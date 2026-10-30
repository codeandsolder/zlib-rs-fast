use crate::{
    deflate::{
        drain_pending_output, fill_window_core,
        longest_match::{execute_match_search_batch, MatchSearchResult, MatchSearchWork},
        BlockState, Compressor, DrainOutcome, InputCursor, State, Strategy, MIN_LOOKAHEAD,
    },
    DeflateFlush,
};

mod fast;
mod huff;
mod medium;
mod quick;
mod rle;
mod slow;
mod stored;

const LOCAL_MATCH_SEARCH_MAX_CHAIN: u16 = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum InputStatus {
    Ready,
    NeedMore,
    Exhausted,
}

/// Refill the compression window until the algorithm has its preferred lookahead, or until the
/// caller must provide more input / the current input is exhausted.
///
/// A flushing caller may process a short tail below `preferred_lookahead`; a `NoFlush` caller
/// preserves zlib's streaming contract and yields `NeedMore` instead.
pub(super) fn ensure_input(
    stream: &mut Compressor<'_, '_>,
    preferred_lookahead: usize,
    flush: DeflateFlush,
) -> InputStatus {
    if stream.state.lookahead < preferred_lookahead {
        stream.fill_window();
    }

    if stream.state.lookahead >= preferred_lookahead {
        InputStatus::Ready
    } else if flush == DeflateFlush::NoFlush {
        InputStatus::NeedMore
    } else if stream.state.lookahead == 0 {
        InputStatus::Exhausted
    } else {
        InputStatus::Ready
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BlockContinuation {
    None,
    AdvanceLazyMatch { match_len: u16 },
}

impl BlockContinuation {
    fn apply(self, state: &mut State<'_>) {
        match self {
            Self::None => {}
            Self::AdvanceLazyMatch { match_len } => {
                state.prev_length = match_len;
                state.strstart += 1;
                state.lookahead -= 1;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProduceStatus {
    NeedInput,
    MatchSearch(MatchSearchWork),
    BlockReady(BlockContinuation),
}

/// Owned work produced when `SymBuf` contains a complete non-final token block.
///
/// The post-block continuation is intentionally hidden inside this value: schedulers queue and
/// execute one block action rather than separately remembering an algorithm-specific mutation that
/// must occur immediately after encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TokenBlockWork {
    continuation: BlockContinuation,
}

impl TokenBlockWork {
    fn new(continuation: BlockContinuation) -> Self {
        Self { continuation }
    }

    pub(super) fn encode(self, state: &mut State<'_>) {
        state.encode_block(false);
        self.continuation.apply(state);
    }
}

/// How orchestration wants to close the currently exhausted input segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EndBlockKind {
    FlushBoundary,
    Final,
}

/// State-only work produced when the current input segment is exhausted.
///
/// The producer supplies the tail insertion state; orchestration separately decides whether zlib
/// requested a non-final flush boundary or the final DEFLATE block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EndOfInputWork {
    insert: usize,
}

impl EndOfInputWork {
    fn new(insert: usize) -> Self {
        Self { insert }
    }

    /// Apply tail state and encode the required block, returning whether pending output was added.
    pub(super) fn encode(self, state: &mut State<'_>, kind: EndBlockKind) -> bool {
        state.insert = self.insert;

        match kind {
            EndBlockKind::Final => {
                // A final empty block is still required when there are no resident tokens.
                state.encode_block(true);
                true
            }
            EndBlockKind::FlushBoundary if !state.sym_buf.is_empty() => {
                state.encode_block(false);
                true
            }
            EndBlockKind::FlushBoundary => false,
        }
    }
}

/// A resumable token producer over already-buffered compression state.
///
/// Implementations may update the LZ/hash/token state, but they never refill raw caller input,
/// encode a DEFLATE block, or drain output. Those I/O boundaries belong to `drive_standard`.
pub(super) trait TokenProducer {
    fn preferred_lookahead(&self) -> usize {
        MIN_LOOKAHEAD
    }

    /// Produce tokens until either the token buffer is full or more window input is required.
    /// `allow_short_tail` means the caller is flushing, so the final bytes below the preferred
    /// lookahead may be consumed instead of requesting another refill first.
    fn produce(&mut self, state: &mut State<'_>, allow_short_tail: bool) -> ProduceStatus;

    /// Consume a scheduler-computed match-search result for a producer that yielded one.
    fn complete_match_search(&mut self, _state: &mut State<'_>, _result: MatchSearchResult) {
        panic!("producer received a match-search result it did not request");
    }

    /// A refill may slide the window, invalidating speculative positions cached by a producer.
    fn after_refill(&mut self, _state: &mut State<'_>) {}

    /// Finalize algorithm-local state once no buffered or caller input remains and return the tail
    /// insertion count expected by the common block finalizer.
    fn finish(&mut self, state: &mut State<'_>) -> usize;
}

/// Uniform producer implementation hidden behind [`StandardCore`].
///
/// The concrete Fast/Medium/Slow/Huffman/RLE variants stay private so orchestration cannot depend
/// on algorithm-specific continuation details.
struct StandardProducer(StandardProducerKind);

enum StandardProducerKind {
    Fast(fast::FastProducer),
    Medium(medium::MediumProducer),
    Slow(slow::SlowProducer),
    Huffman(huff::HuffmanProducer),
    Rle(rle::RleProducer),
}

impl StandardProducer {
    fn for_algorithm(algorithm: Algorithm, state: &State<'_>) -> Option<Self> {
        let kind = match algorithm {
            Algorithm::Fast => StandardProducerKind::Fast(fast::FastProducer::new()),
            Algorithm::Medium => StandardProducerKind::Medium(medium::MediumProducer::new(state)),
            Algorithm::Slow => StandardProducerKind::Slow(slow::SlowProducer::new(state)),
            Algorithm::Huffman => StandardProducerKind::Huffman(huff::HuffmanProducer),
            Algorithm::Rle => StandardProducerKind::Rle(rle::RleProducer),
            Algorithm::Stored | Algorithm::Quick => return None,
        };
        Some(Self(kind))
    }
}

impl TokenProducer for StandardProducer {
    fn preferred_lookahead(&self) -> usize {
        match &self.0 {
            StandardProducerKind::Fast(producer) => producer.preferred_lookahead(),
            StandardProducerKind::Medium(producer) => producer.preferred_lookahead(),
            StandardProducerKind::Slow(producer) => producer.preferred_lookahead(),
            StandardProducerKind::Huffman(producer) => producer.preferred_lookahead(),
            StandardProducerKind::Rle(producer) => producer.preferred_lookahead(),
        }
    }

    fn produce(&mut self, state: &mut State<'_>, allow_short_tail: bool) -> ProduceStatus {
        match &mut self.0 {
            StandardProducerKind::Fast(producer) => producer.produce(state, allow_short_tail),
            StandardProducerKind::Medium(producer) => producer.produce(state, allow_short_tail),
            StandardProducerKind::Slow(producer) => producer.produce(state, allow_short_tail),
            StandardProducerKind::Huffman(producer) => producer.produce(state, allow_short_tail),
            StandardProducerKind::Rle(producer) => producer.produce(state, allow_short_tail),
        }
    }

    fn complete_match_search(&mut self, state: &mut State<'_>, result: MatchSearchResult) {
        match &mut self.0 {
            StandardProducerKind::Fast(producer) => producer.complete_match_search(state, result),
            StandardProducerKind::Medium(producer) => producer.complete_match_search(state, result),
            StandardProducerKind::Slow(producer) => producer.complete_match_search(state, result),
            StandardProducerKind::Huffman(producer) => {
                producer.complete_match_search(state, result)
            }
            StandardProducerKind::Rle(producer) => producer.complete_match_search(state, result),
        }
    }

    fn after_refill(&mut self, state: &mut State<'_>) {
        match &mut self.0 {
            StandardProducerKind::Fast(producer) => producer.after_refill(state),
            StandardProducerKind::Medium(producer) => producer.after_refill(state),
            StandardProducerKind::Slow(producer) => producer.after_refill(state),
            StandardProducerKind::Huffman(producer) => producer.after_refill(state),
            StandardProducerKind::Rle(producer) => producer.after_refill(state),
        }
    }

    fn finish(&mut self, state: &mut State<'_>) -> usize {
        match &mut self.0 {
            StandardProducerKind::Fast(producer) => producer.finish(state),
            StandardProducerKind::Medium(producer) => producer.finish(state),
            StandardProducerKind::Slow(producer) => producer.finish(state),
            StandardProducerKind::Huffman(producer) => producer.finish(state),
            StandardProducerKind::Rle(producer) => producer.finish(state),
        }
    }
}

/// Whether the producer may consume a short tail below its preferred lookahead.
///
/// `FinalTail` is a fact supplied by orchestration after it has tried to satisfy `NeedInput` and
/// knows the current zlib call is flushing all caller input. Keeping that fact outside `State`
/// prevents caller-I/O policy from leaking back into the compression core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputPhase {
    Open,
    FinalTail,
}

/// Whether orchestration expects more caller input for the current segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum InputCompletion {
    MoreMayArrive,
    EndOfSegment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StandardPhase {
    Producing,
    NeedsMatchSearch,
    NeedsDrain,
}

/// A resumable request to drain bytes produced by block encoding.
///
/// `OutputFull` returns this token unchanged so a persistent scheduler can resume the same drain
/// before advancing token production again.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PendingOutputWork;

#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PendingOutputOutcome {
    Ready,
    OutputFull(PendingOutputWork),
}

/// Per-call scheduling state for the ordinary token-producing algorithms.
///
/// It owns the heterogeneous producer plus the orchestration state needed between scheduler turns:
/// whether a short buffered tail is final and whether encoded output must be drained before token
/// production may resume. It borrows no stream memory and allocates nothing, so schedulers can hold
/// one beside each active stream.
pub(super) struct StandardCore {
    producer: StandardProducer,
    input_phase: InputPhase,
    phase: StandardPhase,
}

impl StandardCore {
    pub(super) fn for_algorithm(algorithm: Algorithm, state: &State<'_>) -> Option<Self> {
        Some(Self {
            producer: StandardProducer::for_algorithm(algorithm, state)?,
            input_phase: InputPhase::Open,
            phase: StandardPhase::Producing,
        })
    }

    pub(super) fn next_event(&mut self, state: &mut State<'_>) -> CoreEvent {
        self.assert_producing();
        let event = next_core_event(state, self.input_phase, &mut self.producer);
        if matches!(&event, CoreEvent::MatchSearch(_)) {
            self.phase = StandardPhase::NeedsMatchSearch;
        }
        event
    }

    pub(super) fn complete_match_search(
        &mut self,
        state: &mut State<'_>,
        result: MatchSearchResult,
    ) {
        assert_eq!(
            self.phase,
            StandardPhase::NeedsMatchSearch,
            "match-search completion requires outstanding search work"
        );
        self.producer.complete_match_search(state, result);
        self.phase = StandardPhase::Producing;
    }

    pub(super) fn satisfy_input(
        &mut self,
        state: &mut State<'_>,
        input: &mut InputCursor,
        demand: InputDemand,
        completion: InputCompletion,
    ) -> RefillOutcome {
        let outcome = satisfy_input_demand(state, input, &mut self.producer, demand, completion);
        if outcome == RefillOutcome::FinalTail {
            self.input_phase = InputPhase::FinalTail;
        }
        outcome
    }

    pub(super) fn encode_token_block(
        &mut self,
        state: &mut State<'_>,
        work: TokenBlockWork,
    ) -> PendingOutputWork {
        self.assert_producing();
        work.encode(state);
        self.phase = StandardPhase::NeedsDrain;
        PendingOutputWork
    }

    pub(super) fn encode_end_of_input(
        &mut self,
        state: &mut State<'_>,
        work: EndOfInputWork,
        kind: EndBlockKind,
    ) -> Option<PendingOutputWork> {
        self.assert_producing();
        if work.encode(state, kind) {
            self.phase = StandardPhase::NeedsDrain;
            Some(PendingOutputWork)
        } else {
            None
        }
    }

    pub(super) fn drain_output(
        &mut self,
        state: &mut State<'_>,
        output: &mut crate::deflate::OutputCursor,
        work: PendingOutputWork,
    ) -> PendingOutputOutcome {
        assert_eq!(
            self.phase,
            StandardPhase::NeedsDrain,
            "drain work requires an encoded block"
        );

        match drain_pending_output(state, output) {
            DrainOutcome::Ready => {
                self.phase = StandardPhase::Producing;
                PendingOutputOutcome::Ready
            }
            DrainOutcome::OutputFull => PendingOutputOutcome::OutputFull(work),
        }
    }

    fn assert_producing(&self) {
        assert_eq!(
            self.phase,
            StandardPhase::Producing,
            "outstanding match search or pending output must complete before advancing standard compression"
        );
    }
}

/// Scheduler-level policy for one input segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StandardSegmentMode {
    /// More caller input may arrive; a short lookahead must yield instead of consuming a tail.
    Continue,
    /// This call ends an input segment but not the DEFLATE stream.
    FlushBoundary,
    /// This is the final input segment and must emit the terminal DEFLATE block.
    Final,
}

impl StandardSegmentMode {
    fn input_completion(self) -> InputCompletion {
        match self {
            Self::Continue => InputCompletion::MoreMayArrive,
            Self::FlushBoundary | Self::Final => InputCompletion::EndOfSegment,
        }
    }

    fn end_kind(self) -> Option<EndBlockKind> {
        match self {
            Self::Continue => None,
            Self::FlushBoundary => Some(EndBlockKind::FlushBoundary),
            Self::Final => Some(EndBlockKind::Final),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AfterDrain {
    ResumeProduction,
    EndSegment(EndBlockKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingDrain {
    work: PendingOutputWork,
    after: AfterDrain,
}

/// Result of exactly one schedulable standard-compression stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StandardTurnOutcome {
    Progress,
    MatchSearch(MatchSearchWork),
    NeedMoreCallerInput,
    NeedMoreOutput { final_block: bool },
    SegmentDone(EndBlockKind),
}

/// Reusable single-stream scheduler state for the standard token-producing algorithms.
///
/// Each `step` performs at most one stage: refill, token production, block encoding, or pending
/// output drain. A synchronous zlib adapter can call it in a tight loop; a batch scheduler can call
/// it once per ready stream and rotate fairly without duplicating stage sequencing rules.
pub(super) struct StandardDriver {
    core: StandardCore,
    pending_drain: Option<PendingDrain>,
}

impl StandardDriver {
    pub(super) fn for_algorithm(algorithm: Algorithm, state: &State<'_>) -> Option<Self> {
        Some(Self {
            core: StandardCore::for_algorithm(algorithm, state)?,
            pending_drain: None,
        })
    }

    pub(super) fn step(
        &mut self,
        state: &mut State<'_>,
        input: &mut InputCursor,
        output: &mut crate::deflate::OutputCursor,
        mode: StandardSegmentMode,
    ) -> StandardTurnOutcome {
        self.step_inner(state, input, output, mode)
    }

    #[inline(always)]
    fn step_inner(
        &mut self,
        state: &mut State<'_>,
        input: &mut InputCursor,
        output: &mut crate::deflate::OutputCursor,
        mode: StandardSegmentMode,
    ) -> StandardTurnOutcome {
        if let Some(pending) = self.pending_drain.take() {
            return match self.core.drain_output(state, output, pending.work) {
                PendingOutputOutcome::Ready => match pending.after {
                    AfterDrain::ResumeProduction => StandardTurnOutcome::Progress,
                    AfterDrain::EndSegment(kind) => StandardTurnOutcome::SegmentDone(kind),
                },
                PendingOutputOutcome::OutputFull(work) => {
                    self.pending_drain = Some(PendingDrain {
                        work,
                        after: pending.after,
                    });
                    StandardTurnOutcome::NeedMoreOutput {
                        final_block: pending.after == AfterDrain::EndSegment(EndBlockKind::Final),
                    }
                }
            };
        }

        enum OwnedCoreEvent {
            NeedInput(InputDemand),
            TokenBlockReady(TokenBlockWork),
            EndOfInput(EndOfInputWork),
        }

        let event = match self.core.next_event(state) {
            CoreEvent::MatchSearch(request) => return StandardTurnOutcome::MatchSearch(request),
            CoreEvent::NeedInput(demand) => OwnedCoreEvent::NeedInput(demand),
            CoreEvent::TokenBlockReady(work) => OwnedCoreEvent::TokenBlockReady(work),
            CoreEvent::EndOfInput(work) => OwnedCoreEvent::EndOfInput(work),
        };

        match event {
            OwnedCoreEvent::NeedInput(demand) => {
                match self
                    .core
                    .satisfy_input(state, input, demand, mode.input_completion())
                {
                    RefillOutcome::Ready | RefillOutcome::FinalTail => {
                        StandardTurnOutcome::Progress
                    }
                    RefillOutcome::NeedMoreCallerInput => StandardTurnOutcome::NeedMoreCallerInput,
                }
            }
            OwnedCoreEvent::TokenBlockReady(work) => {
                self.pending_drain = Some(PendingDrain {
                    work: self.core.encode_token_block(state, work),
                    after: AfterDrain::ResumeProduction,
                });
                StandardTurnOutcome::Progress
            }
            OwnedCoreEvent::EndOfInput(work) => {
                let kind = mode
                    .end_kind()
                    .expect("end-of-input requires a segment-ending scheduler mode");
                match self.core.encode_end_of_input(state, work, kind) {
                    Some(work) => {
                        self.pending_drain = Some(PendingDrain {
                            work,
                            after: AfterDrain::EndSegment(kind),
                        });
                        StandardTurnOutcome::Progress
                    }
                    None => StandardTurnOutcome::SegmentDone(kind),
                }
            }
        }
    }

    pub(super) fn complete_match_search(
        &mut self,
        state: &mut State<'_>,
        result: MatchSearchResult,
    ) {
        self.core.complete_match_search(state, result);
    }
}

/// A request to refill one standard compression stream to its preferred lookahead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct InputDemand {
    preferred_lookahead: usize,
}

impl InputDemand {
    fn new(preferred_lookahead: usize) -> Self {
        Self {
            preferred_lookahead,
        }
    }

    fn is_satisfied(self, state: &State<'_>) -> bool {
        state.lookahead >= self.preferred_lookahead
    }
}

/// A scheduler-visible boundary reached while advancing one ordinary compression stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CoreEvent {
    /// The producer needs more buffered lookahead before it can continue normally.
    NeedInput(InputDemand),
    /// The producer has paused on immutable longest-match work that may execute elsewhere.
    MatchSearch(MatchSearchWork),
    /// A complete resident token block is ready to run as one owned block-encoding action.
    TokenBlockReady(TokenBlockWork),
    /// No more input remains for this call; orchestration must close this input segment.
    EndOfInput(EndOfInputWork),
}

/// Advance already-buffered compression state until it reaches a scheduling boundary.
///
/// This is deliberately I/O-free: it cannot refill caller input, encode a token block, or drain
/// pending output. A scheduler decides how to satisfy `NeedInput`, then calls this again with either
/// more buffered data or `FinalTail` once it knows no more input belongs to the current flush.
fn next_core_event(
    state: &mut State<'_>,
    input_phase: InputPhase,
    producer: &mut StandardProducer,
) -> CoreEvent {
    let preferred_lookahead = producer.preferred_lookahead();

    if state.lookahead < preferred_lookahead {
        match input_phase {
            InputPhase::Open => {
                return CoreEvent::NeedInput(InputDemand::new(preferred_lookahead));
            }
            InputPhase::FinalTail if state.lookahead == 0 => {
                return CoreEvent::EndOfInput(EndOfInputWork::new(producer.finish(state)));
            }
            InputPhase::FinalTail => {}
        }
    }

    let allow_short_tail = input_phase == InputPhase::FinalTail;
    enum OwnedProduceStatus {
        NeedInput,
        BlockReady(BlockContinuation),
    }

    let status = match producer.produce(state, allow_short_tail) {
        ProduceStatus::MatchSearch(request) => return CoreEvent::MatchSearch(request),
        ProduceStatus::NeedInput => OwnedProduceStatus::NeedInput,
        ProduceStatus::BlockReady(continuation) => OwnedProduceStatus::BlockReady(continuation),
    };

    match status {
        OwnedProduceStatus::NeedInput => {
            debug_assert!(state.lookahead < preferred_lookahead);
            match input_phase {
                InputPhase::Open => CoreEvent::NeedInput(InputDemand::new(preferred_lookahead)),
                InputPhase::FinalTail => {
                    debug_assert_eq!(state.lookahead, 0);
                    CoreEvent::EndOfInput(EndOfInputWork::new(producer.finish(state)))
                }
            }
        }
        OwnedProduceStatus::BlockReady(continuation) => {
            CoreEvent::TokenBlockReady(TokenBlockWork::new(continuation))
        }
    }
}

/// Result of trying to satisfy an [`InputDemand`] from the current zlib caller buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RefillOutcome {
    Ready,
    NeedMoreCallerInput,
    FinalTail,
}

/// Perform only the caller-I/O refill stage for one standard compression stream.
///
/// The token producer is notified after every refill because the window may slide and invalidate
/// absolute speculative positions. No token production, block encoding, or output draining occurs
/// here.
fn satisfy_input_demand(
    state: &mut State<'_>,
    input: &mut InputCursor,
    producer: &mut StandardProducer,
    demand: InputDemand,
    completion: InputCompletion,
) -> RefillOutcome {
    debug_assert!(!demand.is_satisfied(state));

    fill_window_core(state, input);
    producer.after_refill(state);

    if demand.is_satisfied(state) {
        RefillOutcome::Ready
    } else if completion == InputCompletion::MoreMayArrive {
        RefillOutcome::NeedMoreCallerInput
    } else {
        RefillOutcome::FinalTail
    }
}

/// Compatibility driver for the ordinary zlib streaming API.
///
/// It currently consumes core events immediately. Future batch scheduling can replace this loop
/// while reusing `StandardCore`, its state-only token producers, and the independent refill, encode,
/// and drain stages unchanged.
pub(super) fn drive_standard(
    state: &mut State<'_>,
    input: &mut InputCursor,
    output: &mut crate::deflate::OutputCursor,
    flush: DeflateFlush,
    driver: &mut StandardDriver,
) -> BlockState {
    let mode = match flush {
        DeflateFlush::NoFlush => StandardSegmentMode::Continue,
        DeflateFlush::Finish => StandardSegmentMode::Final,
        _ => StandardSegmentMode::FlushBoundary,
    };

    loop {
        match driver.step_inner(state, input, output, mode) {
            StandardTurnOutcome::Progress => {}
            StandardTurnOutcome::MatchSearch(work) => {
                let request = work.request(state);
                let mut results = [MatchSearchResult {
                    length: 0,
                    start: 0,
                }];
                execute_match_search_batch(core::slice::from_ref(&request), &mut results);
                driver.complete_match_search(state, results[0]);
            }
            StandardTurnOutcome::NeedMoreCallerInput => return BlockState::NeedMore,
            StandardTurnOutcome::NeedMoreOutput { final_block: true } => {
                return BlockState::FinishStarted;
            }
            StandardTurnOutcome::NeedMoreOutput { final_block: false } => {
                return BlockState::NeedMore;
            }
            StandardTurnOutcome::SegmentDone(EndBlockKind::Final) => {
                return BlockState::FinishDone;
            }
            StandardTurnOutcome::SegmentDone(EndBlockKind::FlushBoundary) => {
                return BlockState::BlockDone;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Algorithm {
    Stored,
    Quick,
    Fast,
    Medium,
    Slow,
    Huffman,
    Rle,
}

impl Algorithm {
    pub(super) fn for_stream(level: i8, strategy: Strategy) -> Self {
        if level == 0 {
            return Self::Stored;
        }

        match strategy {
            Strategy::HuffmanOnly => Self::Huffman,
            Strategy::Rle => Self::Rle,
            Strategy::Default | Strategy::Filtered | Strategy::Fixed => {
                level_config(level).algorithm
            }
        }
    }

    fn run(self, stream: &mut Compressor<'_, '_>, flush: DeflateFlush) -> BlockState {
        match self {
            Self::Stored => stored::deflate_stored(stream, flush),
            Self::Quick => quick::deflate_quick(stream, flush),
            algorithm => {
                let mut driver = StandardDriver::for_algorithm(algorithm, stream.state)
                    .expect("ordinary algorithm must have a token producer");
                drive_standard(
                    stream.state,
                    &mut stream.input,
                    &mut stream.output,
                    flush,
                    &mut driver,
                )
            }
        }
    }
}

pub(super) fn run(stream: &mut Compressor<'_, '_>, flush: DeflateFlush) -> BlockState {
    let algorithm = Algorithm::for_stream(stream.state.level, stream.state.strategy);
    algorithm.run(stream, flush)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LevelConfig {
    pub good_length: u16,
    pub max_lazy: u16,
    pub nice_length: u16,
    pub max_chain: u16,
    pub algorithm: Algorithm,
}

impl LevelConfig {
    const fn new(
        good_length: u16,
        max_lazy: u16,
        nice_length: u16,
        max_chain: u16,
        algorithm: Algorithm,
    ) -> Self {
        Self {
            good_length,
            max_lazy,
            nice_length,
            max_chain,
            algorithm,
        }
    }
}

const LEVEL_CONFIGS: [LevelConfig; 10] = [
    LevelConfig::new(0, 0, 0, 0, Algorithm::Stored),
    LevelConfig::new(0, 0, 0, 0, Algorithm::Quick),
    LevelConfig::new(4, 4, 8, 4, Algorithm::Fast),
    LevelConfig::new(4, 6, 16, 6, Algorithm::Medium),
    LevelConfig::new(4, 12, 32, 24, Algorithm::Medium),
    LevelConfig::new(8, 16, 32, 32, Algorithm::Medium),
    LevelConfig::new(8, 16, 128, 128, Algorithm::Medium),
    LevelConfig::new(8, 32, 128, 256, Algorithm::Slow),
    LevelConfig::new(32, 128, 258, 1024, Algorithm::Slow),
    LevelConfig::new(32, 258, 258, 4096, Algorithm::Slow),
];

pub(crate) fn level_config(level: i8) -> LevelConfig {
    LEVEL_CONFIGS[level as usize]
}

pub(crate) fn change_requires_block_boundary(
    current_level: i8,
    current_strategy: Strategy,
    next_level: i8,
    next_strategy: Strategy,
) -> bool {
    current_strategy != next_strategy
        || level_config(current_level).algorithm != level_config(next_level).algorithm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_levels_have_explicit_algorithm_families() {
        let expected = [
            Algorithm::Stored,
            Algorithm::Quick,
            Algorithm::Fast,
            Algorithm::Medium,
            Algorithm::Medium,
            Algorithm::Medium,
            Algorithm::Medium,
            Algorithm::Slow,
            Algorithm::Slow,
            Algorithm::Slow,
        ];

        for (level, expected) in expected.into_iter().enumerate() {
            assert_eq!(level_config(level as i8).algorithm, expected);
        }
    }

    #[test]
    fn strategies_select_specialized_algorithms() {
        assert_eq!(
            Algorithm::for_stream(6, Strategy::Default),
            Algorithm::Medium
        );
        assert_eq!(
            Algorithm::for_stream(6, Strategy::Filtered),
            Algorithm::Medium
        );
        assert_eq!(Algorithm::for_stream(6, Strategy::Fixed), Algorithm::Medium);
        assert_eq!(
            Algorithm::for_stream(6, Strategy::HuffmanOnly),
            Algorithm::Huffman
        );
        assert_eq!(Algorithm::for_stream(6, Strategy::Rle), Algorithm::Rle);
        assert_eq!(
            Algorithm::for_stream(0, Strategy::HuffmanOnly),
            Algorithm::Stored
        );
        assert_eq!(Algorithm::for_stream(0, Strategy::Rle), Algorithm::Stored);
    }

    #[test]
    fn parameter_changes_flush_only_across_algorithm_or_strategy_boundaries() {
        assert!(!change_requires_block_boundary(
            3,
            Strategy::Default,
            6,
            Strategy::Default
        ));
        assert!(!change_requires_block_boundary(
            7,
            Strategy::Default,
            9,
            Strategy::Default
        ));
        assert!(change_requires_block_boundary(
            6,
            Strategy::Default,
            7,
            Strategy::Default
        ));
        assert!(change_requires_block_boundary(
            6,
            Strategy::Default,
            6,
            Strategy::Fixed
        ));
    }
}
