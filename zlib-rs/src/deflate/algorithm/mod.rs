use crate::{
    deflate::{
        drain_pending_output, with_compressor, BlockState, Compressor, DeflateStream, DrainOutcome,
        State, Strategy, MIN_LOOKAHEAD,
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
            Algorithm::Fast => StandardProducerKind::Fast(fast::FastProducer),
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

/// Per-call scheduling state for the ordinary token-producing algorithms.
///
/// It owns the heterogeneous producer plus the only orchestration state the core needs: whether a
/// short buffered tail has been certified final for this zlib call. It borrows no stream memory and
/// allocates nothing, so schedulers can hold one beside each active stream.
pub(super) struct StandardCore {
    producer: StandardProducer,
    input_phase: InputPhase,
}

impl StandardCore {
    pub(super) fn for_algorithm(algorithm: Algorithm, state: &State<'_>) -> Option<Self> {
        Some(Self {
            producer: StandardProducer::for_algorithm(algorithm, state)?,
            input_phase: InputPhase::Open,
        })
    }

    pub(super) fn next_event(&mut self, state: &mut State<'_>) -> CoreEvent {
        next_core_event(state, self.input_phase, &mut self.producer)
    }

    pub(super) fn satisfy_input(
        &mut self,
        stream: &mut Compressor<'_, '_>,
        demand: InputDemand,
        flush: DeflateFlush,
    ) -> RefillOutcome {
        let outcome = satisfy_input_demand(stream, &mut self.producer, demand, flush);
        if outcome == RefillOutcome::FinalTail {
            self.input_phase = InputPhase::FinalTail;
        }
        outcome
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
    match producer.produce(state, allow_short_tail) {
        ProduceStatus::NeedInput => {
            debug_assert!(state.lookahead < preferred_lookahead);
            match input_phase {
                InputPhase::Open => CoreEvent::NeedInput(InputDemand::new(preferred_lookahead)),
                InputPhase::FinalTail => {
                    debug_assert_eq!(state.lookahead, 0);
                    CoreEvent::EndOfInput(EndOfInputWork::new(producer.finish(state)))
                }
            }
        }
        ProduceStatus::BlockReady(continuation) => {
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
    stream: &mut Compressor<'_, '_>,
    producer: &mut StandardProducer,
    demand: InputDemand,
    flush: DeflateFlush,
) -> RefillOutcome {
    debug_assert!(!demand.is_satisfied(stream.state));

    stream.fill_window();
    producer.after_refill(stream.state);

    if demand.is_satisfied(stream.state) {
        RefillOutcome::Ready
    } else if flush == DeflateFlush::NoFlush {
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
    stream: &mut Compressor<'_, '_>,
    flush: DeflateFlush,
    core: &mut StandardCore,
) -> BlockState {
    loop {
        match core.next_event(stream.state) {
            CoreEvent::NeedInput(demand) => match core.satisfy_input(stream, demand, flush) {
                RefillOutcome::Ready | RefillOutcome::FinalTail => {}
                RefillOutcome::NeedMoreCallerInput => return BlockState::NeedMore,
            },
            CoreEvent::TokenBlockReady(work) => {
                work.encode(stream.state);
                if drain_pending_output(stream.state, &mut stream.output)
                    == DrainOutcome::OutputFull
                {
                    return BlockState::NeedMore;
                }
            }
            CoreEvent::EndOfInput(work) => {
                let kind = if flush == DeflateFlush::Finish {
                    EndBlockKind::Final
                } else {
                    EndBlockKind::FlushBoundary
                };
                let encoded = work.encode(stream.state, kind);
                let output_full = encoded
                    && drain_pending_output(stream.state, &mut stream.output)
                        == DrainOutcome::OutputFull;

                return match (kind, output_full) {
                    (EndBlockKind::Final, true) => BlockState::FinishStarted,
                    (EndBlockKind::Final, false) => BlockState::FinishDone,
                    (EndBlockKind::FlushBoundary, true) => BlockState::NeedMore,
                    (EndBlockKind::FlushBoundary, false) => BlockState::BlockDone,
                };
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
    fn for_stream(level: i8, strategy: Strategy) -> Self {
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
                let mut core = StandardCore::for_algorithm(algorithm, stream.state)
                    .expect("ordinary algorithm must have a token producer");
                drive_standard(stream, flush, &mut core)
            }
        }
    }
}

pub fn run(stream: &mut DeflateStream, flush: DeflateFlush) -> BlockState {
    let algorithm = Algorithm::for_stream(stream.state.level, stream.state.strategy);
    with_compressor(stream, |compressor| algorithm.run(compressor, flush))
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
