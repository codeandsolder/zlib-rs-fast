use crate::{
    deflate::{
        with_compressor, BlockState, Compressor, DeflateStream, State, Strategy, MIN_LOOKAHEAD,
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

/// A scheduler-visible boundary reached while advancing one ordinary compression stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CoreEvent {
    /// Caller input was exhausted before the producer's preferred lookahead could be restored.
    NeedInput,
    /// A complete token block is resident in `State::sym_buf` and is ready for block encoding.
    TokenBlockReady(BlockContinuation),
    /// No more input remains for this call. `insert` is the tail state needed by a later refill.
    Finished { insert: usize },
}

/// Advance one stream only until it reaches a scheduling boundary.
///
/// This function may refill the LZ window, but it never encodes a token block and never drains
/// output. A batch scheduler can therefore collect `TokenBlockReady` events from multiple streams
/// before choosing where and when to run the block-encoding stage.
pub(super) fn next_core_event<P: TokenProducer>(
    stream: &mut Compressor<'_, '_>,
    flush: DeflateFlush,
    producer: &mut P,
) -> CoreEvent {
    loop {
        let preferred_lookahead = producer.preferred_lookahead();
        let mut allow_short_tail = false;

        if stream.state.lookahead < preferred_lookahead {
            stream.fill_window();
            producer.after_refill(stream.state);

            if stream.state.lookahead < preferred_lookahead {
                if flush == DeflateFlush::NoFlush {
                    return CoreEvent::NeedInput;
                }
                if stream.state.lookahead == 0 {
                    return CoreEvent::Finished {
                        insert: producer.finish(stream.state),
                    };
                }
                allow_short_tail = true;
            }
        }

        match producer.produce(stream.state, allow_short_tail) {
            ProduceStatus::NeedInput => {
                debug_assert!(stream.state.lookahead < preferred_lookahead);
            }
            ProduceStatus::BlockReady(continuation) => {
                return CoreEvent::TokenBlockReady(continuation);
            }
        }
    }
}

/// Compatibility driver for the ordinary zlib streaming API.
///
/// It currently consumes core events immediately. Future batch scheduling can replace this loop
/// while reusing `next_core_event`, the state-only token producers, and the independent
/// encode/drain stages unchanged.
pub(super) fn drive_standard<P: TokenProducer>(
    stream: &mut Compressor<'_, '_>,
    flush: DeflateFlush,
    producer: &mut P,
) -> BlockState {
    loop {
        match next_core_event(stream, flush, producer) {
            CoreEvent::NeedInput => return BlockState::NeedMore,
            CoreEvent::TokenBlockReady(continuation) => {
                stream.encode_block(false);
                continuation.apply(stream.state);
                stream.flush_pending();
                if stream.io.avail_out == 0 {
                    return BlockState::NeedMore;
                }
            }
            CoreEvent::Finished { insert } => return finish_standard(stream, flush, insert),
        }
    }
}

/// Finalize the ordinary token-producing algorithms after input processing stops.
///
/// `insert` records how much tail state must be rebuilt when more input arrives. Stored and Quick
/// have different block mechanics and intentionally do not use this path.
pub(super) fn finish_standard(
    stream: &mut Compressor<'_, '_>,
    flush: DeflateFlush,
    insert: usize,
) -> BlockState {
    stream.state.insert = insert;

    if flush == DeflateFlush::Finish {
        stream.encode_block(true);
        stream.flush_pending();
        return if stream.io.avail_out == 0 {
            BlockState::FinishStarted
        } else {
            BlockState::FinishDone
        };
    }

    if !stream.state.sym_buf.is_empty() {
        stream.encode_block(false);
        stream.flush_pending();
        if stream.io.avail_out == 0 {
            return BlockState::NeedMore;
        }
    }

    BlockState::BlockDone
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
            Self::Fast => fast::deflate_fast(stream, flush),
            Self::Medium => medium::deflate_medium(stream, flush),
            Self::Slow => slow::deflate_slow(stream, flush),
            Self::Huffman => huff::deflate_huff(stream, flush),
            Self::Rle => rle::deflate_rle(stream, flush),
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
