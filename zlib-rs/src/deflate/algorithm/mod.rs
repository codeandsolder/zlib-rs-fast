use crate::{
    deflate::{with_compressor, BlockState, Compressor, DeflateStream, Strategy},
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
pub(super) enum ProduceStatus {
    NeedInput,
    BlockReady,
    Finished { insert: usize },
}

/// A resumable token producer. It may fill `State::sym_buf`, but it never encodes or drains a
/// DEFLATE block itself. The driver owns those orchestration boundaries.
pub(super) trait TokenProducer {
    fn produce(&mut self, stream: &mut Compressor<'_, '_>, flush: DeflateFlush) -> ProduceStatus;

    /// Apply state changes that zlib intentionally performs after a full block has been encoded.
    /// Most algorithms have no such continuation; Slow uses it for one lazy-match transition.
    fn after_block_flush(&mut self, _stream: &mut Compressor<'_, '_>) {}
}

/// Drive a token producer until the public streaming contract requires a return.
///
/// Token production is deliberately separate from block encoding and output draining here. This is
/// the scheduling seam for later multi-stream batching: producers yield at full token blocks, while
/// the driver decides when those blocks are encoded and copied to the caller.
pub(super) fn drive_standard<P: TokenProducer>(
    stream: &mut Compressor<'_, '_>,
    flush: DeflateFlush,
    producer: &mut P,
) -> BlockState {
    loop {
        match producer.produce(stream, flush) {
            ProduceStatus::NeedInput => return BlockState::NeedMore,
            ProduceStatus::BlockReady => {
                let blocked = stream.flush_block(false);
                producer.after_block_flush(stream);
                if let Some(blocked) = blocked {
                    return blocked;
                }
            }
            ProduceStatus::Finished { insert } => return finish_standard(stream, flush, insert),
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
        if let Some(blocked) = stream.flush_block(true) {
            return blocked;
        }
        return BlockState::FinishDone;
    }

    if !stream.state.sym_buf.is_empty() {
        if let Some(blocked) = stream.flush_block(false) {
            return blocked;
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
