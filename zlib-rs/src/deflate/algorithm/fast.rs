#![forbid(unsafe_code)]

use super::{ensure_input, finish_standard, InputStatus};
use crate::deflate::hash_calc::StandardHashCalc;
use crate::{
    deflate::{BlockState, Compressor, MIN_LOOKAHEAD, STD_MIN_MATCH, WANT_MIN_MATCH},
    DeflateFlush,
};

pub fn deflate_fast(stream: &mut Compressor, flush: DeflateFlush) -> BlockState {
    loop {
        match ensure_input(stream, MIN_LOOKAHEAD, flush) {
            InputStatus::Ready => {}
            InputStatus::NeedMore => return BlockState::NeedMore,
            InputStatus::Exhausted => break,
        }

        let state = &mut stream.state;

        // Insert the string window[strstart .. strstart+2] in the
        // dictionary, and set hash_head to the head of the hash chain:

        // Literal character to output if there is no match.
        let lc = if state.lookahead >= WANT_MIN_MATCH {
            let val = u32::from_le_bytes(
                state.window.filled()[state.strstart..state.strstart + 4]
                    .try_into()
                    .unwrap(),
            );
            let hash_head = StandardHashCalc::quick_insert_value(state, state.strstart, val);
            let dist = state.strstart as isize - hash_head as isize;

            // Find the longest match for the string starting at offset state.strstart.
            if dist <= state.max_dist() as isize && dist > 0 && hash_head != 0 {
                // To simplify the code, we prevent matches with the string
                // of window index 0 (in particular we have to avoid a match
                // of the string with itself at the start of the input file).
                let mut match_len;
                (match_len, state.match_start) =
                    crate::deflate::longest_match::longest_match(state, hash_head);
                if match_len >= WANT_MIN_MATCH {
                    let bflush = state.tally_dist(
                        state.strstart - state.match_start as usize,
                        match_len - STD_MIN_MATCH,
                    );

                    state.lookahead -= match_len;

                    /* Insert new strings in the hash table only if the match length
                     * is not too large. This saves time but degrades compression.
                     */
                    if match_len <= usize::from(state.max_insert_length)
                        && state.lookahead >= WANT_MIN_MATCH
                    {
                        match_len -= 1; /* string at strstart already in table */
                        state.strstart += 1;

                        state.insert_string(state.strstart, match_len);
                        state.strstart += match_len;
                    } else {
                        state.strstart += match_len;
                        StandardHashCalc::quick_insert_string(
                            state,
                            state.strstart + 2 - STD_MIN_MATCH,
                        );

                        /* If lookahead < STD_MIN_MATCH, ins_h is garbage, but it does not
                         * matter since it will be recomputed at next deflate call.
                         */
                    }
                    if bflush {
                        if let Some(blocked) = stream.flush_block(false) {
                            return blocked;
                        }
                    }
                    continue;
                }
            }
            val as u8
        } else {
            state.window.filled()[state.strstart]
        };
        /* No match, output a literal byte */
        let bflush = state.tally_lit(lc);
        state.lookahead -= 1;
        state.strstart += 1;
        if bflush {
            if let Some(blocked) = stream.flush_block(false) {
                return blocked;
            }
        }
    }

    let insert = Ord::min(stream.state.strstart, STD_MIN_MATCH - 1);
    finish_standard(stream, flush, insert)
}
