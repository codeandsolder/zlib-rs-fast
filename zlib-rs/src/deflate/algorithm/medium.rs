#![forbid(unsafe_code)]

use super::{
    BlockContinuation, MatchSourceKind, ProduceStatus, TokenProducer, LOCAL_MATCH_SEARCH_MAX_CHAIN,
};
use crate::deflate::hash_calc::{hash_chain_position, StandardHashCalc};
use crate::deflate::longest_match::{
    MatchSearchError, MatchSearchKind, MatchSearchResult, MatchSearchWork,
};
use crate::deflate::{MatchCandidate, State, MIN_LOOKAHEAD, STD_MIN_MATCH, WANT_MIN_MATCH};

type Match = MatchCandidate;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MediumPhase {
    Ready,
    AwaitingCurrentSearch,
    AfterCurrentSearch,
    AwaitingFutureSearch { current_lookahead: usize },
    AfterFutureSearch,
}

pub(super) struct MediumProducer {
    early_exit: bool,
    current_match: Match,
    next_match: Match,
    phase: MediumPhase,
    match_source: MatchSourceKind,
}

impl MediumProducer {
    pub(super) const fn new(state: &State<'_>, match_source: MatchSourceKind) -> Self {
        Self {
            early_exit: state.level < 5,
            current_match: Match::EMPTY,
            next_match: state.medium_next_match,
            phase: MediumPhase::Ready,
            match_source,
        }
    }

    const fn clear_saved_next(&mut self, state: &mut State<'_>) {
        self.next_match = Match::EMPTY;
        state.medium_next_match = Match::EMPTY;
    }

    const fn save_next(&self, state: &mut State<'_>) {
        state.medium_next_match = self.next_match;
    }

    fn apply_current_search(&mut self, state: &mut State<'_>, result: MatchSearchResult) {
        state.match_start = result.start;
        self.current_match.match_length = result.length;
        self.current_match.match_start = result.start;
        if usize::from(self.current_match.match_length) < WANT_MIN_MATCH
            || self.current_match.match_start >= self.current_match.strstart
        {
            self.current_match.match_length = 1;
        }
    }

    fn apply_future_search(&mut self, state: &mut State<'_>, result: MatchSearchResult) {
        state.match_start = result.start;
        self.next_match.match_length = result.length;
        self.next_match.match_start = result.start;

        if self.next_match.match_start >= self.next_match.strstart {
            self.next_match.match_length = 1;
        }
        if usize::from(self.next_match.match_length) < WANT_MIN_MATCH {
            self.next_match.match_length = 1;
        } else {
            fizzle_matches(
                state.window.filled(),
                state.max_dist(),
                &mut self.current_match,
                &mut self.next_match,
            );
        }
    }

    /// Stock Medium levels never need scheduler suspension: their longest hash chain is 128.
    /// Keep their hot-loop match state local so LLVM can retain it in registers, while the
    /// resumable path below remains available for streams tuned beyond the local-search limit.
    fn produce_local(&mut self, state: &mut State<'_>, allow_short_tail: bool) -> ProduceStatus {
        debug_assert_eq!(self.phase, MediumPhase::Ready);
        debug_assert_eq!(self.match_source, MatchSourceKind::HashChain);
        debug_assert!(state.max_chain_length <= LOCAL_MATCH_SEARCH_MAX_CHAIN);

        let mut current_match = Match::EMPTY;
        let mut next_match = self.next_match;

        loop {
            if state.lookahead == 0 || (!allow_short_tail && state.lookahead < MIN_LOOKAHEAD) {
                self.next_match = next_match;
                return ProduceStatus::NeedInput;
            }

            if !self.early_exit && next_match.match_length > 0 {
                current_match = next_match;
                next_match.match_length = 0;
            } else {
                let hash_head = if state.lookahead >= WANT_MIN_MATCH {
                    StandardHashCalc::quick_insert_string(state, state.strstart)
                } else {
                    0
                };

                current_match.strstart = hash_chain_position(state.strstart);
                current_match.orgstart = current_match.strstart;

                if hash_head != 0 && state.recent_hash_distance(hash_head).is_some() {
                    let result = match MatchSearchWork::new(hash_head, MatchSearchKind::Standard)
                        .request(state)
                    {
                        Ok(request) => request.execute(),
                        Err(error) => return ProduceStatus::MatchSearchError(error),
                    };
                    state.match_start = result.start;
                    current_match.match_length = result.length;
                    current_match.match_start = result.start;
                    if usize::from(current_match.match_length) < WANT_MIN_MATCH
                        || current_match.match_start >= current_match.strstart
                    {
                        current_match.match_length = 1;
                    }
                } else {
                    current_match.match_start = 0;
                    current_match.match_length = 1;
                }
            }

            insert_match(state, current_match, MatchSourceKind::HashChain);

            let current_len = usize::from(current_match.match_length);
            let future_start = usize::from(current_match.strstart) + current_len;
            let future_lookahead = state.lookahead.saturating_sub(current_len);
            if !self.early_exit
                && state.lookahead > MIN_LOOKAHEAD
                && future_lookahead >= WANT_MIN_MATCH
                && future_start < state.window_size - MIN_LOOKAHEAD
            {
                let current_lookahead = state.lookahead;
                state.strstart = future_start;
                state.lookahead = future_lookahead;
                let hash_head = StandardHashCalc::quick_insert_string(state, state.strstart);

                next_match.strstart = hash_chain_position(state.strstart);
                next_match.orgstart = next_match.strstart;

                if hash_head != 0 && state.recent_hash_distance(hash_head).is_some() {
                    let result = match MatchSearchWork::new(hash_head, MatchSearchKind::Standard)
                        .request(state)
                    {
                        Ok(request) => request.execute(),
                        Err(error) => return ProduceStatus::MatchSearchError(error),
                    };
                    state.match_start = result.start;
                    next_match.match_length = result.length;
                    next_match.match_start = result.start;
                    if next_match.match_start >= next_match.strstart
                        || usize::from(next_match.match_length) < WANT_MIN_MATCH
                    {
                        next_match.match_length = 1;
                    } else {
                        fizzle_matches(
                            state.window.filled(),
                            state.max_dist(),
                            &mut current_match,
                            &mut next_match,
                        );
                    }
                } else {
                    next_match.match_start = 0;
                    next_match.match_length = 1;
                }

                state.lookahead = current_lookahead;
                state.strstart = usize::from(current_match.strstart);
            } else {
                next_match.match_length = 0;
            }

            let block_ready = emit_match(state, current_match);
            state.strstart += usize::from(current_match.match_length);

            if block_ready {
                self.next_match = next_match;
                self.save_next(state);
                return ProduceStatus::BlockReady(BlockContinuation::None);
            }
        }
    }
}

impl MediumProducer {
    /// Prepare the current lazy-match candidate and optionally hand search work to the scheduler.
    /// A None result means another phase can execute immediately on the same buffered window.
    fn prepare_current_search(
        &mut self,
        state: &mut State<'_>,
        allow_short_tail: bool,
    ) -> Option<ProduceStatus> {
        if state.lookahead == 0 || (!allow_short_tail && state.lookahead < MIN_LOOKAHEAD) {
            return Some(ProduceStatus::NeedInput);
        }

        // A future match computed in the previous round already had its hash insertion
        // and search performed. Reuse it directly; otherwise prepare a current search.
        if !self.early_exit && self.next_match.match_length > 0 {
            self.current_match = self.next_match;
            self.next_match.match_length = 0;
            self.phase = MediumPhase::AfterCurrentSearch;
            return None;
        }

        self.current_match.strstart = hash_chain_position(state.strstart);
        self.current_match.orgstart = self.current_match.strstart;

        if self.match_source == MatchSourceKind::External {
            if state.lookahead >= WANT_MIN_MATCH {
                self.phase = MediumPhase::AwaitingCurrentSearch;
                return Some(ProduceStatus::MatchSearch(MatchSearchWork::external()));
            }
        } else {
            let hash_head = if state.lookahead >= WANT_MIN_MATCH {
                StandardHashCalc::quick_insert_string(state, state.strstart)
            } else {
                0
            };
            if hash_head != 0 && state.recent_hash_distance(hash_head).is_some() {
                let work = MatchSearchWork::new(hash_head, MatchSearchKind::Standard);
                if state.max_chain_length <= LOCAL_MATCH_SEARCH_MAX_CHAIN {
                    let result = match work.request(state) {
                        Ok(request) => request.execute(),
                        Err(error) => return Some(ProduceStatus::MatchSearchError(error)),
                    };
                    self.apply_current_search(state, result);
                    self.phase = MediumPhase::AfterCurrentSearch;
                    return None;
                }
                self.phase = MediumPhase::AwaitingCurrentSearch;
                return Some(ProduceStatus::MatchSearch(work));
            }
        }

        self.current_match.match_start = 0;
        self.current_match.match_length = 1;
        self.phase = MediumPhase::AfterCurrentSearch;
        None
    }

    /// Insert the current candidate and speculatively prepare the next match.
    /// Detached requests retain the temporary future cursor until completion restores it.
    fn prepare_future_search(&mut self, state: &mut State<'_>) -> Option<ProduceStatus> {
        insert_match(state, self.current_match, self.match_source);

        // Probe one match ahead. Keep the temporary future position installed while the
        // immutable request is outstanding; completion restores the real current view.
        let current_len = usize::from(self.current_match.match_length);
        let future_start = usize::from(self.current_match.strstart) + current_len;
        let future_lookahead = state.lookahead.saturating_sub(current_len);
        if !self.early_exit
            && state.lookahead > MIN_LOOKAHEAD
            && future_lookahead >= WANT_MIN_MATCH
            && future_start < state.window_size - MIN_LOOKAHEAD
        {
            let current_lookahead = state.lookahead;
            state.strstart = future_start;
            state.lookahead = future_lookahead;
            self.next_match.strstart = hash_chain_position(state.strstart);
            self.next_match.orgstart = self.next_match.strstart;

            if self.match_source == MatchSourceKind::External {
                self.phase = MediumPhase::AwaitingFutureSearch { current_lookahead };
                return Some(ProduceStatus::MatchSearch(MatchSearchWork::external()));
            }

            let hash_head = StandardHashCalc::quick_insert_string(state, state.strstart);
            if hash_head != 0 && state.recent_hash_distance(hash_head).is_some() {
                let work = MatchSearchWork::new(hash_head, MatchSearchKind::Standard);
                if state.max_chain_length <= LOCAL_MATCH_SEARCH_MAX_CHAIN {
                    let result = match work.request(state) {
                        Ok(request) => request.execute(),
                        Err(error) => return Some(ProduceStatus::MatchSearchError(error)),
                    };
                    self.apply_future_search(state, result);
                    state.lookahead = current_lookahead;
                    state.strstart = usize::from(self.current_match.strstart);
                    self.phase = MediumPhase::AfterFutureSearch;
                    return None;
                }
                self.phase = MediumPhase::AwaitingFutureSearch { current_lookahead };
                return Some(ProduceStatus::MatchSearch(work));
            }

            self.next_match.match_start = 0;
            self.next_match.match_length = 1;
            state.lookahead = current_lookahead;
            state.strstart = usize::from(self.current_match.strstart);
        } else {
            self.next_match.match_length = 0;
        }
        self.phase = MediumPhase::AfterFutureSearch;
        None
    }
}

impl TokenProducer for MediumProducer {
    fn produce(&mut self, state: &mut State<'_>, allow_short_tail: bool) -> ProduceStatus {
        if self.match_source == MatchSourceKind::HashChain
            && self.phase == MediumPhase::Ready
            && state.max_chain_length <= LOCAL_MATCH_SEARCH_MAX_CHAIN
        {
            return self.produce_local(state, allow_short_tail);
        }

        loop {
            match self.phase {
                MediumPhase::Ready => {
                    if let Some(status) = self.prepare_current_search(state, allow_short_tail) {
                        return status;
                    }
                }
                MediumPhase::AwaitingCurrentSearch | MediumPhase::AwaitingFutureSearch { .. } => {
                    return ProduceStatus::MatchSearchError(MatchSearchError::OutstandingSearch);
                }
                MediumPhase::AfterCurrentSearch => {
                    if let Some(status) = self.prepare_future_search(state) {
                        return status;
                    }
                }
                MediumPhase::AfterFutureSearch => {
                    let block_ready = emit_match(state, self.current_match);
                    state.strstart += usize::from(self.current_match.match_length);
                    self.phase = MediumPhase::Ready;

                    if block_ready {
                        self.save_next(state);
                        return ProduceStatus::BlockReady(BlockContinuation::None);
                    }
                }
            }
        }
    }

    fn complete_match_search(
        &mut self,
        state: &mut State<'_>,
        result: MatchSearchResult,
    ) -> Result<(), MatchSearchError> {
        match self.phase {
            MediumPhase::AwaitingCurrentSearch => {
                self.apply_current_search(state, result);
                self.phase = MediumPhase::AfterCurrentSearch;
            }
            MediumPhase::AwaitingFutureSearch { current_lookahead } => {
                self.apply_future_search(state, result);
                state.lookahead = current_lookahead;
                state.strstart = usize::from(self.current_match.strstart);
                self.phase = MediumPhase::AfterFutureSearch;
            }
            _ => return Err(MatchSearchError::UnexpectedMatchResult),
        }
        Ok(())
    }

    fn after_refill(&mut self, state: &mut State<'_>) {
        assert_eq!(
            self.phase,
            MediumPhase::Ready,
            "medium refill cannot occur while match work is outstanding"
        );
        // A refill may slide the window, invalidating a speculative absolute position.
        self.clear_saved_next(state);
    }

    fn finish(&mut self, state: &mut State<'_>) -> usize {
        assert_eq!(
            self.phase,
            MediumPhase::Ready,
            "medium producer cannot finish while match search is outstanding"
        );
        self.clear_saved_next(state);
        state.strstart.min(STD_MIN_MATCH - 1)
    }
}

fn emit_match(state: &mut State, m: Match) -> bool {
    let mut bflush = false;

    /* matches that are not long enough we need to emit as literals */
    if (m.match_length as usize) < WANT_MIN_MATCH {
        for lc in &state.window.filled()[state.strstart..][..m.match_length as usize] {
            bflush |= State::tally_lit_help(&mut state.sym_buf, &mut state.l_desc, *lc);
        }
    } else {
        // check_match(s, m.strstart, m.match_start, m.match_length);

        bflush |= state.tally_dist(
            (m.strstart - m.match_start) as usize,
            m.match_length as usize - STD_MIN_MATCH,
        );
    }
    state.lookahead -= m.match_length as usize;

    bflush
}

#[inline]
fn insert_match(state: &mut State, mut m: Match, match_source: MatchSourceKind) {
    if match_source == MatchSourceKind::External {
        return;
    }

    if state.lookahead <= (m.match_length as usize + WANT_MIN_MATCH) {
        return;
    }

    /* matches that are not long enough we need to emit as literals */
    if (m.match_length as usize) < WANT_MIN_MATCH {
        m.strstart += 1;
        m.match_length -= 1;
        if m.match_length > 0 && m.strstart >= m.orgstart {
            if m.strstart + m.match_length > m.orgstart {
                state.insert_string(m.strstart as usize, m.match_length as usize);
            } else {
                state.insert_string(m.strstart as usize, (m.orgstart - m.strstart + 1) as usize);
            }
        }
        return;
    }

    // Insert new strings in the hash table only if the match length
    // is not too large. This saves time but degrades compression.
    if usize::from(m.match_length) <= 16 * usize::from(state.max_insert_length)
        && state.lookahead >= WANT_MIN_MATCH
    {
        m.match_length -= 1; /* string at strstart already in table */
        m.strstart += 1;

        if m.strstart >= m.orgstart {
            if m.strstart + m.match_length > m.orgstart {
                state.insert_string(m.strstart as usize, m.match_length as usize);
            } else {
                state.insert_string(m.strstart as usize, (m.orgstart - m.strstart + 1) as usize);
            }
        } else if m.orgstart < m.strstart + m.match_length {
            state.insert_string(
                m.orgstart as usize,
                (m.strstart + m.match_length - m.orgstart) as usize,
            );
        }
    } else {
        m.strstart += m.match_length;
        m.match_length = 0;

        if (m.strstart as usize) >= (STD_MIN_MATCH - 2) {
            StandardHashCalc::quick_insert_string(state, m.strstart as usize + 2 - STD_MIN_MATCH);
        }

        /* If lookahead < WANT_MIN_MATCH, ins_h is garbage, but it does not
         * matter since it will be recomputed at next deflate call.
         */
    }
}

fn fizzle_matches(window: &[u8], max_dist: usize, current: &mut Match, next: &mut Match) {
    /* step zero: sanity checks */

    if current.match_length <= 1 {
        return;
    }

    // A possible leftward shift must remain inside both history positions.
    // Keep this in unsigned address space; signed intermediates could wrap
    // before being cast back into a window index.
    let shift = usize::from(current.match_length - 1);
    let Some(match_start) = usize::from(next.match_start).checked_sub(shift) else {
        return;
    };
    let Some(origin_start) = usize::from(next.strstart).checked_sub(shift) else {
        return;
    };
    let (Some(&match_byte), Some(&origin_byte)) =
        (window.get(match_start), window.get(origin_start))
    else {
        return;
    };

    /* quick exit check.. if this fails then don't bother with anything else */
    if match_byte != origin_byte {
        return;
    }

    /* step one: try to move the "next" match to the left as much as possible */
    let limit = next
        .strstart
        .saturating_sub(u16::try_from(max_dist).unwrap_or(u16::MAX));

    let mut c = *current;
    let mut n = *next;
    let (Some(m), Some(orig)) = (
        window.get(..usize::from(n.match_start)),
        window.get(..usize::from(n.strstart)),
    ) else {
        return;
    };

    let mut m = m.iter().rev();
    let mut orig = orig.iter().rev();

    let mut changed = 0;

    while m.next() == orig.next() {
        if c.match_length < 1 {
            break;
        }
        if n.strstart <= limit {
            break;
        }
        if n.match_length >= 256 {
            break;
        }
        if n.match_start <= 1 {
            break;
        }

        n.strstart -= 1;
        n.match_start -= 1;
        n.match_length += 1;
        c.match_length -= 1;
        changed += 1;
    }

    if changed == 0 {
        return;
    }

    if c.match_length <= 1 && n.match_length != 2 {
        n.orgstart += 1;
        *current = c;
        *next = n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_result_completion_requires_an_awaiting_phase() -> Result<(), crate::ReturnCode> {
        use crate::deflate::{DeflateConfig, DeflateStream};
        let mut stream = DeflateStream::new(DeflateConfig::new(6))?;
        let state = &mut *stream.state;
        let mut producer = MediumProducer::new(state, MatchSourceKind::HashChain);
        let result = MatchSearchResult::new(0, 0);

        for phase in [
            MediumPhase::Ready,
            MediumPhase::AfterCurrentSearch,
            MediumPhase::AfterFutureSearch,
        ] {
            producer.phase = phase;
            assert_eq!(
                producer.complete_match_search(state, result),
                Err(MatchSearchError::UnexpectedMatchResult)
            );
            assert_eq!(producer.phase, phase);
        }
        producer.phase = MediumPhase::AwaitingCurrentSearch;
        assert_eq!(producer.complete_match_search(state, result), Ok(()));
        assert_eq!(producer.phase, MediumPhase::AfterCurrentSearch);
        producer.phase = MediumPhase::AwaitingFutureSearch {
            current_lookahead: state.lookahead,
        };
        assert_eq!(producer.complete_match_search(state, result), Ok(()));
        assert_eq!(producer.phase, MediumPhase::AfterFutureSearch);
        Ok(())
    }

    #[test]
    fn producing_with_unfinished_search_returns_typed_error() -> Result<(), crate::ReturnCode> {
        use crate::deflate::{DeflateConfig, DeflateStream};

        let mut stream = DeflateStream::new(DeflateConfig::new(6))?;
        let state = &mut *stream.state;
        let mut producer = MediumProducer::new(state, MatchSourceKind::HashChain);

        for phase in [
            MediumPhase::AwaitingCurrentSearch,
            MediumPhase::AwaitingFutureSearch {
                current_lookahead: 11,
            },
        ] {
            producer.phase = phase;
            assert!(matches!(
                producer.produce(state, false),
                ProduceStatus::MatchSearchError(MatchSearchError::OutstandingSearch)
            ));
        }
        Ok(())
    }

    #[test]
    fn fizzle_uses_valid_history_indices_and_preserves_match_distance() {
        let history = [b'a'; 32];
        let current = Match {
            match_start: 4,
            match_length: 4,
            strstart: 10,
            orgstart: 10,
        };
        let next = Match {
            match_start: 8,
            match_length: 6,
            strstart: 14,
            orgstart: 14,
        };

        let mut left = current;
        let mut right = next;
        fizzle_matches(&history, 64, &mut left, &mut right);
        assert!(left.match_length < current.match_length);
        assert!(right.match_length > next.match_length);
        assert_eq!(
            right.strstart - right.match_start,
            next.strstart - next.match_start
        );

        let mut malformed_next = next;
        malformed_next.match_start = u16::MAX;
        malformed_next.strstart = u16::MAX;
        let mut unchanged_current = current;
        fizzle_matches(&history, 64, &mut unchanged_current, &mut malformed_next);
        assert_eq!(unchanged_current, current);
        assert_eq!(malformed_next.match_start, u16::MAX);

        let mut underflow_next = next;
        underflow_next.match_start = 1;
        let mut unchanged_current = current;
        fizzle_matches(&history, 64, &mut unchanged_current, &mut underflow_next);
        assert_eq!(unchanged_current, current);
        assert_eq!(underflow_next.match_start, 1);
    }
}
