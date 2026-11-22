#![forbid(unsafe_code)]

/// Compact Huffman lookup entry. Operation zero carries a byte (or a
/// code-length alphabet symbol), never an arbitrary u16. All construction
/// crosses this invariant boundary, so consumers need not revalidate literals.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Code {
    op: u8,
    bits: u8,
    val: u16,
}

impl Code {
    const fn valid_parts(op: u8, val: u16) -> bool {
        op != 0 || val <= 255
    }

    /// Checked construction for entries derived from an input stream.
    #[inline]
    pub(crate) const fn try_new(op: u8, bits: u8, val: u16) -> Option<Self> {
        if Self::valid_parts(op, val) {
            Some(Self { op, bits, val })
        } else {
            None
        }
    }

    /// Constant table entries and known internal sentinel entries.
    pub(crate) const fn new(op: u8, bits: u8, val: u16) -> Self {
        assert!(Self::valid_parts(op, val), "literal must be a byte");
        Self { op, bits, val }
    }

    #[inline]
    pub(crate) const fn op(self) -> u8 {
        self.op
    }

    #[inline]
    pub(crate) const fn bits(self) -> u8 {
        self.bits
    }

    #[inline]
    pub(crate) const fn val(self) -> u16 {
        self.val
    }

    /// The representation invariant makes extraction lossless for op-zero entries.
    #[inline]
    pub(crate) const fn literal(self) -> Option<u8> {
        if self.op == 0 {
            Some(self.val.to_le_bytes()[0])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Code;

    #[test]
    fn literal_invariant_is_checked_at_construction() {
        assert_eq!(core::mem::size_of::<Code>(), 4);
        for value in 0..=u16::MAX {
            assert_eq!(
                Code::try_new(0, 9, value).and_then(Code::literal),
                u8::try_from(value).ok()
            );
            assert_eq!(Code::try_new(16, 9, value).map(Code::val), Some(value));
        }
        assert_eq!(Code::new(96, 7, 0).literal(), None);
        assert_eq!(Code::default().literal(), Some(0));
    }
}
