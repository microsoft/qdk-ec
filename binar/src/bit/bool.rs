use crate::{BitLength, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IntoBitIterator};

impl Bitwise for bool {
    #[inline]
    fn index(&self, index: usize) -> bool {
        assert_eq!(index, 0);
        *self
    }

    #[inline]
    fn weight(&self) -> usize {
        usize::from(*self)
    }

    #[inline]
    fn support(&self) -> impl sorted_iter::SortedIterator<Item = usize> {
        if *self { 0..1 } else { 0..0 }
    }
}

impl BitwiseMut for bool {
    #[inline]
    fn assign_index(&mut self, index: usize, to: bool) {
        assert_eq!(index, 0);
        *self = to;
    }

    #[inline]
    fn negate_index(&mut self, index: usize) {
        assert_eq!(index, 0);
        *self ^= true;
    }

    #[inline]
    fn clear_bits(&mut self) {
        *self = false;
    }
}

impl BitwisePair for bool {
    #[inline]
    fn dot(&self, other: &bool) -> bool {
        *self && *other
    }

    #[inline]
    fn and_weight(&self, other: &bool) -> usize {
        usize::from(*self && *other)
    }

    #[inline]
    fn or_weight(&self, other: &bool) -> usize {
        usize::from(*self || *other)
    }

    #[inline]
    fn xor_weight(&self, other: &bool) -> usize {
        usize::from(*self ^ *other)
    }
}

impl BitwisePairMut for bool {
    #[inline]
    fn assign(&mut self, other: &bool) {
        *self = *other;
    }

    #[inline]
    fn bitxor_assign(&mut self, other: &bool) {
        *self ^= *other;
    }

    #[inline]
    fn bitand_assign(&mut self, other: &Self) {
        *self &= *other;
    }

    #[inline]
    fn bitor_assign(&mut self, other: &Self) {
        *self |= *other;
    }
}

impl IntoBitIterator for &bool {
    type BitIterator = std::iter::Once<bool>;
    fn iter_bits(self) -> Self::BitIterator {
        std::iter::once(*self)
    }
}

impl BitLength for bool {
    #[inline]
    fn bit_len(&self) -> usize {
        1
    }
    const BLOCK_BIT_LEN: usize = 1;
}
