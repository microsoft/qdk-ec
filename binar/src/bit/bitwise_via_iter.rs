use crate::{BitLength, BitwiseMut, IntoBitIterator};

pub trait BitwisePairViaIter<Other: ?Sized>
where
    Self: BitLength,
    Other: BitLength,
    for<'life> &'life Self: IntoBitIterator,
    for<'life> &'life Other: IntoBitIterator,
{
    #[inline]
    fn dot(&self, other: &Other) -> bool {
        self.iter_bits()
            .zip(other.iter_bits())
            .map(|(a, b)| usize::from(a & b))
            .sum::<usize>()
            % 2
            == 1
    }
    #[inline]
    fn and_weight(&self, other: &Other) -> usize {
        self.iter_bits()
            .zip(other.iter_bits())
            .map(|(a, b)| usize::from(a & b))
            .sum::<usize>()
    }
    #[inline]
    fn or_weight(&self, other: &Other) -> usize {
        self.iter_bits()
            .zip(other.iter_bits())
            .map(|(a, b)| usize::from(a | b))
            .sum::<usize>()
    }
    #[inline]
    fn xor_weight(&self, other: &Other) -> usize {
        self.iter_bits()
            .zip(other.iter_bits())
            .map(|(a, b)| usize::from(a ^ b))
            .sum::<usize>()
    }
}

impl<T: ?Sized, Other: ?Sized> BitwisePairViaIter<Other> for T
where
    Self: BitLength,
    Other: BitLength,
    for<'life> &'life Self: IntoBitIterator<BitIterator: 'life>,
    for<'life> &'life Other: IntoBitIterator<BitIterator: 'life>,
{
}

fn assert_length_at_least<T1: BitLength + ?Sized, T2: BitLength + ?Sized>(left: &T1, right: &T2) {
    assert!(
        left.bit_len() >= right.bit_len(),
        "Left-hand side ({} bits) to be at least as long as the right-hand side ({} bits)",
        left.bit_len(),
        right.bit_len()
    );
}

pub trait BitwisePairMutViaIter<Other: ?Sized>
where
    Self: BitwiseMut + BitLength,
    Other: BitLength,
    for<'life> &'life Other: IntoBitIterator,
{
    #[inline]
    fn assign(&mut self, other: &Other) {
        assert_length_at_least(self, other);
        for (index, bit) in other.iter_bits().enumerate() {
            self.assign_index(index, bit);
        }
    }
    #[inline]
    fn bitxor_assign(&mut self, other: &Other) {
        assert_length_at_least(self, other);
        for (index, bit) in other.iter_bits().enumerate() {
            if bit {
                self.negate_index(index);
            }
        }
    }
    #[inline]
    fn bitand_assign(&mut self, other: &Other) {
        assert_length_at_least(self, other);
        for (index, bit) in other.iter_bits().enumerate() {
            if !bit {
                self.assign_index(index, false);
            }
        }
    }
    #[inline]
    fn bitor_assign(&mut self, other: &Other) {
        assert_length_at_least(self, other);
        for (index, bit) in other.iter_bits().enumerate() {
            if bit {
                self.assign_index(index, true);
            }
        }
    }
}

impl<Bits: ?Sized, Other: ?Sized> BitwisePairMutViaIter<Other> for Bits
where
    Bits: BitwiseMut + BitLength,
    Other: BitLength,
    for<'life> &'life Other: IntoBitIterator,
{
}
