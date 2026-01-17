use crate::{BitLength, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut};
use sorted_iter::assume::AssumeSortedByItemExt;

pub trait BitwiseTruncated<Inner: ?Sized>
where
    Self: AsRef<Inner> + BitLength,
    Inner: Bitwise,
{
    #[inline]
    fn index(&self, index: usize) -> bool {
        assert!(index < self.bit_len());
        self.as_ref().index(index)
    }
    #[inline]
    fn weight(&self) -> usize {
        self.as_ref().weight()
    }
    #[inline]
    fn parity(&self) -> bool {
        self.as_ref().parity()
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.as_ref().is_zero()
    }
    #[inline]
    fn is_unit(&self, index: usize) -> bool {
        assert!(index < self.bit_len());
        self.as_ref().is_unit(index)
    }
    #[inline]
    fn support<'a>(&'a self) -> impl sorted_iter::SortedIterator<Item = usize>
    where
        Inner: 'a,
    {
        self.as_ref().support().take(self.bit_len()).assume_sorted_by_item()
    }
    #[inline]
    fn max_support(&self) -> Option<usize> {
        self.as_ref().max_support()
    }
    #[inline]
    fn min_support(&self) -> Option<usize> {
        self.as_ref().min_support()
    }
}

impl<T, Inner> BitwiseTruncated<Inner> for T
where
    T: AsRef<Inner> + BitLength,
    Inner: ?Sized + Bitwise,
{
}

pub trait BitwiseMutTruncated<Inner: ?Sized>
where
    Self: AsMut<Inner> + BitLength,
    Inner: BitwiseMut,
{
    #[inline]
    fn assign_index(&mut self, index: usize, to: bool) {
        assert!(index < self.bit_len());
        self.as_mut().assign_index(index, to);
    }

    #[inline]
    fn negate_index(&mut self, index: usize) {
        assert!(index < self.bit_len());
        self.as_mut().negate_index(index);
    }

    #[inline]
    fn clear_bits(&mut self) {
        self.as_mut().clear_bits();
    }

    fn assign_random(&mut self, bit_count: usize, random_number_generator: &mut impl rand::Rng)
    where
        Self: BitLength,
    {
        for j in 0..bit_count {
            self.assign_index(j, random_number_generator.r#gen());
        }
    }
}

impl<T, Inner> BitwiseMutTruncated<Inner> for T
where
    T: AsMut<Inner> + BitLength,
    Inner: ?Sized + BitwiseMut,
{
}

pub trait BitwisePairTruncated<Other: ?Sized, Inner: ?Sized, InnerOther: ?Sized = Inner>
where
    Self: AsRef<Inner> + BitLength,
    Other: AsRef<InnerOther> + BitLength,
    Inner: Bitwise,
    InnerOther: Bitwise,
    Inner: BitwisePair<InnerOther>,
{
    #[inline]
    fn dot(&self, other: &Other) -> bool {
        assert_eq!(self.bit_len(), other.bit_len());
        self.as_ref().dot(other.as_ref())
    }
    #[inline]
    fn and_weight(&self, other: &Other) -> usize {
        assert_eq!(self.bit_len(), other.bit_len());
        self.as_ref().and_weight(other.as_ref())
    }
    #[inline]
    fn or_weight(&self, other: &Other) -> usize {
        assert_eq!(self.bit_len(), other.bit_len());
        self.as_ref().or_weight(other.as_ref())
    }
    #[inline]
    fn xor_weight(&self, other: &Other) -> usize {
        assert_eq!(self.bit_len(), other.bit_len());
        self.as_ref().xor_weight(other.as_ref())
    }
}

impl<T, Other, Inner, InnerOther> BitwisePairTruncated<Other, Inner, InnerOther> for T
where
    T: ?Sized + AsRef<Inner> + BitLength,
    Other: ?Sized + AsRef<InnerOther> + BitLength,
    Inner: ?Sized + Bitwise,
    InnerOther: ?Sized + Bitwise,
    Inner: BitwisePair<InnerOther>,
{
}

pub trait BitwisePairMutTruncated<Other: ?Sized, BorrowedSelf: ?Sized, BorrowedOther: ?Sized = BorrowedSelf>
where
    Self: AsMut<BorrowedSelf> + BitLength,
    Other: AsRef<BorrowedOther> + BitLength,
    BorrowedSelf: Bitwise,
    BorrowedOther: Bitwise,
    BorrowedSelf: BitwisePairMut<BorrowedOther>,
{
    #[inline]
    fn assign(&mut self, other: &Other) {
        assert_eq!(self.bit_len(), other.bit_len());
        self.as_mut().assign(other.as_ref());
    }
    #[inline]
    fn bitxor_assign(&mut self, other: &Other) {
        assert_eq!(self.bit_len(), other.bit_len());
        self.as_mut().bitxor_assign(other.as_ref());
    }
    #[inline]
    fn bitand_assign(&mut self, other: &Other) {
        assert_eq!(self.bit_len(), other.bit_len());
        self.as_mut().bitand_assign(other.as_ref());
    }
    #[inline]
    fn bitor_assign(&mut self, other: &Other) {
        assert_eq!(self.bit_len(), other.bit_len());
        self.as_mut().bitor_assign(other.as_ref());
    }
}

impl<Bits, Other, BorrowedSelf, BorrowedOther> BitwisePairMutTruncated<Other, BorrowedSelf, BorrowedOther> for Bits
where
    Bits: ?Sized + AsMut<BorrowedSelf> + BitLength,
    Other: ?Sized + AsRef<BorrowedOther> + BitLength,
    BorrowedSelf: ?Sized + Bitwise,
    BorrowedOther: ?Sized + Bitwise,
    BorrowedSelf: BitwisePairMut<BorrowedOther>,
{
}
