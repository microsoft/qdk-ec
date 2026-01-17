use crate::{BitLength, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut};
use std::borrow::{Borrow, BorrowMut};

pub trait BitwiseViaBorrow<BorrowedSelf: ?Sized>
where
    Self: std::borrow::Borrow<BorrowedSelf>,
    BorrowedSelf: Bitwise,
{
    #[inline]
    fn index(&self, index: usize) -> bool {
        self.borrow().index(index)
    }
    #[inline]
    fn weight(&self) -> usize {
        self.borrow().weight()
    }
    #[inline]
    fn parity(&self) -> bool {
        self.borrow().parity()
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.borrow().is_zero()
    }
    #[inline]
    fn is_unit(&self, index: usize) -> bool {
        self.borrow().is_unit(index)
    }
    #[inline]
    fn support<'a>(&'a self) -> impl sorted_iter::SortedIterator<Item = usize>
    where
        BorrowedSelf: 'a,
    {
        self.borrow().support()
    }
    #[inline]
    fn max_support(&self) -> Option<usize> {
        self.borrow().max_support()
    }
    #[inline]
    fn min_support(&self) -> Option<usize> {
        self.borrow().min_support()
    }
}

impl<T, BorrowedSelf> BitwiseViaBorrow<BorrowedSelf> for T
where
    T: std::borrow::Borrow<BorrowedSelf>,
    BorrowedSelf: ?Sized + Bitwise,
{
}

pub trait BitwiseMutViaBorrow<BorrowedSelf: ?Sized>
where
    Self: std::borrow::BorrowMut<BorrowedSelf>,
    BorrowedSelf: BitwiseMut,
{
    #[inline]
    fn assign_index(&mut self, index: usize, to: bool) {
        self.borrow_mut().assign_index(index, to);
    }

    #[inline]
    fn negate_index(&mut self, index: usize) {
        self.borrow_mut().negate_index(index);
    }

    #[inline]
    fn clear_bits(&mut self) {
        self.borrow_mut().clear_bits();
    }

    #[inline]
    fn assign_random(&mut self, bit_count: usize, random_number_generator: &mut impl rand::Rng)
    where
        BorrowedSelf: BitLength,
    {
        self.borrow_mut().assign_random(bit_count, random_number_generator);
    }
}

impl<T, BorrowedSelf> BitwiseMutViaBorrow<BorrowedSelf> for T
where
    T: std::borrow::BorrowMut<BorrowedSelf>,
    BorrowedSelf: ?Sized + BitwiseMut,
{
}

pub trait BitwisePairViaBorrow<Other: ?Sized, BorrowedSelf: ?Sized, BorrowedOther: ?Sized = BorrowedSelf>
where
    Self: Borrow<BorrowedSelf>,
    Other: Borrow<BorrowedOther>,
    BorrowedSelf: Bitwise,
    BorrowedOther: Bitwise,
    BorrowedSelf: BitwisePair<BorrowedOther>,
{
    #[inline]
    fn dot(&self, other: &Other) -> bool {
        self.borrow().dot(other.borrow())
    }
    #[inline]
    fn and_weight(&self, other: &Other) -> usize {
        self.borrow().and_weight(other.borrow())
    }
    #[inline]
    fn or_weight(&self, other: &Other) -> usize {
        self.borrow().or_weight(other.borrow())
    }
    #[inline]
    fn xor_weight(&self, other: &Other) -> usize {
        self.borrow().xor_weight(other.borrow())
    }
}

impl<T, Other, BorrowedSelf, BorrowedOther> BitwisePairViaBorrow<Other, BorrowedSelf, BorrowedOther> for T
where
    T: ?Sized + Borrow<BorrowedSelf>,
    Other: ?Sized + Borrow<BorrowedOther>,
    BorrowedSelf: ?Sized + Bitwise,
    BorrowedOther: ?Sized + Bitwise,
    BorrowedSelf: BitwisePair<BorrowedOther>,
{
}

pub trait BitwisePairMutViaBorrow<Other: ?Sized, BorrowedSelf: ?Sized, BorrowedOther: ?Sized = BorrowedSelf>
where
    Self: BorrowMut<BorrowedSelf>,
    Other: Borrow<BorrowedOther>,
    BorrowedSelf: Bitwise,
    BorrowedOther: Bitwise,
    BorrowedSelf: BitwisePairMut<BorrowedOther>,
{
    #[inline]
    fn assign(&mut self, other: &Other) {
        self.borrow_mut().assign(other.borrow());
    }
    #[inline]
    fn bitxor_assign(&mut self, other: &Other) {
        self.borrow_mut().bitxor_assign(other.borrow());
    }
    #[inline]
    fn bitand_assign(&mut self, other: &Other) {
        self.borrow_mut().bitand_assign(other.borrow());
    }
    #[inline]
    fn bitor_assign(&mut self, other: &Other) {
        self.borrow_mut().bitor_assign(other.borrow());
    }
}

impl<Bits, Other, BorrowedSelf, BorrowedOther> BitwisePairMutViaBorrow<Other, BorrowedSelf, BorrowedOther> for Bits
where
    Bits: ?Sized + BorrowMut<BorrowedSelf>,
    Other: ?Sized + Borrow<BorrowedOther>,
    BorrowedSelf: ?Sized + Bitwise,
    BorrowedOther: ?Sized + Bitwise,
    BorrowedSelf: BitwisePairMut<BorrowedOther>,
{
}
