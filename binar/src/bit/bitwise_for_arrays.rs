use crate::{BitwisePair, BitwisePairMut};
use std::borrow::{Borrow, BorrowMut};

pub trait BitwisePairForArray<const SIZE: usize, BitBlock>
where
    Self: Borrow<[BitBlock; SIZE]>,
    BitBlock: BitwisePair,
{
    #[inline]
    fn dot(&self, other: &Self) -> bool {
        let mut result = 0usize;
        for block_id in 0..SIZE {
            unsafe {
                result += usize::from(
                    self.borrow()
                        .get_unchecked(block_id)
                        .dot(other.borrow().get_unchecked(block_id)),
                );
            }
        }
        result % 2 == 1
    }
    #[inline]
    fn and_weight(&self, other: &Self) -> usize {
        let mut weight = 0usize;
        for block_id in 0..SIZE {
            unsafe {
                weight += self
                    .borrow()
                    .get_unchecked(block_id)
                    .and_weight(other.borrow().get_unchecked(block_id));
            }
        }
        weight
    }
    #[inline]
    fn or_weight(&self, other: &Self) -> usize {
        let mut weight = 0usize;
        for block_id in 0..SIZE {
            unsafe {
                weight += self
                    .borrow()
                    .get_unchecked(block_id)
                    .or_weight(other.borrow().get_unchecked(block_id));
            }
        }
        weight
    }
    #[inline]
    fn xor_weight(&self, other: &Self) -> usize {
        let mut weight = 0usize;
        for block_id in 0..SIZE {
            unsafe {
                weight += self
                    .borrow()
                    .get_unchecked(block_id)
                    .xor_weight(other.borrow().get_unchecked(block_id));
            }
        }
        weight
    }
}

impl<const SIZE: usize, Bits, BitBlock> BitwisePairForArray<SIZE, BitBlock> for Bits
where
    Bits: ?Sized + Borrow<[BitBlock; SIZE]>,
    BitBlock: BitwisePair,
{
}

pub trait BitwisePairMutForArray<const SIZE: usize, BitBlock>
where
    Self: BorrowMut<[BitBlock; SIZE]>,
    BitBlock: BitwisePairMut,
{
    #[inline]
    fn assign(&mut self, other: &Self) {
        for block_id in 0..SIZE {
            unsafe {
                self.borrow_mut()
                    .get_unchecked_mut(block_id)
                    .assign(other.borrow().get_unchecked(block_id));
            }
        }
    }
    #[inline]
    fn bitxor_assign(&mut self, other: &Self) {
        for block_id in 0..SIZE {
            unsafe {
                self.borrow_mut()
                    .get_unchecked_mut(block_id)
                    .bitxor_assign(other.borrow().get_unchecked(block_id));
            }
        }
    }
    #[inline]
    fn bitand_assign(&mut self, other: &Self) {
        for block_id in 0..SIZE {
            unsafe {
                self.borrow_mut()
                    .get_unchecked_mut(block_id)
                    .bitand_assign(other.borrow().get_unchecked(block_id));
            }
        }
    }
    #[inline]
    fn bitor_assign(&mut self, other: &Self) {
        for block_id in 0..SIZE {
            unsafe {
                self.borrow_mut()
                    .get_unchecked_mut(block_id)
                    .bitor_assign(other.borrow().get_unchecked(block_id));
            }
        }
    }
}

impl<const SIZE: usize, Bits, BitBlock> BitwisePairMutForArray<SIZE, BitBlock> for Bits
where
    Bits: ?Sized + BorrowMut<[BitBlock; SIZE]>,
    BitBlock: BitwisePairMut,
{
}
