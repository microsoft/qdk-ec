use crate::bit::bitwise_for_slice as slice;
use crate::bit::bitwise_via_borrow as borrow;
use crate::bit::unsigned_integers::BitIterator;
use crate::{BitLength, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IntoBitIterator};
use std::borrow::{Borrow, BorrowMut};
use std::ops::{Deref, DerefMut};

pub const BIT_BLOCK_WORD_COUNT: usize = 8usize;
pub type Word = u64;
pub type BitBlockInnerArray = [Word; BIT_BLOCK_WORD_COUNT];

/// `BitBlock` is designed so that LLVM can optimize operations on it using SIMD instructions.
/// To see the extent to which this is true run
/// `cargo asm "<binar::bit::bitblock::BitBlock as binar::bit::bitwise::BitwisePairMut>::bitxor_assign"`
#[repr(C, align(64))]
#[derive(Eq, Clone, Debug, Hash, PartialEq, Default)]
pub struct BitBlock {
    pub blocks: BitBlockInnerArray,
}

impl Borrow<BitBlockInnerArray> for BitBlock {
    fn borrow(&self) -> &BitBlockInnerArray {
        &self.blocks
    }
}

impl BorrowMut<BitBlockInnerArray> for BitBlock {
    fn borrow_mut(&mut self) -> &mut BitBlockInnerArray {
        &mut self.blocks
    }
}

impl Deref for BitBlock {
    type Target = BitBlockInnerArray;
    fn deref(&self) -> &Self::Target {
        &self.blocks
    }
}

impl DerefMut for BitBlock {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.blocks
    }
}

impl BitLength for BitBlock {
    fn bit_len(&self) -> usize {
        Self::BLOCK_BIT_LEN
    }
    const BLOCK_BIT_LEN: usize = 8 * size_of::<BitBlockInnerArray>();
}

delegate_bitwise!(BitBlock, borrow::BitwiseViaBorrow<BitBlockInnerArray>);
delegate_bitwise_mut!(BitBlock, borrow::BitwiseMutViaBorrow<BitBlockInnerArray>);
delegate_bitwise_pair!(BitBlock, BitBlock, borrow::BitwisePairViaBorrow<BitBlockInnerArray, BitBlockInnerArray>);
delegate_bitwise_pair_mut!(BitBlock, BitBlock, borrow::BitwisePairMutViaBorrow<BitBlockInnerArray, BitBlockInnerArray>);

impl From<BitBlockInnerArray> for BitBlock {
    fn from(blocks: BitBlockInnerArray) -> Self {
        Self { blocks }
    }
}

impl IntoBitIterator for BitBlock {
    type BitIterator = <BitBlockInnerArray as IntoBitIterator>::BitIterator;

    fn iter_bits(self) -> Self::BitIterator {
        self.blocks.iter_bits()
    }
}

impl<'life> IntoBitIterator for &'life BitBlock {
    type BitIterator = <&'life BitBlockInnerArray as IntoBitIterator>::BitIterator;
    fn iter_bits(self) -> Self::BitIterator {
        <BitBlock as Borrow<BitBlockInnerArray>>::borrow(self).iter_bits()
    }
}

impl<'life> IntoIterator for &'life BitBlock {
    type Item = bool;
    type IntoIter = BitIterator<'life>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_bits()
    }
}

impl BitBlock {
    #[must_use]
    pub fn iter(&self) -> BitIterator<'_> {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl FromIterator<bool> for BitBlock {
    fn from_iter<Iterator: IntoIterator<Item = bool>>(iterator: Iterator) -> Self {
        let mut res: BitBlock = BitBlock::default();
        for (index, bit) in iterator.into_iter().enumerate() {
            res.assign_index(index, bit);
        }
        res
    }
}

// Slices of Bitblocks

delegate_bitwise!([BitBlock], slice::BitwiseForSlice<BitBlock>);
delegate_bitwise_mut!([BitBlock], slice::BitwiseMutForSlice<BitBlock>);
delegate_bitwise_pair!([BitBlock], [BitBlock], slice::BitwisePairForSlice<BitBlock>);
delegate_bitwise_pair_mut!([BitBlock], [BitBlock], slice::BitwisePairMutForSlice<BitBlock>);

impl<'life> IntoBitIterator for &'life [BitBlock] {
    type BitIterator = <Self as slice::IntoBitIteratorForSlice<'life, BitBlock>>::BitIterator;

    fn iter_bits(self) -> Self::BitIterator {
        <Self as slice::IntoBitIteratorForSlice<BitBlock>>::iter_bits(self)
    }
}

impl BitLength for [BitBlock] {
    fn bit_len(&self) -> usize {
        self.len() * BitBlock::BLOCK_BIT_LEN
    }
    const BLOCK_BIT_LEN: usize = BitBlock::BLOCK_BIT_LEN;
}

impl BitBlock {
    /// [`BitBlock`] with all bits set to one.
    #[must_use]
    pub fn ones() -> Self {
        [u64::MAX; BIT_BLOCK_WORD_COUNT].into()
    }

    /// Set all bits to zero.
    pub fn clear(&mut self) {
        self.blocks.fill(0);
    }
}

#[must_use]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct BitAccessor {
    word_index: usize,
    bitmask: Word,
}

impl BitAccessor {
    /// # Panics
    ///
    /// Will panic index is out of range
    pub fn for_index<T>(index: usize) -> Self
    where
        T: BitLength,
    {
        assert!(index < T::BLOCK_BIT_LEN);
        unsafe { Self::for_index_unchecked(index) }
    }

    /// # Safety
    /// Does not check if index is out of bounds
    pub unsafe fn for_index_unchecked(index: usize) -> Self {
        let word_index = index / (Word::BITS as usize);
        let bit_index = index % (Word::BITS as usize);
        Self {
            word_index,
            bitmask: 1 << bit_index,
        }
    }

    #[must_use]
    pub fn array_value_of<const SIZE: usize>(&self, block: &[Word; SIZE]) -> bool {
        let word = unsafe { block.get_unchecked(self.word_index) };
        (*word & self.bitmask) != 0
    }

    pub fn array_bitxor<const SIZE: usize>(&self, block: &mut [Word; SIZE]) {
        let word: &mut Word = unsafe { block.get_unchecked_mut(self.word_index) };
        *word ^= self.bitmask;
    }

    pub fn array_set_value_of<const SIZE: usize>(&self, block: &mut [Word; SIZE], to: bool) {
        let word = unsafe { block.get_unchecked_mut(self.word_index) };
        if to {
            *word |= self.bitmask;
        } else {
            *word &= !self.bitmask;
        }
    }
}
