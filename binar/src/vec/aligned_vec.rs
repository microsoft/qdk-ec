use core::iter::FlatMap;
use std::borrow::{Borrow, BorrowMut};
use std::hash::Hash;
use std::ops::{Shl, ShlAssign};

use crate::bit::bitblock::{BIT_BLOCK_WORD_COUNT, Word};
use crate::bit::bitwise_via_borrow as borrow;
use crate::vec::aligned_view::{AlignedBitView, AlignedBitViewMut};
use crate::{BitBlock, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IntoBitIterator};
use crate::{
    BitLength, delegate_bitwise, delegate_bitwise_body, delegate_bitwise_mut, delegate_bitwise_mut_body,
    delegate_bitwise_pair, delegate_bitwise_pair_body, delegate_bitwise_pair_mut, delegate_bitwise_pair_mut_body,
    into_iterator_via_bit_iterator_body,
};

#[must_use]
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct AlignedBitVec {
    pub(crate) blocks: Vec<BitBlock>,
}

impl Hash for AlignedBitVec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.blocks.hash(state);
    }
}

impl Borrow<[BitBlock]> for AlignedBitVec {
    fn borrow(&self) -> &[BitBlock] {
        &self.blocks
    }
}

impl BorrowMut<[BitBlock]> for AlignedBitVec {
    fn borrow_mut(&mut self) -> &mut [BitBlock] {
        &mut self.blocks
    }
}

impl BitLength for AlignedBitVec {
    fn bit_len(&self) -> usize {
        self.blocks.bit_len()
    }
    const BLOCK_BIT_LEN: usize = BitBlock::BLOCK_BIT_LEN;
}

delegate_bitwise!(AlignedBitVec, borrow::BitwiseViaBorrow<[BitBlock]>);
delegate_bitwise_mut!(AlignedBitVec, borrow::BitwiseMutViaBorrow<[BitBlock]>);
delegate_bitwise_pair!(AlignedBitVec, AlignedBitVec, borrow::BitwisePairViaBorrow<AlignedBitVec, [BitBlock]>);
delegate_bitwise_pair_mut!(AlignedBitVec, AlignedBitVec, borrow::BitwisePairMutViaBorrow<AlignedBitVec,[BitBlock]>);

impl IntoBitIterator for AlignedBitVec {
    type BitIterator = FlatMap<
        <Vec<BitBlock> as IntoIterator>::IntoIter,
        <BitBlock as IntoBitIterator>::BitIterator,
        fn(BitBlock) -> <BitBlock as IntoBitIterator>::BitIterator,
    >;

    fn iter_bits(self) -> Self::BitIterator {
        self.blocks
            .into_iter()
            .flat_map(<BitBlock as IntoBitIterator>::iter_bits)
    }
}

impl<'life> IntoBitIterator for &'life AlignedBitVec {
    type BitIterator = <&'life [BitBlock] as IntoBitIterator>::BitIterator;

    fn iter_bits(self) -> Self::BitIterator {
        <&'life [BitBlock] as IntoBitIterator>::iter_bits(self.borrow())
    }
}

impl AlignedBitVec {
    const fn bits_per_block() -> usize {
        BitBlock::BLOCK_BIT_LEN
    }

    pub fn of_length(unaligned_length: usize) -> AlignedBitVec {
        Self::zeros(unaligned_length)
    }

    pub fn zeros(unaligned_length: usize) -> AlignedBitVec {
        AlignedBitVec {
            blocks: vec![BitBlock::default(); block_count(unaligned_length, Self::bits_per_block())],
        }
    }

    pub fn ones(unaligned_length: usize) -> AlignedBitVec {
        AlignedBitVec {
            blocks: vec![BitBlock::ones(); block_count(unaligned_length, Self::bits_per_block())],
        }
    }

    /// Set all bits to zero.
    pub fn clear(&mut self) {
        for block in &mut self.blocks {
            block.clear();
        }
    }

    pub fn with_length_from_iter<Iterator: IntoIterator<Item = bool>>(iterator: Iterator) -> (Self, usize) {
        let mut iterator = iterator.into_iter();
        let mut blocks = Vec::with_capacity(iterator.size_hint().0 / Self::bits_per_block() + 1);
        let mut length = 0;

        // Note: once `Iterator::array_chunks` is stabilized, we can use that instead.
        loop {
            let mut block = BitBlock::default();
            for index in 0..Self::bits_per_block() {
                match iterator.next() {
                    Some(true) => block.assign_index(index, true),
                    Some(false) => (),
                    None if index == 0 => return (AlignedBitVec { blocks }, length),
                    None => {
                        blocks.push(block);
                        return (AlignedBitVec { blocks }, length);
                    }
                }
                length += 1;
            }
            blocks.push(block);
        }
    }

    #[must_use]
    pub fn top(&self) -> u64 {
        self.blocks[0][0]
    }

    pub fn top_mut(&mut self) -> &mut u64 {
        &mut self.blocks[0][0]
    }

    pub fn from_view(view: &AlignedBitView) -> AlignedBitVec {
        AlignedBitVec {
            blocks: view.blocks.to_vec(),
        }
    }

    pub fn from_view_mut(view: &AlignedBitViewMut) -> AlignedBitVec {
        AlignedBitVec {
            blocks: view.blocks.to_vec(),
        }
    }

    /// View the data as a flat slice of words (u64s) for efficient serialization.
    /// The bit length information is lost - use with care.
    #[must_use]
    pub fn as_words(&self) -> &[Word] {
        unsafe {
            std::slice::from_raw_parts(
                self.blocks.as_ptr().cast::<Word>(),
                self.blocks.len() * BIT_BLOCK_WORD_COUNT,
            )
        }
    }

    /// View the data as a byte slice (native endianness).
    /// Use for fast serialization when endianness is known to match.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.blocks.as_ptr().cast::<u8>(),
                self.blocks.len() * std::mem::size_of::<BitBlock>(),
            )
        }
    }

    /// Reconstruct from a flat vector of words (u64s).
    /// Words are grouped into `BitBlock`s of `BIT_BLOCK_WORD_COUNT` words each.
    pub fn from_words(words: &[Word]) -> Self {
        let mut blocks = Vec::with_capacity(words.len().div_ceil(BIT_BLOCK_WORD_COUNT));
        for chunk in words.chunks(BIT_BLOCK_WORD_COUNT) {
            let mut block_words = [0u64; BIT_BLOCK_WORD_COUNT];
            block_words[..chunk.len()].copy_from_slice(chunk);
            blocks.push(BitBlock { blocks: block_words });
        }
        AlignedBitVec { blocks }
    }

    /// Reconstruct from bytes (native endianness).
    /// Use for fast deserialization when endianness is known to match.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a multiple of `size_of::<BitBlock>()`.
    pub fn from_bytes(data: &[u8]) -> Self {
        let block_size = std::mem::size_of::<BitBlock>();
        assert!(
            data.len().is_multiple_of(block_size),
            "bytes length {} must be a multiple of BitBlock size ({})",
            data.len(),
            block_size
        );
        let blocks: Vec<BitBlock> = data
            .chunks_exact(block_size)
            .map(|chunk| {
                let mut block = BitBlock::default();
                for (i, word_bytes) in chunk.chunks_exact(8).enumerate() {
                    block.blocks[i] = u64::from_ne_bytes(word_bytes.try_into().unwrap());
                }
                block
            })
            .collect();
        AlignedBitVec { blocks }
    }

    pub fn selected_from<'life, Iterable>(view: &'life AlignedBitView, support: Iterable) -> AlignedBitVec
    where
        Iterable: IntoIterator<Item = &'life usize>,
        Iterable::IntoIter: ExactSizeIterator<Item = &'life usize>,
    {
        let support_iterator = support.into_iter();
        let mut bits = AlignedBitVec::of_length(support_iterator.len());
        for index in support_iterator {
            bits.assign_index(*index, view.index(*index));
        }
        bits
    }

    pub fn as_view(&self) -> AlignedBitView<'_> {
        AlignedBitView { blocks: &self.blocks }
    }

    pub fn as_view_mut(&mut self) -> AlignedBitViewMut<'_> {
        AlignedBitViewMut {
            blocks: &mut self.blocks,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.bit_len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.len() == 0
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.blocks.bit_len()
    }

    /// Resize the bit vector to a new length, preserving existing data.
    /// New bits are filled with zeros.
    pub fn resize(&mut self, new_length: usize) {
        let old_length = self.len();
        if new_length == old_length {
            return; // No-op
        }
        if new_length > old_length {
            self.blocks
                .resize(block_count(new_length, Self::bits_per_block()), BitBlock::default());
        } else {
            let mut new_vec = AlignedBitVec::zeros(new_length);
            for j in 0..old_length.min(new_length) {
                new_vec.assign_index(j, self.index(j));
            }
            *self = new_vec;
        }
    }
}

#[must_use]
pub fn block_count(length: usize, bits_per_block: usize) -> usize {
    let mut block_count = length / bits_per_block;
    if !length.is_multiple_of(bits_per_block) {
        block_count += 1;
    }
    block_count
}

impl FromIterator<bool> for AlignedBitVec {
    fn from_iter<Iterator: IntoIterator<Item = bool>>(iterator: Iterator) -> Self {
        AlignedBitVec::with_length_from_iter(iterator).0
    }
}

impl FromIterator<Word> for AlignedBitVec {
    fn from_iter<Iterator: IntoIterator<Item = Word>>(iterator: Iterator) -> Self {
        let words: Vec<Word> = iterator.into_iter().collect();
        AlignedBitVec::from_words(&words)
    }
}

impl<'life> IntoIterator for &'life AlignedBitVec {
    into_iterator_via_bit_iterator_body!(&'life AlignedBitVec);
}

impl AlignedBitVec {
    #[must_use]
    pub fn iter(&self) -> <&Self as IntoBitIterator>::BitIterator {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl AlignedBitVec {
    /// Extract a subvector from `start` to `stop`.
    /// Uses optimized block-level copying and shifting.
    ///
    /// # Panics
    ///
    /// Panics if `start + length > self.bit_len()`
    pub fn extract(&self, start: usize, stop: usize) -> AlignedBitVec {
        assert!(start <= stop && stop <= self.bit_len(), "Invalid extraction range.");

        let start_block = start / BitBlock::BLOCK_BIT_LEN;
        let stop_block = stop.div_ceil(BitBlock::BLOCK_BIT_LEN);

        let result = AlignedBitVec {
            blocks: self.blocks[start_block..stop_block.min(self.blocks.len())].to_vec(),
        };

        result << (start % BitBlock::BLOCK_BIT_LEN)
    }
}

impl Shl<usize> for AlignedBitVec {
    type Output = AlignedBitVec;

    fn shl(mut self, count: usize) -> Self::Output {
        self <<= count;
        self
    }
}

impl ShlAssign<usize> for AlignedBitVec {
    fn shl_assign(&mut self, count: usize) {
        let len = self.bit_len();

        if count == 0 || len == 0 {
            return;
        }

        if count >= len {
            self.clear_bits();
            return;
        }

        let bits_per_word = Word::BITS as usize;
        let word_shift = count / bits_per_word;
        let bit_shift = count % bits_per_word;

        if bit_shift == 0 {
            self.shift_left_aligned(word_shift);
        } else {
            self.shift_left_unaligned(word_shift, bit_shift);
        }
    }
}

impl AlignedBitVec {
    /// Shift left by whole words when word-aligned
    fn shift_left_aligned(&mut self, word_shift: usize) {
        let total_words = self.blocks.len() * BIT_BLOCK_WORD_COUNT;

        if word_shift >= total_words {
            self.clear_bits();
            return;
        }

        // Flatten blocks to a slice of words for efficient copying
        let words: &mut [Word] =
            unsafe { std::slice::from_raw_parts_mut(self.blocks.as_mut_ptr().cast::<Word>(), total_words) };

        // Use copy_within for efficient memmove
        words.copy_within(word_shift..total_words, 0);

        // Zero out the tail
        words[total_words - word_shift..].fill(0);
    }

    fn shift_left_unaligned(&mut self, word_shift: usize, bit_shift: usize) {
        let total_words = self.blocks.len() * BIT_BLOCK_WORD_COUNT;

        let words: &mut [Word] =
            unsafe { std::slice::from_raw_parts_mut(self.blocks.as_mut_ptr().cast::<Word>(), total_words) };

        if word_shift >= total_words {
            words.fill(0);
            return;
        }

        let words_to_copy = total_words - word_shift;
        let complement_shift = Word::BITS as usize - bit_shift;
        let mut destination = words.as_mut_ptr();
        let mut source = unsafe { destination.add(word_shift) };

        for _ in 0..(words_to_copy - 1) {
            unsafe {
                let low_bits = *source >> bit_shift;
                source = source.add(1);
                let high_bits = *source << complement_shift;
                *destination = low_bits | high_bits;
                destination = destination.add(1);
            }
        }

        // Handle last word (no high bits from next word)
        unsafe {
            *destination = *source >> bit_shift;
            destination = destination.add(1);
        }

        // Zero out remaining words
        let end = unsafe { words.as_mut_ptr().add(total_words) };
        let remaining = unsafe { usize::try_from(end.offset_from(destination)).unwrap() };
        if remaining > 0 {
            unsafe { std::ptr::write_bytes(destination, 0, remaining) };
        }
    }
}
