use sorted_iter::{SortedIterator, assume::AssumeSortedByItemExt};

use crate::BitLength;

// Helper functions for primitive type implementations

pub fn support_iterator(iter: impl Iterator<Item = bool>) -> impl SortedIterator<Item = usize> {
    iter.enumerate()
        .filter(|pair| pair.1)
        .map(|pair| pair.0)
        .assume_sorted_by_item()
}

// Common helper for calculating block and bit indices
#[inline]
#[must_use]
pub fn block_and_bit_index<T: BitLength>(index: usize) -> (usize, usize) {
    let block_index = index / T::BLOCK_BIT_LEN;
    let bit_index = index % T::BLOCK_BIT_LEN;
    (block_index, bit_index)
}
