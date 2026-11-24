use sorted_iter::{SortedIterator, assume::AssumeSortedByItemExt};

use crate::bit::bitblock::Word;
use crate::bit::bitwise_via_std::UnsignedInt;
use crate::{BitLength, Bitwise};

struct WordSupportIterator<'life, UnsignedWord> {
    words: std::slice::Iter<'life, UnsignedWord>,
    remaining: UnsignedWord,
    next_word_offset: usize,
}

impl<UnsignedWord: UnsignedInt> Iterator for WordSupportIterator<'_, UnsignedWord> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let zero = UnsignedWord::default();
        while self.remaining == zero {
            self.remaining = *self.words.next()?;
            self.next_word_offset += UnsignedWord::BLOCK_BIT_LEN;
        }
        let bit_offset = self.remaining.trailing_zeros();
        self.remaining = self.remaining.clear_lowest_set_bit();
        Some(self.next_word_offset - UnsignedWord::BLOCK_BIT_LEN + bit_offset)
    }
}

impl<UnsignedWord: UnsignedInt> std::iter::FusedIterator for WordSupportIterator<'_, UnsignedWord> {}

#[inline]
pub(crate) fn words_support<UnsignedWord: UnsignedInt>(
    word_slice: &[UnsignedWord],
) -> impl SortedIterator<Item = usize> {
    WordSupportIterator {
        words: word_slice.iter(),
        remaining: UnsignedWord::default(),
        next_word_offset: 0,
    }
    .assume_sorted_by_item()
}

struct WordsOrSupportIterator<'life> {
    left: &'life [Word],
    right: &'life [Word],
    word_index: usize,
    remaining: Word,
    current_offset: usize,
}

impl Iterator for WordsOrSupportIterator<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        while self.remaining == 0 {
            let word = match (
                self.left.get(self.word_index).copied(),
                self.right.get(self.word_index).copied(),
            ) {
                (Some(left_word), Some(right_word)) => left_word | right_word,
                (Some(left_word), None) => left_word,
                (None, Some(right_word)) => right_word,
                (None, None) => return None,
            };
            self.remaining = word;
            self.current_offset = self.word_index * (Word::BITS as usize);
            self.word_index += 1;
        }
        let bit_offset = self.remaining.trailing_zeros() as usize;
        self.remaining &= self.remaining - 1;
        Some(self.current_offset + bit_offset)
    }
}

impl std::iter::FusedIterator for WordsOrSupportIterator<'_> {}

enum OrSupportIterator<'life, Fallback> {
    Words(WordsOrSupportIterator<'life>),
    Fallback(Fallback),
}

impl<Fallback: Iterator<Item = usize>> Iterator for OrSupportIterator<'_, Fallback> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        match self {
            Self::Words(iterator) => iterator.next(),
            Self::Fallback(iterator) => iterator.next(),
        }
    }
}

/// Returns a sorted iterator over indices set in at least one of `left` or `right`.
///
/// When both operands expose aligned words via [`Bitwise::aligned_words`], this combines
/// corresponding words with bitwise OR in a single pass, including unequal storage lengths.
/// Otherwise it falls back to the sorted union of `left.support()` and `right.support()`.
#[inline]
pub fn or_support<'life, Left, Right>(
    left: &'life Left,
    right: &'life Right,
) -> impl SortedIterator<Item = usize> + 'life
where
    Left: Bitwise + ?Sized,
    Right: Bitwise + ?Sized,
{
    if let (Some(left_words), Some(right_words)) = (left.aligned_words(), right.aligned_words()) {
        OrSupportIterator::Words(WordsOrSupportIterator {
            left: left_words,
            right: right_words,
            word_index: 0,
            remaining: 0,
            current_offset: 0,
        })
    } else {
        OrSupportIterator::Fallback(left.support().union(right.support()))
    }
    .assume_sorted_by_item()
}

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
