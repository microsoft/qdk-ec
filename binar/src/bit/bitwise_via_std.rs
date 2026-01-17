use crate::{BitLength, IntoBitIterator, bit::standard_types::support_iterator};
use rand::distributions::{Distribution, Standard};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, ShlAssign, Shr};

pub trait Counts {
    fn count_ones(&self) -> usize;
    fn leading_zeros(&self) -> usize;
    fn trailing_zeros(&self) -> usize;
}

macro_rules! counts_impl {
    ($word_type:ty) => {
        impl Counts for $word_type {
            #[inline]
            fn count_ones(&self) -> usize {
                <$word_type>::count_ones(*self) as usize
            }
            #[inline]
            fn leading_zeros(&self) -> usize {
                <$word_type>::leading_zeros(*self) as usize
            }
            #[inline]
            fn trailing_zeros(&self) -> usize {
                <$word_type>::trailing_zeros(*self) as usize
            }
        }
    };
}

counts_impl!(u8);
counts_impl!(u16);
counts_impl!(u32);
counts_impl!(u64);
counts_impl!(u128);

pub trait Rotate: Copy {
    #[must_use]
    fn rotate_left(self, n: u32) -> Self;
    #[must_use]
    fn rotate_right(self, n: u32) -> Self;
}

macro_rules! rotate_impl {
    ($word_type:ty) => {
        impl Rotate for $word_type {
            #[inline]
            fn rotate_left(self, n: u32) -> Self {
                <$word_type>::rotate_left(self, n)
            }
            #[inline]
            fn rotate_right(self, n: u32) -> Self {
                <$word_type>::rotate_right(self, n)
            }
        }
    };
}

rotate_impl!(u8);
rotate_impl!(u16);
rotate_impl!(u32);
rotate_impl!(u64);
rotate_impl!(u128);

pub trait UnsignedInt
where
    Self: From<bool>
        + Counts
        + BitLength
        + BitAndAssign
        + BitXorAssign
        + BitOrAssign
        + Not<Output = Self>
        + Copy
        + Default
        + Shl<usize, Output = Self>
        + ShlAssign<usize>
        + Shr<usize, Output = Self>
        + PartialEq
        + BitAnd<Self, Output = Self>
        + BitOr<Self, Output = Self>
        + BitXor<Self, Output = Self>
        + Rotate,
{
}

impl<T> UnsignedInt for T where
    T: From<bool>
        + Counts
        + BitLength
        + BitAndAssign
        + BitXorAssign
        + BitOrAssign
        + Copy
        + Not<Output = T>
        + Default
        + Shl<usize, Output = Self>
        + ShlAssign<usize>
        + Shr<usize, Output = Self>
        + PartialEq
        + BitAnd<T, Output = T>
        + BitOr<T, Output = T>
        + BitXor<T, Output = T>
        + Rotate
{
}

#[inline]
fn one<Word: UnsignedInt>() -> Word {
    Word::from(true)
}

#[inline]
fn zero<Word: UnsignedInt>() -> Word {
    Word::default()
}

pub trait BitwiseViaStd
where
    Self: UnsignedInt,
    for<'a> &'a Self: IntoBitIterator,
{
    #[inline]
    fn index(&self, index: usize) -> bool {
        assert!(index < Self::BLOCK_BIT_LEN);
        let mask = one::<Self>() << index;
        mask & *self == mask
    }

    #[inline]
    fn support(&self) -> impl sorted_iter::SortedIterator<Item = usize> {
        support_iterator(<&'_ Self as IntoBitIterator>::iter_bits(self))
    }

    #[inline]
    fn weight(&self) -> usize {
        self.count_ones()
    }

    #[inline]
    fn parity(&self) -> bool {
        (self.weight() % 2) == 1
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == zero::<Self>()
    }

    #[inline]
    fn is_unit(&self, index: usize) -> bool {
        *self == one::<Self>() << index
    }

    #[inline]
    fn min_support(&self) -> Option<usize> {
        if *self == zero::<Self>() {
            return None;
        }
        Some(self.trailing_zeros())
    }

    #[inline]
    fn max_support(&self) -> Option<usize> {
        if *self == zero::<Self>() {
            return None;
        }
        Some(Self::BLOCK_BIT_LEN - self.leading_zeros() - 1)
    }
}

impl<T> BitwiseViaStd for T
where
    T: UnsignedInt,
    for<'a> &'a T: IntoBitIterator,
{
}

pub trait BitwisePairViaStd
where
    Self: UnsignedInt,
    for<'a> &'a Self: IntoBitIterator,
{
    #[inline]
    fn dot(&self, other: &Self) -> bool {
        (*self & *other).count_ones() & 1 == 1
    }

    #[inline]
    fn and_weight(&self, other: &Self) -> usize {
        (*self & *other).count_ones()
    }

    #[inline]
    fn or_weight(&self, other: &Self) -> usize {
        (*self | *other).count_ones()
    }

    #[inline]
    fn xor_weight(&self, other: &Self) -> usize {
        (*self ^ *other).count_ones()
    }
}

impl<T> BitwisePairViaStd for T
where
    T: UnsignedInt,
    for<'a> &'a T: IntoBitIterator,
{
}

pub trait BitwiseMutViaStd
where
    Self: UnsignedInt,
    for<'a> &'a Self: IntoBitIterator,
{
    #[inline]
    fn assign_index(&mut self, index: usize, to: bool) {
        assert!(index < Self::BLOCK_BIT_LEN);
        let mask = one::<Self>() << index;
        if to {
            *self |= mask;
        } else {
            *self &= !mask;
        }
    }

    #[inline]
    fn negate_index(&mut self, index: usize) {
        assert!(
            index < Self::BLOCK_BIT_LEN,
            "index {} is larger than capacity {}",
            index,
            Self::BLOCK_BIT_LEN
        );
        *self ^= one::<Self>() << index;
    }

    #[inline]
    fn clear_bits(&mut self) {
        *self = zero::<Self>();
    }

    fn assign_random(&mut self, bit_count: usize, random_number_generator: &mut impl rand::Rng)
    where
        Standard: Distribution<Self>,
    {
        assert!(bit_count <= Self::BLOCK_BIT_LEN);
        let mask = (!Self::from(false)) >> (Self::BLOCK_BIT_LEN - bit_count);
        *self = random_number_generator.r#gen::<Self>() & mask;
    }
}

impl<T> BitwiseMutViaStd for T
where
    T: UnsignedInt,
    for<'a> &'a T: IntoBitIterator,
{
}

pub trait BitwisePairMutViaStd
where
    Self: UnsignedInt,
    for<'a> &'a Self: IntoBitIterator,
{
    #[inline]
    fn assign(&mut self, other: &Self) {
        *self = *other;
    }

    #[inline]
    fn bitxor_assign(&mut self, other: &Self) {
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

impl<T> BitwisePairMutViaStd for T
where
    T: UnsignedInt,
    for<'a> &'a T: IntoBitIterator,
{
}

pub struct BitIteratorForUnsignedIntSlice<'life, Word: UnsignedInt> {
    word_index: usize,
    word_mask: Word,
    bits: &'life [Word],
}

impl<'life, Word: UnsignedInt> BitIteratorForUnsignedIntSlice<'life, Word> {
    #[must_use]
    pub fn from_bits(bits: &'life [Word]) -> BitIteratorForUnsignedIntSlice<'life, Word> {
        BitIteratorForUnsignedIntSlice {
            word_index: 0,
            word_mask: one::<Word>(),
            bits,
        }
    }
}

impl<Word: UnsignedInt> Iterator for BitIteratorForUnsignedIntSlice<'_, Word> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.word_index < self.bits.len() {
            let value = (*unsafe { self.bits.get_unchecked(self.word_index) }) & self.word_mask != zero::<Word>();
            self.word_mask = self.word_mask.rotate_left(1);
            self.word_index += usize::from(self.word_mask == one::<Word>());
            return Some(value);
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = <Word>::BLOCK_BIT_LEN * self.bits.len();
        (size, Some(size))
    }
}

impl<WordType: UnsignedInt> ExactSizeIterator for BitIteratorForUnsignedIntSlice<'_, WordType> {
    fn len(&self) -> usize {
        <WordType>::BLOCK_BIT_LEN * self.bits.len()
    }
}

pub struct BitIteratorForUnsignedInt<WordType: UnsignedInt> {
    word_mask: WordType,
    word: WordType,
}

impl<WordType: UnsignedInt> BitIteratorForUnsignedInt<WordType> {
    #[must_use]
    pub fn from_bits(word: &WordType) -> BitIteratorForUnsignedInt<WordType> {
        BitIteratorForUnsignedInt {
            word_mask: one::<WordType>(),
            word: *word,
        }
    }
}

impl<WordType: UnsignedInt> Iterator for BitIteratorForUnsignedInt<WordType> {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        if self.word_mask == zero::<WordType>() {
            None
        } else {
            let value = (self.word & self.word_mask) == self.word_mask;
            self.word_mask <<= 1;
            Some(value)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = <WordType>::BLOCK_BIT_LEN;
        (size, Some(size))
    }
}

impl<WordType: UnsignedInt> ExactSizeIterator for BitIteratorForUnsignedInt<WordType> {
    fn len(&self) -> usize {
        <WordType>::BLOCK_BIT_LEN
    }
}
