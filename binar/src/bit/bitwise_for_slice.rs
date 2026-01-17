use crate::bit::standard_types::support_iterator;
use crate::{
    BitLength, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IntoBitIterator,
    bit::standard_types::block_and_bit_index,
};
use std::borrow::{Borrow, BorrowMut};
use std::iter::{FlatMap, zip};

pub trait BitwiseForSlice<BitBlock>
where
    Self: Borrow<[BitBlock]>,
    BitBlock: Bitwise + Sized + 'static + BitLength,
    for<'life> &'life [BitBlock]: IntoBitIterator,
{
    #[inline]
    fn index(&self, index: usize) -> bool {
        let (block_index, bit_index) = block_and_bit_index::<BitBlock>(index);
        self.borrow()[block_index].index(bit_index)
    }
    #[inline]
    fn weight(&self) -> usize {
        self.borrow().iter().map(Bitwise::weight).sum()
    }
    #[inline]
    fn parity(&self) -> bool {
        self.borrow()
            .iter()
            .map(|block| usize::from(block.parity()))
            .sum::<usize>()
            % 2
            == 1
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.borrow().iter().all(Bitwise::is_zero)
    }
    #[inline]
    fn is_unit(&self, index: usize) -> bool {
        let (block_index, bit_index) = block_and_bit_index::<BitBlock>(index);
        self.borrow()[block_index].is_unit(bit_index)
    }
    #[inline]
    fn max_support(&self) -> Option<usize> {
        let bits = BitBlock::BLOCK_BIT_LEN;
        for (chunk_index, chunk) in self.borrow().iter().enumerate().rev() {
            if let Some(bit_index) = chunk.max_support() {
                return Some(chunk_index * bits + bit_index);
            }
        }
        None
    }
    #[inline]
    fn min_support(&self) -> Option<usize> {
        let bits = BitBlock::BLOCK_BIT_LEN;
        for (chunk_index, chunk) in self.borrow().iter().enumerate() {
            if let Some(bit_index) = chunk.min_support() {
                return Some(chunk_index * bits + bit_index);
            }
        }
        None
    }

    fn support(&self) -> impl sorted_iter::SortedIterator<Item = usize> {
        support_iterator(self.borrow().iter_bits())
    }
}

impl<Bits, BitBlock> BitwiseForSlice<BitBlock> for Bits
where
    Bits: ?Sized + Borrow<[BitBlock]>,
    BitBlock: Bitwise + Sized + 'static + BitLength,
    for<'life> &'life [BitBlock]: IntoBitIterator,
{
}

pub trait BitwiseMutForSlice<BitBlock>
where
    Self: BorrowMut<[BitBlock]>,
    BitBlock: BitwiseMut + Sized + BitLength,
{
    #[inline]
    fn assign_index(&mut self, index: usize, to: bool) {
        let (block_index, bit_index) = block_and_bit_index::<BitBlock>(index);
        self.borrow_mut()[block_index].assign_index(bit_index, to);
    }

    #[inline]
    fn negate_index(&mut self, index: usize) {
        let (block_index, bit_index) = block_and_bit_index::<BitBlock>(index);
        self.borrow_mut()[block_index].negate_index(bit_index);
    }

    #[inline]
    fn clear_bits(&mut self) {
        for chunk in self.borrow_mut().iter_mut() {
            chunk.clear_bits();
        }
    }

    #[inline]
    fn assign_random(&mut self, bit_count: usize, random_number_generator: &mut impl rand::Rng)
    where
        BitBlock: BitLength,
    {
        for j in 0..bit_count {
            self.assign_index(j, random_number_generator.r#gen());
        }
    }
}

impl<Bits, BitBlock> BitwiseMutForSlice<BitBlock> for Bits
where
    Bits: ?Sized + BorrowMut<[BitBlock]>,
    BitBlock: BitwiseMut + Sized + BitLength,
{
}

pub trait BitwisePairForSlice<BitBlock>
where
    Self: Borrow<[BitBlock]>,
    BitBlock: BitwisePair,
{
    #[inline]
    fn dot(&self, other: &Self) -> bool {
        zip(self.borrow().iter(), other.borrow().iter())
            .map(|(left, right)| usize::from(left.dot(right)))
            .sum::<usize>()
            % 2
            == 1
    }
    #[inline]
    fn and_weight(&self, other: &Self) -> usize {
        zip(self.borrow().iter(), other.borrow().iter())
            .map(|(left, right)| left.and_weight(right))
            .sum()
    }
    #[inline]
    fn or_weight(&self, other: &Self) -> usize {
        zip(self.borrow().iter(), other.borrow().iter())
            .map(|(left, right)| left.or_weight(right))
            .sum()
    }
    #[inline]
    fn xor_weight(&self, other: &Self) -> usize {
        zip(self.borrow().iter(), other.borrow().iter())
            .map(|(left, right)| left.xor_weight(right))
            .sum()
    }
}

impl<Bits, BitBlock> BitwisePairForSlice<BitBlock> for Bits
where
    Bits: ?Sized + Borrow<[BitBlock]>,
    BitBlock: BitwisePair,
{
}

pub trait BitwisePairMutForSlice<BitBlock>
where
    Self: BorrowMut<[BitBlock]>,
    BitBlock: BitwisePairMut,
{
    #[inline]
    fn assign(&mut self, other: &Self) {
        zip(self.borrow_mut().iter_mut(), other.borrow().iter()).for_each(|(left, right)| left.assign(right));
    }
    #[inline]
    fn bitxor_assign(&mut self, other: &Self) {
        zip(self.borrow_mut().iter_mut(), other.borrow().iter()).for_each(|(left, right)| left.bitxor_assign(right));
    }
    #[inline]
    fn bitand_assign(&mut self, other: &Self) {
        zip(self.borrow_mut().iter_mut(), other.borrow().iter()).for_each(|(left, right)| left.bitand_assign(right));
    }
    #[inline]
    fn bitor_assign(&mut self, other: &Self) {
        zip(self.borrow_mut().iter_mut(), other.borrow().iter()).for_each(|(left, right)| left.bitor_assign(right));
    }
}

impl<Bits, BitBlock> BitwisePairMutForSlice<BitBlock> for Bits
where
    Bits: ?Sized + BorrowMut<[BitBlock]>,
    BitBlock: BitwisePairMut,
{
}

pub trait IntoBitIteratorForSlice<'life, BitBlock>
where
    Self: Borrow<[BitBlock]>,
    &'life BitBlock: IntoBitIterator,
    BitBlock: 'life,
{
    type BitIterator: Iterator<Item = bool>;
    fn iter_bits(self) -> Self::BitIterator;
}

impl<'life, BitBlock> IntoBitIteratorForSlice<'life, BitBlock> for &'life [BitBlock]
where
    &'life BitBlock: IntoBitIterator,
    BitBlock: 'life,
{
    type BitIterator = FlatMap<
        <&'life [BitBlock] as IntoIterator>::IntoIter,
        <&'life BitBlock as IntoBitIterator>::BitIterator,
        fn(&'life BitBlock) -> <&'life BitBlock as IntoBitIterator>::BitIterator,
    >;

    fn iter_bits(self) -> Self::BitIterator {
        self.iter().flat_map(<&'life BitBlock as IntoBitIterator>::iter_bits)
    }
}
