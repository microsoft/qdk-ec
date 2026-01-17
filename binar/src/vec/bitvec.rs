use sorted_iter::assume::AssumeSortedByItemExt;
use sorted_iter::sorted_iterator::AssumeSortedByItem;

use crate::bit::bitwise_truncated as truncated;
use crate::bit::bitwise_via_borrow as borrow;
use crate::vec::BitView;
use crate::vec::{AlignedBitVec, BitViewMut, Word};
use crate::{BitBlock, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IntoBitIterator};
use crate::{
    BitLength, delegate_bitwise, delegate_bitwise_body, delegate_bitwise_mut, delegate_bitwise_mut_body,
    delegate_bitwise_pair, delegate_bitwise_pair_body, delegate_bitwise_pair_mut, delegate_bitwise_pair_mut_body,
};
use std::borrow::{Borrow, BorrowMut};
use std::iter::Take;

type BitVecInner<Bits> = crate::bit::truncated::BitsTruncated<Bits>;

impl<Bits> AsRef<[BitBlock]> for BitVecInner<Bits>
where
    Bits: Borrow<[BitBlock]> + Bitwise,
{
    fn as_ref(&self) -> &[BitBlock] {
        self.bits.borrow()
    }
}

impl<Bits> AsMut<[BitBlock]> for BitVecInner<Bits>
where
    Bits: BorrowMut<[BitBlock]> + Bitwise,
{
    fn as_mut(&mut self) -> &mut [BitBlock] {
        self.bits.borrow_mut()
    }
}

impl<Bits> std::hash::Hash for BitVecInner<Bits>
where
    Bits: Borrow<[BitBlock]> + Bitwise,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bit_length.hash(state);
        self.bits.borrow().hash(state);
    }
}

impl<Bits> IntoBitIterator for BitVecInner<Bits>
where
    Bits: IntoBitIterator + Bitwise + BitLength,
{
    type BitIterator = Take<<Bits as IntoBitIterator>::BitIterator>;

    fn iter_bits(self) -> Self::BitIterator {
        let length = self.bit_len();
        <Bits as IntoBitIterator>::iter_bits(self.bits).take(length)
    }
}

impl<'life, Bits> IntoBitIterator for &'life BitVecInner<Bits>
where
    Bits: Borrow<[BitBlock]> + Bitwise + BitLength,
{
    type BitIterator = AssumeSortedByItem<Take<<&'life [BitBlock] as IntoBitIterator>::BitIterator>>;

    fn iter_bits(self) -> Self::BitIterator {
        <&'life [BitBlock] as IntoBitIterator>::iter_bits(self.bits.borrow())
            .take(self.bit_len())
            .assume_sorted_by_item()
    }
}

impl<Bits> Bitwise for BitVecInner<Bits>
where
    Bits: Borrow<[BitBlock]> + Bitwise + BitLength,
{
    delegate_bitwise_body!(truncated::BitwiseTruncated<[BitBlock]>);
}

impl<Bits> BitwiseMut for BitVecInner<Bits>
where
    Bits: BorrowMut<[BitBlock]> + BitLength + Bitwise,
{
    delegate_bitwise_mut_body!(truncated::BitwiseMutTruncated<[BitBlock]>);
}

impl<Bits, OtherBits> BitwisePair<BitVecInner<OtherBits>> for BitVecInner<Bits>
where
    Bits: Borrow<[BitBlock]> + BitLength + Bitwise,
    OtherBits: Borrow<[BitBlock]> + BitLength + Bitwise,
{
    delegate_bitwise_pair_body!(
        BitVecInner<OtherBits>,
        truncated::BitwisePairTruncated<BitVecInner<OtherBits>, [BitBlock], [BitBlock]>
    );
}

impl<Bits, OtherBits> BitwisePairMut<BitVecInner<OtherBits>> for BitVecInner<Bits>
where
    Bits: BorrowMut<[BitBlock]> + BitLength + Bitwise,
    OtherBits: Borrow<[BitBlock]> + BitLength + Bitwise,
{
    delegate_bitwise_pair_mut_body!(
        BitVecInner<OtherBits>,
        truncated::BitwisePairMutTruncated<BitVecInner<OtherBits>, [BitBlock], [BitBlock]>
    );
}

type VecInner = BitVecInner<AlignedBitVec>;

#[must_use]
#[derive(Eq, Clone, Debug)]
pub struct BitVec {
    pub(crate) inner: VecInner,
}

impl From<VecInner> for BitVec {
    fn from(inner: VecInner) -> Self {
        Self { inner }
    }
}

impl From<BitVec> for AlignedBitVec {
    fn from(vec: BitVec) -> Self {
        vec.inner.bits
    }
}

impl<'life> From<&'life AlignedBitVec> for AlignedBitVec {
    fn from(bits: &'life AlignedBitVec) -> Self {
        bits.clone()
    }
}

impl std::ops::Deref for BitVec {
    type Target = VecInner;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for BitVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Borrow<VecInner> for BitVec {
    fn borrow(&self) -> &VecInner {
        &self.inner
    }
}

impl BorrowMut<VecInner> for BitVec {
    fn borrow_mut(&mut self) -> &mut VecInner {
        &mut self.inner
    }
}

impl std::hash::Hash for BitVec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl PartialEq for BitVec {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl BitVec {
    pub fn of_length(length: usize) -> BitVec {
        Self::zeros(length)
    }

    pub fn from_aligned(bit_length: usize, bits: AlignedBitVec) -> BitVec {
        VecInner { bits, bit_length }.into()
    }

    pub fn zeros(length: usize) -> BitVec {
        VecInner {
            bit_length: length,
            bits: AlignedBitVec::zeros(length),
        }
        .into()
    }

    pub fn ones(length: usize) -> BitVec {
        let mut bits = AlignedBitVec::ones(length);
        for index in length..bits.len() {
            bits.assign_index(index, false);
        }
        BitVec::from_aligned(length, bits)
    }

    #[must_use]
    pub fn top(&self) -> u64 {
        self.bits.top()
    }

    pub fn top_mut(&mut self) -> &mut u64 {
        self.bits.top_mut()
    }

    pub fn from_view(view: &BitView) -> BitVec {
        Self::from_aligned(view.len(), AlignedBitVec::from_view(&view.bits))
    }

    pub fn from_view_mut(view: &BitViewMut) -> BitVec {
        Self::from_aligned(view.len(), AlignedBitVec::from_view_mut(&view.bits))
    }

    pub fn selected_from<'life, Iterable>(view: &'life BitView, support: Iterable) -> BitVec
    where
        Iterable: IntoIterator<Item = &'life usize>,
        Iterable::IntoIter: ExactSizeIterator<Item = &'life usize>,
    {
        let support_iterator = support.into_iter();
        let bit_length = support_iterator.len();
        VecInner {
            bit_length,
            bits: AlignedBitVec::selected_from(&view.bits, support_iterator),
        }
        .into()
    }

    pub fn as_view(&self) -> BitView<'_> {
        BitView::from_aligned(self.bit_length, self.bits.as_view())
    }

    pub fn as_view_mut(&mut self) -> BitViewMut<'_> {
        BitViewMut::from_aligned(self.bit_length, self.bits.as_view_mut())
    }

    /// View the data as a flat slice of words (u64s) for efficient serialization.
    #[must_use]
    pub fn as_words(&self) -> &[Word] {
        self.bits.as_words()
    }

    /// View the data as a byte slice (native endianness).
    /// Use for fast serialization when endianness is known to match.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.bits.as_bytes()
    }

    /// Deserialize a vector from words (u64s).
    pub fn from_words(length: usize, words: &[Word]) -> Self {
        Self::from_aligned(length, AlignedBitVec::from_words(words))
    }

    /// Deserialize a vector from bytes (native endianness).
    /// Use for fast deserialization when endianness is known to match.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a multiple of `size_of::<BitBlock>()`.
    pub fn from_bytes(length: usize, data: &[u8]) -> Self {
        Self::from_aligned(length, AlignedBitVec::from_bytes(data))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bit_length
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.bits.bit_len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bit_length == 0
    }

    #[must_use]
    pub fn iter(&self) -> <&Self as IntoBitIterator>::BitIterator {
        <&Self as IntoBitIterator>::iter_bits(self)
    }

    /// Resize the bit vector to a new length, preserving existing data.
    /// New bits are filled with zeros.
    pub fn resize(&mut self, new_length: usize) {
        self.bits.resize(new_length);
        self.bit_length = new_length;
    }
}

impl FromIterator<bool> for BitVec {
    fn from_iter<Iterator: IntoIterator<Item = bool>>(iterator: Iterator) -> Self {
        let (aligned, length) = AlignedBitVec::with_length_from_iter(iterator);
        Self::from_aligned(length, aligned)
    }
}

impl BitLength for BitVec {
    fn bit_len(&self) -> usize {
        self.bit_length
    }
    const BLOCK_BIT_LEN: usize = VecInner::BLOCK_BIT_LEN;
}

impl IntoBitIterator for BitVec {
    type BitIterator = <VecInner as IntoBitIterator>::BitIterator;

    fn iter_bits(self) -> Self::BitIterator {
        <VecInner as IntoBitIterator>::iter_bits(self.inner)
    }
}

impl<'life> IntoBitIterator for &'life BitVec {
    type BitIterator = <&'life VecInner as IntoBitIterator>::BitIterator;

    fn iter_bits(self) -> Self::BitIterator {
        <&'life VecInner as IntoBitIterator>::iter_bits(self.borrow())
    }
}

delegate_bitwise!(BitVec, borrow::BitwiseViaBorrow<VecInner>);
delegate_bitwise_mut!(BitVec, borrow::BitwiseMutViaBorrow<VecInner>);
delegate_bitwise_pair!(BitVec, BitVec, borrow::BitwisePairViaBorrow<BitVec, VecInner>);
delegate_bitwise_pair_mut!(BitVec, BitVec, borrow::BitwisePairMutViaBorrow<BitVec, VecInner>);

impl BitVec {
    pub fn extract(&self, start: usize, stop: usize) -> BitVec {
        BitVec::from_aligned(stop - start, AlignedBitVec::extract(&self.bits, start, stop))
    }
}
