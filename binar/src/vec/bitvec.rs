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

/// A dynamically-sized bit vector with a convenient, user-friendly API.
///
/// `BitVec` is the primary bit vector type in this crate, wrapping [`AlignedBitVec`]
/// to provide an easier-to-use interface while maintaining the same performance characteristics.
/// It stores a sequence of bits (0 or 1) with cache-aligned memory for efficient operations.
///
/// # When to Use
///
/// Use `BitVec` for:
/// - Working with bit sequences as first-class values
/// - Implementing algorithms over GF(2) (binary field)
/// - Storing compact boolean arrays
/// - Performing bitwise operations on large datasets
///
/// Consider [`vec::AlignedBitVec`](crate::vec::AlignedBitVec) directly when you need more
/// control over memory layout or specific performance tuning.
///
/// # Construction
///
/// ```
/// use binar::BitVec;
///
/// // Create from length
/// let zeros = BitVec::zeros(100);
/// let ones = BitVec::ones(100);
///
/// // Create from iterator
/// let from_iter: BitVec = [true, false, true, false].into_iter().collect();
///
/// // Create from bytes or words
/// let from_words = BitVec::from_words(64, &[0xFFFF_0000_FFFF_0000u64]);
/// ```
///
/// # Bit Operations
///
/// `BitVec` implements the [`Bitwise`] family of traits, providing both read-only
/// and mutable operations:
///
/// ```
/// use binar::{BitVec, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut};
///
/// let mut v = BitVec::zeros(10);
///
/// // Set and read individual bits
/// v.assign_index(3, true);
/// v.assign_index(7, true);
/// assert_eq!(v.index(3), true);
/// assert_eq!(v.weight(), 2);  // Count set bits
///
/// // Iterate over set bit indices
/// let indices: Vec<_> = v.support().collect();
/// assert_eq!(indices, vec![3, 7]);
///
/// // Binary operations
/// let mut other = BitVec::ones(10);
/// v.bitxor_assign(&other);  // XOR (addition in GF(2))
/// assert_eq!(v.weight(), 8);  // All bits except 3 and 7
/// ```
///
/// # Serialization
///
/// For efficient serialization, use [`as_words`](BitVec::as_words) or
/// [`as_bytes`](BitVec::as_bytes):
///
/// ```
/// use binar::BitVec;
///
/// let v = BitVec::ones(128);
/// let words = v.as_words();
/// let restored = BitVec::from_words(128, words);
/// assert_eq!(v, restored);
/// ```
///
/// # See Also
///
/// - [`vec::AlignedBitVec`](crate::vec::AlignedBitVec) - The underlying aligned type
/// - [`BitView`](crate::BitView) and [`BitViewMut`](crate::BitViewMut) - Non-owning views
/// - [`IndexSet`](crate::IndexSet) - Sparse representation for vectors with few set bits
/// - [`BitMatrix`](crate::BitMatrix) - 2D matrix of bits
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
    /// Creates a new `BitVec` with all bits set to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitVec, Bitwise};
    ///
    /// let v = BitVec::zeros(100);
    /// assert_eq!(v.len(), 100);
    /// assert!(v.is_zero());
    /// ```
    pub fn of_length(length: usize) -> BitVec {
        Self::zeros(length)
    }

    /// Creates a `BitVec` from an aligned bit vector and length.
    ///
    /// This is useful when working directly with [`AlignedBitVec`] and needing
    /// to convert to the more convenient `BitVec` API.
    pub fn from_aligned(bit_length: usize, bits: AlignedBitVec) -> BitVec {
        VecInner { bits, bit_length }.into()
    }

    /// Creates a new `BitVec` with all bits set to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitVec, Bitwise};
    ///
    /// let v = BitVec::zeros(100);
    /// assert_eq!(v.len(), 100);
    /// assert!(v.is_zero());
    /// ```
    pub fn zeros(length: usize) -> BitVec {
        VecInner {
            bit_length: length,
            bits: AlignedBitVec::zeros(length),
        }
        .into()
    }

    /// Creates a new `BitVec` with all bits set to one.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitVec, Bitwise};
    ///
    /// let v = BitVec::ones(8);
    /// assert_eq!(v.weight(), 8);
    /// ```
    pub fn ones(length: usize) -> BitVec {
        let mut bits = AlignedBitVec::ones(length);
        for index in length..bits.len() {
            bits.assign_index(index, false);
        }
        BitVec::from_aligned(length, bits)
    }

    /// Returns the first word (u64) of the underlying storage.
    ///
    /// This is useful for direct bit manipulation of the first 64 bits.
    #[must_use]
    pub fn top(&self) -> u64 {
        self.bits.top()
    }

    /// Returns a mutable reference to the first word (u64) of the underlying storage.
    ///
    /// This is useful for direct bit manipulation of the first 64 bits.
    pub fn top_mut(&mut self) -> &mut u64 {
        self.bits.top_mut()
    }

    /// Creates a `BitVec` by copying data from a `BitView`.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitVec;
    ///
    /// let original = BitVec::ones(10);
    /// let view = original.as_view();
    /// let copy = BitVec::from_view(&view);
    /// assert_eq!(copy, original);
    /// ```
    pub fn from_view(view: &BitView) -> BitVec {
        Self::from_aligned(view.len(), AlignedBitVec::from_view(&view.bits))
    }

    /// Creates a `BitVec` by copying data from a `BitViewMut`.
    pub fn from_view_mut(view: &BitViewMut) -> BitVec {
        Self::from_aligned(view.len(), AlignedBitVec::from_view_mut(&view.bits))
    }

    /// Creates a new `BitVec` by selecting specific bit indices from a view.
    ///
    /// The resulting vector contains the bits at the specified indices,
    /// packed contiguously starting from index 0.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitVec, BitwiseMut};
    ///
    /// let mut source = BitVec::zeros(10);
    /// source.assign_index(2, true);
    /// source.assign_index(5, true);
    /// source.assign_index(7, true);
    ///
    /// let indices = vec![2, 5, 7];
    /// let selected = BitVec::selected_from(&source.as_view(), &indices);
    /// assert_eq!(selected.len(), 3);
    /// assert_eq!(selected.weight(), 3);  // All selected bits were set
    /// ```
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

    /// Returns a non-owning view of this bit vector.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitVec, Bitwise};
    ///
    /// let v = BitVec::ones(10);
    /// let view = v.as_view();
    /// assert_eq!(view.len(), 10);
    /// assert_eq!(view.weight(), 10);
    /// ```
    pub fn as_view(&self) -> BitView<'_> {
        BitView::from_aligned(self.bit_length, self.bits.as_view())
    }

    /// Returns a mutable non-owning view of this bit vector.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitVec, BitwiseMut};
    ///
    /// let mut v = BitVec::zeros(10);
    /// let mut view = v.as_view_mut();
    /// view.assign_index(3, true);
    /// assert_eq!(v.index(3), true);
    /// ```
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

    /// Returns the number of bits in the vector.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitVec;
    ///
    /// let v = BitVec::zeros(42);
    /// assert_eq!(v.len(), 42);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.bit_length
    }

    /// Returns the capacity of the underlying storage in bits.
    ///
    /// This may be larger than [`len()`](BitVec::len) due to alignment.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.bits.bit_len()
    }

    /// Returns `true` if the vector has a length of zero.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::BitVec;
    ///
    /// let empty = BitVec::zeros(0);
    /// assert!(empty.is_empty());
    ///
    /// let non_empty = BitVec::zeros(10);
    /// assert!(!non_empty.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bit_length == 0
    }

    /// Returns an iterator over the bits as boolean values.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitVec, BitwiseMut};
    ///
    /// let mut v = BitVec::zeros(5);
    /// v.assign_index(1, true);
    /// v.assign_index(3, true);
    ///
    /// let bits: Vec<bool> = v.iter().collect();
    /// assert_eq!(bits, vec![false, true, false, true, false]);
    /// ```
    #[must_use]
    pub fn iter(&self) -> <&Self as IntoBitIterator>::BitIterator {
        <&Self as IntoBitIterator>::iter_bits(self)
    }

    /// Resizes the bit vector to a new length, preserving existing data.
    ///
    /// If `new_length` is greater than the current length, new bits are set to zero.
    /// If `new_length` is less than the current length, the vector is truncated.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitVec, BitwiseMut};
    ///
    /// let mut v = BitVec::zeros(5);
    /// v.assign_index(2, true);
    /// v.resize(10);
    /// assert_eq!(v.len(), 10);
    /// assert_eq!(v.index(2), true);
    /// ```
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
    /// Extracts a contiguous slice of bits into a new `BitVec`.
    ///
    /// Returns a new vector containing bits from `start` (inclusive) to `stop` (exclusive).
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{BitVec, BitwiseMut};
    ///
    /// let mut v = BitVec::zeros(10);
    /// v.assign_index(2, true);
    /// v.assign_index(4, true);
    /// v.assign_index(6, true);
    ///
    /// let slice = v.extract(2, 7);
    /// assert_eq!(slice.len(), 5);
    /// assert_eq!(slice.index(0), true);  // Originally index 2
    /// assert_eq!(slice.index(2), true);  // Originally index 4
    /// ```
    pub fn extract(&self, start: usize, stop: usize) -> BitVec {
        BitVec::from_aligned(stop - start, AlignedBitVec::extract(&self.bits, start, stop))
    }
}
