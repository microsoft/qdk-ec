use crate::BitView;
use crate::matrix::Column;
use crate::vec::{AlignedBitVec, AlignedBitView, AlignedBitViewMut, BitBlock, BitVec, BitViewMut};
use crate::{BitLength, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, FromBits};
use sorted_iter::{SortedIterator, assume::AssumeSortedByItemExt};
use sorted_vec::SortedSet;

/// A sparse representation of a bit vector using a sorted set of indices.
///
/// `IndexSet` stores only the indices of set bits, making it memory-efficient for sparse
/// bit vectors (those with few set bits relative to their length). It implements the same
/// [`Bitwise`] trait family as [`BitVec`], allowing it to be used interchangeably in
/// generic code.
///
/// # When to Use
///
/// Use `IndexSet` when:
/// - Your bit vector is sparse (most bits are 0)
/// - You need to iterate over set bits frequently
/// - Memory usage is more important than constant-time random access
///
/// Use [`BitVec`] when:
/// - Your bit vector is dense or of unknown density
/// - You need constant-time indexed access
/// - You perform many bitwise operations
///
/// # Example
///
/// ```
/// use binar::{IndexSet, BitVec, Bitwise, BitwiseMut, BitwisePairMut};
///
/// // Create a sparse set with just a few bits set
/// let mut sparse = IndexSet::new();
/// sparse.assign_index(10, true);
/// sparse.assign_index(100, true);
/// sparse.assign_index(1000, true);
///
/// assert_eq!(sparse.weight(), 3);
///
/// // Can convert to BitVec when needed
/// let mut dense = BitVec::zeros(1001);
/// dense.assign(&sparse);
/// assert_eq!(dense.weight(), 3);
/// assert_eq!(dense.index(100), true);
/// ```
///
/// # Performance Characteristics
///
/// - **Memory**: O(k) where k is the number of set bits
/// - **index()**: O(log k) via binary search
/// - **assign_index()**: O(k) due to sorted insertion
/// - **support()**: O(k) iteration over set bits
/// - **weight()**: O(1)
///
/// # See Also
///
/// - [`BitVec`](crate::BitVec) - Dense bit vector representation
/// - [`Bitwise`] - Read-only bit operations trait
/// - [`BitwiseMut`] - Mutable bit operations trait
#[must_use]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct IndexSet {
    indexes: SortedSet<usize>,
}

impl IndexSet {
    /// Creates a new empty `IndexSet`.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{IndexSet, Bitwise};
    ///
    /// let set = IndexSet::new();
    /// assert_eq!(set.weight(), 0);
    /// assert!(set.is_zero());
    /// ```
    pub fn new() -> IndexSet {
        IndexSet {
            indexes: SortedSet::new(),
        }
    }

    /// Creates an `IndexSet` with a single bit set at the given index.
    ///
    /// # Example
    ///
    /// ```
    /// use binar::{IndexSet, Bitwise};
    ///
    /// let set = IndexSet::singleton(42);
    /// assert_eq!(set.weight(), 1);
    /// assert_eq!(set.index(42), true);
    /// assert_eq!(set.index(41), false);
    /// ```
    pub fn singleton(value: usize) -> Self {
        IndexSet {
            indexes: unsafe { SortedSet::from_sorted(vec![value]) },
        }
    }
}

impl Default for IndexSet {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<usize> for IndexSet {
    fn from_iter<Iterator: IntoIterator<Item = usize>>(iterator: Iterator) -> Self {
        let indexes = SortedSet::from_unsorted(iterator.into_iter().collect());
        IndexSet { indexes }
    }
}

// Should this be a bool iterator instead?
impl IntoIterator for IndexSet {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.indexes.into_iter()
    }
}

impl Extend<usize> for IndexSet {
    fn extend<T: IntoIterator<Item = usize>>(&mut self, iter: T) {
        for index in iter {
            self.indexes.push(index);
        }
    }
}

impl Bitwise for IndexSet {
    #[inline]
    fn index(&self, index: usize) -> bool {
        self.indexes.binary_search(&index).is_ok()
    }

    #[inline]
    fn weight(&self) -> usize {
        self.indexes.len()
    }

    #[inline]
    fn support(&self) -> impl SortedIterator<Item = usize> {
        self.indexes.iter().copied().assume_sorted_by_item()
    }
}

impl<Bits: Bitwise> BitwisePairMut<Bits> for IndexSet {
    #[inline]
    fn assign(&mut self, other: &Bits) {
        let indexes: Vec<usize> = other.support().collect();
        self.indexes = indexes.into();
    }

    #[inline]
    fn bitxor_assign(&mut self, other: &Bits) {
        for index in other.support() {
            let found = self.indexes.find_or_insert(index);
            if found.is_found() {
                self.indexes.remove_index(found.index());
            }
        }
    }

    #[inline]
    fn bitand_assign(&mut self, other: &Bits) {
        let self_support = self.indexes.iter().copied().assume_sorted_by_item();
        let other_support = other.support();
        let indexes: Vec<usize> = self_support.intersection(other_support).collect();
        self.indexes = indexes.into();
    }

    #[inline]
    fn bitor_assign(&mut self, other: &Bits) {
        let self_support = self.indexes.iter().copied().assume_sorted_by_item();
        let other_support = other.support();
        let indexes: Vec<usize> = self_support.union(other_support).collect();
        self.indexes = indexes.into();
    }
}

impl BitwiseMut for IndexSet {
    #[inline]
    fn assign_index(&mut self, index: usize, to: bool) {
        if to {
            self.indexes.push(index);
        } else {
            self.indexes.remove_item(&index);
        }
    }

    #[inline]
    fn negate_index(&mut self, index: usize) {
        if self.indexes.contains(&index) {
            self.indexes.remove_item(&index);
        } else {
            self.indexes.push(index);
        }
    }

    #[inline]
    fn clear_bits(&mut self) {
        self.indexes.clear();
    }
}

impl<Bits: Bitwise> BitwisePair<Bits> for IndexSet {
    #[inline]
    fn dot(&self, other: &Bits) -> bool {
        let mut res = false;
        for index in &self.indexes {
            res ^= other.index(*index);
        }
        res
    }

    #[inline]
    fn and_weight(&self, other: &Bits) -> usize {
        self.support().intersection(other.support()).count()
    }

    #[inline]
    fn or_weight(&self, other: &Bits) -> usize {
        self.support().union(other.support()).count()
    }

    #[inline]
    fn xor_weight(&self, other: &Bits) -> usize {
        self.support().symmetric_difference(other.support()).count()
    }
}

impl<T: BitwiseMut + BitLength> BitwisePair<IndexSet> for T {
    #[inline]
    fn dot(&self, other: &IndexSet) -> bool {
        other.dot(self)
    }

    #[inline]
    fn and_weight(&self, other: &IndexSet) -> usize {
        other.and_weight(self)
    }

    #[inline]
    fn or_weight(&self, other: &IndexSet) -> usize {
        other.or_weight(self)
    }

    #[inline]
    fn xor_weight(&self, other: &IndexSet) -> usize {
        other.xor_weight(self)
    }
}

impl<T: BitwiseMut + BitLength> BitwisePairMut<IndexSet> for T {
    #[inline]
    fn assign(&mut self, other: &IndexSet) {
        self.clear_bits();
        for index in &other.indexes {
            self.assign_index(*index, true);
        }
    }

    #[inline]
    fn bitxor_assign(&mut self, other: &IndexSet) {
        for index in other.support() {
            self.negate_index(index);
        }
    }

    #[inline]
    fn bitand_assign(&mut self, other: &IndexSet) {
        let intersection: Vec<usize> = self.support().intersection(other.support()).collect();
        self.clear_bits();
        for k in intersection {
            self.assign_index(k, true);
        }
    }

    #[inline]
    fn bitor_assign(&mut self, other: &IndexSet) {
        let union: Vec<usize> = self.support().union(other.support()).collect();
        self.clear_bits();
        for k in union {
            self.assign_index(k, true);
        }
    }
}

impl<T: BitwiseMut + BitLength> PartialEq<T> for IndexSet {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.support().eq(other.support())
    }
}

impl<Other: Bitwise> FromBits<Other> for IndexSet {
    fn from_bits(other: &Other) -> Self {
        IndexSet {
            indexes: other.support().collect::<Vec<usize>>().into(),
        }
    }
}

impl<'life, T> From<&'life T> for IndexSet
where
    T: Bitwise + 'life,
{
    fn from(value: &'life T) -> Self {
        unsafe {
            IndexSet {
                indexes: SortedSet::from_sorted(value.support().collect::<Vec<_>>()),
            }
        }
    }
}

pub fn remapped(bits: &IndexSet, support: &[usize]) -> IndexSet {
    bits.support().map(|id| support[id]).collect()
}

impl PartialEq<IndexSet> for AlignedBitVec {
    fn eq(&self, other: &IndexSet) -> bool {
        self.support().eq(other.support())
    }
}

impl PartialEq<IndexSet> for AlignedBitView<'_> {
    fn eq(&self, other: &IndexSet) -> bool {
        self.support().eq(other.support())
    }
}

impl PartialEq<IndexSet> for AlignedBitViewMut<'_> {
    fn eq(&self, other: &IndexSet) -> bool {
        self.support().eq(other.support())
    }
}

impl PartialEq<IndexSet> for BitVec {
    fn eq(&self, other: &IndexSet) -> bool {
        self.support().eq(other.support())
    }
}

impl PartialEq<IndexSet> for BitView<'_> {
    fn eq(&self, other: &IndexSet) -> bool {
        self.support().eq(other.support())
    }
}

impl PartialEq<IndexSet> for BitViewMut<'_> {
    fn eq(&self, other: &IndexSet) -> bool {
        self.support().eq(other.support())
    }
}

impl PartialEq<IndexSet> for Column<'_> {
    fn eq(&self, other: &IndexSet) -> bool {
        self.support().eq(other.support())
    }
}

impl PartialEq<IndexSet> for BitBlock {
    fn eq(&self, other: &IndexSet) -> bool {
        self.support().eq(other.support())
    }
}
