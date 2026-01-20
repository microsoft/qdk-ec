/// Trait for read-only bit operations.
///
/// `Bitwise` provides methods for reading and querying bit-like data structures.
/// It's the foundation of the bit manipulation trait hierarchy in this crate.
///
/// # Core Operations
///
/// - [`index`](Bitwise::index): Read a specific bit
/// - [`support`](Bitwise::support): Iterate over indices of set bits (bits equal to 1)
/// - [`weight`](Bitwise::weight): Count the number of set bits
///
/// # Example
///
/// ```
/// use binar::{BitVec, Bitwise, BitwiseMut};
///
/// let mut bits = BitVec::zeros(10);
/// bits.assign_index(2, true);
/// bits.assign_index(7, true);
///
/// assert_eq!(bits.weight(), 2);
/// assert_eq!(bits.index(2), true);
/// assert_eq!(bits.index(3), false);
///
/// // Iterate over indices of set bits
/// let indices: Vec<_> = bits.support().collect();
/// assert_eq!(indices, vec![2, 7]);
/// ```
///
/// # Implementors
///
/// This trait is implemented for:
/// - [`BitVec`](crate::BitVec) and [`vec::AlignedBitVec`](crate::vec::AlignedBitVec)
/// - [`BitMatrix`](crate::BitMatrix) rows and [`matrix::AlignedBitMatrix`](crate::matrix::AlignedBitMatrix) rows
/// - [`IndexSet`](crate::IndexSet) (sparse bit representation)
/// - View types like [`BitView`](crate::BitView)
/// - Standard types: `u64`, `u32`, `u16`, `u8`, `usize`, arrays like `[u64; N]`, and slices
///
/// See also [`BitwiseMut`], [`BitwisePair`], and [`BitwisePairMut`].
pub trait Bitwise {
    /// Returns the value of the bit at the given index.
    ///
    /// # Panics
    ///
    /// May panic if `index` is out of bounds, depending on the implementing type.
    fn index(&self, index: usize) -> bool;

    /// Returns a sorted iterator over the indices of all set bits (bits equal to 1).
    ///
    /// The iterator yields indices in ascending order.
    fn support(&self) -> impl sorted_iter::SortedIterator<Item = usize>;

    /// Returns the index of the first set bit, if any.
    #[inline]
    fn min_support(&self) -> Option<usize> {
        self.support().next()
    }

    /// Returns the index of the last set bit, if any.
    #[inline]
    fn max_support(&self) -> Option<usize> {
        self.support().last()
    }

    /// Returns the number of set bits (Hamming weight).
    #[inline]
    fn weight(&self) -> usize {
        self.support().count()
    }

    /// Returns `true` if the number of set bits is odd.
    #[inline]
    fn parity(&self) -> bool {
        (self.weight() % 2) == 1
    }

    /// Returns `true` if no bits are set.
    #[inline]
    fn is_zero(&self) -> bool {
        self.weight() == 0
    }

    /// Returns `true` if exactly one bit is set and it's at the given index.
    #[inline]
    fn is_unit(&self, index: usize) -> bool {
        self.weight() == 1 && self.index(index)
    }
}

#[macro_export]
macro_rules! delegate_bitwise_body {
    ($delegate_trait:path) => {
        #[inline]
        fn index(&self, index: usize) -> bool {
            <Self as $delegate_trait>::index(self, index)
        }

        #[inline]
        fn support(&self) -> impl sorted_iter::SortedIterator<Item = usize> {
            <Self as $delegate_trait>::support(self)
        }

        #[inline]
        fn max_support(&self) -> Option<usize> {
            <Self as $delegate_trait>::max_support(self)
        }

        #[inline]
        fn min_support(&self) -> Option<usize> {
            <Self as $delegate_trait>::min_support(self)
        }

        #[inline]
        fn weight(&self) -> usize {
            <Self as $delegate_trait>::weight(self)
        }

        #[inline]
        fn parity(&self) -> bool {
            <Self as $delegate_trait>::parity(self)
        }

        #[inline]
        fn is_zero(&self) -> bool {
            <Self as $delegate_trait>::is_zero(self)
        }

        #[inline]
        fn is_unit(&self, index: usize) -> bool {
            <Self as $delegate_trait>::is_unit(self, index)
        }
    };
}
pub use delegate_bitwise_body;

/// Trait for mutable bit operations.
///
/// `BitwiseMut` extends [`Bitwise`] with methods for modifying bit values.
///
/// # Example
///
/// ```
/// use binar::{BitVec, BitwiseMut, Bitwise};
///
/// let mut bits = BitVec::zeros(10);
///
/// // Set individual bits
/// bits.assign_index(3, true);
/// bits.assign_index(7, true);
///
/// // Toggle a bit
/// bits.negate_index(3);  // Now false
/// assert_eq!(bits.index(3), false);
///
/// // Clear all bits
/// bits.clear_bits();
/// assert!(bits.is_zero());
/// ```
///
/// # Implementors
///
/// This trait is implemented for the same types as [`Bitwise`], including standard mutable types
/// like `&mut u64`, `&mut [u64]`, and `&mut [u64; N]`.
///
/// See also [`Bitwise`], [`BitwisePair`], and [`BitwisePairMut`].
pub trait BitwiseMut: Bitwise {
    /// Sets the bit at the given index to the specified value.
    ///
    /// # Panics
    ///
    /// May panic if `index` is out of bounds, depending on the implementing type.
    fn assign_index(&mut self, index: usize, to: bool);

    /// Toggles the bit at the given index (false → true, true → false).
    ///
    /// # Panics
    ///
    /// May panic if `index` is out of bounds, depending on the implementing type.
    fn negate_index(&mut self, index: usize);

    /// Sets all bits to false (clears the entire bit structure).
    fn clear_bits(&mut self);

    /// Assigns random values to the first `bit_count` bits.
    fn assign_random(&mut self, bit_count: usize, random_number_generator: &mut impl rand::Rng) {
        for j in 0..bit_count {
            self.assign_index(j, random_number_generator.r#gen());
        }
    }
}

#[macro_export]
macro_rules! delegate_bitwise_mut_body {
    ($delegate_trait:path) => {
        #[inline]
        fn assign_index(&mut self, index: usize, to: bool) {
            <Self as $delegate_trait>::assign_index(self, index, to)
        }
        #[inline]
        fn negate_index(&mut self, index: usize) {
            <Self as $delegate_trait>::negate_index(self, index)
        }
        #[inline]
        fn clear_bits(&mut self) {
            <Self as $delegate_trait>::clear_bits(self)
        }
        #[inline]
        fn assign_random(&mut self, bit_count: usize, random_number_generator: &mut impl rand::Rng) {
            <Self as $delegate_trait>::assign_random(self, bit_count, random_number_generator)
        }
    };
}
pub use delegate_bitwise_mut_body;

/// Trait for binary operations between two bit structures.
///
/// `BitwisePair` provides methods for computing relationships between two bit-like types,
/// such as dot products (inner products over GF(2)) and population counts of logical operations.
///
/// # Example
///
/// ```
/// use binar::{BitVec, BitwiseMut, BitwisePair};
///
/// let mut v1 = BitVec::zeros(8);
/// v1.assign_index(1, true);
/// v1.assign_index(3, true);
/// v1.assign_index(5, true);
///
/// let mut v2 = BitVec::zeros(8);
/// v2.assign_index(1, true);
/// v2.assign_index(5, true);
/// v2.assign_index(7, true);
///
/// // Dot product over GF(2): counts common set bits mod 2
/// assert_eq!(v1.dot(&v2), false);  // 2 common bits: even
///
/// // Weight of AND: number of bits set in both
/// assert_eq!(v1.and_weight(&v2), 2);  // Bits 1 and 5
///
/// // Weight of XOR: number of bits that differ
/// assert_eq!(v1.xor_weight(&v2), 2);  // Bits 3 and 7
/// ```
///
/// # Implementors
///
/// This trait is implemented for all types implementing [`Bitwise`], including standard types
/// like `u64`, arrays, and slices, allowing operations between different bit representations.
///
/// See also [`Bitwise`], [`BitwiseMut`], and [`BitwisePairMut`].
pub trait BitwisePair<Other: ?Sized = Self> {
    /// Computes the dot product (inner product) over GF(2).
    ///
    /// Returns `true` if an odd number of bits are set in both structures,
    /// `false` if an even number (including zero) are set in both.
    fn dot(&self, other: &Other) -> bool;

    /// Returns the number of bits set in both structures (Hamming weight of AND).
    fn and_weight(&self, other: &Other) -> usize;

    /// Returns the number of bits set in at least one structure (Hamming weight of OR).
    fn or_weight(&self, other: &Other) -> usize;

    /// Returns the number of bits that differ (Hamming weight of XOR/Hamming distance).
    fn xor_weight(&self, other: &Other) -> usize;
}

#[macro_export]
macro_rules! delegate_bitwise_pair_body {
    ($other_type:ty, $delegate_trait:path) => {
        #[inline]
        fn dot(&self, other: &$other_type) -> bool {
            <Self as $delegate_trait>::dot(self, other)
        }
        #[inline]
        fn and_weight(&self, other: &$other_type) -> usize {
            <Self as $delegate_trait>::and_weight(self, other)
        }
        #[inline]
        fn or_weight(&self, other: &$other_type) -> usize {
            <Self as $delegate_trait>::or_weight(self, other)
        }
        #[inline]
        fn xor_weight(&self, other: &$other_type) -> usize {
            <Self as $delegate_trait>::xor_weight(self, other)
        }
    };
}
pub use delegate_bitwise_pair_body;

/// Trait for mutable binary operations between two bit structures.
///
/// `BitwisePairMut` provides in-place logical operations for modifying one bit structure
/// based on another. These operations are performed over GF(2), where XOR is addition
/// and AND is multiplication.
///
/// # Example
///
/// ```
/// use binar::{BitVec, BitwisePairMut, Bitwise, BitwiseMut};
///
/// let mut v1 = BitVec::zeros(8);
/// v1.assign_index(1, true);
/// v1.assign_index(3, true);
/// v1.assign_index(5, true);
///
/// let mut v2 = BitVec::zeros(8);
/// v2.assign_index(3, true);
/// v2.assign_index(5, true);
/// v2.assign_index(7, true);
///
/// // XOR: add over GF(2)
/// v1.bitxor_assign(&v2);
/// assert_eq!(v1.weight(), 2);  // Bits 1 and 7 remain
///
/// // Copy another vector
/// v1.assign(&v2);
/// assert_eq!(v1.weight(), v2.weight());
/// ```
///
/// # Implementors
///
/// This trait is implemented for all types implementing [`BitwiseMut`], including mutable
/// standard types like `&mut u64` and `&mut [u64; N]`.
///
/// See also [`Bitwise`], [`BitwiseMut`], and [`BitwisePair`].
pub trait BitwisePairMut<Other: ?Sized + Bitwise = Self>: Bitwise + BitwiseMut + BitwisePair<Other> {
    /// Assigns all bits from `other` to `self`.
    fn assign(&mut self, other: &Other);

    /// Performs in-place bitwise AND with `other`.
    fn bitand_assign(&mut self, other: &Other);

    /// Performs in-place bitwise OR with `other`.
    fn bitor_assign(&mut self, other: &Other);

    /// Performs in-place bitwise XOR with `other` (addition over GF(2)).
    fn bitxor_assign(&mut self, other: &Other);

    fn assign_with_offset(&mut self, other: &Other, start_bit: usize, num_bits: usize) {
        for bit_index in 0..num_bits {
            self.assign_index(bit_index + start_bit, other.index(bit_index));
        }
    }

    fn assign_from_interval(&mut self, other: &Other, start_bit: usize, num_bits: usize) {
        for bit_index in 0..num_bits {
            self.assign_index(bit_index, other.index(start_bit + bit_index));
        }
    }
}

#[macro_export]
macro_rules! delegate_bitwise_pair_mut_body {
    ($other_type:ty, $delegate_trait:path) => {
        #[inline]
        fn assign(&mut self, other: &$other_type) {
            <Self as $delegate_trait>::assign(self, other);
        }
        #[inline]
        fn bitand_assign(&mut self, other: &$other_type) {
            <Self as $delegate_trait>::bitand_assign(self, other);
        }
        #[inline]
        fn bitor_assign(&mut self, other: &$other_type) {
            <Self as $delegate_trait>::bitor_assign(self, other);
        }
        #[inline]
        fn bitxor_assign(&mut self, other: &$other_type) {
            <Self as $delegate_trait>::bitxor_assign(self, other);
        }
    };
}
pub use delegate_bitwise_pair_mut_body;

pub trait FromBits<Other> {
    fn from_bits(other: &Other) -> Self;
}

pub trait IntoBitIterator {
    type BitIterator: Iterator<Item = bool>;
    fn iter_bits(self) -> Self::BitIterator;
}

#[macro_export]
macro_rules! into_iterator_via_bit_iterator_body {
    ($type:ty) => {
        type Item = bool;
        type IntoIter = <$type as IntoBitIterator>::BitIterator;
        fn into_iter(self) -> Self::IntoIter {
            self.iter_bits()
        }
    };
}
pub use into_iterator_via_bit_iterator_body;

pub trait BitLength {
    fn bit_len(&self) -> usize;
    const BLOCK_BIT_LEN: usize;
}

#[macro_export]
macro_rules! delegate_bitwise {
    ($type:ty, $delegate_trait:path) => {
        impl Bitwise for $type {
            delegate_bitwise_body! {$delegate_trait}
        }
    };
}
pub use delegate_bitwise;

#[macro_export]
macro_rules! delegate_bitwise_mut {
    ($type:ty, $delegate_trait:path) => {
        impl BitwiseMut for $type {
            delegate_bitwise_mut_body! {$delegate_trait}
        }
    };
}
pub use delegate_bitwise_mut;

#[macro_export]
macro_rules! delegate_bitwise_pair {
    ($type:ty, $other_type:ty, $delegate_trait:path) => {
        impl BitwisePair<$other_type> for $type {
            delegate_bitwise_pair_body! {$other_type, $delegate_trait}
        }
    };
}
pub use delegate_bitwise_pair;

#[macro_export]
macro_rules! delegate_bitwise_pair_mut {
    ($type:ty, $other_type:ty, $delegate_trait:path) => {
        impl BitwisePairMut<$other_type> for $type {
            delegate_bitwise_pair_mut_body! {$other_type, $delegate_trait}
        }
    };
}
pub use delegate_bitwise_pair_mut;
