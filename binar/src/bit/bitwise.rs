/// See also [`BitwiseMut`], [`BitwisePair`], and [`BitwisePairMut`].
pub trait Bitwise {
    fn index(&self, index: usize) -> bool;
    fn support(&self) -> impl sorted_iter::SortedIterator<Item = usize>;
    #[inline]
    fn min_support(&self) -> Option<usize> {
        self.support().next()
    }
    #[inline]
    fn max_support(&self) -> Option<usize> {
        self.support().last()
    }
    #[inline]
    fn weight(&self) -> usize {
        self.support().count()
    }
    #[inline]
    fn parity(&self) -> bool {
        (self.weight() % 2) == 1
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.weight() == 0
    }
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

/// See also [`Bitwise`], [`BitwisePair`], and [`BitwisePairMut`].
pub trait BitwiseMut: Bitwise {
    fn assign_index(&mut self, index: usize, to: bool);
    fn negate_index(&mut self, index: usize);
    fn clear_bits(&mut self);

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

/// See also [`Bitwise`], [`BitwiseMut`], and [`BitwisePairMut`].
pub trait BitwisePair<Other: ?Sized = Self> {
    fn dot(&self, other: &Other) -> bool;
    fn and_weight(&self, other: &Other) -> usize;
    fn or_weight(&self, other: &Other) -> usize;
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
            <Self as $delegate_trait>::or_weight(self, other)
        }
    };
}
pub use delegate_bitwise_pair_body;

/// See also [`Bitwise`], [`BitwiseMut`], and [`BitwisePair`].
pub trait BitwisePairMut<Other: ?Sized + Bitwise = Self>: Bitwise + BitwiseMut + BitwisePair<Other> {
    fn assign(&mut self, other: &Other);
    fn bitand_assign(&mut self, other: &Other);
    fn bitor_assign(&mut self, other: &Other);
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
