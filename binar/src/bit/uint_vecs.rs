use crate::{
    BitLength, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IntoBitIterator,
    bit::bitwise_for_slice::{
        BitwiseForSlice, BitwiseMutForSlice, BitwisePairForSlice, BitwisePairMutForSlice, IntoBitIteratorForSlice,
    },
};
use core::iter::FlatMap;

macro_rules! uint_vecs {
    ($uint:ty) => {
        delegate_bitwise!(Vec<$uint>, BitwiseForSlice<$uint>);
        delegate_bitwise!([$uint], BitwiseForSlice<$uint>);

        delegate_bitwise_mut!(Vec<$uint>, BitwiseMutForSlice<$uint>);
        delegate_bitwise_mut!([$uint], BitwiseMutForSlice<$uint>);

        delegate_bitwise_pair!(Vec<$uint>, Vec<$uint>, BitwisePairForSlice<$uint>);
        delegate_bitwise_pair!([$uint], [$uint], BitwisePairForSlice<$uint>);

        delegate_bitwise_pair_mut!(Vec<$uint>, Vec<$uint>, BitwisePairMutForSlice<$uint>);
        delegate_bitwise_pair_mut!([$uint], [$uint], BitwisePairMutForSlice<$uint>);

        impl BitLength for Vec<$uint> {
            #[inline]
            fn bit_len(&self) -> usize {
                Self::BLOCK_BIT_LEN * self.len()
            }
            const BLOCK_BIT_LEN: usize = <$uint>::BLOCK_BIT_LEN;
        }

        impl BitLength for [$uint] {
            #[inline]
            fn bit_len(&self) -> usize {
                Self::BLOCK_BIT_LEN * self.len()
            }
            const BLOCK_BIT_LEN: usize = <$uint>::BLOCK_BIT_LEN;
        }
    };
}

uint_vecs!(u16);
uint_vecs!(u32);
uint_vecs!(u64);
uint_vecs!(u128);

macro_rules! uint_array_vecs {
    ($uint:ty) => {
        impl<const SIZE: usize> Bitwise for Vec<[$uint; SIZE]> {
            delegate_bitwise_body! {
                BitwiseForSlice:: <[$uint;SIZE]>
            }
        }
        impl<const SIZE: usize> Bitwise for [[$uint; SIZE]] {
            delegate_bitwise_body! {
                BitwiseForSlice:: <[$uint;SIZE]>
            }
        }

        impl<const SIZE: usize> BitwiseMut for Vec<[$uint; SIZE]> {
            delegate_bitwise_mut_body! {
                BitwiseMutForSlice:: <[$uint;SIZE]>
            }
        }
        impl<const SIZE: usize> BitwiseMut for [[$uint; SIZE]] {
            delegate_bitwise_mut_body! {
                BitwiseMutForSlice:: <[$uint;SIZE]>
            }
        }

        impl<const SIZE: usize> BitwisePair<Vec<[$uint; SIZE]>> for Vec<[$uint; SIZE]> {
            delegate_bitwise_pair_body! {
                Vec<[$uint;SIZE]> ,BitwisePairForSlice:: <[$uint;SIZE]>
            }
        }
        impl<const SIZE: usize> BitwisePair<[[$uint; SIZE]]> for [[$uint; SIZE]] {
            delegate_bitwise_pair_body! {
                [[$uint;SIZE]],BitwisePairForSlice:: <[$uint;SIZE]>
            }
        }

        impl<const SIZE: usize> BitwisePairMut<Vec<[$uint; SIZE]>> for Vec<[$uint; SIZE]> {
            delegate_bitwise_pair_mut_body! {
                Vec<[$uint;SIZE]> ,BitwisePairMutForSlice:: <[$uint;SIZE]>
            }
        }
        impl<const SIZE: usize> BitwisePairMut<[[$uint; SIZE]]> for [[$uint; SIZE]] {
            delegate_bitwise_pair_mut_body! {
                [[$uint;SIZE]],BitwisePairMutForSlice:: <[$uint;SIZE]>
            }
        }

        impl<const SIZE: usize> crate::IntoBitIterator for Vec<[$uint; SIZE]> {
            type BitIterator = FlatMap<
                <Vec<[$uint; SIZE]> as IntoIterator>::IntoIter,
                <[$uint; SIZE] as IntoBitIterator>::BitIterator,
                fn([$uint; SIZE]) -> <[$uint; SIZE] as IntoBitIterator>::BitIterator,
            >;

            fn iter_bits(self) -> Self::BitIterator {
                self.into_iter()
                    .flat_map(<[$uint; SIZE] as IntoBitIterator>::iter_bits)
            }
        }

        impl<'life, const SIZE: usize> crate::IntoBitIterator for &'life Vec<[$uint; SIZE]> {
            type BitIterator = <&'life [[$uint; SIZE]] as IntoBitIteratorForSlice<'life, [$uint; SIZE]>>::BitIterator;

            fn iter_bits(self) -> Self::BitIterator {
                IntoBitIteratorForSlice::<'life, [$uint; SIZE]>::iter_bits(self.as_slice())
            }
        }

        impl<'life, const SIZE: usize> crate::IntoBitIterator for &'life [[$uint; SIZE]] {
            type BitIterator = <&'life [[$uint; SIZE]] as IntoBitIteratorForSlice<'life, [$uint; SIZE]>>::BitIterator;

            fn iter_bits(self) -> Self::BitIterator {
                IntoBitIteratorForSlice::<'life, [$uint; SIZE]>::iter_bits(self)
            }
        }

        impl<const SIZE: usize> BitLength for Vec<[$uint; SIZE]> {
            #[inline]
            fn bit_len(&self) -> usize {
                Self::BLOCK_BIT_LEN * self.len()
            }
            const BLOCK_BIT_LEN: usize = <[$uint; SIZE]>::BLOCK_BIT_LEN;
        }

        impl<const SIZE: usize> BitLength for [[$uint; SIZE]] {
            #[inline]
            fn bit_len(&self) -> usize {
                Self::BLOCK_BIT_LEN * self.len()
            }
            const BLOCK_BIT_LEN: usize = <[$uint; SIZE]>::BLOCK_BIT_LEN;
        }
    };
}

uint_array_vecs!(u16);
uint_array_vecs!(u32);
uint_array_vecs!(u64);
uint_array_vecs!(u128);
