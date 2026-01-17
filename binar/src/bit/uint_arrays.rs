use crate::{
    BitLength, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IntoBitIterator,
    bit::bitwise_for_arrays::{BitwisePairForArray, BitwisePairMutForArray},
    bit::bitwise_for_slice::{BitwiseForSlice, BitwiseMutForSlice},
    bit::bitwise_via_borrow::{BitwiseMutViaBorrow, BitwisePairMutViaBorrow, BitwisePairViaBorrow, BitwiseViaBorrow},
};
use core::iter::FlatMap;

macro_rules! uint_arrays {
    ($uint:ty) => {
        impl<const SIZE: usize> Bitwise for [$uint;SIZE] {
            delegate_bitwise_body!{BitwiseForSlice<$uint>}
        }

        impl<const SIZE: usize> BitwiseMut for [$uint;SIZE] {
            delegate_bitwise_mut_body!{BitwiseMutForSlice<$uint>}
        }

        impl<const SIZE: usize> BitwisePair for [$uint;SIZE] {
            delegate_bitwise_pair_body!{[$uint;SIZE], BitwisePairForArray<SIZE, $uint>}
        }

        impl<const SIZE: usize> BitwisePairMut for [$uint;SIZE] {
            delegate_bitwise_pair_mut_body!{[$uint;SIZE], BitwisePairMutForArray<SIZE, $uint>}
        }

        impl<const SIZE: usize> IntoBitIterator for [$uint;SIZE] {
            type BitIterator = FlatMap<
                <[$uint;SIZE] as IntoIterator>::IntoIter,
                <$uint as IntoBitIterator>::BitIterator,
                fn($uint) -> <$uint as IntoBitIterator>::BitIterator,
            >;

            fn iter_bits(self) -> Self::BitIterator {
                self.into_iter().flat_map(<$uint as IntoBitIterator>::iter_bits)
            }
        }

        impl<'life, const SIZE: usize> IntoBitIterator for &'life [$uint;SIZE] {
            type BitIterator = <&'life[$uint] as IntoBitIterator>::BitIterator;

            fn iter_bits(self) -> Self::BitIterator {
                <&'_[$uint] as IntoBitIterator>::iter_bits(self.as_slice())
            }
        }

        impl<const SIZE: usize> BitLength for [$uint; SIZE] {
            #[inline]
            fn bit_len(&self) -> usize {
                Self::BLOCK_BIT_LEN
            }

            const BLOCK_BIT_LEN: usize = SIZE * <$uint>::BLOCK_BIT_LEN;
        }

        // Bitwise for array references

        impl<const SIZE: usize> Bitwise for &[$uint;SIZE] {
            delegate_bitwise_body!(BitwiseViaBorrow<[$uint;SIZE]>);
        }
        impl<const SIZE: usize> BitwisePair for &[$uint;SIZE] {
            delegate_bitwise_pair_body!(&[$uint;SIZE], BitwisePairViaBorrow<&[$uint;SIZE], [$uint;SIZE]>);
        }
        impl<const SIZE: usize> BitwisePair<[$uint;SIZE]> for &[$uint;SIZE] {
            delegate_bitwise_pair_body!([$uint;SIZE], BitwisePairViaBorrow<[$uint;SIZE], [$uint;SIZE]>);
        }
        impl<const SIZE: usize> BitwisePair<&mut[$uint;SIZE]> for &[$uint;SIZE] {
            delegate_bitwise_pair_body!(&mut[$uint;SIZE], BitwisePairViaBorrow<&mut[$uint;SIZE], [$uint;SIZE]>);
        }

        impl<const SIZE: usize> Bitwise for &mut [$uint;SIZE] {
            delegate_bitwise_body!(BitwiseViaBorrow<[$uint;SIZE]>);
        }
        impl<const SIZE: usize> BitwisePair for &mut [$uint;SIZE] {
            delegate_bitwise_pair_body!(&mut[$uint;SIZE], BitwisePairViaBorrow<&mut[$uint;SIZE], [$uint;SIZE]>);
        }
        impl<const SIZE: usize> BitwisePair<[$uint;SIZE]> for &mut [$uint;SIZE] {
            delegate_bitwise_pair_body!([$uint;SIZE], BitwisePairViaBorrow<[$uint;SIZE], [$uint;SIZE]>);
        }
        impl<const SIZE: usize> BitwisePair<&[$uint;SIZE]> for &mut [$uint;SIZE] {
            delegate_bitwise_pair_body!(&[$uint;SIZE], BitwisePairViaBorrow<&[$uint;SIZE], [$uint;SIZE]>);
        }
        impl<const SIZE: usize> BitwisePair<&mut[$uint;SIZE]> for [$uint;SIZE] {
            delegate_bitwise_pair_body!(&mut[$uint;SIZE], BitwisePairViaBorrow<&mut[$uint;SIZE], [$uint;SIZE]>);
        }
        impl<const SIZE: usize> BitwisePair<&[$uint;SIZE]> for [$uint;SIZE] {
            delegate_bitwise_pair_body!(&[$uint;SIZE], BitwisePairViaBorrow<&[$uint;SIZE], [$uint;SIZE]>);
        }

        impl<const SIZE: usize> BitwiseMut for &mut [$uint;SIZE] {
            delegate_bitwise_mut_body!(BitwiseMutViaBorrow<[$uint;SIZE]>);
        }

        impl<const SIZE: usize> BitwisePairMut for &mut [$uint;SIZE] {
            delegate_bitwise_pair_mut_body!(&mut[$uint;SIZE], BitwisePairMutViaBorrow<&mut[$uint;SIZE], [$uint;SIZE]>);
        }
        impl<const SIZE: usize> BitwisePairMut<[$uint;SIZE]> for &mut [$uint;SIZE] {
            delegate_bitwise_pair_mut_body!([$uint;SIZE], BitwisePairMutViaBorrow<[$uint;SIZE], [$uint;SIZE]>);
        }
        impl<const SIZE: usize> BitwisePairMut<&[$uint;SIZE]> for &mut [$uint;SIZE] {
            delegate_bitwise_pair_mut_body!(&[$uint;SIZE], BitwisePairMutViaBorrow<&[$uint;SIZE], [$uint;SIZE]>);
        }
        impl<const SIZE: usize> BitwisePairMut<&mut[$uint;SIZE]> for [$uint;SIZE] {
            delegate_bitwise_pair_mut_body!(&mut[$uint;SIZE], BitwisePairMutViaBorrow<&mut[$uint;SIZE], [$uint;SIZE]>);
        }
        impl<const SIZE: usize> BitwisePairMut<&[$uint;SIZE]> for [$uint;SIZE] {
            delegate_bitwise_pair_mut_body!(&[$uint;SIZE], BitwisePairMutViaBorrow<&[$uint;SIZE], [$uint;SIZE]>);
        }

        impl<'life, const SIZE: usize> IntoBitIterator for &'life &[$uint;SIZE] {
            type BitIterator = <&'life[$uint] as IntoBitIterator>::BitIterator;

            fn iter_bits(self) -> Self::BitIterator {
                self.as_slice().iter_bits()
            }
        }

        impl<'life, const SIZE: usize> IntoBitIterator for &'life &mut[$uint;SIZE] {
            type BitIterator = <&'life[$uint] as IntoBitIterator>::BitIterator;

            fn iter_bits(self) -> Self::BitIterator {
                self.as_slice().iter_bits()
            }
        }

        impl<const SIZE: usize> BitLength for &mut[$uint; SIZE] {
            fn bit_len(&self) -> usize {
                Self::BLOCK_BIT_LEN
            }

            const BLOCK_BIT_LEN: usize = SIZE * <$uint>::BLOCK_BIT_LEN;
        }
    };
}

uint_arrays!(u16);
uint_arrays!(u32);
uint_arrays!(u64);
uint_arrays!(u128);
