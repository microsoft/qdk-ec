use crate::bit::bitwise_via_std as via;
use crate::{BitLength, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IntoBitIterator};

macro_rules! implement_unsigned_int_traits {
    ($word_type:ty) => {
        impl BitLength for $word_type {
            fn bit_len(&self) -> usize {
                Self::BLOCK_BIT_LEN
            }
            const BLOCK_BIT_LEN: usize = <$word_type>::BITS as usize;
        }

        delegate_bitwise!($word_type, via::BitwiseViaStd);
        delegate_bitwise_mut!($word_type, via::BitwiseMutViaStd);
        delegate_bitwise_pair!($word_type, $word_type, via::BitwisePairViaStd);
        delegate_bitwise_pair_mut!($word_type, $word_type, via::BitwisePairMutViaStd);

        impl IntoBitIterator for $word_type {
            type BitIterator = via::BitIteratorForUnsignedInt<$word_type>;
            fn iter_bits(self) -> Self::BitIterator {
                via::BitIteratorForUnsignedInt::<$word_type>::from_bits(&self)
            }
        }

        impl IntoBitIterator for &$word_type {
            type BitIterator = via::BitIteratorForUnsignedInt<$word_type>;
            fn iter_bits(self) -> Self::BitIterator {
                via::BitIteratorForUnsignedInt::<$word_type>::from_bits(self)
            }
        }

        impl<'life> IntoBitIterator for &'life [$word_type] {
            type BitIterator = via::BitIteratorForUnsignedIntSlice<'life, $word_type>;
            fn iter_bits(self) -> Self::BitIterator {
                via::BitIteratorForUnsignedIntSlice::<'_, $word_type>::from_bits(self)
            }
        }
    };
}

pub type BitIterator<'life> = via::BitIteratorForUnsignedIntSlice<'life, u64>;

implement_unsigned_int_traits!(u16);
implement_unsigned_int_traits!(u32);
implement_unsigned_int_traits!(u64);
implement_unsigned_int_traits!(u128);

// macro_rules! delegate_bitwise_traits {
//     ($uint:ty) => {
//         delegate_bitwise!(&$uint, BitwiseViaBorrow<$uint>);
//         delegate_bitwise!(&mut $uint, BitwiseViaBorrow<$uint>);
//         delegate_bitwise_mut!(&mut $uint, BitwiseMutViaBorrow<$uint>);
//         delegate_bitwise_pair!(&$uint, $uint, BitwisePairViaBorrow<$uint,$uint,$uint>);
//         delegate_bitwise_pair!(&$uint, &$uint, BitwisePairViaBorrow<&$uint,$uint,$uint>);
//         delegate_bitwise_pair!(&$uint, &mut $uint, BitwisePairViaBorrow<&mut $uint,$uint,$uint>);
//         delegate_bitwise_pair!(&mut $uint, $uint, BitwisePairViaBorrow<$uint,$uint,$uint>);
//         delegate_bitwise_pair!(&mut $uint, &$uint, BitwisePairViaBorrow<&$uint,$uint,$uint>);
//         delegate_bitwise_pair!(&mut $uint, &mut $uint, BitwisePairViaBorrow<&mut $uint,$uint,$uint>);
//     };
// }

// delegate_bitwise_traits!(u16);
// delegate_bitwise_traits!(u32);
// delegate_bitwise_traits!(u64);
// delegate_bitwise_traits!(u128);
