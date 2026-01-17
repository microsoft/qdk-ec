use crate::bit::bitwise_via_borrow::{
    BitwiseMutViaBorrow, BitwisePairMutViaBorrow, BitwisePairViaBorrow, BitwiseViaBorrow,
};
use crate::bit::bitwise_via_iter::{BitwisePairMutViaIter, BitwisePairViaIter};
use crate::{BitLength, bit::standard_types::support_iterator};
use crate::{Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IntoBitIterator};
use std::iter::Copied;

// Bit traits for Vec<bool>, [bool], [bool;SIZE], Vec<bool>

impl<'life> IntoBitIterator for &'life [bool] {
    type BitIterator = Copied<<&'life [bool] as IntoIterator>::IntoIter>;
    fn iter_bits(self) -> Self::BitIterator {
        self.iter().copied()
    }
}

impl<'life> IntoBitIterator for &'life Vec<bool> {
    type BitIterator = Copied<<&'life [bool] as IntoIterator>::IntoIter>;
    fn iter_bits(self) -> Self::BitIterator {
        self.iter().copied()
    }
}

impl BitLength for Vec<bool> {
    #[inline]
    fn bit_len(&self) -> usize {
        self.len()
    }
    const BLOCK_BIT_LEN: usize = 1;
}

impl BitLength for [bool] {
    #[inline]
    fn bit_len(&self) -> usize {
        self.len()
    }
    const BLOCK_BIT_LEN: usize = 1;
}

impl Bitwise for [bool] {
    #[inline]
    fn index(&self, index: usize) -> bool {
        self[index]
    }

    #[inline]
    fn weight(&self) -> usize {
        self.iter().filter(|bit| **bit).count()
    }

    #[inline]
    fn support(&self) -> impl sorted_iter::SortedIterator<Item = usize> {
        support_iterator(self.iter_bits())
    }
}

impl BitwiseMut for [bool] {
    #[inline]
    fn assign_index(&mut self, index: usize, to: bool) {
        self[index] = to;
    }

    #[inline]
    fn negate_index(&mut self, index: usize) {
        self[index] ^= true;
    }

    #[inline]
    fn clear_bits(&mut self) {
        for val in self.iter_mut() {
            *val = false;
        }
    }
}

delegate_bitwise_pair!([bool], [bool], BitwisePairViaIter<[bool]>);
delegate_bitwise_pair_mut!([bool], [bool], BitwisePairMutViaIter<[bool]>);

delegate_bitwise!(Vec<bool>, BitwiseViaBorrow<[bool]>);
impl<const SIZE: usize> Bitwise for [bool; SIZE] {
    delegate_bitwise_body! {BitwiseViaBorrow<[bool]>}
}

delegate_bitwise_mut!(Vec<bool>, BitwiseMutViaBorrow<[bool]>);
impl<const SIZE: usize> BitwiseMut for [bool; SIZE] {
    delegate_bitwise_mut_body! {BitwiseMutViaBorrow<[bool]>}
}

delegate_bitwise_pair!([bool], Vec<bool>, BitwisePairViaBorrow<Vec<bool>, [bool]>);
delegate_bitwise_pair!(Vec<bool>, [bool], BitwisePairViaBorrow<[bool], [bool]>);
delegate_bitwise_pair!(Vec<bool>, Vec<bool>, BitwisePairViaBorrow<Vec<bool>, [bool]>);
impl<const SIZE: usize> BitwisePair for [bool; SIZE] {
    delegate_bitwise_pair_body! {[bool; SIZE], BitwisePairViaBorrow<[bool; SIZE],[bool]>}
}

delegate_bitwise_pair_mut!([bool], Vec<bool>, BitwisePairMutViaBorrow<Vec<bool>, [bool]>);
delegate_bitwise_pair_mut!(Vec<bool>, [bool], BitwisePairMutViaBorrow<[bool], [bool]>);
delegate_bitwise_pair_mut!(Vec<bool>, Vec<bool>, BitwisePairMutViaBorrow<Vec<bool>, [bool]>);
impl<const SIZE: usize> BitwisePairMut for [bool; SIZE] {
    delegate_bitwise_pair_mut_body! {[bool; SIZE], BitwisePairMutViaBorrow<[bool; SIZE],[bool]>}
}

delegate_bitwise!(&[bool], BitwiseViaBorrow<[bool]>);
delegate_bitwise!(&mut [bool], BitwiseViaBorrow<[bool]>);
delegate_bitwise_mut!(&mut [bool], BitwiseMutViaBorrow<[bool]>);

delegate_bitwise_pair!(&[bool], &[bool], BitwisePairViaBorrow<&[bool], [bool]>);
delegate_bitwise_pair!(&mut [bool], &mut [bool], BitwisePairViaBorrow<&mut [bool], [bool]>);
