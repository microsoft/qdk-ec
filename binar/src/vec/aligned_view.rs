use std::borrow::{Borrow, BorrowMut};
use std::hash::Hash;

use crate::bit::bitwise_via_borrow::{
    BitwiseMutViaBorrow, BitwisePairMutViaBorrow, BitwisePairViaBorrow, BitwiseViaBorrow,
};
use crate::vec::AlignedBitVec;
use crate::{
    BitBlock, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, delegate_bitwise, delegate_bitwise_body,
    delegate_bitwise_mut, delegate_bitwise_mut_body, delegate_bitwise_pair, delegate_bitwise_pair_body,
    delegate_bitwise_pair_mut, delegate_bitwise_pair_mut_body,
};
use crate::{BitLength, IntoBitIterator, into_iterator_via_bit_iterator_body};

#[must_use]
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct AlignedBitView<'life> {
    pub(crate) blocks: &'life [BitBlock],
}

#[must_use]
#[derive(PartialEq, Eq, Debug)]
pub struct AlignedBitViewMut<'life> {
    pub(crate) blocks: &'life mut [BitBlock],
}

impl Borrow<[BitBlock]> for AlignedBitView<'_> {
    fn borrow(&self) -> &[BitBlock] {
        self.blocks
    }
}

impl Borrow<[BitBlock]> for AlignedBitViewMut<'_> {
    fn borrow(&self) -> &[BitBlock] {
        self.blocks
    }
}

impl BorrowMut<[BitBlock]> for AlignedBitViewMut<'_> {
    fn borrow_mut(&mut self) -> &mut [BitBlock] {
        self.blocks
    }
}

impl BitLength for AlignedBitView<'_> {
    fn bit_len(&self) -> usize {
        self.blocks.bit_len()
    }
    const BLOCK_BIT_LEN: usize = BitBlock::BLOCK_BIT_LEN;
}

impl BitLength for AlignedBitViewMut<'_> {
    fn bit_len(&self) -> usize {
        self.blocks.bit_len()
    }
    const BLOCK_BIT_LEN: usize = BitBlock::BLOCK_BIT_LEN;
}

impl AlignedBitView<'_> {
    #[must_use]
    pub fn top(&self) -> u64 {
        self.blocks[0][0]
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bit_len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bit_len() == 0
    }
}

impl AlignedBitViewMut<'_> {
    #[must_use]
    pub fn top(&self) -> u64 {
        self.blocks[0][0]
    }

    pub fn top_mut(&mut self) -> &mut u64 {
        &mut self.blocks[0][0]
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bit_len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bit_len() == 0
    }
}

impl Hash for AlignedBitView<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.blocks.hash(state);
    }
}

impl Hash for AlignedBitViewMut<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.blocks.hash(state);
    }
}

impl<'ref_live> IntoBitIterator for &'ref_live AlignedBitView<'_> {
    type BitIterator = <&'ref_live [BitBlock] as IntoBitIterator>::BitIterator;

    fn iter_bits(self) -> Self::BitIterator {
        <&'ref_live [BitBlock] as IntoBitIterator>::iter_bits(self.blocks)
    }
}

impl<'ref_live> IntoBitIterator for &'ref_live AlignedBitViewMut<'_> {
    type BitIterator = <&'ref_live [BitBlock] as IntoBitIterator>::BitIterator;

    fn iter_bits(self) -> Self::BitIterator {
        <&'ref_live [BitBlock] as IntoBitIterator>::iter_bits(self.blocks)
    }
}

impl<'ref_life> IntoIterator for &'ref_life AlignedBitView<'_> {
    into_iterator_via_bit_iterator_body!(&'ref_life AlignedBitView<'ref_life>);
}

impl AlignedBitView<'_> {
    #[must_use]
    pub fn iter(&self) -> <&Self as IntoBitIterator>::BitIterator {
        <&Self as IntoIterator>::into_iter(self)
    }
}

impl<'ref_life> IntoIterator for &'ref_life AlignedBitViewMut<'_> {
    into_iterator_via_bit_iterator_body!(&'ref_life AlignedBitViewMut<'ref_life>);
}

impl AlignedBitViewMut<'_> {
    #[must_use]
    pub fn iter(&self) -> <&Self as IntoBitIterator>::BitIterator {
        <&Self as IntoIterator>::into_iter(self)
    }
}

delegate_bitwise!(AlignedBitViewMut<'_>, BitwiseViaBorrow<[BitBlock]>);
delegate_bitwise_mut!(AlignedBitViewMut<'_>, BitwiseMutViaBorrow<[BitBlock]>);

delegate_bitwise_pair!(
    AlignedBitViewMut<'_>,
    AlignedBitViewMut<'_>,
    BitwisePairViaBorrow<AlignedBitViewMut<'_>, [BitBlock]>
);
delegate_bitwise_pair!(
    AlignedBitViewMut<'_>,
    AlignedBitView<'_>,
    BitwisePairViaBorrow<AlignedBitView<'_>, [BitBlock]>
);
delegate_bitwise_pair!(AlignedBitViewMut<'_>, AlignedBitVec, BitwisePairViaBorrow<AlignedBitVec,[BitBlock]>);
delegate_bitwise_pair!(
    AlignedBitVec,
    AlignedBitViewMut<'_>,
    BitwisePairViaBorrow<AlignedBitViewMut<'_>, [BitBlock]>
);

delegate_bitwise_pair_mut!(
    AlignedBitViewMut<'_>,
    AlignedBitViewMut<'_>,
    BitwisePairMutViaBorrow<AlignedBitViewMut<'_>, [BitBlock]>
);
delegate_bitwise_pair_mut!(
    AlignedBitViewMut<'_>,
    AlignedBitView<'_>,
    BitwisePairMutViaBorrow<AlignedBitView<'_>, [BitBlock]>
);
delegate_bitwise_pair_mut!(AlignedBitViewMut<'_>, AlignedBitVec, BitwisePairMutViaBorrow<AlignedBitVec, [BitBlock]>);
delegate_bitwise_pair_mut!(
    AlignedBitVec,
    AlignedBitViewMut<'_>,
    BitwisePairMutViaBorrow<AlignedBitViewMut<'_>, [BitBlock]>
);
delegate_bitwise_pair_mut!(
    AlignedBitVec,
    AlignedBitView<'_>,
    BitwisePairMutViaBorrow<AlignedBitView<'_>, [BitBlock]>
);

delegate_bitwise!(AlignedBitView<'_>, BitwiseViaBorrow<[BitBlock]>);

delegate_bitwise_pair!(
    AlignedBitView<'_>,
    AlignedBitView<'_>,
    BitwisePairViaBorrow<AlignedBitView<'_>, [BitBlock]>
);
delegate_bitwise_pair!(
    AlignedBitView<'_>,
    AlignedBitViewMut<'_>,
    BitwisePairViaBorrow<AlignedBitViewMut<'_>, [BitBlock]>
);
delegate_bitwise_pair!(AlignedBitView<'_>, AlignedBitVec, BitwisePairViaBorrow<AlignedBitVec, [BitBlock]>);
delegate_bitwise_pair!(
    AlignedBitVec,
    AlignedBitView<'_>,
    BitwisePairViaBorrow<AlignedBitView<'_>, [BitBlock]>
);

impl<'life> From<AlignedBitView<'life>> for AlignedBitVec {
    fn from(value: AlignedBitView<'life>) -> Self {
        Self::from_view(&value)
    }
}

impl<'life> From<AlignedBitViewMut<'life>> for AlignedBitVec {
    fn from(value: AlignedBitViewMut<'life>) -> Self {
        Self::from_view_mut(&value)
    }
}

impl PartialEq<AlignedBitView<'_>> for AlignedBitVec {
    fn eq(&self, other: &AlignedBitView<'_>) -> bool {
        self.blocks.as_slice() == other.blocks
    }
}

impl PartialEq<AlignedBitViewMut<'_>> for AlignedBitVec {
    fn eq(&self, other: &AlignedBitViewMut<'_>) -> bool {
        self.blocks.as_slice() == other.blocks
    }
}

impl PartialEq<AlignedBitView<'_>> for AlignedBitViewMut<'_> {
    fn eq(&self, other: &AlignedBitView<'_>) -> bool {
        self.blocks == other.blocks
    }
}

impl PartialEq<AlignedBitVec> for AlignedBitView<'_> {
    fn eq(&self, other: &AlignedBitVec) -> bool {
        self.blocks == other.blocks.as_slice()
    }
}

impl PartialEq<AlignedBitVec> for AlignedBitViewMut<'_> {
    fn eq(&self, other: &AlignedBitVec) -> bool {
        self.blocks == other.blocks.as_slice()
    }
}
