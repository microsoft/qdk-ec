use std::{
    borrow::{Borrow, BorrowMut},
    ops::{Deref, DerefMut},
};

use super::BitVec;
use crate::{
    BitLength, IntoBitIterator,
    bit::{
        bitwise_via_borrow::{BitwiseMutViaBorrow, BitwisePairMutViaBorrow, BitwisePairViaBorrow, BitwiseViaBorrow},
        truncated::BitsTruncated,
    },
    delegate_bitwise, delegate_bitwise_body, delegate_bitwise_mut, delegate_bitwise_mut_body,
    delegate_bitwise_pair_body, delegate_bitwise_pair_mut_body,
    vec::{
        AlignedBitVec,
        aligned_view::{AlignedBitView, AlignedBitViewMut},
    },
};
use crate::{Bitwise, BitwiseMut, BitwisePair, BitwisePairMut};

// Should we use convention <TypeName>Mutable or Mutable<TypeName> ?

pub type ViewInner<'life> = BitsTruncated<AlignedBitView<'life>>;
#[must_use]
#[derive(Eq, Debug, Hash)]
pub struct BitView<'life> {
    pub(crate) inner: ViewInner<'life>,
}

impl<'life> Deref for BitView<'life> {
    type Target = ViewInner<'life>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'life> Borrow<ViewInner<'life>> for BitView<'life> {
    fn borrow(&self) -> &ViewInner<'life> {
        &self.inner
    }
}

impl<'life> From<ViewInner<'life>> for BitView<'life> {
    fn from(inner: ViewInner<'life>) -> Self {
        Self { inner }
    }
}

impl<'life> From<BitView<'life>> for AlignedBitView<'life> {
    fn from(vec: BitView<'life>) -> Self {
        vec.inner.bits
    }
}

impl PartialEq for BitView<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

pub type ViewInnerMut<'life> = BitsTruncated<AlignedBitViewMut<'life>>;
#[must_use]
#[derive(Eq, Debug, Hash)]
pub struct BitViewMut<'life> {
    pub(crate) inner: ViewInnerMut<'life>,
}

impl<'life> Deref for BitViewMut<'life> {
    type Target = ViewInnerMut<'life>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for BitViewMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'life> Borrow<ViewInnerMut<'life>> for BitViewMut<'life> {
    fn borrow(&self) -> &ViewInnerMut<'life> {
        &self.inner
    }
}

impl<'life> BorrowMut<ViewInnerMut<'life>> for BitViewMut<'life> {
    fn borrow_mut(&mut self) -> &mut ViewInnerMut<'life> {
        &mut self.inner
    }
}

impl<'life> From<ViewInnerMut<'life>> for BitViewMut<'life> {
    fn from(inner: ViewInnerMut<'life>) -> Self {
        Self { inner }
    }
}

impl<'life> From<BitViewMut<'life>> for AlignedBitViewMut<'life> {
    fn from(vec: BitViewMut<'life>) -> Self {
        vec.inner.bits
    }
}

impl PartialEq for BitViewMut<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl BitView<'_> {
    pub fn from_aligned(bit_length: usize, bits: AlignedBitView<'_>) -> BitView<'_> {
        ViewInner { bits, bit_length }.into()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bit_len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bit_len() == 0
    }

    #[must_use]
    pub fn top(&self) -> u64 {
        self.bits.top()
    }

    #[must_use]
    pub fn iter(&self) -> <&Self as IntoBitIterator>::BitIterator {
        <&Self as IntoBitIterator>::iter_bits(self)
    }
}

impl<'life> IntoBitIterator for &'life BitView<'_> {
    type BitIterator = <&'life VecInner as IntoBitIterator>::BitIterator;

    fn iter_bits(self) -> Self::BitIterator {
        <&'life ViewInner as IntoBitIterator>::iter_bits(self.borrow())
    }
}

impl<'life> BitLength for BitView<'life> {
    fn bit_len(&self) -> usize {
        self.bit_length
    }
    const BLOCK_BIT_LEN: usize = AlignedBitView::<'life>::BLOCK_BIT_LEN;
}

type VecInner = BitsTruncated<AlignedBitVec>;
delegate_bitwise!(BitView<'_>, BitwiseViaBorrow<ViewInner<'_>>);
impl<'life1> BitwisePair<BitVec> for BitView<'life1> {
    delegate_bitwise_pair_body!(BitVec, BitwisePairViaBorrow<BitVec, ViewInner<'life1>, VecInner>);
}
impl<'life1, 'life2> BitwisePair<BitView<'life2>> for BitView<'life1> {
    delegate_bitwise_pair_body!(
        BitView<'life2>,
        BitwisePairViaBorrow<BitView<'life2>, ViewInner<'life1>, ViewInner<'life2>>
    );
}
impl<'life1, 'life2> BitwisePair<BitViewMut<'life2>> for BitView<'life1> {
    delegate_bitwise_pair_body!(
        BitViewMut<'life2>,
        BitwisePairViaBorrow<BitViewMut<'life2>, ViewInner<'life1>, ViewInnerMut<'life2>>
    );
}
impl<'life1> BitwisePair<BitView<'life1>> for BitVec {
    delegate_bitwise_pair_body!(
        BitView<'life1>,
        BitwisePairViaBorrow<BitView<'life1>, VecInner, ViewInner<'life1>>
    );
}

impl BitViewMut<'_> {
    pub fn from_aligned(bit_length: usize, bits: AlignedBitViewMut<'_>) -> BitViewMut<'_> {
        ViewInnerMut { bits, bit_length }.into()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bit_length
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bit_length == 0
    }

    #[must_use]
    pub fn top(&self) -> u64 {
        self.bits.top()
    }

    pub fn top_mut(&mut self) -> &mut u64 {
        self.bits.top_mut()
    }

    #[must_use]
    pub fn iter(&self) -> <&Self as IntoBitIterator>::BitIterator {
        <&Self as IntoBitIterator>::iter_bits(self)
    }
}

impl<'life> BitLength for BitViewMut<'life> {
    fn bit_len(&self) -> usize {
        self.inner.bit_length
    }
    const BLOCK_BIT_LEN: usize = AlignedBitView::<'life>::BLOCK_BIT_LEN;
}

impl<'life> IntoBitIterator for &'life BitViewMut<'_> {
    type BitIterator = <&'life ViewInnerMut<'life> as IntoBitIterator>::BitIterator;

    fn iter_bits(self) -> Self::BitIterator {
        <&'life ViewInnerMut as IntoBitIterator>::iter_bits(self.borrow())
    }
}

delegate_bitwise!(BitViewMut<'_>, BitwiseViaBorrow<ViewInnerMut<'_>>);
delegate_bitwise_mut!(BitViewMut<'_>, BitwiseMutViaBorrow<ViewInnerMut<'_>>);

impl<'life1> BitwisePair<BitVec> for BitViewMut<'life1> {
    delegate_bitwise_pair_body!(BitVec, BitwisePairViaBorrow<BitVec, ViewInnerMut<'life1>, VecInner>);
}
impl<'life1, 'life2> BitwisePair<BitViewMut<'life2>> for BitViewMut<'life1> {
    delegate_bitwise_pair_body!(
        BitViewMut<'life2>,
        BitwisePairViaBorrow<BitViewMut<'life2>, ViewInnerMut<'life1>, ViewInnerMut<'life2>>
    );
}
impl<'life1, 'life2> BitwisePair<BitView<'life2>> for BitViewMut<'life1> {
    delegate_bitwise_pair_body!(
        BitView<'life2>,
        BitwisePairViaBorrow<BitView<'life2>, ViewInnerMut<'life1>, ViewInner<'life2>>
    );
}
impl<'life1> BitwisePair<BitViewMut<'life1>> for BitVec {
    delegate_bitwise_pair_body!(
        BitViewMut<'life1>,
        BitwisePairViaBorrow<BitViewMut<'life1>, VecInner, ViewInnerMut<'life1>>
    );
}

impl<'life1> BitwisePairMut<BitVec> for BitViewMut<'life1> {
    delegate_bitwise_pair_mut_body!(BitVec, BitwisePairMutViaBorrow<BitVec, ViewInnerMut<'life1>, VecInner>);
}
impl<'life1, 'life2> BitwisePairMut<BitViewMut<'life2>> for BitViewMut<'life1> {
    delegate_bitwise_pair_mut_body!(
        BitViewMut<'life2>,
        BitwisePairMutViaBorrow<BitViewMut<'life2>, ViewInnerMut<'life1>, ViewInnerMut<'life2>>
    );
}
impl<'life1, 'life2> BitwisePairMut<BitView<'life2>> for BitViewMut<'life1> {
    delegate_bitwise_pair_mut_body!(
        BitView<'life2>,
        BitwisePairMutViaBorrow<BitView<'life2>, ViewInnerMut<'life1>, ViewInner<'life2>>
    );
}
impl<'life1> BitwisePairMut<BitViewMut<'life1>> for BitVec {
    delegate_bitwise_pair_mut_body!(
        BitViewMut<'life1>,
        BitwisePairMutViaBorrow<BitViewMut<'life1>, VecInner, ViewInnerMut<'life1>>
    );
}
impl<'life1> BitwisePairMut<BitView<'life1>> for BitVec {
    delegate_bitwise_pair_mut_body!(
        BitView<'life1>,
        BitwisePairMutViaBorrow<BitView<'life1>, VecInner, ViewInner<'life1>>
    );
}

impl<'life> From<&'life BitView<'life>> for BitVec {
    fn from(value: &'life BitView<'life>) -> Self {
        Self::from_view(value)
    }
}

impl<'life> From<&'life BitVec> for BitVec {
    fn from(value: &'life BitVec) -> Self {
        value.clone()
    }
}

impl<'life> From<&'life BitViewMut<'life>> for BitVec {
    fn from(value: &'life BitViewMut<'life>) -> Self {
        Self::from_view_mut(value)
    }
}
