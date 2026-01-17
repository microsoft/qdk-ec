use std::{iter::zip, ops::Range};

use sorted_iter::SortedIterator;

use crate::bit::bitwise_via_iter::{BitwisePairMutViaIter, BitwisePairViaIter};
use crate::bit::standard_types::support_iterator;
use crate::{
    BitLength, BitVec, BitView, BitViewMut, Bitwise, BitwisePair, BitwisePairMut, IntoBitIterator,
    delegate_bitwise_pair, delegate_bitwise_pair_body, delegate_bitwise_pair_mut, delegate_bitwise_pair_mut_body,
    vec::{AlignedBitVec, AlignedBitView, AlignedBitViewMut, BitAccessor, BitBlock},
};

#[derive(Clone, Debug, Hash)]
pub struct Column<'life> {
    pub(crate) rows: &'life [*mut BitBlock],
    pub(crate) accessor: BitAccessor,
    pub(crate) block_index: usize,
}

impl Column<'_> {
    #[must_use]
    pub fn iter(&self) -> ColumnIterator<'_> {
        <&Self as IntoIterator>::into_iter(self)
    }
}

pub struct ColumnIterator<'life> {
    column: &'life Column<'life>,
    row_index: usize,
}

impl Iterator for ColumnIterator<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_index >= self.column.rows.len() {
            return None;
        }
        let output = self.column.index(self.row_index);
        self.row_index += 1;
        Some(output)
    }
}

impl<'life> IntoIterator for &'life Column<'_> {
    type Item = bool;
    type IntoIter = ColumnIterator<'life>;
    fn into_iter(self) -> Self::IntoIter {
        ColumnIterator {
            column: self,
            row_index: 0,
        }
    }
}

impl<'ref_life> IntoBitIterator for &'ref_life Column<'_> {
    type BitIterator = ColumnIterator<'ref_life>;
    fn iter_bits(self) -> Self::BitIterator {
        self.iter()
    }
}

impl BitLength for Column<'_> {
    fn bit_len(&self) -> usize {
        self.rows.len()
    }

    const BLOCK_BIT_LEN: usize = 1;
}

impl PartialEq for Column<'_> {
    fn eq(&self, other: &Self) -> bool {
        if self.bit_len() != other.bit_len() {
            return false;
        }
        for (a, b) in zip(self.iter_bits(), other.iter_bits()) {
            if a != b {
                return false;
            }
        }
        true
    }
}

impl Eq for Column<'_> {}

impl Column<'_> {
    #[must_use]
    pub fn slice(&self, range: Range<usize>) -> Self {
        Column {
            rows: &self.rows[range],
            accessor: self.accessor.clone(),
            block_index: self.block_index,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Bitwise for Column<'_> {
    fn support(&self) -> impl SortedIterator<Item = usize> {
        support_iterator(self.iter_bits())
    }

    fn index(&self, index: usize) -> bool {
        let block = unsafe { &*self.rows[index].add(self.block_index) };
        self.accessor.array_value_of(block)
    }
}

delegate_bitwise_pair!(Column<'_>, Column<'_>, BitwisePairViaIter<Column<'_>>);

delegate_bitwise_pair!(Column<'_>, BitVec, BitwisePairViaIter<BitVec>);
delegate_bitwise_pair!(BitVec, Column<'_>, BitwisePairViaIter<Column<'_>>);

delegate_bitwise_pair!(Column<'_>, BitView<'_>, BitwisePairViaIter<BitView<'_>>);
delegate_bitwise_pair!(BitView<'_>, Column<'_>, BitwisePairViaIter<Column<'_>>);

delegate_bitwise_pair!(Column<'_>, BitViewMut<'_>, BitwisePairViaIter<BitViewMut<'_>>);
delegate_bitwise_pair!(BitViewMut<'_>, Column<'_>, BitwisePairViaIter<Column<'_>>);

delegate_bitwise_pair_mut!(BitViewMut<'_>, Column<'_>, BitwisePairMutViaIter<Column<'_>>);
delegate_bitwise_pair_mut!(BitVec, Column<'_>, BitwisePairMutViaIter<Column<'_>>);

delegate_bitwise_pair!(Column<'_>, AlignedBitVec, BitwisePairViaIter<AlignedBitVec>);
delegate_bitwise_pair!(AlignedBitVec, Column<'_>, BitwisePairViaIter<Column<'_>>);

delegate_bitwise_pair!(Column<'_>, AlignedBitView<'_>, BitwisePairViaIter<AlignedBitView<'_>>);
delegate_bitwise_pair!(AlignedBitView<'_>, Column<'_>, BitwisePairViaIter<Column<'_>>);

delegate_bitwise_pair!(
    Column<'_>,
    AlignedBitViewMut<'_>,
    BitwisePairViaIter<AlignedBitViewMut<'_>>
);
delegate_bitwise_pair!(AlignedBitViewMut<'_>, Column<'_>, BitwisePairViaIter<Column<'_>>);

delegate_bitwise_pair_mut!(AlignedBitViewMut<'_>, Column<'_>, BitwisePairMutViaIter<Column<'_>>);
delegate_bitwise_pair_mut!(AlignedBitVec, Column<'_>, BitwisePairMutViaIter<Column<'_>>);
