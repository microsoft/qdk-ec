//! Shared frame transforms specialized for zero, copy, and sparse matrices.

use crate::misc::bit_matrix::{apply_modifier, sparse_to_dense};
use crate::{bin, util};
use binar::{BitMatrix, BitVec, Bitwise, BitwiseMut, BitwisePair};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;

enum Plan {
    Zero,
    CopyPrefix,
    Sparse(Vec<(usize, usize)>),
    Dense,
}

pub(crate) struct FrameMatrix {
    dense: BitMatrix,
    plan: Plan,
}

impl FrameMatrix {
    fn new(dense: BitMatrix) -> Self {
        let entries: usize = dense.rows().map(|row| row.weight()).sum();
        let plan = if entries == 0 {
            Plan::Zero
        } else if dense.row_count() <= dense.column_count()
            && entries == dense.row_count()
            && dense.rows().enumerate().all(|(index, row)| row.index(index))
        {
            Plan::CopyPrefix
        } else if entries <= dense.as_words().len() {
            let mut indices = Vec::with_capacity(entries);
            for (row_index, row) in dense.rows().enumerate() {
                indices.extend(row.support().map(|column| (row_index, column)));
            }
            Plan::Sparse(indices)
        } else {
            Plan::Dense
        };
        Self { dense, plan }
    }

    pub(crate) fn multiply(&self, input: &BitVec) -> BitVec {
        assert_eq!(input.len(), self.column_count());
        if matches!(self.plan, Plan::Dense) {
            return &self.dense * &input.as_view();
        }
        let mut result = BitVec::zeros(self.row_count());
        self.xor_into(&mut result, input);
        result
    }

    pub(crate) fn xor_into(&self, target: &mut BitVec, input: &BitVec) {
        assert_eq!(target.len(), self.row_count());
        assert_eq!(input.len(), self.column_count());
        match &self.plan {
            Plan::Zero => {}
            Plan::CopyPrefix => {
                let length = target.len();
                let word_count = length.div_ceil(64);
                let source = input.as_words();
                let blocks: &mut [binar::BitBlock] = target.as_mut();
                for (index, word) in blocks
                    .iter_mut()
                    .flat_map(|block| block.blocks.iter_mut())
                    .take(word_count)
                    .enumerate()
                {
                    let mask = if index + 1 == word_count && !length.is_multiple_of(64) {
                        u64::MAX >> (64 - length % 64)
                    } else {
                        u64::MAX
                    };
                    *word ^= source[index] & mask;
                }
            }
            Plan::Sparse(indices) => {
                for &(row, column) in indices {
                    if input.index(column) {
                        target.negate_index(row);
                    }
                }
            }
            Plan::Dense => {
                let input = input.as_view();
                for (index, row) in self.dense.rows().enumerate() {
                    if row.dot(&input) {
                        target.negate_index(index);
                    }
                }
            }
        }
    }
}

impl std::ops::Deref for FrameMatrix {
    type Target = BitMatrix;

    fn deref(&self) -> &Self::Target {
        &self.dense
    }
}

struct CachedMatrix {
    source: Option<util::BitMatrix>,
    prepared: Arc<FrameMatrix>,
}

pub(crate) struct FrameMatrixCache {
    entries: lru::LruCache<u64, CachedMatrix>,
}

impl Default for FrameMatrixCache {
    fn default() -> Self {
        Self {
            entries: lru::LruCache::new(NonZeroUsize::new(128).unwrap()),
        }
    }
}

impl FrameMatrixCache {
    pub(crate) fn prepare(
        &mut self,
        source: Option<&util::BitMatrix>,
        modifier: Option<&bin::BitMatrixModifier>,
    ) -> Arc<FrameMatrix> {
        if let Some(modifier) = modifier {
            let base = self.prepare(source, None);
            return Arc::new(FrameMatrix::new(apply_modifier(base.dense.clone(), modifier)));
        }
        let mut hasher = DefaultHasher::new();
        source.is_some().hash(&mut hasher);
        if let Some(source) = source {
            source.rows.hash(&mut hasher);
            source.cols.hash(&mut hasher);
            source.i.hash(&mut hasher);
            source.j.hash(&mut hasher);
        }
        let key = hasher.finish();
        if let Some(cached) = self.entries.get(&key)
            && cached.source.as_ref() == source
        {
            return Arc::clone(&cached.prepared);
        }
        let dense = source.map_or_else(|| BitMatrix::zeros(0, 0), sparse_to_dense);
        let prepared = Arc::new(FrameMatrix::new(dense));
        self.entries.put(
            key,
            CachedMatrix {
                source: source.cloned(),
                prepared: Arc::clone(&prepared),
            },
        );
        prepared
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::bit_matrix::{append_bit, zeros};
    use binar::BitwisePairMut;

    #[test]
    fn plans_match_dense_products_and_preserve_padding_at_word_and_block_boundaries() {
        for rows in [0, 1, 7, 50, 65, 511, 512, 513] {
            for columns in [0, 1, 51, 64, 65, 101, 511, 512, 513] {
                for pattern in 0..4 {
                    let mut dense = BitMatrix::zeros(rows, columns);
                    for row in 0..rows {
                        for column in 0..columns {
                            let value = match pattern {
                                0 => false,
                                1 => row == column,
                                2 => (row * 7 + column) % 97 == 0,
                                _ => (row + column) % 3 != 0,
                            };
                            dense.set((row, column), value);
                        }
                    }
                    let input: BitVec = (0..columns).map(|index| index % 5 < 2).collect();
                    let product = &dense * &input.as_view();
                    let compiled = FrameMatrix::new(dense);
                    assert_eq!(compiled.multiply(&input), product);
                    let mut actual: BitVec = (0..rows).map(|index| index % 3 == 1).collect();
                    let mut expected = actual.clone();
                    expected.bitxor_assign(&product);
                    compiled.xor_into(&mut actual, &input);
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn cached_matrices_are_shared_without_reusing_changed_sources_or_mutating_the_base() {
        let mut source = zeros(2, 3);
        append_bit(&mut source, 0, 1);
        let source = Some(source);
        let mut cache = FrameMatrixCache::default();
        let first = cache.prepare(source.as_ref(), None);
        assert!(Arc::ptr_eq(&first, &cache.prepare(source.as_ref(), None)));
        let input: BitVec = [false, true, false].into_iter().collect();
        let expected = first.multiply(&input);
        let modifier = bin::BitMatrixModifier {
            toggle: source.clone(),
            overwrite: None,
        };
        let toggled = cache.prepare(source.as_ref(), Some(&modifier));
        assert!(toggled.multiply(&input).is_zero());
        let mut shape = zeros(1, 3);
        append_bit(&mut shape, 0, 2);
        let transform = bin::BitMatrixModifier {
            toggle: source.clone(),
            overwrite: Some(shape),
        };
        assert_eq!(
            cache.prepare(source.as_ref(), Some(&transform)).multiply(&input),
            BitVec::zeros(1)
        );
        assert_eq!(first.multiply(&input), expected);
        let mut replacement = source.clone();
        replacement.as_mut().unwrap().j[0] = 2;
        assert!(cache.prepare(replacement.as_ref(), None).multiply(&input).is_zero());
        cache.clear();
        assert_eq!(first.multiply(&input), expected);
    }
}
