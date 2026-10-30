pub const WILDCARD: u64 = 0;
pub const FUTURE_CHECK_CID: u64 = u64::MAX - 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorIndex {
    pub eid: usize,
    pub error_index: usize,
}

// let the coordinator keep the internal loaded library
pub const KEEP_FLAG_LIBRARY: u64 = 1 << 0;
// let the coordinator keep the loaded decoders; note that since decoders usually
// depend on the library, keeping the decoders while not keeping the library may
// lead to inconsistent result
pub const KEEP_FLAG_DECODER: u64 = 1 << 1;
// keep everything possible
pub const KEEP_FLAG_EVERYTHING: u64 = !0;

pub(crate) type CollectedIds = rangemap::RangeInclusiveSet<u64>;

#[cfg(test)]
mod collected_id_tests {
    use super::CollectedIds;

    #[test]
    fn merges_intervals_without_losing_sparse_or_extreme_ids() {
        let mut ids = CollectedIds::default();
        for id in [0, 5, 3, 4, 2, 1, 5, u64::MAX, u64::MAX - 1] {
            ids.insert(id..=id);
        }
        assert_eq!(ids.len(), 2);
        for id in [0, 1, 2, 3, 4, 5, u64::MAX - 1, u64::MAX] {
            assert!(ids.contains(&id));
        }
        assert!(!ids.contains(&6));
        assert!(!ids.contains(&(u64::MAX - 2)));
        ids.clear();
        assert_eq!(ids.len(), 0);
        assert!(!ids.contains(&0));
    }
}
