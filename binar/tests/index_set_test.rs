use binar::{Bitwise, IndexSet};
use proptest::prelude::*;

proptest! {
    #[test]
    fn extremes_match_support(indexes in prop::collection::vec(0..10_000usize, 0..64)) {
        let set: IndexSet = indexes.iter().copied().collect();
        prop_assert_eq!(set.max_support(), set.support().last());
        prop_assert_eq!(set.min_support(), set.support().next());
        prop_assert_eq!(set.max_support(), indexes.iter().copied().max());
        prop_assert_eq!(set.min_support(), indexes.iter().copied().min());
    }
}
