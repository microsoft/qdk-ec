use binar::vec::AlignedBitVec;
use binar::{
    BitBlock, BitLength, BitVec, BitView, BitViewMut, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IndexSet,
    IntoBitIterator, or_support,
};
use proptest::prelude::*;

#[test]
fn bitview_from_aligned_validates_bounds_and_tail_padding() {
    for length in [1, 63, 64, 511, 513] {
        let mut dirty = AlignedBitVec::zeros(length + 512);
        dirty.assign_index(length, true);
        assert!(std::panic::catch_unwind(|| BitView::from_aligned(length, dirty.as_view())).is_err());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = BitViewMut::from_aligned(length, dirty.as_view_mut());
            }))
            .is_err()
        );
    }

    let mut short = AlignedBitVec::zeros(512);
    assert!(std::panic::catch_unwind(|| BitView::from_aligned(513, short.as_view())).is_err());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = BitViewMut::from_aligned(513, short.as_view_mut());
        }))
        .is_err()
    );

    let mut dirty_later_word = AlignedBitVec::zeros(512);
    dirty_later_word.assign_index(511, true);
    assert!(std::panic::catch_unwind(|| BitView::from_aligned(1, dirty_later_word.as_view())).is_err());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = BitViewMut::from_aligned(1, dirty_later_word.as_view_mut());
        }))
        .is_err()
    );
}

proptest! {
    #[test]
    fn from_iter(bits in prop::collection::vec(any::<bool>(), 0..2000)) {
        let bitvec = AlignedBitVec::from_iter(bits.clone());
        let actual: Vec<bool> = bitvec.into_iter().collect();
        assert_eq!(bits.support().collect::<Vec<usize>>(), actual.support().collect::<Vec<usize>>());
    }

    #[test]
    fn index(bits in prop::collection::vec(any::<bool>(), 0..2000)) {
        let bitvec = AlignedBitVec::from_iter(bits.clone());
        for (index, expected) in bits.iter().enumerate() {
            assert_eq!(bitvec.index(index), *expected);
        }
    }

    #[test]
    fn weight(bits in arbitrary_bitvec(2000)) {
        let ones = bits.into_iter().filter(|bit| *bit);
        assert_eq!(ones.count(), bits.weight());
    }

    #[test]
    fn support(bits in arbitrary_bitvec(2000)) {
        let support: Vec<usize> = bits.support().collect();
        assert_eq!(support.len(), bits.weight());
        for index in support {
            assert!(bits.index(index));
        }
    }

    #[test]
    fn assign(bits in prop::collection::vec(any::<bool>(), 0..10)) {
        let mut bitvec = AlignedBitVec::of_length(bits.len());
        for (index, bit) in bits.iter().enumerate() {
            bitvec.assign_index(index, *bit);
        }
        let actual: Vec<bool> = bitvec.into_iter().take(bits.len()).collect();
        assert_eq!(bits, actual);
    }

    #[test]
    fn bitxor_assign((left, right) in equal_length_bitvecs(2000)) {
        let mut xored = left.clone();
        xored.bitxor_assign(&right);
        for (index, result) in xored.into_iter().enumerate() {
            assert_eq!(result, left.index(index) ^ right.index(index));
        }
    }

    #[test]
    fn and_weight((left, right) in equal_length_bitvecs(2000)) {
        let expected = (0..left.bit_len())
            .filter(|&i| left.index(i) && right.index(i))
            .count();
        assert_eq!(left.and_weight(&right), expected);
    }

    #[test]
    fn or_weight((left, right) in equal_length_bitvecs(2000)) {
        let expected = (0..left.bit_len())
            .filter(|&i| left.index(i) || right.index(i))
            .count();
        assert_eq!(left.or_weight(&right), expected);
    }

    #[test]
    fn xor_weight((left, right) in equal_length_bitvecs(2000)) {
        let expected = (0..left.bit_len())
            .filter(|&i| left.index(i) ^ right.index(i))
            .count();
        assert_eq!(left.xor_weight(&right), expected);
    }

    #[test]
    fn interval_copy_matches_bit_assignments(
        source in arbitrary_bitvec(2048),
        destination in arbitrary_bitvec(2048),
        offset in 0usize..4096,
        length in 0usize..4096,
    ) {
        let start = offset % (source.bit_len() + 1);
        let count = length % ((source.bit_len() - start).min(destination.bit_len()) + 1);
        let mut expected = destination.clone();
        for index in 0..count {
            expected.assign_index(index, source.index(start + index));
        }
        let mut actual = destination.clone();
        actual.assign_from_interval(&source, start, count);
        prop_assert_eq!(&actual, &expected);
        let mut via_view = destination;
        via_view.as_view_mut().assign_from_interval(&source.as_view(), start, count);
        prop_assert_eq!(via_view, expected);
    }

    #[test]
    fn dot_product((left, right) in equal_length_bitvecs(2000)) {
        let common_bits = (0..left.bit_len())
            .filter(|&i| left.index(i) && right.index(i))
            .count();
        let expected = (common_bits % 2) == 1;
        assert_eq!(left.dot(&right), expected);
    }

}

#[test]
fn bitvec_is_unit_rejects_extra_bits_in_other_blocks() {
    let first_bit_index = 0;
    let second_block_bit_index = BitBlock::BLOCK_BIT_LEN;
    let width = second_block_bit_index + 1;
    let mut bitvec = BitVec::zeros(width);
    bitvec.assign_index(first_bit_index, true);
    bitvec.assign_index(second_block_bit_index, true);

    assert_eq!(bitvec.weight(), 2);
    assert!(!bitvec.is_unit(first_bit_index));
    assert!(!bitvec.is_unit(second_block_bit_index));
}

#[test]
fn aligned_bitvec_is_unit_rejects_extra_bits_in_other_blocks() {
    let first_bit_index = 0;
    let second_block_bit_index = BitBlock::BLOCK_BIT_LEN;
    let width = second_block_bit_index + 1;
    let mut bitvec = AlignedBitVec::zeros(width);
    bitvec.assign_index(first_bit_index, true);
    bitvec.assign_index(second_block_bit_index, true);

    assert_eq!(bitvec.weight(), 2);
    assert!(!bitvec.is_unit(first_bit_index));
    assert!(!bitvec.is_unit(second_block_bit_index));
}

#[test]
fn interval_copy_preserves_suffix_across_word_and_block_boundaries() {
    for start in [
        0, 1, 2, 31, 32, 63, 64, 65, 127, 128, 129, 510, 511, 512, 513, 1023, 1024, 1025,
    ] {
        for count in [
            0, 1, 2, 31, 32, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025,
        ] {
            let mut source = AlignedBitVec::ones(start + count);
            for index in 0..source.bit_len() {
                source.assign_index(index, (index * 37) % 67 < 33);
            }
            let mut destination = AlignedBitVec::ones(count + 65);
            assert!(source.aligned_words().is_some());
            assert!(destination.aligned_words_mut().is_some());
            let mut expected = destination.clone();
            for index in 0..count {
                expected.assign_index(index, source.index(start + index));
            }
            destination.assign_from_interval(&source, start, count);
            assert_eq!(destination, expected, "start={start}, count={count}");
        }
    }
}

#[test]
fn empty_interval_copy_does_not_access_storage() {
    let source = AlignedBitVec::zeros(0);
    let mut destination = AlignedBitVec::zeros(0);
    destination.assign_from_interval(&source, usize::MAX, 0);
    assert_eq!(destination, AlignedBitVec::zeros(0));
}

#[test]
fn interval_copy_keeps_the_non_aligned_fallback() {
    let source = 0xa529_7cf0_56ab_9123u64;
    assert!(source.aligned_words().is_none());
    for start in 0..=64 {
        for count in 0..=64 - start {
            let mut destination = 0x7f31_2a95_c604_b8edu64;
            let mut expected = destination;
            for index in 0..count {
                expected.assign_index(index, source.index(start + index));
            }
            destination.assign_from_interval(&source, start, count);
            assert_eq!(destination, expected, "start={start}, count={count}");
        }
    }
}

#[test]
fn or_support_handles_unequal_lengths_truncated_views_and_fallback() {
    use sorted_iter::SortedIterator;

    let empty = AlignedBitVec::zeros(0);
    let mut short = AlignedBitVec::zeros(64);
    short.assign_index(0, true);
    short.assign_index(63, true);
    let mut long = AlignedBitVec::zeros(600);
    long.assign_index(63, true);
    long.assign_index(64, true);
    long.assign_index(512, true);

    assert_eq!(or_support(&empty, &empty).collect::<Vec<_>>(), Vec::<usize>::new());
    assert_eq!(or_support(&empty, &short).collect::<Vec<_>>(), vec![0, 63]);
    assert_eq!(or_support(&short, &long).collect::<Vec<_>>(), vec![0, 63, 64, 512]);
    assert_eq!(or_support(&long, &short).collect::<Vec<_>>(), vec![0, 63, 64, 512]);

    let storage = BitVec::ones(3);
    let view = storage.as_view();
    let sparse: IndexSet = [1, 5].into_iter().collect();
    assert!(view.aligned_words_with_bit_len().is_some());
    assert_eq!(or_support(&view, &sparse).collect::<Vec<_>>(), vec![0, 1, 2, 5]);
    assert_eq!(or_support(&view, &short).collect::<Vec<_>>(), vec![0, 1, 2, 63]);
    assert_eq!(or_support(&view, &view).collect::<Vec<_>>(), vec![0, 1, 2]);

    let left_vec = BitVec::zeros(1);
    let left_view = left_vec.as_view();
    let mut right_vec = BitVec::zeros(70);
    right_vec.assign_index(64, true);
    right_vec.assign_index(69, true);
    let right_view = right_vec.as_view();
    let expected: Vec<_> = left_view.support().union(right_view.support()).collect();
    assert_eq!(expected, vec![64, 69]);
    assert_eq!(or_support(&left_view, &right_view).collect::<Vec<_>>(), expected);
    assert_eq!(or_support(&right_view, &left_view).collect::<Vec<_>>(), expected);

    let ones_block = AlignedBitVec::ones(512);
    let zero_view = BitView::from_aligned(0, ones_block.as_view());
    let mut empty_or = or_support(&left_view, &zero_view);
    assert_eq!(empty_or.by_ref().collect::<Vec<_>>(), Vec::<usize>::new());
    assert_eq!(empty_or.next(), None);

    let mut left_mut_vec = BitVec::zeros(65);
    left_mut_vec.assign_index(64, true);
    let mut right_mut_vec = BitVec::zeros(513);
    right_mut_vec.assign_index(512, true);
    assert!(left_mut_vec.aligned_words_with_bit_len().is_some());
    assert_eq!(
        or_support(&left_mut_vec, &right_mut_vec).collect::<Vec<_>>(),
        vec![64, 512]
    );
    let left_mut = left_mut_vec.as_view_mut();
    let right_mut = right_mut_vec.as_view_mut();
    assert!(left_mut.aligned_words_with_bit_len().is_some());
    let mut mut_iter = or_support(&left_mut, &right_mut);
    assert_eq!(mut_iter.by_ref().collect::<Vec<_>>(), vec![64, 512]);
    assert_eq!(mut_iter.next(), None);
}

#[test]
fn selected_from_packs_bits_contiguously_across_block_boundaries() {
    let mut source = BitVec::zeros(1050);
    for bit in [2, 5, 64, 511, 512, 513, 1024, 1049] {
        source.assign_index(bit, true);
    }
    let indices = [1049, 512, 5, 511, 0, 1024, 600, 2];
    let expected = vec![0, 1, 2, 3, 5, 7];

    let selected = BitVec::selected_from(&source.as_view(), &indices);
    assert_eq!(selected.len(), indices.len());
    assert_eq!(selected.support().collect::<Vec<_>>(), expected);

    let aligned_source: AlignedBitVec = source.into();
    let aligned = AlignedBitVec::selected_from(&aligned_source.as_view(), &indices);
    assert_eq!(aligned.support().collect::<Vec<_>>(), expected);
}

#[test]
#[should_panic(expected = "index < self.bit_len()")]
fn bitvec_selected_from_panics_on_out_of_bounds_view_index() {
    let source = BitVec::ones(10);
    let _ = BitVec::selected_from(&source.as_view(), &[0, 10]);
}

#[test]
fn bitvec_from_aligned_clears_padding_and_trims_blocks() {
    for length in [0, 5, 64, 512, 513] {
        let cleaned = BitVec::from_aligned(length, AlignedBitVec::ones(length + 512));
        let expected = BitVec::ones(length);
        assert_eq!(cleaned.weight(), length);
        assert_eq!(cleaned.max_support(), length.checked_sub(1));
        assert_eq!(cleaned, expected);
    }

    assert_eq!(BitVec::from_words(5, &[0xFF]), BitVec::ones(5));
    assert_eq!(BitVec::ones(70).extract(2, 7), BitVec::ones(5));
    assert_eq!(BitVec::from_aligned(600, AlignedBitVec::zeros(0)), BitVec::zeros(600));
    let mut expected_600 = BitVec::zeros(600);
    expected_600.assign_index(0, true);
    expected_600.assign_index(2, true);
    assert_eq!(BitVec::from_words(600, &[0b101]), expected_600);
}

#[test]
fn bitview_from_aligned_trims_excess_trailing_blocks() {
    for length in [0, 5, 512] {
        let mut storage = AlignedBitVec::zeros(1024);
        let mut expected = BitVec::zeros(length);
        if length > 0 {
            storage.assign_index(0, true);
            expected.assign_index(0, true);
        }
        let trailing = length.div_ceil(BitBlock::BLOCK_BIT_LEN) * BitBlock::BLOCK_BIT_LEN;
        storage.assign_index(trailing, true);

        let view = BitView::from_aligned(length, storage.as_view());
        assert_eq!(view.weight(), expected.weight());
        assert_eq!(view, expected.as_view());

        BitViewMut::from_aligned(length, storage.as_view_mut()).clear_bits();
        assert!(storage.index(trailing));
    }
}

#[test]
fn or_support_falls_back_when_aligned_words_with_bit_len_is_none() {
    use sorted_iter::{SortedIterator, assume::AssumeSortedByItemExt};

    struct Low3(BitBlock);

    impl Bitwise for Low3 {
        fn index(&self, i: usize) -> bool {
            assert!(i < 3);
            self.0.index(i)
        }

        fn support(&self) -> impl SortedIterator<Item = usize> {
            (0..3).filter(|&i| self.index(i)).assume_sorted_by_item()
        }

        fn aligned_words(&self) -> Option<&[u64]> {
            Some(&self.0.blocks[..1])
        }
    }

    let custom_view = Low3(BitBlock::from([0b1000, 0, 0, 0, 0, 0, 0, 0]));
    assert!(custom_view.aligned_words().is_some());
    assert!(custom_view.aligned_words_with_bit_len().is_none());
    assert_eq!(custom_view.support().next(), None);
    assert_eq!(or_support(&custom_view, &custom_view).next(), None);
}

#[test]
fn aligned_bitvec_resize_shrink_clears_retained_block_tail() {
    for new_length in [0, 1, 63, 64, 65, 500, 512, 513, 1024] {
        let mut bits = AlignedBitVec::ones(1024);
        for index in (0..new_length).step_by(3) {
            bits.assign_index(index, false);
        }
        bits.resize(new_length);
        let expected_blocks = new_length.div_ceil(BitBlock::BLOCK_BIT_LEN);
        assert_eq!(bits.len(), expected_blocks * BitBlock::BLOCK_BIT_LEN);
        for index in 0..bits.len() {
            let expected = index < new_length && !index.is_multiple_of(3);
            assert_eq!(bits.index(index), expected, "new_length={new_length}, index={index}");
        }
    }
}

#[test]
fn with_length_from_iter_handles_word_and_block_boundaries() {
    for length in [0usize, 1, 63, 64, 65, 511, 512, 513, 1024] {
        let expected_capacity = length.div_ceil(BitBlock::BLOCK_BIT_LEN) * BitBlock::BLOCK_BIT_LEN;
        let (all_false, false_len) = AlignedBitVec::with_length_from_iter(std::iter::repeat_n(false, length));
        assert_eq!(false_len, length);
        assert_eq!(all_false.len(), expected_capacity);
        assert!(all_false.is_zero());

        let (mixed, mixed_len) = AlignedBitVec::with_length_from_iter((0..length).map(|index| index % 5 == 0));
        assert_eq!(mixed_len, length);
        assert_eq!(mixed.len(), expected_capacity);
        for index in 0..mixed.len() {
            assert_eq!(mixed.index(index), index < length && index % 5 == 0);
        }
    }
}

#[test]
fn unsigned_int_slice_support_matches_boolean_oracle() {
    let words: Vec<u64> = vec![0, 1, 1 << 63, u64::MAX, 0, 0xA5A5_0000_0000_5A5A];
    let expected: Vec<usize> = words
        .as_slice()
        .iter_bits()
        .enumerate()
        .filter_map(|(index, bit)| bit.then_some(index))
        .collect();
    assert_eq!(words.support().collect::<Vec<_>>(), expected);
    assert_eq!(words.as_slice().support().collect::<Vec<_>>(), expected);

    let short_words: [u16; 4] = [0, 0x8001, 0, 0xFFFF];
    let short_expected: Vec<usize> = short_words
        .as_slice()
        .iter_bits()
        .enumerate()
        .filter_map(|(index, bit)| bit.then_some(index))
        .collect();
    assert_eq!(short_words.support().collect::<Vec<_>>(), short_expected);
}

fn arbitrary_bitvec(max_length: usize) -> impl Strategy<Value = AlignedBitVec> {
    prop::collection::vec(any::<bool>(), 0..max_length).prop_map(AlignedBitVec::from_iter)
}

fn equal_length_bitvecs(max_length: usize) -> impl Strategy<Value = (AlignedBitVec, AlignedBitVec)> {
    (0..max_length).prop_flat_map(|length| {
        (
            prop::collection::vec(any::<bool>(), length).prop_map(AlignedBitVec::from_iter),
            prop::collection::vec(any::<bool>(), length).prop_map(AlignedBitVec::from_iter),
        )
    })
}
