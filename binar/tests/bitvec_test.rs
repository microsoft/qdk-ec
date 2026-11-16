use binar::vec::AlignedBitVec;
use binar::{BitBlock, BitLength, BitVec, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut};
use proptest::prelude::*;

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
