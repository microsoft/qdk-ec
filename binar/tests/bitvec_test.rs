use binar::vec::AlignedBitVec;
use binar::{BitLength, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut};
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
    fn dot_product((left, right) in equal_length_bitvecs(2000)) {
        let common_bits = (0..left.bit_len())
            .filter(|&i| left.index(i) && right.index(i))
            .count();
        let expected = (common_bits % 2) == 1;
        assert_eq!(left.dot(&right), expected);
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
