use binar::vec::BitBlock;
use binar::{BitLength, Bitwise, BitwiseMut, BitwisePairMut, IntoBitIterator};
use proptest::prelude::*;

proptest! {
    #[test]
    fn support_matches_boolean_enumeration(words in prop::collection::vec(any::<[u64; 8]>(), 0..20)) {
        let blocks: Vec<BitBlock> = words.into_iter().map(BitBlock::from).collect();
        let expected: Vec<_> = blocks.as_slice().iter_bits().enumerate()
            .filter_map(|(index, bit)| bit.then_some(index)).collect();
        prop_assert_eq!(blocks.as_slice().support().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn from_array(bits in arbitrary_bool_array()) {
        let block = array_to_bitblock(&bits);
        for (index, &expected) in bits.iter().enumerate() {
            assert_eq!(block.index(index), expected);
        }
    }

    #[test]
    fn set(block in arbitrary_bitblock(), index in 0..BITS) {
        let mut clone = block.clone();
        for value in [true, false] {
            clone.assign_index(index, value);
            assert_eq!(clone.index(index), value);
            for index2 in 0..BITS {
                if index != index2 {
                    assert_eq!(clone.index(index2), block.index(index2));
                }
            }
        }
    }

    #[test]
    fn xor(left in arbitrary_bitblock(), right in arbitrary_bitblock()) {
        let mut xor = left.clone();
        xor.bitxor_assign(&right);
        for index in 0..BITS {
            assert_eq!(xor.index(index), left.index(index) ^ right.index(index));
        }
    }

    #[test]
    fn xor_assign(left in arbitrary_bitblock(), right in arbitrary_bitblock()) {
        let mut xor = left.clone();
        xor.bitxor_assign(&right);
        xor.bitxor_assign(&right);
        assert_eq!(left, xor);
    }

    #[test]
    fn and(left in arbitrary_bitblock(), right in arbitrary_bitblock()) {
        let mut and = left.clone();
        and.bitand_assign(&right);
        for index in 0..BITS {
            assert_eq!(and.index(index), left.index(index) & right.index(index));
        }
    }

    #[test]
    fn and_assign(left in arbitrary_bitblock(), right in arbitrary_bitblock()) {
        let mut and = left.clone();
        and.bitand_assign(&right);
        let mut and2 = and.clone();
        and2.bitand_assign(&right);
        assert_eq!(and, and2);
    }

}

#[test]
fn support_word_and_block_boundaries() {
    for blocks in [
        vec![],
        vec![BitBlock::default(); 3],
        vec![BitBlock::ones(); 3],
        vec![
            BitBlock::from([1, 1 << 63, 0, 0, 0, 0, 0, 1 << 63]),
            BitBlock::default(),
            BitBlock::from([1, 0, u64::MAX, 0, 0, 0, 0, 1 << 63]),
        ],
    ] {
        let expected: Vec<_> = blocks
            .as_slice()
            .iter_bits()
            .enumerate()
            .filter_map(|(index, bit)| bit.then_some(index))
            .collect();
        let mut support = blocks.as_slice().support();
        assert_eq!(support.by_ref().collect::<Vec<_>>(), expected);
        assert_eq!(support.next(), None);
        assert_eq!(support.next(), None);
        for block in &blocks {
            let expected: Vec<_> = block
                .iter_bits()
                .enumerate()
                .filter_map(|(index, bit)| bit.then_some(index))
                .collect();
            assert_eq!(block.support().collect::<Vec<_>>(), expected);
        }
    }
}

#[test]
fn zeros() {
    let block = BitBlock::default();
    for index in 0..BITS {
        assert!(!block.index(index), "{}", index);
    }
}

#[test]
fn ones() {
    let block = BitBlock::ones();
    for index in 0..BITS {
        assert!(block.index(index), "{}", index);
    }
}

#[test]
fn all() {
    for value in [true, false] {
        let block = if value { BitBlock::ones() } else { BitBlock::default() };
        for index in 0..BITS {
            assert_eq!(block.index(index), value, "{index}");
        }
    }
}

const BITS: usize = BitBlock::BLOCK_BIT_LEN;

fn arbitrary_bool_array() -> impl Strategy<Value = [bool; BITS]> {
    proptest::array::uniform::<proptest::bool::Any, BITS>(proptest::bool::ANY)
}

fn array_to_bitblock(bits: &[bool; BITS]) -> BitBlock {
    bits.iter().copied().collect::<BitBlock>()
}

fn arbitrary_bitblock() -> impl Strategy<Value = BitBlock> {
    arbitrary_bool_array().prop_map(|bits| array_to_bitblock(&bits))
}
