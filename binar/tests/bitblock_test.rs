use binar::vec::BitBlock;
use binar::{BitLength, Bitwise, BitwiseMut, BitwisePairMut};
use proptest::prelude::*;

proptest! {
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
