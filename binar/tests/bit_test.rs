use binar::{BitVec, BitwisePairMut, IndexSet, vec::BitBlock};

/// # Panics
///
/// Will panic
pub fn test_one_bit_index<T: BitwisePairMut + Clone>(mut bits: T, index: usize) {
    bits.clear_bits();
    assert!(bits.is_zero());
    bits.assign_index(index, true);
    assert!(bits.index(index));
    assert!(bits.support().any(|elt| elt == index));
    assert_eq!(bits.support().count(), 1);
    assert_eq!(bits.weight(), 1);
    assert!(bits.parity());
    bits.negate_index(index);
    assert!(!bits.index(index));
    assert_eq!(bits.support().count(), 0);
    assert_eq!(bits.weight(), 0);
    assert!(!bits.parity());

    let mut other_bits = bits.clone();
    other_bits.clear_bits();
    assert_eq!(other_bits.weight(), 0);
    bits.negate_index(index);
    assert!(other_bits.is_zero());
    other_bits.bitxor_assign(&bits);
    assert_eq!(other_bits.support().count(), 1);
    assert_eq!(other_bits.weight(), 1);
    assert!(other_bits.parity());
    bits.clear_bits();

    other_bits.negate_index(index + 1);
    other_bits.bitand_assign(&bits);
    assert_eq!(bits.weight(), 0);
    assert!(!bits.parity());
}

/// # Panics
///
/// Will panic
pub fn test_unary_bit_traits<T: BitwisePairMut + Default + Clone>(size: usize, index: usize) {
    assert!(index + 1 < size);
    let item = T::default();
    test_one_bit_index(item, index);
}

macro_rules! call_test_per_uint {
    ($function:ident,$uint:ty,$first:expr $(, $rest:expr)*) => {
        $function::<$uint>($first,$($rest),*);
        $function::<[$uint;ARRAY_SIZE]>($first,$($rest),*);
    };
}

macro_rules! call_test {
    ($function:ident,$first:expr $(, $rest:expr)*) => {
        call_test_per_uint!($function,u16,$first,$($rest),*);
        call_test_per_uint!($function,u32,$first,$($rest),*);
        call_test_per_uint!($function,u64,$first,$($rest),*);
        call_test_per_uint!($function,u128,$first,$($rest),*);
        $function::<IndexSet>($first,$($rest),*);
    };
}

#[test]
fn bit_test() {
    const ARRAY_SIZE: usize = 4;
    let size = 10;
    let index = 7;
    call_test!(test_unary_bit_traits, size, index);
    test_one_bit_index(BitBlock::default(), index);
    test_one_bit_index(BitVec::zeros(size), index);
    test_one_bit_index(vec![0u16; size / 16 + 1], index);
    test_one_bit_index(vec![0u32; size / 32 + 1], index);
    test_one_bit_index(vec![0u64; size / 64 + 1], index);
    test_one_bit_index(vec![0u128; size / 128 + 1], index);
    test_one_bit_index(vec![[0u16; ARRAY_SIZE]; 1], index);
    test_one_bit_index(vec![[0u32; ARRAY_SIZE]; 1], index);
    test_one_bit_index(vec![[0u64; ARRAY_SIZE]; 1], index);
    test_one_bit_index(vec![[0u128; ARRAY_SIZE]; 1], index);
}
