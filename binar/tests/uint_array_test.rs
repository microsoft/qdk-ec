use binar::{Bitwise, BitwiseMut};

#[test]
fn test_negate_index_basic_operations() {
    // Test single word and multi-word operations with various sizes
    let test_cases = [
        (1, 64),  // Single u64 word
        (2, 128), // Two words, tests 64-bit boundary
        (4, 256), // Four words, tests multiple boundaries
    ];

    for (word_count, total_bits) in test_cases {
        let mut arr = vec![0u64; word_count];

        // Test key boundary positions and some random positions
        let test_indices: Vec<usize> = (0..word_count)
            .flat_map(|word| [word * 64, word * 64 + 1, word * 64 + 31, word * 64 + 63])
            .filter(|&i| i < total_bits)
            .collect();

        for &index in &test_indices {
            // Set bit
            arr.negate_index(index);

            let expected_word = index / 64;
            let expected_bit = index % 64;
            let expected_value = 1u64 << expected_bit;

            assert_eq!(
                arr[expected_word], expected_value,
                "Failed to set bit at index {index} (word {expected_word}, bit {expected_bit})",
            ); // Verify other words are unaffected
            for (word_idx, &word_value) in arr.iter().enumerate() {
                if word_idx != expected_word {
                    assert_eq!(word_value, 0, "Word {word_idx} should be 0 when setting bit {index}");
                }
            }

            // Clear bit
            arr.negate_index(index);
            assert_eq!(arr[expected_word], 0, "Failed to clear bit at index {index}");
        }
    }
}

#[test]
fn test_negate_index_different_word_types() {
    // Test with different word types to ensure generic implementation works

    // Test u32 arrays
    let mut arr_u32: [u32; 4] = [0; 4];
    let u32_test_indices = [0, 31, 32, 63, 64, 95, 96, 127];

    for &index in &u32_test_indices {
        arr_u32.negate_index(index);

        let expected_word = index / 32;
        let expected_bit = index % 32;
        let expected_value = 1u32 << expected_bit;

        assert_eq!(
            arr_u32[expected_word], expected_value,
            "u32 array failed at index {index}"
        );
        arr_u32.negate_index(index); // Reset
    }

    // Test u64 arrays
    let mut arr_u64: [u64; 2] = [0; 2];
    let u64_test_indices = [0, 63, 64, 127];

    for &index in &u64_test_indices {
        arr_u64.negate_index(index);

        let expected_word = index / 64;
        let expected_bit = index % 64;
        let expected_value = 1u64 << expected_bit;

        assert_eq!(
            arr_u64[expected_word], expected_value,
            "u64 array failed at index {index}"
        );
        arr_u64.negate_index(index); // Reset
    }
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn test_negate_index_out_of_bounds() {
    let mut arr: [u64; 2] = [0; 2];
    arr.negate_index(128); // Beyond 2 * 64 = 128 bits
}

#[test]
fn max_support_test() {
    for j in 1..u64::BITS as usize {
        let x = 1u64 << j;
        assert_eq!(x.support().last().unwrap(), x.max_support().unwrap());
        assert_eq!(x.max_support().unwrap(), x.min_support().unwrap());
    }
}
