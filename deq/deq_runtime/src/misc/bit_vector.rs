use crate::util::BitVector;

pub fn pack_bits(array: &[bool]) -> Vec<u8> {
    let mut packed: Vec<u8> = vec![0; bit_vector_len(array.len() as u64)];
    for (i, &bit) in array.iter().enumerate() {
        if bit {
            packed[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    packed
}

pub fn unpack_bits(data: &[u8], size: u64) -> Vec<bool> {
    debug_assert!(data.len() == bit_vector_len(size));
    assert!((size as usize) <= data.len() * 8);
    let mut bits: Vec<bool> = vec![false; size as usize];
    for i in 0..size as usize {
        bits[i] = (data[i / 8] & (1 << (7 - (i % 8)))) != 0;
    }
    bits
}

pub fn xor_bits(a: &BitVector, b: &BitVector) -> BitVector {
    let mut result = a.clone();
    xor_bits_in_place(&mut result, b);
    result
}

pub fn xor_bits_in_place(a: &mut BitVector, b: &BitVector) {
    debug_assert!(a.size == b.size);
    for i in 0..a.data.len() {
        a.data[i] ^= b.data[i];
    }
}

#[inline]
pub fn byte_bit_index(index: u64) -> (usize, u8) {
    let byte_index = (index / 8) as usize;
    let bit_index = 7 - ((index % 8) as u8);
    (byte_index, bit_index)
}

#[inline]
pub fn bit_vector_len(size: u64) -> usize {
    size.div_ceil(8) as usize // (size + 7) / 8
}

pub fn from_sparse_indices(size: u64, indices: &[u64]) -> BitVector {
    let mut data = vec![0u8; bit_vector_len(size)];
    for &idx in indices {
        let (byte_index, bit_index) = byte_bit_index(idx);
        data[byte_index] |= 1 << bit_index;
    }
    BitVector { size, data }
}

pub fn to_sparse_indices(bit_vector: &BitVector) -> Vec<u64> {
    let mut indices = vec![];
    for index in 0..bit_vector.size {
        if get_bit(bit_vector, index) {
            indices.push(index);
        }
    }
    indices
}

/// Return whether every meaningful bit is zero, ignoring unused padding bits in
/// the final byte.
pub fn is_zero(bit_vector: &BitVector) -> bool {
    debug_assert!(crate::misc::validation::validate_data_len(bit_vector, "bit vector").is_ok());
    let full_bytes = (bit_vector.size / 8) as usize;
    if bit_vector.data[..full_bytes].iter().any(|&byte| byte != 0) {
        return false;
    }
    let remainder = bit_vector.size % 8;
    remainder == 0 || bit_vector.data[full_bytes] >> (8 - remainder) == 0
}

pub fn extend_num_bits(bit_vector: &mut BitVector, extending_length: u64) {
    bit_vector.size += extending_length;
    let new_data_len = bit_vector_len(bit_vector.size);
    bit_vector.data.resize(new_data_len, 0u8);
}

pub fn get_bit(bit_vector: &BitVector, index: u64) -> bool {
    debug_assert!(index < bit_vector.size);
    let (byte_index, bit_index) = byte_bit_index(index);
    (bit_vector.data[byte_index] & (1 << bit_index)) != 0
}

pub fn set_bit(bit_vector: &mut BitVector, index: u64, value: bool) {
    debug_assert!(index < bit_vector.size);
    let (byte_index, bit_index) = byte_bit_index(index);
    if value {
        bit_vector.data[byte_index] |= 1 << bit_index;
    } else {
        bit_vector.data[byte_index] &= !(1 << bit_index);
    }
}

pub fn flip_bit(bit_vector: &mut BitVector, index: u64) {
    debug_assert!(index < bit_vector.size);
    let (byte_index, bit_index) = byte_bit_index(index);
    bit_vector.data[byte_index] ^= 1 << bit_index;
}

pub fn bit_vector_to_string(bit_vector: &BitVector) -> String {
    let mut s = String::with_capacity(bit_vector.size as usize);
    for i in 0..bit_vector.size {
        if get_bit(bit_vector, i) {
            s.push('1');
        } else {
            s.push('0');
        }
    }
    s
}

pub fn binar_bitvec_to_bit_vector(bit_vec: &binar::BitVec, length: usize) -> BitVector {
    let mut data = vec![0u8; bit_vector_len(length as u64)];
    for (index, value) in (0..length).zip(bit_vec.iter()) {
        let (byte_index, bit_index) = byte_bit_index(index as u64);
        data[byte_index] |= (value as u8) << bit_index;
    }
    BitVector {
        size: length as u64,
        data,
    }
}

pub(crate) fn bit_vector_to_binar_bitvec(bit_vector: &BitVector) -> binar::BitVec {
    let length = usize::try_from(bit_vector.size).expect("bit vector length exceeds addressable memory");
    let mut words: Vec<_> = bit_vector.data[..bit_vector_len(bit_vector.size)]
        .chunks(8)
        .map(|chunk| {
            let mut bytes = [0; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            u64::from_be_bytes(bytes).reverse_bits()
        })
        .collect();
    if let Some(last) = words.last_mut()
        && !length.is_multiple_of(64)
    {
        *last &= u64::MAX >> (64 - length % 64);
    }
    binar::BitVec::from_words(length, &words)
}

/// Extract a contiguous slice of bits from a BitVector
pub fn slice(bit_vector: &BitVector, start: usize, length: usize) -> BitVector {
    debug_assert!(start + length <= bit_vector.size as usize);
    let byte_start = start / 8;
    let shift = start % 8;
    let byte_count = bit_vector_len(length as u64);
    let mut data = if shift == 0 {
        bit_vector.data[byte_start..byte_start + byte_count].to_vec()
    } else {
        let mut data = Vec::with_capacity(byte_count);
        for index in byte_start..byte_start + byte_count {
            let mut byte = bit_vector.data[index] << shift;
            if let Some(next) = bit_vector.data.get(index + 1) {
                byte |= next >> (8 - shift);
            }
            data.push(byte);
        }
        data
    };
    if let Some(last) = data.last_mut()
        && !length.is_multiple_of(8)
    {
        *last &= u8::MAX << (8 - length % 8);
    }
    BitVector {
        size: length as u64,
        data,
    }
}

/// Append bits from another `BitVector` to the end of this one.
///
/// # Panics
/// Panics if either vector has insufficient storage or the combined length
/// cannot be represented in memory.
pub fn append(bit_vector: &mut BitVector, other: &BitVector) {
    if other.size == 0 {
        return;
    }
    let original_size = bit_vector.size;
    extend_num_bits(bit_vector, other.size);
    let byte_start = usize::try_from(original_size / 8).unwrap();
    let shift = original_size % 8;
    let source = &other.data[..bit_vector_len(other.size)];
    if shift == 0 {
        bit_vector.data[byte_start..byte_start + source.len()].copy_from_slice(source);
    } else {
        bit_vector.data[byte_start] &= u8::MAX << (8 - shift);
        for (index, &byte) in source.iter().enumerate() {
            bit_vector.data[byte_start + index] |= byte >> shift;
            if let Some(next) = bit_vector.data.get_mut(byte_start + index + 1) {
                *next |= byte << (8 - shift);
            }
        }
    }
    if !bit_vector.size.is_multiple_of(8) {
        *bit_vector.data.last_mut().unwrap() &= u8::MAX << (8 - bit_vector.size % 8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(string: &str) -> Vec<bool> {
        string
            .chars()
            .map(|c| match c {
                '0' => false,
                '1' => true,
                _ => panic!("invalid character in bit string"),
            })
            .collect()
    }

    #[test]
    fn bit_pack() {
        // cargo test bit_pack -- --nocapture

        // np.packbits([1,0,0,0,0,0,0,0]) = array([128], dtype=uint8)
        assert!(pack_bits(&bits("10000000")) == vec![128u8]);
        // np.packbits([1,0,0,0,0,0,0,0,0,1]) = array([128,  64], dtype=uint8)
        assert!(pack_bits(&bits("1000000001")) == vec![128u8, 64u8]);
        // np.packbits([0,0,0,1,0,1,0,0,1,1]) = array([ 20, 192], dtype=uint8)
        assert!(pack_bits(&bits("0001010011")) == vec![20u8, 192u8]);

        assert!(unpack_bits(&[128u8], 8) == bits("10000000"));
        assert!(unpack_bits(&[128u8, 64u8], 10) == bits("1000000001"));
        assert!(unpack_bits(&[20u8, 192u8], 10) == bits("0001010011"));
    }

    #[test]
    fn packed_slices_preserve_every_alignment_and_clear_padding() {
        let bits: Vec<bool> = (0..137).map(|index| index % 11 == 3 || index % 7 < 3).collect();
        let mut source = BitVector {
            size: u64::try_from(bits.len()).unwrap(),
            data: pack_bits(&bits),
        };
        *source.data.last_mut().unwrap() |= 0x7f;
        for start in 0..=bits.len() {
            for length in 0..=bits.len() - start {
                let result = slice(&source, start, length);
                assert_eq!(unpack_bits(&result.data, result.size), bits[start..start + length]);
                if !length.is_multiple_of(8) {
                    assert_eq!(result.data.last().unwrap() & (u8::MAX >> (length % 8)), 0);
                }
            }
        }
    }

    #[test]
    fn packed_appends_preserve_every_alignment_without_exposing_padding() {
        for left_length in 0..=73 {
            for right_length in 0..=73 {
                let left: Vec<bool> = (0..left_length).map(|index| index % 7 < 3).collect();
                let right: Vec<bool> = (0..right_length).map(|index| index % 5 == 1).collect();
                let packed = |bits: &[bool]| {
                    let mut data = pack_bits(bits);
                    if let Some(last) = data.last_mut()
                        && !bits.len().is_multiple_of(8)
                    {
                        *last |= u8::MAX >> (bits.len() % 8);
                    }
                    BitVector {
                        size: u64::try_from(bits.len()).unwrap(),
                        data,
                    }
                };
                let mut result = packed(&left);
                append(&mut result, &packed(&right));
                let expected: Vec<_> = left.into_iter().chain(right).collect();
                assert_eq!(unpack_bits(&result.data, result.size), expected);
                if right_length > 0 && !expected.len().is_multiple_of(8) {
                    assert_eq!(result.data.last().unwrap() & (u8::MAX >> (expected.len() % 8)), 0);
                }
            }
        }
    }

    #[test]
    fn packed_binar_conversion_preserves_word_boundaries_and_ignores_padding() {
        for length in 0usize..=513 {
            let bits: Vec<bool> = (0..length).map(|index| index % 17 < 3 || index % 13 == 4).collect();
            let mut source = BitVector {
                size: u64::try_from(length).unwrap(),
                data: pack_bits(&bits),
            };
            if let Some(last) = source.data.last_mut()
                && !length.is_multiple_of(8)
            {
                *last |= u8::MAX >> (length % 8);
            }
            assert_eq!(
                bit_vector_to_binar_bitvec(&source),
                bits.into_iter().collect::<binar::BitVec>()
            );
        }
    }

    #[test]
    fn zero_check_ignores_padding_bits() {
        assert!(is_zero(&BitVector { size: 0, data: vec![] }));
        assert!(is_zero(&BitVector {
            size: 1,
            data: vec![0b0111_1111]
        }));
        assert!(!is_zero(&BitVector {
            size: 1,
            data: vec![0b1000_0000]
        }));
        assert!(is_zero(&BitVector {
            size: 9,
            data: vec![0, 0b0111_1111],
        }));
        assert!(!is_zero(&BitVector {
            size: 9,
            data: vec![0, 0b1000_0000],
        }));
    }
}
