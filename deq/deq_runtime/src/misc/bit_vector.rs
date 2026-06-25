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

/// Validate that `bit_vector.data` has exactly the number of bytes required
/// for `bit_vector.size` bits. Returns `Ok(())` on success, or a descriptive
/// error string on failure.
pub fn validate_data_len(bit_vector: &BitVector, name: &str) -> Result<(), String> {
    let required = bit_vector_len(bit_vector.size);
    if bit_vector.data.len() != required {
        Err(format!(
            "{name} data length ({}) does not match required length ({required}) for {} bits",
            bit_vector.data.len(),
            bit_vector.size
        ))
    } else {
        Ok(())
    }
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

/// Extract a contiguous slice of bits from a BitVector
pub fn slice(bit_vector: &BitVector, start: usize, length: usize) -> BitVector {
    debug_assert!(start + length <= bit_vector.size as usize);
    let mut result = BitVector {
        size: length as u64,
        data: vec![0u8; bit_vector_len(length as u64)],
    };
    for i in 0..length {
        if get_bit(bit_vector, (start + i) as u64) {
            set_bit(&mut result, i as u64, true);
        }
    }
    result
}

/// Append bits from another BitVector to the end of this one
pub fn append(bit_vector: &mut BitVector, other: &BitVector) {
    let original_size = bit_vector.size;
    extend_num_bits(bit_vector, other.size);
    for i in 0..other.size {
        if get_bit(other, i) {
            set_bit(bit_vector, original_size + i, true);
        }
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
}
