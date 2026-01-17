/// Transpose 2^k x 2^k blocks in place
/// Block size is 2^k ( that is 1<<k ), mask must be for every other 2^k bits in 64 bit word
fn swap_blocks_64<const BLOCK_SIZE: usize, const BLOCK_MASK: u64>(matrix: &mut [u64; 64]) {
    for block_top_row in (0..64).step_by(2 * BLOCK_SIZE) {
        for row_offset in 0..BLOCK_SIZE {
            let row_id_from = block_top_row + row_offset;
            let row_id_to = row_id_from + BLOCK_SIZE;
            let row_from = matrix[row_id_from];
            let row_to = matrix[row_id_to];
            // swap via 3 xor trick: swap(a,b) ~ (a = a^b , b = a^b , a = a^b)
            let diff = ((row_from >> BLOCK_SIZE) ^ row_to) & BLOCK_MASK;
            matrix[row_id_to] = row_to ^ diff;
            matrix[row_id_from] = row_from ^ (diff << BLOCK_SIZE);
        }
    }
}

#[allow(
    clippy::unusual_byte_groupings,
    reason = "Better readability via nested groups of every 4 and every 16 bits"
)]
pub fn transpose_64x64_inplace(matrix: &mut [u64; 64]) {
    // mask for every other 2^k bits in 64 bit word for k from 0 to 5
    const BLOCK_MASKS: [u64; 6] = [
        0b___0101_0101_0101_0101___0101_0101_0101_0101___0101_0101_0101_0101___0101_0101_0101_0101,
        0b___0011_0011_0011_0011___0011_0011_0011_0011___0011_0011_0011_0011___0011_0011_0011_0011,
        0b___0000_1111_0000_1111___0000_1111_0000_1111___0000_1111_0000_1111___0000_1111_0000_1111,
        0b___0000_0000_1111_1111___0000_0000_1111_1111___0000_0000_1111_1111___0000_0000_1111_1111,
        0b___0000_0000_0000_0000___1111_1111_1111_1111___0000_0000_0000_0000___1111_1111_1111_1111,
        0b___0000_0000_0000_0000___0000_0000_0000_0000___1111_1111_1111_1111___1111_1111_1111_1111,
    ];

    // See Hacker’s Delight, Second Edition Henry S. Warren Jr., Chapter 7.3 Transpose of a Bit Matrix
    // We perform transpose of 2x2, 4x4, ..., 32x32 blocks
    swap_blocks_64::<{ 1 << 0 }, { BLOCK_MASKS[0] }>(matrix);
    swap_blocks_64::<{ 1 << 1 }, { BLOCK_MASKS[1] }>(matrix);
    swap_blocks_64::<{ 1 << 2 }, { BLOCK_MASKS[2] }>(matrix);
    swap_blocks_64::<{ 1 << 3 }, { BLOCK_MASKS[3] }>(matrix);
    swap_blocks_64::<{ 1 << 4 }, { BLOCK_MASKS[4] }>(matrix);
    swap_blocks_64::<{ 1 << 5 }, { BLOCK_MASKS[5] }>(matrix);
}

/// Delta swap: swap bits at positions i and i+delta where mask bit is 1.
#[inline]
fn delta_swap(x: u64, mask: u64, delta: usize) -> u64 {
    let t = ((x >> delta) ^ x) & mask;
    x ^ t ^ (t << delta)
}

/// Transpose an 8x64 bit matrix (8 u64s) into 64 8-bit patterns.
/// Output word i contains patterns 8*i to 8*i+7 packed as bytes.
#[inline]
pub(super) fn transpose_8x64(w: [u64; 8]) -> [u64; 8] {
    let mut out = [0u64; 8];

    // Process 8 columns at a time (one byte from each input word)
    for (byte_idx, out_word) in out.iter_mut().enumerate() {
        // Pack byte byte_idx from each of 8 words into one u64
        let mut x = 0u64;
        for (word_idx, word) in w.iter().enumerate() {
            x |= ((word >> (byte_idx * 8)) & 0xFF) << (word_idx * 8);
        }

        // Transpose the 8x8 bit matrix using delta swaps
        let x = delta_swap(x, 0x00AA_00AA_00AA_00AA, 7);
        let x = delta_swap(x, 0x0000_CCCC_0000_CCCC, 14);
        let x = delta_swap(x, 0x0000_0000_F0F0_F0F0, 28);

        // Unpack: now byte j contains pattern for column 8*byte_idx + j
        *out_word = x;
    }

    out
}
