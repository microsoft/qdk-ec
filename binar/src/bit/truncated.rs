use std::borrow::{Borrow, BorrowMut};

use crate::{BitLength, Bitwise};

#[derive(Debug, Clone, Eq)]
pub struct BitsTruncated<Bits>
where
    Bits: Bitwise,
{
    pub bits: Bits,
    pub bit_length: usize,
}

impl<Bits> Borrow<Bits> for BitsTruncated<Bits>
where
    Bits: Bitwise,
{
    fn borrow(&self) -> &Bits {
        &self.bits
    }
}

impl<Bits> BorrowMut<Bits> for BitsTruncated<Bits>
where
    Bits: Bitwise,
{
    fn borrow_mut(&mut self) -> &mut Bits {
        &mut self.bits
    }
}

impl<Bits> BitLength for BitsTruncated<Bits>
where
    Bits: Bitwise + BitLength,
{
    fn bit_len(&self) -> usize {
        self.bit_length
    }
    const BLOCK_BIT_LEN: usize = Bits::BLOCK_BIT_LEN;
}

impl<Bits> PartialEq for BitsTruncated<Bits>
where
    Bits: Bitwise + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.bit_length == other.bit_length && self.bits == other.bits
    }
}
