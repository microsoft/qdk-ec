use binar::{Bitwise, BitwiseMut, BitwisePairMut};

/// Trait for types that have a neutral/zero element
pub trait NeutralElement {
    type NeutralElementType: 'static;
    fn neutral_element(&self) -> Self::NeutralElementType;
    fn default_size_neutral_element() -> Self::NeutralElementType;
    fn neutral_element_of_size(size: usize) -> Self::NeutralElementType;
}

// Implementations for basic types
impl NeutralElement for u8 {
    type NeutralElementType = u8;

    #[inline]
    fn neutral_element(&self) -> Self::NeutralElementType {
        0u8
    }

    #[inline]
    fn default_size_neutral_element() -> Self::NeutralElementType {
        0u8
    }

    #[inline]
    fn neutral_element_of_size(_size: usize) -> Self::NeutralElementType {
        0u8
    }
}

impl NeutralElement for &u8 {
    type NeutralElementType = u8;

    #[inline]
    fn neutral_element(&self) -> Self::NeutralElementType {
        0u8
    }

    #[inline]
    fn default_size_neutral_element() -> Self::NeutralElementType {
        0u8
    }

    #[inline]
    fn neutral_element_of_size(_size: usize) -> Self::NeutralElementType {
        0u8
    }
}

impl NeutralElement for &mut u8 {
    type NeutralElementType = u8;

    #[inline]
    fn neutral_element(&self) -> Self::NeutralElementType {
        0u8
    }

    #[inline]
    fn default_size_neutral_element() -> Self::NeutralElementType {
        0u8
    }

    #[inline]
    fn neutral_element_of_size(_size: usize) -> Self::NeutralElementType {
        0u8
    }
}

// Unary traits, involve only one type
pub trait BitwiseNeutralElement:
    Bitwise + NeutralElement<NeutralElementType: BitwisePairMut<Self> + NeutralElement + BitwiseMut>
{
}
