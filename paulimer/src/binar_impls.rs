// Implementations of NeutralElement and BitwiseNeutralElement for binar types
use crate::traits::{BitwiseNeutralElement, NeutralElement};
use binar::matrix::Column;
use binar::vec::{AlignedBitVec, AlignedBitView, AlignedBitViewMut, BitBlock};
use binar::{BitLength, BitVec, BitView, BitViewMut, IndexSet};

// BitVec implementations
macro_rules! vecview_neutral_element_body {
    () => {
        type NeutralElementType = BitVec;

        fn neutral_element(&self) -> <Self as NeutralElement>::NeutralElementType {
            BitVec::default_size_neutral_element()
        }

        fn default_size_neutral_element() -> <Self as NeutralElement>::NeutralElementType {
            BitVec::zeros(0)
        }

        fn neutral_element_of_size(size: usize) -> <Self as NeutralElement>::NeutralElementType {
            BitVec::zeros(size)
        }
    };
}

impl NeutralElement for BitVec {
    vecview_neutral_element_body!();
}

impl NeutralElement for BitViewMut<'_> {
    vecview_neutral_element_body!();
}

impl NeutralElement for BitView<'_> {
    vecview_neutral_element_body!();
}

impl BitwiseNeutralElement for BitVec {}
impl BitwiseNeutralElement for BitViewMut<'_> {}
impl BitwiseNeutralElement for BitView<'_> {}

// AlignedBitVec implementations
macro_rules! blocks_neutral_element_body {
    () => {
        type NeutralElementType = AlignedBitVec;

        fn neutral_element(&self) -> <Self as NeutralElement>::NeutralElementType {
            AlignedBitVec::zeros(self.len())
        }

        fn default_size_neutral_element() -> <Self as NeutralElement>::NeutralElementType {
            AlignedBitVec::zeros(0)
        }

        fn neutral_element_of_size(size: usize) -> <Self as NeutralElement>::NeutralElementType {
            AlignedBitVec::zeros(size)
        }
    };
}

impl NeutralElement for AlignedBitVec {
    blocks_neutral_element_body!();
}

impl NeutralElement for AlignedBitViewMut<'_> {
    blocks_neutral_element_body!();
}

impl NeutralElement for AlignedBitView<'_> {
    blocks_neutral_element_body!();
}

impl BitwiseNeutralElement for AlignedBitVec {}
impl BitwiseNeutralElement for AlignedBitViewMut<'_> {}
impl BitwiseNeutralElement for AlignedBitView<'_> {}

// BitBlock implementations
impl NeutralElement for BitBlock {
    type NeutralElementType = BitBlock;

    fn neutral_element(&self) -> <Self as NeutralElement>::NeutralElementType {
        Self::default()
    }

    fn default_size_neutral_element() -> <Self as NeutralElement>::NeutralElementType {
        Self::default()
    }

    fn neutral_element_of_size(size: usize) -> <Self as NeutralElement>::NeutralElementType {
        assert!(size <= BitBlock::BLOCK_BIT_LEN);
        Self::default_size_neutral_element()
    }
}

impl BitwiseNeutralElement for BitBlock {}

// IndexSet implementations
impl NeutralElement for IndexSet {
    type NeutralElementType = IndexSet;

    fn neutral_element(&self) -> Self::NeutralElementType {
        IndexSet::new()
    }

    fn default_size_neutral_element() -> Self::NeutralElementType {
        IndexSet::new()
    }

    fn neutral_element_of_size(_size: usize) -> Self::NeutralElementType {
        IndexSet::new()
    }
}

impl BitwiseNeutralElement for IndexSet {}

// AlignedBitMatrix Column implementations
impl NeutralElement for Column<'_> {
    type NeutralElementType = BitVec;

    fn neutral_element(&self) -> Self::NeutralElementType {
        BitVec::zeros(self.len())
    }

    fn default_size_neutral_element() -> Self::NeutralElementType {
        BitVec::zeros(0)
    }

    fn neutral_element_of_size(size: usize) -> Self::NeutralElementType {
        BitVec::zeros(size)
    }
}

impl BitwiseNeutralElement for Column<'_> {}

// Implementations for arrays of unsigned integers
macro_rules! implement_neutral_element_for_uint_array {
    ($uint_type:ty) => {
        impl<const WORD_COUNT: usize> NeutralElement for [$uint_type; WORD_COUNT] {
            type NeutralElementType = [$uint_type; WORD_COUNT];

            fn neutral_element(&self) -> Self::NeutralElementType {
                [0; WORD_COUNT]
            }

            fn default_size_neutral_element() -> Self::NeutralElementType {
                [0; WORD_COUNT]
            }

            fn neutral_element_of_size(_size: usize) -> Self::NeutralElementType {
                [0; WORD_COUNT]
            }
        }

        impl<const WORD_COUNT: usize> BitwiseNeutralElement for [$uint_type; WORD_COUNT] {}

        impl<const WORD_COUNT: usize> NeutralElement for &[$uint_type; WORD_COUNT] {
            type NeutralElementType = [$uint_type; WORD_COUNT];

            fn neutral_element(&self) -> Self::NeutralElementType {
                [0; WORD_COUNT]
            }

            fn default_size_neutral_element() -> Self::NeutralElementType {
                [0; WORD_COUNT]
            }

            fn neutral_element_of_size(_size: usize) -> Self::NeutralElementType {
                [0; WORD_COUNT]
            }
        }

        impl<const WORD_COUNT: usize> BitwiseNeutralElement for &[$uint_type; WORD_COUNT] {}

        impl<const WORD_COUNT: usize> NeutralElement for &mut [$uint_type; WORD_COUNT] {
            type NeutralElementType = [$uint_type; WORD_COUNT];

            fn neutral_element(&self) -> Self::NeutralElementType {
                [0; WORD_COUNT]
            }

            fn default_size_neutral_element() -> Self::NeutralElementType {
                [0; WORD_COUNT]
            }

            fn neutral_element_of_size(_size: usize) -> Self::NeutralElementType {
                [0; WORD_COUNT]
            }
        }

        impl<const WORD_COUNT: usize> BitwiseNeutralElement for &mut [$uint_type; WORD_COUNT] {}
    };
}

implement_neutral_element_for_uint_array!(u16);
implement_neutral_element_for_uint_array!(u32);
implement_neutral_element_for_uint_array!(u64);
implement_neutral_element_for_uint_array!(u128);

impl NeutralElement for Vec<bool> {
    type NeutralElementType = Vec<bool>;

    fn neutral_element(&self) -> Self::NeutralElementType {
        vec![false; self.len()]
    }

    fn default_size_neutral_element() -> Self::NeutralElementType {
        vec![]
    }

    fn neutral_element_of_size(size: usize) -> Self::NeutralElementType {
        vec![false; size]
    }
}
impl BitwiseNeutralElement for Vec<bool> {}

// Implementations for arrays of unsigned integers
macro_rules! implement_neutral_element_for_uint {
    ($uint_type:ty) => {
        impl NeutralElement for $uint_type {
            type NeutralElementType = $uint_type;

            fn neutral_element(&self) -> Self::NeutralElementType {
                0
            }

            fn default_size_neutral_element() -> Self::NeutralElementType {
                0
            }

            fn neutral_element_of_size(_size: usize) -> Self::NeutralElementType {
                0
            }
        }

        impl BitwiseNeutralElement for $uint_type {}

        // impl NeutralElement for &$uint_type {
        //     type NeutralElementType = $uint_type;

        //     fn neutral_element(&self) -> Self::NeutralElementType {
        //         0
        //     }

        //     fn default_size_neutral_element() -> Self::NeutralElementType {
        //         0
        //     }

        //     fn neutral_element_of_size(_size: usize) -> Self::NeutralElementType {
        //         0
        //     }
        // }

        // impl BitwiseNeutralElement for &$uint_type {}

        // impl NeutralElement for &mut $uint_type {
        //     type NeutralElementType = $uint_type;

        //     fn neutral_element(&self) -> Self::NeutralElementType {
        //         0
        //     }

        //     fn default_size_neutral_element() -> Self::NeutralElementType {
        //         0
        //     }

        //     fn neutral_element_of_size(_size: usize) -> Self::NeutralElementType {
        //         0
        //     }
        // }

        // impl BitwiseNeutralElement for &mut $uint_type {}
    };
}

implement_neutral_element_for_uint!(u16);
implement_neutral_element_for_uint!(u32);
implement_neutral_element_for_uint!(u64);
implement_neutral_element_for_uint!(u128);
