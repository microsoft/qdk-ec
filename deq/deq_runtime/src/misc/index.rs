pub const WILDCARD: u64 = 0;

#[derive(Debug, Clone)]
pub struct ErrorIndex {
    pub eid: u64,
    pub error_index: u64,
}

// let the coordinator keep the internal loaded library
pub const KEEP_FLAG_LIBRARY: u64 = 1 << 0;
// let the coordinator keep the loaded decoders; note that since decoders usually
// depend on the library, keeping the decoders while not keeping the library may
// lead to inconsistent result
pub const KEEP_FLAG_DECODER: u64 = 1 << 1;
// keep everything possible
pub const KEEP_FLAG_EVERYTHING: u64 = !0;
