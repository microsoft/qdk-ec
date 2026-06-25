//! cxx FFI bridge for the Tesseract C++ decoder core.

use cxx::UniquePtr;

#[cxx::bridge(namespace = "tesseract_bridge")]
mod ffi {
    unsafe extern "C++" {
        include!("tesseract_bridge.h");

        type TesseractDecoderHandle;

        #[allow(clippy::too_many_arguments)]
        fn new_tesseract_decoder(
            num_detectors: u64,
            edge_vertices: &[u64],
            edge_offsets: &[u64],
            edge_probabilities: &[f64],
            det_beam: i32,
            beam_climbing: bool,
            no_revisit_dets: bool,
            merge_errors: bool,
            pqlimit: u64,
            det_penalty: f64,
        ) -> Result<UniquePtr<TesseractDecoderHandle>>;

        fn decode_to_errors(handle: Pin<&mut TesseractDecoderHandle>, detections: &[u64]) -> Vec<u64>;
    }
}

pub struct TesseractCxxDecoder {
    inner: UniquePtr<ffi::TesseractDecoderHandle>,
}

// SAFETY: Each instance is only accessed by one thread at a time via DecoderInstance.
unsafe impl Send for TesseractCxxDecoder {}

pub struct TesseractCxxConfig {
    pub det_beam: i32,
    pub beam_climbing: bool,
    pub no_revisit_dets: bool,
    pub merge_errors: bool,
    pub pqlimit: u64,
    pub det_penalty: f64,
}

impl Default for TesseractCxxConfig {
    fn default() -> Self {
        Self {
            det_beam: 5,
            beam_climbing: false,
            no_revisit_dets: true,
            merge_errors: true,
            pqlimit: 200_000,
            det_penalty: 0.0,
        }
    }
}

impl TesseractCxxDecoder {
    pub fn new(
        num_detectors: u64,
        edge_vertices: &[u64],
        edge_offsets: &[u64],
        edge_probabilities: &[f64],
        config: &TesseractCxxConfig,
    ) -> Self {
        let inner = ffi::new_tesseract_decoder(
            num_detectors,
            edge_vertices,
            edge_offsets,
            edge_probabilities,
            config.det_beam,
            config.beam_climbing,
            config.no_revisit_dets,
            config.merge_errors,
            config.pqlimit,
            config.det_penalty,
        )
        .expect("failed to create TesseractDecoder");
        Self { inner }
    }

    pub fn decode(&mut self, detections: &[u64]) -> Vec<u64> {
        ffi::decode_to_errors(self.inner.pin_mut(), detections)
    }
}
