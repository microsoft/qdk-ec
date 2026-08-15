#[cfg(feature = "cli")]
use crate::misc::util::help_message;
#[cfg(feature = "cli")]
use clap::ValueEnum;
use serde::Serialize;
use std::sync::Arc;
#[cfg(feature = "cli")]
use tonic::transport::server::Router;
use tonic::{Request, Status};

use thread_pooling::DecoderFeatures;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Debug)]
#[cfg_attr(feature = "cli", derive(ValueEnum))]
pub enum DecoderType {
    /// a naive decoder that returns no errors
    BlackBoxNaive,
    /// using the public `relay-bp` crate as a blackbox (default f64)
    BlackBoxRelayBP,
    BlackBoxRelayBpF32,
    /// using a Python-defined decoder as a blackbox
    #[cfg(feature = "python")]
    BlackBoxPython,
    /// using Google's Tesseract beam-search decoder as a blackbox
    #[cfg(feature = "tesseract")]
    BlackBoxTesseract,
    /// loading a decoder from a binary-only shared library at runtime via the C ABI
    #[cfg(feature = "dylib")]
    BlackBoxDynLib,
    /// a mock decoder that returns no errors, with configurable latency
    Mock,
}

impl crate::controller::ParseByName for DecoderType {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "black-box-naive" => Some(Self::BlackBoxNaive),
            "black-box-relay-bp" => Some(Self::BlackBoxRelayBP),
            "black-box-relay-bp-f32" => Some(Self::BlackBoxRelayBpF32),
            #[cfg(feature = "python")]
            "black-box-python" => Some(Self::BlackBoxPython),
            #[cfg(feature = "tesseract")]
            "black-box-tesseract" => Some(Self::BlackBoxTesseract),
            #[cfg(feature = "dylib")]
            "black-box-dyn-lib" => Some(Self::BlackBoxDynLib),
            "mock" => Some(Self::Mock),
            _ => None,
        }
    }

    fn variant_names() -> Vec<&'static str> {
        #[allow(unused_mut)]
        let mut names = vec!["black-box-naive", "black-box-relay-bp", "black-box-relay-bp-f32"];
        #[cfg(feature = "python")]
        names.push("black-box-python");
        #[cfg(feature = "tesseract")]
        names.push("black-box-tesseract");
        #[cfg(feature = "dylib")]
        names.push("black-box-dyn-lib");
        names.push("mock");
        names
    }
}

pub mod blackbox_decoder {
    include!("proto/deq.decoder.blackbox_decoder.rs");
}

pub mod blackbox_util;
pub mod mock_decoder;
pub mod test_harness;
pub mod test_problems;
pub mod thread_pooling;

pub mod naive_decoder;
pub use mock_decoder::MockDecoder;
pub use naive_decoder::NaiveDecoder;

pub mod relay_bp_decoder;
pub use relay_bp_decoder::RelayBPDecoder;

#[cfg(feature = "dylib")]
pub mod dyn_lib_decoder;
#[cfg(feature = "dylib")]
pub use dyn_lib_decoder::DynLibDecoder;

#[cfg(feature = "python")]
pub mod python_decoder;
#[cfg(feature = "python")]
pub use python_decoder::PythonDecoder;

#[cfg(feature = "tesseract")]
pub mod tesseract_decoder;
#[cfg(feature = "tesseract")]
mod tesseract_ffi;
#[cfg(feature = "tesseract")]
pub use tesseract_decoder::TesseractDecoder;

impl DecoderType {
    pub fn create(&self, config: serde_json::Value) -> DynDecoder {
        match self {
            Self::BlackBoxNaive => DynDecoder::BlackBoxNaive(Arc::new(NaiveDecoder::new(config))),
            Self::BlackBoxRelayBP => DynDecoder::BlackBoxRelayBP(Arc::new(RelayBPDecoder::new(config))),
            Self::BlackBoxRelayBpF32 => DynDecoder::BlackBoxRelayBpF32(Arc::new(RelayBPDecoder::<f32>::new(config))),
            #[cfg(feature = "python")]
            Self::BlackBoxPython => DynDecoder::BlackBoxPython(Arc::new(PythonDecoder::new(config))),
            #[cfg(feature = "tesseract")]
            Self::BlackBoxTesseract => DynDecoder::BlackBoxTesseract(Arc::new(TesseractDecoder::new(config))),
            #[cfg(feature = "dylib")]
            Self::BlackBoxDynLib => DynDecoder::BlackBoxDynLib(Arc::new(DynLibDecoder::new(config))),
            Self::Mock => DynDecoder::Mock(Arc::new(MockDecoder::from_config(config))),
        }
    }

    #[cfg(feature = "cli")]
    pub fn config_help() -> String {
        help_message::<naive_decoder::NaiveDecoderConfig>("NaiveDecoderConfig:")
            + &*help_message::<relay_bp_decoder::RelayBPDecoderConfig>("RelayBPDecoderConfig:")
            + &*{
                #[cfg(feature = "python")]
                {
                    help_message::<python_decoder::PythonDecoderConfig>("PythonDecoderConfig:")
                }
                #[cfg(not(feature = "python"))]
                {
                    String::new()
                }
            }
            + &*{
                #[cfg(feature = "tesseract")]
                {
                    help_message::<tesseract_decoder::TesseractDecoderConfig>("TesseractDecoderConfig:")
                }
                #[cfg(not(feature = "tesseract"))]
                {
                    String::new()
                }
            }
            + &*{
                #[cfg(feature = "dylib")]
                {
                    help_message::<dyn_lib_decoder::DynLibDecoderConfig>("DynLibDecoderConfig:")
                }
                #[cfg(not(feature = "dylib"))]
                {
                    String::new()
                }
            }
            + &*help_message::<mock_decoder::MockDecoderConfig>("MockDecoderConfig:")
    }

    #[cfg(not(feature = "cli"))]
    pub fn config_help() -> String {
        String::new()
    }
}

#[derive(Clone)]
pub enum DynDecoder {
    BlackBoxNaive(Arc<NaiveDecoder>),
    BlackBoxRelayBP(Arc<RelayBPDecoder>),
    BlackBoxRelayBpF32(Arc<RelayBPDecoder<f32>>),
    #[cfg(feature = "python")]
    BlackBoxPython(Arc<PythonDecoder>),
    #[cfg(feature = "tesseract")]
    BlackBoxTesseract(Arc<TesseractDecoder>),
    #[cfg(feature = "dylib")]
    BlackBoxDynLib(Arc<DynLibDecoder>),
    Mock(Arc<MockDecoder>),
}

impl DynDecoder {
    #[cfg(feature = "cli")]
    pub fn add_service(&self, router: Router) -> Router {
        match self {
            DynDecoder::BlackBoxNaive(decoder) => NaiveDecoder::add_service(decoder, router),
            DynDecoder::BlackBoxRelayBP(decoder) => RelayBPDecoder::add_service(decoder, router),
            DynDecoder::BlackBoxRelayBpF32(decoder) => RelayBPDecoder::<f32>::add_service(decoder, router),
            #[cfg(feature = "python")]
            DynDecoder::BlackBoxPython(decoder) => PythonDecoder::add_service(decoder, router),
            #[cfg(feature = "tesseract")]
            DynDecoder::BlackBoxTesseract(decoder) => TesseractDecoder::add_service(decoder, router),
            #[cfg(feature = "dylib")]
            DynDecoder::BlackBoxDynLib(decoder) => DynLibDecoder::add_service(decoder, router),
            DynDecoder::Mock(decoder) => MockDecoder::add_service(decoder, router),
        }
    }

    fn inner(&self) -> &dyn blackbox_decoder::black_box_decoder_server::BlackBoxDecoder {
        match self {
            DynDecoder::BlackBoxNaive(decoder) => decoder.as_ref(),
            DynDecoder::BlackBoxRelayBP(decoder) => decoder.as_ref(),
            DynDecoder::BlackBoxRelayBpF32(decoder) => decoder.as_ref(),
            #[cfg(feature = "python")]
            DynDecoder::BlackBoxPython(decoder) => decoder.as_ref(),
            #[cfg(feature = "tesseract")]
            DynDecoder::BlackBoxTesseract(decoder) => decoder.as_ref(),
            #[cfg(feature = "dylib")]
            DynDecoder::BlackBoxDynLib(decoder) => decoder.as_ref(),
            DynDecoder::Mock(decoder) => decoder.as_ref(),
        }
    }

    #[must_use]
    pub fn features(&self) -> DecoderFeatures {
        match self {
            DynDecoder::BlackBoxNaive(decoder) => decoder.supported_features(),
            DynDecoder::BlackBoxRelayBP(decoder) => decoder.features(),
            DynDecoder::BlackBoxRelayBpF32(decoder) => decoder.features(),
            #[cfg(feature = "python")]
            DynDecoder::BlackBoxPython(decoder) => decoder.features(),
            #[cfg(feature = "tesseract")]
            DynDecoder::BlackBoxTesseract(decoder) => decoder.features(),
            #[cfg(feature = "dylib")]
            DynDecoder::BlackBoxDynLib(decoder) => decoder.features(),
            DynDecoder::Mock(decoder) => decoder.supported_features(),
        }
    }

    fn require_features(&self, required: DecoderFeatures) -> Result<(), Status> {
        let unsupported = required.difference(self.features());
        if unsupported.is_empty() {
            Ok(())
        } else {
            Err(Status::failed_precondition(format!(
                "unsupported decoder features: {unsupported}"
            )))
        }
    }

    pub async fn decode(
        &self,
        problem: blackbox_decoder::DecodingProblem,
    ) -> Result<blackbox_decoder::ParityFactor, Status> {
        if problem.loss.is_some() {
            self.require_features(DecoderFeatures::LOSS)?;
        }
        self.inner().decode(Request::new(problem)).await.map(|v| v.into_inner())
    }

    pub async fn load_hypergraph(
        &self,
        hypergraph: blackbox_decoder::DecodingHypergraph,
    ) -> Result<blackbox_decoder::LoadHypergraphResponse, Status> {
        self.inner()
            .load_hypergraph(Request::new(hypergraph))
            .await
            .map(|v| v.into_inner())
    }

    pub async fn decode_loaded(
        &self,
        problem: blackbox_decoder::LoadedDecodingProblem,
    ) -> Result<blackbox_decoder::ParityFactor, Status> {
        let mut required = DecoderFeatures::empty();
        if !problem.reweights.is_empty() {
            required = required | DecoderFeatures::REWEIGHTS;
        }
        if problem.loss.is_some() {
            required = required | DecoderFeatures::LOSS;
        }
        self.require_features(required)?;
        self.inner()
            .decode_loaded(Request::new(problem))
            .await
            .map(|v| v.into_inner())
    }

    pub async fn reset(&self, flags: blackbox_decoder::ResetRequest) -> Result<(), Status> {
        self.inner().reset(Request::new(flags)).await.map(|_| ())
    }
}
