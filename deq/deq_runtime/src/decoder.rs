#[cfg(feature = "cli")]
use crate::misc::util::help_message;
#[cfg(feature = "cli")]
use clap::ValueEnum;
use serde::Serialize;
use std::sync::Arc;
#[cfg(feature = "cli")]
use tonic::transport::server::Router;
use tonic::{Request, Status};

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
            + &*help_message::<mock_decoder::MockDecoderConfig>("MockDecoderConfig:")
    }

    #[cfg(not(feature = "cli"))]
    pub fn config_help() -> String {
        String::new()
    }
}

pub enum DynDecoder {
    None,
    BlackBoxNaive(Arc<NaiveDecoder>),
    BlackBoxRelayBP(Arc<RelayBPDecoder>),
    BlackBoxRelayBpF32(Arc<RelayBPDecoder<f32>>),
    #[cfg(feature = "python")]
    BlackBoxPython(Arc<PythonDecoder>),
    #[cfg(feature = "tesseract")]
    BlackBoxTesseract(Arc<TesseractDecoder>),
    Mock(Arc<MockDecoder>),
}

impl DynDecoder {
    #[cfg(feature = "cli")]
    pub fn add_service(&self, router: Router) -> Router {
        match self {
            DynDecoder::None => router,
            DynDecoder::BlackBoxNaive(decoder) => NaiveDecoder::add_service(decoder, router),
            DynDecoder::BlackBoxRelayBP(decoder) => RelayBPDecoder::add_service(decoder, router),
            DynDecoder::BlackBoxRelayBpF32(decoder) => RelayBPDecoder::<f32>::add_service(decoder, router),
            #[cfg(feature = "python")]
            DynDecoder::BlackBoxPython(decoder) => PythonDecoder::add_service(decoder, router),
            #[cfg(feature = "tesseract")]
            DynDecoder::BlackBoxTesseract(decoder) => TesseractDecoder::add_service(decoder, router),
            DynDecoder::Mock(decoder) => MockDecoder::add_service(decoder, router),
        }
    }

    pub fn as_black_box_decoder(&self) -> Option<DynBlackBoxDecoder> {
        match self {
            DynDecoder::BlackBoxNaive(v) => Some(DynBlackBoxDecoder::BlackBoxNaive(v.clone())),
            DynDecoder::BlackBoxRelayBP(v) => Some(DynBlackBoxDecoder::BlackBoxRelayBP(v.clone())),
            DynDecoder::BlackBoxRelayBpF32(v) => Some(DynBlackBoxDecoder::BlackBoxRelayBpF32(v.clone())),
            #[cfg(feature = "python")]
            DynDecoder::BlackBoxPython(v) => Some(DynBlackBoxDecoder::BlackBoxPython(v.clone())),
            #[cfg(feature = "tesseract")]
            DynDecoder::BlackBoxTesseract(v) => Some(DynBlackBoxDecoder::BlackBoxTesseract(v.clone())),
            DynDecoder::Mock(v) => Some(DynBlackBoxDecoder::MockDecoder(v.clone())),
            _ => None,
        }
    }

    #[cfg(feature = "cli")]
    pub async fn as_black_box_decoder_client(
        &self,
        endpoint: Option<&tonic::transport::Endpoint>,
    ) -> Option<BlackBoxDecoderClient> {
        match self.as_black_box_decoder() {
            Some(black_box_decoder) => Some(if let Some(endpoint) = endpoint {
                BlackBoxDecoderClient::from_endpoint(endpoint.clone()).await
            } else {
                BlackBoxDecoderClient::Local(black_box_decoder)
            }),
            None => None,
        }
    }

    #[cfg(not(feature = "cli"))]
    pub async fn as_black_box_decoder_client(&self) -> Option<BlackBoxDecoderClient> {
        self.as_black_box_decoder().map(BlackBoxDecoderClient::Local)
    }
}

#[derive(Clone)]
pub enum DynBlackBoxDecoder {
    BlackBoxNaive(Arc<NaiveDecoder>),
    BlackBoxRelayBP(Arc<RelayBPDecoder>),
    BlackBoxRelayBpF32(Arc<RelayBPDecoder<f32>>),
    #[cfg(feature = "python")]
    BlackBoxPython(Arc<PythonDecoder>),
    #[cfg(feature = "tesseract")]
    BlackBoxTesseract(Arc<TesseractDecoder>),
    MockDecoder(Arc<MockDecoder>),
}

impl DynBlackBoxDecoder {
    pub fn inner(&self) -> Arc<dyn blackbox_decoder::black_box_decoder_server::BlackBoxDecoder> {
        match self {
            DynBlackBoxDecoder::BlackBoxNaive(v) => v.clone(),
            DynBlackBoxDecoder::BlackBoxRelayBP(v) => v.clone(),
            DynBlackBoxDecoder::BlackBoxRelayBpF32(v) => v.clone(),
            #[cfg(feature = "python")]
            DynBlackBoxDecoder::BlackBoxPython(v) => v.clone(),
            #[cfg(feature = "tesseract")]
            DynBlackBoxDecoder::BlackBoxTesseract(v) => v.clone(),
            DynBlackBoxDecoder::MockDecoder(v) => v.clone(),
        }
    }
}

/// a client wrapper that can either be a remote gRPC client or a local reference
#[derive(Clone)]
pub enum BlackBoxDecoderClient {
    #[cfg(feature = "cli")]
    Remote(blackbox_decoder::black_box_decoder_client::BlackBoxDecoderClient<tonic::transport::Channel>),
    Local(DynBlackBoxDecoder),
}

impl BlackBoxDecoderClient {
    #[cfg(feature = "cli")]
    pub async fn from_endpoint(endpoint: tonic::transport::Endpoint) -> Self {
        Self::Remote(
            crate::decoder::blackbox_decoder::black_box_decoder_client::BlackBoxDecoderClient::connect(endpoint)
                .await
                .unwrap(),
        )
    }

    /// Create a client from a MockDecoder for testing
    pub fn from_mock(mock: Arc<MockDecoder>) -> Self {
        Self::Local(DynBlackBoxDecoder::MockDecoder(mock))
    }

    pub async fn decode(
        &mut self,
        problem: blackbox_decoder::DecodingProblem,
    ) -> Result<blackbox_decoder::ParityFactor, Status> {
        let request = Request::new(problem);
        (match self {
            #[cfg(feature = "cli")]
            BlackBoxDecoderClient::Remote(client) => client.decode(request).await,
            BlackBoxDecoderClient::Local(local) => local.inner().decode(request).await,
        })
        .map(|v| v.into_inner())
    }

    pub async fn load_hypergraph(
        &mut self,
        hypergraph: blackbox_decoder::DecodingHypergraph,
    ) -> Result<blackbox_decoder::LoadHypergraphResponse, Status> {
        let request = Request::new(hypergraph);
        (match self {
            #[cfg(feature = "cli")]
            BlackBoxDecoderClient::Remote(client) => client.load_hypergraph(request).await,
            BlackBoxDecoderClient::Local(local) => local.inner().load_hypergraph(request).await,
        })
        .map(|v| v.into_inner())
    }

    pub async fn decode_loaded(
        &mut self,
        problem: blackbox_decoder::LoadedDecodingProblem,
    ) -> Result<blackbox_decoder::ParityFactor, Status> {
        let request = Request::new(problem);
        (match self {
            #[cfg(feature = "cli")]
            BlackBoxDecoderClient::Remote(client) => client.decode_loaded(request).await,
            BlackBoxDecoderClient::Local(local) => local.inner().decode_loaded(request).await,
        })
        .map(|v| v.into_inner())
    }

    pub async fn reset(&mut self, flags: blackbox_decoder::ResetRequest) -> Result<(), Status> {
        let request = Request::new(flags);
        (match self {
            #[cfg(feature = "cli")]
            BlackBoxDecoderClient::Remote(client) => client.reset(request).await,
            BlackBoxDecoderClient::Local(local) => local.inner().reset(request).await,
        })
        .map(|_| ())
    }
}
