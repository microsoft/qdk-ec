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
    #[cfg_attr(feature = "cli", value(name = "naive", alias = "black-box-naive"))]
    BlackBoxNaive,
    /// using the native `fusion-blossom` MWPM decoder
    #[cfg(feature = "mwpm")]
    #[cfg_attr(feature = "cli", value(name = "mwpm", alias = "black-box-mwpm"))]
    BlackBoxMwpm,
    /// using the native hypergraph MWPF decoder
    #[cfg(feature = "mwpf")]
    #[cfg_attr(feature = "cli", value(name = "mwpf", alias = "black-box-mwpf"))]
    BlackBoxMwpf,
    /// using Fusion Blossom in union-find mode
    #[cfg(feature = "mwpm")]
    #[cfg_attr(feature = "cli", value(name = "uf", alias = "black-box-uf"))]
    BlackBoxUf,
    /// using MWPF in hypergraph union-find mode
    #[cfg(feature = "mwpf")]
    #[cfg_attr(feature = "cli", value(name = "huf", alias = "black-box-huf"))]
    BlackBoxHuf,
    /// using the public `relay-bp` crate as a blackbox (default f64)
    #[cfg_attr(feature = "cli", value(name = "relay-bp", alias = "black-box-relay-bp"))]
    BlackBoxRelayBP,
    #[cfg_attr(feature = "cli", value(name = "relay-bp-f32", alias = "black-box-relay-bp-f32"))]
    BlackBoxRelayBpF32,
    /// using a Python-defined decoder as a blackbox
    #[cfg(feature = "python")]
    #[cfg_attr(feature = "cli", value(name = "python", alias = "black-box-python"))]
    BlackBoxPython,
    /// using Google's Tesseract beam-search decoder as a blackbox
    #[cfg(feature = "tesseract")]
    #[cfg_attr(feature = "cli", value(name = "tesseract", alias = "black-box-tesseract"))]
    BlackBoxTesseract,
    /// loading a decoder from a binary-only shared library at runtime via the C ABI
    #[cfg(feature = "dylib")]
    #[cfg_attr(feature = "cli", value(name = "dyn-lib", alias = "black-box-dyn-lib"))]
    BlackBoxDynLib,
    /// a mock decoder that returns no errors, with configurable latency
    Mock,
}

impl crate::controller::ParseByName for DecoderType {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "naive" | "black-box-naive" => Some(Self::BlackBoxNaive),
            #[cfg(feature = "mwpm")]
            "mwpm" | "black-box-mwpm" => Some(Self::BlackBoxMwpm),
            #[cfg(feature = "mwpf")]
            "mwpf" | "black-box-mwpf" => Some(Self::BlackBoxMwpf),
            #[cfg(feature = "mwpm")]
            "uf" | "black-box-uf" => Some(Self::BlackBoxUf),
            #[cfg(feature = "mwpf")]
            "huf" | "black-box-huf" => Some(Self::BlackBoxHuf),
            "relay-bp" | "black-box-relay-bp" => Some(Self::BlackBoxRelayBP),
            "relay-bp-f32" | "black-box-relay-bp-f32" => Some(Self::BlackBoxRelayBpF32),
            #[cfg(feature = "python")]
            "python" | "black-box-python" => Some(Self::BlackBoxPython),
            #[cfg(feature = "tesseract")]
            "tesseract" | "black-box-tesseract" => Some(Self::BlackBoxTesseract),
            #[cfg(feature = "dylib")]
            "dyn-lib" | "black-box-dyn-lib" => Some(Self::BlackBoxDynLib),
            "mock" => Some(Self::Mock),
            _ => None,
        }
    }

    fn variant_names() -> Vec<&'static str> {
        #[allow(unused_mut)]
        let mut names = vec![
            "naive",
            "relay-bp",
            "relay-bp-f32",
            "black-box-naive",
            "black-box-relay-bp",
            "black-box-relay-bp-f32",
        ];
        #[cfg(feature = "mwpm")]
        names.extend(["mwpm", "uf", "black-box-mwpm", "black-box-uf"]);
        #[cfg(feature = "mwpf")]
        names.extend(["mwpf", "huf", "black-box-mwpf", "black-box-huf"]);
        #[cfg(feature = "python")]
        names.extend(["python", "black-box-python"]);
        #[cfg(feature = "tesseract")]
        names.extend(["tesseract", "black-box-tesseract"]);
        #[cfg(feature = "dylib")]
        names.extend(["dyn-lib", "black-box-dyn-lib"]);
        names.push("mock");
        names
    }
}

pub mod blackbox_decoder {
    include!("proto/deq.decoder.blackbox_decoder.rs");
}

pub mod blackbox_util;
pub mod decoder_features;
pub use decoder_features::DecoderFeatures;
pub mod mock_decoder;
pub mod test_harness;
pub mod test_problems;
pub mod thread_pooling;

pub mod naive_decoder;
pub use mock_decoder::MockDecoder;
pub use naive_decoder::NaiveDecoder;

#[cfg(feature = "mwpm")]
pub mod mwpm_decoder;
#[cfg(feature = "mwpm")]
pub use mwpm_decoder::MwpmDecoder;

#[cfg(feature = "mwpf")]
pub mod mwpf_decoder;
#[cfg(feature = "mwpf")]
pub use mwpf_decoder::MwpfDecoder;

#[cfg(feature = "mwpm")]
pub mod uf_decoder;
#[cfg(feature = "mwpm")]
pub use uf_decoder::UfDecoder;

#[cfg(feature = "mwpf")]
pub mod huf_decoder;
#[cfg(feature = "mwpf")]
pub use huf_decoder::HufDecoder;

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
        self.create_with_thread_pool(config, None)
    }

    pub(crate) fn create_with_thread_pool(
        self,
        config: serde_json::Value,
        thread_pool: Option<Arc<rayon::ThreadPool>>,
    ) -> DynDecoder {
        match self {
            Self::BlackBoxNaive => DynDecoder::BlackBoxNaive(Arc::new(NaiveDecoder::new(config))),
            #[cfg(feature = "mwpm")]
            Self::BlackBoxMwpm => DynDecoder::BlackBoxMwpm(Arc::new(MwpmDecoder::with_thread_pool(config, thread_pool))),
            #[cfg(feature = "mwpf")]
            Self::BlackBoxMwpf => DynDecoder::BlackBoxMwpf(Arc::new(MwpfDecoder::with_thread_pool(config, thread_pool))),
            #[cfg(feature = "mwpm")]
            Self::BlackBoxUf => DynDecoder::BlackBoxUf(Arc::new(UfDecoder::with_thread_pool(config, thread_pool))),
            #[cfg(feature = "mwpf")]
            Self::BlackBoxHuf => DynDecoder::BlackBoxHuf(Arc::new(HufDecoder::with_thread_pool(config, thread_pool))),
            Self::BlackBoxRelayBP => {
                DynDecoder::BlackBoxRelayBP(Arc::new(RelayBPDecoder::with_thread_pool(config, thread_pool)))
            }
            Self::BlackBoxRelayBpF32 => {
                DynDecoder::BlackBoxRelayBpF32(Arc::new(RelayBPDecoder::<f32>::with_thread_pool(config, thread_pool)))
            }
            #[cfg(feature = "python")]
            Self::BlackBoxPython => {
                DynDecoder::BlackBoxPython(Arc::new(PythonDecoder::with_thread_pool(config, thread_pool)))
            }
            #[cfg(feature = "tesseract")]
            Self::BlackBoxTesseract => {
                DynDecoder::BlackBoxTesseract(Arc::new(TesseractDecoder::with_thread_pool(config, thread_pool)))
            }
            #[cfg(feature = "dylib")]
            Self::BlackBoxDynLib => {
                DynDecoder::BlackBoxDynLib(Arc::new(DynLibDecoder::with_thread_pool(config, thread_pool)))
            }
            Self::Mock => DynDecoder::Mock(Arc::new(MockDecoder::from_config(config))),
        }
    }

    #[cfg(feature = "cli")]
    pub fn config_help() -> String {
        help_message::<naive_decoder::NaiveDecoderConfig>("NaiveDecoderConfig:")
            + &*{
                #[cfg(feature = "mwpm")]
                {
                    help_message::<mwpm_decoder::MwpmDecoderConfig>("MwpmDecoderConfig:")
                        + &*help_message::<uf_decoder::UfDecoderConfig>("UfDecoderConfig:")
                }
                #[cfg(not(feature = "mwpm"))]
                {
                    String::new()
                }
            }
            + &*{
                #[cfg(feature = "mwpf")]
                {
                    help_message::<mwpf_decoder::MwpfDecoderConfig>("MwpfDecoderConfig:")
                        + &*help_message::<huf_decoder::HufDecoderConfig>("HufDecoderConfig:")
                }
                #[cfg(not(feature = "mwpf"))]
                {
                    String::new()
                }
            }
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
    #[cfg(feature = "mwpm")]
    BlackBoxMwpm(Arc<MwpmDecoder>),
    #[cfg(feature = "mwpf")]
    BlackBoxMwpf(Arc<MwpfDecoder>),
    #[cfg(feature = "mwpm")]
    BlackBoxUf(Arc<UfDecoder>),
    #[cfg(feature = "mwpf")]
    BlackBoxHuf(Arc<HufDecoder>),
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

// Keep the backend list in one place without erasing pooled types or boxing their futures.
macro_rules! dispatch_decoder {
    ($value:expr, |$decoder:ident| $body:expr) => {
        dispatch_decoder!($value, |$decoder| $body, $decoder => $body)
    };
    ($value:expr, |$decoder:ident| $pooled:expr, $unpooled:pat => $fallback:expr) => {
        match $value {
            DynDecoder::BlackBoxNaive($unpooled) => $fallback,
            DynDecoder::Mock($unpooled) => $fallback,
            #[cfg(feature = "mwpm")]
            DynDecoder::BlackBoxMwpm($decoder) => $pooled,
            #[cfg(feature = "mwpf")]
            DynDecoder::BlackBoxMwpf($decoder) => $pooled,
            #[cfg(feature = "mwpm")]
            DynDecoder::BlackBoxUf($decoder) => $pooled,
            #[cfg(feature = "mwpf")]
            DynDecoder::BlackBoxHuf($decoder) => $pooled,
            DynDecoder::BlackBoxRelayBP($decoder) => $pooled,
            DynDecoder::BlackBoxRelayBpF32($decoder) => $pooled,
            #[cfg(feature = "python")]
            DynDecoder::BlackBoxPython($decoder) => $pooled,
            #[cfg(feature = "tesseract")]
            DynDecoder::BlackBoxTesseract($decoder) => $pooled,
            #[cfg(feature = "dylib")]
            DynDecoder::BlackBoxDynLib($decoder) => $pooled,
        }
    };
}

impl DynDecoder {
    pub(crate) async fn unload_hypergraphs(&self, hids: &[u64]) {
        dispatch_decoder!(self, |decoder| decoder.unload_hypergraphs(hids).await, _ => {
            // The other unpooled backend, NaiveDecoder, retains no graph state.
            if let Self::Mock(decoder) = self {
                decoder.unload_hypergraphs(hids).await;
            }
        });
    }

    pub(crate) fn thread_pool(&self) -> Option<&Arc<rayon::ThreadPool>> {
        dispatch_decoder!(self, |decoder| Some(&decoder.thread_pool), _ => None)
    }

    #[cfg(feature = "cli")]
    pub fn add_service(&self, router: Router) -> Router {
        dispatch_decoder!(self, |decoder| decoder.add_service(router))
    }

    fn inner(&self) -> &dyn blackbox_decoder::black_box_decoder_server::BlackBoxDecoder {
        dispatch_decoder!(self, |decoder| decoder.as_ref())
    }

    #[must_use]
    pub fn features(&self) -> DecoderFeatures {
        dispatch_decoder!(self, |decoder| decoder.features(), decoder => decoder.supported_features())
    }

    pub(crate) fn edge_filter(&self) -> Option<thread_pooling::EdgeFilter> {
        dispatch_decoder!(self, |decoder| decoder.edge_filter(), _ => None)
    }

    fn require_features(&self, required: DecoderFeatures) -> Result<(), Status> {
        required
            .require_supported_by(self.features())
            .map_err(|unsupported| Status::failed_precondition(format!("unsupported decoder features: {unsupported}")))
    }

    pub async fn decode(
        &self,
        problem: blackbox_decoder::DecodingProblem,
    ) -> Result<blackbox_decoder::ParityFactor, Status> {
        self.require_features(DecoderFeatures::required(
            problem.decoder_seed.is_some(),
            false,
            problem.loss.is_some(),
        ))?;
        self.inner().decode(Request::new(problem)).await.map(|v| v.into_inner())
    }

    pub(crate) async fn decode_shared(
        &self,
        hypergraph: Arc<blackbox_decoder::DecodingHypergraph>,
        syndrome: crate::util::BitVector,
        loss: Option<blackbox_decoder::LossInfo>,
        decoder_seed: Option<u64>,
    ) -> Result<blackbox_decoder::ParityFactor, Status> {
        self.require_features(DecoderFeatures::required(decoder_seed.is_some(), false, loss.is_some()))?;
        dispatch_decoder!(self, |decoder| decoder.decode_shared(hypergraph, syndrome, loss, decoder_seed).await,
            _ => self.decode(blackbox_decoder::DecodingProblem {
                hypergraph: Some(hypergraph.as_ref().clone()),
                syndrome: Some(syndrome),
                loss,
                decoder_seed,
            }).await
        )
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
        let required = DecoderFeatures::required(
            problem.decoder_seed.is_some(),
            !problem.reweights.is_empty(),
            problem.loss.is_some(),
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::ParseByName;

    #[tokio::test]
    async fn common_dispatch_preserves_pools_capabilities_and_shared_decoding() {
        let backends = [
            (DecoderType::BlackBoxNaive, false, false),
            (DecoderType::Mock, false, false),
            (DecoderType::BlackBoxRelayBP, true, false),
            (DecoderType::BlackBoxRelayBpF32, true, false),
            #[cfg(feature = "mwpm")]
            (DecoderType::BlackBoxMwpm, true, true),
            #[cfg(feature = "mwpm")]
            (DecoderType::BlackBoxUf, true, true),
            #[cfg(feature = "mwpf")]
            (DecoderType::BlackBoxMwpf, true, false),
            #[cfg(feature = "mwpf")]
            (DecoderType::BlackBoxHuf, true, false),
            #[cfg(feature = "tesseract")]
            (DecoderType::BlackBoxTesseract, true, false),
        ];
        let graph = Arc::new(blackbox_decoder::DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![blackbox_decoder::Hyperedge {
                vertices: vec![0],
                probability: 0.1,
                observable_flips: None,
            }],
        });
        let syndrome = crate::util::BitVector {
            size: 1,
            data: vec![0x80],
        };
        for (kind, pooled, filters_edges) in backends {
            let decoder = kind.create(if pooled {
                serde_json::json!({"parallel": 1})
            } else {
                serde_json::json!({})
            });
            assert_eq!(decoder.thread_pool().is_some(), pooled, "{kind:?}");
            assert_eq!(decoder.edge_filter().is_some(), filters_edges, "{kind:?}");
            let config = serde_json::json!({"decoder_preprocessing": true});
            let coordinator = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::coordinator::WindowCoordinator::new(config, decoder.clone())
            }));
            assert_eq!(coordinator.is_ok(), filters_edges, "{kind:?}");
            if let Some(filter) = decoder.edge_filter() {
                for (vertices, probability, retained) in [
                    (vec![], 0.1, false),
                    (vec![0], 0.1, true),
                    (vec![0, 1], 0.5, true),
                    (vec![0, 1, 2], 0.1, false),
                    (vec![0], 0.0, false),
                ] {
                    let edge = blackbox_decoder::Hyperedge {
                        vertices,
                        probability,
                        observable_flips: None,
                    };
                    assert_eq!(filter(&edge), retained, "{kind:?}");
                }
            }
            #[cfg(feature = "cli")]
            assert_eq!(
                decoder.features().to_proto(),
                decoder.inner().get_capabilities(Request::new(())).await.unwrap().into_inner(),
                "{kind:?}",
            );
            let expected = decoder
                .decode(blackbox_decoder::DecodingProblem {
                    hypergraph: Some(graph.as_ref().clone()),
                    syndrome: Some(syndrome.clone()),
                    loss: None,
                    decoder_seed: None,
                })
                .await
                .unwrap();
            let actual = decoder
                .decode_shared(Arc::clone(&graph), syndrome.clone(), None, None)
                .await
                .unwrap();
            assert_eq!(actual, expected, "{kind:?}");
        }
    }

    #[test]
    fn manual_parser_accepts_short_and_legacy_names() {
        let names = [
            ("naive", "black-box-naive", DecoderType::BlackBoxNaive),
            #[cfg(feature = "mwpm")]
            ("mwpm", "black-box-mwpm", DecoderType::BlackBoxMwpm),
            #[cfg(feature = "mwpf")]
            ("mwpf", "black-box-mwpf", DecoderType::BlackBoxMwpf),
            #[cfg(feature = "mwpm")]
            ("uf", "black-box-uf", DecoderType::BlackBoxUf),
            #[cfg(feature = "mwpf")]
            ("huf", "black-box-huf", DecoderType::BlackBoxHuf),
            ("relay-bp", "black-box-relay-bp", DecoderType::BlackBoxRelayBP),
            ("relay-bp-f32", "black-box-relay-bp-f32", DecoderType::BlackBoxRelayBpF32),
            #[cfg(feature = "python")]
            ("python", "black-box-python", DecoderType::BlackBoxPython),
            #[cfg(feature = "tesseract")]
            ("tesseract", "black-box-tesseract", DecoderType::BlackBoxTesseract),
            #[cfg(feature = "dylib")]
            ("dyn-lib", "black-box-dyn-lib", DecoderType::BlackBoxDynLib),
        ];
        for (short, legacy, expected) in names {
            assert_eq!(DecoderType::from_name(short), Some(expected));
            assert_eq!(DecoderType::from_name(legacy), Some(expected));
            assert!(DecoderType::variant_names().contains(&short));
            assert!(DecoderType::variant_names().contains(&legacy));
        }
    }
}
