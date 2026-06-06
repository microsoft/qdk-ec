//! Mock decoder for testing
//!
//! Records all decoder calls for verification, allowing inspection of
//! what coordinators send to the decoder at runtime.

use crate::decoder::blackbox_decoder::{self, black_box_decoder_server};
use crate::util::BitVector;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use std::sync::Arc;
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::RwLock;
#[cfg(feature = "cli")]
use tonic::transport::server::Router;
use tonic::{Request, Response, Status};

/// Configuration for the mock decoder when used from the CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct MockDecoderConfig {
    /// Simulated decode latency in milliseconds (applied per decode call).
    #[serde(default)]
    pub decode_delay_ms: u64,
}

/// A mock decoder that records all operations for testing.
pub struct MockDecoder {
    pub state: RwLock<MockDecoderState>,
    /// Optional delay to simulate decoder latency (applied per decode call).
    pub decode_delay: std::sync::Mutex<Option<std::time::Duration>>,
}

#[derive(Default)]
pub struct MockDecoderState {
    /// All decode calls with hypergraph and syndrome
    pub decode_calls: Vec<DecodeProblem>,
    /// All loaded hypergraphs by hid
    pub loaded_hypergraphs: HashMap<u64, blackbox_decoder::DecodingHypergraph>,
    /// All decode_loaded calls
    pub decode_loaded_calls: Vec<LoadedDecodeProblem>,
    /// Next hid to assign
    pub next_hid: u64,
    /// Number of reset calls
    pub reset_count: usize,
    /// Custom response provider: if set, returns this subgraph for decode/decode_loaded
    /// Key is the syndrome data, value is the subgraph to return
    pub custom_responses: HashMap<Vec<u8>, Vec<u64>>,
}

/// Captured decode problem
#[derive(Clone, Debug)]
pub struct DecodeProblem {
    pub hypergraph: blackbox_decoder::DecodingHypergraph,
    pub syndrome: BitVector,
}

/// Captured loaded decode problem
#[derive(Clone, Debug)]
pub struct LoadedDecodeProblem {
    pub hid: u64,
    pub syndrome: BitVector,
}

impl MockDecoder {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(MockDecoderState {
                next_hid: 1,
                ..Default::default()
            }),
            decode_delay: std::sync::Mutex::new(None),
        }
    }

    /// Create a MockDecoder from a JSON config value (for CLI use).
    pub fn from_config(config: serde_json::Value) -> Self {
        let config: MockDecoderConfig = serde_json::from_value(config).unwrap();
        let delay = if config.decode_delay_ms > 0 {
            Some(std::time::Duration::from_millis(config.decode_delay_ms))
        } else {
            None
        };
        Self {
            state: RwLock::new(MockDecoderState {
                next_hid: 1,
                ..Default::default()
            }),
            decode_delay: std::sync::Mutex::new(delay),
        }
    }

    #[cfg(feature = "cli")]
    pub fn add_service(self: &Arc<Self>, router: Router) -> Router {
        let service =
            black_box_decoder_server::BlackBoxDecoderServer::from_arc(self.clone()).max_decoding_message_size(usize::MAX);
        router.add_service(service)
    }

    /// Clear all recorded state
    pub async fn clear(&self) {
        let mut state = self.state.write().await;
        state.decode_calls.clear();
        state.loaded_hypergraphs.clear();
        state.decode_loaded_calls.clear();
        state.next_hid = 1;
        state.reset_count = 0;
        state.custom_responses.clear();
    }

    /// Set a custom response for a specific syndrome
    pub async fn set_response(&self, syndrome_data: Vec<u8>, subgraph: Vec<u64>) {
        let mut state = self.state.write().await;
        state.custom_responses.insert(syndrome_data, subgraph);
    }

    /// Set the decode delay for simulating decoder latency.
    pub fn set_decode_delay(&self, delay: std::time::Duration) {
        *self.decode_delay.lock().unwrap() = Some(delay);
    }

    /// Get the subgraph response for a syndrome, or empty if not set
    fn get_response(state: &MockDecoderState, syndrome: &BitVector) -> Vec<u64> {
        state.custom_responses.get(&syndrome.data).cloned().unwrap_or_default()
    }

    /// Apply decode delay if configured.
    async fn apply_delay(&self) {
        let delay = *self.decode_delay.lock().unwrap();
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }
    }
}

impl Default for MockDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MockDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockDecoder").finish()
    }
}

#[tonic::async_trait]
impl black_box_decoder_server::BlackBoxDecoder for MockDecoder {
    async fn decode(
        &self,
        request: Request<blackbox_decoder::DecodingProblem>,
    ) -> Result<Response<blackbox_decoder::ParityFactor>, Status> {
        let problem = request.into_inner();
        let hypergraph = problem
            .hypergraph
            .ok_or_else(|| Status::invalid_argument("missing hypergraph"))?;
        let syndrome = problem.syndrome.ok_or_else(|| Status::invalid_argument("missing syndrome"))?;

        let mut state = self.state.write().await;
        state.decode_calls.push(DecodeProblem {
            hypergraph: hypergraph.clone(),
            syndrome: syndrome.clone(),
        });

        let subgraph = Self::get_response(&state, &syndrome);
        drop(state);
        self.apply_delay().await;
        Ok(Response::new(blackbox_decoder::ParityFactor { subgraph }))
    }

    async fn load_hypergraph(
        &self,
        request: Request<blackbox_decoder::DecodingHypergraph>,
    ) -> Result<Response<blackbox_decoder::LoadHypergraphResponse>, Status> {
        let hypergraph = request.into_inner();

        let mut state = self.state.write().await;
        let hid = state.next_hid;
        state.next_hid += 1;
        state.loaded_hypergraphs.insert(hid, hypergraph);

        Ok(Response::new(blackbox_decoder::LoadHypergraphResponse { hid }))
    }

    async fn decode_loaded(
        &self,
        request: Request<blackbox_decoder::LoadedDecodingProblem>,
    ) -> Result<Response<blackbox_decoder::ParityFactor>, Status> {
        let problem = request.into_inner();
        let syndrome = problem.syndrome.ok_or_else(|| Status::invalid_argument("missing syndrome"))?;

        let mut state = self.state.write().await;
        if !state.loaded_hypergraphs.contains_key(&problem.hid) {
            return Err(Status::not_found(format!("hid={}", problem.hid)));
        }

        state.decode_loaded_calls.push(LoadedDecodeProblem {
            hid: problem.hid,
            syndrome: syndrome.clone(),
        });

        let subgraph = Self::get_response(&state, &syndrome);
        drop(state);
        self.apply_delay().await;
        Ok(Response::new(blackbox_decoder::ParityFactor { subgraph }))
    }

    async fn reset(&self, request: Request<blackbox_decoder::ResetRequest>) -> Result<Response<()>, Status> {
        let flags = request.into_inner();
        let mut state = self.state.write().await;
        state.reset_count += 1;
        if flags.reset_hypergraphs {
            state.loaded_hypergraphs.clear();
            state.next_hid = 1;
        }
        Ok(().into())
    }
}
