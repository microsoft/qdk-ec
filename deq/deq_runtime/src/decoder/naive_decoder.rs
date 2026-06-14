//! Naive decoder
//!
//! A decoder that does nothing but returning zero error;
//!

use crate::decoder::blackbox_decoder::{self, black_box_decoder_server};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cli")]
use std::sync::Arc;
#[cfg(feature = "cli")]
use structdoc::StructDoc;
#[cfg(feature = "cli")]
use tonic::transport::server::Router;
use tonic::{Request, Response, Status};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct NaiveDecoderConfig {}

pub struct NaiveDecoder {
    pub config: NaiveDecoderConfig,
}

impl NaiveDecoder {
    pub fn new(config: serde_json::Value) -> Self {
        let config: NaiveDecoderConfig = serde_json::from_value(config).unwrap();
        Self { config }
    }

    #[cfg(feature = "cli")]
    pub fn add_service(self: &Arc<Self>, router: Router) -> Router {
        let service =
            black_box_decoder_server::BlackBoxDecoderServer::from_arc(self.clone()).max_decoding_message_size(usize::MAX);
        router.add_service(service)
    }
}

#[tonic::async_trait]
impl black_box_decoder_server::BlackBoxDecoder for NaiveDecoder {
    async fn decode(
        &self,
        _request: Request<blackbox_decoder::DecodingProblem>,
    ) -> Result<Response<blackbox_decoder::ParityFactor>, Status> {
        Ok(Response::new(blackbox_decoder::ParityFactor { subgraph: vec![] }))
    }

    async fn load_hypergraph(
        &self,
        _request: Request<blackbox_decoder::DecodingHypergraph>,
    ) -> Result<Response<blackbox_decoder::LoadHypergraphResponse>, Status> {
        Ok(Response::new(blackbox_decoder::LoadHypergraphResponse { hid: 1 }))
    }

    async fn decode_loaded(
        &self,
        _request: Request<blackbox_decoder::LoadedDecodingProblem>,
    ) -> Result<Response<blackbox_decoder::ParityFactor>, Status> {
        Ok(Response::new(blackbox_decoder::ParityFactor { subgraph: vec![] }))
    }

    async fn reset(&self, _request: Request<blackbox_decoder::ResetRequest>) -> Result<Response<()>, Status> {
        Ok(().into())
    }
}
