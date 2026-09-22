#[cfg(feature = "cli")]
use crate::misc::fastrace::FileReporter;
use crate::misc::parser::SerdeJsonParser;
use crate::{controller, coordinator, decoder, simulator};
use clap::Parser;
use clap::builder::{TypedValueParser, ValueParser};
use coordinator::CoordinatorClient;
#[cfg(feature = "cli")]
use fastrace::collector::Config;
use futures_util::FutureExt;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};

include!("proto/deq.server.rs");

#[derive(Parser, Clone, Debug)]
pub struct ServerConfigs {
    /// the socket address to bind the server
    #[clap(long, default_value_t = format!("[::]:50051"))]
    pub addr: String,
    /// the type of the decoder algorithm
    #[clap(short = 'd', long, value_enum, default_value_t = decoder::DecoderType::BlackBoxNaive)]
    pub decoder: decoder::DecoderType,
    #[clap(
        long,
        default_value_t = json!({}),
        value_parser = ValueParser::new(SerdeJsonParser),
        help = decoder::DecoderType::config_help()
    )]
    pub decoder_config: serde_json::Value,
    /// Decoder for forced-gap alternatives; omitted options reuse the hard decoder.
    #[clap(long, value_enum)]
    pub gap_decoder: Option<decoder::DecoderType>,
    /// Gap search configuration; shares the hard decoder's thread pool. Set parallel only in decoder-config.
    /// A config-only override uses the hard decoder's type.
    #[clap(long, value_parser = SerdeJsonParser.try_map(validate_gap_decoder_config))]
    pub gap_decoder_config: Option<serde_json::Value>,
    /// the type of the decoding coordinator
    #[clap(short = 'c', long, value_enum, default_value_t = coordinator::CoordinatorType::Naive)]
    pub coordinator: coordinator::CoordinatorType,
    #[clap(
        long,
        default_value_t = json!({}),
        value_parser = ValueParser::new(SerdeJsonParser),
        help = coordinator::CoordinatorType::config_help()
    )]
    pub coordinator_config: serde_json::Value,
    /// the type of the controller (optional)
    #[clap(long, value_enum, default_value_t = controller::ControllerType::None)]
    pub controller: controller::ControllerType,
    #[clap(
        long,
        default_value_t = json!({}),
        value_parser = ValueParser::new(SerdeJsonParser),
        help = controller::ControllerType::config_help()
    )]
    pub controller_config: serde_json::Value,
    #[clap(long, default_value_t = false)]
    pub controller_use_remote_client: bool,
    /// the type of the simulator (optional)
    #[clap(short = 's', long, value_enum, default_value_t = simulator::SimulatorType::None)]
    pub simulator: simulator::SimulatorType,
    #[clap(
        long,
        default_value_t = json!({}),
        value_parser = ValueParser::new(SerdeJsonParser),
        help = simulator::SimulatorType::config_help()
    )]
    pub simulator_config: serde_json::Value,
    #[clap(long)]
    pub trace: Option<String>,
}

fn validate_gap_decoder_config(config: serde_json::Value) -> Result<serde_json::Value, String> {
    if config.get("parallel").is_some() {
        return Err(
            "gap decoder configuration must not contain 'parallel'; set the shared thread pool size in --decoder-config"
                .into(),
        );
    }
    Ok(config)
}

impl ServerConfigs {
    pub async fn run(self) {
        let addr: core::net::SocketAddr = self.addr.parse().unwrap();
        let mut server = tonic::transport::Server::builder();
        let tcp_nodelay = true; // enabled by default
        let tcp_keepalive = None; // disabled by default
        let incoming = tonic::transport::server::TcpIncoming::bind(addr)
            .unwrap()
            .with_nodelay(Some(tcp_nodelay))
            .with_keepalive(tcp_keepalive);
        let addr = incoming.local_addr().unwrap();
        // When the server binds to an unspecified address ([::]  or 0.0.0.0),
        // internal clients must connect via the corresponding loopback address
        // since the unspecified address is not routable (especially on Windows).
        let client_ip = match addr.ip() {
            std::net::IpAddr::V6(ip) if ip.is_unspecified() => std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST),
            std::net::IpAddr::V4(ip) if ip.is_unspecified() => std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            other => other,
        };
        let client_addr = std::net::SocketAddr::new(client_ip, addr.port());
        let url = format!("http://{client_addr}");
        println!("server running on {:?} (port={})", url, addr.port());
        // Flush to ensure the startup message is visible when stdout is piped
        let _ = std::io::Write::flush(&mut std::io::stdout());

        if let Some(trace_file) = self.trace.as_ref() {
            #[cfg(feature = "cli")]
            fastrace::set_reporter(FileReporter::new(trace_file), Config::default());
            #[cfg(not(feature = "cli"))]
            eprintln!("warning: --trace requires the 'fastrace' feature; ignoring trace_file={trace_file}");
        }

        let endpoint = tonic::transport::Endpoint::from_shared(url).unwrap();
        // add the server control services
        let router =
            server.add_service(server_server::ServerServer::new(ServerState {}).max_decoding_message_size(usize::MAX));
        // add the decoder service
        let (decoder, gap_decoder) = self.create_decoders();
        let router = decoder.add_service(router);
        // add coordinator service
        let coordinator =
            self.coordinator
                .create_with_gap_decoder(self.coordinator_config.clone(), decoder.clone(), gap_decoder);
        let router = coordinator.add_service(router);
        coordinator.start().await;
        // create the controller
        let controller = self.controller.create(self.controller_config);
        let router = controller.add_service(router);
        let coordinator_client = if self.controller_use_remote_client {
            CoordinatorClient::from_endpoint(endpoint.clone()).await
        } else {
            CoordinatorClient::Local(coordinator.clone())
        };
        controller.start(coordinator_client).await;
        // create a handle to shutdown the server
        let (tx, rx) = oneshot::channel::<()>();
        // lastly, create the simulator and start it. the simulator acts as a client to the services
        let simulator = self.simulator.create(self.simulator_config);
        tokio::spawn(async move { simulator.start(endpoint, tx).await });

        // serve the gRPC requests
        router.serve_with_incoming_shutdown(incoming, rx.map(drop)).await.unwrap();

        #[cfg(feature = "cli")]
        fastrace::flush();
    }

    fn create_decoders(&self) -> (decoder::DynDecoder, Option<decoder::DynDecoder>) {
        let gap_config = validate_gap_decoder_config(self.gap_decoder_config.clone().unwrap_or_else(|| json!({})))
            .expect("invalid gap decoder configuration");
        let hard = self.decoder.create(self.decoder_config.clone());
        let gap = if self.gap_decoder.is_none() && self.gap_decoder_config.is_none() {
            None
        } else {
            Some(
                self.gap_decoder
                    .unwrap_or(self.decoder)
                    .create_with_thread_pool(gap_config, hard.thread_pool().cloned()),
            )
        };
        (hard, gap)
    }

    /// Build an in-process [`LocalServer`] from this config without binding to
    /// a network address. The `controller_use_remote_client` flag is ignored;
    /// in-process callers have no reason to pay gRPC overhead. Use
    /// [`LocalServer::bind_grpc`] to optionally expose a network endpoint on top.
    pub async fn build_local(self) -> Arc<LocalServer> {
        let (decoder, gap_decoder) = self.create_decoders();
        let coordinator = self
            .coordinator
            .create_with_gap_decoder(self.coordinator_config, decoder.clone(), gap_decoder);
        coordinator.start().await;
        let controller = self.controller.create(self.controller_config);
        let coordinator_client = CoordinatorClient::Local(coordinator.clone());
        controller.start(coordinator_client).await;
        Arc::new(LocalServer {
            decoder,
            coordinator,
            controller,
            bind_state: Mutex::new(BindState::Unbound),
        })
    }
}

/// State tracking the optional gRPC binding of a [`LocalServer`].
enum BindState {
    /// Not bound to any network address.
    Unbound,
    /// Bound: holds the shutdown signal and the join handle for the serve loop.
    Bound {
        shutdown_tx: oneshot::Sender<()>,
        serve_handle: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
        bound_port: u16,
        url: String,
    },
    /// Shutdown has been triggered; the serve handle is still available to await.
    ShuttingDown {
        serve_handle: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
    },
    /// Finished serving and joined.
    Finished,
}

/// A locally constructed `deq` runtime that owns the decoder, coordinator and
/// (optional) controller services in process. Local clients call directly into
/// the service traits with no serialization; an optional gRPC server can be
/// bound on top so remote clients can connect concurrently.
pub struct LocalServer {
    decoder: decoder::DynDecoder,
    coordinator: coordinator::DynCoordinator,
    controller: controller::DynController,
    bind_state: Mutex<BindState>,
}

impl LocalServer {
    /// In-process client to the coordinator service. Always available because
    /// every server is configured with one.
    pub fn coordinator_client(&self) -> CoordinatorClient {
        CoordinatorClient::Local(self.coordinator.clone())
    }

    /// Access the underlying decoder.
    pub fn decoder(&self) -> &decoder::DynDecoder {
        &self.decoder
    }

    /// Access the (possibly None) controller.
    pub fn controller(&self) -> &controller::DynController {
        &self.controller
    }

    /// In-process handle to the JIT controller, when one is configured.
    /// Returns `None` for `controller="none"` or `controller="static"`.
    pub fn jit_controller(&self) -> Option<Arc<controller::JitController>> {
        match &self.controller {
            controller::DynController::Jit(c) => Some(c.clone()),
            _ => None,
        }
    }

    /// In-process handle to the static controller, when one is configured.
    pub fn static_controller(&self) -> Option<Arc<controller::StaticController>> {
        match &self.controller {
            controller::DynController::Static(c) => Some(c.clone()),
            _ => None,
        }
    }

    /// Bind a gRPC server on `addr`. Returns the actual URL clients should use
    /// to connect (with the unspecified-address rewrite applied).
    pub async fn bind_grpc(self: &Arc<Self>, addr: core::net::SocketAddr) -> Result<String, BindError> {
        let mut guard = self.bind_state.lock().await;
        if !matches!(*guard, BindState::Unbound) {
            return Err(BindError::AlreadyBound);
        }

        let mut server = tonic::transport::Server::builder();
        let tcp_nodelay = true;
        let tcp_keepalive = None;
        let incoming = tonic::transport::server::TcpIncoming::bind(addr)
            .map_err(|e| BindError::Bind(e.to_string()))?
            .with_nodelay(Some(tcp_nodelay))
            .with_keepalive(tcp_keepalive);
        let bound_addr = incoming.local_addr().map_err(|e| BindError::Bind(e.to_string()))?;
        let client_ip = match bound_addr.ip() {
            std::net::IpAddr::V6(ip) if ip.is_unspecified() => std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST),
            std::net::IpAddr::V4(ip) if ip.is_unspecified() => std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            other => other,
        };
        let client_addr = std::net::SocketAddr::new(client_ip, bound_addr.port());
        let url = format!("http://{client_addr}");

        let router =
            server.add_service(server_server::ServerServer::new(ServerState {}).max_decoding_message_size(usize::MAX));
        let router = self.decoder.add_service(router);
        let router = self.coordinator.add_service(router);
        let router = self.controller.add_service(router);

        let (tx, rx) = oneshot::channel::<()>();
        let serve_handle = tokio::spawn(async move { router.serve_with_incoming_shutdown(incoming, rx.map(drop)).await });

        *guard = BindState::Bound {
            shutdown_tx: tx,
            serve_handle,
            bound_port: bound_addr.port(),
            url: url.clone(),
        };
        Ok(url)
    }

    /// Returns the bound port if a gRPC server is currently bound.
    pub fn bound_port(&self) -> Option<u16> {
        match self.bind_state.try_lock() {
            Ok(g) => match &*g {
                BindState::Bound { bound_port, .. } => Some(*bound_port),
                _ => None,
            },
            Err(_) => None,
        }
    }

    /// Returns the URL that should be used to connect to the bound server.
    pub async fn bound_url(&self) -> Option<String> {
        match &*self.bind_state.lock().await {
            BindState::Bound { url, .. } => Some(url.clone()),
            _ => None,
        }
    }

    /// Signal the gRPC server to shut down. Returns immediately; await
    /// [`Self::serve_handle`] to wait for the serve loop to finish.
    pub async fn shutdown_grpc(&self) {
        let mut guard = self.bind_state.lock().await;
        let state = std::mem::replace(&mut *guard, BindState::Unbound);
        match state {
            BindState::Bound {
                shutdown_tx,
                serve_handle,
                ..
            } => {
                let _ = shutdown_tx.send(());
                *guard = BindState::ShuttingDown { serve_handle };
            }
            other => *guard = other,
        }
    }

    /// Fully shut the runtime down: propagate cancellation into the
    /// controller and coordinator services so any in-flight RPCs (notably
    /// `decode` calls waiting on an open frontier) return promptly, then
    /// stop the gRPC serve loop (if bound) and await it.
    ///
    /// After this returns, in-process Python callers awaiting any service
    /// method receive `Status::cancelled` rather than hanging forever, so the
    /// Python event loop can finalize cleanly without leaked pending tasks.
    pub async fn shutdown(&self) -> Result<(), String> {
        self.controller.cancel_pending().await;
        self.coordinator.cancel_pending().await;
        self.shutdown_grpc().await;
        self.serve_handle().await
    }

    /// Wait for the serve loop (if one was bound) to finish. Returns Ok(()) if
    /// no server is bound or once the serve task completes.
    pub async fn serve_handle(&self) -> Result<(), String> {
        let mut guard = self.bind_state.lock().await;
        let state = std::mem::replace(&mut *guard, BindState::Finished);
        match state {
            BindState::Unbound | BindState::Finished => {
                *guard = BindState::Finished;
                Ok(())
            }
            BindState::Bound {
                shutdown_tx,
                serve_handle,
                ..
            } => {
                drop(shutdown_tx);
                let res = serve_handle.await.map_err(|e| e.to_string())?;
                res.map_err(|e| e.to_string())
            }
            BindState::ShuttingDown { serve_handle } => {
                let res = serve_handle.await.map_err(|e| e.to_string())?;
                res.map_err(|e| e.to_string())
            }
        }
    }
}

/// Error returned by [`LocalServer::bind_grpc`].
#[derive(Debug)]
pub enum BindError {
    AlreadyBound,
    Bind(String),
}

impl std::fmt::Display for BindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindError::AlreadyBound => write!(f, "gRPC server is already bound"),
            BindError::Bind(msg) => write!(f, "failed to bind: {msg}"),
        }
    }
}

impl std::error::Error for BindError {}

pub struct ServerState {
    // pub shutdown_tx,
}

#[tonic::async_trait]
impl server_server::Server for ServerState {
    async fn shutdown(&self, _request: Request<()>) -> std::result::Result<Response<()>, Status> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_decoder_thread_pool_preserves_separate_configs() {
        for explicit_type in [false, true] {
            let mut arguments = vec![
                "server",
                "--decoder",
                "black-box-relay-bp",
                "--decoder-config",
                "{\"parallel\":2,\"seed\":17}",
                "--gap-decoder-config",
                "{\"seed\":23}",
            ];
            if explicit_type {
                arguments.extend(["--gap-decoder", "black-box-relay-bp"]);
            }
            let config = ServerConfigs::try_parse_from(arguments).unwrap();
            let (hard, gap) = config.create_decoders();
            let decoder::DynDecoder::BlackBoxRelayBP(hard) = hard else {
                panic!("expected relay-bp hard decoder")
            };
            let Some(decoder::DynDecoder::BlackBoxRelayBP(gap)) = gap else {
                panic!("expected relay-bp gap decoder")
            };
            assert!(!Arc::ptr_eq(&hard, &gap));
            assert!(Arc::ptr_eq(&hard.thread_pool, &gap.thread_pool));
            assert_eq!(hard.thread_pool.current_num_threads(), 2);
            assert_eq!(gap.thread_pool.current_num_threads(), 2);
            assert_eq!(gap.config.parallel, 2);
            assert_eq!(hard.original_config["seed"], 17);
            assert_eq!(gap.original_config["seed"], 23);
        }
    }

    #[test]
    fn shared_decoder_thread_pool_supports_different_backends() {
        let backends = [
            "black-box-relay-bp",
            "black-box-relay-bp-f32",
            #[cfg(feature = "tesseract")]
            "black-box-tesseract",
        ];
        for hard_type in backends {
            for gap_type in backends {
                let config = ServerConfigs::try_parse_from([
                    "server",
                    "--decoder",
                    hard_type,
                    "--decoder-config",
                    "{\"parallel\":1}",
                    "--gap-decoder",
                    gap_type,
                ])
                .unwrap();
                let (hard, gap) = config.create_decoders();
                let gap = gap.unwrap();
                assert!(Arc::ptr_eq(hard.thread_pool().unwrap(), gap.thread_pool().unwrap()));
                assert_eq!(gap.thread_pool().unwrap().current_num_threads(), 1);
            }
        }
    }

    #[tokio::test]
    async fn shared_decoder_thread_pool_keeps_caches_and_resets_separate() {
        use crate::decoder::blackbox_decoder::{DecodingHypergraph, Hyperedge, LoadedDecodingProblem, ResetRequest};
        use crate::util::BitVector;

        let config = ServerConfigs::try_parse_from([
            "server",
            "--decoder",
            "black-box-relay-bp",
            "--decoder-config",
            "{\"parallel\":1}",
            "--gap-decoder",
            "black-box-relay-bp-f32",
        ])
        .unwrap();
        let (hard, gap) = config.create_decoders();
        let gap = gap.unwrap();
        let hard_graph = DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![Hyperedge {
                vertices: vec![0],
                probability: 0.1,
            }],
        };
        let gap_graph = DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![
                Hyperedge {
                    vertices: vec![],
                    probability: 0.0,
                },
                Hyperedge {
                    vertices: vec![0],
                    probability: 0.2,
                },
            ],
        };
        let problem = |hid| LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector {
                size: 1,
                data: vec![0x80],
            }),
            ..Default::default()
        };
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let (hard_load, gap_load) =
                tokio::join!(hard.load_hypergraph(hard_graph.clone()), gap.load_hypergraph(gap_graph));
            let hard_hid = hard_load.unwrap().hid;
            let gap_hid = gap_load.unwrap().hid;
            assert_eq!(hard_hid, gap_hid);
            let (hard_result, gap_result) =
                tokio::join!(hard.decode_loaded(problem(hard_hid)), gap.decode_loaded(problem(gap_hid)));
            assert_eq!(hard_result.unwrap().subgraph, vec![0]);
            assert_eq!(gap_result.unwrap().subgraph, vec![1]);

            let reset = ResetRequest {
                reset_hypergraphs: true,
                ..Default::default()
            };
            hard.reset(reset.clone()).await.unwrap();
            assert_eq!(
                hard.decode_loaded(problem(hard_hid)).await.unwrap_err().code(),
                tonic::Code::NotFound
            );
            assert_eq!(gap.decode_loaded(problem(gap_hid)).await.unwrap().subgraph, vec![1]);
            let reloaded = hard.load_hypergraph(hard_graph).await.unwrap().hid;
            gap.reset(reset).await.unwrap();
            assert_eq!(hard.decode_loaded(problem(reloaded)).await.unwrap().subgraph, vec![0]);
        })
        .await
        .expect("shared-pool decoding and reset must not deadlock");
    }

    #[test]
    fn gap_decoder_config_rejects_pool_size() {
        for parallel in [json!(0), json!(1), json!(2), json!(null), json!("auto")] {
            let gap_config = json!({"parallel": parallel}).to_string();
            for explicit_type in [false, true] {
                let mut arguments = vec![
                    "server",
                    "--decoder",
                    "black-box-relay-bp",
                    "--decoder-config",
                    "{\"parallel\":2}",
                    "--gap-decoder-config",
                    &gap_config,
                ];
                if explicit_type {
                    arguments.extend(["--gap-decoder", "black-box-relay-bp"]);
                }
                let error = ServerConfigs::try_parse_from(arguments).unwrap_err();
                assert!(error.to_string().contains("parallel"));
                assert!(error.to_string().contains("--decoder-config"));
            }
        }
    }

    #[test]
    fn omitted_gap_options_reuse_the_hard_decoder() {
        let config = ServerConfigs::try_parse_from(["server", "--decoder", "mock"]).unwrap();
        let (hard, gap) = config.create_decoders();
        assert!(matches!(hard, decoder::DynDecoder::Mock(_)));
        assert!(gap.is_none());
    }

    #[test]
    fn explicit_gap_decoder_does_not_inherit_unrelated_config() {
        let config = ServerConfigs::try_parse_from([
            "server",
            "--decoder",
            "black-box-relay-bp",
            "--decoder-config",
            "{\"parallel\":1}",
            "--gap-decoder",
            "mock",
        ])
        .unwrap();
        let (hard, gap) = config.create_decoders();
        assert!(matches!(hard, decoder::DynDecoder::BlackBoxRelayBP(_)));
        assert!(matches!(gap, Some(decoder::DynDecoder::Mock(_))));
    }

    #[test]
    fn gap_config_only_selects_the_primary_decoder_type() {
        let config = ServerConfigs::try_parse_from(["server", "--decoder", "mock", "--gap-decoder-config", "{}"]).unwrap();
        let (hard, gap) = config.create_decoders();
        let decoder::DynDecoder::Mock(hard) = hard else {
            panic!("expected mock hard decoder")
        };
        let Some(decoder::DynDecoder::Mock(gap)) = gap else {
            panic!("expected mock gap decoder")
        };
        assert!(!Arc::ptr_eq(&hard, &gap));
    }
}
