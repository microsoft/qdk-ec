#[cfg(feature = "cli")]
use crate::misc::fastrace::FileReporter;
use crate::misc::parser::SerdeJsonParser;
use crate::{controller, coordinator, decoder, simulator};
use clap::Parser;
use clap::builder::ValueParser;
use coordinator::CoordinatorClient;
#[cfg(feature = "cli")]
use fastrace::collector::Config;
use futures_util::FutureExt;
use serde_json::json;
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
    #[clap(long, default_value_t = false)]
    pub coordinator_use_remote_client: bool,
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
        let decoder = self.decoder.create(self.decoder_config);
        let router = decoder.add_service(router);
        let black_box_decoder = decoder
            .as_black_box_decoder_client(self.coordinator_use_remote_client.then_some(&endpoint))
            .await;
        // add coordinator service
        let coordinator = self.coordinator.create(self.coordinator_config.clone(), black_box_decoder);
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
}

pub struct ServerState {
    // pub shutdown_tx,
}

#[tonic::async_trait]
impl server_server::Server for ServerState {
    async fn shutdown(&self, _request: Request<()>) -> std::result::Result<Response<()>, Status> {
        unimplemented!()
    }
}
