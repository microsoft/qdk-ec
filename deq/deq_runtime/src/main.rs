use clap::Parser;
use deq_runtime::cli::*;

#[tokio::main]
pub async fn main() {
    Cli::parse().run().await;
}
