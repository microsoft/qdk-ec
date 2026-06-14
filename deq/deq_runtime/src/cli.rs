use crate::benchmark::BenchmarkCommands;
use crate::server::ServerConfigs;
use clap::builder::styling;
use clap::{Parser, Subcommand};
use std::env;

#[derive(Parser, Clone)]
#[clap(author = clap::crate_authors!(", "))]
#[clap(version = env!("CARGO_PKG_VERSION"))]
#[clap(about = env!("CARGO_PKG_DESCRIPTION"))]
#[clap(color = clap::ColorChoice::Auto)]
#[clap(
    styles = styling::Styles
        ::styled()
        .header(styling::AnsiColor::Green.on_default() | styling::Effects::BOLD)
        .usage(styling::AnsiColor::Green.on_default() | styling::Effects::BOLD)
        .literal(styling::AnsiColor::Cyan.on_default() | styling::Effects::BOLD)
        .placeholder(styling::AnsiColor::Cyan.on_default())
)]
#[clap(propagate_version = true)]
#[clap(subcommand_required = true)]
#[clap(arg_required_else_help = true)]
pub struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Clone)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    Server(ServerConfigs),
    /// built-in benchmarks
    Benchmark {
        #[clap(subcommand)]
        command: BenchmarkCommands,
    },
}

impl Cli {
    pub async fn run(self) {
        match self.command {
            Commands::Server(configs) => {
                configs.run().await;
            }
            Commands::Benchmark { command } => {
                command.run().await;
            }
        }
    }
}

pub async fn execute_in_cli<'a>(iter: impl Iterator<Item = &'a String> + Clone, print_command: bool) {
    if print_command {
        print!("[command]");
        for word in iter.clone() {
            if word.contains(char::is_whitespace) {
                print!("'{word}' ");
            } else {
                print!("{word} ");
            }
        }
        println!();
    }
    Cli::parse_from(iter).run().await;
}
