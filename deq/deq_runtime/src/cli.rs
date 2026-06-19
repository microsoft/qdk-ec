use crate::benchmark::BenchmarkCommands;
#[cfg(feature = "python")]
use crate::misc::parser::SerdeJsonParser;
use crate::server::ServerConfigs;
#[cfg(feature = "python")]
use clap::builder::ValueParser;
use clap::builder::styling;
use clap::{Parser, Subcommand};
#[cfg(feature = "python")]
use serde_json::json;
use std::env;
#[cfg(feature = "python")]
use std::path::PathBuf;

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
    /// Run the standard decoder test suite against a user-provided decoder
    #[cfg(feature = "python")]
    Test {
        #[clap(subcommand)]
        command: TestCommands,
    },
}

#[cfg(feature = "python")]
#[derive(Subcommand, Clone)]
enum TestCommands {
    /// Run the standard suite against a Python decoder defined in a *.py file
    PythonDecoder {
        /// Path to the Python decoder file (must expose a `Decoder` class with `__init__(hypergraph, config)`, `decode(...)`, `reset()`)
        #[clap(long)]
        file: PathBuf,
        /// Optional Python-decoder configuration as a JSON object
        #[clap(long, default_value_t = json!({}), value_parser = ValueParser::new(SerdeJsonParser))]
        py_config: serde_json::Value,
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
            #[cfg(feature = "python")]
            Commands::Test { command } => {
                command.run().await;
            }
        }
    }
}

#[cfg(feature = "python")]
impl TestCommands {
    pub async fn run(self) {
        match self {
            TestCommands::PythonDecoder { file, py_config } => {
                run_python_decoder_test(file, py_config).await;
            }
        }
    }
}

#[cfg(feature = "python")]
async fn run_python_decoder_test(file: PathBuf, py_config: serde_json::Value) {
    use crate::decoder::test_harness::run_standard_suite;
    use crate::decoder::{BlackBoxDecoderClient, DynBlackBoxDecoder, PythonDecoder};
    use std::sync::Arc;

    let config = serde_json::json!({
        "file": file.to_string_lossy(),
        "py_config": py_config,
    });
    let decoder = Arc::new(PythonDecoder::new(config));
    let mut client = BlackBoxDecoderClient::Local(DynBlackBoxDecoder::BlackBoxPython(decoder));
    let report = run_standard_suite(&mut client).await;
    for line in report.summary_lines() {
        println!("{line}");
    }
    let passed = report.pass_count();
    let total = report.total();
    println!("passed: {passed}/{total}");
    if passed != total {
        std::process::exit(1);
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
