#![cfg(all(feature = "cli", feature = "simulator"))]

use deq_runtime::jit::JitLibrary;
use prost::Message;
use std::fs;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

fn simulate(script: &str) -> (Output, Vec<u8>) {
    let directory = tempfile::tempdir().unwrap();
    let circuit = directory.path().join("circuit.stim");
    let library = directory.path().join("library.deq.jit");
    let trace = directory.path().join("trace.pb");
    let stdout = directory.path().join("stdout");
    let stderr = directory.path().join("stderr");
    fs::write(&circuit, format!("#!rhai\n# {script}\nR 0\nM 0\n")).unwrap();
    fs::write(&library, JitLibrary::default().encode_to_vec()).unwrap();
    let config = serde_json::json!({
        "filepath": circuit,
        "jit_library_filepath": library,
        "shots": 1,
        "errors": 0,
        "seed": 1,
        "strict_timing": true
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_deq-runtime-cli"))
        .args([
            "server",
            "--addr",
            "127.0.0.1:0",
            "--controller",
            "jit",
            "--coordinator",
            "window",
            "--simulator",
            "jit-static",
            "--simulator-config",
            &config.to_string(),
            "--trace",
        ])
        .arg(&trace)
        .stdout(Stdio::from(fs::File::create(&stdout).unwrap()))
        .stderr(Stdio::from(fs::File::create(&stderr).unwrap()))
        .spawn()
        .unwrap();
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if started.elapsed() > Duration::from_secs(30) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!(
                "simulator did not terminate:\n{}\n{}",
                fs::read_to_string(&stdout).unwrap(),
                fs::read_to_string(&stderr).unwrap()
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    (
        Output {
            status,
            stdout: fs::read(stdout).unwrap(),
            stderr: fs::read(stderr).unwrap(),
        },
        fs::read(trace).unwrap(),
    )
}

#[test]
fn spawned_simulator_startup_panic_is_a_failed_process() {
    let (output, _) = simulate("fn is_logical_error( {");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("server running on"), "{stdout}");
    assert!(stderr.contains("Failed to compile Rhai script"), "{stderr}");
    assert!(!output.status.success(), "{stdout}\n{stderr}");
    assert!(stderr.contains("simulator task failed"), "{stderr}");
    assert!(!stdout.contains("=== Simulation Complete ==="), "{stdout}");
}

#[test]
fn spawned_simulator_runtime_panic_flushes_trace_before_failing() {
    let (output, trace) = simulate("fn is_logical_error(shot, readouts, measurements) { throw \"simulator-regression\"; }");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Rhai is_logical_error() failed"), "{stderr}");
    assert!(!output.status.success(), "{stdout}\n{stderr}");
    assert!(stderr.contains("simulator task failed"), "{stderr}");
    assert!(!stdout.contains("=== Simulation Complete ==="), "{stdout}");
    assert!(!trace.is_empty(), "the failed shot's trace must be flushed");
}

#[test]
fn successful_simulation_exits_successfully() {
    let (output, _) = simulate("fn is_logical_error(shot, readouts, measurements) { false }");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(stdout.contains("=== Simulation Complete ==="), "{stdout}");
    assert!(stdout.contains("Shots: 1/1"), "{stdout}");
    assert!(stdout.contains("Logical errors: 0"), "{stdout}");
    assert!(stdout.contains("Failed shots: 0"), "{stdout}");
}

#[test]
fn valid_logical_errors_are_successful_simulation_results() {
    let (output, _) = simulate("fn is_logical_error(shot, readouts, measurements) { true }");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(stdout.contains("=== Simulation Complete ==="), "{stdout}");
    assert!(stdout.contains("Logical errors: 1"), "{stdout}");
    assert!(stdout.contains("Failed shots: 0"), "{stdout}");
}
