//! Compile `examples/demo.c` against the generated header and the staticlib,
//! run it, and check the output.
//!
//! This is the only test that proves the header and the library actually work
//! from a C compiler; the other tests call the shims from Rust, which cannot
//! catch a bad header. Missing build prerequisites fail the test.

#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::process::Command;

fn staticlib() -> PathBuf {
    let executable = std::env::current_exe().expect("locate test executable");
    let profile = executable
        .parent()
        .and_then(Path::parent)
        .expect("test is in profile/deps");
    let library = profile.join("libqodec_c.a");
    assert!(
        library.is_file(),
        "{} is missing; run cargo build -p qodec-c with the same target and profile",
        library.display()
    );
    library
}

fn c_compiler() -> Option<String> {
    std::env::var("CC").ok().or_else(|| {
        ["cc", "gcc", "clang"]
            .iter()
            .find(|name| {
                Command::new(name)
                    .arg("--version")
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .is_ok_and(|status| status.success())
            })
            .map(|name| (*name).to_owned())
    })
}

#[test]
fn a_c_program_can_drive_the_abi() {
    let compiler = c_compiler().expect("no C compiler found; set CC to override");
    let library = staticlib();
    let (_out_dir, binary) = compile_demo(&compiler, &library);
    assert_demo_reads_protocol(&binary);
    assert_demo_preserves_argument_shapes(&binary);
    assert_demo_reports_unparsable_circuit(&binary);
    assert_demo_rejects_directory(&binary);
}

fn compile_demo(compiler: &str, library: &Path) -> (tempfile::TempDir, PathBuf) {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    // A TempDir rather than a name derived from the process id: the compiled
    // binary must not land on a predictable path in a shared /tmp, and it is
    // removed when the test ends.
    let out_dir = tempfile::tempdir().expect("create temp dir");
    let binary = out_dir.path().join("demo");

    let compile = Command::new(compiler)
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        // -pedantic keeps the generated header valid ISO C rather than merely
        // GCC-acceptable: it is what catches constructs like a zero-length
        // array, which MSVC also rejects.
        .arg("-pedantic")
        .arg("-I")
        .arg(manifest.join("include"))
        .arg(manifest.join("examples/demo.c"))
        .arg(library)
        // A Rust staticlib pulls in the platform's threading, dl and math bits.
        .args(["-lpthread", "-ldl", "-lm"])
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("run the C compiler");
    assert!(
        compile.status.success(),
        "compiling demo.c failed:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    (out_dir, binary)
}

fn run_demo(binary: &Path, fixture: &str) -> std::process::Output {
    Command::new(binary)
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join(fixture))
        .output()
        .expect("run C demo")
}

fn successful_output(run: std::process::Output) -> String {
    let stdout = String::from_utf8(run.stdout).expect("UTF-8 demo output");
    assert!(
        run.status.success(),
        "demo exited with {:?}:\nstdout: {stdout}\nstderr: {}",
        run.status.code(),
        String::from_utf8_lossy(&run.stderr)
    );
    stdout
}

fn assert_demo_reads_protocol(binary: &Path) {
    let stdout = successful_output(run_demo(binary, "../../examples/repetition3/repetition3.qodec.yaml"));
    assert!(stdout.contains("layers: 2"), "got: {stdout}");
    assert!(stdout.contains("repetition3"), "got: {stdout}");
    assert!(stdout.contains("unknown mnemonic rejected:"), "got: {stdout}");
    // Traversing the tree must render real values, which only works if every
    // struct layout agrees between the header and the library.
    assert!(stdout.contains("circuit.readouts["), "got: {stdout}");
    assert!(stdout.contains("in[0].stabilizers["), "got: {stdout}");
    assert!(stdout.contains("code repetition3"), "got: {stdout}");
    assert!(stdout.contains("Z_0 Z_1"), "got: {stdout}");
    // The action tagged union, across three of its five shapes.
    assert!(stdout.contains("action[0]: stabilize Z_0"), "got: {stdout}");
    assert!(stdout.contains("action[0]: observe Z_0"), "got: {stdout}");
    assert!(stdout.contains("action[0]: rotate Z_0 by <theta>"), "got: {stdout}");
    assert!(
        stdout.contains("circuit did not parse: No source parser registered for '.stim'"),
        "got: {stdout}"
    );
    assert!(stdout.contains("circuit targets stim+rz (stim)"), "got: {stdout}");
    assert!(stdout.trim_end().ends_with("ok"), "got: {stdout}");
}

fn assert_demo_preserves_argument_shapes(binary: &Path) {
    let stdout = successful_output(run_demo(
        binary,
        "tests/fixtures/argument-shapes/argument-shapes.qodec.yaml",
    ));
    assert!(
        stdout.contains("probe q0 angle=0.5 count=-1 gated=rec[0] label=tag names=[alpha beta] targets=[q1 q2]"),
        "got: {stdout}"
    );
    assert!(stdout.contains("program: 6 calls"), "got: {stdout}");
    assert!(stdout.contains("select q0 select=-2 select{select=0}"), "got: {stdout}");
    assert!(
        stdout.lines().any(|line| line.trim() == "select q0 select=-3"),
        "got: {stdout}"
    );
    assert!(
        stdout.contains("boolean q0 disabled=false one=1 select=true zero=0 select{select=1}"),
        "got: {stdout}"
    );
    assert!(
        stdout
            .lines()
            .any(|line| line.trim() == "boolean q0 disabled=true one=1 select=false zero=0"),
        "got: {stdout}"
    );
}

fn assert_demo_reports_unparsable_circuit(binary: &Path) {
    let stdout = successful_output(run_demo(binary, "tests/fixtures/unparsable-body/unparsable.qodec.yaml"));
    assert!(
        stdout.contains("circuit did not parse: No source parser registered for '.openqasm'"),
        "got: {stdout}"
    );
}

fn assert_demo_rejects_directory(binary: &Path) {
    let directory = run_demo(binary, "../../examples/c4c6");
    let stderr = String::from_utf8_lossy(&directory.stderr);
    assert_eq!(directory.status.code(), Some(1), "got: {stderr}");
    assert!(stderr.contains("expected a manifest file path"), "got: {stderr}");
}

#[test]
fn argument_layout_matches_between_c_and_rust() {
    let compiler = c_compiler().expect("no C compiler found; set CC to override");
    let directory = std::env::temp_dir().join(format!("qodec-c-argument-layout-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("create temp dir");
    let binary = directory.join("argument_layout");
    compile_argument_layout_probe(&compiler, &binary);
    let run = Command::new(&binary).output().expect("run argument layout checks");
    assert!(run.status.success());
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert_eq!(stdout.trim(), rust_argument_layout());
    eprintln!("ABI 1 argument layout: {}", stdout.trim());
    std::fs::remove_dir_all(directory).expect("remove temp dir");
}

fn compile_argument_layout_probe(compiler: &str, binary: &Path) {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let compile = Command::new(compiler)
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I"])
        .arg(manifest.join("include"))
        .arg(manifest.join("tests/argument_layout.c"))
        .arg("-o")
        .arg(binary)
        .output()
        .expect("compile argument layout checks");
    assert!(compile.status.success(), "{}", String::from_utf8_lossy(&compile.stderr));
}

fn rust_argument_layout() -> String {
    use qodec_c::{QodecArgument, QodecArgumentValue};
    use std::mem::{align_of, offset_of, size_of};
    let argument = QodecArgumentValue::Boolean { value: true };
    let QodecArgumentValue::Boolean { value } = &argument else {
        unreachable!("constructed a Boolean");
    };
    let payload_offset = std::ptr::from_ref(value).addr() - std::ptr::from_ref(&argument).addr();
    let tag = unsafe { *std::ptr::from_ref(&argument).cast::<u8>() };
    assert_eq!(tag, 7);
    format!(
        "value size={} align={} payload={payload_offset}; argument size={} align={} value={}; Boolean tag={tag}",
        size_of::<QodecArgumentValue>(),
        align_of::<QodecArgumentValue>(),
        size_of::<QodecArgument>(),
        align_of::<QodecArgument>(),
        offset_of!(QodecArgument, value),
    )
}
