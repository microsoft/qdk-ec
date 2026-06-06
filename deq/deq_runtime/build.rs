use std::fs;
use std::path::Path;

const PROTO_DIR: &str = "../proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-changed={PROTO_DIR}");

    // ── Tesseract decoder C++ bridge (feature-gated) ────────────────
    #[cfg(feature = "tesseract")]
    build_tesseract_bridge();

    // Added early return, so proto compilation is skipped during package verification
    if !Path::new(PROTO_DIR).exists() {
        return Ok(());
    }

    let mut proto_files = vec![];
    if let Ok(entries) = fs::read_dir(PROTO_DIR) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && let Some(ext) = path.extension()
                && ext == "proto"
            {
                proto_files.push(path.to_str().unwrap().to_string());
            }
        }
    }

    tonic_prost_build::configure()
        .build_server(true)
        .client_mod_attribute(".", "#[cfg(feature = \"cli\")]")
        .out_dir("src/proto")
        .type_attribute(
            "deq.coordinator.window_coordinator.Event.event",
            "#[allow(clippy::large_enum_variant)]",
        )
        .compile_protos(&proto_files, &[PROTO_DIR.to_string()])?;

    Ok(())
}

// ── Tesseract decoder C++ bridge ────────────────────────────────────

#[cfg(feature = "tesseract")]
fn build_tesseract_bridge() {
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let tesseract_cpp = manifest_dir.join("cpp").join("tesseract");

    println!("cargo::rerun-if-changed=cpp/tesseract/tesseract_core.h");
    println!("cargo::rerun-if-changed=cpp/tesseract/tesseract_bridge.h");
    println!("cargo::rerun-if-changed=cpp/tesseract/tesseract_bridge.cc");
    println!("cargo::rerun-if-changed=src/decoder/tesseract_ffi.rs");

    let boost_dir = download_boost(&out_dir);

    let mut build = cxx_build::bridge("src/decoder/tesseract_ffi.rs");
    build
        .file(tesseract_cpp.join("tesseract_bridge.cc"))
        .std("c++20")
        .warnings(false)
        .include(&tesseract_cpp)
        .include(&boost_dir);

    if std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() == "msvc" {
        build.flag("/EHsc");
    }

    build.compile("tesseract-bridge");
}

// ── Boost download (header-only, for dynamic_bitset) ────────────────

#[cfg(feature = "tesseract")]
fn download_boost(out_dir: &std::path::Path) -> std::path::PathBuf {
    const BOOST_VERSION: &str = "1.83.0";
    const BOOST_SHA256: &str = "c0685b68dd44cc46574cce86c4e17c0f611b15e195be9848dfd0769a0a207628";

    let boost_dir = out_dir.join("boost_src");
    let marker = boost_dir.join(".boost_extracted");
    if marker.exists() {
        return boost_dir;
    }

    let version_underscore = BOOST_VERSION.replace('.', "_");
    let archive_name = format!("boost_{version_underscore}");
    let url = format!("https://archives.boost.io/release/{BOOST_VERSION}/source/{archive_name}.tar.gz");
    let tarball = out_dir.join(format!("{archive_name}.tar.gz"));

    if !tarball.exists() {
        eprintln!("cargo:warning=Downloading Boost {BOOST_VERSION} headers...");
        let status = std::process::Command::new("curl")
            .args(["-sSfL", "-o"])
            .arg(&tarball)
            .arg(&url)
            .status()
            .expect("failed to run curl");
        assert!(status.success(), "failed to download Boost");
    }

    let data = fs::read(&tarball).expect("failed to read tarball");
    let digest = sha256_digest(&data);
    assert_eq!(
        digest, BOOST_SHA256,
        "Boost tarball SHA256 mismatch: expected {BOOST_SHA256}, got {digest}"
    );

    eprintln!("cargo:warning=Extracting Boost headers...");
    fs::create_dir_all(&boost_dir).ok();
    let status = std::process::Command::new("tar")
        .args(["xzf"])
        .arg(&tarball)
        .arg("-C")
        .arg(out_dir)
        .status()
        .expect("failed to run tar");
    assert!(status.success(), "failed to extract Boost tarball");

    let full_dir = out_dir.join(&archive_name);
    let dst_boost = boost_dir.join("boost");
    if dst_boost.exists() {
        fs::remove_dir_all(&dst_boost).ok();
    }
    copy_dir_recursive(&full_dir.join("boost"), &dst_boost);
    fs::remove_dir_all(&full_dir).ok();
    fs::write(&marker, "ok").ok();
    boost_dir
}

#[cfg(feature = "tesseract")]
fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) {
    fs::create_dir_all(dst).expect("failed to create dir");
    for entry in fs::read_dir(src).expect("failed to read dir") {
        let entry = entry.expect("failed to read dir entry");
        let (s, d) = (entry.path(), dst.join(entry.file_name()));
        if s.is_dir() {
            copy_dir_recursive(&s, &d);
        } else {
            fs::copy(&s, &d).expect("failed to copy file");
        }
    }
}

#[cfg(feature = "tesseract")]
fn sha256_digest(data: &[u8]) -> String {
    let tmp = std::env::temp_dir().join("deq_boost_sha256_check");
    fs::write(&tmp, data).expect("write tmp");

    let output = std::process::Command::new("sha256sum")
        .arg(&tmp)
        .output()
        .or_else(|_| std::process::Command::new("shasum").args(["-a", "256"]).arg(&tmp).output());
    fs::remove_file(&tmp).ok();

    if let Ok(output) = output
        && output.status.success()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let token = stdout.split_whitespace().next().unwrap_or("");
        return token
            .chars()
            .filter(|c| c.is_ascii_hexdigit())
            .collect::<String>()
            .to_lowercase();
    }

    let tmp2 = std::env::temp_dir().join("deq_boost_sha256_check2");
    fs::write(&tmp2, data).expect("write tmp");
    let output = std::process::Command::new("certutil")
        .args(["-hashfile"])
        .arg(&tmp2)
        .arg("SHA256")
        .output()
        .expect("failed to run certutil");
    fs::remove_file(&tmp2).ok();
    assert!(output.status.success(), "certutil failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .nth(1)
        .unwrap_or("")
        .chars()
        .filter(|c| c.is_ascii_hexdigit())
        .collect::<String>()
        .to_lowercase()
}
