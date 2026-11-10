use deq_runtime::jit::{JitInstruction, JitLibrary};
use flate2::read::GzDecoder;
use prost::Message;
use serde::Deserialize;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct FixtureManifest {
    pub schema_version: u32,
    pub distance: u32,
    pub probability: f64,
    pub compiler_threads: usize,
    pub maximum_mean_epoch_ms: f64,
    pub cases: Vec<CaseMetadata>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CaseMetadata {
    pub name: String,
    pub width: u64,
    pub layers: u64,
    pub epochs: u64,
    pub gadgets: u64,
    pub checks: u64,
    pub errors: u64,
    pub protobuf_crc32: u32,
}

pub struct Fixture {
    pub manifest: FixtureManifest,
    pub manifest_crc32: u32,
    pub library: JitLibrary,
    pub programs: Vec<(CaseMetadata, Vec<JitInstruction>)>,
}

fn decode(data: &[u8], expected_crc: u32) -> Result<JitLibrary, Box<dyn std::error::Error>> {
    let mut protobuf = Vec::new();
    GzDecoder::new(data).read_to_end(&mut protobuf)?;
    if crc32fast::hash(&protobuf) != expected_crc {
        return Err("fixture protobuf checksum does not match the manifest".into());
    }
    Ok(JitLibrary::decode(protobuf.as_slice())?)
}

pub fn load() -> Result<Fixture, Box<dyn std::error::Error>> {
    load_from(&Path::new(env!("CARGO_MANIFEST_DIR")).join("benches/fixtures"))
}

pub fn load_from(directory: &Path) -> Result<Fixture, Box<dyn std::error::Error>> {
    let manifest_bytes = std::fs::read(directory.join("manifest.json")).map_err(|error| {
        std::io::Error::new(
            error.kind(),
            format!(
                "cannot read JIT benchmark fixtures at {}: {error}; run python deq/deq_runtime/benches/generate_jit_benchmark_fixture.py from the repository root",
                directory.display()
            ),
        )
    })?;
    let manifest_crc32 = crc32fast::hash(&manifest_bytes);
    let manifest: FixtureManifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest.schema_version != 1 || manifest.distance != 7 || manifest.probability.to_bits() != 0.003_f64.to_bits() {
        return Err("unsupported JIT benchmark fixture".into());
    }
    let small = manifest
        .cases
        .iter()
        .find(|case| case.width == 4)
        .ok_or("missing small case")?;
    let large = manifest
        .cases
        .iter()
        .find(|case| case.width == 1000)
        .ok_or("missing width-1000 case")?;
    let mut library = decode(
        &std::fs::read(directory.join("lattice_surgery_d7_w1000_l50.jit.gz"))?,
        large.protobuf_crc32,
    )?;
    let small_program = decode(
        &std::fs::read(directory.join("lattice_surgery_d7_w4_l2.program.jit.gz"))?,
        small.protobuf_crc32,
    )?;
    let large_program = std::mem::take(&mut library.program);
    let programs = vec![(small.clone(), small_program.program), (large.clone(), large_program)];
    for (metadata, program) in &programs {
        if u64::try_from(program.len())? != metadata.gadgets || metadata.epochs != 4 * metadata.layers + 2 {
            return Err("program size or cadence does not match its fixture metadata".into());
        }
    }
    Ok(Fixture {
        manifest,
        manifest_crc32,
        library,
        programs,
    })
}
