use clap::Subcommand;
use std::path::PathBuf;

#[derive(Subcommand, Clone, Debug)]
pub enum BenchmarkCommands {
    /// Benchmark JIT compilation latency.
    ///
    /// Loads a .deq.jit file, deserializes it once, then compiles it
    /// N times and prints per-trial durations as JSON to stdout.
    JitCompile {
        /// Path to the .deq.jit file
        file: PathBuf,
        /// Number of compilation trials
        #[clap(long, default_value = "100")]
        trials: u32,
    },
}

impl BenchmarkCommands {
    pub async fn run(self) {
        match self {
            Self::JitCompile { file, trials } => {
                run_jit_compile_benchmark(&file, trials).await;
            }
        }
    }
}

async fn run_jit_compile_benchmark(file: &std::path::Path, trials: u32) {
    use crate::jit::{JitLibrary, static_jit_compile};
    use prost::Message;

    let bytes = std::fs::read(file).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", file.display());
        std::process::exit(1);
    });
    let jit_library = JitLibrary::decode(&*bytes).unwrap_or_else(|e| {
        eprintln!("Failed to decode JitLibrary: {e}");
        std::process::exit(1);
    });

    let n_gadgets = jit_library.program.len();
    eprintln!("Benchmarking {} ({} gadgets, {} trials)", file.display(), n_gadgets, trials);

    // Measure the cost of deep-cloning all used gadget types as a baseline.
    // Each gadget type is cloned once per instruction that uses it, reflecting
    // the fact that the JIT compiler clones check/error models per gadget instance.
    {
        use hashbrown::HashMap;
        let gtype_map: HashMap<u64, &_> = jit_library
            .gadget_types
            .iter()
            .filter_map(|gt| gt.base.as_ref().map(|b| (b.gtype, gt)))
            .collect();
        // Collect the gadget types in program order, with duplicates
        let types_to_clone: Vec<&_> = jit_library
            .program
            .iter()
            .filter_map(|i| i.gadget.as_ref().and_then(|g| gtype_map.get(&g.gtype).copied()))
            .collect();
        let t = std::time::Instant::now();
        for _ in 0..trials {
            for gt in &types_to_clone {
                std::hint::black_box((*gt).clone());
            }
        }
        let clone_us = t.elapsed().as_micros() / trials as u128;
        eprintln!(
            "  clone baseline ({} gadget type copies): {clone_us} µs",
            types_to_clone.len()
        );
    }

    let mut durations = Vec::with_capacity(trials as usize);
    for i in 0..trials {
        let jit_clone = jit_library.clone();
        let t0 = std::time::Instant::now();
        static_jit_compile(jit_clone).await;
        durations.push(t0.elapsed().as_secs_f64());
        if (i + 1) % 1000 == 0 {
            eprintln!("  {}/{}", i + 1, trials);
        }
    }

    // Output JSON to stdout
    print!("[");
    for (i, d) in durations.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{d}");
    }
    println!("]");
}
