//! Opt-in harness for measuring the production shared JIT path without decoding.

use super::{JitInstruction, JitLibrary, jit_compiler::JitCompiler};
use hashbrown::{HashMap, HashSet};
use serde::Serialize;
use std::hint::black_box;
use std::sync::Arc;
use tokio::sync::watch;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

/// A dependency-ordered program prepared outside the measured interval.
pub struct PreparedProgram {
    instructions: Vec<PreparedInstruction>,
}

struct PreparedInstruction {
    instruction: JitInstruction,
    dependencies: Vec<watch::Receiver<bool>>,
    compiled: watch::Sender<bool>,
}

impl PreparedProgram {
    /// Allocate dependency notifications without executing any compiler work.
    ///
    /// # Errors
    /// Rejects missing gadgets, zero/duplicate IDs, and non-topological dependencies.
    pub fn new(program: Vec<JitInstruction>) -> Result<Self, String> {
        let mut signals = HashMap::<u64, watch::Sender<bool>>::new();
        let mut instructions = Vec::with_capacity(program.len());
        for instruction in program {
            let gadget = instruction.gadget.as_ref().ok_or("instruction has no gadget")?;
            if gadget.gid == 0 || signals.contains_key(&gadget.gid) {
                return Err(format!("gadget ID {} is zero or duplicated", gadget.gid));
            }
            let dependencies = gadget
                .submission_dependencies()
                .map(|gid| {
                    signals
                        .get(&gid)
                        .map(watch::Sender::subscribe)
                        .ok_or_else(|| format!("gid={} references unscheduled predecessor {gid}", gadget.gid))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let compiled = watch::channel(false).0;
            signals.insert(gadget.gid, compiled.clone());
            instructions.push(PreparedInstruction {
                instruction,
                dependencies,
                compiled,
            });
        }
        Ok(Self { instructions })
    }
}

/// Counters that make skipped compilation, unresolved futures, and retained state visible.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CompilationSummary {
    pub gadgets: u64,
    pub checks: u64,
    pub errors: u64,
    pub peak_live_gadgets: usize,
    pub retained_gadgets: usize,
}

/// A loaded compiler with no controller, coordinator, simulator, or decoder.
pub struct CompilerHarness {
    compiler: Arc<JitCompiler>,
}

impl CompilerHarness {
    /// Load immutable type definitions outside the measured interval.
    ///
    /// # Errors
    /// Rejects a library carrying executable instructions or duplicate type IDs.
    pub async fn new(library: JitLibrary) -> Result<Self, String> {
        if !library.program.is_empty() {
            return Err("pass type definitions separately from the executable program".to_string());
        }
        let mut type_ids = HashSet::new();
        for definition in &library.gadget_types {
            let base = definition.base.as_ref().ok_or("gadget type has no base definition")?;
            if !type_ids.insert(base.gtype) {
                return Err(format!("duplicate gadget type {}", base.gtype));
            }
        }
        let compiler = JitCompiler::new();
        compiler.load_library(library).await;
        Ok(Self { compiler })
    }

    /// Clear execution state between iterations while preserving immutable templates.
    pub async fn reset(&self) {
        self.compiler.reset().await;
    }

    /// Compile all instructions, resolve every error model, and collect retired JIT state.
    ///
    /// # Errors
    /// Reports cancelled/failed tasks, mismatched model owners, and undrained state.
    pub async fn run(&self, program: PreparedProgram) -> Result<CompilationSummary, String> {
        let token = CancellationToken::new();
        let mut tasks = JoinSet::new();
        for prepared in program.instructions {
            let compiler = Arc::clone(&self.compiler);
            let token = token.clone();
            tasks.spawn(async move {
                for mut dependency in prepared.dependencies {
                    dependency
                        .wait_for(|compiled| *compiled)
                        .await
                        .map_err(|error| format!("predecessor compilation was cancelled: {error}"))?;
                }
                let (gadget, check_type, check, error_future) = compiler.compile_shared(prepared.instruction, token).await;
                let gid = gadget.gid;
                let checks = u64::try_from(check_type.checks.len()).map_err(|error| error.to_string())?;
                black_box((&gadget, &check_type, &check));
                prepared.compiled.send_replace(true);
                drop(gadget);
                drop(check_type);
                drop(check);
                let (error_type, error_model) = error_future.await;
                if error_model.cid != gid {
                    return Err(format!("gid={gid} received an error model for cid={}", error_model.cid));
                }
                let errors = u64::try_from(error_type.errors.len()).map_err(|error| error.to_string())?;
                black_box((&error_type, &error_model));
                drop(error_type);
                drop(error_model);
                compiler.mark_error_model_ready(gid).await;
                let graph = compiler.gadgets.write().await;
                let live = graph.len();
                compiler.collect_garbage(graph).await;
                Ok::<_, String>((checks, errors, live))
            });
        }
        let mut summary = CompilationSummary::default();
        while let Some(result) = tasks.join_next().await {
            let (checks, errors, live) = result.map_err(|error| format!("JIT task failed: {error}"))??;
            summary.gadgets += 1;
            summary.checks += checks;
            summary.errors += errors;
            summary.peak_live_gadgets = summary.peak_live_gadgets.max(live);
        }
        self.compiler.collect_garbage(self.compiler.gadgets.write().await).await;
        summary.retained_gadgets = self.compiler.gadgets.read().await.len();
        if summary.retained_gadgets != 0 {
            return Err(format!(
                "{} JIT gadgets remained after program completion",
                summary.retained_gadgets
            ));
        }
        Ok(summary)
    }
}
