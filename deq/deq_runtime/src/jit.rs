include!("proto/deq.jit.rs");

#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod benchmark;
pub mod jit_compiler;
pub mod loss_compiler;

use crate::bin;
use crate::controller::jit_controller::TypeCache;
use tokio_util::sync::CancellationToken;

pub async fn static_jit_compile(jit_library: JitLibrary) -> bin::Library {
    static_jit_compile_impl(jit_library, false).await
}

/// Compile a static library with shared check/error types and per-instance bindings.
///
/// Uses the production controller's structural cache keys and avoids materializing
/// a separate large error-type declaration for every gadget instance.
pub async fn static_jit_compile_compact(jit_library: JitLibrary) -> bin::Library {
    static_jit_compile_impl(jit_library, true).await
}

async fn static_jit_compile_impl(mut jit_library: JitLibrary, compact: bool) -> bin::Library {
    let compiler = jit_compiler::JitCompiler::new();
    let program = std::mem::take(&mut jit_library.program);
    let token = CancellationToken::new();
    // copy the port types and gadget types from the JIT library
    let mut library = bin::Library {
        metadata: jit_library.metadata.clone(),
        ..Default::default()
    };
    for port_type in jit_library.port_types.iter() {
        library.port_types.push(port_type.base.as_ref().unwrap().clone());
    }
    for gadget_type in jit_library.gadget_types.iter() {
        library.gadget_types.push(gadget_type.base.as_ref().unwrap().clone());
    }
    compiler.load_library(jit_library).await;

    if program.is_empty() {
        return library;
    }

    // Check if all gids are pre-assigned (required for parallel compilation)
    let all_preassigned = program.iter().all(|i| i.gadget.as_ref().is_some_and(|g| g.gid != 0));

    if compact && all_preassigned && program_is_dependency_ordered(&program) {
        return static_jit_compile_ordered(compiler, program, token, library).await;
    }

    if !all_preassigned || program.len() < 64 {
        // Fall back to sequential compilation when gids are auto-assigned
        // or the program is small enough that spawn overhead dominates.
        return static_jit_compile_sequential(compiler, program, token, library, compact).await;
    }

    // Parallel compilation: spawn each gadget as a tokio task once its
    // input dependencies are compiled. Error model futures are spawned
    // immediately so they can resolve concurrently with later compilations.

    use std::collections::HashMap;
    use tokio::sync::watch;

    // Create a ready-signal channel for each gid
    let mut ready_txs: HashMap<u64, watch::Sender<bool>> = HashMap::new();

    struct TaskInfo {
        idx: usize,
        instruction: JitInstruction,
        gid: u64,
        dep_gids: Vec<u64>,
    }

    let mut tasks: Vec<TaskInfo> = Vec::with_capacity(program.len());
    for (idx, instruction) in program.into_iter().enumerate() {
        let gadget = instruction.gadget.as_ref().unwrap();
        let gid = gadget.gid;
        let dep_gids: Vec<u64> = gadget.connectors.iter().map(|c| c.gid).collect();
        let (tx, _) = watch::channel(false);
        ready_txs.insert(gid, tx);
        tasks.push(TaskInfo {
            idx,
            instruction,
            gid,
            dep_gids,
        });
    }

    // Subscribe to all dependency channels BEFORE moving senders into tasks,
    // so that all senders are still alive during subscription.
    let mut all_dep_rxs: Vec<Vec<watch::Receiver<bool>>> = Vec::with_capacity(tasks.len());
    for info in &tasks {
        let rxs: Vec<watch::Receiver<bool>> = info.dep_gids.iter().map(|d| ready_txs[d].subscribe()).collect();
        all_dep_rxs.push(rxs);
    }

    // Spawn compilation tasks
    let n = tasks.len();
    let mut handles = Vec::with_capacity(n);

    for (info, dep_rxs) in tasks.into_iter().zip(all_dep_rxs) {
        let ready_tx = ready_txs.remove(&info.gid).unwrap();
        let comp = std::sync::Arc::clone(&compiler);
        let tok = token.clone();

        handles.push(tokio::spawn(async move {
            // Wait for all input gadgets to finish compiling
            for mut rx in dep_rxs {
                let _ = rx.wait_for(|v| *v).await;
            }

            // Compile gadget (acquires write lock on gadgets, then releases)
            let (gadget, cmt, cm, error_future) = compile_for_export(&comp, info.instruction, tok, compact).await;

            // Signal that this gadget is compiled — dependents can proceed,
            // and input gadgets' error model futures can now resolve
            let _ = ready_tx.send(true);

            // Spawn error model resolution as a concurrent task so it can
            // overlap with later gadget compilations on other threads
            let error_handle = tokio::spawn(error_future);

            (info.idx, gadget, cmt, cm, error_handle)
        }));
    }

    // Collect all gadget compilation results
    let mut results: Vec<_> = futures_util::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // Sort by original program index to maintain deterministic output order
    results.sort_by_key(|(idx, _, _, _, _)| *idx);

    // Build library in program order
    let mut cache = TypeCache::new();
    let mut error_handles = Vec::with_capacity(n);
    for (_, gadget, cmt, mut cm, error_handle) in results {
        cm.terminal_error_model = None;
        library.program.push(bin::Instruction {
            create: Some(bin::instruction::Create::Gadget(gadget)),
        });
        if compact {
            cache_check_type(&mut library, &mut cache, cmt, &mut cm);
        } else {
            library.check_model_types.push(cmt);
        }
        library.program.push(bin::Instruction {
            create: Some(bin::instruction::Create::CheckModel(cm)),
        });
        error_handles.push(error_handle);
    }

    // Wait for all error model resolutions
    let error_results = futures_util::future::join_all(error_handles).await;
    for result in error_results {
        let (error_model_type, mut error_model) = result.unwrap();
        if compact {
            cache_error_type(&mut library, &mut cache, &error_model_type, &mut error_model);
        } else {
            library
                .error_model_types
                .push(std::sync::Arc::unwrap_or_clone(error_model_type));
        }
        library.program.push(bin::Instruction {
            create: Some(bin::instruction::Create::ErrorModel(error_model)),
        });
    }

    library
}

fn program_is_dependency_ordered(program: &[JitInstruction]) -> bool {
    let mut preceding = std::collections::HashSet::with_capacity(program.len());
    program.iter().all(|instruction| {
        let gadget = instruction.gadget.as_ref().unwrap();
        gadget.connectors.iter().all(|connector| preceding.contains(&connector.gid)) && preceding.insert(gadget.gid)
    })
}

async fn static_jit_compile_ordered(
    compiler: std::sync::Arc<jit_compiler::JitCompiler>,
    program: Vec<JitInstruction>,
    token: CancellationToken,
    mut library: bin::Library,
) -> bin::Library {
    use futures_util::{FutureExt as _, StreamExt as _, stream::FuturesOrdered};
    use tokio_util::task::AbortOnDropHandle;

    let mut cache = TypeCache::new();
    let mut pending_errors = FuturesOrdered::new();
    let mut errors = Vec::with_capacity(program.len());
    for (index, instruction) in program.into_iter().enumerate() {
        let (gadget, check_type, mut check, error_future) =
            compile_for_export(&compiler, instruction, token.clone(), true).await;
        let gid = gadget.gid;
        check.terminal_error_model = None;
        cache_check_type(&mut library, &mut cache, check_type, &mut check);
        library.program.push(bin::Instruction {
            create: Some(bin::instruction::Create::Gadget(gadget)),
        });
        library.program.push(bin::Instruction {
            create: Some(bin::instruction::Create::CheckModel(check)),
        });
        let error_compiler = std::sync::Arc::clone(&compiler);
        pending_errors.push_back(AbortOnDropHandle::new(tokio::spawn(async move {
            let result = error_future.await;
            error_compiler.mark_error_model_ready(gid).await;
            result
        })));
        // Poll unresolved errors without waiting for successors that are still uncompiled.
        while let Some(Some(result)) = pending_errors.next().now_or_never() {
            let (definition, mut model) = result.expect("error model export task failed");
            cache_error_type(&mut library, &mut cache, &definition, &mut model);
            errors.push(bin::Instruction {
                create: Some(bin::instruction::Create::ErrorModel(model)),
            });
        }
        if (index + 1).is_multiple_of(256) {
            compiler.collect_garbage(compiler.gadgets.write().await).await;
        }
    }
    while let Some(result) = pending_errors.next().await {
        let (definition, mut model) = result.expect("error model export task failed");
        cache_error_type(&mut library, &mut cache, &definition, &mut model);
        errors.push(bin::Instruction {
            create: Some(bin::instruction::Create::ErrorModel(model)),
        });
    }
    compiler.collect_garbage(compiler.gadgets.write().await).await;
    library.program.extend(errors);
    library
}

/// Sequential fallback for when gids are not pre-assigned.
async fn static_jit_compile_sequential(
    compiler: std::sync::Arc<jit_compiler::JitCompiler>,
    program: Vec<JitInstruction>,
    token: CancellationToken,
    mut library: bin::Library,
    compact: bool,
) -> bin::Library {
    let mut cache = TypeCache::new();
    let mut error_model_futures = vec![];
    for instruction in program {
        let (gadget, check_model_type, mut check_model, error_model_future) =
            compile_for_export(&compiler, instruction, token.clone(), compact).await;
        check_model.terminal_error_model = None;
        error_model_futures.push(error_model_future);
        library.program.push(bin::Instruction {
            create: Some(bin::instruction::Create::Gadget(gadget)),
        });
        if compact {
            cache_check_type(&mut library, &mut cache, check_model_type, &mut check_model);
        } else {
            library.check_model_types.push(check_model_type);
        }
        library.program.push(bin::Instruction {
            create: Some(bin::instruction::Create::CheckModel(check_model)),
        });
    }
    let error_models = futures_util::future::join_all(error_model_futures).await;
    for (error_model_type, mut error_model) in error_models {
        if compact {
            cache_error_type(&mut library, &mut cache, &error_model_type, &mut error_model);
        } else {
            library
                .error_model_types
                .push(std::sync::Arc::unwrap_or_clone(error_model_type));
        }
        library.program.push(bin::Instruction {
            create: Some(bin::instruction::Create::ErrorModel(error_model)),
        });
    }
    library
}

async fn compile_for_export(
    compiler: &std::sync::Arc<jit_compiler::JitCompiler>,
    instruction: JitInstruction,
    token: CancellationToken,
    compact: bool,
) -> (
    bin::Gadget,
    bin::CheckModelType,
    bin::CheckModel,
    futures_util::future::BoxFuture<'static, (std::sync::Arc<bin::ErrorModelType>, bin::ErrorModel)>,
) {
    if compact {
        let (gadget, check_type, check, errors) = compiler.compile_shared(instruction, token).await;
        (gadget, check_type, check, Box::pin(errors))
    } else {
        let (gadget, check_type, check, errors) = compiler.compile(instruction, token).await;
        (
            gadget,
            check_type,
            check,
            Box::pin(async move {
                let (definition, model) = errors.await;
                (std::sync::Arc::new(definition), model)
            }),
        )
    }
}

fn cache_check_type(
    library: &mut bin::Library,
    cache: &mut TypeCache,
    definition: bin::CheckModelType,
    model: &mut bin::CheckModel,
) {
    model.modifier = crate::controller::jit_controller::build_check_model_modifier(&definition);
    model.ctype = if let Some(existing) = cache.get_check_model_type(&definition) {
        existing
    } else {
        let id = definition.ctype;
        cache.insert_check_model_type(&definition, id);
        library.check_model_types.push(definition);
        id
    };
}

fn cache_error_type(
    library: &mut bin::Library,
    cache: &mut TypeCache,
    definition: &bin::ErrorModelType,
    model: &mut bin::ErrorModel,
) {
    model.modifier = crate::controller::jit_controller::build_error_model_modifier(definition, model);
    model.etype = if let Some(existing) = cache.get_error_model_type(definition) {
        existing
    } else {
        let mut definition = definition.clone();
        definition.etype = model.etype;
        cache.insert_error_model_type(&definition, definition.etype);
        library.error_model_types.push(definition);
        model.etype
    };
}

/// Input the serialized JitLibrary, output independently materialized types.
///
/// Releases the GIL during compilation and periodically polls
/// ``Python::check_signals`` from the main Python thread so that Ctrl+C
/// (``KeyboardInterrupt``) is honored: Rust/PyO3 bindings are not
/// naturally interruptible — a long running native call keeps the GIL
/// and blocks Python's signal handler from running, so a hang or
/// expensive compile would otherwise be uninterruptible by the user.
///
/// ``PyErr_CheckSignals`` only processes signal handlers when invoked
/// from the main interpreter thread, so the actual compilation runs on
/// a background OS thread while this function (still on the main
/// thread) periodically wakes up to check for pending signals.  When a
/// signal is detected, the cancellation token is fired so the tokio
/// tasks driving ``static_jit_compile`` unblock and the worker thread
/// exits.
#[cfg(feature = "python_binding")]
#[pyo3::pyfunction]
#[pyo3(name="static_jit_compile", signature = (jit_library))]
pub fn py_static_jit_compile(py: pyo3::Python<'_>, jit_library: Vec<u8>) -> pyo3::PyResult<Vec<u8>> {
    py_static_jit_compile_impl(py, jit_library, false)
}

/// Input the serialized JitLibrary, output shared types with per-instance bindings.
///
/// Compilation uses the same interruptible worker as [`py_static_jit_compile`].
#[cfg(feature = "python_binding")]
#[pyo3::pyfunction]
#[pyo3(name="static_jit_compile_compact", signature = (jit_library))]
pub fn py_static_jit_compile_compact(py: pyo3::Python<'_>, jit_library: Vec<u8>) -> pyo3::PyResult<Vec<u8>> {
    py_static_jit_compile_impl(py, jit_library, true)
}

#[cfg(feature = "python_binding")]
fn py_static_jit_compile_impl(py: pyo3::Python<'_>, jit_library: Vec<u8>, compact: bool) -> pyo3::PyResult<Vec<u8>> {
    use prost::Message;
    use std::sync::mpsc::{RecvTimeoutError, sync_channel};
    use std::time::Duration;
    let jit_library = JitLibrary::decode(&*jit_library).unwrap();

    let cancel = CancellationToken::new();
    let cancel_for_worker = cancel.clone();
    let (tx, rx) = sync_channel::<bin::Library>(1);
    let worker = std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let library = rt.block_on(async move {
            tokio::select! {
                library = static_jit_compile_impl(jit_library, compact) => Some(library),
                _ = cancel_for_worker.cancelled() => None,
            }
        });
        if let Some(library) = library {
            let _ = tx.send(library);
        }
        // Drop the runtime to abort any still-running tasks (e.g.
        // dependency-wait awaits in the parallel path that don't
        // observe the cancellation token directly).
        drop(rt);
    });

    let outcome = loop {
        // ``recv_timeout`` blocks the calling thread (this is the main
        // Python thread, holding the GIL) for up to 50ms.  Holding the
        // GIL for that long is acceptable: ``check_signals`` requires
        // the main thread anyway, the cadence is short enough that
        // Ctrl+C still feels responsive, and no other Python thread can
        // make progress while the JIT runtime is the only thing
        // happening regardless.
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(library) => break Ok(library),
            Err(RecvTimeoutError::Disconnected) => {
                // Worker exited without sending a result — it must have
                // observed cancellation. Surface the pending signal
                // (typically ``KeyboardInterrupt``) if any.
                py.check_signals()?;
                break Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "static_jit_compile worker exited unexpectedly",
                ));
            }
            Err(RecvTimeoutError::Timeout) => {
                if let Err(err) = py.check_signals() {
                    cancel.cancel();
                    // Drain any final result the worker may have
                    // produced before observing the cancel.
                    let _ = rx.recv();
                    break Err(err);
                }
            }
        }
    };

    let _ = worker.join();

    let library = outcome?;
    let mut buf = Vec::with_capacity(library.encoded_len());
    library.encode(&mut buf).unwrap();
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn abandoned_ordered_export_aborts_unresolved_error_tasks() {
        use futures_util::FutureExt as _;

        let compiler = jit_compiler::JitCompiler::new();
        compiler
            .load_library(JitLibrary {
                port_types: vec![JitPortType {
                    base: Some(bin::PortType {
                        ptype: 1,
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                gadget_types: vec![JitGadgetType {
                    base: Some(bin::GadgetType {
                        gtype: 1,
                        outputs: vec![bin::gadget_type::Port {
                            ptype: 1,
                            ..Default::default()
                        }],
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            })
            .await;
        let weak = std::sync::Arc::downgrade(&compiler);
        let mut export = Box::pin(static_jit_compile_ordered(
            std::sync::Arc::clone(&compiler),
            vec![JitInstruction {
                gadget: Some(bin::Gadget {
                    gid: 1,
                    gtype: 1,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            CancellationToken::new(),
            bin::Library::default(),
        ));
        assert!(export.as_mut().now_or_never().is_none());
        assert!(compiler.contains_gid(1).await);
        drop(export);
        drop(compiler);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while weak.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("abandoned error tasks retained the compiler");
    }

    #[tokio::test]
    async fn ordered_compact_export_retires_state_and_matches_legacy_export() {
        let source = JitLibrary {
            gadget_types: vec![JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 1,
                    measurements: vec![bin::gadget_type::Measurement::default()],
                    ..Default::default()
                }),
                finished_checks: vec![jit_gadget_type::Check {
                    base: Some(bin::check_model_type::Check::default()),
                    measurements: vec![jit_gadget_type::PresentMeasurement::default()],
                }],
                errors: vec![jit_gadget_type::Error {
                    base: Some(bin::error_model_type::Error {
                        probability: 0.125,
                        ..Default::default()
                    }),
                    finished_checks: vec![0],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let program: Vec<_> = (1..=600)
            .map(|gid| JitInstruction {
                gadget: Some(bin::Gadget {
                    gid,
                    gtype: 1,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .collect();
        let compiler = jit_compiler::JitCompiler::new();
        compiler.load_library(source.clone()).await;
        let legacy = jit_compiler::JitCompiler::new();
        legacy.load_library(source).await;
        let expected = static_jit_compile_sequential(
            legacy,
            program.clone(),
            CancellationToken::new(),
            bin::Library::default(),
            true,
        )
        .await;
        let actual = static_jit_compile_ordered(
            std::sync::Arc::clone(&compiler),
            program,
            CancellationToken::new(),
            bin::Library::default(),
        )
        .await;
        assert_eq!(actual, expected);
        assert!(compiler.gadgets.read().await.is_empty());
    }

    #[tokio::test]
    async fn static_compile_preserves_metadata() {
        let metadata = prost_types::Struct {
            fields: [(
                "source".to_string(),
                prost_types::Value {
                    kind: Some(prost_types::value::Kind::StringValue("unit-test".to_string())),
                },
            )]
            .into(),
        };
        let library = static_jit_compile(JitLibrary {
            metadata: Some(metadata.clone()),
            ..Default::default()
        })
        .await;

        assert_eq!(library.metadata, Some(metadata));
    }
}
