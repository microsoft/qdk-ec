use super::*;

#[tokio::test]
async fn zero_error_template_capacity_preserves_models_without_caching() {
    let source = Arc::new(jit::JitGadgetType {
        base: Some(bin::GadgetType {
            gtype: 7,
            ..Default::default()
        }),
        errors: vec![jit::jit_gadget_type::Error {
            base: Some(bin::error_model_type::Error {
                probability: 0.125,
                residual: vec![2],
                readout_flips: vec![1],
                ..Default::default()
            }),
            finished_checks: vec![0],
            ..Default::default()
        }],
        ..Default::default()
    });
    let mut models = vec![];
    for capacity in [DEFAULT_ERROR_TEMPLATE_CACHE_CAPACITY, 1, 0] {
        let compiler = JitCompiler::with_error_template_cache_capacity(capacity);
        let key = || ErrorTemplateKey {
            gtype: 7,
            output_checks: vec![],
        };
        let first = compiler.error_template(&source, key()).await;
        let second = compiler.error_template(&source, key()).await;
        assert_eq!(Arc::ptr_eq(&first, &second), capacity > 0);
        assert_eq!(first.model_type, second.model_type);
        if let Some(cache) = &compiler.error_templates {
            let cache = cache.lock().await;
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.cap().get(), capacity);
        } else {
            assert_eq!(capacity, 0);
        }
        models.push(Arc::clone(&first.model_type));
        compiler.reset_library().await;
        if let Some(cache) = &compiler.error_templates {
            assert!(cache.lock().await.is_empty());
        }
    }
    assert_eq!(models[0], models[1]);
    assert_eq!(models[0], models[2]);
}

#[tokio::test]
async fn error_template_capacity_controls_lru_eviction() {
    let compiler = JitCompiler::with_error_template_cache_capacity(2);
    let sources: Vec<_> = (1..=3)
        .map(|gtype| {
            Arc::new(jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype,
                    ..Default::default()
                }),
                ..Default::default()
            })
        })
        .collect();
    let key = |gtype| ErrorTemplateKey {
        gtype,
        output_checks: vec![],
    };
    let first = compiler.error_template(&sources[0], key(1)).await;
    let second = compiler.error_template(&sources[1], key(2)).await;
    assert!(Arc::ptr_eq(&first, &compiler.error_template(&sources[0], key(1)).await));
    compiler.error_template(&sources[2], key(3)).await;
    assert!(Arc::ptr_eq(&first, &compiler.error_template(&sources[0], key(1)).await));
    assert!(!Arc::ptr_eq(&second, &compiler.error_template(&sources[1], key(2)).await));
    assert_eq!(compiler.error_templates.as_ref().unwrap().lock().await.len(), 2);
}

#[test]
fn controller_forwards_error_template_cache_capacity() {
    for capacity in [0, 1, DEFAULT_ERROR_TEMPLATE_CACHE_CAPACITY] {
        let controller = crate::controller::JitController::new(serde_json::json!({
            "error_template_cache_capacity": capacity
        }));
        let actual = controller
            .compiler
            .error_templates
            .as_ref()
            .map_or(0, |cache| cache.try_lock().unwrap().cap().get());
        assert_eq!(actual, capacity);
    }
}

#[tokio::test]
async fn incremental_collection_matches_every_four_vertex_dag_and_root_set() {
    let ids = [40, 1, 90, 7];
    let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for edge_mask in 0_u32..64 {
        for root_mask in 0_u32..16 {
            let compiler = JitCompiler::new();
            let mut graph = compiler.gadgets.write().await;
            for (index, &gid) in ids.iter().enumerate() {
                let targets: Vec<_> = edges
                    .iter()
                    .enumerate()
                    .filter_map(|(bit, &(source, target))| {
                        (source == index && edge_mask & (1 << bit) != 0).then_some(ids[target])
                    })
                    .collect();
                let rooted = root_mask & (1 << index) != 0;
                let outputs = targets
                    .iter()
                    .map(|&gid| watch::channel(Some(bin::gadget::Connector { gid, port: 0 })).0)
                    .collect();
                graph.insert(
                    gid,
                    JitGadgetState {
                        error_model_ready: !rooted,
                        retention_count: usize::from(rooted),
                        gtype: 0,
                        outputs,
                        output_checks: vec![vec![]; targets.len()],
                        input_virtual_check_map: Arc::new(vec![]),
                        output_virtual_checks: watch::channel(Some(Arc::new(vec![
                            targets.into_iter().map(|cid| ExplicitCheck { cid, check_index: 0 }).collect(),
                        ])))
                        .0,
                    },
                );
            }
            for (bit, &(_, target)) in edges.iter().enumerate() {
                if edge_mask & (1 << bit) != 0 {
                    JitCompiler::retain_reference(&mut graph, ids[target]);
                    JitCompiler::retain_reference(&mut graph, ids[target]);
                }
            }
            compiler.collection_candidates.lock().unwrap().extend(
                graph
                    .iter()
                    .filter_map(|(&gid, gadget)| (gadget.retention_count == 0).then_some(gid)),
            );
            compiler.collect_garbage(graph).await;
            let mut reachable = root_mask;
            for source in 0..4 {
                if reachable & (1 << source) != 0 {
                    for (bit, &(from, target)) in edges.iter().enumerate() {
                        if from == source && edge_mask & (1 << bit) != 0 {
                            reachable |= 1 << target;
                        }
                    }
                }
            }
            for (index, &gid) in ids.iter().enumerate() {
                let expected = reachable & (1 << index) != 0;
                assert_eq!(compiler.contains_gid(gid).await, expected);
                assert_eq!(compiler.was_collected(gid).await, !expected);
            }
        }
    }
}

#[test]
fn normalized_error_templates_preserve_absolute_checks_and_error_effects() {
    let source = Arc::new(jit::JitGadgetType {
        base: Some(bin::GadgetType {
            gtype: 7,
            ..Default::default()
        }),
        errors: vec![
            jit::jit_gadget_type::Error {
                base: Some(bin::error_model_type::Error {
                    probability: 0.125,
                    residual: vec![2],
                    readout_flips: vec![1],
                    ..Default::default()
                }),
                finished_checks: vec![0],
                unfinished_checks: vec![0, 1],
            },
            jit::jit_gadget_type::Error {
                base: Some(bin::error_model_type::Error {
                    probability: 0.25,
                    ..Default::default()
                }),
                unfinished_checks: vec![0, 2],
                ..Default::default()
            },
            jit::jit_gadget_type::Error {
                base: Some(bin::error_model_type::Error {
                    probability: 0.375,
                    ..Default::default()
                }),
                finished_checks: vec![3],
                ..Default::default()
            },
        ],
        ..Default::default()
    });
    let checks: Vec<HashSet<_>> = [
        [(20, 5), (30, 10)],
        [(20, 5), (30, 11)],
        [(20, 7), (30, 10)],
        [(20, 1), (30, 2)],
    ]
    .into_iter()
    .map(|checks| {
        checks
            .into_iter()
            .map(|(cid, check_index)| ExplicitCheck { cid, check_index })
            .collect()
    })
    .collect();
    let (key, sources) = normalize_error_context(7, 10, &checks);
    let template = build_error_template(&source, &key);
    for (input, output) in source.errors.iter().zip(&template.model_type.errors) {
        let restored: HashSet<_> = output
            .checks
            .iter()
            .map(|check| {
                if let Some(remote) = check.remote_check_model {
                    let (source, bias) = template.remote_bindings[usize::try_from(remote).unwrap()];
                    ExplicitCheck {
                        cid: sources[source].0,
                        check_index: check.check_index + sources[source].1 + bias,
                    }
                } else {
                    ExplicitCheck {
                        cid: 10,
                        check_index: check.check_index,
                    }
                }
            })
            .collect();
        assert_eq!(restored, expand_error_checks(input, &checks, 10));
        let base = input.base.as_ref().unwrap();
        assert_eq!(output.probability.to_bits(), base.probability.to_bits());
        assert_eq!(output.residual, base.residual);
        assert_eq!(output.readout_flips, base.readout_flips);
    }
    let shifted: Vec<HashSet<_>> = checks
        .iter()
        .map(|checks| {
            checks
                .iter()
                .map(|check| ExplicitCheck {
                    cid: check.cid + 100,
                    check_index: check.check_index + 10,
                })
                .collect()
        })
        .collect();
    let (shifted_key, shifted_sources) = normalize_error_context(7, 110, &shifted);
    assert!(key == shifted_key);
    assert_ne!(sources, shifted_sources);
}
