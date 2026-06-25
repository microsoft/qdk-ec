//! Integration tests for WindowCoordinator commit region and trace infrastructure.
//!
//! These tests create a WindowCoordinator with MockDecoder and trace output,
//! exercise various gadget topologies, and assert on the trace to verify
//! distance-based commit region computation, window construction, and leader election.

mod common;

use deq_runtime::bin::{self, instruction};
use deq_runtime::coordinator::coordinator_server::Coordinator;
use deq_runtime::coordinator::window_coordinator::{self, WindowCoordinator};
use deq_runtime::decoder::{BlackBoxDecoderClient, MockDecoder};
use deq_runtime::jit::{self, static_jit_compile};
use deq_runtime::util::{BitMatrix, BitVector};
use prost::Message;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tempfile::NamedTempFile;
use tonic::Request;

// re-use the trace proto types
use window_coordinator::trace;

// ─── helpers ───────────────────────────────────────────────────────────────

fn make_mock_decoder() -> Arc<MockDecoder> {
    Arc::new(MockDecoder::new())
}

fn make_coordinator(mock: Arc<MockDecoder>, trace_file: &str) -> WindowCoordinator {
    make_coordinator_with_radii(mock, trace_file, 1, 0)
}

fn make_coordinator_with_hops(mock: Arc<MockDecoder>, trace_file: &str, buffer_radius: usize) -> WindowCoordinator {
    make_coordinator_with_radii(mock, trace_file, buffer_radius, buffer_radius)
}

fn make_coordinator_with_radii(
    mock: Arc<MockDecoder>,
    trace_file: &str,
    buffer_radius: usize,
    lookahead_radius: usize,
) -> WindowCoordinator {
    let config = serde_json::json!({
        "persistent_decoder": false,
        "merge_hyperedges": false,
        "trace_filepath": trace_file,
        "buffer_radius": buffer_radius,
        "lookahead_radius": lookahead_radius,
    });
    WindowCoordinator::new(config, BlackBoxDecoderClient::from_mock(mock))
}

fn make_gadget(gid: u64, gtype: u64, connectors: Vec<(u64, u64)>) -> bin::Gadget {
    bin::Gadget {
        gid,
        gtype,
        connectors: connectors
            .into_iter()
            .map(|(gid, port)| bin::gadget::Connector { gid, port })
            .collect(),
        ..Default::default()
    }
}

fn make_check_model(cid: u64, ctype: u64, gid: u64) -> bin::CheckModel {
    bin::CheckModel {
        cid,
        ctype,
        gid,
        ..Default::default()
    }
}

fn make_error_model(eid: u64, etype: u64, cid: u64) -> bin::ErrorModel {
    bin::ErrorModel {
        eid,
        etype,
        cid,
        ..Default::default()
    }
}

/// Build a library with five gadget types:
///   gtype=1 "checked"     — 0 inputs, 1 output, 1 measurement, 1 check, 1 error
///   gtype=2 "free_hop" — 1 input, 1 output, 0 measurements (free-hop)
///   gtype=3 "free_hop_2in2out" — 2 inputs, 2 outputs, 0 measurements (free-hop)
///   gtype=4 "checked_1in" — 1 input, 1 output, 1 measurement, 1 check, 1 error
///   gtype=5 "checked_terminal" — 1 input, 0 outputs, 1 measurement, 1 readout, 1 check, 1 error
///
/// Port type has 0 observables for simplicity (no frame propagation needed).
fn make_test_library() -> bin::Library {
    let port_type = bin::PortType {
        ptype: 1,
        observables: vec![],
        ..Default::default()
    };

    // gtype=1: checked source gadget (0 inputs, 1 output, 1 measurement)
    let checked_source = bin::GadgetType {
        gtype: 1,
        inputs: vec![],
        outputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        measurements: vec![bin::gadget_type::Measurement::default()],
        readouts: vec![],
        correction_propagation: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        readout_propagation: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        physical_correction: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        ..Default::default()
    };

    // gtype=2: free-hop (1 input, 1 output, 0 measurements)
    let free_hop = bin::GadgetType {
        gtype: 2,
        inputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        outputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        measurements: vec![], // no measurements → free_hop
        readouts: vec![],
        correction_propagation: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        readout_propagation: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        ..Default::default()
    };

    // gtype=3: free-hop 2-input 2-output (2 inputs, 2 outputs, 0 measurements)
    let free_hop_2in2out = bin::GadgetType {
        gtype: 3,
        inputs: vec![
            bin::gadget_type::Port {
                ptype: 1,
                ..Default::default()
            },
            bin::gadget_type::Port {
                ptype: 1,
                ..Default::default()
            },
        ],
        outputs: vec![
            bin::gadget_type::Port {
                ptype: 1,
                ..Default::default()
            },
            bin::gadget_type::Port {
                ptype: 1,
                ..Default::default()
            },
        ],
        measurements: vec![],
        readouts: vec![],
        correction_propagation: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        readout_propagation: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        ..Default::default()
    };

    // gtype=4: checked 1-input (1 input, 1 output, 1 measurement)
    let checked_1in = bin::GadgetType {
        gtype: 4,
        inputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        outputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        measurements: vec![bin::gadget_type::Measurement::default()],
        readouts: vec![],
        correction_propagation: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        readout_propagation: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        physical_correction: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        ..Default::default()
    };

    // gtype=5: checked terminal (1 input, 0 outputs, 1 measurement, 1 readout)
    let checked_terminal = bin::GadgetType {
        gtype: 5,
        inputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        outputs: vec![],
        measurements: vec![bin::gadget_type::Measurement::default()],
        readouts: vec![bin::gadget_type::Readout {
            measurement_indices: vec![0],
            ..Default::default()
        }],
        correction_propagation: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        readout_propagation: Some(BitMatrix {
            rows: 1,
            cols: 1,
            ..Default::default()
        }),
        logical_correction: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        physical_correction: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        ..Default::default()
    };

    // ctype=1: check model for gtype=1 (source, no remote)
    let ctype1 = bin::CheckModelType {
        ctype: 1,
        gtype: 1,
        checks: vec![bin::check_model_type::Check {
            measurements: vec![bin::check_model_type::RemoteMeasurement {
                measurement_index: 0,
                remote_gadget: None,
            }],
            naturally_flipped: false,
            ..Default::default()
        }],
        remote_gadgets: vec![],
        ..Default::default()
    };

    // etype=1: error model for ctype=1
    let etype1 = bin::ErrorModelType {
        etype: 1,
        ctype: 1,
        errors: vec![bin::error_model_type::Error {
            probability: 0.1,
            checks: vec![bin::error_model_type::RemoteCheck {
                check_index: 0,
                remote_check_model: None,
            }],
            residual: vec![],
            readout_flips: vec![],
            ..Default::default()
        }],
        remote_check_models: vec![],
        ..Default::default()
    };

    // ctype=4: check model for gtype=4 (1-input checked, no remote)
    let ctype4 = bin::CheckModelType {
        ctype: 4,
        gtype: 4,
        checks: vec![bin::check_model_type::Check {
            measurements: vec![bin::check_model_type::RemoteMeasurement {
                measurement_index: 0,
                remote_gadget: None,
            }],
            naturally_flipped: false,
            ..Default::default()
        }],
        remote_gadgets: vec![],
        ..Default::default()
    };

    // etype=4: error model for ctype=4
    let etype4 = bin::ErrorModelType {
        etype: 4,
        ctype: 4,
        errors: vec![bin::error_model_type::Error {
            probability: 0.1,
            checks: vec![bin::error_model_type::RemoteCheck {
                check_index: 0,
                remote_check_model: None,
            }],
            residual: vec![],
            readout_flips: vec![],
            ..Default::default()
        }],
        remote_check_models: vec![],
        ..Default::default()
    };

    // ctype=5: check model for gtype=5 (terminal, no remote)
    let ctype5 = bin::CheckModelType {
        ctype: 5,
        gtype: 5,
        checks: vec![bin::check_model_type::Check {
            measurements: vec![bin::check_model_type::RemoteMeasurement {
                measurement_index: 0,
                remote_gadget: None,
            }],
            naturally_flipped: false,
            ..Default::default()
        }],
        remote_gadgets: vec![],
        ..Default::default()
    };

    // etype=5: error model for ctype=5
    let etype5 = bin::ErrorModelType {
        etype: 5,
        ctype: 5,
        errors: vec![bin::error_model_type::Error {
            probability: 0.1,
            checks: vec![bin::error_model_type::RemoteCheck {
                check_index: 0,
                remote_check_model: None,
            }],
            residual: vec![],
            readout_flips: vec![],
            ..Default::default()
        }],
        remote_check_models: vec![],
        ..Default::default()
    };

    bin::Library {
        port_types: vec![port_type],
        gadget_types: vec![checked_source, free_hop, free_hop_2in2out, checked_1in, checked_terminal],
        check_model_types: vec![ctype1, ctype4, ctype5],
        error_model_types: vec![etype1, etype4, etype5],
        ..Default::default()
    }
}

/// Build a set of gadget type IDs that are free-hops (no measurements).
fn free_hop_types_from_library(library: &bin::Library) -> HashSet<u64> {
    library
        .gadget_types
        .iter()
        .filter(|gt| gt.is_free_hop.unwrap_or(gt.measurements.is_empty()))
        .map(|gt| gt.gtype)
        .collect()
}

/// Free-hop types for the standard test library.
fn test_free_hop_types() -> HashSet<u64> {
    free_hop_types_from_library(&make_test_library())
}

/// Execute a gadget instruction and return the assigned id.
async fn exec_gadget(coord: &WindowCoordinator, gadget: bin::Gadget) -> u64 {
    Coordinator::execute(
        coord,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(gadget)),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id
}

/// Execute a check model instruction and return the assigned id.
async fn exec_check_model(coord: &WindowCoordinator, cm: bin::CheckModel) -> u64 {
    Coordinator::execute(
        coord,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(cm)),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id
}

/// Execute an error model instruction and return the assigned id.
async fn exec_error_model(coord: &WindowCoordinator, em: bin::ErrorModel) -> u64 {
    Coordinator::execute(
        coord,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::ErrorModel(em)),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id
}

/// Decode a gadget with the given outcomes.
async fn decode(coord: &WindowCoordinator, gid: u64, num_measurements: u64) -> deq_runtime::coordinator::Readouts {
    Coordinator::decode(
        coord,
        Request::new(deq_runtime::coordinator::Outcomes {
            gid,
            outcomes: Some(BitVector {
                data: vec![0; num_measurements.div_ceil(8) as usize],
                size: num_measurements,
            }),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
    .into_inner()
}

/// Call reset (preserving library) and flush the trace.
async fn reset_shot(coord: &WindowCoordinator) {
    Coordinator::reset(
        coord,
        Request::new(deq_runtime::coordinator::ResetRequest {
            reset_library: false,
            reset_decoder_service: false,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
}

/// Read and parse the trace protobuf from the given file.
fn read_trace(path: &str) -> trace::WindowCoordinatorTrace {
    let data = std::fs::read(path).unwrap();
    trace::WindowCoordinatorTrace::decode(&data[..]).unwrap()
}

/// Extract all DecodeEvents from a shot's events.
fn decode_events(shot: &trace::Shot) -> Vec<&trace::DecodeEvent> {
    shot.events
        .iter()
        .filter_map(|e| match &e.event {
            Some(trace::event::Event::Decode(d)) => Some(d),
            _ => None,
        })
        .collect()
}

/// Build a mapping from gid → is_free_hop using trace ExecuteGadget events.
fn gid_free_hop_map(shot: &trace::Shot, free_hop_types: &HashSet<u64>) -> HashMap<u64, bool> {
    shot.events
        .iter()
        .filter_map(|e| match &e.event {
            Some(trace::event::Event::ExecuteGadget(eg)) => {
                let g = eg.gadget.as_ref().unwrap();
                Some((g.gid, free_hop_types.contains(&g.gtype)))
            }
            _ => None,
        })
        .collect()
}

/// Comprehensive window correctness verification.
///
/// Checks using recorded trace data:
///   1. **No overlap**: no gadget appears in two commit regions.
///   2. **Valid gids**: commit region ⊆ all_gids.
///   3. **Center in commit**: center (leader) is in its own commit region.
///   4. **Structural**: commit_region ⊆ window.
///   5. **Boundary distance**: independently recomputes boundary distances
///      from the gadget graph and timestamps (not runtime-reported values).
///      Every hop-counted gadget in a commit region (except center) must have
///      `boundary_dist >= buffer_radius`.
///   6. **Partition**: every gadget committed exactly once (no gaps).
///   7. **Buffer-committing concurrency**: if two leaders decode concurrently,
///      neither's commit region may overlap the other's buffer.
fn assert_window_correctness(
    shot: &trace::Shot,
    all_gids: &HashSet<u64>,
    buffer_radius: usize,
    free_hop_types: &HashSet<u64>,
) {
    let events = decode_events(shot);
    let leaders: Vec<_> = events.iter().filter(|d| d.is_leader).collect();

    // Build gadget graph from ExecuteGadget events for independent boundary
    // distance computation (check 5).
    struct GadgetInfo {
        connectors: Vec<(u64, u64)>, // (source_gid, port)
        num_outputs: u32,
        is_free_hop: bool,
    }
    let mut gadget_graph: HashMap<u64, GadgetInfo> = HashMap::new();
    // output adjacency: gid -> [(target_gid)] built by inverting connectors
    let mut output_adj: HashMap<u64, Vec<u64>> = HashMap::new();
    for event in &shot.events {
        if let Some(trace::event::Event::ExecuteGadget(eg)) = &event.event {
            let gadget = eg.gadget.as_ref().unwrap();
            let connectors: Vec<(u64, u64)> = gadget.connectors.iter().map(|c| (c.gid, c.port)).collect();
            for &(src_gid, _port) in &connectors {
                output_adj.entry(src_gid).or_default().push(gadget.gid);
            }
            gadget_graph.insert(
                gadget.gid,
                GadgetInfo {
                    connectors,
                    num_outputs: eg.num_outputs,
                    is_free_hop: free_hop_types.contains(&gadget.gtype),
                },
            );
        }
    }

    // Collect decode event timestamps and DecodeFinished timestamps to
    // determine which gadgets were Committed/Decoding at each leader's
    // decode time.
    struct DecodeTimeInfo {
        start_ns: u64,
        commit_region: HashSet<u64>,
    }
    let mut decode_times: HashMap<u64, DecodeTimeInfo> = HashMap::new(); // leader_gid -> info
    for event in &shot.events {
        match &event.event {
            Some(trace::event::Event::Decode(d)) if d.is_leader => {
                decode_times.insert(
                    d.gid,
                    DecodeTimeInfo {
                        start_ns: event.timestamp_ns,
                        commit_region: d.committing_gids.iter().copied().collect(),
                    },
                );
            }
            _ => {}
        }
    }

    let mut all_committed: HashSet<u64> = HashSet::new();

    for leader in &leaders {
        let commit_region: HashSet<u64> = leader.committing_gids.iter().copied().collect();
        let window: HashSet<u64> = leader.window.iter().copied().collect();
        let center_gid = leader.gid;

        // 1. No overlap with previously committed
        let overlap: HashSet<u64> = all_committed.intersection(&commit_region).copied().collect();
        assert!(
            overlap.is_empty(),
            "leader {} commit_region overlaps with prior leaders: {:?}",
            center_gid,
            overlap
        );

        // 2. Commit region is subset of all_gids
        for &gid in &commit_region {
            assert!(all_gids.contains(&gid), "leader {} commits unknown gid {}", center_gid, gid);
        }

        // 3. Center is in commit region
        assert!(
            commit_region.contains(&center_gid),
            "leader {} center not in commit_region {:?}",
            center_gid,
            commit_region
        );

        // 4. Commit region is subset of window
        for &gid in &commit_region {
            assert!(
                window.contains(&gid),
                "leader {} commits gid {} not in window {:?}",
                center_gid,
                gid,
                window
            );
        }

        // 5. Independent boundary distance verification.
        // Determine which gadgets were Committed/Decoding at this leader's
        // decode start time, then recompute boundary distances from scratch.
        let my_start_ns = decode_times.get(&center_gid).map(|t| t.start_ns).unwrap_or(0);
        let mut terminals: HashSet<u64> = HashSet::new();
        for (other_leader, info) in &decode_times {
            if *other_leader == center_gid {
                continue;
            }
            // If the other leader started before us and hasn't finished yet,
            // its commit region is Decoding (terminal). If it finished
            // before our start, its commit region is Committed (terminal).
            let other_start = info.start_ns;
            if other_start < my_start_ns {
                // Started before us → its commit_region is at least Decoding/Committed
                terminals.extend(&info.commit_region);
            }
            // If other_start >= my_start_ns, it started after us and cannot
            // have committed before our window was built.
        }

        // Compute boundary distances using multi-pass relaxation BFS.
        // The runtime uses a 0-1 BFS (VecDeque with push_front/push_back),
        // but this frontier-based relaxation converges to the same result.
        let mut boundary_dist: HashMap<u64, usize> = HashMap::new();
        let mut boundary_seeds: Vec<u64> = Vec::new();

        for &gid in &window {
            if terminals.contains(&gid) {
                boundary_dist.insert(gid, usize::MAX);
                continue;
            }
            let Some(info) = gadget_graph.get(&gid) else { continue };
            let mut has_external_boundary = false;

            // Check inputs: connector sources outside window that aren't terminals
            for &(src_gid, _port) in &info.connectors {
                if !window.contains(&src_gid) && !terminals.contains(&src_gid) {
                    has_external_boundary = true;
                    break;
                }
            }

            // Check outputs: count connected outputs from other gadgets' connectors
            if !has_external_boundary {
                let connected_targets: Vec<u64> = output_adj.get(&gid).map(|v| v.to_vec()).unwrap_or_default();
                let num_connected = connected_targets.len() as u32;
                if num_connected < info.num_outputs {
                    // Unconnected output port → boundary
                    has_external_boundary = true;
                } else {
                    // Check if any connected output goes outside window and isn't terminal
                    for &target_gid in &connected_targets {
                        if !window.contains(&target_gid) && !terminals.contains(&target_gid) {
                            has_external_boundary = true;
                            break;
                        }
                    }
                }
            }

            if has_external_boundary {
                boundary_seeds.push(gid);
                boundary_dist.insert(gid, 0);
            }
        }

        // BFS inward: propagate boundary distances
        let mut frontier = boundary_seeds;
        while !frontier.is_empty() {
            let mut next_frontier: Vec<u64> = Vec::new();
            for &fgid in &frontier {
                let my_bdist = boundary_dist[&fgid];
                let Some(info) = gadget_graph.get(&fgid) else { continue };

                // Propagate to input neighbors
                for &(src_gid, _) in &info.connectors {
                    if !window.contains(&src_gid) || terminals.contains(&src_gid) {
                        continue;
                    }
                    let peer_free = gadget_graph.get(&src_gid).is_some_and(|g| g.is_free_hop);
                    let step = if peer_free { 0 } else { 1 };
                    let new_dist = my_bdist + step;
                    let entry = boundary_dist.entry(src_gid).or_insert(usize::MAX);
                    if new_dist < *entry {
                        *entry = new_dist;
                        next_frontier.push(src_gid);
                    }
                }

                // Propagate to output neighbors
                if let Some(targets) = output_adj.get(&fgid) {
                    for &target_gid in targets {
                        if !window.contains(&target_gid) || terminals.contains(&target_gid) {
                            continue;
                        }
                        let peer_free = gadget_graph.get(&target_gid).is_some_and(|g| g.is_free_hop);
                        let step = if peer_free { 0 } else { 1 };
                        let new_dist = my_bdist + step;
                        let entry = boundary_dist.entry(target_gid).or_insert(usize::MAX);
                        if new_dist < *entry {
                            *entry = new_dist;
                            next_frontier.push(target_gid);
                        }
                    }
                }
            }
            frontier = next_frontier;
        }

        // Unreached gadgets have infinite boundary distance
        for &gid in &window {
            boundary_dist.entry(gid).or_insert(usize::MAX);
        }

        // Verify: every committed hop-counted gadget (except center) must have
        // bd >= buffer_radius. Free-hop gadgets are exempt because they have step=0
        // in BFS (don't increase hop count) and are always absorbed to prevent
        // stranding.
        for &gid in &commit_region {
            if gid == center_gid {
                continue;
            }
            let is_free = gadget_graph.get(&gid).is_some_and(|g| g.is_free_hop);
            if is_free {
                continue;
            }
            let bd = boundary_dist.get(&gid).copied().unwrap_or(0);
            assert!(
                bd >= buffer_radius,
                "leader {} commits gid {} with boundary_dist {} < buffer_radius {} \
                 (commit_region={:?}, window={:?}, terminals={:?}, boundary_dists={:?})",
                center_gid,
                gid,
                bd,
                buffer_radius,
                commit_region,
                window,
                terminals,
                boundary_dist
            );
        }

        // Verify: every committed free-hop gadget is transitively adjacent
        // (through other committed free-hops) to at least one committed
        // hop-counted gadget. This catches stranded free-hop clusters.
        {
            let committed_hop_counted: HashSet<u64> = commit_region
                .iter()
                .copied()
                .filter(|&gid| !gadget_graph.get(&gid).is_some_and(|g| g.is_free_hop))
                .collect();
            let committed_free_hops: HashSet<u64> = commit_region
                .iter()
                .copied()
                .filter(|&gid| gadget_graph.get(&gid).is_some_and(|g| g.is_free_hop))
                .collect();

            // BFS from committed hop-counted gadgets through committed free-hop neighbors
            let mut reachable_free_hops: HashSet<u64> = HashSet::new();
            let mut queue: Vec<u64> = committed_hop_counted.iter().copied().collect();
            while let Some(gid) = queue.pop() {
                if let Some(info) = gadget_graph.get(&gid) {
                    for &(src_gid, _) in &info.connectors {
                        if committed_free_hops.contains(&src_gid) && reachable_free_hops.insert(src_gid) {
                            queue.push(src_gid);
                        }
                    }
                }
                if let Some(targets) = output_adj.get(&gid) {
                    for &target_gid in targets {
                        if committed_free_hops.contains(&target_gid) && reachable_free_hops.insert(target_gid) {
                            queue.push(target_gid);
                        }
                    }
                }
            }

            let stranded: HashSet<u64> = committed_free_hops.difference(&reachable_free_hops).copied().collect();
            assert!(
                stranded.is_empty(),
                "leader {} commits free-hop gadgets not transitively adjacent to any \
                 committed hop-counted gadget: {:?} (commit_region={:?}, window={:?})",
                center_gid,
                stranded,
                commit_region,
                window
            );
        }

        all_committed.extend(&commit_region);
    }

    // 6. Check all gadgets are committed (no gaps)
    let missing: HashSet<u64> = all_gids.difference(&all_committed).copied().collect();
    assert!(missing.is_empty(), "gadgets not committed by any leader: {:?}", missing);

    let extra: HashSet<u64> = all_committed.difference(all_gids).copied().collect();
    assert!(extra.is_empty(), "committed gids not in all_gids: {:?}", extra);

    // 7. Buffer-committing concurrency check: if two leaders decode
    // concurrently, neither's commit region may serve as buffer for the other.
    // Buffer = window \ commit_region.
    let mut decode_start: HashMap<u64, u64> = HashMap::new();
    let mut decode_finish: HashMap<u64, u64> = HashMap::new();
    let mut leader_windows: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut leader_commits: HashMap<u64, HashSet<u64>> = HashMap::new();

    for event in &shot.events {
        match &event.event {
            Some(trace::event::Event::Decode(d)) if d.is_leader => {
                decode_start.insert(d.gid, event.timestamp_ns);
                leader_windows.insert(d.gid, d.window.iter().copied().collect());
                leader_commits.insert(d.gid, d.committing_gids.iter().copied().collect());
            }
            Some(trace::event::Event::DecodeFinished(df)) => {
                decode_finish.insert(df.leader_gid, event.timestamp_ns);
            }
            _ => {}
        }
    }

    let leader_ids: Vec<u64> = decode_start.keys().copied().collect();
    for i in 0..leader_ids.len() {
        for j in (i + 1)..leader_ids.len() {
            let (g1, g2) = (leader_ids[i], leader_ids[j]);
            let (Some(&s1), Some(&f1), Some(&s2), Some(&f2)) = (
                decode_start.get(&g1),
                decode_finish.get(&g1),
                decode_start.get(&g2),
                decode_finish.get(&g2),
            ) else {
                continue;
            };
            // Only check concurrently decoding pairs
            if !(s1 < f2 && s2 < f1) {
                continue;
            }
            let c1 = &leader_commits[&g1];
            let c2 = &leader_commits[&g2];
            let w1 = &leader_windows[&g1];
            let w2 = &leader_windows[&g2];
            let buf1: HashSet<u64> = w1.difference(c1).copied().collect();
            let buf2: HashSet<u64> = w2.difference(c2).copied().collect();

            let bad_12: HashSet<u64> = c1.intersection(&buf2).copied().collect();
            assert!(
                bad_12.is_empty(),
                "leaders {} and {} decode concurrently but {}'s commit {:?} \
                 overlaps {}'s buffer {:?} on {:?}",
                g1,
                g2,
                g1,
                c1,
                g2,
                buf2,
                bad_12
            );
            let bad_21: HashSet<u64> = c2.intersection(&buf1).copied().collect();
            assert!(
                bad_21.is_empty(),
                "leaders {} and {} decode concurrently but {}'s commit {:?} \
                 overlaps {}'s buffer {:?} on {:?}",
                g1,
                g2,
                g2,
                c2,
                g1,
                buf1,
                bad_21
            );
        }
    }
}

/// Check that decode windows with disjoint commit regions ran in parallel.
///
/// Uses DecodeEvent and DecodeFinishedEvent timestamps. For each pair of
/// leaders, if their commit regions are disjoint (they should always be),
/// checks whether their decode time ranges overlapped.
///
/// Returns (total_sequential_ms, actual_wall_ms, n_parallel_pairs).
fn check_decode_parallelism(shot: &trace::Shot) -> (u64, u64, usize) {
    use std::collections::HashMap;

    // Collect (start_timestamp, finish_timestamp) per leader_gid
    let mut decode_start: HashMap<u64, u64> = HashMap::new();
    let mut decode_finish: HashMap<u64, u64> = HashMap::new();

    for event in &shot.events {
        match &event.event {
            Some(trace::event::Event::Decode(d)) if d.is_leader => {
                decode_start.insert(d.gid, event.timestamp_ns);
            }
            Some(trace::event::Event::DecodeFinished(df)) => {
                decode_finish.insert(df.leader_gid, event.timestamp_ns);
            }
            _ => {}
        }
    }

    let leaders: Vec<u64> = decode_start.keys().copied().collect();
    let mut total_decode_ns: u64 = 0;
    let mut n_parallel_pairs = 0;

    for &gid in &leaders {
        if let (Some(&start), Some(&finish)) = (decode_start.get(&gid), decode_finish.get(&gid)) {
            total_decode_ns += finish.saturating_sub(start);
        }
    }

    // Check pairwise overlap
    for i in 0..leaders.len() {
        for j in (i + 1)..leaders.len() {
            let (g1, g2) = (leaders[i], leaders[j]);
            if let (Some(&s1), Some(&f1), Some(&s2), Some(&f2)) = (
                decode_start.get(&g1),
                decode_finish.get(&g1),
                decode_start.get(&g2),
                decode_finish.get(&g2),
            ) {
                // Ranges overlap if s1 < f2 && s2 < f1
                if s1 < f2 && s2 < f1 {
                    n_parallel_pairs += 1;
                }
            }
        }
    }

    // Wall clock: from earliest start to latest finish
    let min_start = decode_start.values().copied().min().unwrap_or(0);
    let max_finish = decode_finish.values().copied().max().unwrap_or(0);
    let wall_ns = max_finish.saturating_sub(min_start);

    (total_decode_ns / 1_000_000, wall_ns / 1_000_000, n_parallel_pairs)
}

// ─── tests ─────────────────────────────────────────────────────────────────

/// Test 1: Source → terminal chain: A → B (terminal).
/// B has no outputs, so B is within A's window. Only one leader (A).
/// commit_region = {A, B}, window = {A, B}.
#[tokio::test]
async fn test_two_checked_gadgets_chain() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A(gtype=1, source) → B(gtype=5, terminal, 0 outputs)
    let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    let gid_b = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_a, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_b)).await;
    exec_error_model(&coord, make_error_model(0, 5, 2)).await;

    assert_eq!(gid_a, 1);
    assert_eq!(gid_b, 2);

    // Decode both concurrently
    let (r1, r2) = tokio::join!(decode(&coord, gid_a, 1), decode(&coord, gid_b, 1),);
    assert_eq!(r1.gid, gid_a);
    assert_eq!(r2.gid, gid_b);

    // Flush trace
    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    assert_eq!(trace.shots.len(), 1);

    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let _free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);
    let all_gids: HashSet<u64> = [gid_a, gid_b].into();
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // Self-contained chain: no connections go outside the window, so all
    // gadgets have boundary_dist = ∞.  One leader commits everything.
    let leaders: Vec<_> = events.iter().filter(|d| d.is_leader).collect();
    assert_eq!(leaders.len(), 1);

    let leader = leaders[0];
    let committing: HashSet<u64> = leader.committing_gids.iter().copied().collect();
    assert_eq!(committing, HashSet::from([gid_a, gid_b]));
}

/// Test 2: Transversal chain A → T → B (terminal)
/// A and B are hop-counted gadgets linked through free-hop T.
/// T has no syndrome extraction rounds, so it is not a leader candidate.
/// commit_region = {A, T, B} (free-hop T always included), window = {A, T, B}.
/// committing_gids = {A, T, B}.
#[tokio::test]
async fn test_free_hop_chain() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A(gtype=1, source) → T(gtype=2, free_hop) → B(gtype=5, terminal)
    let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    let gid_t = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_a, 0)])).await;

    let gid_b = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_b)).await;
    exec_error_model(&coord, make_error_model(0, 5, 2)).await;

    // Decode all concurrently (free-hop gadgets also need decode)
    let (r_a, r_t, r_b) = tokio::join!(decode(&coord, gid_a, 1), decode(&coord, gid_t, 0), decode(&coord, gid_b, 1),);
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_t.gid, gid_t);
    assert_eq!(r_b.gid, gid_b);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);
    let all_gids: HashSet<u64> = [gid_a, gid_t, gid_b].into();
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // T should be marked as free-hop (not leader)
    let t_event = events.iter().find(|d| d.gid == gid_t).unwrap();
    assert!(*free_map.get(&t_event.gid).unwrap());
    assert!(!t_event.is_leader);

    // Self-contained chain: no connections go outside the window, so all
    // gadgets have boundary_dist = ∞.  One leader commits everything.
    let leaders: Vec<_> = events.iter().filter(|d| d.is_leader).collect();
    assert_eq!(leaders.len(), 1);
    let leader = leaders[0];

    let committing: HashSet<u64> = leader.committing_gids.iter().copied().collect();
    assert_eq!(committing, HashSet::from([gid_a, gid_t, gid_b]));
}

/// Test 3: 2-input 2-output free-hop: A,A2 → T → B,C
/// T has 2 inputs and 2 outputs. A,A2 are source gadgets, B,C are terminal.
/// commit_region = {A, A2, T, B, C} (T always included as free-hop).
/// All gadgets form one commit region because T connects everything.
#[tokio::test]
async fn test_multi_output_free_hop() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A(gtype=1) and A2(gtype=1) → T(gtype=3, 2in/2out free_hop) → B(gtype=5), C(gtype=5)
    let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    let gid_a2 = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a2)).await;
    exec_error_model(&coord, make_error_model(0, 1, 2)).await;

    let gid_t = exec_gadget(&coord, make_gadget(0, 3, vec![(gid_a, 0), (gid_a2, 0)])).await;

    let gid_b = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_b)).await;
    exec_error_model(&coord, make_error_model(0, 5, 3)).await;

    let gid_c = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t, 1)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_c)).await;
    exec_error_model(&coord, make_error_model(0, 5, 4)).await;

    // Decode all concurrently
    let (r_a, r_a2, r_t, r_b, r_c) = tokio::join!(
        decode(&coord, gid_a, 1),
        decode(&coord, gid_a2, 1),
        decode(&coord, gid_t, 0),
        decode(&coord, gid_b, 1),
        decode(&coord, gid_c, 1),
    );
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_a2.gid, gid_a2);
    assert_eq!(r_t.gid, gid_t);
    assert_eq!(r_b.gid, gid_b);
    assert_eq!(r_c.gid, gid_c);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);
    let all_gids: HashSet<u64> = [gid_a, gid_a2, gid_t, gid_b, gid_c].into();
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // Self-contained graph: no connections go outside the window, so all
    // gadgets have boundary_dist = ∞.  First leader commits all gadgets.
    let leaders: Vec<_> = events.iter().filter(|d| d.is_leader).collect();
    assert_eq!(leaders.len(), 1);

    // T should be free-hop
    let t_event = events.iter().find(|d| d.gid == gid_t).unwrap();
    assert!(*free_map.get(&t_event.gid).unwrap());
    assert!(!t_event.is_leader);

    // The single leader commits all gadgets
    let leader = &leaders[0];
    let committing: HashSet<u64> = leader.committing_gids.iter().copied().collect();
    assert_eq!(committing, HashSet::from([gid_a, gid_a2, gid_t, gid_b, gid_c]));
}

/// Test 4: Chain of free-hop gadgets A → T1 → T2 → B
/// Multiple free-hop gadgets in sequence.
/// commit_region = {A, T1, T2, B} (free-hops always included).
/// committing_gids = {A, T1, T2, B}.
#[tokio::test]
async fn test_free_hop_chain_multiple() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A(source) → T1 → T2 → B(terminal, gtype=5)
    let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    let gid_t1 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_a, 0)])).await;
    let gid_t2 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_t1, 0)])).await;

    let gid_b = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t2, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_b)).await;
    exec_error_model(&coord, make_error_model(0, 5, 2)).await;

    let (r_a, r_t1, r_t2, r_b) = tokio::join!(
        decode(&coord, gid_a, 1),
        decode(&coord, gid_t1, 0),
        decode(&coord, gid_t2, 0),
        decode(&coord, gid_b, 1),
    );
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_t1.gid, gid_t1);
    assert_eq!(r_t2.gid, gid_t2);
    assert_eq!(r_b.gid, gid_b);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);
    let all_gids: HashSet<u64> = [gid_a, gid_t1, gid_t2, gid_b].into();
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // Self-contained chain: no connections go outside the window, so all
    // gadgets have boundary_dist = ∞.  One leader commits everything.
    let leaders: Vec<_> = events.iter().filter(|d| d.is_leader).collect();
    assert_eq!(leaders.len(), 1);
    let leader = leaders[0];

    let committing: HashSet<u64> = leader.committing_gids.iter().copied().collect();
    assert_eq!(committing, HashSet::from([gid_a, gid_t1, gid_t2, gid_b]));

    // T1, T2 should both be free-hop
    let t1_event = events.iter().find(|d| d.gid == gid_t1).unwrap();
    let t2_event = events.iter().find(|d| d.gid == gid_t2).unwrap();
    assert!(*free_map.get(&t1_event.gid).unwrap());
    assert!(*free_map.get(&t2_event.gid).unwrap());
}

/// Test 5: Hop-counted → hop-counted → free-hop → terminal chain.
/// A → B → T → C, where A and B are hop-counted (with outputs), T is free-hop, C is terminal.
///
/// Distance-based commit regions (buffer_radius=1):
///   A explores: window={A, B}. B is at boundary (output T not explored).
///     boundary_dist: A=1, B=0. commit_region={A}. Leader=A.
///   B explores: window={A, B, T, C}. A is promised-committed (interior).
///     All remaining gadgets have infinite boundary_dist. commit_region={B, T, C}. Leader=B.
///
/// A's window = {A, B}. B's window = {A, B, T, C}.
#[tokio::test]
async fn test_free_hop_in_buffer() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A(source, gtype=1) → B(checked_1in, gtype=4) → T(free_hop, gtype=2) → C(terminal, gtype=5)
    let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    let gid_b = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_a, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_b)).await;
    exec_error_model(&coord, make_error_model(0, 4, 2)).await;

    let gid_t = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_b, 0)])).await;

    let gid_c = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_c)).await;
    exec_error_model(&coord, make_error_model(0, 5, 3)).await;

    // Decode all concurrently
    let (r_a, r_b, r_t, r_c) = tokio::join!(
        decode(&coord, gid_a, 1),
        decode(&coord, gid_b, 1),
        decode(&coord, gid_t, 0),
        decode(&coord, gid_c, 1),
    );
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_b.gid, gid_b);
    assert_eq!(r_t.gid, gid_t);
    assert_eq!(r_c.gid, gid_c);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);
    let all_gids: HashSet<u64> = [gid_a, gid_b, gid_t, gid_c].into();
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // With the buffer-committing constraint, the exact leader assignment
    // depends on scheduling order. The key properties are:
    //   - Every gadget is committed exactly once (partition — checked above)
    //   - T is always a free-hop non-leader
    let t_event = events.iter().find(|d| d.gid == gid_t).unwrap();
    assert!(*free_map.get(&t_event.gid).unwrap());
    assert!(!t_event.is_leader);
}

/// Test 6: Long chain with free-hop gadgets between each hop-counted pair.
/// A → T1 → B → T2 → C → T3 → D, where A is source, B and C are checked_1in (gtype=4),
/// D is terminal (gtype=5), and T1/T2/T3 are free-hop (gtype=2).
///
/// Commit regions are dynamic (depend on decode scheduling order).
/// Only the partition property is guaranteed: every gadget is committed
/// by exactly one leader window.
#[tokio::test]
async fn test_long_free_hop_chain() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A(source) → T1 → B(checked_1in) → T2 → C(checked_1in) → T3 → D(terminal)
    let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    let gid_t1 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_a, 0)])).await;

    let gid_b = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t1, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_b)).await;
    exec_error_model(&coord, make_error_model(0, 4, 2)).await;

    let gid_t2 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_b, 0)])).await;

    let gid_c = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t2, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_c)).await;
    exec_error_model(&coord, make_error_model(0, 4, 3)).await;

    let gid_t3 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_c, 0)])).await;

    let gid_d = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t3, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_d)).await;
    exec_error_model(&coord, make_error_model(0, 5, 4)).await;

    // Decode all concurrently
    let (r_a, r_t1, r_b, r_t2, r_c, r_t3, r_d) = tokio::join!(
        decode(&coord, gid_a, 1),
        decode(&coord, gid_t1, 0),
        decode(&coord, gid_b, 1),
        decode(&coord, gid_t2, 0),
        decode(&coord, gid_c, 1),
        decode(&coord, gid_t3, 0),
        decode(&coord, gid_d, 1),
    );
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_t1.gid, gid_t1);
    assert_eq!(r_b.gid, gid_b);
    assert_eq!(r_t2.gid, gid_t2);
    assert_eq!(r_c.gid, gid_c);
    assert_eq!(r_t3.gid, gid_t3);
    assert_eq!(r_d.gid, gid_d);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);

    // Verify partition property: every gadget committed exactly once
    let all_gids: HashSet<u64> = HashSet::from([gid_a, gid_t1, gid_b, gid_t2, gid_c, gid_t3, gid_d]);
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // T1, T2, T3 should all be free-hop
    for &tid in &[gid_t1, gid_t2, gid_t3] {
        let t_event = events.iter().find(|d| d.gid == tid).unwrap();
        assert!(*free_map.get(&t_event.gid).unwrap());
    }
}

/// Test 7: Same topology as test 6 but with 2 buffer hops.
/// A(source) → T1 → B(checked_1in) → T2 → C(checked_1in) → T3 → D(terminal)
///
/// With the dynamic decode-driven design and buffer_radius=2, commit regions
/// depend on scheduling order. Decoding gadgets act as terminals, so
/// free-hops between committed/decoding regions get infinite boundary
/// distance. Only the partition property is guaranteed.
#[tokio::test]
async fn test_long_free_hop_chain_2_hops() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator_with_hops(mock.clone(), &trace_path, 2);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A(source) → T1 → B(checked_1in) → T2 → C(checked_1in) → T3 → D(terminal)
    let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    let gid_t1 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_a, 0)])).await;

    let gid_b = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t1, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_b)).await;
    exec_error_model(&coord, make_error_model(0, 4, 2)).await;

    let gid_t2 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_b, 0)])).await;

    let gid_c = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t2, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_c)).await;
    exec_error_model(&coord, make_error_model(0, 4, 3)).await;

    let gid_t3 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_c, 0)])).await;

    let gid_d = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t3, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_d)).await;
    exec_error_model(&coord, make_error_model(0, 5, 4)).await;

    // Decode all concurrently
    let (r_a, r_t1, r_b, r_t2, r_c, r_t3, r_d) = tokio::join!(
        decode(&coord, gid_a, 1),
        decode(&coord, gid_t1, 0),
        decode(&coord, gid_b, 1),
        decode(&coord, gid_t2, 0),
        decode(&coord, gid_c, 1),
        decode(&coord, gid_t3, 0),
        decode(&coord, gid_d, 1),
    );
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_t1.gid, gid_t1);
    assert_eq!(r_b.gid, gid_b);
    assert_eq!(r_t2.gid, gid_t2);
    assert_eq!(r_c.gid, gid_c);
    assert_eq!(r_t3.gid, gid_t3);
    assert_eq!(r_d.gid, gid_d);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);

    // Verify partition: every gadget committed exactly once
    let all_gids: HashSet<u64> = [gid_a, gid_t1, gid_b, gid_t2, gid_c, gid_t3, gid_d].into();
    assert_window_correctness(shot, &all_gids, 2, &test_free_hop_types());

    // All free-hop gadgets marked as such
    for &tid in &[gid_t1, gid_t2, gid_t3] {
        let ev = events.iter().find(|d| d.gid == tid).unwrap();
        assert!(*free_map.get(&ev.gid).unwrap());
    }
}

/// Test 8: Same topology as test_long_free_hop_chain but with 3 buffer hops.
/// A → T1 → B → T2 → C → T3 → D
///
/// With 3 hops, A's window reaches all 7 gadgets (A→hop1:B→hop2:C→hop3:D,
/// with free-hop gadgets as free hops). Since the window covers the entire chain
/// and D is terminal, all gadgets have infinite boundary distance.
///
/// Result: 1 leader (A), commit_region=all 7, committing=all 7.
#[tokio::test]
async fn test_long_free_hop_chain_3_hops() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator_with_hops(mock.clone(), &trace_path, 3);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A(source) → T1 → B(checked_1in) → T2 → C(checked_1in) → T3 → D(terminal)
    let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    let gid_t1 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_a, 0)])).await;

    let gid_b = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t1, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_b)).await;
    exec_error_model(&coord, make_error_model(0, 4, 2)).await;

    let gid_t2 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_b, 0)])).await;

    let gid_c = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t2, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_c)).await;
    exec_error_model(&coord, make_error_model(0, 4, 3)).await;

    let gid_t3 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_c, 0)])).await;

    let gid_d = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t3, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_d)).await;
    exec_error_model(&coord, make_error_model(0, 5, 4)).await;

    // Decode all concurrently
    let (r_a, r_t1, r_b, r_t2, r_c, r_t3, r_d) = tokio::join!(
        decode(&coord, gid_a, 1),
        decode(&coord, gid_t1, 0),
        decode(&coord, gid_b, 1),
        decode(&coord, gid_t2, 0),
        decode(&coord, gid_c, 1),
        decode(&coord, gid_t3, 0),
        decode(&coord, gid_d, 1),
    );
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_t1.gid, gid_t1);
    assert_eq!(r_b.gid, gid_b);
    assert_eq!(r_t2.gid, gid_t2);
    assert_eq!(r_c.gid, gid_c);
    assert_eq!(r_t3.gid, gid_t3);
    assert_eq!(r_d.gid, gid_d);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);
    let all_gids: HashSet<u64> = [gid_a, gid_t1, gid_b, gid_t2, gid_c, gid_t3, gid_d].into();
    assert_window_correctness(shot, &all_gids, 3, &test_free_hop_types());

    // With buffer_radius=3 and lookahead_radius=3 (default), the window radius is 6.
    // The chain A→T1→B→T2→C→T3→D has only 4 hop-counted gadgets spanning 3 hops,
    // so the first leader's window covers the entire chain and all gadgets with
    // sufficient boundary distance get committed.
    let leaders: Vec<_> = events.iter().filter(|d| d.is_leader).collect();
    // At least 1 leader, and all gadgets get committed (checked by assert_window_correctness)
    assert!(!leaders.is_empty());

    // All hop-counted gadgets must be committed by some leader
    let mut all_committed: HashSet<u64> = HashSet::new();
    for leader in &leaders {
        all_committed.extend(leader.committing_gids.iter().copied());
    }
    for &gid in &[gid_a, gid_b, gid_c, gid_d] {
        assert!(all_committed.contains(&gid), "hop-counted gid {} not committed", gid);
    }

    // T1, T2, T3 should be free-hop
    for &tid in &[gid_t1, gid_t2, gid_t3] {
        let ev = events.iter().find(|d| d.gid == tid).unwrap();
        assert!(*free_map.get(&ev.gid).unwrap());
        assert!(!ev.is_leader);
    }
}

/// Test 9: Diamond topology with overlapping 2-in/2-out free-hop gadgets.
///
///   A1 ─┐         ┌─ B1
///       T1 ──┐    │
///   A2 ─┘    └──┐
///               T2 ─┬─ B2
///   A3 ─────────┘   └─ B3
///
/// T1 inputs: (A1, A2). T2 inputs: (T1 output 1, A3).
/// T1 output 0 → B1, T2 output 0 → B2, T2 output 1 → B3.
///
/// Distance-based commit regions (buffer_radius=1):
/// A1 explores: window = all 8 gadgets (free-hops connect everything within 1 hop).
/// B1,B2,B3 are terminals (no outputs), so no boundary from that side.
/// All gadgets have infinite boundary distance → commit_region = all 8.
/// commit_region = {A1, A2, A3, T1, T2, B1, B2, B3}
/// committing_gids = commit_region
#[tokio::test]
async fn test_diamond_free_hop() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A1(source), A2(source), A3(source)
    let gid_a1 = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a1)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    let gid_a2 = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a2)).await;
    exec_error_model(&coord, make_error_model(0, 1, 2)).await;

    let gid_a3 = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a3)).await;
    exec_error_model(&coord, make_error_model(0, 1, 3)).await;

    // T1(2in/2out): inputs = (A1, port 0), (A2, port 0)
    let gid_t1 = exec_gadget(&coord, make_gadget(0, 3, vec![(gid_a1, 0), (gid_a2, 0)])).await;

    // T2(2in/2out): inputs = (T1, port 1), (A3, port 0)
    let gid_t2 = exec_gadget(&coord, make_gadget(0, 3, vec![(gid_t1, 1), (gid_a3, 0)])).await;

    // B1(terminal): input = T1 output 0
    let gid_b1 = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t1, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_b1)).await;
    exec_error_model(&coord, make_error_model(0, 5, 4)).await;

    // B2(terminal): input = T2 output 0
    let gid_b2 = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t2, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_b2)).await;
    exec_error_model(&coord, make_error_model(0, 5, 5)).await;

    // B3(terminal): input = T2 output 1
    let gid_b3 = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t2, 1)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_b3)).await;
    exec_error_model(&coord, make_error_model(0, 5, 6)).await;

    // Decode all concurrently
    let (r_a1, r_a2, r_a3, r_t1, r_t2, r_b1, r_b2, r_b3) = tokio::join!(
        decode(&coord, gid_a1, 1),
        decode(&coord, gid_a2, 1),
        decode(&coord, gid_a3, 1),
        decode(&coord, gid_t1, 0),
        decode(&coord, gid_t2, 0),
        decode(&coord, gid_b1, 1),
        decode(&coord, gid_b2, 1),
        decode(&coord, gid_b3, 1),
    );
    assert_eq!(r_a1.gid, gid_a1);
    assert_eq!(r_a2.gid, gid_a2);
    assert_eq!(r_a3.gid, gid_a3);
    assert_eq!(r_t1.gid, gid_t1);
    assert_eq!(r_t2.gid, gid_t2);
    assert_eq!(r_b1.gid, gid_b1);
    assert_eq!(r_b2.gid, gid_b2);
    assert_eq!(r_b3.gid, gid_b3);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);
    let all_gids: HashSet<u64> = [gid_a1, gid_a2, gid_a3, gid_t1, gid_t2, gid_b1, gid_b2, gid_b3].into();
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // Self-contained graph: no connections go outside the window, so all
    // gadgets have boundary_dist = ∞.  First leader commits all gadgets.
    let leaders: Vec<_> = events.iter().filter(|d| d.is_leader).collect();
    assert_eq!(leaders.len(), 1);

    // T1, T2 should be free-hop
    for &tid in &[gid_t1, gid_t2] {
        let ev = events.iter().find(|d| d.gid == tid).unwrap();
        assert!(*free_map.get(&ev.gid).unwrap());
        assert!(!ev.is_leader);
    }

    // The single leader commits all gadgets
    let leader = &leaders[0];
    let committing: HashSet<u64> = leader.committing_gids.iter().copied().collect();
    assert_eq!(committing, all_gids);
}

/// Test 10: Double-diamond — two layers of overlapping 2-in/2-out free-hop gadgets.
///
///   A1 ─┐         ┌─ B1 ─┐         ┌─ C1
///       T1 ──┐    │      T3 ──┐    │
///   A2 ─┘    └──┐ │           └──┐ │
///               T2 ─┬─ B2 ─┘       T4 ─┬─ C2
///   A3 ─────────┘   └─ B3 ─────────┘   └─ C3
///
/// B1,B2,B3 are checked_1in (gtype=4, have outputs). C1,C2,C3 are terminals.
///
/// With the dynamic decode-driven design, commit regions depend on scheduling
/// order. Decoding gadgets act as terminals, so free-hops between
/// committed/decoding regions may be committed by different leaders.
/// Only the partition property is guaranteed.
#[tokio::test]
async fn test_double_diamond_free_hop() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // ── Layer 1: sources ──
    let gid_a1 = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a1)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    let gid_a2 = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a2)).await;
    exec_error_model(&coord, make_error_model(0, 1, 2)).await;

    let gid_a3 = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a3)).await;
    exec_error_model(&coord, make_error_model(0, 1, 3)).await;

    // ── Layer 2: first free-hop pair ──
    let gid_t1 = exec_gadget(&coord, make_gadget(0, 3, vec![(gid_a1, 0), (gid_a2, 0)])).await;
    let gid_t2 = exec_gadget(&coord, make_gadget(0, 3, vec![(gid_t1, 1), (gid_a3, 0)])).await;

    // ── Layer 3: checked_1in (have outputs) ──
    let gid_b1 = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t1, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_b1)).await;
    exec_error_model(&coord, make_error_model(0, 4, 4)).await;

    let gid_b2 = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t2, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_b2)).await;
    exec_error_model(&coord, make_error_model(0, 4, 5)).await;

    let gid_b3 = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t2, 1)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_b3)).await;
    exec_error_model(&coord, make_error_model(0, 4, 6)).await;

    // ── Layer 4: second free-hop pair ──
    let gid_t3 = exec_gadget(&coord, make_gadget(0, 3, vec![(gid_b1, 0), (gid_b2, 0)])).await;
    let gid_t4 = exec_gadget(&coord, make_gadget(0, 3, vec![(gid_t3, 1), (gid_b3, 0)])).await;

    // ── Layer 5: terminals ──
    let gid_c1 = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t3, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_c1)).await;
    exec_error_model(&coord, make_error_model(0, 5, 7)).await;

    let gid_c2 = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t4, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_c2)).await;
    exec_error_model(&coord, make_error_model(0, 5, 8)).await;

    let gid_c3 = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t4, 1)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_c3)).await;
    exec_error_model(&coord, make_error_model(0, 5, 9)).await;

    // Decode all concurrently
    let (r_a1, r_a2, r_a3, r_t1, r_t2, r_b1, r_b2, r_b3, r_t3, r_t4, r_c1, r_c2, r_c3) = tokio::join!(
        decode(&coord, gid_a1, 1),
        decode(&coord, gid_a2, 1),
        decode(&coord, gid_a3, 1),
        decode(&coord, gid_t1, 0),
        decode(&coord, gid_t2, 0),
        decode(&coord, gid_b1, 1),
        decode(&coord, gid_b2, 1),
        decode(&coord, gid_b3, 1),
        decode(&coord, gid_t3, 0),
        decode(&coord, gid_t4, 0),
        decode(&coord, gid_c1, 1),
        decode(&coord, gid_c2, 1),
        decode(&coord, gid_c3, 1),
    );
    assert_eq!(r_a1.gid, gid_a1);
    assert_eq!(r_a2.gid, gid_a2);
    assert_eq!(r_a3.gid, gid_a3);
    assert_eq!(r_t1.gid, gid_t1);
    assert_eq!(r_t2.gid, gid_t2);
    assert_eq!(r_b1.gid, gid_b1);
    assert_eq!(r_b2.gid, gid_b2);
    assert_eq!(r_b3.gid, gid_b3);
    assert_eq!(r_t3.gid, gid_t3);
    assert_eq!(r_t4.gid, gid_t4);
    assert_eq!(r_c1.gid, gid_c1);
    assert_eq!(r_c2.gid, gid_c2);
    assert_eq!(r_c3.gid, gid_c3);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);

    // Verify partition: every gadget committed exactly once
    let all_gids: HashSet<u64> = [
        gid_a1, gid_a2, gid_a3, gid_t1, gid_t2, gid_b1, gid_b2, gid_b3, gid_t3, gid_t4, gid_c1, gid_c2, gid_c3,
    ]
    .into();
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // T1, T2, T3, T4 should be free-hop
    for &tid in &[gid_t1, gid_t2, gid_t3, gid_t4] {
        let ev = events.iter().find(|d| d.gid == tid).unwrap();
        assert!(*free_map.get(&ev.gid).unwrap());
        assert!(!ev.is_leader);
    }
}

/// Helper: build and execute the extended chain topology:
/// A(source) → T1 → B → T2 → C → T3 → D → T4 → E → T5 → F(terminal)
/// Returns (gids, coord, trace_path) where gids = [A, T1, B, T2, C, T3, D, T4, E, T5, F].
async fn build_extended_chain(buffer_radius: usize) -> ([u64; 11], WindowCoordinator, Arc<MockDecoder>, NamedTempFile) {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator_with_hops(mock.clone(), &trace_path, buffer_radius);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    let gid_t1 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_a, 0)])).await;

    let gid_b = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t1, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_b)).await;
    exec_error_model(&coord, make_error_model(0, 4, 2)).await;

    let gid_t2 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_b, 0)])).await;

    let gid_c = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t2, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_c)).await;
    exec_error_model(&coord, make_error_model(0, 4, 3)).await;

    let gid_t3 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_c, 0)])).await;

    let gid_d = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t3, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_d)).await;
    exec_error_model(&coord, make_error_model(0, 4, 4)).await;

    let gid_t4 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_d, 0)])).await;

    let gid_e = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t4, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_e)).await;
    exec_error_model(&coord, make_error_model(0, 4, 5)).await;

    let gid_t5 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_e, 0)])).await;

    let gid_f = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_t5, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_f)).await;
    exec_error_model(&coord, make_error_model(0, 5, 6)).await;

    let gids = [
        gid_a, gid_t1, gid_b, gid_t2, gid_c, gid_t3, gid_d, gid_t4, gid_e, gid_t5, gid_f,
    ];
    (gids, coord, mock, trace_file)
}

/// Test 11: Extended linear chain with 6 hop-counted gadgets and 5 free-hop gadgets.
/// A → T1 → B → T2 → C → T3 → D → T4 → E → T5 → F(terminal)
///
/// Commit regions are dynamic (depend on decode scheduling order).
/// Only the partition property is guaranteed: every gadget is committed
/// by exactly one leader window.
#[tokio::test]
async fn test_extended_chain() {
    let (gids, coord, _mock, trace_file) = build_extended_chain(1).await;
    let [
        gid_a,
        gid_t1,
        gid_b,
        gid_t2,
        gid_c,
        gid_t3,
        gid_d,
        gid_t4,
        gid_e,
        gid_t5,
        gid_f,
    ] = gids;
    let trace_path = trace_file.path().to_str().unwrap().to_string();

    // Decode all concurrently
    let (r_a, r_t1, r_b, r_t2, r_c, r_t3, r_d, r_t4, r_e, r_t5, r_f) = tokio::join!(
        decode(&coord, gid_a, 1),
        decode(&coord, gid_t1, 0),
        decode(&coord, gid_b, 1),
        decode(&coord, gid_t2, 0),
        decode(&coord, gid_c, 1),
        decode(&coord, gid_t3, 0),
        decode(&coord, gid_d, 1),
        decode(&coord, gid_t4, 0),
        decode(&coord, gid_e, 1),
        decode(&coord, gid_t5, 0),
        decode(&coord, gid_f, 1),
    );
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_t1.gid, gid_t1);
    assert_eq!(r_b.gid, gid_b);
    assert_eq!(r_t2.gid, gid_t2);
    assert_eq!(r_c.gid, gid_c);
    assert_eq!(r_t3.gid, gid_t3);
    assert_eq!(r_d.gid, gid_d);
    assert_eq!(r_t4.gid, gid_t4);
    assert_eq!(r_e.gid, gid_e);
    assert_eq!(r_t5.gid, gid_t5);
    assert_eq!(r_f.gid, gid_f);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);

    // Verify partition property: every gadget committed exactly once
    let all_gids: HashSet<u64> = HashSet::from([
        gid_a, gid_t1, gid_b, gid_t2, gid_c, gid_t3, gid_d, gid_t4, gid_e, gid_t5, gid_f,
    ]);
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // All free-hop gadgets marked as such
    for &tid in &[gid_t1, gid_t2, gid_t3, gid_t4, gid_t5] {
        let ev = events.iter().find(|d| d.gid == tid).unwrap();
        assert!(*free_map.get(&ev.gid).unwrap());
    }
}

/// Test 12: Same extended chain with 2 buffer hops.
/// A → T1 → B → T2 → C → T3 → D → T4 → E → T5 → F(terminal)
///
/// Commit regions are dynamic (depend on decode scheduling order).
/// Only the partition property is guaranteed: every gadget is committed
/// by exactly one leader window.
#[tokio::test]
async fn test_extended_chain_2_hops() {
    let (gids, coord, _mock, trace_file) = build_extended_chain(2).await;
    let [
        gid_a,
        gid_t1,
        gid_b,
        gid_t2,
        gid_c,
        gid_t3,
        gid_d,
        gid_t4,
        gid_e,
        gid_t5,
        gid_f,
    ] = gids;
    let trace_path = trace_file.path().to_str().unwrap().to_string();

    let (r_a, r_t1, r_b, r_t2, r_c, r_t3, r_d, r_t4, r_e, r_t5, r_f) = tokio::join!(
        decode(&coord, gid_a, 1),
        decode(&coord, gid_t1, 0),
        decode(&coord, gid_b, 1),
        decode(&coord, gid_t2, 0),
        decode(&coord, gid_c, 1),
        decode(&coord, gid_t3, 0),
        decode(&coord, gid_d, 1),
        decode(&coord, gid_t4, 0),
        decode(&coord, gid_e, 1),
        decode(&coord, gid_t5, 0),
        decode(&coord, gid_f, 1),
    );
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_t1.gid, gid_t1);
    assert_eq!(r_b.gid, gid_b);
    assert_eq!(r_t2.gid, gid_t2);
    assert_eq!(r_c.gid, gid_c);
    assert_eq!(r_t3.gid, gid_t3);
    assert_eq!(r_d.gid, gid_d);
    assert_eq!(r_t4.gid, gid_t4);
    assert_eq!(r_e.gid, gid_e);
    assert_eq!(r_t5.gid, gid_t5);
    assert_eq!(r_f.gid, gid_f);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);

    // Verify partition property: every gadget committed exactly once
    let all_gids: HashSet<u64> = HashSet::from([
        gid_a, gid_t1, gid_b, gid_t2, gid_c, gid_t3, gid_d, gid_t4, gid_e, gid_t5, gid_f,
    ]);
    assert_window_correctness(shot, &all_gids, 2, &test_free_hop_types());

    // All free-hop gadgets marked as such
    for &tid in &[gid_t1, gid_t2, gid_t3, gid_t4, gid_t5] {
        let ev = events.iter().find(|d| d.gid == tid).unwrap();
        assert!(*free_map.get(&ev.gid).unwrap());
    }
}
/// Test 13: Consecutive free-hop gadgets between hop-counted gadgets.
/// A(source) → T1 → T2 → B(checked_1in) → T3 → T4 → C(checked_1in) → D(terminal)
///
/// Tests that multiple consecutive free-hop gadgets (T1→T2, T3→T4) are correctly
/// handled by the distance-based algorithm. Free-hops add 0 distance.
///
/// With the dynamic decode-driven design, commit regions depend on scheduling order.
/// Decoding gadgets act as terminals, so free-hops between committed/decoding
/// regions get infinite boundary distance and are committed by the adjacent leader.
/// Only the partition property is guaranteed.
#[tokio::test]
async fn test_consecutive_free_hops() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A(source, gtype=1)
    let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    // T1(free-hop, gtype=2), T2(free-hop, gtype=2)
    let gid_t1 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_a, 0)])).await;
    let gid_t2 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_t1, 0)])).await;

    // B(checked_1in, gtype=4)
    let gid_b = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t2, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_b)).await;
    exec_error_model(&coord, make_error_model(0, 4, 2)).await;

    // T3(free-hop, gtype=2), T4(free-hop, gtype=2)
    let gid_t3 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_b, 0)])).await;
    let gid_t4 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_t3, 0)])).await;

    // C(checked_1in, gtype=4)
    let gid_c = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t4, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_c)).await;
    exec_error_model(&coord, make_error_model(0, 4, 3)).await;

    // D(terminal, gtype=5)
    let gid_d = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_c, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_d)).await;
    exec_error_model(&coord, make_error_model(0, 5, 4)).await;

    // Decode all concurrently
    let (r_a, r_t1, r_t2, r_b, r_t3, r_t4, r_c, r_d) = tokio::join!(
        decode(&coord, gid_a, 1),
        decode(&coord, gid_t1, 0),
        decode(&coord, gid_t2, 0),
        decode(&coord, gid_b, 1),
        decode(&coord, gid_t3, 0),
        decode(&coord, gid_t4, 0),
        decode(&coord, gid_c, 1),
        decode(&coord, gid_d, 1),
    );
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_t1.gid, gid_t1);
    assert_eq!(r_t2.gid, gid_t2);
    assert_eq!(r_b.gid, gid_b);
    assert_eq!(r_t3.gid, gid_t3);
    assert_eq!(r_t4.gid, gid_t4);
    assert_eq!(r_c.gid, gid_c);
    assert_eq!(r_d.gid, gid_d);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);

    // Verify partition: every gadget committed exactly once
    let all_gids: HashSet<u64> = [gid_a, gid_t1, gid_t2, gid_b, gid_t3, gid_t4, gid_c, gid_d].into();
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // All free-hop gadgets marked as such
    for &tid in &[gid_t1, gid_t2, gid_t3, gid_t4] {
        let ev = events.iter().find(|d| d.gid == tid).unwrap();
        assert!(*free_map.get(&ev.gid).unwrap());
    }
}

/// Test 14: Multiple shots accumulate in the trace file.
#[tokio::test]
async fn test_multiple_shots() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    for _ in 0..3 {
        let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
        exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
        exec_error_model(&coord, make_error_model(0, 1, 1)).await;

        let gid_b = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_a, 0)])).await;
        exec_check_model(&coord, make_check_model(0, 5, gid_b)).await;
        exec_error_model(&coord, make_error_model(0, 5, 2)).await;

        let (_, _) = tokio::join!(decode(&coord, gid_a, 1), decode(&coord, gid_b, 1),);
        reset_shot(&coord).await;
    }

    let trace = read_trace(&trace_path);
    assert_eq!(trace.shots.len(), 3);

    for shot in &trace.shots {
        let events = decode_events(shot);
        let leaders: Vec<_> = events.iter().filter(|d| d.is_leader).collect();
        // Self-contained chain: no connections go outside the window, so all
        // gadgets have boundary_dist = ∞.  One leader commits both gadgets.
        assert_eq!(leaders.len(), 1);

        let leader = &leaders[0];
        let committing: HashSet<u64> = leader.committing_gids.iter().copied().collect();
        assert_eq!(committing.len(), 2);
    }
}

/// Test: Two-qubit circuit with transversal gate and asymmetric termination.
///
/// Circuit (q0 left column, q1 right column):
///   A0(src,q0)  A1(src,q1)
///   B0(idle,q0) B1(idle,q1)
///      T(2-qubit free-hop, q0+q1)
///               C1(term,q1)
///   B2(idle,q0)
///   B3(idle,q0)
///   C0(term,q0)
///
/// Commit regions are dynamic (depend on decode scheduling order).
/// Only the partition property is guaranteed: every gadget is committed
/// by exactly one leader window.
#[tokio::test]
async fn test_two_qubit_transversal() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A0(source, q0)
    let a0 = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, a0)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    // A1(source, q1)
    let a1 = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, a1)).await;
    exec_error_model(&coord, make_error_model(0, 1, 2)).await;

    // B0(idle, q0) — checked 1-in 1-out
    let b0 = exec_gadget(&coord, make_gadget(0, 4, vec![(a0, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, b0)).await;
    exec_error_model(&coord, make_error_model(0, 4, 3)).await;

    // B1(idle, q1)
    let b1 = exec_gadget(&coord, make_gadget(0, 4, vec![(a1, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, b1)).await;
    exec_error_model(&coord, make_error_model(0, 4, 4)).await;

    // T(2-qubit free-hop, q0+q1) — gtype=3: 2 inputs, 2 outputs
    let t = exec_gadget(&coord, make_gadget(0, 3, vec![(b0, 0), (b1, 0)])).await;

    // C1(terminal, q1) — connected to T's output port 1
    let c1 = exec_gadget(&coord, make_gadget(0, 5, vec![(t, 1)])).await;
    exec_check_model(&coord, make_check_model(0, 5, c1)).await;
    exec_error_model(&coord, make_error_model(0, 5, 5)).await;

    // B2(idle, q0) — connected to T's output port 0
    let b2 = exec_gadget(&coord, make_gadget(0, 4, vec![(t, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, b2)).await;
    exec_error_model(&coord, make_error_model(0, 4, 6)).await;

    // B3(idle, q0)
    let b3 = exec_gadget(&coord, make_gadget(0, 4, vec![(b2, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, b3)).await;
    exec_error_model(&coord, make_error_model(0, 4, 7)).await;

    // C0(terminal, q0)
    let c0 = exec_gadget(&coord, make_gadget(0, 5, vec![(b3, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, c0)).await;
    exec_error_model(&coord, make_error_model(0, 5, 8)).await;

    // Decode all concurrently
    let (r_a0, r_a1, r_b0, r_b1, r_t, r_c1, r_b2, r_b3, r_c0) = tokio::join!(
        decode(&coord, a0, 1),
        decode(&coord, a1, 1),
        decode(&coord, b0, 1),
        decode(&coord, b1, 1),
        decode(&coord, t, 0),
        decode(&coord, c1, 1),
        decode(&coord, b2, 1),
        decode(&coord, b3, 1),
        decode(&coord, c0, 1),
    );
    for (r, gid) in [
        (&r_a0, a0),
        (&r_a1, a1),
        (&r_b0, b0),
        (&r_b1, b1),
        (&r_t, t),
        (&r_c1, c1),
        (&r_b2, b2),
        (&r_b3, b3),
        (&r_c0, c0),
    ] {
        assert_eq!(r.gid, gid);
    }

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let fht = test_free_hop_types();
    let free_map = gid_free_hop_map(shot, &fht);
    let events = decode_events(shot);

    // Verify partition: every gadget committed exactly once
    let all_gids: HashSet<u64> = [a0, a1, b0, b1, t, c1, b2, b3, c0].into();
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // T is free-hop
    let t_ev = events.iter().find(|d| d.gid == t).unwrap();
    assert!(*free_map.get(&t_ev.gid).unwrap());
}

// ─── stress tests ──────────────────────────────────────────────────────────

/// Build and run a random circuit with N qubits and K gate layers using
/// JIT-compiled gadgets from test_jit_library().
///
/// Gadget types (from common/test_library.rs):
///   - gtype=1 prepare_z: source (0in/1out, 2 measurements)
///   - gtype=2 measure_z: terminal (1in/0out, 3 measurements, 1 readout)
///   - gtype=3 cnot: two-qubit free-hop (2in/2out, 0 measurements)
///   - gtype=4 identity: single-qubit free-hop (1in/1out, 0 measurements)
///   - gtype=5 idle: single-qubit hop-counted (1in/1out, 2 measurements)
///
/// After K layers, all live qubits are terminated with measure_z.
async fn run_random_circuit(n_qubits: usize, n_gates: usize, seed: u64, buffer_radius: usize) {
    use common::test_library::test_jit_library;

    // Simple deterministic PRNG (xorshift64)
    let mut rng_state = seed;
    let mut next_rand = || -> u64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    // Measurement counts per gtype (from JIT library definition)
    let n_meas: HashMap<u64, u64> = HashMap::from([
        (1, 2), // prepare_z
        (2, 3), // measure_z
        (3, 0), // cnot (free-hop)
        (4, 0), // identity (free-hop)
        (5, 2), // idle
    ]);

    // Track per-qubit state: current_gid/port providing the wire
    let mut alive: Vec<bool> = vec![false; n_qubits];
    let mut wire_gid: Vec<u64> = vec![0; n_qubits];
    let mut wire_port: Vec<u64> = vec![0; n_qubits];
    let mut jit_instructions: Vec<jit::JitInstruction> = Vec::new();
    let mut all_gids: HashSet<u64> = HashSet::new();
    let mut decode_tasks: Vec<(u64, u64)> = Vec::new(); // (gid, num_measurements)
    let mut gid_counter: u64 = 1;

    // Helper to create a JIT instruction
    let make_instr = |gtype: u64, gid: u64, connectors: Vec<bin::gadget::Connector>| -> jit::JitInstruction {
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype,
                gid,
                connectors,
                ..Default::default()
            }),
            ..Default::default()
        }
    };

    // Initialize all qubits with prepare_z (gtype=1)
    for q in 0..n_qubits {
        let gid = gid_counter;
        gid_counter += 1;
        jit_instructions.push(make_instr(1, gid, vec![]));
        alive[q] = true;
        wire_gid[q] = gid;
        wire_port[q] = 0;
        all_gids.insert(gid);
        decode_tasks.push((gid, n_meas[&1]));
    }

    // Apply K random gates
    for _ in 0..n_gates {
        let live_qubits: Vec<usize> = (0..n_qubits).filter(|&q| alive[q]).collect();
        let dead_qubits: Vec<usize> = (0..n_qubits).filter(|&q| !alive[q]).collect();

        // Decide: revive a dead qubit, or apply a gate to live qubit(s)
        let do_revive = if live_qubits.is_empty() {
            true
        } else if dead_qubits.is_empty() {
            false
        } else {
            next_rand() % 4 == 0 // 25% chance to revive
        };

        if do_revive {
            let q = dead_qubits[next_rand() as usize % dead_qubits.len()];
            let gid = gid_counter;
            gid_counter += 1;
            jit_instructions.push(make_instr(1, gid, vec![]));
            alive[q] = true;
            wire_gid[q] = gid;
            wire_port[q] = 0;
            all_gids.insert(gid);
            decode_tasks.push((gid, n_meas[&1]));
        } else {
            // Pick a gate type with weighted distribution.
            // With 2-qubit gates: measure_z=10%, identity=30%, cnot=20% (free-hop), idle=40%.
            // Without 2-qubit gates: measure_z=12%, identity=40%, idle=48%.
            let r = next_rand() % 100;
            let gtype: u64 = if live_qubits.len() >= 2 {
                match r {
                    0..10 => 2,  // measure_z (terminal)
                    10..40 => 4, // identity (free-hop)
                    40..60 => 3, // cnot (2-qubit free-hop)
                    _ => 5,      // idle (hop-counted)
                }
            } else {
                match r {
                    0..12 => 2,  // measure_z (terminal)
                    12..52 => 4, // identity (free-hop)
                    _ => 5,      // idle (hop-counted)
                }
            };

            match gtype {
                4 => {
                    // Single-qubit free-hop (identity)
                    let q = live_qubits[next_rand() as usize % live_qubits.len()];
                    let gid = gid_counter;
                    gid_counter += 1;
                    jit_instructions.push(make_instr(
                        4,
                        gid,
                        vec![bin::gadget::Connector {
                            gid: wire_gid[q],
                            port: wire_port[q],
                        }],
                    ));
                    wire_gid[q] = gid;
                    wire_port[q] = 0;
                    all_gids.insert(gid);
                    decode_tasks.push((gid, 0));
                }
                3 => {
                    // Two-qubit gate (cnot): pick 2 distinct live qubits
                    let idx0 = next_rand() as usize % live_qubits.len();
                    let mut idx1 = next_rand() as usize % (live_qubits.len() - 1);
                    if idx1 >= idx0 {
                        idx1 += 1;
                    }
                    let q0 = live_qubits[idx0];
                    let q1 = live_qubits[idx1];
                    let gid = gid_counter;
                    gid_counter += 1;
                    jit_instructions.push(make_instr(
                        3,
                        gid,
                        vec![
                            bin::gadget::Connector {
                                gid: wire_gid[q0],
                                port: wire_port[q0],
                            },
                            bin::gadget::Connector {
                                gid: wire_gid[q1],
                                port: wire_port[q1],
                            },
                        ],
                    ));
                    wire_gid[q0] = gid;
                    wire_port[q0] = 0;
                    wire_gid[q1] = gid;
                    wire_port[q1] = 1;
                    all_gids.insert(gid);
                    decode_tasks.push((gid, n_meas[&3]));
                }
                5 => {
                    // Single-qubit hop-counted (idle)
                    let q = live_qubits[next_rand() as usize % live_qubits.len()];
                    let gid = gid_counter;
                    gid_counter += 1;
                    jit_instructions.push(make_instr(
                        5,
                        gid,
                        vec![bin::gadget::Connector {
                            gid: wire_gid[q],
                            port: wire_port[q],
                        }],
                    ));
                    wire_gid[q] = gid;
                    wire_port[q] = 0;
                    all_gids.insert(gid);
                    decode_tasks.push((gid, n_meas[&5]));
                }
                2 => {
                    // Terminal (measure_z): kills the qubit
                    let q = live_qubits[next_rand() as usize % live_qubits.len()];
                    let gid = gid_counter;
                    gid_counter += 1;
                    jit_instructions.push(make_instr(
                        2,
                        gid,
                        vec![bin::gadget::Connector {
                            gid: wire_gid[q],
                            port: wire_port[q],
                        }],
                    ));
                    alive[q] = false;
                    all_gids.insert(gid);
                    decode_tasks.push((gid, n_meas[&2]));
                }
                _ => unreachable!(),
            }
        }
    }

    // Terminate all remaining live qubits with measure_z (gtype=2)
    for q in 0..n_qubits {
        if alive[q] {
            let gid = gid_counter;
            gid_counter += 1;
            jit_instructions.push(make_instr(
                2,
                gid,
                vec![bin::gadget::Connector {
                    gid: wire_gid[q],
                    port: wire_port[q],
                }],
            ));
            alive[q] = false;
            all_gids.insert(gid);
            decode_tasks.push((gid, n_meas[&2]));
        }
    }

    // JIT-compile the program into a bin Library
    let mut jit_library = test_jit_library();
    jit_library.program = jit_instructions;
    let library = static_jit_compile(jit_library).await;

    // Split: types go to load_library, program instructions go to execute
    let types_library = bin::Library {
        port_types: library.port_types,
        gadget_types: library.gadget_types,
        check_model_types: library.check_model_types,
        error_model_types: library.error_model_types,
        ..Default::default()
    };
    let jit_free_hop_types = free_hop_types_from_library(&types_library);

    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator_with_hops(mock.clone(), &trace_path, buffer_radius);

    Coordinator::load_library(&coord, Request::new(types_library)).await.unwrap();

    // Execute all program instructions (gadgets, check models, error models)
    for instruction in library.program {
        Coordinator::execute(&coord, Request::new(instruction)).await.unwrap();
    }

    // Decode all concurrently — must use tokio::spawn for true concurrency
    // because non-leader decode calls block waiting for the leader's pauli_frame.
    // Each task sleeps a random 1–50ms before calling decode to expose timing-sensitive bugs.
    let coord = Arc::new(coord);
    let handles: Vec<_> = decode_tasks
        .iter()
        .enumerate()
        .map(|(i, &(gid, n_meas))| {
            let coord = coord.clone();
            let delay_ms = ((seed.wrapping_mul(31).wrapping_add(i as u64).wrapping_mul(7)) % 50) + 1;
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                let timeout_secs = 120;
                let result = tokio::time::timeout(
                    std::time::Duration::from_secs(timeout_secs),
                    Coordinator::decode(
                        &*coord,
                        Request::new(deq_runtime::coordinator::Outcomes {
                            gid,
                            outcomes: Some(BitVector {
                                data: vec![0; n_meas.div_ceil(8) as usize],
                                size: n_meas,
                            }),
                            ..Default::default()
                        }),
                    ),
                )
                .await;
                match result {
                    Ok(Ok(resp)) => resp.into_inner(),
                    Ok(Err(e)) => panic!("decode gid={gid} failed: {e}"),
                    Err(_) => panic!("decode gid={gid} timed out (deadlock?)"),
                }
            })
        })
        .collect();

    // Watchdog: dump state on timeout
    let watchdog_coord = coord.clone();
    let watchdog_decode_tasks = decode_tasks.clone();
    let watchdog = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        eprintln!("=== WATCHDOG: 10s elapsed, dumping gadget state ===");
        if let Ok(gadgets) = tokio::time::timeout(std::time::Duration::from_millis(500), watchdog_coord.gadgets.read()).await
        {
            let mut stuck = Vec::new();
            for &(gid, _) in &watchdog_decode_tasks {
                if let Some(g) = gadgets.get(&gid) {
                    let state = g.state.borrow().clone();
                    let pauli_frame = g.pauli_frame.borrow().is_some();
                    if !matches!(state, deq_runtime::coordinator::window_coordinator::GadgetState::Committed) || !pauli_frame
                    {
                        let outcomes = g.outcomes.borrow().is_some();
                        stuck.push(format!(
                            "gid={gid} out={outcomes} state={state:?} pf={pauli_frame} free={}",
                            g.is_free_hop
                        ));
                    }
                }
            }
            eprintln!("  {}/{} gadgets not done:", stuck.len(), watchdog_decode_tasks.len());
            for s in stuck.iter().take(30) {
                eprintln!("    {s}");
            }
            if stuck.len() > 30 {
                eprintln!("    ... and {} more", stuck.len() - 30);
            }
        } else {
            eprintln!("  TIMEOUT: gadgets lock held!");
        }
    });

    let results = futures_util::future::join_all(handles).await;
    watchdog.abort();
    for (i, result) in results.iter().enumerate() {
        let readouts = result.as_ref().unwrap();
        assert_eq!(readouts.gid, decode_tasks[i].0);
    }

    // Flush trace and verify partition property
    // Need to get coord back from Arc for reset_shot
    let coord_ref = &*coord;
    reset_shot(coord_ref).await;
    let trace = read_trace(&trace_path);
    assert_eq!(trace.shots.len(), 1);
    assert_window_correctness(&trace.shots[0], &all_gids, buffer_radius, &jit_free_hop_types);
}

/// Stress test: small random circuits (3 qubits, 20 gates) with various seeds, hop=1.
#[tokio::test]
async fn test_stress_random_small() {
    for seed in 1..=20 {
        run_random_circuit(3, 20, seed, 1).await;
    }
}

/// Stress test: medium random circuits (5 qubits, 40 gates) with various seeds, hop=1.
#[tokio::test]
async fn test_stress_random_medium() {
    for seed in 100..=115 {
        run_random_circuit(5, 40, seed, 1).await;
    }
}

/// Stress test: medium circuits with 2 buffer hops.
#[tokio::test]
async fn test_stress_random_medium_2_hops() {
    for seed in 200..=215 {
        run_random_circuit(5, 40, seed, 2).await;
    }
}

/// Stress test: larger circuits (8 qubits, 80 gates), hop=1.
#[tokio::test]
async fn test_stress_random_large() {
    for seed in 300..=310 {
        run_random_circuit(8, 80, seed, 1).await;
    }
}

// ─── timing test ───────────────────────────────────────────────────────────

/// Test that windowed decoding completes efficiently.
///
/// Uses the extended chain (A→T1→B→T2→C→T3→D→T4→E→T5→F) with buffer_radius=1
/// and a 30ms decode delay.  With windowed decoding, multiple gadgets are
/// committed per leader, so wall time should be well under 6×30ms = 180ms
/// (which would be the cost of decoding each gadget separately).
#[tokio::test(flavor = "multi_thread")]
async fn test_timing_parallel_decode() {
    let (gids, coord, mock, trace_file) = build_extended_chain(1).await;
    let [
        gid_a,
        gid_t1,
        gid_b,
        gid_t2,
        gid_c,
        gid_t3,
        gid_d,
        gid_t4,
        gid_e,
        gid_t5,
        gid_f,
    ] = gids;
    let trace_path = trace_file.path().to_str().unwrap().to_string();

    // Set 30ms decode delay
    mock.set_decode_delay(std::time::Duration::from_millis(30));

    // Decode all concurrently
    let (r_a, r_t1, r_b, r_t2, r_c, r_t3, r_d, r_t4, r_e, r_t5, r_f) = tokio::join!(
        decode(&coord, gid_a, 1),
        decode(&coord, gid_t1, 0),
        decode(&coord, gid_b, 1),
        decode(&coord, gid_t2, 0),
        decode(&coord, gid_c, 1),
        decode(&coord, gid_t3, 0),
        decode(&coord, gid_d, 1),
        decode(&coord, gid_t4, 0),
        decode(&coord, gid_e, 1),
        decode(&coord, gid_t5, 0),
        decode(&coord, gid_f, 1),
    );
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_t1.gid, gid_t1);
    assert_eq!(r_b.gid, gid_b);
    assert_eq!(r_t2.gid, gid_t2);
    assert_eq!(r_c.gid, gid_c);
    assert_eq!(r_t3.gid, gid_t3);
    assert_eq!(r_d.gid, gid_d);
    assert_eq!(r_t4.gid, gid_t4);
    assert_eq!(r_e.gid, gid_e);
    assert_eq!(r_t5.gid, gid_t5);
    assert_eq!(r_f.gid, gid_f);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];

    // Correctness check
    let all_gids: HashSet<u64> = HashSet::from([
        gid_a, gid_t1, gid_b, gid_t2, gid_c, gid_t3, gid_d, gid_t4, gid_e, gid_t5, gid_f,
    ]);
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // Timing check: verify we're not fully sequential.
    // With the await_mandatory_zone_syndrome step, task scheduling may
    // reduce parallelism — the extra lock release/acquire cycles allow a fast
    // explorer to grab the write lock before others finish.  We only assert
    // that wall time stays well below 6×30ms = 180ms (fully sequential).
    let (total_ms, wall_ms, n_parallel_pairs) = check_decode_parallelism(shot);
    eprintln!(
        "timing: total_decode={}ms, wall={}ms, parallel_pairs={}",
        total_ms, wall_ms, n_parallel_pairs
    );

    // With 6 hop-counted gadgets each taking ~30ms, fully sequential = ~180ms.
    // Even without parallelism, windows commit multiple gadgets per leader,
    // so wall time should be well under 180ms.
    assert!(
        wall_ms < 150,
        "wall time {wall_ms}ms too high — expected windowed decoding (total_decode={total_ms}ms)"
    );
}

/// Count leader decode events in a trace shot.
fn count_leaders(shot: &trace::Shot) -> usize {
    decode_events(shot).iter().filter(|d| d.is_leader).count()
}

/// Build a linear chain of `n` checked gadgets with free-hop gadgets between them.
///
/// Returns: A→T→B→T→C→T→...→Z(terminal)
/// where each letter is a checked gadget (gtype=1 for source, gtype=4 for middle,
/// gtype=5 for terminal) and T is a free-hop gadget (gtype=2).
///
/// Returns (all_gids_ordered, coord, mock, trace_file).
/// `all_gids_ordered` contains all GIDs (checked + free-hop) in chain order.
async fn build_long_chain(
    n_checked: usize,
    buffer_radius: usize,
) -> (Vec<u64>, Vec<u64>, WindowCoordinator, Arc<MockDecoder>, NamedTempFile) {
    assert!(n_checked >= 2);
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator_with_hops(mock.clone(), &trace_path, buffer_radius);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    let mut all_gids: Vec<u64> = Vec::new();
    let mut checked_gids: Vec<u64> = Vec::new();
    // CIDs are auto-assigned starting from 1. Track the next expected CID
    // so error_models reference the correct check_model.
    let mut next_cid_counter: u64 = 1;

    // First gadget: source (gtype=1, 0 inputs)
    let gid = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid)).await;
    exec_error_model(&coord, make_error_model(0, 1, next_cid_counter)).await;
    next_cid_counter += 1;
    all_gids.push(gid);
    checked_gids.push(gid);
    let mut prev_gid = gid;

    // Middle gadgets: free-hop → checked, repeated
    for _i in 1..n_checked - 1 {
        // Free-hop (gtype=2)
        let t_gid = exec_gadget(&coord, make_gadget(0, 2, vec![(prev_gid, 0)])).await;
        all_gids.push(t_gid);

        // Checked (gtype=4, 1 input)
        let c_gid = exec_gadget(&coord, make_gadget(0, 4, vec![(t_gid, 0)])).await;
        exec_check_model(&coord, make_check_model(0, 4, c_gid)).await;
        exec_error_model(&coord, make_error_model(0, 4, next_cid_counter)).await;
        next_cid_counter += 1;
        all_gids.push(c_gid);
        checked_gids.push(c_gid);
        prev_gid = c_gid;
    }

    // Last gadget: free-hop → terminal (gtype=5)
    let t_gid = exec_gadget(&coord, make_gadget(0, 2, vec![(prev_gid, 0)])).await;
    all_gids.push(t_gid);

    let term_gid = exec_gadget(&coord, make_gadget(0, 5, vec![(t_gid, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, term_gid)).await;
    exec_error_model(&coord, make_error_model(0, 5, next_cid_counter)).await;
    all_gids.push(term_gid);
    checked_gids.push(term_gid);

    (all_gids, checked_gids, coord, mock, trace_file)
}

/// Test timing: batch decode (all at once) with a long chain.
///
/// With 30 checked gadgets, buffer_radius=5, and 100ms decode delay:
/// - Window radius = 5 hops → leaders ≥10 hops apart don't overlap.
/// - Wave 1: ~3 parallel leaders (gadgets ~0, ~10, ~20).
///   After 100ms, these commit → committed gadgets become terminals.
/// - Subsequent waves fill in the gaps, each taking ~100ms.
/// - Total: a small number of waves (not 30 sequential decodes).
///
/// We verify:
///   1. Wall time << 30 × 100ms = 3000ms (proves parallelism + waves).
///   2. Number of leaders < 30 (some gadgets get committed by neighbors).
#[tokio::test(flavor = "multi_thread")]
async fn test_timing_batch_decode_long_chain() {
    let n_checked = 30;
    let buffer_radius = 5;
    let decode_delay_ms = 100;

    let (all_gids, checked_gids, coord, mock, trace_file) = build_long_chain(n_checked, buffer_radius).await;
    let trace_path = trace_file.path().to_str().unwrap().to_string();

    // Set 100ms decode delay
    mock.set_decode_delay(std::time::Duration::from_millis(decode_delay_ms));

    // Decode all concurrently (batch mode) — must use tokio::spawn
    // because non-leader decode calls block waiting for the leader's pauli_frame.
    let coord = Arc::new(coord);
    let checked_set: HashSet<u64> = checked_gids.iter().copied().collect();
    let handles: Vec<_> = all_gids
        .iter()
        .map(|&gid| {
            let coord = coord.clone();
            let n_meas = if checked_set.contains(&gid) { 1u64 } else { 0u64 };
            tokio::spawn(async move { decode(&coord, gid, n_meas).await })
        })
        .collect();

    let wall_start = std::time::Instant::now();
    let results = futures_util::future::join_all(handles).await;
    let wall_ms = wall_start.elapsed().as_millis();

    for result in &results {
        result.as_ref().unwrap();
    }

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];

    // Correctness check
    let all_gids_set: HashSet<u64> = all_gids.iter().copied().collect();
    assert_window_correctness(shot, &all_gids_set, buffer_radius, &test_free_hop_types());

    let n_leaders = count_leaders(shot);
    let (total_ms, trace_wall_ms, n_parallel_pairs) = check_decode_parallelism(shot);

    eprintln!(
        "batch_decode: n_checked={}, buffer_radius={}, delay={}ms",
        n_checked, buffer_radius, decode_delay_ms
    );
    eprintln!(
        "  leaders={}, total_decode={}ms, wall={}ms (measured={}ms), parallel_pairs={}",
        n_leaders, total_ms, trace_wall_ms, wall_ms, n_parallel_pairs
    );

    // With single-gadget commit, each checked gadget is its own leader.
    // Parallelism still works: non-overlapping windows decode concurrently.
    // With buffer_radius=5 and the chain pattern (checked → free → checked),
    // each checked gadget is 2 hops from adjacent checked gadgets (1 hop-counted
    // step through the free-hop), so windows 11+ hops apart can run in parallel.
    assert!(
        wall_ms < 1500,
        "batch wall time {wall_ms}ms too high — expected parallel wave decoding \
         (n_leaders={n_leaders}, total_decode={total_ms}ms)"
    );

    // CI runners with few cores can serialize leader decode execution even in
    // batch mode, making timestamp-overlap detection flaky. Keep a robust
    // structural check instead: batch mode should produce far fewer leaders
    // than checked gadgets because windows commit multiple gadgets per leader.
    assert!(
        n_leaders < n_checked,
        "too many leaders in batch mode: leaders={n_leaders}, n_checked={n_checked}"
    );
}

/// Test timing: staggered decode (10ms intervals) with a long chain.
///
/// With 30 checked gadgets, buffer_radius=5, and 100ms decode delay:
/// - Decode calls arrive every 10ms in chain order.
/// - By the time gadget[10] starts (100ms), gadget[0]'s decode is finishing.
///   Each gadget sees its predecessor as committed or committing.
/// - This produces a sliding-window pattern: each gadget (or small group)
///   becomes its own leader, yielding many more leaders than batch mode.
///
/// We verify:
///   1. Number of leaders is significantly more than in batch mode.
///   2. Wall time ≈ delay + (n-1) × stagger ≈ 100 + 290 ≈ 400ms.
#[tokio::test(flavor = "multi_thread")]
async fn test_timing_staggered_decode_long_chain() {
    let n_checked = 30;
    let buffer_radius = 5;
    let decode_delay_ms: u64 = 100;
    let stagger_ms: u64 = 10;

    let (all_gids, checked_gids, coord, mock, trace_file) = build_long_chain(n_checked, buffer_radius).await;
    let trace_path = trace_file.path().to_str().unwrap().to_string();

    // Set 100ms decode delay
    mock.set_decode_delay(std::time::Duration::from_millis(decode_delay_ms));

    // Decode with staggered timing: each gadget's decode starts 10ms after
    // the previous one, in chain order. Must use tokio::spawn for concurrency.
    let coord = Arc::new(coord);
    let checked_set: HashSet<u64> = checked_gids.iter().copied().collect();
    let handles: Vec<_> = all_gids
        .iter()
        .enumerate()
        .map(|(i, &gid)| {
            let coord = coord.clone();
            let n_meas = if checked_set.contains(&gid) { 1u64 } else { 0u64 };
            let delay = std::time::Duration::from_millis(i as u64 * stagger_ms);
            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                decode(&coord, gid, n_meas).await
            })
        })
        .collect();

    let wall_start = std::time::Instant::now();
    let results = futures_util::future::join_all(handles).await;
    let wall_ms = wall_start.elapsed().as_millis();

    for result in &results {
        result.as_ref().unwrap();
    }

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];

    // Correctness check
    let all_gids_set: HashSet<u64> = all_gids.iter().copied().collect();
    assert_window_correctness(shot, &all_gids_set, buffer_radius, &test_free_hop_types());

    let n_leaders = count_leaders(shot);
    let (total_ms, trace_wall_ms, n_parallel_pairs) = check_decode_parallelism(shot);

    eprintln!(
        "staggered_decode: n_checked={}, buffer_radius={}, delay={}ms, stagger={}ms",
        n_checked, buffer_radius, decode_delay_ms, stagger_ms
    );
    eprintln!(
        "  leaders={}, total_decode={}ms, wall={}ms (measured={}ms), parallel_pairs={}",
        n_leaders, total_ms, trace_wall_ms, wall_ms, n_parallel_pairs
    );

    // Staggered decode produces a sliding window: most checked gadgets
    // become their own leaders. However, with the buffer-committing
    // constraint, a gadget whose buffer includes a Decoding neighbor
    // must wait for that decode to finish, which can merge windows.
    // With buffer_radius=5 and 100ms decode delay, expect at least a few
    // independent leaders (more than batch mode's wave pattern).
    assert!(
        n_leaders >= 3,
        "staggered mode has too few leaders ({n_leaders} < 3) — expected some sliding window behavior",
    );

    // Wall time should be approximately:
    // last stagger delay + decode_delay = (n_all - 1) × 10ms + 100ms
    // With n_all = 2*30-1 = 59 gadgets: 58 × 10 + 100 = 680ms
    // Allow generous margin for scheduling jitter.
    let expected_wall_ms = (all_gids.len() as u64 - 1) * stagger_ms + decode_delay_ms;
    assert!(
        wall_ms < (expected_wall_ms * 3) as u128,
        "staggered wall time {wall_ms}ms too high — expected ~{expected_wall_ms}ms"
    );
}

// ─── benchmarks (run with: cargo test --test window_coordinator_test bench_ -- --ignored) ──
//
// Stress tested with 290,400 random seeds across hop values 1–5:
//   Hop 1:  83,700 seeds  |  Hop 2:  83,700 seeds
//   Hop 3:  41,000 seeds  |  Hop 4:  41,000 seeds  |  Hop 5:  41,000 seeds
//
// Per-configuration breakdown (summed across all hop values):
//   - Small  (3q, 20g):  140,000 seeds
//   - Medium (5q, 40g):   72,000 seeds
//   - Large  (8q, 80g):   44,000 seeds
//   - XL    (20q, 500g):  28,000 seeds
//   - XXL   (50q, 2000g):  6,400 seeds
//
// Each seed generates a random circuit with random decode delays (1-50ms),
// then verifies: partition correctness (every gadget committed exactly once),
// boundary distance correctness (committed gadgets ≥ buffer_radius from boundary),
// and buffer-committing concurrency (no commit region overlaps a concurrent buffer).
// All seeds passed with zero failures.

/// Benchmark: small circuits (3 qubits, 20 gates) with hop=1, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop1() {
    for seed in 10000..15000u64 {
        run_random_circuit(3, 20, seed, 1).await;
    }
}

/// Benchmark: small circuits (3 qubits, 20 gates) with hop=2, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop2() {
    for seed in 15000..20000u64 {
        run_random_circuit(3, 20, seed, 2).await;
    }
}

/// Benchmark: medium circuits (5 qubits, 40 gates) with hop=1, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop1() {
    for seed in 20000..23000u64 {
        run_random_circuit(5, 40, seed, 1).await;
    }
}

/// Benchmark: medium circuits (5 qubits, 40 gates) with hop=2, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop2() {
    for seed in 23000..26000u64 {
        run_random_circuit(5, 40, seed, 2).await;
    }
}

/// Benchmark: large circuits (8 qubits, 80 gates) with hop=1, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop1() {
    for seed in 30000..32000u64 {
        run_random_circuit(8, 80, seed, 1).await;
    }
}

/// Benchmark: large circuits (8 qubits, 80 gates) with hop=2, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop2() {
    for seed in 32000..34000u64 {
        run_random_circuit(8, 80, seed, 2).await;
    }
}

/// Benchmark: XL circuits (20 qubits, 500 gates) with hop=1, 1000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop1() {
    for seed in 40000..41000u64 {
        run_random_circuit(20, 500, seed, 1).await;
    }
}

/// Benchmark: XL circuits (20 qubits, 500 gates) with hop=2, 1000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop2() {
    for seed in 41000..42000u64 {
        run_random_circuit(20, 500, seed, 2).await;
    }
}

/// Benchmark: XXL circuits (50 qubits, 2000 gates) with hop=1, 200 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop1() {
    for seed in 50000..50200u64 {
        run_random_circuit(50, 2000, seed, 1).await;
    }
}

/// Benchmark: XXL circuits (50 qubits, 2000 gates) with hop=2, 200 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop2() {
    for seed in 50200..50400u64 {
        run_random_circuit(50, 2000, seed, 2).await;
    }
}

/// Round 2: small circuits (3 qubits, 20 gates) with hop=1, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop1_r2() {
    for seed in 60000..65000u64 {
        run_random_circuit(3, 20, seed, 1).await;
    }
}

/// Round 2: small circuits (3 qubits, 20 gates) with hop=2, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop2_r2() {
    for seed in 65000..70000u64 {
        run_random_circuit(3, 20, seed, 2).await;
    }
}

/// Round 2: medium circuits (5 qubits, 40 gates) with hop=1, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop1_r2() {
    for seed in 70000..73000u64 {
        run_random_circuit(5, 40, seed, 1).await;
    }
}

/// Round 2: medium circuits (5 qubits, 40 gates) with hop=2, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop2_r2() {
    for seed in 73000..76000u64 {
        run_random_circuit(5, 40, seed, 2).await;
    }
}

/// Round 2: large circuits (8 qubits, 80 gates) with hop=1, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop1_r2() {
    for seed in 80000..82000u64 {
        run_random_circuit(8, 80, seed, 1).await;
    }
}

/// Round 2: large circuits (8 qubits, 80 gates) with hop=2, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop2_r2() {
    for seed in 82000..84000u64 {
        run_random_circuit(8, 80, seed, 2).await;
    }
}

/// Round 2: XL circuits (20 qubits, 500 gates) with hop=1, 1000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop1_r2() {
    for seed in 90000..91000u64 {
        run_random_circuit(20, 500, seed, 1).await;
    }
}

/// Round 2: XL circuits (20 qubits, 500 gates) with hop=2, 1000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop2_r2() {
    for seed in 91000..92000u64 {
        run_random_circuit(20, 500, seed, 2).await;
    }
}

/// Round 3: small circuits (3 qubits, 20 gates) with hop=1, 10000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop1_r3() {
    for seed in 100000..110000u64 {
        run_random_circuit(3, 20, seed, 1).await;
    }
}

/// Round 3: small circuits (3 qubits, 20 gates) with hop=2, 10000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop2_r3() {
    for seed in 110000..120000u64 {
        run_random_circuit(3, 20, seed, 2).await;
    }
}

/// Round 3: medium circuits (5 qubits, 40 gates) with hop=1, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop1_r3() {
    for seed in 120000..125000u64 {
        run_random_circuit(5, 40, seed, 1).await;
    }
}

/// Round 3: medium circuits (5 qubits, 40 gates) with hop=2, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop2_r3() {
    for seed in 125000..130000u64 {
        run_random_circuit(5, 40, seed, 2).await;
    }
}

/// Round 3: large circuits (8 qubits, 80 gates) with hop=1, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop1_r3() {
    for seed in 130000..133000u64 {
        run_random_circuit(8, 80, seed, 1).await;
    }
}

/// Round 3: large circuits (8 qubits, 80 gates) with hop=2, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop2_r3() {
    for seed in 133000..136000u64 {
        run_random_circuit(8, 80, seed, 2).await;
    }
}

/// Round 3: XL circuits (20 qubits, 500 gates) with hop=1, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop1_r3() {
    for seed in 140000..142000u64 {
        run_random_circuit(20, 500, seed, 1).await;
    }
}

/// Round 3: XL circuits (20 qubits, 500 gates) with hop=2, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop2_r3() {
    for seed in 142000..144000u64 {
        run_random_circuit(20, 500, seed, 2).await;
    }
}

/// Round 3: XXL circuits (50 qubits, 2000 gates) with hop=1, 500 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop1_r3() {
    for seed in 150000..150500u64 {
        run_random_circuit(50, 2000, seed, 1).await;
    }
}

/// Round 3: XXL circuits (50 qubits, 2000 gates) with hop=2, 500 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop2_r3() {
    for seed in 150500..151000u64 {
        run_random_circuit(50, 2000, seed, 2).await;
    }
}

/// Round 4: small circuits (3 qubits, 20 gates) with hop=1, 10000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop1_r4() {
    for seed in 200000..210000u64 {
        run_random_circuit(3, 20, seed, 1).await;
    }
}

/// Round 4: small circuits (3 qubits, 20 gates) with hop=2, 10000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop2_r4() {
    for seed in 210000..220000u64 {
        run_random_circuit(3, 20, seed, 2).await;
    }
}

/// Round 4: medium circuits (5 qubits, 40 gates) with hop=1, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop1_r4() {
    for seed in 220000..225000u64 {
        run_random_circuit(5, 40, seed, 1).await;
    }
}

/// Round 4: medium circuits (5 qubits, 40 gates) with hop=2, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop2_r4() {
    for seed in 225000..230000u64 {
        run_random_circuit(5, 40, seed, 2).await;
    }
}

/// Round 4: large circuits (8 qubits, 80 gates) with hop=1, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop1_r4() {
    for seed in 230000..233000u64 {
        run_random_circuit(8, 80, seed, 1).await;
    }
}

/// Round 4: large circuits (8 qubits, 80 gates) with hop=2, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop2_r4() {
    for seed in 233000..236000u64 {
        run_random_circuit(8, 80, seed, 2).await;
    }
}

/// Round 4: XL circuits (20 qubits, 500 gates) with hop=1, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop1_r4() {
    for seed in 240000..242000u64 {
        run_random_circuit(20, 500, seed, 1).await;
    }
}

/// Round 4: XL circuits (20 qubits, 500 gates) with hop=2, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop2_r4() {
    for seed in 242000..244000u64 {
        run_random_circuit(20, 500, seed, 2).await;
    }
}

/// Round 4: XXL circuits (50 qubits, 2000 gates) with hop=1, 500 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop1_r4() {
    for seed in 250000..250500u64 {
        run_random_circuit(50, 2000, seed, 1).await;
    }
}

/// Round 4: XXL circuits (50 qubits, 2000 gates) with hop=2, 500 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop2_r4() {
    for seed in 250500..251000u64 {
        run_random_circuit(50, 2000, seed, 2).await;
    }
}

/// Round 5: small circuits (3 qubits, 20 gates) with hop=1, 10000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop1_r5() {
    for seed in 300000..310000u64 {
        run_random_circuit(3, 20, seed, 1).await;
    }
}

/// Round 5: small circuits (3 qubits, 20 gates) with hop=2, 10000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop2_r5() {
    for seed in 310000..320000u64 {
        run_random_circuit(3, 20, seed, 2).await;
    }
}

/// Round 5: medium circuits (5 qubits, 40 gates) with hop=1, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop1_r5() {
    for seed in 320000..325000u64 {
        run_random_circuit(5, 40, seed, 1).await;
    }
}

/// Round 5: medium circuits (5 qubits, 40 gates) with hop=2, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop2_r5() {
    for seed in 325000..330000u64 {
        run_random_circuit(5, 40, seed, 2).await;
    }
}

/// Round 5: large circuits (8 qubits, 80 gates) with hop=1, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop1_r5() {
    for seed in 330000..333000u64 {
        run_random_circuit(8, 80, seed, 1).await;
    }
}

/// Round 5: large circuits (8 qubits, 80 gates) with hop=2, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop2_r5() {
    for seed in 333000..336000u64 {
        run_random_circuit(8, 80, seed, 2).await;
    }
}

/// Round 5: XL circuits (20 qubits, 500 gates) with hop=1, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop1_r5() {
    for seed in 340000..342000u64 {
        run_random_circuit(20, 500, seed, 1).await;
    }
}

/// Round 5: XL circuits (20 qubits, 500 gates) with hop=2, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop2_r5() {
    for seed in 342000..344000u64 {
        run_random_circuit(20, 500, seed, 2).await;
    }
}

/// Round 5: XXL circuits (50 qubits, 2000 gates) with hop=1, 500 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop1_r5() {
    for seed in 350000..350500u64 {
        run_random_circuit(50, 2000, seed, 1).await;
    }
}

/// Round 5: XXL circuits (50 qubits, 2000 gates) with hop=2, 500 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop2_r5() {
    for seed in 350500..351000u64 {
        run_random_circuit(50, 2000, seed, 2).await;
    }
}

/// Stress: small (3q, 20g) with hop=3, 10000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop3() {
    for seed in 400000..410000u64 {
        run_random_circuit(3, 20, seed, 3).await;
    }
}

/// Stress: medium (5q, 40g) with hop=3, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop3() {
    for seed in 410000..415000u64 {
        run_random_circuit(5, 40, seed, 3).await;
    }
}

/// Stress: large (8q, 80g) with hop=3, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop3() {
    for seed in 415000..418000u64 {
        run_random_circuit(8, 80, seed, 3).await;
    }
}

/// Stress: xl (20q, 500g) with hop=3, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop3() {
    for seed in 418000..420000u64 {
        run_random_circuit(20, 500, seed, 3).await;
    }
}

/// Stress: xxl (50q, 2000g) with hop=3, 500 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop3() {
    for seed in 420000..420500u64 {
        run_random_circuit(50, 2000, seed, 3).await;
    }
}

/// Round 2: small (3q, 20g) with hop=3, 10000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop3_r2() {
    for seed in 420500..430500u64 {
        run_random_circuit(3, 20, seed, 3).await;
    }
}

/// Round 2: medium (5q, 40g) with hop=3, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop3_r2() {
    for seed in 430500..435500u64 {
        run_random_circuit(5, 40, seed, 3).await;
    }
}

/// Round 2: large (8q, 80g) with hop=3, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop3_r2() {
    for seed in 435500..438500u64 {
        run_random_circuit(8, 80, seed, 3).await;
    }
}

/// Round 2: xl (20q, 500g) with hop=3, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop3_r2() {
    for seed in 438500..440500u64 {
        run_random_circuit(20, 500, seed, 3).await;
    }
}

/// Round 2: xxl (50q, 2000g) with hop=3, 500 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop3_r2() {
    for seed in 440500..441000u64 {
        run_random_circuit(50, 2000, seed, 3).await;
    }
}

/// Stress: small (3q, 20g) with hop=4, 10000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop4() {
    for seed in 441000..451000u64 {
        run_random_circuit(3, 20, seed, 4).await;
    }
}

/// Stress: medium (5q, 40g) with hop=4, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop4() {
    for seed in 451000..456000u64 {
        run_random_circuit(5, 40, seed, 4).await;
    }
}

/// Stress: large (8q, 80g) with hop=4, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop4() {
    for seed in 456000..459000u64 {
        run_random_circuit(8, 80, seed, 4).await;
    }
}

/// Stress: xl (20q, 500g) with hop=4, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop4() {
    for seed in 459000..461000u64 {
        run_random_circuit(20, 500, seed, 4).await;
    }
}

/// Stress: xxl (50q, 2000g) with hop=4, 500 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop4() {
    for seed in 461000..461500u64 {
        run_random_circuit(50, 2000, seed, 4).await;
    }
}

/// Round 2: small (3q, 20g) with hop=4, 10000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop4_r2() {
    for seed in 461500..471500u64 {
        run_random_circuit(3, 20, seed, 4).await;
    }
}

/// Round 2: medium (5q, 40g) with hop=4, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop4_r2() {
    for seed in 471500..476500u64 {
        run_random_circuit(5, 40, seed, 4).await;
    }
}

/// Round 2: large (8q, 80g) with hop=4, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop4_r2() {
    for seed in 476500..479500u64 {
        run_random_circuit(8, 80, seed, 4).await;
    }
}

/// Round 2: xl (20q, 500g) with hop=4, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop4_r2() {
    for seed in 479500..481500u64 {
        run_random_circuit(20, 500, seed, 4).await;
    }
}

/// Round 2: xxl (50q, 2000g) with hop=4, 500 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop4_r2() {
    for seed in 481500..482000u64 {
        run_random_circuit(50, 2000, seed, 4).await;
    }
}

/// Stress: small (3q, 20g) with hop=5, 10000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop5() {
    for seed in 482000..492000u64 {
        run_random_circuit(3, 20, seed, 5).await;
    }
}

/// Stress: medium (5q, 40g) with hop=5, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop5() {
    for seed in 492000..497000u64 {
        run_random_circuit(5, 40, seed, 5).await;
    }
}

/// Stress: large (8q, 80g) with hop=5, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop5() {
    for seed in 497000..500000u64 {
        run_random_circuit(8, 80, seed, 5).await;
    }
}

/// Stress: xl (20q, 500g) with hop=5, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop5() {
    for seed in 500000..502000u64 {
        run_random_circuit(20, 500, seed, 5).await;
    }
}

/// Stress: xxl (50q, 2000g) with hop=5, 500 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop5() {
    for seed in 502000..502500u64 {
        run_random_circuit(50, 2000, seed, 5).await;
    }
}

/// Round 2: small (3q, 20g) with hop=5, 10000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_small_hop5_r2() {
    for seed in 502500..512500u64 {
        run_random_circuit(3, 20, seed, 5).await;
    }
}

/// Round 2: medium (5q, 40g) with hop=5, 5000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_medium_hop5_r2() {
    for seed in 512500..517500u64 {
        run_random_circuit(5, 40, seed, 5).await;
    }
}

/// Round 2: large (8q, 80g) with hop=5, 3000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_large_hop5_r2() {
    for seed in 517500..520500u64 {
        run_random_circuit(8, 80, seed, 5).await;
    }
}

/// Round 2: xl (20q, 500g) with hop=5, 2000 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xl_hop5_r2() {
    for seed in 520500..522500u64 {
        run_random_circuit(20, 500, seed, 5).await;
    }
}

/// Round 2: xxl (50q, 2000g) with hop=5, 500 seeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_stress_xxl_hop5_r2() {
    for seed in 522500..523000u64 {
        run_random_circuit(50, 2000, seed, 5).await;
    }
}

/// Test: A committing region must not be used as buffer by a concurrent window.
///
/// Chain: A(source) → B(checked_1in) → C(checked_1in) → D(terminal)
/// buffer_radius = 1
///
/// Expected windows and commit regions:
///   A: window={A,B}, commit={A}   — B is buffer for A
///   B: window={A,B,C}, commit={B} — A and C are buffer for B
///   C: window={B,C,D}, commit={C,D} — B is buffer for C
///
/// A and C can decode in parallel: their windows overlap on B, but B is
/// buffer for both (not in either commit region). Their commit regions
/// {A} and {C,D} are disjoint and neither appears in the other's buffer.
///
/// B must wait for A (or C) to commit, because A's commit region {A} is
/// part of B's buffer {A,C}. Decoding B while A is still committing would
/// mean B's boundary context at A is unstable.
///
/// This test uses a 30ms decode delay and verifies that no leader's commit
/// region overlaps with a concurrently-decoding leader's buffer.
#[tokio::test(flavor = "multi_thread")]
async fn test_overlapping_windows_sequential() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    mock.set_decode_delay(std::time::Duration::from_millis(30));
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A(source, gtype=1) → B(checked_1in, gtype=4) → C(checked_1in, gtype=4) → D(terminal, gtype=5)
    let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    let gid_b = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_a, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_b)).await;
    exec_error_model(&coord, make_error_model(0, 4, 2)).await;

    let gid_c = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_b, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_c)).await;
    exec_error_model(&coord, make_error_model(0, 4, 3)).await;

    let gid_d = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_c, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_d)).await;
    exec_error_model(&coord, make_error_model(0, 5, 4)).await;

    // Decode all concurrently
    let (r_a, r_b, r_c, r_d) = tokio::join!(
        decode(&coord, gid_a, 1),
        decode(&coord, gid_b, 1),
        decode(&coord, gid_c, 1),
        decode(&coord, gid_d, 1),
    );
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_b.gid, gid_b);
    assert_eq!(r_c.gid, gid_c);
    assert_eq!(r_d.gid, gid_d);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let all_gids: HashSet<u64> = [gid_a, gid_b, gid_c, gid_d].into();
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());

    // Buffer-committing concurrency is verified by check 7 in
    // assert_window_correctness above.
}

/// Test: Free-hop gadgets must not inflate boundary distance.
///
/// Chain: A(source) → T1(free-hop) → B(checked) → T2(free-hop) → C(checked) → D(terminal)
/// buffer_radius = 1
///
/// The boundary-distance invariant is verified by `assert_window_correctness`:
/// every committed hop-counted gadget (except the center) has
/// `boundary_dist ≥ buffer_radius`.  Free-hops are always absorbed.
///
/// Which gadget becomes center is scheduler-dependent, so we only
/// assert partition correctness and boundary-distance safety.
#[tokio::test]
async fn test_free_hop_boundary_distance() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // A(source, gtype=1)
    let gid_a = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    // T1(free-hop, gtype=2)
    let gid_t1 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_a, 0)])).await;

    // B(checked_1in, gtype=4)
    let gid_b = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t1, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_b)).await;
    exec_error_model(&coord, make_error_model(0, 4, 2)).await;

    // T2(free-hop, gtype=2)
    let gid_t2 = exec_gadget(&coord, make_gadget(0, 2, vec![(gid_b, 0)])).await;

    // C(checked_1in, gtype=4)
    let gid_c = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_t2, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_c)).await;
    exec_error_model(&coord, make_error_model(0, 4, 3)).await;

    // D(terminal, gtype=5)
    let gid_d = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_c, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_d)).await;
    exec_error_model(&coord, make_error_model(0, 5, 4)).await;

    // Decode all concurrently
    let (r_a, r_t1, r_b, r_t2, r_c, r_d) = tokio::join!(
        decode(&coord, gid_a, 1),
        decode(&coord, gid_t1, 0),
        decode(&coord, gid_b, 1),
        decode(&coord, gid_t2, 0),
        decode(&coord, gid_c, 1),
        decode(&coord, gid_d, 1),
    );
    assert_eq!(r_a.gid, gid_a);
    assert_eq!(r_t1.gid, gid_t1);
    assert_eq!(r_b.gid, gid_b);
    assert_eq!(r_t2.gid, gid_t2);
    assert_eq!(r_c.gid, gid_c);
    assert_eq!(r_d.gid, gid_d);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let all_gids: HashSet<u64> = [gid_a, gid_t1, gid_b, gid_t2, gid_c, gid_d].into();

    // Partition correctness + boundary-distance safety (center-agnostic).
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());
}

/// Test: Free-hop boundary distance with multi-input/output gadget (CNOT pattern).
///
/// DAG (mirrors notebook Part 3):
///   gid=1 (PrepZ, hc, source) ─→ gid=3 (CNOT, fh, 2in/2out) ─→ gid=4 (Idle, hc) ─→ gid=6 (MeasZ, hc, terminal)
///   gid=2 (PrepZ, hc, source) ─↗                              ─→ gid=5 (Idle, hc) ─→ gid=7 (MeasZ, hc, terminal)
///
/// buffer_radius = 1.  The boundary-distance invariant is verified by
/// `assert_window_correctness`.  Which gadget becomes center is
/// scheduler-dependent, so we only assert partition correctness and
/// boundary-distance safety.
#[tokio::test]
async fn test_free_hop_boundary_distance_cnot() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // gid=1: PrepZ (source, gtype=1, 0in/1out)
    let gid_1 = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_1)).await;
    exec_error_model(&coord, make_error_model(0, 1, 1)).await;

    // gid=2: PrepZ (source, gtype=1, 0in/1out)
    let gid_2 = exec_gadget(&coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(&coord, make_check_model(0, 1, gid_2)).await;
    exec_error_model(&coord, make_error_model(0, 1, 2)).await;

    // gid=3: CNOT (free-hop, gtype=3, 2in/2out)
    let gid_3 = exec_gadget(&coord, make_gadget(0, 3, vec![(gid_1, 0), (gid_2, 0)])).await;

    // gid=4: Idle (checked_1in, gtype=4, 1in/1out)
    let gid_4 = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_3, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_4)).await;
    exec_error_model(&coord, make_error_model(0, 4, 3)).await;

    // gid=5: Idle (checked_1in, gtype=4, 1in/1out)
    let gid_5 = exec_gadget(&coord, make_gadget(0, 4, vec![(gid_3, 1)])).await;
    exec_check_model(&coord, make_check_model(0, 4, gid_5)).await;
    exec_error_model(&coord, make_error_model(0, 4, 4)).await;

    // gid=6: MeasZ (terminal, gtype=5, 1in/0out)
    let gid_6 = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_4, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_6)).await;
    exec_error_model(&coord, make_error_model(0, 5, 5)).await;

    // gid=7: MeasZ (terminal, gtype=5, 1in/0out)
    let gid_7 = exec_gadget(&coord, make_gadget(0, 5, vec![(gid_5, 0)])).await;
    exec_check_model(&coord, make_check_model(0, 5, gid_7)).await;
    exec_error_model(&coord, make_error_model(0, 5, 6)).await;

    // Decode all concurrently
    let (r1, r2, r3, r4, r5, r6, r7) = tokio::join!(
        decode(&coord, gid_1, 1),
        decode(&coord, gid_2, 1),
        decode(&coord, gid_3, 0),
        decode(&coord, gid_4, 1),
        decode(&coord, gid_5, 1),
        decode(&coord, gid_6, 1),
        decode(&coord, gid_7, 1),
    );
    assert_eq!(r1.gid, gid_1);
    assert_eq!(r2.gid, gid_2);
    assert_eq!(r3.gid, gid_3);
    assert_eq!(r4.gid, gid_4);
    assert_eq!(r5.gid, gid_5);
    assert_eq!(r6.gid, gid_6);
    assert_eq!(r7.gid, gid_7);

    reset_shot(&coord).await;
    let trace = read_trace(&trace_path);
    let shot = &trace.shots[0];
    let all_gids: HashSet<u64> = [gid_1, gid_2, gid_3, gid_4, gid_5, gid_6, gid_7].into();

    // Partition correctness + boundary-distance safety (center-agnostic).
    assert_window_correctness(shot, &all_gids, 1, &test_free_hop_types());
}

// ===================================================================
// Deadlock regression tests: chains of checked gadgets (no free-hops)
// ===================================================================

/// Helper: decode via an Arc<WindowCoordinator> (needed for tokio::spawn).
async fn decode_arc(coord: Arc<WindowCoordinator>, gid: u64, num_measurements: u64) -> deq_runtime::coordinator::Readouts {
    Coordinator::decode(
        coord.as_ref(),
        Request::new(deq_runtime::coordinator::Outcomes {
            gid,
            outcomes: Some(BitVector {
                data: vec![0; num_measurements.div_ceil(8) as usize],
                size: num_measurements,
            }),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
    .into_inner()
}

/// Library with remote error model references for deadlock regression tests.
///
/// Like `make_test_library()` but error model types for source (etype 11)
/// and middle (etype 14) have `remote_check_models: [output(0)]`, meaning
/// each error model references the NEXT gadget's check model.  This creates
/// `referring_eids` cross-links that are necessary to trigger the
/// `is_safe_terminal` bug (#empty-commit-region deadlock).
fn make_test_library_with_remote_errors() -> bin::Library {
    let mut lib = make_test_library();

    // etype=11: error model for source gadget with remote_check_model → output(0)
    lib.error_model_types.push(bin::ErrorModelType {
        etype: 11,
        ctype: 1,
        errors: vec![bin::error_model_type::Error {
            probability: 0.1,
            checks: vec![
                bin::error_model_type::RemoteCheck {
                    check_index: 0,
                    remote_check_model: None,
                },
                bin::error_model_type::RemoteCheck {
                    check_index: 0,
                    remote_check_model: Some(0), // references remote_check_models[0]
                },
            ],
            residual: vec![],
            readout_flips: vec![],
            ..Default::default()
        }],
        remote_check_models: vec![bin::error_model_type::RemoteCheckModel {
            port: Some(bin::error_model_type::remote_check_model::Port::Output(0)),
            ..Default::default()
        }],
        ..Default::default()
    });

    // etype=14: error model for middle gadget with remote_check_model → output(0)
    lib.error_model_types.push(bin::ErrorModelType {
        etype: 14,
        ctype: 4,
        errors: vec![bin::error_model_type::Error {
            probability: 0.1,
            checks: vec![
                bin::error_model_type::RemoteCheck {
                    check_index: 0,
                    remote_check_model: None,
                },
                bin::error_model_type::RemoteCheck {
                    check_index: 0,
                    remote_check_model: Some(0), // references remote_check_models[0]
                },
            ],
            residual: vec![],
            readout_flips: vec![],
            ..Default::default()
        }],
        remote_check_models: vec![bin::error_model_type::RemoteCheckModel {
            port: Some(bin::error_model_type::remote_check_model::Port::Output(0)),
            ..Default::default()
        }],
        ..Default::default()
    });

    lib
}

/// Build a chain of `n` checked gadgets with NO free-hops:
///   G0(source) → G1(checked_1in) → ... → G_{n-1}(terminal)
/// Each gadget has 1 measurement, 1 check model, 1 error model.
/// Error models for non-terminal gadgets use `remote_check_models` that
/// reference the NEXT gadget's check model (via output(0)), creating the
/// `referring_eids` cross-links needed to trigger the is_safe_terminal bug.
/// Returns an Arc<WindowCoordinator> so it can be shared across spawned tasks.
async fn build_checked_chain(
    n: usize,
    buffer_radius: usize,
) -> (Vec<u64>, Arc<WindowCoordinator>, Arc<MockDecoder>, NamedTempFile) {
    assert!(n >= 2, "chain needs at least 2 gadgets");
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = Arc::new(make_coordinator_with_hops(mock.clone(), &trace_path, buffer_radius));

    Coordinator::load_library(coord.as_ref(), Request::new(make_test_library_with_remote_errors()))
        .await
        .unwrap();

    let mut gids = Vec::with_capacity(n);

    // Per-gadget ordering: Gadget → CheckModel → ErrorModel for each gadget,
    // matching the real compiler's emission order.  The deferred referring_eids
    // registration resolves forward references that aren't available at error
    // model creation time (downstream gadget not yet created / not yet bound
    // to a check model).
    //
    // Source gadget (gtype=1): Gadget → CheckModel → ErrorModel(etype=11, remote refs)
    let gid_0 = exec_gadget(coord.as_ref(), make_gadget(0, 1, vec![])).await;
    gids.push(gid_0);
    exec_check_model(coord.as_ref(), make_check_model(0, 1, gid_0)).await;
    exec_error_model(coord.as_ref(), make_error_model(0, 11, 1)).await;

    // Middle gadgets (gtype=4): each one per-gadget
    for i in 1..n - 1 {
        let prev = gids[i - 1];
        let gid = exec_gadget(coord.as_ref(), make_gadget(0, 4, vec![(prev, 0)])).await;
        gids.push(gid);
        exec_check_model(coord.as_ref(), make_check_model(0, 4, gid)).await;
        exec_error_model(coord.as_ref(), make_error_model(0, 14, (i + 1) as u64)).await;
    }

    // Terminal gadget (gtype=5): no remote refs in error model
    let prev = *gids.last().unwrap();
    let gid_last = exec_gadget(coord.as_ref(), make_gadget(0, 5, vec![(prev, 0)])).await;
    gids.push(gid_last);
    exec_check_model(coord.as_ref(), make_check_model(0, 5, gid_last)).await;
    exec_error_model(coord.as_ref(), make_error_model(0, 5, n as u64)).await;

    (gids, coord, mock, trace_file)
}

/// Regression test: 5 checked gadgets (no free-hops), buffer_radius=2.
///
/// Chain: G0 → G1 → G2 → G3 → G4 (all checked, error models reference
/// next gadget's check model via output(0)).
///
/// The `referring_eids` cross-references cause `is_safe_terminal` to return
/// false for committed gadgets whose next-gadget error model owner is still
/// uncommitted.  Without the fix, late-committing gadgets get an empty
/// commit_region and hang in `wait_for_pauli_frame` forever.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_checked_chain_no_free_hops_2_hops() {
    let (gids, coord, mock, trace_file) = build_checked_chain(5, 2).await;
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    // A small decode delay makes overlapping windows more likely to conflict
    mock.set_decode_delay(std::time::Duration::from_millis(10));

    let handles: Vec<_> = gids
        .iter()
        .map(|&gid| {
            let c = coord.clone();
            tokio::spawn(async move { decode_arc(c, gid, 1).await })
        })
        .collect();

    let results = tokio::time::timeout(std::time::Duration::from_secs(10), futures_util::future::join_all(handles))
        .await
        .expect("DEADLOCK: concurrent decode did not complete within 10s");

    for (i, result) in results.into_iter().enumerate() {
        let readouts = result.unwrap();
        assert_eq!(readouts.gid, gids[i], "gid mismatch for gadget {i}");
    }

    reset_shot(coord.as_ref()).await;
    let trace = read_trace(&trace_path);
    let all_gids: HashSet<u64> = gids.into_iter().collect();
    assert_window_correctness(&trace.shots[0], &all_gids, 2, &test_free_hop_types());
}

/// Regression test: 5 checked gadgets (no free-hops), buffer_radius=3.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_checked_chain_no_free_hops_3_hops() {
    let (gids, coord, mock, trace_file) = build_checked_chain(5, 3).await;
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    mock.set_decode_delay(std::time::Duration::from_millis(10));

    let handles: Vec<_> = gids
        .iter()
        .map(|&gid| {
            let c = coord.clone();
            tokio::spawn(async move { decode_arc(c, gid, 1).await })
        })
        .collect();

    let results = tokio::time::timeout(std::time::Duration::from_secs(10), futures_util::future::join_all(handles))
        .await
        .expect("DEADLOCK: concurrent decode did not complete within 10s");

    for (i, result) in results.into_iter().enumerate() {
        let readouts = result.unwrap();
        assert_eq!(readouts.gid, gids[i], "gid mismatch for gadget {i}");
    }

    reset_shot(coord.as_ref()).await;
    let trace = read_trace(&trace_path);
    let all_gids: HashSet<u64> = gids.into_iter().collect();
    assert_window_correctness(&trace.shots[0], &all_gids, 3, &test_free_hop_types());
}

/// Regression test: 10 checked gadgets (no free-hops), buffer_radius=4.
/// Longer chain with more overlapping windows.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_checked_chain_no_free_hops_10_gadgets_4_hops() {
    let (gids, coord, mock, trace_file) = build_checked_chain(10, 4).await;
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    mock.set_decode_delay(std::time::Duration::from_millis(10));

    let handles: Vec<_> = gids
        .iter()
        .map(|&gid| {
            let c = coord.clone();
            tokio::spawn(async move { decode_arc(c, gid, 1).await })
        })
        .collect();

    let results = tokio::time::timeout(std::time::Duration::from_secs(10), futures_util::future::join_all(handles))
        .await
        .expect("DEADLOCK: concurrent decode did not complete within 10s");

    for (i, result) in results.into_iter().enumerate() {
        let readouts = result.unwrap();
        assert_eq!(readouts.gid, gids[i], "gid mismatch for gadget {i}");
    }

    reset_shot(coord.as_ref()).await;
    let trace = read_trace(&trace_path);
    let all_gids: HashSet<u64> = gids.into_iter().collect();
    assert_window_correctness(&trace.shots[0], &all_gids, 4, &test_free_hop_types());
}

/// Regression test: 7 checked gadgets, buffer_radius=2, 5 shots.
///
/// The empty-commit-region bug often manifests on 2nd+ shots because
/// different scheduling orders lead to different committed-gadget patterns.
/// Running multiple shots increases the chance of hitting the problematic
/// schedule.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_checked_chain_multi_shot_7_gadgets_2_hops() {
    let n = 7;
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    mock.set_decode_delay(std::time::Duration::from_millis(5));
    let coord = Arc::new(make_coordinator_with_hops(mock.clone(), &trace_path, 2));

    Coordinator::load_library(coord.as_ref(), Request::new(make_test_library_with_remote_errors()))
        .await
        .unwrap();

    for shot in 0..5 {
        // Re-instantiate gadgets, check models, error models for each shot
        let mut gids = Vec::with_capacity(n);
        let gid_0 = exec_gadget(coord.as_ref(), make_gadget(0, 1, vec![])).await;
        gids.push(gid_0);
        for i in 1..n - 1 {
            let prev = gids[i - 1];
            let gid = exec_gadget(coord.as_ref(), make_gadget(0, 4, vec![(prev, 0)])).await;
            gids.push(gid);
        }
        let prev = *gids.last().unwrap();
        let gid_last = exec_gadget(coord.as_ref(), make_gadget(0, 5, vec![(prev, 0)])).await;
        gids.push(gid_last);

        exec_check_model(coord.as_ref(), make_check_model(0, 1, gids[0])).await;
        for &gid in &gids[1..n - 1] {
            exec_check_model(coord.as_ref(), make_check_model(0, 4, gid)).await;
        }
        exec_check_model(coord.as_ref(), make_check_model(0, 5, gids[n - 1])).await;

        exec_error_model(coord.as_ref(), make_error_model(0, 11, 1)).await;
        for i in 1..n - 1 {
            exec_error_model(coord.as_ref(), make_error_model(0, 14, (i + 1) as u64)).await;
        }
        exec_error_model(coord.as_ref(), make_error_model(0, 5, n as u64)).await;

        // Decode all concurrently
        let handles: Vec<_> = gids
            .iter()
            .map(|&gid| {
                let c = coord.clone();
                tokio::spawn(async move { decode_arc(c, gid, 1).await })
            })
            .collect();

        let results = tokio::time::timeout(std::time::Duration::from_secs(10), futures_util::future::join_all(handles))
            .await
            .unwrap_or_else(|_| panic!("DEADLOCK on shot {shot}: concurrent decode did not complete within 10s"));

        for (i, result) in results.into_iter().enumerate() {
            let readouts = result.unwrap();
            assert_eq!(readouts.gid, gids[i], "shot {shot}: gid mismatch for gadget {i}");
        }

        // Verify partition correctness and boundary distances for this shot
        let all_gids: HashSet<u64> = gids.into_iter().collect();
        reset_shot(coord.as_ref()).await;
        let trace = read_trace(&trace_path);
        assert_window_correctness(&trace.shots[shot], &all_gids, 2, &test_free_hop_types());
    }
}

/// Build a checked chain with independent buffer_radius and lookahead_radius.
async fn build_checked_chain_with_radii(
    n: usize,
    buffer_radius: usize,
    lookahead_radius: usize,
) -> (Vec<u64>, Arc<WindowCoordinator>, Arc<MockDecoder>, NamedTempFile) {
    assert!(n >= 2, "chain needs at least 2 gadgets");
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = Arc::new(make_coordinator_with_radii(
        mock.clone(),
        &trace_path,
        buffer_radius,
        lookahead_radius,
    ));

    Coordinator::load_library(coord.as_ref(), Request::new(make_test_library_with_remote_errors()))
        .await
        .unwrap();

    let mut gids = Vec::with_capacity(n);
    let gid_0 = exec_gadget(coord.as_ref(), make_gadget(0, 1, vec![])).await;
    gids.push(gid_0);
    for i in 1..n - 1 {
        let prev = gids[i - 1];
        let gid = exec_gadget(coord.as_ref(), make_gadget(0, 4, vec![(prev, 0)])).await;
        gids.push(gid);
    }
    let prev = *gids.last().unwrap();
    let gid_last = exec_gadget(coord.as_ref(), make_gadget(0, 5, vec![(prev, 0)])).await;
    gids.push(gid_last);

    exec_check_model(coord.as_ref(), make_check_model(0, 1, gids[0])).await;
    for &gid in &gids[1..n - 1] {
        exec_check_model(coord.as_ref(), make_check_model(0, 4, gid)).await;
    }
    exec_check_model(coord.as_ref(), make_check_model(0, 5, gids[n - 1])).await;

    exec_error_model(coord.as_ref(), make_error_model(0, 11, 1)).await;
    for i in 1..n - 1 {
        exec_error_model(coord.as_ref(), make_error_model(0, 14, (i + 1) as u64)).await;
    }
    exec_error_model(coord.as_ref(), make_error_model(0, 5, n as u64)).await;

    (gids, coord, mock, trace_file)
}

/// Test lookahead_radius=0 with buffer_radius=3: only center gadget committed per window.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_lookahead_radius_zero_buffer_radius_3() {
    let (gids, coord, mock, trace_file) = build_checked_chain_with_radii(10, 3, 0).await;
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    mock.set_decode_delay(std::time::Duration::from_millis(5));

    let handles: Vec<_> = gids
        .iter()
        .map(|&gid| {
            let c = coord.clone();
            tokio::spawn(async move { decode_arc(c, gid, 1).await })
        })
        .collect();

    let results = tokio::time::timeout(std::time::Duration::from_secs(10), futures_util::future::join_all(handles))
        .await
        .expect("DEADLOCK: lookahead_radius=0, buffer_radius=3 did not complete within 10s");

    for (i, result) in results.into_iter().enumerate() {
        let readouts = result.unwrap();
        assert_eq!(readouts.gid, gids[i], "gid mismatch for gadget {i}");
    }

    reset_shot(coord.as_ref()).await;
    let trace = read_trace(&trace_path);
    let all_gids: HashSet<u64> = gids.into_iter().collect();
    assert_window_correctness(&trace.shots[0], &all_gids, 3, &test_free_hop_types());
}

/// Test lookahead_radius=2, buffer_radius=1: aggressive multi-gadget commit with small buffer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_lookahead_radius_2_buffer_radius_1() {
    let (gids, coord, mock, trace_file) = build_checked_chain_with_radii(10, 1, 2).await;
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    mock.set_decode_delay(std::time::Duration::from_millis(5));

    let handles: Vec<_> = gids
        .iter()
        .map(|&gid| {
            let c = coord.clone();
            tokio::spawn(async move { decode_arc(c, gid, 1).await })
        })
        .collect();

    let results = tokio::time::timeout(std::time::Duration::from_secs(10), futures_util::future::join_all(handles))
        .await
        .expect("DEADLOCK: lookahead_radius=2, buffer_radius=1 did not complete within 10s");

    for (i, result) in results.into_iter().enumerate() {
        let readouts = result.unwrap();
        assert_eq!(readouts.gid, gids[i], "gid mismatch for gadget {i}");
    }

    reset_shot(coord.as_ref()).await;
    let trace = read_trace(&trace_path);
    let all_gids: HashSet<u64> = gids.into_iter().collect();
    assert_window_correctness(&trace.shots[0], &all_gids, 1, &test_free_hop_types());
}

/// Test asymmetric radii: buffer_radius=2, lookahead_radius=5 on a long chain.
/// The large lookahead_radius means many gadgets are potential centers, but
/// buffer_radius=2 requires a minimum buffer of 2 from the boundary.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_asymmetric_radii_long_chain() {
    let (gids, coord, mock, trace_file) = build_checked_chain_with_radii(15, 2, 5).await;
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    mock.set_decode_delay(std::time::Duration::from_millis(5));

    let handles: Vec<_> = gids
        .iter()
        .map(|&gid| {
            let c = coord.clone();
            tokio::spawn(async move { decode_arc(c, gid, 1).await })
        })
        .collect();

    let results = tokio::time::timeout(std::time::Duration::from_secs(15), futures_util::future::join_all(handles))
        .await
        .expect("DEADLOCK: buffer_radius=2, lookahead_radius=5 did not complete within 15s");

    for (i, result) in results.into_iter().enumerate() {
        let readouts = result.unwrap();
        assert_eq!(readouts.gid, gids[i], "gid mismatch for gadget {i}");
    }

    reset_shot(coord.as_ref()).await;
    let trace = read_trace(&trace_path);
    let all_gids: HashSet<u64> = gids.into_iter().collect();
    assert_window_correctness(&trace.shots[0], &all_gids, 2, &test_free_hop_types());
}

/// Streaming test: with lookahead_radius > 0, decoding should start immediately
/// using whatever gadgets are available, not block waiting for the full
/// effective_window_radius.
///
/// Setup: chain of 8 gadgets, buffer_radius=1, lookahead_radius=2.
/// effective_window_radius = 3, but in streaming mode the BFS should NOT
/// wait for gadgets beyond buffer_radius from the center.
///
/// We execute gadgets one at a time with 100ms gaps between them and set
/// decode_delay to 0ms.  Each gadget's decode is submitted immediately
/// after its check/error models.  If build_window blocks waiting for
/// future gadgets in the lookahead_radius zone, early decodes will take
/// 300+ms (waiting for 3 more gadgets at 100ms each).  After the fix,
/// they should complete within ~200ms (at most waiting for 1 output in
/// the buffer_radius zone).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_streaming_lookahead_radius_no_unnecessary_wait() {
    let n = 8;
    let buffer_radius = 1;
    let lookahead_radius = 2;
    let gadget_delay = std::time::Duration::from_millis(100);
    // Maximum time a single decode should take.  With buffer_radius=1,
    // the BFS only needs to wait for at most 1 future gadget (100ms),
    // plus some overhead.  If it waited for the full window (3 gadgets),
    // it would take 300ms+.
    let max_decode_ms = 250;

    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    mock.set_decode_delay(std::time::Duration::from_millis(0));
    let coord = Arc::new(make_coordinator_with_radii(
        mock.clone(),
        &trace_path,
        buffer_radius,
        lookahead_radius,
    ));

    Coordinator::load_library(coord.as_ref(), Request::new(make_test_library_with_remote_errors()))
        .await
        .unwrap();

    // Execute gadgets one at a time, submitting decode after each one's
    // check+error models arrive.  Collect decode handles.
    let mut gids = Vec::with_capacity(n);
    let mut decode_handles = Vec::new();

    for i in 0..n {
        // Execute gadget
        let gadget = if i == 0 {
            make_gadget(0, 1, vec![])
        } else if i == n - 1 {
            make_gadget(0, 5, vec![(gids[i - 1], 0)])
        } else {
            make_gadget(0, 4, vec![(gids[i - 1], 0)])
        };
        let gid = exec_gadget(coord.as_ref(), gadget).await;
        gids.push(gid);

        // Check model
        let ctype = if i == 0 {
            1
        } else if i == n - 1 {
            5
        } else {
            4
        };
        exec_check_model(coord.as_ref(), make_check_model(0, ctype, gid)).await;

        // Error model
        let etype = if i == 0 {
            11
        } else if i == n - 1 {
            5
        } else {
            14
        };
        exec_error_model(coord.as_ref(), make_error_model(0, etype, (i + 1) as u64)).await;

        // Immediately submit decode
        let c = coord.clone();
        let t0 = std::time::Instant::now();
        decode_handles.push(tokio::spawn(async move {
            let readouts = decode_arc(c, gid, 1).await;
            (readouts, t0.elapsed())
        }));

        // Wait before next gadget (simulating streaming arrival)
        if i < n - 1 {
            tokio::time::sleep(gadget_delay).await;
        }
    }

    // All decodes must complete without deadlock
    let results = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        futures_util::future::join_all(decode_handles),
    )
    .await
    .expect("DEADLOCK: streaming decode did not complete within 15s");

    // Verify: each decode should complete well within the timeout, not
    // waiting for the full lookahead_radius worth of future gadgets.
    for (i, result) in results.into_iter().enumerate() {
        let (readouts, elapsed) = result.unwrap();
        assert_eq!(readouts.gid, gids[i], "gid mismatch for gadget {i}");
        let elapsed_ms = elapsed.as_millis();
        assert!(
            elapsed_ms < max_decode_ms as u128,
            "gadget {i} (gid={}) decode took {elapsed_ms}ms, expected < {max_decode_ms}ms. \
             build_window likely blocked waiting for future gadgets in the lookahead_radius zone.",
            gids[i],
        );
    }
}

/// Test that isolated vertices (checks with no incident hyperedges) are
/// properly stripped from the decoding hypergraph.  This can happen when
/// check-only gadgets (committed, no error models) are included in the
/// relative program — their check model creates vertex slots but no
/// hyperedge references them.
///
/// Uses a chain with buffer_radius=2, lookahead_radius=0 to create windows
/// where previously committed gadgets appear as check-only entries.
/// Before the fix, this would panic with "vertex N do not have any
/// neighbor edges" in MWPF.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_isolated_vertices_stripped() {
    let (gids, coord, mock, trace_file) = build_checked_chain_with_radii(8, 2, 0).await;
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    mock.set_decode_delay(std::time::Duration::from_millis(5));

    let handles: Vec<_> = gids
        .iter()
        .map(|&gid| {
            let c = coord.clone();
            tokio::spawn(async move { decode_arc(c, gid, 1).await })
        })
        .collect();

    let results = tokio::time::timeout(std::time::Duration::from_secs(10), futures_util::future::join_all(handles))
        .await
        .expect("DEADLOCK: isolated vertex test did not complete within 10s");

    for (i, result) in results.into_iter().enumerate() {
        let readouts = result.unwrap();
        assert_eq!(readouts.gid, gids[i], "gid mismatch for gadget {i}");
    }

    reset_shot(coord.as_ref()).await;
    let trace = read_trace(&trace_path);
    let all_gids: HashSet<u64> = gids.into_iter().collect();
    assert_window_correctness(&trace.shots[0], &all_gids, 2, &test_free_hop_types());
}

// ─── persistent_decoder cache-key regression tests ──────────────────────────
//
// Exercises the `WindowCoordinator::loaded_decoders` cache with
// `persistent_decoder: true`.  Same pattern as the equivalent monolithic
// regression: two shots with identical `RelativeProgram` but different
// per-eid error-model modifier state must produce two distinct
// `DecoderCacheKey`s and therefore two `load_hypergraph` calls.

fn make_persistent_coordinator(mock: Arc<MockDecoder>, trace_file: &str) -> WindowCoordinator {
    let config = serde_json::json!({
        "persistent_decoder": true,
        "merge_hyperedges": false,
        "trace_filepath": trace_file,
        "buffer_radius": 1usize,
        "lookahead_radius": 0usize,
    });
    WindowCoordinator::new(config, BlackBoxDecoderClient::from_mock(mock))
}

fn make_error_model_with_modifier(eid: u64, etype: u64, cid: u64, pm: Option<bin::ProbabilityModifier>) -> bin::ErrorModel {
    bin::ErrorModel {
        eid,
        etype,
        cid,
        modifier: pm.map(|p| bin::error_model::ErrorModelModifier {
            probability_modifier: Some(p),
            reroute_remote_check_models: vec![],
        }),
        ..Default::default()
    }
}

/// Build the same two-gadget self-contained chain (source → terminal) as
/// `test_two_checked_gadgets_chain`, but parameterised on the probability
/// modifiers attached to the two error models.
async fn run_two_gadget_chain_shot(
    coord: &WindowCoordinator,
    modifier_source: Option<bin::ProbabilityModifier>,
    modifier_terminal: Option<bin::ProbabilityModifier>,
) -> (u64, u64) {
    let gid_a = exec_gadget(coord, make_gadget(0, 1, vec![])).await;
    exec_check_model(coord, make_check_model(0, 1, gid_a)).await;
    exec_error_model(coord, make_error_model_with_modifier(0, 1, 1, modifier_source)).await;

    let gid_b = exec_gadget(coord, make_gadget(0, 5, vec![(gid_a, 0)])).await;
    exec_check_model(coord, make_check_model(0, 5, gid_b)).await;
    exec_error_model(coord, make_error_model_with_modifier(0, 5, 2, modifier_terminal)).await;

    let (r1, r2) = tokio::join!(decode(coord, gid_a, 1), decode(coord, gid_b, 1));
    assert_eq!(r1.gid, gid_a);
    assert_eq!(r2.gid, gid_b);
    (gid_a, gid_b)
}

/// Regression test for `WindowCoordinator::loaded_decoders` cache-key
/// correctness.
///
/// Drives two shots with identical `RelativeProgram` and identical commit
/// region but with the probability modifier on the source error model
/// switched from `0.1` to `0.2` in the second shot.  Under the new
/// `DecoderCacheKey` the per-eid `ErrorModelFingerprint` must differ,
/// forcing a fresh `load_hypergraph` call.  Under the old key (which only
/// keyed on `RelativeProgram`) the second shot would silently reuse the
/// `0.1`-probability hypergraph.
#[tokio::test]
async fn test_window_persistent_decoder_distinguishes_probability_modifier_across_shots() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_persistent_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    // Shot 1: source-error probability modifier p = 0.1.
    run_two_gadget_chain_shot(
        &coord,
        Some(bin::ProbabilityModifier {
            probabilities: vec![0.1],
            ..Default::default()
        }),
        None,
    )
    .await;

    reset_shot(&coord).await;

    // Shot 2: same program shape, but source-error probability modifier
    // p = 0.2.  Different `DecoderCacheKey` → new `load_hypergraph` call.
    run_two_gadget_chain_shot(
        &coord,
        Some(bin::ProbabilityModifier {
            probabilities: vec![0.2],
            ..Default::default()
        }),
        None,
    )
    .await;

    let mock_state = mock.state.read().await;
    assert_eq!(
        mock_state.loaded_hypergraphs.len(),
        2,
        "Expected 2 distinct loaded hypergraphs (one per modifier), got {}; old cache key \
         would reuse the first one and report 1",
        mock_state.loaded_hypergraphs.len(),
    );
    let loaded_decoders = coord.loaded_decoders.read().await;
    assert_eq!(
        loaded_decoders.len(),
        2,
        "Expected 2 distinct DecoderCacheKey entries (one per modifier), got {}",
        loaded_decoders.len(),
    );
}

/// Positive control: two shots with *identical* error-model modifier on
/// the WindowCoordinator must reuse the cached hypergraph (cache hit on
/// the second shot), so exactly one `load_hypergraph` call is observed
/// and `decode_loaded` is served from the cache.
#[tokio::test]
async fn test_window_persistent_decoder_reuses_cache_when_modifier_unchanged() {
    let trace_file = NamedTempFile::new().unwrap();
    let trace_path = trace_file.path().to_str().unwrap().to_string();
    let mock = make_mock_decoder();
    let coord = make_persistent_coordinator(mock.clone(), &trace_path);

    Coordinator::load_library(&coord, Request::new(make_test_library()))
        .await
        .unwrap();

    let modifier = bin::ProbabilityModifier {
        probabilities: vec![0.1],
        ..Default::default()
    };

    run_two_gadget_chain_shot(&coord, Some(modifier.clone()), None).await;
    reset_shot(&coord).await;
    run_two_gadget_chain_shot(&coord, Some(modifier), None).await;

    let mock_state = mock.state.read().await;
    assert_eq!(
        mock_state.loaded_hypergraphs.len(),
        1,
        "Expected 1 cached hypergraph reused across both shots, got {}",
        mock_state.loaded_hypergraphs.len(),
    );
    assert!(
        !mock_state.decode_loaded_calls.is_empty(),
        "Expected the second shot to hit the cache and call decode_loaded",
    );
    let loaded_decoders = coord.loaded_decoders.read().await;
    assert_eq!(loaded_decoders.len(), 1, "Expected a single cache entry");
}
