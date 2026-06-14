//! Integration tests for MockCoordinator

use deq_runtime::bin::{self, instruction};
use deq_runtime::coordinator::MockCoordinator;
use deq_runtime::coordinator::coordinator_server;
use tonic::Request;

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

fn make_library() -> bin::Library {
    bin::Library { ..Default::default() }
}

/// Load a library with common types used by tests into the coordinator.
async fn load_test_library(coordinator: &MockCoordinator) {
    let mut library = make_library();
    library.gadget_types.push(bin::GadgetType {
        gtype: 1,
        ..Default::default()
    });
    library.check_model_types.push(bin::CheckModelType {
        ctype: 2,
        gtype: 1,
        ..Default::default()
    });
    library.error_model_types.push(bin::ErrorModelType {
        etype: 3,
        ctype: 2,
        ..Default::default()
    });
    coordinator_server::Coordinator::load_library(coordinator, Request::new(library))
        .await
        .unwrap();
}

#[tokio::test]
async fn test_load_library_stores_types() {
    let coordinator = MockCoordinator::new();

    let mut library = make_library();
    library.gadget_types.push(bin::GadgetType {
        gtype: 1,
        ..Default::default()
    });
    library.check_model_types.push(bin::CheckModelType {
        ctype: 2,
        gtype: 1,
        ..Default::default()
    });
    library.error_model_types.push(bin::ErrorModelType {
        etype: 3,
        ctype: 2,
        ..Default::default()
    });

    coordinator_server::Coordinator::load_library(&*coordinator, Request::new(library))
        .await
        .unwrap();

    let state = coordinator.state.read().await;
    assert!(state.gadget_types.contains_key(&1));
    assert!(state.check_model_types.contains_key(&2));
    assert!(state.error_model_types.contains_key(&3));
    assert_eq!(state.libraries.len(), 1);
}

#[tokio::test]
async fn test_execute_creates_gadget() {
    let coordinator = MockCoordinator::new();
    load_test_library(&coordinator).await;

    let gadget = make_gadget(0, 1, vec![]);
    let instruction = bin::Instruction {
        create: Some(instruction::Create::Gadget(gadget)),
    };

    let response = coordinator_server::Coordinator::execute(&*coordinator, Request::new(instruction))
        .await
        .unwrap();

    let assigned_gid = response.into_inner().id;
    assert_eq!(assigned_gid, 1);

    let state = coordinator.state.read().await;
    assert!(state.gadgets.contains_key(&assigned_gid));
    assert_eq!(state.next_gid, 2);
}

#[tokio::test]
async fn test_execute_creates_check_model() {
    let coordinator = MockCoordinator::new();
    load_test_library(&coordinator).await;

    let check_model = make_check_model(0, 2, 1);
    let instruction = bin::Instruction {
        create: Some(instruction::Create::CheckModel(check_model)),
    };

    let response = coordinator_server::Coordinator::execute(&*coordinator, Request::new(instruction))
        .await
        .unwrap();

    let assigned_cid = response.into_inner().id;
    assert_eq!(assigned_cid, 1);

    let state = coordinator.state.read().await;
    assert!(state.check_models.contains_key(&assigned_cid));
    assert_eq!(state.next_cid, 2);
}

#[tokio::test]
async fn test_execute_creates_error_model() {
    let coordinator = MockCoordinator::new();
    load_test_library(&coordinator).await;

    let error_model = make_error_model(0, 3, 1);
    let instruction = bin::Instruction {
        create: Some(instruction::Create::ErrorModel(error_model)),
    };

    let response = coordinator_server::Coordinator::execute(&*coordinator, Request::new(instruction))
        .await
        .unwrap();

    let assigned_eid = response.into_inner().id;
    assert_eq!(assigned_eid, 1);

    let state = coordinator.state.read().await;
    assert!(state.error_models.contains_key(&assigned_eid));
    assert_eq!(state.next_eid, 2);
}

#[tokio::test]
async fn test_gadget_connectors_build_outputs_map() {
    let coordinator = MockCoordinator::new();
    load_test_library(&coordinator).await;

    // Gadget 1 has no inputs (will be assigned gid=1)
    let g1 = make_gadget(0, 1, vec![]);
    // Gadget 2 connects to gadget 1's output port 0 (will be assigned gid=2)
    let g2 = make_gadget(0, 1, vec![(1, 0)]);
    // Gadget 3 connects to gadget 1's output port 1 and gadget 2's output port 0 (will be assigned gid=3)
    let g3 = make_gadget(0, 1, vec![(1, 1), (2, 0)]);

    for gadget in [g1, g2, g3] {
        let instruction = bin::Instruction {
            create: Some(instruction::Create::Gadget(gadget)),
        };
        coordinator_server::Coordinator::execute(&*coordinator, Request::new(instruction))
            .await
            .unwrap();
    }

    let state = coordinator.state.read().await;
    // (1, 0) -> 2: gadget 1's output 0 connects to gadget 2
    assert_eq!(state.outputs.get(&(1, 0)), Some(&2));
    // (1, 1) -> 3: gadget 1's output 1 connects to gadget 3
    assert_eq!(state.outputs.get(&(1, 1)), Some(&3));
    // (2, 0) -> 3: gadget 2's output 0 connects to gadget 3
    assert_eq!(state.outputs.get(&(2, 0)), Some(&3));
}

#[tokio::test]
async fn test_clear_resets_state() {
    let coordinator = MockCoordinator::new();
    load_test_library(&coordinator).await;
    let gadget = make_gadget(0, 1, vec![]);
    let instruction = bin::Instruction {
        create: Some(instruction::Create::Gadget(gadget)),
    };
    coordinator_server::Coordinator::execute(&*coordinator, Request::new(instruction))
        .await
        .unwrap();

    // Clear
    coordinator.clear().await;

    let state = coordinator.state.read().await;
    assert!(state.gadgets.is_empty());
    assert!(state.instructions.is_empty());
    assert_eq!(state.next_gid, 1);
}

#[tokio::test]
async fn test_get_effective_types_basic() {
    let coordinator = MockCoordinator::new();

    // Load library with types
    let mut library = make_library();
    library.gadget_types.push(bin::GadgetType {
        gtype: 1,
        ..Default::default()
    });
    library.check_model_types.push(bin::CheckModelType {
        ctype: 2,
        gtype: 1,
        checks: vec![],
        remote_gadgets: vec![],
        ..Default::default()
    });
    library.error_model_types.push(bin::ErrorModelType {
        etype: 3,
        ctype: 2,
        errors: vec![],
        remote_check_models: vec![],
        ..Default::default()
    });
    coordinator_server::Coordinator::load_library(&*coordinator, Request::new(library))
        .await
        .unwrap();

    // Create instances (coordinator assigns gid=1, cid=1, eid=1)
    let gadget = make_gadget(0, 1, vec![]);
    let gid = coordinator_server::Coordinator::execute(
        &*coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(gadget)),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id;

    let check_model = make_check_model(0, 2, gid);
    let cid = coordinator_server::Coordinator::execute(
        &*coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(check_model)),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id;

    let error_model = make_error_model(0, 3, cid);
    coordinator_server::Coordinator::execute(
        &*coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::ErrorModel(error_model)),
        }),
    )
    .await
    .unwrap();

    // Get effective types
    let effective = coordinator.get_effective_types().await;

    assert_eq!(effective.check_model_types.len(), 1);
    assert_eq!(effective.error_model_types.len(), 1);

    let eff_check = effective.check_model_types.get(&1).unwrap();
    assert_eq!(eff_check.ctype, 2);
    assert_eq!(eff_check.gtype, 1);

    let eff_error = effective.error_model_types.get(&1).unwrap();
    assert_eq!(eff_error.etype, 3);
    assert_eq!(eff_error.ctype, 2);
}

#[tokio::test]
async fn test_mixed_user_and_auto_assigned_ids() {
    let coordinator = MockCoordinator::new();
    load_test_library(&coordinator).await;

    // User explicitly requests gid=2
    let g_user = make_gadget(2, 1, vec![]);
    let response = coordinator_server::Coordinator::execute(
        &*coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(g_user)),
        }),
    )
    .await
    .unwrap();
    assert_eq!(response.into_inner().id, 2);

    // Auto-assign should get gid=1 (first unused)
    let g_auto1 = make_gadget(0, 1, vec![]);
    let response = coordinator_server::Coordinator::execute(
        &*coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(g_auto1)),
        }),
    )
    .await
    .unwrap();
    assert_eq!(response.into_inner().id, 1);

    // Auto-assign should skip gid=2 (already used) and get gid=3
    let g_auto2 = make_gadget(0, 1, vec![]);
    let response = coordinator_server::Coordinator::execute(
        &*coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(g_auto2)),
        }),
    )
    .await
    .unwrap();
    assert_eq!(response.into_inner().id, 3);

    let state = coordinator.state.read().await;
    assert!(state.gadgets.contains_key(&1));
    assert!(state.gadgets.contains_key(&2));
    assert!(state.gadgets.contains_key(&3));
    assert_eq!(state.gadgets.len(), 3);
}
