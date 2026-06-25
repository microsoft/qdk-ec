//! Effective type comparison for testing
//!
//! Provides utilities to compare recorded coordinator state (with modifiers expanded)
//! against expected static library output, normalizing type IDs and ignoring
//! visualization-only fields.
//!
//! This module is used for testing JIT controller output by verifying that the
//! types and instances sent to the MockCoordinator match expected values.

use deq_runtime::bin::{self, error_model_type};
use deq_runtime::coordinator::mock_coordinator::{
    EffectiveCheckModelType, EffectiveErrorModelType, EffectiveTypes, MockCoordinator,
};
use hashbrown::HashMap;

/// Normalized check for structural comparison (excludes visualization).
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormalizedCheck {
    pub measurements: Vec<NormalizedRemoteMeasurement>,
    pub naturally_flipped: bool,
}

/// Normalized remote measurement for structural comparison.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormalizedRemoteMeasurement {
    /// The resolved gid of the remote gadget, or `None` for local measurements.
    pub resolved_gid: Option<u64>,
    pub measurement_index: u64,
}

/// Normalized error for structural comparison (excludes visualization).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormalizedError {
    pub checks: Vec<NormalizedRemoteCheck>,
    pub residual: Vec<u64>,
    pub readout_flips: Vec<u64>,
    pub probability_bits: u64,
}

impl NormalizedError {
    #[allow(dead_code)]
    fn with_resolved_cids(error: &error_model_type::Error, resolved_cids: &[u64]) -> Self {
        Self {
            checks: error
                .checks
                .iter()
                .map(|rc| NormalizedRemoteCheck {
                    resolved_cid: rc.remote_check_model.map(|idx| resolved_cids[idx as usize]),
                    check_index: rc.check_index,
                })
                .collect(),
            residual: error.residual.clone(),
            readout_flips: error.readout_flips.clone(),
            probability_bits: error.probability.to_bits(),
        }
    }
}

/// Normalized remote check for structural comparison.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormalizedRemoteCheck {
    /// The resolved cid of the remote check model, or `None` for local checks.
    pub resolved_cid: Option<u64>,
    pub check_index: u64,
}

/// Normalized check model type for structural comparison.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedCheckModelType {
    pub gtype: u64,
    pub checks: Vec<NormalizedCheck>,
    pub resolved_remote_gadgets: Vec<u64>,
}

/// Normalized error model type for structural comparison.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedErrorModelType {
    pub ctype: u64,
    pub errors: Vec<NormalizedError>,
    pub resolved_remote_check_models: Vec<u64>,
}

/// Result of equivalence comparison.
#[derive(Debug)]
pub struct EquivalenceResult {
    pub equivalent: bool,
    pub differences: Vec<String>,
}

impl EquivalenceResult {
    pub fn assert_equivalent(&self) {
        if !self.equivalent {
            panic!("Libraries are not equivalent:\n{}", self.differences.join("\n"));
        }
    }
}

/// Compare effective types from MockCoordinator against expected static output.
///
/// This function assumes one-to-one gid correspondence between effective and expected:
/// - cid == gid for check models
/// - eid == cid == gid for error models
///
/// It directly matches check/error models by their gid, making the comparison O(n).
pub fn compare_effective_types(effective: &EffectiveTypes, expected: &bin::Library) -> EquivalenceResult {
    let mut differences = Vec::new();

    // Build expected type maps by ctype/etype
    let expected_cmts: HashMap<u64, &bin::CheckModelType> =
        expected.check_model_types.iter().map(|cmt| (cmt.ctype, cmt)).collect();
    let expected_emts: HashMap<u64, &bin::ErrorModelType> =
        expected.error_model_types.iter().map(|emt| (emt.etype, emt)).collect();

    // Build gid -> ctype and cid -> etype from program instructions
    let mut gid_to_ctype: HashMap<u64, u64> = HashMap::new();
    let mut cid_to_etype: HashMap<u64, u64> = HashMap::new();

    for instruction in &expected.program {
        if let Some(create) = &instruction.create {
            match create {
                bin::instruction::Create::CheckModel(cm) => {
                    gid_to_ctype.insert(cm.gid, cm.ctype);
                }
                bin::instruction::Create::ErrorModel(em) => {
                    cid_to_etype.insert(em.cid, em.etype);
                }
                _ => {}
            }
        }
    }

    // Check counts
    if effective.check_model_types.len() != gid_to_ctype.len() {
        differences.push(format!(
            "Check model count mismatch: got {}, expected {}",
            effective.check_model_types.len(),
            gid_to_ctype.len()
        ));
    }
    if effective.error_model_types.len() != cid_to_etype.len() {
        differences.push(format!(
            "Error model count mismatch: got {}, expected {}",
            effective.error_model_types.len(),
            cid_to_etype.len()
        ));
    }

    // Compare check models by gid (cid == gid in our system)
    for (&cid, effective_cmt) in &effective.check_model_types {
        let gid = cid;
        match gid_to_ctype.get(&gid) {
            Some(&expected_ctype) => match expected_cmts.get(&expected_ctype) {
                Some(&expected_cmt) => {
                    if !compare_check_model_type_structure(effective_cmt, expected_cmt) {
                        differences.push(format!(
                            "Check model type mismatch at gid={}: effective {:?}",
                            gid, effective_cmt
                        ));
                    }
                }
                None => {
                    differences.push(format!(
                        "Expected check model type not found for ctype={} at gid={}",
                        expected_ctype, gid
                    ));
                }
            },
            None => {
                differences.push(format!("No expected check model at gid={}", gid));
            }
        }
    }

    // Compare error models by cid (eid == cid == gid in our system)
    for (&eid, effective_emt) in &effective.error_model_types {
        let cid = eid;
        match cid_to_etype.get(&cid) {
            Some(&expected_etype) => match expected_emts.get(&expected_etype) {
                Some(&expected_emt) => {
                    if !compare_error_model_type_structure(effective_emt, expected_emt) {
                        differences.push(format!(
                            "Error model type mismatch at cid={}: effective {:?}",
                            cid, effective_emt
                        ));
                    }
                }
                None => {
                    differences.push(format!(
                        "Expected error model type not found for etype={} at cid={}",
                        expected_etype, cid
                    ));
                }
            },
            None => {
                differences.push(format!("No expected error model at cid={}", cid));
            }
        }
    }

    EquivalenceResult {
        equivalent: differences.is_empty(),
        differences,
    }
}

fn compare_check_model_type_structure(effective: &EffectiveCheckModelType, expected: &bin::CheckModelType) -> bool {
    if effective.gtype != expected.gtype {
        return false;
    }

    if effective.checks.len() != expected.checks.len() {
        return false;
    }

    // Compare checks - need to match by structure, not order
    // For now, use ordered comparison but compare measurements as sets
    for (eff_check, exp_check) in effective.checks.iter().zip(expected.checks.iter()) {
        if eff_check.naturally_flipped != exp_check.naturally_flipped {
            return false;
        }
        if eff_check.measurements.len() != exp_check.measurements.len() {
            return false;
        }
        // Compare measurements as sets (order-independent)
        let eff_meas_set: std::collections::HashSet<_> = eff_check
            .measurements
            .iter()
            .map(|m| {
                // Normalize remote_gadget: map index to resolved gid for comparison
                let resolved_gid = m.remote_gadget.map(|idx| effective.remote_gadgets[idx as usize]);
                (resolved_gid, m.measurement_index)
            })
            .collect();
        let exp_meas_set: std::collections::HashSet<_> = exp_check
            .measurements
            .iter()
            .map(|m| {
                // For expected: remote_gadget index maps to expected.remote_gadgets
                let resolved_gid = m.remote_gadget.map(|idx| {
                    expected.remote_gadgets[idx as usize]
                        .absolute_gid
                        .expect("expected remote gadget should have absolute_gid")
                });
                (resolved_gid, m.measurement_index)
            })
            .collect();
        if eff_meas_set != exp_meas_set {
            return false;
        }
    }

    true
}

fn compare_error_model_type_structure(effective: &EffectiveErrorModelType, expected: &bin::ErrorModelType) -> bool {
    // We're matching by gid, so we don't need to compare ctype here
    // (the caller already matched by gid)

    if effective.errors.len() != expected.errors.len() {
        return false;
    }

    for (eff_error, exp_error) in effective.errors.iter().zip(expected.errors.iter()) {
        if (eff_error.probability - exp_error.probability).abs() > 1e-9 {
            return false;
        }
        if eff_error.residual != exp_error.residual {
            return false;
        }
        if eff_error.readout_flips != exp_error.readout_flips {
            return false;
        }
        if eff_error.checks.len() != exp_error.checks.len() {
            return false;
        }
        // Compare checks as a set (order-independent)
        let eff_checks_set: std::collections::HashSet<_> = eff_error
            .checks
            .iter()
            .map(|c| {
                let resolved_cid = c.remote_check_model.map(|idx| {
                    expected.remote_check_models[idx as usize]
                        .absolute_cid
                        .expect("expected remote check model should have absolute_cid")
                });
                (resolved_cid, c.check_index)
            })
            .collect();
        let exp_checks_set: std::collections::HashSet<_> = exp_error
            .checks
            .iter()
            .map(|c| {
                let resolved_cid = c.remote_check_model.map(|idx| {
                    expected.remote_check_models[idx as usize]
                        .absolute_cid
                        .expect("expected remote check model should have absolute_cid")
                });
                (resolved_cid, c.check_index)
            })
            .collect();
        if eff_checks_set != exp_checks_set {
            return false;
        }
    }

    true
}

/// Assert that the recorded state in MockCoordinator is semantically equivalent
/// to the expected static library output.
///
/// This is the main entry point for testing JIT controller output.
pub async fn assert_effective_types_equivalent(mock: &MockCoordinator, expected: &bin::Library) {
    let effective = mock.get_effective_types().await;
    let result = compare_effective_types(&effective, expected);
    result.assert_equivalent();
}

/// Compare gadget types for structural equivalence (excluding visualization).
pub fn compare_gadget_types(actual: &bin::GadgetType, expected: &bin::GadgetType) -> bool {
    if actual.gtype != expected.gtype {
        return false;
    }

    if actual.measurements.len() != expected.measurements.len() {
        return false;
    }

    if actual.inputs.len() != expected.inputs.len() {
        return false;
    }
    for (a, e) in actual.inputs.iter().zip(expected.inputs.iter()) {
        if a.ptype != e.ptype {
            return false;
        }
    }

    if actual.outputs.len() != expected.outputs.len() {
        return false;
    }
    for (a, e) in actual.outputs.iter().zip(expected.outputs.iter()) {
        if a.ptype != e.ptype {
            return false;
        }
    }

    if actual.readouts.len() != expected.readouts.len() {
        return false;
    }
    for (a, e) in actual.readouts.iter().zip(expected.readouts.iter()) {
        if a.measurement_indices != e.measurement_indices {
            return false;
        }
    }

    if actual.correction_propagation != expected.correction_propagation {
        return false;
    }
    if actual.readout_propagation != expected.readout_propagation {
        return false;
    }
    if actual.logical_correction != expected.logical_correction {
        return false;
    }
    if actual.physical_correction != expected.physical_correction {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use deq_runtime::bin::gadget_type;

    #[test]
    fn test_normalized_error_probability_bits() {
        let error1 = error_model_type::Error {
            probability: 0.1,
            ..Default::default()
        };
        let error2 = error_model_type::Error {
            probability: 0.1,
            ..Default::default()
        };
        let error3 = error_model_type::Error {
            probability: 0.2,
            ..Default::default()
        };

        let norm1 = NormalizedError::with_resolved_cids(&error1, &[]);
        let norm2 = NormalizedError::with_resolved_cids(&error2, &[]);
        let norm3 = NormalizedError::with_resolved_cids(&error3, &[]);

        assert_eq!(norm1, norm2);
        assert_ne!(norm1, norm3);
    }

    #[test]
    fn test_compare_gadget_types_equal() {
        let gt1 = bin::GadgetType {
            gtype: 1,
            name: "test1".to_string(),
            measurements: vec![gadget_type::Measurement::default()],
            ..Default::default()
        };
        let gt2 = bin::GadgetType {
            gtype: 1,
            name: "different_name".to_string(),
            measurements: vec![gadget_type::Measurement {
                tag: "different_tag".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };

        assert!(compare_gadget_types(&gt1, &gt2));
    }

    #[test]
    fn test_compare_gadget_types_different_measurements() {
        let gt1 = bin::GadgetType {
            gtype: 1,
            measurements: vec![gadget_type::Measurement::default()],
            ..Default::default()
        };
        let gt2 = bin::GadgetType {
            gtype: 1,
            measurements: vec![gadget_type::Measurement::default(), gadget_type::Measurement::default()],
            ..Default::default()
        };

        assert!(!compare_gadget_types(&gt1, &gt2));
    }
}
