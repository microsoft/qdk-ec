//! Pauli Frame Tracker
//!
//! Tracking how the logical readout changes on the gadget correction, and
//! propagate the changes accordingly
//!
//! When using this Pauli frame tracker in a committed decoding (e.g. window decoding)
//! where the decoded result of each gadget is never changed once it's committed,
//! this tracker ensures that it will return the `PauliFrameUpdate` only once (but not
//! necessarily in the function call to that gadget, because it's history gadget may
//! not have been committed yet)
//!
//! When the decoder may sometimes update the committed result, it's still permitted
//! but the function will only return the updated logical readouts and ignore those
//! unchanged ones
//!
//! Note that this is a preliminary implementation, where it might have to visit
//! the same gadget multiple times in a single update; a more mature implementation
//! should sort the gadgets so that visiting each gadget once should work. However
//! that would require building more complicated data structures.
//!

use crate::misc::bit_matrix::{apply_modifier, optional_sparse_to_dense};
use crate::misc::bit_vector::get_bit;
use crate::{bin, util};
use binar::{BitMatrix, BitVec, BitwisePairMut};
use hashbrown::HashMap;
use std::sync::Arc;

pub struct PauliFrameTracker {
    pub gadgets: HashMap<u64, PauliFrameGadget>,
}

pub struct PauliFrameGadget {
    /// the current error corrected readout and residual
    frame: Option<PauliFrame>,

    /// the raw value of the readouts before error correction; the values
    /// are loaded only after the measurements are ready for the gadget
    raw_readouts: Option<BitVec>,
    /// the raw measurement outcomes for this gadget; needed for physical_correction
    raw_measurements: Option<BitVec>,
    /// the correction from the decoder; the values are loaded only
    /// after the decoder has committed the errors; note that we still allow
    /// user to update this field, which will trigger an update of the error
    /// corrected readouts of this gadget and/or future gadgets
    decoded: Option<PauliFrame>,

    /// input observable to output observable
    correction_propagation: BitMatrix,
    /// input observable to readouts
    readout_propagation: BitMatrix,
    /// readouts to output observables
    logical_correction: BitMatrix,
    /// measurements to output observables
    physical_correction: BitMatrix,
    /// remote readouts to output observables
    /// (remote_readouts, correction_matrix)
    remote_conditional_correction: Option<(Vec<bin::remote_conditional_correction::RemoteReadout>, BitMatrix)>,

    /// the input connections
    inputs: Vec<bin::gadget::Connector>,
    /// the output peers
    outputs: Vec<Option<bin::gadget::Connector>>,
    /// gadgets that depend on this gadget's readout via remote_conditional_correction
    remote_dependents: Vec<u64>,
    /// the output bias for each port
    output_bias: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauliFrame {
    /// the value of readouts
    readouts: BitVec,
    /// the value of output observables
    residual: BitVec,
}

impl PauliFrameGadget {
    pub fn num_readouts(&self) -> usize {
        self.logical_correction.column_count()
    }

    pub fn num_measurements(&self) -> usize {
        self.physical_correction.column_count()
    }

    pub fn num_input_observables(&self) -> usize {
        self.correction_propagation.column_count() - 1
    }

    pub fn num_output_observables(&self) -> usize {
        self.correction_propagation.row_count()
    }

    pub fn get_residual_slice(&self, residual: &BitVec, port: u64) -> Vec<bool> {
        let start = self.output_bias[port as usize];
        let end = if (port as usize) < self.output_bias.len() - 1 {
            self.output_bias[(port as usize) + 1]
        } else {
            self.num_output_observables()
        };
        use binar::Bitwise;
        (start..end).map(|i| residual.index(i)).collect()
    }
}

impl PauliFrameTracker {
    pub fn new() -> Self {
        Self {
            gadgets: Default::default(),
        }
    }

    pub fn reset(&mut self) {
        self.gadgets.clear();
    }

    pub fn add_gadget(
        &mut self,
        gid: u64,
        gadget_type: &bin::GadgetType,
        gadget_modifier: Option<&bin::GadgetModifier>,
        port_types: &HashMap<u64, Arc<bin::PortType>>,
        connectors: &[bin::gadget::Connector],
    ) {
        debug_assert!(!self.gadgets.contains_key(&gid));
        debug_assert!(gadget_type.inputs.len() == connectors.len());
        // get the port biases
        let mut output_observable_count = 0;
        let mut output_bias = Vec::with_capacity(gadget_type.outputs.len());
        for port in gadget_type.outputs.iter() {
            let port_type = port_types.get(&port.ptype).unwrap();
            output_bias.push(output_observable_count);
            output_observable_count += port_type.observables.len();
        }
        // get matrices and apply modifier if present
        let mut correction_propagation = optional_sparse_to_dense(&gadget_type.correction_propagation);
        let mut readout_propagation = optional_sparse_to_dense(&gadget_type.readout_propagation);
        let mut logical_correction = optional_sparse_to_dense(&gadget_type.logical_correction);
        let mut physical_correction = optional_sparse_to_dense(&gadget_type.physical_correction);
        let mut remote_conditional_correction = None;
        if let Some(modifier) = gadget_modifier {
            if let Some(mod_) = &modifier.correction_propagation_mod {
                correction_propagation = apply_modifier(correction_propagation, mod_);
            }
            if let Some(mod_) = &modifier.readout_propagation_mod {
                readout_propagation = apply_modifier(readout_propagation, mod_);
            }
            if let Some(mod_) = &modifier.logical_correction_mod {
                logical_correction = apply_modifier(logical_correction, mod_);
            }
            if let Some(mod_) = &modifier.physical_correction_mod {
                physical_correction = apply_modifier(physical_correction, mod_);
            }
            if let Some(remote_cc) = &modifier.remote_conditional_correction {
                let matrix = optional_sparse_to_dense(&remote_cc.correction);
                remote_conditional_correction = Some((remote_cc.remote_readouts.clone(), matrix));
            }
        }
        // create the gadget
        let gadget = PauliFrameGadget {
            frame: None,
            raw_readouts: None,
            raw_measurements: None,
            decoded: None,
            correction_propagation,
            readout_propagation,
            logical_correction,
            physical_correction,
            remote_conditional_correction: remote_conditional_correction.clone(),
            inputs: connectors.to_vec(),
            outputs: vec![None; gadget_type.outputs.len()],
            remote_dependents: vec![],
            output_bias,
        };
        debug_assert!(output_observable_count == gadget.num_output_observables());
        debug_assert!(gadget.logical_correction.row_count() == gadget.num_output_observables());
        debug_assert!(gadget_type.measurements.len() == gadget.num_measurements());
        debug_assert!(gadget.readout_propagation.row_count() == gadget.num_readouts());
        debug_assert!(gadget.readout_propagation.column_count() == gadget.num_input_observables() + 1);
        self.gadgets.insert(gid, gadget);
        for (port, connector) in connectors.iter().enumerate() {
            self.gadgets.get_mut(&connector.gid).unwrap().outputs[connector.port as usize]
                .replace(bin::gadget::Connector { gid, port: port as u64 });
        }
        // register remote dependencies so that remote gadgets can trigger propagation
        if let Some((remote_refs, _)) = remote_conditional_correction {
            for remote_ref in remote_refs {
                self.gadgets.get_mut(&remote_ref.gid).unwrap().remote_dependents.push(gid);
            }
        }
    }

    pub fn load_raw(&mut self, gid: u64, raw_readouts: &[bool], raw_measurements: &util::BitVector) {
        let gadget = self.gadgets.get_mut(&gid).unwrap();
        debug_assert!(gadget.raw_readouts.is_none());
        debug_assert!(gadget.raw_measurements.is_none());
        debug_assert!(gadget.num_readouts() == raw_readouts.len());
        debug_assert_eq!(gadget.num_measurements() as u64, raw_measurements.size);
        gadget.raw_readouts.replace(raw_readouts.iter().cloned().collect());
        gadget
            .raw_measurements
            .replace((0..raw_measurements.size).map(|i| get_bit(raw_measurements, i)).collect());
    }

    pub fn load_correction(&mut self, gid: u64, residual: BitVec, readout_flips: BitVec) -> HashMap<u64, util::BitVector> {
        let gadget = self.gadgets.get_mut(&gid).unwrap();
        debug_assert!(
            gadget.raw_readouts.is_some(),
            "raw readouts must be loaded first for gid={gid}"
        );
        gadget.decoded.replace(PauliFrame {
            residual,
            readouts: readout_flips,
        });
        let mut updates = HashMap::new();
        self.propagate_from(gid, &mut updates);
        updates
    }

    fn propagate_from(&mut self, gid: u64, updates: &mut HashMap<u64, util::BitVector>) {
        // first check if all the input peers have done their work; if not, return nothing
        let gadget = self.gadgets.get(&gid).unwrap();
        if gadget.decoded.is_none() {
            return;
        }
        let mut input_observables: Vec<bool> = vec![];
        for peer in gadget.inputs.iter() {
            let peer_gadget = self.gadgets.get(&peer.gid).unwrap();
            if let Some(frame) = peer_gadget.frame.as_ref() {
                input_observables.append(&mut peer_gadget.get_residual_slice(&frame.residual, peer.port));
            } else {
                return;
            }
        }
        // gather remote readouts for remote conditional correction if present
        let remote_readouts_vec: Option<BitVec> = if let Some((remote_refs, _)) = &gadget.remote_conditional_correction {
            let mut values = Vec::with_capacity(remote_refs.len());
            for remote_ref in remote_refs {
                let remote_gadget = self.gadgets.get(&remote_ref.gid).unwrap();
                if let Some(frame) = remote_gadget.frame.as_ref() {
                    use binar::Bitwise;
                    values.push(frame.readouts.index(remote_ref.readout_index as usize));
                } else {
                    return;
                }
            }
            Some(values.into_iter().collect())
        } else {
            None
        };
        // if input peers are all ready, let's calculate my pauli frame
        let input_observables: BitVec = input_observables.into_iter().chain([true]).collect();
        use std::ops::Mul;
        let mut residual = gadget.correction_propagation.mul(&input_observables.as_view());
        // apply physical_correction * measurements
        let raw_measurements = gadget.raw_measurements.as_ref().unwrap();
        residual.bitxor_assign(&gadget.physical_correction.mul(&raw_measurements.as_view()));
        let mut readouts = gadget.raw_readouts.clone().unwrap();
        let decoded = gadget.decoded.as_ref().unwrap();
        readouts.bitxor_assign(&decoded.readouts);
        readouts.bitxor_assign(&gadget.readout_propagation.mul(&input_observables.as_view()));
        residual.bitxor_assign(&gadget.logical_correction.mul(&readouts.as_view()));
        // apply remote conditional correction if present
        if let Some((_, correction_matrix)) = &gadget.remote_conditional_correction {
            residual.bitxor_assign(&correction_matrix.mul(&remote_readouts_vec.unwrap().as_view()));
        }
        residual.bitxor_assign(&decoded.residual);
        // check if the value has changed; if the value is unchanged, no need to propagate further
        let frame = PauliFrame { residual, readouts };
        if gadget.frame.as_ref() == Some(&frame) {
            return;
        }
        let readouts = crate::misc::bit_vector::binar_bitvec_to_bit_vector(&frame.readouts, gadget.num_readouts());
        updates.insert(gid, readouts);
        let gadget = self.gadgets.get_mut(&gid).unwrap();
        gadget.frame = Some(frame);
        let outputs = gadget.outputs.clone();
        let remote_dependents = gadget.remote_dependents.clone();
        for peer in outputs.iter().flatten() {
            self.propagate_from(peer.gid, updates);
        }
        // also propagate to gadgets that depend on this gadget's readout via remote_conditional_correction
        for dependent_gid in remote_dependents {
            self.propagate_from(dependent_gid, updates);
        }
    }
}

impl Default for PauliFrameTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc::bit_matrix::{append_bit, zeros};
    use binar::BitwiseMut;
    use util::BitVector;

    // the ptype=1 port type has 2 observables
    fn build_port_types() -> HashMap<u64, Arc<bin::PortType>> {
        [(
            1,
            Arc::new(bin::PortType {
                ptype: 1,
                name: "surface code".to_string(),
                description: "".to_string(),
                observables: vec![
                    bin::port_type::Observable {
                        tag: "X".to_string(),
                        ..Default::default()
                    },
                    bin::port_type::Observable {
                        tag: "Z".to_string(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
        )]
        .into_iter()
        .collect()
    }

    fn build_gadget_type(num_inputs: usize, num_outputs: usize, num_readouts: usize) -> bin::GadgetType {
        bin::GadgetType {
            gtype: 0, // invalid but nobody checks
            name: "".to_string(),
            description: "".to_string(),
            measurements: vec![], // don't care
            inputs: vec![
                bin::gadget_type::Port {
                    ptype: 1,
                    tag: "".to_string(),
                    ..Default::default()
                };
                num_inputs
            ],
            outputs: vec![
                bin::gadget_type::Port {
                    ptype: 1,
                    tag: "".to_string(),
                    ..Default::default()
                };
                num_outputs
            ],
            correction_propagation: Some(zeros(num_outputs * 2, num_inputs * 2 + 1)),
            readouts: vec![
                bin::gadget_type::Readout {
                    tag: "".to_string(),
                    measurement_indices: vec![],
                    ..Default::default()
                };
                num_readouts
            ],
            readout_propagation: Some(zeros(num_readouts, num_inputs * 2 + 1)),
            logical_correction: Some(zeros(num_outputs * 2, num_readouts)),
            physical_correction: Some(zeros(num_outputs * 2, 0)),
            ..Default::default()
        }
    }

    fn load_correction_to(
        tracker: &mut PauliFrameTracker,
        gid: u64,
        residual_ones: &[usize],
        readout_flips_ones: &[usize],
    ) -> HashMap<u64, BitVector> {
        let gadget = tracker.gadgets.get(&gid).unwrap();
        let mut residual: BitVec = BitVec::zeros(gadget.num_output_observables());
        let mut readout_flips: BitVec = BitVec::zeros(gadget.num_readouts());
        for &index in residual_ones {
            residual.assign_index(index, true);
        }
        for &index in readout_flips_ones {
            readout_flips.assign_index(index, true);
        }
        tracker.load_correction(gid, residual, readout_flips)
    }

    /*
    G1 -->  G2
    a flip in G1 should result in flipped logical readout in G2
     */
    #[test]
    fn pauli_frame_tracker_1() {
        // cargo test pauli_frame_tracker_1 -- --nocapture
        fn prepare_tracker() -> PauliFrameTracker {
            let mut tracker = PauliFrameTracker::new();
            let port_types = build_port_types();
            let gadget_type_1 = build_gadget_type(0, 1, 0);
            tracker.add_gadget(1, &gadget_type_1, None, &port_types, &[]);
            tracker.load_raw(1, &[], &crate::misc::bit_vector::from_sparse_indices(0, &[]));
            let mut gadget_type_2 = build_gadget_type(1, 0, 1);
            append_bit(gadget_type_2.readout_propagation.as_mut().unwrap(), 0, 0);
            tracker.add_gadget(
                2,
                &gadget_type_2,
                None,
                &port_types,
                &[bin::gadget::Connector { gid: 1, port: 0 }],
            );
            tracker.load_raw(2, &[false], &crate::misc::bit_vector::from_sparse_indices(0, &[]));
            tracker
        }

        // 1: sequentially load the decoded results
        let mut tracker = prepare_tracker();
        assert_eq!(
            load_correction_to(&mut tracker, 1, &[], &[]),
            [(1, BitVector { size: 0, data: vec![] })].into()
        );
        assert_eq!(
            load_correction_to(&mut tracker, 2, &[], &[]),
            [(2, BitVector { size: 1, data: vec![0] })].into()
        );

        // 2. try to load a wrong decode result on gid=1 and expect to see flip on gid=2
        let mut tracker = prepare_tracker();
        assert_eq!(
            load_correction_to(&mut tracker, 1, &[0], &[]),
            [(1, BitVector { size: 0, data: vec![] })].into()
        );
        assert_eq!(
            load_correction_to(&mut tracker, 2, &[], &[]),
            [(
                2,
                BitVector {
                    size: 1,
                    data: vec![1 << 7]
                }
            )]
            .into()
        );

        // 3 load gadget 2's decoded result, which should depend on 1's and does nothing
        let mut tracker = prepare_tracker();
        assert_eq!(load_correction_to(&mut tracker, 2, &[], &[]), [].into());
        assert_eq!(
            load_correction_to(&mut tracker, 1, &[0], &[]),
            [
                (1, BitVector { size: 0, data: vec![] }),
                (
                    2,
                    BitVector {
                        size: 1,
                        data: vec![1 << 7]
                    }
                ),
            ]
            .into()
        );
    }

    /*
    Test race condition fix for remote_conditional_correction:

    G_A (source) -- no connection to G_C via port, only via remote_conditional_correction
    G_B (source) --> G_C (has remote_conditional_correction referencing G_A's readout)

    If corrections are loaded in order: B, C, A
    Without the fix, C would never get its frame computed because A's frame is not ready
    when C is processed, and A doesn't trigger C's propagation.

    With the fix, A's propagate_from triggers C's propagate_from via remote_dependents.
    */
    #[test]
    fn pauli_frame_tracker_remote_conditional_correction_race() {
        fn prepare_tracker() -> PauliFrameTracker {
            let mut tracker = PauliFrameTracker::new();
            let port_types = build_port_types();

            // G_A: source gadget with 1 output, 1 readout
            let gadget_type_a = build_gadget_type(0, 1, 1);
            tracker.add_gadget(1, &gadget_type_a, None, &port_types, &[]);
            tracker.load_raw(1, &[true], &crate::misc::bit_vector::from_sparse_indices(0, &[]));

            // G_B: source gadget with 1 output, 0 readouts
            let gadget_type_b = build_gadget_type(0, 1, 0);
            tracker.add_gadget(2, &gadget_type_b, None, &port_types, &[]);
            tracker.load_raw(2, &[], &crate::misc::bit_vector::from_sparse_indices(0, &[]));

            // G_C: takes input from G_B, has remote_conditional_correction from G_A
            // readout_propagation XORs with input observable[0]
            let gadget_type_c = build_gadget_type(1, 0, 1);

            // Build remote_conditional_correction modifier
            // G_C has 0 output observables (no output ports), so correction matrix is 0x1
            let remote_cc = bin::RemoteConditionalCorrection {
                remote_readouts: vec![bin::remote_conditional_correction::RemoteReadout {
                    gid: 1,           // references G_A
                    readout_index: 0, // G_A's readout[0]
                }],
                correction: Some(zeros(0, 1)), // 0 rows (output obs), 1 col (remote readouts)
            };
            let modifier = bin::GadgetModifier {
                remote_conditional_correction: Some(remote_cc),
                ..Default::default()
            };

            tracker.add_gadget(
                3,
                &gadget_type_c,
                Some(&modifier),
                &port_types,
                &[bin::gadget::Connector { gid: 2, port: 0 }],
            );
            tracker.load_raw(3, &[false], &crate::misc::bit_vector::from_sparse_indices(0, &[]));
            tracker
        }

        // Test the race condition scenario: load in order B, C, A
        let mut tracker = prepare_tracker();

        // Load B first - should compute B's frame
        let updates_b = load_correction_to(&mut tracker, 2, &[], &[]);
        assert_eq!(updates_b.len(), 1);
        assert!(updates_b.contains_key(&2), "B should have its frame computed");

        // Load C - should NOT compute C's frame yet (A is not ready)
        let updates_c = load_correction_to(&mut tracker, 3, &[], &[]);
        assert!(updates_c.is_empty(), "C should not have frame yet (A not ready)");

        // Load A - should compute A's frame AND trigger C's frame computation
        let updates_a = load_correction_to(&mut tracker, 1, &[], &[]);
        assert!(updates_a.contains_key(&1), "A should have its frame computed");
        assert!(
            updates_a.contains_key(&3),
            "C should also have its frame computed via remote_dependents"
        );
    }
}
