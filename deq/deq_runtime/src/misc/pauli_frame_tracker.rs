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

use crate::misc::bit_vector::{bit_vector_to_binar_bitvec, get_bit};
use crate::misc::frame_matrix::{FrameMatrix, FrameMatrixCache};
use crate::{bin, util};
use binar::{BitVec, BitwiseMut, BitwisePairMut};
use hashbrown::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

pub struct PauliFrameTracker {
    pub gadgets: HashMap<u64, PauliFrameGadget>,
    retained_readouts: HashMap<u64, RetainedReadout>,
    matrices: FrameMatrixCache,
}

struct RetainedReadout {
    /// Keeps logical readouts available while positive; this entry is removed at zero.
    remaining: u64,
    /// Packed logical readout bits retained after the source gadget is collected.
    value: util::BitVector,
}

pub struct PauliFrameGadget {
    /// Marks a final frame whose raw measurements and decoded correction have been released.
    finalized: bool,
    /// Positive values retain logical readouts after collection; underdeclared hints saturate at zero.
    expected_remote_references: u64,
    /// Cached remote bits in committing mode; inner None entries await their source frames.
    remote_readout_values: Option<Vec<Option<bool>>>,
    /// the current error corrected readout and residual
    frame: Option<PauliFrame>,

    pub(crate) raw_frame: Arc<RawFrameSlot>,
    /// the correction from the decoder; the values are loaded only
    /// after the decoder has committed the errors; note that we still allow
    /// user to update this field, which will trigger an update of the error
    /// corrected readouts of this gadget and/or future gadgets
    decoded: Option<DecodedFrame>,

    /// input observable to output observable
    pub(crate) correction_propagation: Arc<FrameMatrix>,
    /// input observable to readouts
    pub(crate) readout_propagation: Arc<FrameMatrix>,
    /// readouts to output observables
    pub(crate) logical_correction: Arc<FrameMatrix>,
    /// measurements to output observables
    physical_correction: Arc<FrameMatrix>,
    /// remote readouts to output observables
    /// (remote_readouts, correction_matrix)
    pub(crate) remote_conditional_correction:
        Option<(Vec<bin::remote_conditional_correction::RemoteReadout>, Arc<FrameMatrix>)>,

    /// the input connections
    pub(crate) inputs: Vec<bin::gadget::Connector>,
    /// the output peers
    outputs: Vec<Option<bin::gadget::Connector>>,
    /// gadgets that depend on this gadget's readout via remote_conditional_correction
    remote_dependents: Vec<u64>,
    /// the output bias for each port
    pub(crate) output_bias: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauliFrame {
    /// the value of readouts
    readouts: BitVec,
    /// the value of output observables
    residual: BitVec,
}

enum DecodedFrame {
    Dense(PauliFrame),
    Sparse {
        residual_flips: Vec<usize>,
        readout_flips: Vec<usize>,
    },
}

impl DecodedFrame {
    fn apply(&self, residual: &mut BitVec, readouts: &mut BitVec) {
        match self {
            Self::Dense(frame) => {
                residual.bitxor_assign(&frame.residual);
                readouts.bitxor_assign(&frame.readouts);
            }
            Self::Sparse {
                residual_flips,
                readout_flips,
            } => {
                for &index in residual_flips {
                    residual.negate_index(index);
                }
                for &index in readout_flips {
                    readouts.negate_index(index);
                }
            }
        }
    }
}

pub(crate) struct PreparedRawFrame {
    readouts: BitVec,
    measurements: BitVec,
}

impl PreparedRawFrame {
    pub(crate) fn new(readouts: &[bool], measurements: &util::BitVector) -> Self {
        Self {
            readouts: readouts.iter().copied().collect(),
            measurements: bit_vector_to_binar_bitvec(measurements),
        }
    }
}

/// Per-gadget input publication, independent of the frame propagation worker.
///
/// The coordinator fills this slot before publishing outcomes. Propagation reads
/// it only after that publication, and finalization releases the payload.
#[derive(Default)]
pub struct RawFrameSlot {
    raw: Mutex<Option<PreparedRawFrame>>,
}

impl RawFrameSlot {
    pub(crate) fn load(&self, frame: PreparedRawFrame) -> Result<(), &'static str> {
        let mut raw = self.raw.lock().map_err(|_| "raw frame publication lock poisoned")?;
        if raw.is_some() {
            return Err("raw frame already loaded");
        }
        *raw = Some(frame);
        Ok(())
    }
}

impl PauliFrameGadget {
    #[must_use]
    pub fn num_readouts(&self) -> usize {
        self.logical_correction.column_count()
    }

    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.physical_correction.column_count()
    }

    #[must_use]
    pub fn num_input_observables(&self) -> usize {
        self.correction_propagation.column_count() - 1
    }

    #[must_use]
    pub fn num_output_observables(&self) -> usize {
        self.correction_propagation.row_count()
    }

    pub fn get_residual_slice(&self, residual: &BitVec, port: u64) -> Vec<bool> {
        use binar::Bitwise;
        self.residual_range(port).map(|index| residual.index(index)).collect()
    }

    fn residual_range(&self, port: u64) -> std::ops::Range<usize> {
        let start = self.output_bias[port as usize];
        let end = if (port as usize) < self.output_bias.len() - 1 {
            self.output_bias[(port as usize) + 1]
        } else {
            self.num_output_observables()
        };
        start..end
    }
}

impl PauliFrameTracker {
    pub fn new() -> Self {
        Self {
            gadgets: Default::default(),
            retained_readouts: HashMap::new(),
            matrices: FrameMatrixCache::default(),
        }
    }

    pub fn reset(&mut self) {
        self.gadgets.clear();
        self.retained_readouts.clear();
        self.matrices.clear();
    }

    pub(crate) fn validate_remote_readouts(&self, modifier: Option<&bin::GadgetModifier>) -> Result<(), String> {
        if let Some(remote) = modifier.and_then(|modifier| modifier.remote_conditional_correction.as_ref()) {
            for readout in &remote.remote_readouts {
                let count = if let Some(gadget) = self.gadgets.get(&readout.gid) {
                    gadget.num_readouts() as u64
                } else if let Some(retained) = self.retained_readouts.get(&readout.gid) {
                    retained.value.size
                } else {
                    return Err(format!(
                        "logical readout source gid={} is unknown or already collected",
                        readout.gid
                    ));
                };
                if readout.readout_index >= count {
                    return Err(format!(
                        "readout index {} is out of range for gid={}",
                        readout.readout_index, readout.gid
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn add_committing_gadget(
        &mut self,
        gadget: &bin::Gadget,
        gadget_type: &bin::GadgetType,
        port_types: &HashMap<u64, Arc<bin::PortType>>,
    ) -> Result<Arc<RawFrameSlot>, String> {
        self.validate_remote_readouts(gadget.modifier.as_ref())?;
        let references = gadget
            .modifier
            .as_ref()
            .and_then(|modifier| modifier.remote_conditional_correction.as_ref())
            .map(|remote| remote.remote_readouts.as_slice())
            .unwrap_or_default();
        let values = references
            .iter()
            .map(|readout| {
                if let Some(source) = self.gadgets.get(&readout.gid) {
                    source.frame.as_ref().map(|frame| {
                        use binar::Bitwise;
                        frame.readouts.index(usize::try_from(readout.readout_index).unwrap())
                    })
                } else {
                    Some(get_bit(&self.retained_readouts[&readout.gid].value, readout.readout_index))
                }
            })
            .collect();
        let sources: HashSet<_> = references.iter().map(|readout| readout.gid).collect();
        self.add_gadget_inner(
            gadget.gid,
            gadget_type,
            gadget.modifier.as_ref(),
            port_types,
            &gadget.connectors,
            gadget.expected_remote_references,
            Some(values),
        );
        for gid in sources {
            if let Some(source) = self.gadgets.get_mut(&gid) {
                source.expected_remote_references = source.expected_remote_references.saturating_sub(1);
            } else {
                let source = self.retained_readouts.get_mut(&gid).unwrap();
                source.remaining = source.remaining.saturating_sub(1);
                if source.remaining == 0 {
                    self.retained_readouts.remove(&gid);
                }
            }
        }
        Ok(Arc::clone(&self.gadgets[&gadget.gid].raw_frame))
    }

    pub(crate) fn collect_gadget(&mut self, gid: u64) {
        let gadget = self.gadgets.remove(&gid).expect("collected frame must exist");
        let frame = gadget.frame.expect("cannot collect an unfinished frame");
        if gadget.expected_remote_references > 0 && !frame.readouts.is_empty() {
            self.retained_readouts.insert(
                gid,
                RetainedReadout {
                    remaining: gadget.expected_remote_references,
                    value: crate::misc::bit_vector::binar_bitvec_to_bit_vector(&frame.readouts, frame.readouts.len()),
                },
            );
        }
    }

    pub(crate) fn retained_readout_count(&self) -> usize {
        self.retained_readouts.len()
    }

    pub(crate) fn retained_readout_bytes(&self) -> usize {
        self.retained_readouts.values().map(|readout| readout.value.data.len()).sum()
    }

    /// Discard raw measurements and corrections once a gadget's frame is final.
    /// Retain its frame and port layout for future children and remote readouts.
    ///
    /// # Panics
    /// Panics if the gadget does not exist or its frame is not yet available.
    pub fn finalize(&mut self, gid: u64) {
        let gadget = self.gadgets.get_mut(&gid).expect("unknown finalized gadget");
        assert!(gadget.frame.is_some(), "cannot finalize a pending frame");
        gadget.finalized = true;
        *gadget.raw_frame.raw.lock().unwrap() = None;
        gadget.decoded = None;
    }

    pub fn add_gadget(
        &mut self,
        gid: u64,
        gadget_type: &bin::GadgetType,
        gadget_modifier: Option<&bin::GadgetModifier>,
        port_types: &HashMap<u64, Arc<bin::PortType>>,
        connectors: &[bin::gadget::Connector],
    ) {
        self.add_gadget_inner(gid, gadget_type, gadget_modifier, port_types, connectors, 0, None);
    }

    #[allow(clippy::too_many_arguments)]
    fn add_gadget_inner(
        &mut self,
        gid: u64,
        gadget_type: &bin::GadgetType,
        gadget_modifier: Option<&bin::GadgetModifier>,
        port_types: &HashMap<u64, Arc<bin::PortType>>,
        connectors: &[bin::gadget::Connector],
        expected_remote_references: u64,
        remote_readout_values: Option<Vec<Option<bool>>>,
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
        let correction_propagation = self.matrices.prepare(
            gadget_type.correction_propagation.as_ref(),
            gadget_modifier.and_then(|modifier| modifier.correction_propagation_mod.as_ref()),
        );
        let readout_propagation = self.matrices.prepare(
            gadget_type.readout_propagation.as_ref(),
            gadget_modifier.and_then(|modifier| modifier.readout_propagation_mod.as_ref()),
        );
        let logical_correction = self.matrices.prepare(
            gadget_type.logical_correction.as_ref(),
            gadget_modifier.and_then(|modifier| modifier.logical_correction_mod.as_ref()),
        );
        let physical_correction = self.matrices.prepare(
            gadget_type.physical_correction.as_ref(),
            gadget_modifier.and_then(|modifier| modifier.physical_correction_mod.as_ref()),
        );
        let remote_conditional_correction = gadget_modifier
            .and_then(|modifier| modifier.remote_conditional_correction.as_ref())
            .map(|remote| {
                (
                    remote.remote_readouts.clone(),
                    self.matrices.prepare(remote.correction.as_ref(), None),
                )
            });
        // create the gadget
        let gadget = PauliFrameGadget {
            finalized: false,
            expected_remote_references,
            remote_readout_values,
            frame: None,
            raw_frame: Arc::default(),
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
        debug_assert_eq!(gadget_type.readouts.len(), gadget.num_readouts());
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
            let snapshots = self.gadgets[&gid].remote_readout_values.is_some();
            let mut registered = HashSet::new();
            for remote_ref in remote_refs {
                if !registered.insert(remote_ref.gid) {
                    continue;
                }
                if let Some(source) = self.gadgets.get_mut(&remote_ref.gid)
                    && (!snapshots || source.frame.is_none())
                {
                    source.remote_dependents.push(gid);
                }
            }
        }
    }

    pub fn load_raw(&mut self, gid: u64, raw_readouts: &[bool], raw_measurements: &util::BitVector) {
        self.load_prepared_raw(gid, PreparedRawFrame::new(raw_readouts, raw_measurements));
    }

    pub(crate) fn load_prepared_raw(&mut self, gid: u64, raw: PreparedRawFrame) {
        let gadget = self.gadgets.get_mut(&gid).unwrap();
        debug_assert_eq!(gadget.num_readouts(), raw.readouts.len());
        debug_assert_eq!(gadget.num_measurements(), raw.measurements.len());
        gadget.raw_frame.load(raw).expect("raw frame publication failed");
    }

    pub fn load_correction(&mut self, gid: u64, residual: BitVec, readout_flips: BitVec) -> HashMap<u64, util::BitVector> {
        self.load_frame_correction(
            gid,
            DecodedFrame::Dense(PauliFrame {
                residual,
                readouts: readout_flips,
            }),
        )
    }

    pub(crate) fn load_sparse_correction(
        &mut self,
        gid: u64,
        residual_flips: Vec<usize>,
        readout_flips: Vec<usize>,
    ) -> HashMap<u64, util::BitVector> {
        let gadget = &self.gadgets[&gid];
        assert!(residual_flips.iter().all(|&index| index < gadget.num_output_observables()));
        assert!(readout_flips.iter().all(|&index| index < gadget.num_readouts()));
        self.load_frame_correction(
            gid,
            DecodedFrame::Sparse {
                residual_flips,
                readout_flips,
            },
        )
    }

    fn load_frame_correction(&mut self, gid: u64, correction: DecodedFrame) -> HashMap<u64, util::BitVector> {
        let gadget = self.gadgets.get_mut(&gid).unwrap();
        assert!(!gadget.finalized, "cannot correct a finalized frame");
        debug_assert!(
            gadget.raw_frame.raw.lock().unwrap().is_some(),
            "raw readouts must be loaded first for gid={gid}"
        );
        gadget.decoded = Some(correction);
        let mut updates = HashMap::new();
        self.propagate_from(gid, &mut updates);
        updates
    }

    fn propagate_from(&mut self, gid: u64, updates: &mut HashMap<u64, util::BitVector>) {
        use binar::Bitwise;

        // first check if all the input peers have done their work; if not, return nothing
        let gadget = self.gadgets.get(&gid).unwrap();
        if gadget.decoded.is_none() {
            return;
        }
        if gadget.inputs.iter().any(|peer| self.gadgets[&peer.gid].frame.is_none()) {
            return;
        }
        // gather remote readouts for remote conditional correction if present
        let remote_readouts_vec: Option<BitVec> = if let Some(values) = &gadget.remote_readout_values {
            let Some(values) = values.iter().copied().collect::<Option<Vec<_>>>() else {
                return;
            };
            Some(values.into_iter().collect())
        } else if let Some((remote_refs, _)) = &gadget.remote_conditional_correction {
            let mut values = Vec::with_capacity(remote_refs.len());
            for remote_ref in remote_refs {
                let remote_gadget = self.gadgets.get(&remote_ref.gid).unwrap();
                if let Some(frame) = remote_gadget.frame.as_ref() {
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
        let input_observables: BitVec = gadget
            .inputs
            .iter()
            .flat_map(|peer| {
                let source = &self.gadgets[&peer.gid];
                let frame = source.frame.as_ref().unwrap();
                source.residual_range(peer.port).map(move |index| frame.residual.index(index))
            })
            .chain([true])
            .collect();
        let mut residual = gadget.correction_propagation.multiply(&input_observables);
        // apply physical_correction * measurements
        let raw_frame = gadget.raw_frame.raw.lock().unwrap();
        let raw = raw_frame.as_ref().expect("raw frame must be loaded before propagation");
        gadget.physical_correction.xor_into(&mut residual, &raw.measurements);
        let mut readouts = raw.readouts.clone();
        drop(raw_frame);
        let decoded = gadget.decoded.as_ref().unwrap();
        decoded.apply(&mut residual, &mut readouts);
        gadget.readout_propagation.xor_into(&mut readouts, &input_observables);
        gadget.logical_correction.xor_into(&mut residual, &readouts);
        // apply remote conditional correction if present
        if let Some((_, correction_matrix)) = &gadget.remote_conditional_correction {
            correction_matrix.xor_into(&mut residual, &remote_readouts_vec.unwrap());
        }
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
        let readout_snapshot = gadget.frame.as_ref().unwrap().readouts.clone();
        for &dependent in &remote_dependents {
            let Some(consumer) = self.gadgets.get_mut(&dependent) else {
                continue;
            };
            if let (Some(values), Some((references, _))) =
                (&mut consumer.remote_readout_values, &consumer.remote_conditional_correction)
            {
                for (value, reference) in values.iter_mut().zip(references) {
                    if reference.gid == gid {
                        *value = Some(readout_snapshot.index(usize::try_from(reference.readout_index).unwrap()));
                    }
                }
            }
        }
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

    #[test]
    fn sparse_frame_corrections_match_dense_updates_with_duplicate_and_boundary_flips() {
        let mut kind = build_gadget_type(0, 33, 65);
        append_bit(kind.correction_propagation.as_mut().unwrap(), 0, 0);
        append_bit(kind.readout_propagation.as_mut().unwrap(), 64, 0);
        append_bit(kind.logical_correction.as_mut().unwrap(), 65, 64);
        append_bit(kind.logical_correction.as_mut().unwrap(), 0, 0);
        let prepare = || {
            let mut tracker = PauliFrameTracker::new();
            tracker.add_gadget(1, &kind, None, &build_port_types(), &[]);
            let raw: Vec<_> = (0..65).map(|index| index % 7 < 3).collect();
            tracker.load_raw(1, &raw, &BitVector::default());
            tracker
        };
        for (residual, readouts) in [(vec![], vec![]), (vec![0, 1, 63, 64, 65, 1], vec![0, 1, 63, 64, 1])] {
            let mut dense = prepare();
            let mut dense_residual = BitVec::zeros(66);
            let mut dense_readouts = BitVec::zeros(65);
            for &index in &residual {
                dense_residual.negate_index(index);
            }
            for &index in &readouts {
                dense_readouts.negate_index(index);
            }
            let expected = dense.load_correction(1, dense_residual, dense_readouts);
            let mut sparse = prepare();
            let actual = sparse.load_sparse_correction(1, residual, readouts);
            assert_eq!(actual, expected);
            assert_eq!(sparse.gadgets[&1].frame, dense.gadgets[&1].frame);
        }
    }

    #[test]
    fn prepared_raw_frames_preserve_physical_corrections_across_word_boundaries() {
        let mut tracker = PauliFrameTracker::new();
        let mut kind = build_gadget_type(0, 1, 0);
        kind.measurements = vec![bin::gadget_type::Measurement::default(); 137];
        let mut physical = zeros(2, 137);
        for column in [0, 63, 64] {
            append_bit(&mut physical, 0, column);
        }
        for column in [127, 136] {
            append_bit(&mut physical, 1, column);
        }
        kind.physical_correction = Some(physical);
        tracker.add_gadget(1, &kind, None, &build_port_types(), &[]);
        let mut measurements = crate::misc::bit_vector::from_sparse_indices(137, &[0, 64, 136]);
        *measurements.data.last_mut().unwrap() |= 0x7f;
        tracker.load_prepared_raw(1, PreparedRawFrame::new(&[], &measurements));
        load_correction_to(&mut tracker, 1, &[], &[]);
        assert_eq!(
            tracker.gadgets[&1].frame.as_ref().unwrap().residual,
            [false, true].into_iter().collect::<BitVec>()
        );
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

    #[test]
    fn readout_hints_saturate_and_preserve_the_unsigned_range() {
        let ports = build_port_types();
        let source_type = build_gadget_type(0, 0, 1);
        let consumer_type = build_gadget_type(0, 0, 0);
        let consumer = |gid| bin::Gadget {
            gid,
            modifier: Some(bin::GadgetModifier {
                remote_conditional_correction: Some(bin::RemoteConditionalCorrection {
                    remote_readouts: vec![bin::remote_conditional_correction::RemoteReadout {
                        gid: 1,
                        readout_index: 0,
                    }],
                    correction: Some(zeros(0, 1)),
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        for hint in [0, 1, 1 << 63, u64::MAX] {
            let mut tracker = PauliFrameTracker::new();
            tracker
                .add_committing_gadget(
                    &bin::Gadget {
                        gid: 1,
                        expected_remote_references: hint,
                        ..Default::default()
                    },
                    &source_type,
                    &ports,
                )
                .unwrap();
            tracker.load_raw(1, &[true], &BitVector::default());
            load_correction_to(&mut tracker, 1, &[], &[]);
            for gid in [2, 3] {
                tracker.add_committing_gadget(&consumer(gid), &consumer_type, &ports).unwrap();
            }
            assert_eq!(tracker.gadgets[&1].expected_remote_references, hint.saturating_sub(2));
            tracker.finalize(1);
            tracker.collect_gadget(1);
            if hint > 2 {
                assert_eq!(tracker.retained_readouts[&1].remaining, hint - 2);
                assert!(get_bit(&tracker.retained_readouts[&1].value, 0));
                tracker.add_committing_gadget(&consumer(4), &consumer_type, &ports).unwrap();
                assert_eq!(tracker.retained_readouts[&1].remaining, hint - 3);
            } else {
                assert!(tracker.retained_readouts.is_empty());
            }
        }
    }

    #[test]
    fn finalized_frames_release_payloads_but_support_late_children_and_remote_readouts() {
        let mut tracker = PauliFrameTracker::new();
        let ports = build_port_types();
        let source = build_gadget_type(0, 1, 1);
        tracker.add_gadget(1, &source, None, &ports, &[]);
        tracker.load_raw(1, &[true], &BitVector::default());
        load_correction_to(&mut tracker, 1, &[0], &[]);
        let expected = tracker.gadgets[&1].frame.clone();
        tracker.finalize(1);
        tracker.finalize(1);
        let source = &tracker.gadgets[&1];
        assert_eq!(source.frame, expected);
        assert!(source.decoded.is_none());
        assert!(source.raw_frame.raw.lock().unwrap().is_none());
        assert!(source.finalized);
        assert_eq!(source.num_output_observables(), 2);
        assert_eq!(source.num_readouts(), 1);

        let mut child = build_gadget_type(1, 1, 1);
        append_bit(child.readout_propagation.as_mut().unwrap(), 0, 0);
        let mut remote = zeros(2, 1);
        append_bit(&mut remote, 1, 0);
        let modifier = bin::GadgetModifier {
            remote_conditional_correction: Some(bin::RemoteConditionalCorrection {
                remote_readouts: vec![bin::remote_conditional_correction::RemoteReadout {
                    gid: 1,
                    readout_index: 0,
                }],
                correction: Some(remote),
            }),
            ..Default::default()
        };
        tracker.add_gadget(
            2,
            &child,
            Some(&modifier),
            &ports,
            &[bin::gadget::Connector { gid: 1, port: 0 }],
        );
        tracker.load_raw(2, &[false], &BitVector::default());
        let updates = load_correction_to(&mut tracker, 2, &[], &[]);
        assert!(get_bit(&updates[&2], 0));
        assert_eq!(
            tracker.gadgets[&2].get_residual_slice(&tracker.gadgets[&2].frame.as_ref().unwrap().residual, 0),
            vec![false, true],
        );
    }
}
