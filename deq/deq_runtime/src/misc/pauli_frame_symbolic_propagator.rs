//! Symbolic propagation of decoder correction components through a gadget graph.
//!
//! Unlike [`super::pauli_frame_tracker::PauliFrameTracker`], this module does
//! not store shot values. It propagates formal GF(2) correction variables to
//! determine logical flips within a decode window and dependencies across
//! successive windows.

use super::pauli_frame_tracker::PauliFrameGadget;
use binar::BitMatrix;
use hashbrown::{HashMap, HashSet};
use std::collections::BinaryHeap;

#[cfg_attr(test, derive(Clone))]
pub(crate) struct PauliFrameSymbolicPropagator {
    /// List of symbolic GF(2) expressions.
    nodes: Vec<SymbolicNode>,
    /// Symbolic state keyed by runtime gadget ID.
    gadgets: HashMap<u64, SymbolicGadget>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CorrectionBasis {
    Readout { gid: u64, index: usize },
    Residual { gid: u64, index: usize },
}

/// Index into `PauliFrameSymbolicPropagator::nodes`.
type SymbolicNodeIndex = usize;
const ZERO_NODE: SymbolicNodeIndex = 0;

#[cfg_attr(test, derive(Clone))]
enum SymbolicNode {
    Zero,
    Basis(CorrectionBasis),
    Xor { gid: u64, terms: Box<[SymbolicNodeIndex]> },
}

#[cfg_attr(test, derive(Clone))]
struct SymbolicGadget {
    /// Position in `nodes` for each logical readout expression.
    readout_nodes: Vec<SymbolicNodeIndex>,
    /// Position in `nodes` for each output-observable expression.
    residual_nodes: Vec<SymbolicNodeIndex>,
    /// Gadgets whose readouts are used in remote corrections.
    remote_sources: Vec<u64>,
    /// Gadgets that consume this gadget's readouts remotely.
    remote_dependents: Vec<u64>,
    /// Flattened observable start index for each output port.
    output_bias: Vec<usize>,
}

impl SymbolicGadget {
    fn num_readouts(&self) -> usize {
        self.readout_nodes.len()
    }

    fn output_observable_range(&self, port: u64) -> std::ops::Range<usize> {
        let port = usize::try_from(port).unwrap();
        let start = self.output_bias[port];
        let end = self.output_bias.get(port + 1).copied().unwrap_or(self.residual_nodes.len());
        start..end
    }

    fn residual_slice(&self, port: u64) -> &[SymbolicNodeIndex] {
        &self.residual_nodes[self.output_observable_range(port)]
    }
}

impl PauliFrameSymbolicPropagator {
    pub(crate) fn new() -> Self {
        Self {
            nodes: vec![SymbolicNode::Zero],
            gadgets: HashMap::new(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.nodes.truncate(1);
        self.gadgets.clear();
    }

    fn basis(&mut self, basis: CorrectionBasis) -> SymbolicNodeIndex {
        let node = self.nodes.len();
        self.nodes.push(SymbolicNode::Basis(basis));
        node
    }

    fn xor(&mut self, gid: u64, mut terms: Vec<SymbolicNodeIndex>) -> SymbolicNodeIndex {
        terms.retain(|&term| term != ZERO_NODE);
        terms.sort_unstable();
        let mut reduced = Vec::with_capacity(terms.len());
        let mut start = 0;
        while start < terms.len() {
            let mut end = start + 1;
            while end < terms.len() && terms[end] == terms[start] {
                end += 1;
            }
            if (end - start) % 2 == 1 {
                reduced.push(terms[start]);
            }
            start = end;
        }
        match reduced.as_slice() {
            [] => ZERO_NODE,
            [node] => *node,
            _ => {
                let node = self.nodes.len();
                self.nodes.push(SymbolicNode::Xor {
                    gid,
                    terms: reduced.into_boxed_slice(),
                });
                node
            }
        }
    }

    fn apply_matrix(&mut self, gid: u64, matrix: &BitMatrix, inputs: &[SymbolicNodeIndex]) -> Vec<SymbolicNodeIndex> {
        debug_assert_eq!(matrix.column_count(), inputs.len());
        (0..matrix.row_count())
            .map(|row| {
                let terms = inputs
                    .iter()
                    .enumerate()
                    .filter_map(|(column, &input)| matrix.get((row, column)).then_some(input))
                    .collect();
                self.xor(gid, terms)
            })
            .collect()
    }

    pub(crate) fn add_gadget(&mut self, gid: u64, gadget: &PauliFrameGadget) {
        debug_assert!(!self.gadgets.contains_key(&gid));
        let mut input_nodes = Vec::with_capacity(gadget.num_input_observables() + 1);
        for connector in &gadget.inputs {
            input_nodes.extend_from_slice(self.gadgets[&connector.gid].residual_slice(connector.port));
        }
        input_nodes.push(ZERO_NODE);

        let mut readout_nodes = self.apply_matrix(gid, &gadget.readout_propagation, &input_nodes);
        for (index, readout) in readout_nodes.iter_mut().enumerate() {
            let basis = self.basis(CorrectionBasis::Readout { gid, index });
            *readout = self.xor(gid, vec![*readout, basis]);
        }

        let correction_nodes = self.apply_matrix(gid, &gadget.correction_propagation, &input_nodes);
        let logical_nodes = self.apply_matrix(gid, &gadget.logical_correction, &readout_nodes);
        let remote_nodes = if let Some((remote_readouts, correction_matrix)) = &gadget.remote_conditional_correction {
            let remote_inputs: Vec<_> = remote_readouts
                .iter()
                .map(|remote| self.gadgets[&remote.gid].readout_nodes[usize::try_from(remote.readout_index).unwrap()])
                .collect();
            self.apply_matrix(gid, correction_matrix, &remote_inputs)
        } else {
            vec![ZERO_NODE; gadget.num_output_observables()]
        };

        let mut residual_nodes = Vec::with_capacity(gadget.num_output_observables());
        for index in 0..gadget.num_output_observables() {
            let basis = self.basis(CorrectionBasis::Residual { gid, index });
            residual_nodes.push(self.xor(
                gid,
                vec![correction_nodes[index], logical_nodes[index], remote_nodes[index], basis],
            ));
        }

        self.gadgets.insert(
            gid,
            SymbolicGadget {
                readout_nodes,
                residual_nodes,
                remote_sources: gadget
                    .remote_conditional_correction
                    .iter()
                    .flat_map(|(readouts, _)| readouts.iter().map(|readout| readout.gid))
                    .collect(),
                remote_dependents: vec![],
                output_bias: gadget.output_bias.clone(),
            },
        );
        if let Some((remote_readouts, _)) = &gadget.remote_conditional_correction {
            for remote in remote_readouts {
                self.gadgets.get_mut(&remote.gid).unwrap().remote_dependents.push(gid);
            }
        }
    }

    pub(crate) fn logical_flip_cache(
        &self,
        gadget_ids: impl IntoIterator<Item = u64>,
        targets: &[CorrectionBasis],
    ) -> LogicalFlipCache {
        let active_gids = gadget_ids.into_iter().collect();
        let mut basis_effects = HashMap::<CorrectionBasis, Vec<u64>>::new();
        for (target_index, &target) in targets.iter().enumerate() {
            let root = match target {
                CorrectionBasis::Readout { gid, index } => self.gadgets[&gid].readout_nodes[index],
                CorrectionBasis::Residual { gid, index } => self.gadgets[&gid].residual_nodes[index],
            };
            for component in self.components(root, Some(&active_gids)) {
                basis_effects
                    .entry(component)
                    .or_default()
                    .push(u64::try_from(target_index).unwrap());
            }
        }
        LogicalFlipCache { basis_effects }
    }

    pub(crate) fn remote_dependencies_are_closed(&self, gadget_ids: &[u64]) -> bool {
        let included: HashSet<_> = gadget_ids.iter().copied().collect();
        gadget_ids.iter().all(|gid| {
            let gadget = &self.gadgets[gid];
            let sources_are_included = gadget.remote_sources.iter().all(|source| included.contains(source));
            sources_are_included && gadget.remote_dependents.iter().all(|dependent| included.contains(dependent))
        })
    }

    pub(crate) fn boundary_targets(&self, boundaries: &[(u64, u64)]) -> Vec<CorrectionBasis> {
        boundaries
            .iter()
            .flat_map(|&(gid, port)| {
                self.gadgets[&gid]
                    .output_observable_range(port)
                    .map(move |index| CorrectionBasis::Residual { gid, index })
            })
            .collect()
    }

    pub(crate) fn readout_targets(&self, gids: impl IntoIterator<Item = u64>) -> Vec<CorrectionBasis> {
        gids.into_iter()
            .flat_map(|gid| (0..self.gadgets[&gid].num_readouts()).map(move |index| CorrectionBasis::Readout { gid, index }))
            .collect()
    }

    pub(crate) fn readout_components(&self, gid: u64) -> Vec<Vec<CorrectionBasis>> {
        self.gadgets[&gid]
            .readout_nodes
            .iter()
            .map(|&root| self.components(root, None))
            .collect()
    }

    fn components(&self, root: SymbolicNodeIndex, active_gids: Option<&HashSet<u64>>) -> Vec<CorrectionBasis> {
        let included = |gid| active_gids.is_none_or(|gids| gids.contains(&gid));
        let mut pending = BinaryHeap::from([root]);
        let mut active = HashSet::new();
        active.insert(root);
        let mut components = vec![];
        while let Some(node) = pending.pop() {
            if !active.remove(&node) {
                continue;
            }
            match &self.nodes[node] {
                SymbolicNode::Zero => {}
                SymbolicNode::Basis(component) => {
                    let (CorrectionBasis::Readout { gid, .. } | CorrectionBasis::Residual { gid, .. }) = *component;
                    if included(gid) {
                        components.push(*component);
                    }
                }
                SymbolicNode::Xor { gid, terms } => {
                    if !included(*gid) {
                        continue;
                    }
                    for &term in terms {
                        if active.insert(term) {
                            pending.push(term);
                        } else {
                            active.remove(&term);
                        }
                    }
                }
            }
        }
        components
    }
}

impl Default for PauliFrameSymbolicPropagator {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) struct LogicalFlipCache {
    basis_effects: HashMap<CorrectionBasis, Vec<u64>>,
}

impl LogicalFlipCache {
    pub(crate) fn propagated_logical_flips(
        &self,
        correction_gid: u64,
        residual_indices: &[u64],
        readout_indices: &[u64],
    ) -> Vec<u64> {
        let mut effect = HashSet::new();
        for &index in readout_indices {
            self.xor_basis_effect(
                CorrectionBasis::Readout {
                    gid: correction_gid,
                    index: usize::try_from(index).unwrap(),
                },
                &mut effect,
            );
        }
        for &index in residual_indices {
            self.xor_basis_effect(
                CorrectionBasis::Residual {
                    gid: correction_gid,
                    index: usize::try_from(index).unwrap(),
                },
                &mut effect,
            );
        }
        let mut effect: Vec<_> = effect.into_iter().collect();
        effect.sort_unstable();
        effect
    }

    fn xor_basis_effect(&self, component: CorrectionBasis, effect: &mut HashSet<u64>) {
        for &target in self.basis_effects.get(&component).into_iter().flatten() {
            if !effect.insert(target) {
                effect.remove(&target);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin;
    use crate::misc::bit_matrix::{append_bit, zeros};
    use crate::misc::pauli_frame_tracker::PauliFrameTracker;
    use binar::{BitVec, BitwiseMut};
    use std::sync::Arc;

    fn port_types() -> HashMap<u64, Arc<bin::PortType>> {
        [(
            1,
            Arc::new(bin::PortType {
                ptype: 1,
                observables: vec![bin::port_type::Observable::default(); 2],
                ..Default::default()
            }),
        )]
        .into_iter()
        .collect()
    }

    fn gadget_type(inputs: usize, outputs: usize, readouts: usize) -> bin::GadgetType {
        bin::GadgetType {
            inputs: vec![
                bin::gadget_type::Port {
                    ptype: 1,
                    ..Default::default()
                };
                inputs
            ],
            outputs: vec![
                bin::gadget_type::Port {
                    ptype: 1,
                    ..Default::default()
                };
                outputs
            ],
            readouts: vec![bin::gadget_type::Readout::default(); readouts],
            correction_propagation: Some(zeros(outputs * 2, inputs * 2 + 1)),
            readout_propagation: Some(zeros(readouts, inputs * 2 + 1)),
            logical_correction: Some(zeros(outputs * 2, readouts)),
            physical_correction: Some(zeros(outputs * 2, 0)),
            ..Default::default()
        }
    }

    fn add_gadget(
        concrete: &mut PauliFrameTracker,
        symbolic: &mut PauliFrameSymbolicPropagator,
        gid: u64,
        gadget_type: &bin::GadgetType,
        modifier: Option<&bin::GadgetModifier>,
        connectors: &[bin::gadget::Connector],
        port_types: &HashMap<u64, Arc<bin::PortType>>,
    ) {
        concrete.add_gadget(gid, gadget_type, modifier, port_types, connectors);
        symbolic.add_gadget(gid, &concrete.gadgets[&gid]);
    }

    #[test]
    fn symbolic_logical_flips_match_concrete_frame_changes() {
        let port_types = port_types();
        let gids = [40, 7, 21];
        for pattern in 0_u64..32 {
            let mut types = [gadget_type(0, 1, 1), gadget_type(1, 2, 2), gadget_type(2, 0, 2)];
            for (gadget_index, gadget_type) in types.iter_mut().enumerate() {
                for (matrix_index, matrix) in [
                    gadget_type.correction_propagation.as_mut().unwrap(),
                    gadget_type.readout_propagation.as_mut().unwrap(),
                    gadget_type.logical_correction.as_mut().unwrap(),
                ]
                .into_iter()
                .enumerate()
                {
                    for row in 0..matrix.rows {
                        for column in 0..matrix.cols {
                            let bit =
                                (row + column + u64::try_from(gadget_index).unwrap() + u64::try_from(matrix_index).unwrap())
                                    % 5;
                            if pattern & (1 << bit) != 0 {
                                append_bit(matrix, usize::try_from(row).unwrap(), usize::try_from(column).unwrap());
                            }
                        }
                    }
                }
            }
            let connectors = [
                vec![],
                vec![bin::gadget::Connector { gid: gids[0], port: 0 }],
                vec![
                    bin::gadget::Connector { gid: gids[1], port: 0 },
                    bin::gadget::Connector { gid: gids[1], port: 1 },
                ],
            ];
            let mut concrete = PauliFrameTracker::new();
            let mut symbolic = PauliFrameSymbolicPropagator::new();
            let mut baseline = HashMap::new();
            for ((&gid, gadget_type), inputs) in gids.iter().zip(&types).zip(&connectors) {
                add_gadget(&mut concrete, &mut symbolic, gid, gadget_type, None, inputs, &port_types);
                concrete.load_raw(
                    gid,
                    &vec![false; gadget_type.readouts.len()],
                    &crate::util::BitVector::default(),
                );
                baseline.extend(concrete.load_correction(
                    gid,
                    BitVec::zeros(gadget_type.outputs.len() * 2),
                    BitVec::zeros(gadget_type.readouts.len()),
                ));
            }
            let targets = symbolic.readout_targets(gids);
            let cache = symbolic.logical_flip_cache(gids, &targets);
            for (&gid, gadget_type) in gids.iter().zip(&types) {
                let output_count = gadget_type.outputs.len() * 2;
                let readout_count = gadget_type.readouts.len();
                for component in 0..output_count + readout_count {
                    let mut residual = BitVec::zeros(output_count);
                    let mut readouts = BitVec::zeros(readout_count);
                    let (residual_indices, readout_indices) = if component < output_count {
                        residual.assign_index(component, true);
                        (vec![u64::try_from(component).unwrap()], vec![])
                    } else {
                        readouts.assign_index(component - output_count, true);
                        (vec![], vec![u64::try_from(component - output_count).unwrap()])
                    };
                    let updates = concrete.load_correction(gid, residual, readouts);
                    let expected: Vec<_> = targets
                        .iter()
                        .enumerate()
                        .filter_map(|(target_index, target)| {
                            let CorrectionBasis::Readout { gid, index } = target else {
                                unreachable!()
                            };
                            let changed = updates.get(gid).is_some_and(|readouts| {
                                crate::misc::bit_vector::get_bit(readouts, u64::try_from(*index).unwrap())
                                    != crate::misc::bit_vector::get_bit(&baseline[gid], u64::try_from(*index).unwrap())
                            });
                            changed.then_some(u64::try_from(target_index).unwrap())
                        })
                        .collect();
                    assert_eq!(
                        cache.propagated_logical_flips(gid, &residual_indices, &readout_indices),
                        expected,
                        "pattern={pattern}, gid={gid}, component={component}"
                    );
                    concrete.load_correction(gid, BitVec::zeros(output_count), BitVec::zeros(readout_count));
                }
            }
        }
    }

    #[test]
    fn logical_flip_cache_stays_within_active_region() {
        let port_types = port_types();
        let mut concrete = PauliFrameTracker::new();
        let mut symbolic = PauliFrameSymbolicPropagator::new();
        let source = gadget_type(0, 1, 0);
        add_gadget(&mut concrete, &mut symbolic, 1, &source, None, &[], &port_types);

        let mut pass_through = gadget_type(1, 1, 0);
        append_bit(pass_through.correction_propagation.as_mut().unwrap(), 0, 0);
        add_gadget(
            &mut concrete,
            &mut symbolic,
            2,
            &pass_through,
            None,
            &[bin::gadget::Connector { gid: 1, port: 0 }],
            &port_types,
        );

        let mut terminal = gadget_type(1, 0, 1);
        append_bit(terminal.readout_propagation.as_mut().unwrap(), 0, 0);
        add_gadget(
            &mut concrete,
            &mut symbolic,
            3,
            &terminal,
            None,
            &[bin::gadget::Connector { gid: 2, port: 0 }],
            &port_types,
        );

        let cache = symbolic.logical_flip_cache([1, 3], &[CorrectionBasis::Readout { gid: 3, index: 0 }]);
        assert!(cache.propagated_logical_flips(1, &[0], &[]).is_empty());

        let cache = symbolic.logical_flip_cache([1, 2, 3], &[CorrectionBasis::Readout { gid: 3, index: 0 }]);
        assert_eq!(cache.propagated_logical_flips(1, &[0], &[]), vec![0]);
        assert!(cache.propagated_logical_flips(1, &[0, 0], &[]).is_empty());
        assert_eq!(cache.propagated_logical_flips(3, &[], &[0]), vec![0]);
    }

    #[test]
    fn reconvergent_paths_cancel_symbolically() {
        let port_types = port_types();
        let mut concrete = PauliFrameTracker::new();
        let mut symbolic = PauliFrameSymbolicPropagator::new();
        let mut source = gadget_type(0, 2, 1);
        append_bit(source.logical_correction.as_mut().unwrap(), 0, 0);
        append_bit(source.logical_correction.as_mut().unwrap(), 2, 0);
        add_gadget(&mut concrete, &mut symbolic, 1, &source, None, &[], &port_types);

        let mut branch = gadget_type(1, 1, 0);
        append_bit(branch.correction_propagation.as_mut().unwrap(), 0, 0);
        for (port, gid) in [2, 3].into_iter().enumerate() {
            add_gadget(
                &mut concrete,
                &mut symbolic,
                gid,
                &branch,
                None,
                &[bin::gadget::Connector {
                    gid: 1,
                    port: u64::try_from(port).unwrap(),
                }],
                &port_types,
            );
        }

        let mut terminal = gadget_type(2, 0, 1);
        append_bit(terminal.readout_propagation.as_mut().unwrap(), 0, 0);
        append_bit(terminal.readout_propagation.as_mut().unwrap(), 0, 2);
        add_gadget(
            &mut concrete,
            &mut symbolic,
            4,
            &terminal,
            None,
            &[
                bin::gadget::Connector { gid: 2, port: 0 },
                bin::gadget::Connector { gid: 3, port: 0 },
            ],
            &port_types,
        );

        let source_readout = CorrectionBasis::Readout { gid: 1, index: 0 };
        assert!(!symbolic.readout_components(4)[0].contains(&source_readout));
        let cache = symbolic.logical_flip_cache([1, 2, 3, 4], &[CorrectionBasis::Readout { gid: 4, index: 0 }]);
        assert!(cache.propagated_logical_flips(1, &[], &[0]).is_empty());

        let cache = symbolic.logical_flip_cache([1, 2, 4], &[CorrectionBasis::Readout { gid: 4, index: 0 }]);
        assert_eq!(cache.propagated_logical_flips(1, &[], &[0]), vec![0]);
    }

    #[test]
    fn remote_readout_dependencies_propagate_symbolically() {
        let port_types = port_types();
        let mut concrete = PauliFrameTracker::new();
        let mut symbolic = PauliFrameSymbolicPropagator::new();
        let source = gadget_type(0, 1, 1);
        add_gadget(&mut concrete, &mut symbolic, 1, &source, None, &[], &port_types);

        let mut correction = zeros(2, 1);
        append_bit(&mut correction, 0, 0);
        let modifier = bin::GadgetModifier {
            remote_conditional_correction: Some(bin::RemoteConditionalCorrection {
                remote_readouts: vec![bin::remote_conditional_correction::RemoteReadout {
                    gid: 1,
                    readout_index: 0,
                }],
                correction: Some(correction),
            }),
            ..Default::default()
        };
        let remote = gadget_type(0, 1, 0);
        add_gadget(&mut concrete, &mut symbolic, 2, &remote, Some(&modifier), &[], &port_types);

        let mut terminal = gadget_type(1, 0, 1);
        append_bit(terminal.readout_propagation.as_mut().unwrap(), 0, 0);
        add_gadget(
            &mut concrete,
            &mut symbolic,
            3,
            &terminal,
            None,
            &[bin::gadget::Connector { gid: 2, port: 0 }],
            &port_types,
        );

        assert!(symbolic.readout_components(3)[0].contains(&CorrectionBasis::Readout { gid: 1, index: 0 }));
        assert!(!symbolic.remote_dependencies_are_closed(&[1]));
        assert!(symbolic.remote_dependencies_are_closed(&[1, 2, 3]));

        let targets = [
            CorrectionBasis::Residual { gid: 2, index: 0 },
            CorrectionBasis::Readout { gid: 3, index: 0 },
        ];
        let cache = symbolic.logical_flip_cache([1, 2, 3], &targets);
        assert_eq!(cache.propagated_logical_flips(1, &[], &[0]), vec![0, 1]);
        assert_eq!(cache.propagated_logical_flips(2, &[0], &[]), vec![0, 1]);

        let cache = symbolic.logical_flip_cache([1, 3], &targets[1..]);
        assert!(cache.propagated_logical_flips(1, &[], &[0]).is_empty());
    }
}
