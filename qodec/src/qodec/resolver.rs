//! Cross-file reference resolution: walks the raw artifact maps a loader
//! has materialized, ties gadgets to their source instruction set, realization
//! target, and supporting check / flag / observable files, and produces the
//! [`Layer`] chain held by a fully-resolved [`Qodec`].
//!
//! Also owns the path-normalization helpers (`resolve_relative`,
//! `normalize_relative`, `normalize_path`) used by both the loader and
//! artifact resolution.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::LoadError;
use crate::Code;
use crate::GadgetSpec;
use crate::InstructionSet;
use crate::Sourced;
use crate::resolved::{self, Encoding, Layer};
use crate::{LayerSpec, Manifest};
use crate::{ParityEquation, ReadoutsList};

/// Referenced artifacts used to resolve a lowering chain.
#[derive(Clone, Copy)]
pub(super) struct LoaderArtifacts<'a> {
    pub instruction_sets_by_path: &'a BTreeMap<PathBuf, Arc<InstructionSet>>,
    pub gadgets: &'a BTreeMap<PathBuf, GadgetSpec>,
    pub check_lists: &'a BTreeMap<PathBuf, Vec<ParityEquation>>,
    pub readout_lists: &'a BTreeMap<PathBuf, ReadoutsList>,
    pub codes_by_path: &'a BTreeMap<PathBuf, Arc<Code>>,
    pub source_files: &'a BTreeMap<PathBuf, String>,
}

pub(super) fn build_layers(
    manifest: &Manifest,
    manifest_path: &Path,
    artifacts: &LoaderArtifacts<'_>,
) -> Result<Vec<Layer>, LoadError> {
    let layer_paths: Vec<PathBuf> = manifest
        .layers
        .iter()
        .map(|layer| PathBuf::from(normalize_relative(&layer.instruction_set)))
        .collect();
    let instruction_sets: Vec<Arc<InstructionSet>> = layer_paths
        .iter()
        .map(|path| {
            artifacts
                .instruction_sets_by_path
                .get(path)
                .cloned()
                .ok_or_else(|| LoadError::MissingArtifact {
                    referenced_from: manifest_path.to_path_buf(),
                    path: path.clone(),
                })
        })
        .collect::<Result<_, _>>()?;

    let mut layers: Vec<Layer> = Vec::with_capacity(instruction_sets.len());
    for index in 0..instruction_sets.len() {
        let gadgets = if index + 1 < instruction_sets.len() {
            build_edge_gadgets(
                &manifest.layers[index],
                &layer_paths[index],
                &layer_paths[index + 1],
                &instruction_sets[index],
                &instruction_sets[index + 1],
                artifacts,
            )?
        } else {
            BTreeMap::new()
        };
        layers.push(Layer {
            instruction_set: instruction_sets[index].clone(),
            codes: manifest.layers[index]
                .codes
                .iter()
                .map(|(block, path)| {
                    let path = PathBuf::from(normalize_relative(path));
                    let code =
                        artifacts
                            .codes_by_path
                            .get(&path)
                            .cloned()
                            .ok_or_else(|| LoadError::MissingArtifact {
                                referenced_from: manifest_path.to_path_buf(),
                                path,
                            })?;
                    Ok((block.clone(), code))
                })
                .collect::<Result<_, LoadError>>()?,
            gadgets,
        });
    }

    Ok(layers)
}

/// Build the resolved gadget set for one lowering edge from the layer's
/// explicit `mnemonic → gadget-file` membership map. Each listed gadget is
/// looked up by its resolved path and resolved against the edge's source
/// and target instruction sets.
fn build_edge_gadgets(
    layer: &LayerSpec,
    source_path: &Path,
    target_path: &Path,
    source: &Arc<InstructionSet>,
    target: &Arc<InstructionSet>,
    artifacts: &LoaderArtifacts<'_>,
) -> Result<BTreeMap<String, resolved::Gadget>, LoadError> {
    let edge = LoweringEdge {
        source_path,
        target_path,
        source,
        target,
        layer_codes: &layer.codes,
    };
    let mut resolved_gadgets: BTreeMap<String, resolved::Gadget> = BTreeMap::new();
    for (mnemonic, gadget_file) in &layer.gadgets {
        let gadget_path = PathBuf::from(normalize_relative(gadget_file));
        let gadget = artifacts
            .gadgets
            .get(&gadget_path)
            .ok_or_else(|| LoadError::InvalidGadget {
                gadget: gadget_path.clone(),
                error: format!(
                    "layer instruction set '{}' lists gadget '{mnemonic}' as '{}', which was not loaded",
                    source_path.display(),
                    gadget_path.display()
                ),
            })?;
        let resolved_gadget = build_one_gadget(&gadget_path, gadget, mnemonic, &edge, artifacts)?;
        resolved_gadgets.insert(mnemonic.clone(), resolved_gadget);
    }
    Ok(resolved_gadgets)
}

/// Resolve one gadget whose membership the layer has supplied: its source
/// instruction set is the listing layer's, its target instruction set is the layer below's, and the
/// mnemonic it implements is the map key. If the gadget *also* states its
/// own `implements` / circuit `instruction_set`, those are checked for agreement; when
/// absent, the layer supplies them.
fn build_one_gadget(
    gadget_path: &Path,
    gadget: &GadgetSpec,
    mnemonic: &str,
    edge: &LoweringEdge<'_>,
    artifacts: &LoaderArtifacts<'_>,
) -> Result<resolved::Gadget, LoadError> {
    let invalid = |error: String| LoadError::InvalidGadget {
        gadget: gadget_path.to_path_buf(),
        error,
    };

    edge.check_implements(gadget_path, gadget, mnemonic).map_err(&invalid)?;
    edge.check_circuit_target(gadget_path, gadget).map_err(&invalid)?;
    let instruction = edge
        .source
        .resolve(mnemonic)
        .map_err(|_| LoadError::GadgetUnknownImplements {
            gadget: gadget_path.to_path_buf(),
            implements: mnemonic.to_owned(),
            instruction_set: edge.source.name.clone(),
        })?;

    let (circuit, inputs, outputs) = resolve_realization_for_gadget(
        gadget,
        &instruction,
        edge.layer_codes,
        edge.target,
        artifacts.codes_by_path,
        gadget_path,
        artifacts.source_files,
    )
    .map_err(&invalid)?;
    let checks = resolve_gadget_checks(gadget_path, gadget, artifacts.check_lists)
        .map_err(&invalid)?
        .to_vec();
    let readouts = resolve_gadget_readouts(gadget_path, gadget, artifacts.readout_lists).map_err(&invalid)?;
    let readouts = crate::Readout::resolve_list(readouts, instruction.observe_count());

    Ok(resolved::Gadget {
        implements: instruction,
        circuit,
        inputs,
        outputs,
        parameter_bindings: gadget.parameter_bindings.clone(),
        checks,
        readouts,
        frames: gadget.frames.clone(),
        metadata: gadget.metadata.clone(),
    })
}

struct LoweringEdge<'a> {
    source_path: &'a Path,
    target_path: &'a Path,
    source: &'a InstructionSet,
    target: &'a Arc<InstructionSet>,
    layer_codes: &'a BTreeMap<String, String>,
}

impl LoweringEdge<'_> {
    fn check_implements(&self, path: &Path, gadget: &GadgetSpec, mnemonic: &str) -> Result<(), String> {
        let Some(implements) = &gadget.implements else {
            return Ok(());
        };
        if implements.mnemonic != mnemonic {
            return Err(format!(
                "gadget states `implements` mnemonic '{}', but the layer lists it under '{mnemonic}'",
                implements.mnemonic
            ));
        }
        let resolved = resolve_relative(path, &implements.instruction_set);
        if resolved != self.source_path {
            return Err(format!(
                "gadget states `implements` instruction set '{}' (resolving to {}), but its layer's instruction set is {}",
                implements.instruction_set,
                resolved.display(),
                self.source_path.display()
            ));
        }
        Ok(())
    }

    fn check_circuit_target(&self, path: &Path, gadget: &GadgetSpec) -> Result<(), String> {
        let Some(instruction_set) = &gadget.circuit.instruction_set else {
            return Ok(());
        };
        let resolved = resolve_relative(path, instruction_set);
        if resolved != self.target_path {
            return Err(format!(
                "gadget states circuit `instruction_set` '{}' (resolving to {}), but the layer below's instruction set is {}",
                instruction_set,
                resolved.display(),
                self.target_path.display()
            ));
        }
        Ok(())
    }
}

fn resolve_gadget_readouts<'a>(
    gadget_path: &Path,
    gadget: &'a GadgetSpec,
    readout_lists: &'a BTreeMap<PathBuf, ReadoutsList>,
) -> Result<&'a ReadoutsList, String> {
    match &gadget.readouts {
        Sourced::Inline(list) => Ok(list),
        Sourced::File { path } => {
            let resolved = resolve_relative(gadget_path, path);
            readout_lists
                .get(&resolved)
                .ok_or_else(|| format!("readouts reference {} not found", resolved.display()))
        }
    }
}

/// Resolve a gadget's realization (its resolved [`resolved::Circuit`] plus
/// the boundary `inputs`/`outputs` encodings) into the fields
/// `(circuit, inputs, outputs)`.
///
/// The resolved circuit carries the loaded source text. The code each encoding
/// entry uses is bound on the gadget's
/// source layer (`layer_codes`), keyed by the block type the gadget's
/// `implements` instruction declares at that position; the path is then
/// looked up in `codes` (keyed by qodec-root-relative path).
type ResolvedRealizationFields = (resolved::Circuit, Vec<Encoding>, Vec<Encoding>);

fn resolve_realization_for_gadget(
    gadget: &GadgetSpec,
    instruction: &crate::Instruction,
    layer_codes: &BTreeMap<String, String>,
    target_instruction_set: &Arc<InstructionSet>,
    codes: &BTreeMap<PathBuf, Arc<Code>>,
    gadget_path: &Path,
    source_files: &BTreeMap<PathBuf, String>,
) -> Result<ResolvedRealizationFields, String> {
    let source = match &gadget.circuit.source {
        Sourced::Inline(text) => text.clone(),
        Sourced::File { path } => {
            let resolved_path = resolve_relative(gadget_path, path);
            source_files
                .get(&resolved_path)
                .cloned()
                .ok_or_else(|| format!("circuit source {} was not loaded", resolved_path.display()))?
        }
    };

    let bindings = CodeBindings {
        side: "in",
        layer_codes,
        codes,
        target_instruction_set,
    };
    let inputs = bindings.encodings(&gadget.inputs, &instruction.inputs, &gadget.circuit.inputs)?;
    let outputs = CodeBindings {
        side: "out",
        ..bindings
    }
    .encodings(&gadget.outputs, &instruction.outputs, &gadget.circuit.outputs)?;

    let resolved_circuit = resolved::Circuit {
        instruction_set: Arc::clone(target_instruction_set),
        source,
        format: gadget.circuit.format.clone().or_else(|| {
            gadget
                .circuit
                .source
                .path()
                .and_then(super::parsers::ParserRegistry::format_for_path)
                .map(str::to_owned)
        }),
    };
    Ok((resolved_circuit, inputs, outputs))
}

/// One side of one gadget boundary, with everything needed to bind its codes.
#[derive(Clone, Copy)]
struct CodeBindings<'a> {
    side: &'a str,
    layer_codes: &'a BTreeMap<String, String>,
    codes: &'a BTreeMap<PathBuf, Arc<Code>>,
    target_instruction_set: &'a InstructionSet,
}

impl CodeBindings<'_> {
    /// Resolve one boundary's encodings, filling in the canonical support when
    /// the gadget declared none.
    fn encodings(
        &self,
        entries: &[crate::EncodingSpec],
        operands: &[crate::BlockOperand],
        circuit_side: &BTreeMap<String, String>,
    ) -> Result<Vec<Encoding>, String> {
        let plan = if entries.is_empty() && !operands.is_empty() {
            self.default_support(operands)?
        } else {
            entries
                .iter()
                .enumerate()
                .map(|(index, entry)| self.declared_support(index, entry, operands))
                .collect::<Result<_, _>>()?
        };
        plan.into_iter()
            .enumerate()
            .map(|(index, support)| support.with_block_types(index, circuit_side, self.target_instruction_set))
            .collect()
    }

    fn resolve(&self, entry: usize, block_type: &str) -> Result<Arc<Code>, String> {
        let side = self.side;
        let code_path = self.layer_codes.get(block_type).ok_or_else(|| {
            format!(
                "`{side}` encoding entry {entry} encodes block type '{block_type}', which the layer \
                 does not bind to a code; add it to the layer's `codes` map"
            )
        })?;
        let code_key = PathBuf::from(normalize_relative(code_path));
        let code = self.codes.get(&code_key).ok_or_else(|| {
            format!(
                "code '{code_path}' (for block type '{block_type}') not found (tried key '{}')",
                code_key.display()
            )
        })?;
        Ok(Arc::clone(code))
    }

    fn default_support(&self, operands: &[crate::BlockOperand]) -> Result<Vec<EncodingSupport>, String> {
        let mut plan = Vec::with_capacity(operands.len());
        let mut next_qubit = 0usize;
        for (entry, operand) in operands.iter().enumerate() {
            let code = self.resolve(entry, &operand.block)?;
            let width = code.physical_qubit_count();
            let end = next_qubit.checked_add(width).ok_or_else(|| {
                format!(
                    "`{}` encoding entry {entry}: default support index range overflows",
                    self.side
                )
            })?;
            let mut support = Vec::new();
            support.try_reserve_exact(width).map_err(|error| {
                format!(
                    "`{}` encoding entry {entry}: cannot allocate default support for {width} qubits: {error}",
                    self.side
                )
            })?;
            support.extend((next_qubit..end).map(|qubit| qubit.to_string()));
            next_qubit = end;
            plan.push(EncodingSupport { code, support });
        }
        Ok(plan)
    }

    fn declared_support(
        &self,
        entry: usize,
        spec: &crate::EncodingSpec,
        operands: &[crate::BlockOperand],
    ) -> Result<EncodingSupport, String> {
        let side = self.side;
        let operand = operands.get(entry).ok_or_else(|| format!(
            "`{side}` encoding entry {entry} has no positionally-aligned instruction operand to supply its block type"
        ))?;
        if !operand.block.is_empty() && spec.block_type != operand.block {
            return Err(format!(
                "`{side}` encoding entry {entry} is keyed by block type '{}', but the instruction \
                 declares block type '{}' at this position; the entry key must name the operand's block type",
                spec.block_type, operand.block
            ));
        }
        Ok(EncodingSupport {
            code: self.resolve(entry, &operand.block)?,
            support: spec.support.iter().map(|block| block.0.clone()).collect(),
        })
    }
}

struct EncodingSupport {
    code: Arc<Code>,
    support: Vec<String>,
}

impl EncodingSupport {
    fn with_block_types(
        self,
        entry: usize,
        circuit_side: &BTreeMap<String, String>,
        target: &InstructionSet,
    ) -> Result<Encoding, String> {
        let sole_block_type = match target.blocks.as_slice() {
            [only] => Some(only.name.as_str()),
            _ => None,
        };
        let block_types = self.support.iter().map(|label| {
            circuit_side.get(label).cloned().or_else(|| sole_block_type.map(str::to_owned)).ok_or_else(|| format!(
                "encoding entry {entry} references circuit operand '{label}', but the circuit's \
                 `in`/`out` typing is elided and target instruction set '{}' declares {} block types (elision requires exactly one)",
                target.name, target.blocks.len()
            ))
        }).collect::<Result<_, _>>()?;
        Ok(Encoding {
            code: self.code,
            support: self.support,
            block_types,
        })
    }
}

fn resolve_gadget_checks<'a>(
    gadget_path: &Path,
    gadget: &'a GadgetSpec,
    check_lists: &'a BTreeMap<PathBuf, Vec<ParityEquation>>,
) -> Result<&'a [ParityEquation], String> {
    match &gadget.checks {
        Sourced::Inline(list) => Ok(list.as_slice()),
        Sourced::File { path } => {
            let resolved = resolve_relative(gadget_path, path);
            check_lists
                .get(&resolved)
                .map(Vec::as_slice)
                .ok_or_else(|| format!("checks reference {} not found", resolved.display()))
        }
    }
}

pub(crate) fn resolve_relative(referrer: &Path, target: &str) -> PathBuf {
    let directory = referrer.parent().unwrap_or_else(|| Path::new(""));
    let joined = directory.join(target);
    normalize_path(&joined)
}

pub(crate) fn normalize_relative(path: &str) -> String {
    normalize_path(Path::new(path)).to_string_lossy().into_owned()
}

fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut components: Vec<Component> = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(components.last(), Some(Component::Normal(_))) {
                    components.pop();
                } else if !matches!(components.last(), Some(Component::RootDir | Component::Prefix(_))) {
                    components.push(component);
                }
            }
            other => components.push(other),
        }
    }
    components.iter().collect()
}
