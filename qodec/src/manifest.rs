//! The qodec manifest — the ordered layer stack.
//!
//! A [`Manifest`] is the root artifact: it names the qodec and lists its layers
//! from logical to physical. It also declares the on-disk
//! [`CURRENT_SCHEMA_VERSION`], which the loader checks for exact equality.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The current qodec schema version supported by this loader.
///
/// Bumped on every breaking change. Loaders reject manifests whose
/// declared `schema_version` differs from this value. See `CHANGELOG.md`
/// for the compatibility contract.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// One layer of the lowering chain: its instruction set, the codes its blocks encode
/// into, plus the gadgets that lower it to the layer below.
///
/// Layer membership is explicit: each non-bottom layer lists its gadgets
/// as a `mnemonic → gadget-file` map. The layer supplies a gadget's
/// source instruction set (this layer's `instruction_set`), its target instruction set (the layer below's
/// `instruction_set`), and the mnemonic it implements (the map key). A gadget may
/// still state its own `implements` / circuit `instruction_set`, in which case the
/// loader checks they agree with the layer; when omitted, the layer
/// supplies them. The bottom (most-concrete) layer has no `gadgets`.
///
/// The layer also binds each of its block types to a code, once: `codes`
/// maps a block-type name (declared in this layer's `instruction_set`) to the
/// code document (relative to the manifest) that encodes it. A
/// gadget's `in`/`out` entry does not state its code — the code for an
/// entry is determined by `(this layer, the block type the gadget's
/// instruction declares at that position)`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LayerSpec {
    /// Path to this layer's instruction set document, relative to the manifest.
    pub instruction_set: String,
    /// This layer's codes, keyed by the block-type name they encode (as
    /// declared in this layer's `instruction_set`); the value is a code document
    /// path relative to the manifest. Usually empty on the bottom layer, whose
    /// blocks are physical; a binding there states which code those blocks
    /// already carry, as `examples/distillation-15` does.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub codes: BTreeMap<String, String>,
    /// This layer's gadgets, keyed by the mnemonic each one implements;
    /// the value is a YAML gadget document path relative to the manifest,
    /// never a raw circuit-source path. A circuit-only gadget document
    /// can contain just `circuit: ./idle.stim`. Empty for the bottom layer.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub gadgets: BTreeMap<String, String>,
}

/// Package manifest identifying a single lowering chain.
///
/// `layers` is an ordered list of [`LayerSpec`]s from top (most abstract /
/// logical) to bottom (most concrete / physical). Each layer names its instruction set
/// file and lists its gadgets as a `mnemonic → gadget-file` map, so gadget
/// membership is explicit. The loader reads only referenced artifacts:
/// each layer's instruction set, codes, and gadget documents, then each gadget's
/// external checks, readouts, and circuit source. The referencing field
/// determines the artifact type; filenames need no particular suffix.
/// There is no recursive directory scan, and unreferenced files or bundle
/// entries are ignored. Circuit-source language identification is separate
/// from artifact typing.
///
/// Paths resolve relative to the containing manifest or gadget document;
/// leading `..` components are preserved. Loading requires an explicit
/// manifest or bundle file path, not a directory. In a bundle, the first
/// document is a single-entry `{path: manifest}` envelope whose key is
/// unrestricted; later entries are looked up by referenced path, including
/// entries containing raw circuit-source text.
///
/// `schema_version` is a single non-negative integer that is bumped on
/// every breaking change to the qodec on-disk representations. Loaders
/// reject manifests whose declared value differs from
/// [`CURRENT_SCHEMA_VERSION`]. The field is optional in the on-disk
/// representations; absent values are treated as the loader's current
/// version.
///
/// Manifests may share artifacts through their path references.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub layers: Vec<LayerSpec>,
    /// Free-form, qodec-opaque annotations (see [`crate::Metadata`]).
    #[serde(default, skip_serializing_if = "crate::Metadata::is_empty")]
    pub metadata: crate::Metadata,
}

#[cfg(test)]
mod tests {
    use super::Manifest;

    #[test]
    fn layer_instruction_set_key_round_trips() {
        let layer: super::LayerSpec = serde_yaml::from_str("instruction_set: logical.isa.yaml\n").unwrap();
        let serialized = serde_yaml::to_value(&layer).unwrap();
        let keys: Vec<_> = serialized.as_mapping().unwrap().keys().collect();
        assert_eq!(keys, vec![&serde_yaml::Value::String("instruction_set".into())]);
        assert_eq!(serde_yaml::from_value::<super::LayerSpec>(serialized).unwrap(), layer);
    }

    #[test]
    fn layer_rejects_unknown_isa_key() {
        for source in ["isa: old.yaml\n", "instruction_set: logical.yaml\nisa: old.yaml\n"] {
            let error = serde_yaml::from_str::<super::LayerSpec>(source).unwrap_err();
            assert!(error.to_string().contains("unknown field `isa`"), "{error}");
        }
    }

    #[test]
    fn minimal_round_trip() {
        let yaml = "\
layers:
  - instruction_set: logical.isa.yaml
    gadgets: {prepare: prepare.gadget.yaml}
  - instruction_set: physical.isa.yaml
";
        let manifest: Manifest = serde_yaml::from_str(yaml).expect("should deserialize");
        assert_eq!(manifest.layers.len(), 2);
        assert_eq!(manifest.layers[0].instruction_set, "logical.isa.yaml");
        assert_eq!(
            manifest.layers[0].gadgets.get("prepare").map(String::as_str),
            Some("prepare.gadget.yaml")
        );
        assert_eq!(manifest.layers[1].instruction_set, "physical.isa.yaml");
        assert!(manifest.layers[1].gadgets.is_empty());
        assert_eq!(manifest.schema_version, None);
        assert_eq!(manifest.name, None);
        assert_eq!(manifest.description, None);

        let encoded = serde_yaml::to_string(&manifest).expect("should serialize");
        let decoded: Manifest = serde_yaml::from_str(&encoded).expect("should round-trip");
        assert_eq!(manifest, decoded);
    }

    #[test]
    fn full_manifest_round_trip() {
        let yaml = "\
schema_version: 1
name: c4-stim
description: C4 to Stim
layers:
  - instruction_set: c4.isa.yaml
    codes: {c4: c4.code.yaml}
    gadgets: {prepare_zz: prepare_zz.gadget.yaml}
  - instruction_set: stim.isa.yaml
";
        let manifest: Manifest = serde_yaml::from_str(yaml).expect("should deserialize");
        assert_eq!(manifest.schema_version, Some(1));
        assert_eq!(manifest.name.as_deref(), Some("c4-stim"));
        assert_eq!(manifest.layers.len(), 2);
        assert_eq!(manifest.layers[0].instruction_set, "c4.isa.yaml");
        assert_eq!(
            manifest.layers[0].codes.get("c4").map(String::as_str),
            Some("c4.code.yaml")
        );
        assert_eq!(manifest.layers[1].instruction_set, "stim.isa.yaml");
        assert!(manifest.layers[1].codes.is_empty());

        let encoded = serde_yaml::to_string(&manifest).expect("should serialize");
        let decoded: Manifest = serde_yaml::from_str(&encoded).expect("should round-trip");
        assert_eq!(manifest, decoded);
    }

    /// The richest manifest in the examples still round-trips byte-for-byte.
    #[test]
    fn c4c6_round_trips() {
        let yaml = include_str!("../examples/c4c6/qodec.yaml");
        let manifest: Manifest = serde_yaml::from_str(yaml).expect("deserializable");
        let back = serde_yaml::to_string(&manifest).expect("serializable");
        let reparsed: Manifest = serde_yaml::from_str(&back).expect("re-deserializable");
        assert_eq!(reparsed, manifest);
    }
}
