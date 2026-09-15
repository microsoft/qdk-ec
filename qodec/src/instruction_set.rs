//! Instruction set architectures (instruction sets).
//!
//! An [`InstructionSet`] declares a layer's block types and the instructions
//! that operate on them.

use crate::Instruction;
use serde::{Deserialize, Serialize};

/// A block type declared within an instruction set.
///
/// Serialized as `{name: encodes}`, e.g. `{c4: 2}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Name used by instruction operands to refer to this block type.
    pub name: String,
    /// Number of logical qubits in one block of this type.
    pub encodes: usize,
}

/// Block types and instruction declarations available at one layer.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InstructionSet {
    /// Unique name within a qodec; how a layer refers to this instruction set.
    pub name: String,
    /// Free-form prose for readers; empty when the author supplied none.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// The block types instructions may take as operands, in declaration order.
    #[serde(with = "blocks_serde")]
    pub blocks: Vec<Block>,
    /// The instructions this set declares, with unique mnemonics.
    pub instructions: Vec<Instruction>,
    /// Annotations for external tools; see [`crate::Metadata`].
    #[serde(default, skip_serializing_if = "crate::Metadata::is_empty")]
    pub metadata: crate::Metadata,
}

impl InstructionSet {
    /// Read and validate a standalone instruction set from a YAML file.
    ///
    /// # Errors
    ///
    /// Returns [`crate::LoadError::Io`] if reading fails,
    /// [`crate::LoadError::Yaml`] if parsing fails, or
    /// [`crate::LoadError::InvalidInstructionSet`] if [`Self::validate`] fails.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, crate::LoadError> {
        let path = path.as_ref();
        let instruction_set: Self = crate::qodec::loader::read_yaml(path)?;
        instruction_set
            .validate()
            .map_err(|error| crate::LoadError::InvalidInstructionSet {
                instruction_set: path.to_path_buf(),
                error,
            })?;
        Ok(instruction_set)
    }

    /// Validate this instruction set and write it as YAML, creating parent directories as needed.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if validation, serialization, directory
    /// creation, or writing fails.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::qodec::loader::save_artifact(self, path.as_ref(), Self::validate)
    }
}

impl Serialize for Block {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry(&self.name, &self.encodes)?;
        map.end()
    }
}

/// (De)serialization for an instruction set's `blocks:` list.
///
/// The on-disk form is a plain map `{c4: 2, c6: 3}` of block names to
/// qubit counts. Block order is not semantically meaningful (blocks are
/// referenced by name), but author order is preserved in both directions.
mod blocks_serde {
    use super::Block;
    use serde::de::{MapAccess, Visitor};
    use serde::ser::SerializeMap;
    use serde::{Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S>(blocks: &[Block], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(blocks.len()))?;
        for block in blocks {
            map.serialize_entry(&block.name, &block.encodes)?;
        }
        map.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Block>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct BlocksVisitor;

        impl<'de> Visitor<'de> for BlocksVisitor {
            type Value = Vec<Block>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a map of block names to qubit counts")
            }

            fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut blocks = Vec::new();
                while let Some((name, encodes)) = access.next_entry::<String, usize>()? {
                    blocks.push(Block { name, encodes });
                }
                Ok(blocks)
            }
        }

        deserializer.deserialize_map(BlocksVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::{Block, InstructionSet};

    fn parse(yaml: &str) -> InstructionSet {
        serde_yaml::from_str(yaml).expect("should deserialize")
    }

    #[test]
    fn blocks_round_trip_in_author_order() {
        let instruction_set = parse("name: demo\nblocks: {c6: 3, c4: 2}\ninstructions: []\n");
        assert_eq!(
            instruction_set.blocks,
            vec![
                Block {
                    name: "c6".to_owned(),
                    encodes: 3
                },
                Block {
                    name: "c4".to_owned(),
                    encodes: 2
                },
            ]
        );
        let emitted = serde_yaml::to_string(&instruction_set).expect("should serialize");
        assert!(emitted.contains("c6: 3"), "got: {emitted}");
        assert!(
            emitted.find("c6").unwrap() < emitted.find("c4").unwrap(),
            "author order is preserved on the way out: {emitted}"
        );
    }

    #[test]
    fn a_block_declaration_serializes_as_a_one_entry_map() {
        let block = Block {
            name: "c4".to_owned(),
            encodes: 2,
        };
        assert_eq!(serde_yaml::to_string(&block).expect("serializable").trim(), "c4: 2");
    }

    #[test]
    fn blocks_must_be_a_map() {
        let error = serde_yaml::from_str::<InstructionSet>("name: demo\nblocks: [c4]\ninstructions: []\n")
            .expect_err("a sequence is not a block map");
        assert!(
            error.to_string().contains("a map of block names to qubit counts"),
            "got: {error}"
        );
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let error = serde_yaml::from_str::<InstructionSet>("name: demo\nblocks: {}\ninstructions: []\noops: 1\n")
            .expect_err("deny_unknown_fields");
        assert!(error.to_string().contains("unknown field"), "got: {error}");
    }

    #[test]
    fn empty_description_and_metadata_are_omitted() {
        let instruction_set = parse("name: demo\nblocks: {}\ninstructions: []\n");
        let emitted = serde_yaml::to_string(&instruction_set).expect("serializable");
        assert!(!emitted.contains("description"), "got: {emitted}");
        assert!(!emitted.contains("metadata"), "got: {emitted}");
    }
}

#[cfg(test)]
mod standalone_io_tests {
    use super::InstructionSet;

    fn example_instruction_set() -> InstructionSet {
        serde_yaml::from_str(include_str!("../examples/c4c6/c4c6.isa.yaml")).expect("the example parses")
    }

    #[test]
    fn an_instruction_set_round_trips_through_its_own_file() {
        let directory = tempfile::TempDir::new().expect("temp dir");
        let path = directory.path().join("nested/out.isa.yaml");
        let original = example_instruction_set();
        original.save(&path).expect("saves, creating the parent");
        assert_eq!(InstructionSet::load(&path).expect("loads"), original);
    }

    #[test]
    fn saving_an_ambiguous_instruction_set_does_not_write_it() {
        let mut instruction_set = example_instruction_set();
        let duplicate = instruction_set.blocks[0].clone();
        instruction_set.blocks.push(duplicate);
        let directory = tempfile::TempDir::new().expect("temp dir");
        let path = directory.path().join("nested/ambiguous.isa.yaml");
        let error = instruction_set.save(&path).expect_err("two blocks share a name");
        assert!(
            error.to_string().contains("duplicate block declaration"),
            "got: {error}"
        );
        assert!(!path.exists(), "nothing should be written");
    }

    #[test]
    fn loading_an_ambiguous_instruction_set_names_the_file() {
        let directory = tempfile::TempDir::new().expect("temp dir");
        let path = directory.path().join("ambiguous.isa.yaml");
        std::fs::write(
            &path,
            "name: t\ndescription: d\nblocks: {q: 1}\ninstructions:\n\
             - {mnemonic: R, description: d, out: [q], action: []}\n\
             - {mnemonic: R, description: d, out: [q], action: []}\n",
        )
        .expect("write the fixture");
        let error = InstructionSet::load(&path).expect_err("two instructions share a mnemonic");
        assert!(
            matches!(&error, crate::LoadError::InvalidInstructionSet { instruction_set, .. } if instruction_set == &path),
            "got {error:?}"
        );
        assert!(
            error.to_string().contains("duplicate instruction mnemonic 'R'"),
            "got: {error}"
        );
    }
}
