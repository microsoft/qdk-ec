use schemars::json_schema;
use schemars::JsonSchema;

use crate::clifford::CliffordUnitary;
use crate::pauli::SparsePauli;

impl JsonSchema for CliffordUnitary {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("CliffordUnitary")
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        json_schema!({
            "type": "string",
            "description": "A Clifford unitary represented as a string of Pauli operator mappings. \
                            Format: comma-separated list of mappings like 'Z₀→Z₀, X₀→X₀' where \
                            subscripts are Unicode subscript digits (₀-₉). Alternatively subscript can be _0, _12 etc.\
                            Each mapping shows \
                            how a Pauli operator (X or Z) on a qubit transforms under the unitary.",
            "examples": [
                "Z₀→Z₀, X₀→X₀",
                "Z_1 -> X_1, X_1 -> Z_0",
                "Z₀→Z₀, X₀→X₀, Z₁→Z₀Z₁, X₁→X₁"
            ]
        })
    }
}

impl JsonSchema for SparsePauli {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("SparsePauli")
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        json_schema!({
            "type": "string",
            "description": "A Pauli operator in sparse notation with optional phase prefix. \
                            Format: [phase]<operators> where phase is one of '', '+', '-', '𝑖', '-𝑖', 'i', '-i'\
                            and operators are X, Y, Z followed by Unicode subscript indices (₀-₉). \
                            Alternatively subscripts can be _0 , _12 etc. \
                            Identity is represented as 'I'.",
            "examples": [
                "I",
                "X₀",
                "Z_12",
                "iY₀Z₃",
                "-X₀Y₂Z₃",
                "𝑖X₀"
            ]
        })
    }
}
