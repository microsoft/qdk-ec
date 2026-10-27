//! Final YAML layout shared by displays and persistence.
//!
//! Serde determines document shapes and scalar styles. Libyaml then renders
//! single-line scalar sequences in flow style. It never interprets circuit text.
//! Parser and emitter allocations stay at stable addresses; the emitter consumes
//! each owned document, and its synchronous output callback borrows the local buffer.

use std::ffi::{CStr, c_void};
use std::io;
use std::mem::MaybeUninit;
use std::slice;

use unsafe_libyaml as unsafe_yaml;

pub(crate) fn to_string(value: &impl serde::Serialize) -> io::Result<String> {
    let text = serde_yaml::to_string(value).map_err(io::Error::other)?;
    compact(&text)
}

struct Parser(Box<unsafe_yaml::yaml_parser_t>);

impl Drop for Parser {
    fn drop(&mut self) {
        unsafe { unsafe_yaml::yaml_parser_delete(&raw mut *self.0) };
    }
}

struct Document(Option<unsafe_yaml::yaml_document_t>);

impl Drop for Document {
    fn drop(&mut self) {
        if let Some(document) = &mut self.0 {
            unsafe { unsafe_yaml::yaml_document_delete(document) };
        }
    }
}

struct Emitter(Box<unsafe_yaml::yaml_emitter_t>);

impl Drop for Emitter {
    fn drop(&mut self) {
        unsafe { unsafe_yaml::yaml_emitter_delete(&raw mut *self.0) };
    }
}

fn compact(text: &str) -> io::Result<String> {
    let mut output = Vec::<u8>::new();
    unsafe {
        let mut parser = Box::<unsafe_yaml::yaml_parser_t>::new_uninit();
        if unsafe_yaml::yaml_parser_initialize(parser.as_mut_ptr()).fail {
            return Err(io::Error::other("could not initialize YAML parser"));
        }
        let mut parser = Parser(parser.assume_init());
        unsafe_yaml::yaml_parser_set_input_string(&raw mut *parser.0, text.as_ptr(), text.len() as u64);
        let mut document = MaybeUninit::<unsafe_yaml::yaml_document_t>::uninit();
        if unsafe_yaml::yaml_parser_load(&raw mut *parser.0, document.as_mut_ptr()).fail {
            return Err(problem(parser.0.problem));
        }
        let mut document = Document(Some(document.assume_init()));
        if let Some(document) = document.0.as_mut() {
            compact_sequences(document);
        }

        let mut emitter = Box::<unsafe_yaml::yaml_emitter_t>::new_uninit();
        if unsafe_yaml::yaml_emitter_initialize(emitter.as_mut_ptr()).fail {
            return Err(io::Error::other("could not initialize YAML emitter"));
        }
        let mut emitter = Emitter(emitter.assume_init());
        unsafe_yaml::yaml_emitter_set_unicode(&raw mut *emitter.0, true);
        unsafe_yaml::yaml_emitter_set_width(&raw mut *emitter.0, -1);
        unsafe_yaml::yaml_emitter_set_output(
            &raw mut *emitter.0,
            write_output,
            std::ptr::from_mut(&mut output).cast(),
        );
        if let Some(mut document) = document.0.take()
            && unsafe_yaml::yaml_emitter_dump(&raw mut *emitter.0, &raw mut document).fail
        {
            return Err(problem(emitter.0.problem));
        }
        if unsafe_yaml::yaml_emitter_close(&raw mut *emitter.0).fail {
            return Err(problem(emitter.0.problem));
        }
    }
    String::from_utf8(output).map_err(io::Error::other)
}

unsafe fn problem(message: *const i8) -> io::Error {
    let message = if message.is_null() {
        "YAML formatting failed".to_owned()
    } else {
        unsafe { CStr::from_ptr(message.cast()) }.to_string_lossy().into_owned()
    };
    io::Error::other(message)
}

unsafe fn compact_sequences(document: &mut unsafe_yaml::yaml_document_t) {
    let mut node = document.nodes.start;
    while node < document.nodes.top {
        unsafe {
            if (*node).type_ == unsafe_yaml::YAML_SEQUENCE_NODE {
                let items = (*node).data.sequence.items;
                let mut item = items.start;
                let mut compact = true;
                while item < items.top {
                    let child = unsafe_yaml::yaml_document_get_node(document, *item);
                    if (*child).type_ != unsafe_yaml::YAML_SCALAR_NODE {
                        compact = false;
                        break;
                    }
                    let scalar = (*child).data.scalar;
                    let Ok(length) = usize::try_from(scalar.length) else {
                        compact = false;
                        break;
                    };
                    let value = slice::from_raw_parts(scalar.value, length);
                    if std::str::from_utf8(value).map_or(true, |value| {
                        value.contains(['\n', '\r', '\u{85}', '\u{2028}', '\u{2029}'])
                    }) {
                        compact = false;
                        break;
                    }
                    item = item.add(1);
                }
                if compact {
                    (*node).data.sequence.style = unsafe_yaml::YAML_FLOW_SEQUENCE_STYLE;
                }
            }
            node = node.add(1);
        }
    }
}

unsafe fn write_output(data: *mut c_void, buffer: *mut u8, length: u64) -> i32 {
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    let output = unsafe { &mut *data.cast::<Vec<u8>>() };
    if output.try_reserve(length).is_err() {
        return 0;
    }
    output.extend_from_slice(unsafe { slice::from_raw_parts(buffer, length) });
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn flat_lists_are_compact_but_document_structure_is_not() {
        let value: serde_yaml::Value = serde_yaml::from_str(
            "in:\n- C4: [0, 1, 2, 3]\nchecks:\n- ['circuit.readouts[0]', 'in[0].stabilizers[0]']\n",
        )
        .unwrap();
        let output = to_string(&value).unwrap();
        assert!(output.contains("- C4: [0, 1, 2, 3]"), "{output}");
        assert!(output.contains("checks:\n- ["), "{output}");
        assert_eq!(serde_yaml::from_str::<serde_yaml::Value>(&output).unwrap(), value);
    }

    #[test]
    fn multiline_values_remain_block_scalars() {
        let value = serde_json::json!({"source": "R 0\nM 0\n", "notes": ["one\ntwo\n", "plain"]});
        let output = to_string(&value).unwrap();
        assert!(output.contains("source: |\n"), "{output}");
        assert!(output.contains("notes:\n- |\n"), "{output}");
        assert_eq!(serde_yaml::from_str::<serde_json::Value>(&output).unwrap(), value);
    }

    #[test]
    fn flow_context_preserves_scalar_types_and_quoting() {
        let value = serde_json::json!({
            "strings": ["true", "false", "null", "~", "yes", "0", "1e3", "a,b", "a: b", "[x]", "{a}", "#", " leading", "trailing ", "O'Brien", "\\", "\"", "\u{03bb}"],
            "typed": [true, false, null, 0, 18_446_744_073_709_551_615_u64, -7, 1.25],
            "empty": [],
            "nested": [[1, 2], [3]],
            "mapping": [{"value": [1, 2]}],
        });
        let output = to_string(&value).unwrap();
        assert_eq!(serde_yaml::from_str::<serde_json::Value>(&output).unwrap(), value);
        assert!(output.contains("strings: ["), "{output}");
        assert!(output.contains("nested:\n- [1, 2]"), "{output}");
        assert!(output.contains("mapping:\n- value: [1, 2]"), "{output}");
    }

    #[test]
    fn scalar_tags_and_nonfinite_numbers_are_preserved() {
        let value: serde_yaml::Value = serde_yaml::from_str("items: [!label text, .inf, -.inf, .nan, 'null']").unwrap();
        let output = to_string(&value).unwrap();
        assert_eq!(serde_yaml::from_str::<serde_yaml::Value>(&output).unwrap(), value);
    }

    #[test]
    fn malformed_yaml_reports_an_error() {
        assert!(compact("source: [unterminated").is_err());
    }

    proptest! {
        #[test]
        fn arbitrary_strings_round_trip(values in proptest::collection::vec(any::<String>(), 0..20)) {
            let output = to_string(&values).unwrap();
            prop_assert_eq!(serde_yaml::from_str::<Vec<String>>(&output).unwrap(), values);
        }
    }
}
