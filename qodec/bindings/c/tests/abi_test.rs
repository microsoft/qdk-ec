//! Exercise the C ABI by traversing the view tree, as a C caller would.
//!
//! `cargo test` runs with the crate root as the working directory, so the
//! example qodecs are two levels up.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use qodec_c::*;

const REPETITION3: &str = "../../examples/repetition3/repetition3.qodec.yaml";
const ICEBERG: &str = "../../examples/iceberg/iceberg.qodec.yaml";

struct LoadedView(ptr::NonNull<Qodec>);

impl LoadedView {
    fn load(path: &str) -> Self {
        let c_path = CString::new(path).expect("no interior NUL");
        let mut root = ptr::null_mut();
        let status = unsafe { qodec_load(c_path.as_ptr(), &raw mut root) };
        assert_eq!(status, QODEC_STATUS_OK, "opening {path}: {}", last_error());
        Self(ptr::NonNull::new(root).expect("a successful open hands back the root"))
    }

    fn root(&self) -> &Qodec {
        unsafe { self.0.as_ref() }
    }
}

impl Drop for LoadedView {
    fn drop(&mut self) {
        unsafe { qodec_unload(self.0.as_ptr()) };
    }
}

fn with_view(path: &str, body: impl FnOnce(&Qodec)) {
    let owner = LoadedView::load(path);
    body(owner.root());
}

fn last_error() -> String {
    string_at(qodec_last_error()).unwrap_or_else(|| "(no error)".to_owned())
}

fn string_at(pointer: *const c_char) -> Option<String> {
    if pointer.is_null() {
        return None;
    }
    Some(unsafe { CStr::from_ptr(pointer) }.to_string_lossy().into_owned())
}

fn layers(view: &Qodec) -> &[QodecLayer] {
    if view.layers.count == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(view.layers.items, view.layers.count) }
}

fn gadgets(layer: &QodecLayer) -> &[QodecGadget] {
    if layer.gadgets.count == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(layer.gadgets.items, layer.gadgets.count) }
}

fn instructions(run: &QodecInstructions) -> &[QodecInstruction] {
    if run.count == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(run.items, run.count) }
}

fn steps(run: &QodecActionSteps) -> &[QodecActionStep] {
    if run.count == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(run.items, run.count) }
}

fn operands(run: &QodecBlockOperands) -> &[QodecBlockOperand] {
    if run.count == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(run.items, run.count) }
}

fn blocks(run: &QodecBlocks) -> &[QodecBlock] {
    if run.count == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(run.items, run.count) }
}

fn parameters(run: &QodecParameters) -> &[QodecParameter] {
    if run.count == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(run.items, run.count) }
}

fn calls(run: &QodecInstructionCalls) -> &[QodecInstructionCall] {
    if run.count == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(run.items, run.count) }
}

fn arguments(run: &QodecArguments) -> &[QodecArgument] {
    if run.count == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(run.items, run.count) }
}

fn indices(run: &QodecIndices) -> &[u64] {
    if run.count == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(run.items, run.count) }
}

/// One `select` pattern: an AND-conjunction of `(flag, expected bit)`.
type Pattern = Vec<(String, u8)>;

/// Expand an `select` CSR into its patterns, each a list of `(flag, bit)`.
fn select_patterns(view: QodecSelect) -> Vec<Pattern> {
    if view.count == 0 {
        return Vec::new();
    }
    let offsets = unsafe { std::slice::from_raw_parts(view.offsets, view.count + 1) };
    let constraints = unsafe { std::slice::from_raw_parts(view.constraints, view.total) };
    offsets
        .windows(2)
        .map(|window| {
            constraints[window[0]..window[1]]
                .iter()
                .map(|constraint| (string_at(constraint.flag).expect("a flag is named"), constraint.bit))
                .collect()
        })
        .collect()
}

fn encodings(run: &QodecEncodings) -> &[QodecEncoding] {
    if run.count == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(run.items, run.count) }
}

/// Walk a parity view into one vector of references per equation.
fn equations(view: QodecParity) -> Vec<Vec<QodecReference>> {
    if view.count == 0 {
        assert!(view.offsets.is_null() && view.references.is_null());
        return Vec::new();
    }
    let offsets = unsafe { std::slice::from_raw_parts(view.offsets, view.count + 1) };
    let references = unsafe { std::slice::from_raw_parts(view.references, view.total) };
    assert_eq!(offsets[0], 0, "CSR offsets start at zero");
    assert_eq!(offsets[view.count], view.total, "and end at the total");
    (0..view.count)
        .map(|i| references[offsets[i]..offsets[i + 1]].to_vec())
        .collect()
}

/// Walk a string view, checking each entry is NUL-terminated in place.
fn strings(view: QodecStrings) -> Vec<String> {
    if view.count == 0 {
        assert!(view.offsets.is_null() && view.bytes.is_null());
        return Vec::new();
    }
    let offsets = unsafe { std::slice::from_raw_parts(view.offsets, view.count + 1) };
    assert_eq!(offsets[0], 0, "CSR offsets start at zero");
    assert_eq!(offsets[view.count], view.total, "and end at the total");
    (0..view.count)
        .map(|i| {
            let entry = unsafe { CStr::from_ptr(view.bytes.add(offsets[i])) };
            assert_eq!(
                entry.to_bytes().len(),
                offsets[i + 1] - offsets[i] - 1,
                "each entry is NUL-terminated in place"
            );
            entry.to_string_lossy().into_owned()
        })
        .collect()
}

fn find<'view>(layer: &'view QodecLayer, mnemonic: &str) -> &'view QodecGadget {
    let name = CString::new(mnemonic).expect("no interior NUL");
    let gadget = unsafe { qodec_find_gadget(std::ptr::from_ref(layer), name.as_ptr()) };
    assert!(!gadget.is_null(), "gadget {mnemonic}: {}", last_error());
    unsafe { &*gadget }
}

#[test]
fn abi_version_is_reported() {
    // Comparing the accessor with the constant it returns proves nothing; the
    // literal is what a consumer compiled against ABI 1 actually depends on.
    assert_eq!(QODEC_ABI_VERSION, 1);
    assert_eq!(qodec_abi_version(), 1);
}

fn projection_documents() -> [serde_json::Value; 4] {
    [
        serde_json::json!({"entry": {"layers": [
            {"instruction_set": "top", "gadgets": {"aa": "body", "az": "body"}},
            {"instruction_set": "bottom"}
        ]}}),
        serde_json::json!({"top": {"name": "top", "blocks": {}, "instructions": [
            {"mnemonic": "aa", "description": ""}, {"mnemonic": "az", "description": ""}
        ]}}),
        serde_json::json!({"bottom": {"name": "bottom", "blocks": {}, "instructions": []}}),
        serde_json::json!({"body": {"circuit": []}}),
    ]
}

fn write_projection_bundle(documents: [serde_json::Value; 4]) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("entry");
    std::fs::write(&path, documents.map(|document| document.to_string()).join("\n---\n")).unwrap();
    (directory, path)
}

fn assert_projection_rejected(path: &std::path::Path, value: &str) {
    qodec::Qodec::load(path).expect("the Rust model remains loadable");
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let mut root = ptr::null_mut();
    let status = unsafe { qodec_load(c_path.as_ptr(), &raw mut root) };
    unsafe { qodec_unload(root) };
    assert_eq!(status, QODEC_STATUS_ERROR);
    assert!(root.is_null());
    assert_eq!(
        last_error(),
        format!("qodec_load: C projection cannot represent a string containing NUL: {value:?}")
    );
}

#[test]
fn nul_mnemonic_collision_is_rejected() {
    let mut documents = projection_documents();
    documents[0]["entry"]["layers"][0]["gadgets"] = serde_json::json!({"a\0z": "body", "az": "body"});
    documents[1]["top"]["instructions"][0]["mnemonic"] = serde_json::json!("a\0z");
    let (_directory, path) = write_projection_bundle(documents);
    let model = qodec::Qodec::load(&path).unwrap();
    assert!(model.layers()[0].gadgets.contains_key("a\0z"));
    assert!(model.layers()[0].gadgets.contains_key("az"));
    assert_projection_rejected(&path, "a\0z");
}

#[test]
fn nul_circuit_source_is_rejected() {
    for source in ["\0az", "a\0z", "az\0"] {
        let mut documents = projection_documents();
        documents[3]["body"]["circuit"] = serde_json::json!({"format": "yaml", "source": source});
        let (_directory, path) = write_projection_bundle(documents);
        let model = qodec::Qodec::load(&path).unwrap();
        assert_eq!(model.layers()[0].gadgets["aa"].circuit.source, source);
        assert_projection_rejected(&path, source);
    }
}

#[test]
fn nul_string_collection_entries_are_rejected() {
    for value in ["\0az", "a\0z", "az\0"] {
        let mut documents = projection_documents();
        documents[1]["top"]["instructions"][0]["flags"] = serde_json::json!(["before", value, "after"]);
        let (_directory, path) = write_projection_bundle(documents);
        let model = qodec::Qodec::load(&path).unwrap();
        assert_eq!(
            model.layers()[0].instruction_set.instructions[0].flags,
            ["before", value, "after"]
        );
        assert_projection_rejected(&path, value);
    }
}

#[test]
fn nul_call_arguments_are_rejected() {
    for arguments in [
        serde_json::json!({"label": "a\0z"}),
        serde_json::json!({"names": ["before", "a\0z", "after"]}),
    ] {
        let mut documents = projection_documents();
        documents[2]["bottom"]["instructions"] = serde_json::json!([
            {"mnemonic": "probe", "description": "", "parameters": {"label": "string", "names": "string"}}
        ]);
        let source = serde_json::json!([{"probe": {"arguments": arguments}}]).to_string();
        assert!(!source.contains('\0'));
        documents[3]["body"]["circuit"] = serde_json::json!({"format": "yaml", "source": source});
        let (_directory, path) = write_projection_bundle(documents);
        assert_projection_rejected(&path, "a\0z");
    }
}

#[test]
fn gadgets_are_sorted_by_mnemonic() {
    let mut documents = projection_documents();
    documents[1]["top"]["instructions"][0]["flags"] = serde_json::json!(["before", "a\\0z", "az", "after"]);
    let (_directory, path) = write_projection_bundle(documents);
    with_view(path.to_str().unwrap(), |view| {
        let layer = &layers(view)[0];
        let names: Vec<_> = gadgets(layer)
            .iter()
            .map(|gadget| string_at(gadget.implements.mnemonic).unwrap())
            .collect();
        assert_eq!(names, ["aa", "az"]);
        for name in names {
            let gadget = find(layer, &name);
            assert_eq!(string_at(gadget.implements.mnemonic).unwrap(), name);
            assert_instruction_storage_shared(&gadget.implements, instruction(layer, &name));
        }
        assert_eq!(
            strings(find(layer, "aa").implements.flags),
            ["before", "a\\0z", "az", "after"]
        );
    });
}

#[test]
fn nul_metadata_is_preserved_as_json() {
    let mut documents = projection_documents();
    documents[0]["entry"]["metadata"] = serde_json::json!({"a\0z": "a\0z", "az": "az"});
    let (_directory, path) = write_projection_bundle(documents);
    with_view(path.to_str().unwrap(), |view| {
        let json = string_at(view.metadata_json).unwrap();
        let metadata: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(metadata, serde_json::json!({"a\0z": "a\0z", "az": "az"}));
    });
}

#[test]
fn sparse_frames_and_constant_terms_are_projected() {
    let mut protocol = qodec::Qodec::load(REPETITION3).unwrap();
    let gadget = protocol.layers_mut()[0].gadgets.get_mut("prepare_z").unwrap();
    gadget
        .frames
        .insert(qodec::Reference::parse("out[0].x[0]").unwrap(), Vec::new());
    gadget.frames.insert(
        qodec::Reference::parse("out[0].z[0]").unwrap(),
        vec![qodec::ParityTerm::Bit(true)],
    );
    let directory = std::env::temp_dir().join(format!("qodec-c-frames-{}", std::process::id()));
    protocol.save_bundle(&directory).unwrap();
    with_view(directory.join(protocol.manifest_filename()).to_str().unwrap(), |view| {
        let gadget = find(&layers(view)[0], "prepare_z");
        assert_eq!(strings(gadget.frame_targets), ["out[0].x[0]", "out[0].z[0]"]);
        let terms = equations(gadget.frames);
        assert!(terms[0].is_empty());
        assert_eq!(terms[1][0].tag, QODEC_REFERENCE_CONSTANT);
        assert_eq!(terms[1][0].index, 1);
        assert!(equations(find(&layers(view)[0], "idle").frames).is_empty());
    });
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn unknown_circuit_calls_are_reported_on_the_preserved_circuit() {
    let original =
        std::fs::read_to_string("tests/fixtures/argument-shapes/argument-shapes.qodec.yaml").expect("read fixture");
    let path = std::env::temp_dir().join(format!("qodec-c-unknown-call-{}.yaml", std::process::id()));
    for replacement in ["- missing: [0]", "- missing: {operands: [0]}", "- missing: {}"] {
        let source = original.replacen("- M: [0]", replacement, 1);
        assert_ne!(source, original);
        std::fs::write(&path, source).expect("write invalid bundle");
        with_view(path.to_str().unwrap(), |view| {
            assert_eq!(view.layers.count, 2);
            let circuit = &find(&layers(view)[0], "noop").circuit;
            let message = string_at(circuit.error).expect("per-circuit error");
            assert!(message.contains("unknown instruction"), "{message}");
            assert!(message.contains("missing"), "{message}");
            assert!(string_at(circuit.source).unwrap().contains("missing"));
            assert_eq!(circuit.calls.count, 0);
        });
    }
    std::fs::remove_file(path).expect("remove invalid fixture");
}

#[test]
fn a_bundle_loads_with_arbitrary_outer_filename_and_manifest_key() {
    let directory = std::env::temp_dir().join(format!("qodec-c-reference-loading-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("create temp dir");
    let path = directory.join("archive.data");
    let mut original = qodec::Qodec::load(REPETITION3).expect("load fixture");
    original.set_schema_version(Some(qodec::CURRENT_SCHEMA_VERSION));
    original.set_manifest_filename("entry".to_owned());
    std::fs::write(&path, original.to_bundle_string().expect("serialize fixture")).expect("write bundle");

    with_view(path.to_str().expect("utf-8 path"), |view| {
        assert_eq!(string_at(view.name).as_deref(), Some("repetition3"));
        assert!(view.has_schema_version);
        assert_eq!(view.schema_version, qodec::CURRENT_SCHEMA_VERSION);
        assert_eq!(view.layers.count, 2);
        let gadget = find(&layers(view)[0], "measure_z");
        assert_eq!(
            string_at(gadget.circuit.error).as_deref(),
            Some("No source parser registered for '.stim'")
        );
        assert!(string_at(gadget.circuit.source).is_some_and(|source| source.contains('M')));
        assert_eq!(gadget.circuit.calls.count, 0);
    });

    std::fs::remove_dir_all(&directory).expect("remove temp dir");
}

#[test]
fn the_whole_tree_is_reachable_by_field_access() {
    with_view(REPETITION3, |view| {
        assert_eq!(string_at(view.name).as_deref(), Some("repetition3"));
        assert_eq!(view.layers.count, 2, "a logical layer lowered onto a physical one");

        let [logical, physical] = layers(view) else {
            panic!("expected two layers")
        };
        assert_eq!(string_at(logical.instruction_set_name).as_deref(), Some("repetition3"));
        assert_eq!(string_at(physical.instruction_set_name).as_deref(), Some("stim+rz"));

        let mnemonics: Vec<String> = instructions(&logical.instructions)
            .iter()
            .filter_map(|instruction| string_at(instruction.mnemonic))
            .collect();
        assert!(mnemonics.contains(&"measure_z".to_owned()), "got {mnemonics:?}");

        assert!(logical.gadgets.count > 0, "the logical layer lowers through gadgets");
        assert_eq!(physical.gadgets.count, 0, "the target layer lowers no further");

        let named: Vec<String> = gadgets(logical)
            .iter()
            .filter_map(|gadget| string_at(gadget.implements.mnemonic))
            .collect();
        assert!(named.contains(&"measure_z".to_owned()), "got {named:?}");
    });
}

#[test]
fn a_gadget_carries_its_circuit_and_decoding_surface() {
    with_view(REPETITION3, |view| {
        let logical = &layers(view)[0];
        let gadget = find(logical, "measure_z");

        let source = string_at(gadget.circuit.source).expect("a source");
        assert!(source.contains('M'), "measure_z lowers to measurements: {source:?}");

        let checks = equations(gadget.checks);
        assert!(!checks.is_empty(), "measure_z declares deterministic checks");
        let stabilizer = checks[0]
            .iter()
            .find(|reference| reference.tag == QODEC_REFERENCE_ENCODING_PROPERTY)
            .expect("a check anchors on an encoding sign");
        assert_eq!(stabilizer.boundary, QODEC_BOUNDARY_IN);
        assert_eq!(stabilizer.property, QODEC_PROPERTY_STABILIZER);

        let readouts = equations(gadget.readouts);
        assert_eq!(readouts.len(), 1, "measure_z observes one logical bit");
        assert_eq!(
            strings(gadget.readout_names),
            vec![""],
            "an anonymous readout still has a name-list entry"
        );
    });
}

#[test]
fn a_gadgets_encodings_carry_their_code() {
    with_view(REPETITION3, |view| {
        let gadget = find(&layers(view)[0], "measure_z");
        assert_eq!(gadget.inputs.count, 1, "measure_z consumes one encoded block");
        assert_eq!(gadget.outputs.count, 0, "and is destructive, so produces none");

        let encoding = &encodings(&gadget.inputs)[0];
        assert_eq!(string_at(encoding.code.name).as_deref(), Some("repetition3"));
        assert_eq!(strings(encoding.support), vec!["0", "1", "2"]);
        assert_eq!(strings(encoding.block_types).len(), 3, "one type per support entry");
        assert_eq!(strings(encoding.code.stabilizers), vec!["Z_0 Z_1", "Z_1 Z_2"]);
        assert_eq!(strings(encoding.code.x).len(), 1, "one logical qubit");
        assert_eq!(strings(encoding.code.z).len(), 1);
    });
}

/// A reference's `entry` indexes the gadget's `inputs` / `outputs` run, so the
/// two halves of the model have to line up.
#[test]
fn reference_entries_index_the_encoding_runs() {
    with_view(REPETITION3, |view| {
        for layer in layers(view) {
            for gadget in gadgets(layer) {
                for equation in equations(gadget.checks).iter().chain(&equations(gadget.readouts)) {
                    for reference in equation {
                        if reference.tag != QODEC_REFERENCE_ENCODING_PROPERTY {
                            continue;
                        }
                        let run = if reference.boundary == QODEC_BOUNDARY_IN {
                            gadget.inputs
                        } else {
                            gadget.outputs
                        };
                        assert!(
                            usize::try_from(reference.entry).is_ok_and(|entry| entry < run.count),
                            "entry {} out of range for a run of {}",
                            reference.entry,
                            run.count
                        );
                    }
                }
            }
        }
    });
}

/// The point of returning references rather than the authored strings.
#[test]
fn selectors_are_expanded_before_they_cross_the_boundary() {
    with_view(ICEBERG, |view| {
        let gadget = find(&layers(view)[0], "measure_z_all");
        let checks = equations(gadget.checks);
        assert_eq!(checks.len(), 1, "one global-parity check");

        let mut indices: Vec<u64> = checks[0]
            .iter()
            .filter(|reference| reference.tag == QODEC_REFERENCE_CIRCUIT_READOUT)
            .map(|reference| reference.index)
            .collect();
        assert!(
            indices.len() > 1,
            "the authored `circuit.readouts[0:n]` selector must arrive expanded, got {}",
            indices.len()
        );
        indices.sort_unstable();
        assert_eq!(indices, (0..indices.len() as u64).collect::<Vec<_>>());
    });
}

/// The view tree points into buffers owned by the handle. If any of them moved
/// while `Prepared` was being built, this would read rubbish or fault.
#[test]
fn the_view_tree_survives_being_boxed() {
    // c4c6 has 18 gadgets across two layers, so every vector grows repeatedly
    // during construction.
    with_view("../../examples/c4c6/qodec.yaml", |view| {
        let mut seen = 0;
        for layer in layers(view) {
            assert!(string_at(layer.instruction_set_name).is_some_and(|name| !name.is_empty()));
            for gadget in gadgets(layer) {
                assert!(string_at(gadget.implements.mnemonic).is_some_and(|m| !m.is_empty()));
                assert!(string_at(gadget.circuit.source).is_some());
                assert_gadget_storage_readable(gadget);
                seen += 1;
            }
        }
        assert_eq!(seen, 18, "c4c6 has 18 gadgets; the walker must reach every one");
    });
}

#[test]
fn repeated_reads_return_the_same_borrowed_storage() {
    with_view(REPETITION3, |view| {
        let gadget = find(&layers(view)[0], "measure_z");
        let first = gadget.checks;
        let second = find(&layers(view)[0], "measure_z").checks;
        assert_eq!(first.references, second.references);
        assert_eq!(first.offsets, second.offsets);
    });
}

fn assert_gadget_storage_readable(gadget: &QodecGadget) {
    for call in calls(&gadget.circuit.calls) {
        assert!(string_at(call.mnemonic).is_some_and(|name| !name.is_empty()));
        for argument in arguments(&call.operands).iter().chain(arguments(&call.arguments)) {
            read_argument_storage(argument);
        }
        let _ = select_patterns(call.select);
    }
    for encoding in encodings(&gadget.inputs).iter().chain(encodings(&gadget.outputs)) {
        assert!(string_at(encoding.code.name).is_some_and(|name| !name.is_empty()));
        assert!(!strings(encoding.code.stabilizers).is_empty());
    }
    let _ = equations(gadget.checks);
}

fn read_argument_storage(argument: &QodecArgument) {
    let _ = string_at(argument.name);
    match argument.value {
        QodecArgumentValue::Text { value } => {
            let _ = string_at(value);
        }
        QodecArgumentValue::StringList { strings: list } => {
            let _ = strings(list);
        }
        QodecArgumentValue::QubitList { qubits } => {
            let _ = indices(&qubits);
        }
        QodecArgumentValue::Qubit { .. }
        | QodecArgumentValue::Integer { .. }
        | QodecArgumentValue::Number { .. }
        | QodecArgumentValue::Boolean { .. }
        | QodecArgumentValue::Readout { .. } => {}
    }
}

#[test]
fn a_missing_qodec_reports_an_error() {
    let path = CString::new("definitely/not/here").expect("no interior NUL");
    let mut qodec: *mut Qodec = ptr::null_mut();
    let status = unsafe { qodec_load(path.as_ptr(), &raw mut qodec) };
    assert_eq!(status, QODEC_STATUS_ERROR);
    assert!(qodec.is_null(), "a failed open must not hand back a root");
    assert_ne!(last_error(), "(no error)", "a failing status records a message");
}

#[test]
fn a_directory_containing_a_manifest_reports_an_error() {
    for manifest in [REPETITION3, "../../examples/c4c6/qodec.yaml"] {
        let directory = std::path::Path::new(manifest).parent().expect("manifest directory");
        let path = CString::new(directory.to_str().expect("utf-8 path")).expect("no interior NUL");
        with_view(manifest, |view| {
            for original in [ptr::null_mut(), ptr::from_ref(view).cast_mut()] {
                let mut qodec = original;
                let status = unsafe { qodec_load(path.as_ptr(), &raw mut qodec) };
                assert_eq!(status, QODEC_STATUS_ERROR, "loading {}", directory.display());
                assert_eq!(qodec, original, "a failed load must leave the output pointer unchanged");
                let message = last_error();
                assert!(message.contains("expected a manifest file path"), "got: {message}");
            }
        });
    }
}

#[test]
fn null_inputs_are_rejected_rather_than_dereferenced() {
    assert!(unsafe { qodec_find_gadget(ptr::null(), ptr::null()) }.is_null());

    let mut qodec: *mut Qodec = ptr::null_mut();
    assert_eq!(
        unsafe { qodec_load(ptr::null(), &raw mut qodec) },
        QODEC_STATUS_INVALID_ARG
    );
    let path = CString::new(REPETITION3).expect("no interior NUL");
    assert_eq!(
        unsafe { qodec_load(path.as_ptr(), ptr::null_mut()) },
        QODEC_STATUS_INVALID_ARG
    );

    with_view(REPETITION3, |view| {
        let layer = &layers(view)[0];
        assert!(
            unsafe { qodec_find_gadget(std::ptr::from_ref(layer), ptr::null()) }.is_null(),
            "a null mnemonic is rejected"
        );
        let missing = CString::new("no_such_instruction").expect("no interior NUL");
        assert!(unsafe { qodec_find_gadget(std::ptr::from_ref(layer), missing.as_ptr()) }.is_null());
        assert_ne!(last_error(), "(no error)", "a null return records a message");

        // The bottom layer has no gadgets at all.
        let bottom = &layers(view)[1];
        let name = CString::new("measure_z").expect("no interior NUL");
        assert!(unsafe { qodec_find_gadget(std::ptr::from_ref(bottom), name.as_ptr()) }.is_null());
    });
}

#[test]
fn closing_a_null_qodec_is_a_no_op() {
    unsafe { qodec_unload(ptr::null_mut()) };
}

#[test]
fn invalid_utf8_path_preserves_the_output_pointer() {
    let path = CString::new(vec![0xff]).unwrap();
    with_view(REPETITION3, |view| {
        for original in [ptr::null_mut(), ptr::from_ref(view).cast_mut()] {
            let mut output = original;
            assert_eq!(
                unsafe { qodec_load(path.as_ptr(), &raw mut output) },
                QODEC_STATUS_INVALID_ARG
            );
            assert_eq!(output, original);
            assert_eq!(last_error(), "qodec_load: path is not valid UTF-8");
        }
        assert_eq!(layers(view).len(), 2);
    });
}

#[test]
fn errors_are_thread_local_and_success_does_not_clear_them() {
    let mut output = ptr::null_mut();
    assert_eq!(
        unsafe { qodec_load(ptr::null(), &raw mut output) },
        QODEC_STATUS_INVALID_ARG
    );
    let original = last_error();
    let other = std::thread::spawn(|| {
        assert!(qodec_last_error().is_null());
        assert!(unsafe { qodec_find_gadget(ptr::null(), ptr::null()) }.is_null());
        last_error()
    })
    .join()
    .unwrap();
    assert_eq!(other, "qodec_find_gadget: layer is null");
    with_view(REPETITION3, |_| assert_eq!(last_error(), original));
    assert_eq!(last_error(), original);
}

fn instruction<'a>(layer: &'a QodecLayer, mnemonic: &str) -> &'a QodecInstruction {
    instructions(&layer.instructions)
        .iter()
        .find(|instruction| string_at(instruction.mnemonic).as_deref() == Some(mnemonic))
        .unwrap_or_else(|| panic!("no instruction {mnemonic}"))
}

#[test]
fn an_isa_exposes_its_blocks_and_descriptions() {
    with_view(REPETITION3, |view| {
        let logical = &layers(view)[0];
        assert!(
            string_at(logical.instruction_set_description).is_some_and(|d| d.contains("repetition code")),
            "the instruction set description comes across"
        );

        let declared = blocks(&logical.blocks);
        assert_eq!(declared.len(), 1);
        assert_eq!(string_at(declared[0].name).as_deref(), Some("repetition3"));
        assert_eq!(declared[0].encodes, 1, "the block encodes one logical qubit");
    });
}

#[test]
fn an_instruction_exposes_its_operands_and_description() {
    with_view(REPETITION3, |view| {
        let logical = &layers(view)[0];

        let prepare = instruction(logical, "prepare_z");
        assert!(string_at(prepare.description).is_some_and(|d| d.contains("Prepare")));
        assert_eq!(prepare.inputs.count, 0, "prepare_z consumes nothing");
        assert_eq!(prepare.outputs.count, 1);
        let out = &operands(&prepare.outputs)[0];
        assert_eq!(string_at(out.block).as_deref(), Some("repetition3"));
        assert!(!out.is_variadic);

        let idle = instruction(logical, "idle");
        assert_eq!(idle.inputs.count, 1, "idle passes its block through");
        assert_eq!(idle.outputs.count, 1);
        assert_eq!(idle.action.count, 0, "idle declares no action");
    });
}

#[test]
fn stabilize_and_observe_actions_carry_their_paulis() {
    with_view(REPETITION3, |view| {
        let logical = &layers(view)[0];

        let prepare = steps(&instruction(logical, "prepare_z").action);
        assert_eq!(prepare.len(), 1);
        let QodecAction::Stabilize { paulis } = prepare[0].action else {
            panic!("prepare_z stabilizes, got {:?}", prepare[0].action);
        };
        assert_eq!(strings(paulis), vec!["Z_0"]);
        assert!(!prepare[0].has_condition, "unguarded");

        let measure = steps(&instruction(logical, "measure_z").action);
        let QodecAction::Observe { observables } = measure[0].action else {
            panic!("measure_z observes, got {:?}", measure[0].action);
        };
        assert_eq!(strings(observables), vec!["Z_0"], "one observable per outcome bit");
    });
}

#[test]
fn a_rotation_carries_its_axis_and_parameterized_angle() {
    with_view(REPETITION3, |view| {
        let logical = &layers(view)[0];
        let rotate = instruction(logical, "rotate_z");

        let declared = parameters(&rotate.parameters);
        assert_eq!(declared.len(), 1);
        assert_eq!(string_at(declared[0].name).as_deref(), Some("theta"));
        assert_eq!(declared[0].kind, QODEC_PARAMETER_NUMBER);

        let step = &steps(&rotate.action)[0];
        let QodecAction::Rotate {
            axis,
            angle_is_literal,
            angle_operand,
            ..
        } = step.action
        else {
            panic!("rotate_z rotates, got {:?}", step.action);
        };
        assert_eq!(string_at(axis).as_deref(), Some("Z_0"), "the rotation axis");
        assert!(!angle_is_literal, "this angle forwards to a parameter");
        assert_eq!(string_at(angle_operand).as_deref(), Some("theta"), "and names it");
    });
}

#[test]
fn a_clifford_action_carries_its_tableau() {
    with_view(REPETITION3, |view| {
        let physical = &layers(view)[1];
        let cx = instruction(physical, "CX");

        let step = &steps(&cx.action)[0];
        let QodecAction::Clifford { from, to } = step.action else {
            panic!("CX is a Clifford, got {:?}", step.action);
        };
        let from = strings(from);
        let to = strings(to);
        assert!(!from.is_empty(), "a tableau has entries");
        assert_eq!(from.len(), to.len(), "domain and codomain are parallel");
    });
}

#[test]
fn a_gadget_exposes_its_circuit_and_parameter_forwarding() {
    with_view(REPETITION3, |view| {
        let logical = &layers(view)[0];

        let measure = find(logical, "measure_z");
        assert_eq!(
            string_at(measure.circuit.instruction_set_name).as_deref(),
            Some("stim+rz"),
            "the circuit names the instruction set it calls into"
        );
        assert!(string_at(measure.circuit.source).is_some_and(|s| s.contains('M')));

        // rotate_z forwards its instruction parameter into the circuit source.
        let rotate = find(logical, "rotate_z");
        let names = strings(rotate.parameter_names);
        let targets = strings(rotate.parameter_targets);
        assert_eq!(names.len(), targets.len(), "forwarding is a parallel pair");
        assert!(names.contains(&"theta".to_owned()), "got {names:?}");
    });
}

#[test]
fn gadget_implements_reuses_the_instruction_declaration() {
    with_view(REPETITION3, |view| {
        let logical = &layers(view)[0];
        for gadget in gadgets(logical) {
            let mnemonic = string_at(gadget.implements.mnemonic).expect("a mnemonic");
            let declared = instruction(logical, &mnemonic);
            assert_instruction_storage_shared(&gadget.implements, declared);
        }
    });
}

fn assert_instruction_storage_shared(actual: &QodecInstruction, expected: &QodecInstruction) {
    assert_eq!(actual.mnemonic, expected.mnemonic);
    assert_eq!(actual.description, expected.description);
    assert_eq!(actual.metadata_json, expected.metadata_json);
    assert_eq!(
        (actual.inputs.count, actual.inputs.items),
        (expected.inputs.count, expected.inputs.items)
    );
    assert_eq!(
        (actual.outputs.count, actual.outputs.items),
        (expected.outputs.count, expected.outputs.items)
    );
    assert_eq!(
        (actual.parameters.count, actual.parameters.items),
        (expected.parameters.count, expected.parameters.items)
    );
    assert_eq!(
        (actual.action.count, actual.action.items),
        (expected.action.count, expected.action.items)
    );
    assert_eq!(
        (
            actual.flags.count,
            actual.flags.offsets,
            actual.flags.bytes,
            actual.flags.total
        ),
        (
            expected.flags.count,
            expected.flags.offsets,
            expected.flags.bytes,
            expected.flags.total
        ),
    );
}

#[test]
fn a_code_exposes_its_description() {
    with_view(REPETITION3, |view| {
        let gadget = find(&layers(view)[0], "measure_z");
        let code = &encodings(&gadget.inputs)[0].code;
        assert_eq!(string_at(code.name).as_deref(), Some("repetition3"));
        assert!(string_at(code.description).is_some(), "present, even if empty");
        // repetition3 declares none; the field is still a valid empty list.
    });
}

/// Native YAML is interpreted without an external parser.
#[test]
fn a_circuit_is_parsed_into_instruction_calls() {
    with_view("tests/fixtures/argument-shapes/argument-shapes.qodec.yaml", |view| {
        let idle = find(&layers(view)[0], "noop");
        assert!(idle.circuit.error.is_null(), "{:?}", string_at(idle.circuit.error));

        let program = calls(&idle.circuit.calls);
        let mnemonics: Vec<String> = program
            .iter()
            .map(|call| string_at(call.mnemonic).expect("a call names an instruction"))
            .collect();
        assert_eq!(mnemonics, ["M", "probe", "select", "select", "boolean", "boolean"]);
        assert_eq!(positional_qubits(&program[0]), [0]);
        assert_eq!(positional_qubits(&program[1]), [0]);
    });
}

fn positional_qubits(call: &QodecInstructionCall) -> Vec<u64> {
    arguments(&call.operands)
        .iter()
        .map(|slot| {
            assert!(slot.name.is_null(), "operands are positional");
            let QodecArgumentValue::Qubit { index } = slot.value else {
                panic!("a stim target is a qubit, got {:?}", slot.value);
            };
            index
        })
        .collect()
}

/// Inline-YAML calls bind block operands and supply arguments to declared
/// parameters. Each value's tag identifies its supplied representation.
#[test]
fn a_call_binds_named_operands_with_their_tag() {
    with_view(REPETITION3, |view| {
        let rotate = find(&layers(view)[0], "rotate_z");
        let program = calls(&rotate.circuit.calls);
        assert_eq!(program.len(), 1);
        assert_eq!(string_at(program[0].mnemonic).as_deref(), Some("rotate_z"));

        let operands = arguments(&program[0].operands);
        assert_eq!(operands.len(), 1);
        assert!(matches!(operands[0].value, QodecArgumentValue::Qubit { index: 0 }));

        let operands = arguments(&program[0].arguments);
        assert_eq!(operands.len(), 1, "theta is the one named argument");
        assert_eq!(string_at(operands[0].name).as_deref(), Some("theta"));
        let QodecArgumentValue::Text { value } = operands[0].value else {
            panic!("theta binds a string, got {:?}", operands[0].value);
        };
        assert_eq!(string_at(value).as_deref(), Some("theta"));

        assert!(select_patterns(program[0].select).is_empty());
    });
}

/// A call may assert what its own flags read under noiseless execution.
#[test]
fn a_call_carries_its_select_expectation() {
    with_view("../../examples/c4c6/qodec.yaml", |view| {
        let idle = find(&layers(view)[0], "idle");
        let asserted: Vec<(String, Pattern)> = calls(&idle.circuit.calls)
            .iter()
            .flat_map(|call| {
                let mnemonic = string_at(call.mnemonic).expect("a call names an instruction");
                select_patterns(call.select)
                    .into_iter()
                    .map(move |pattern| (mnemonic.clone(), pattern))
            })
            .collect();

        assert!(!asserted.is_empty(), "c4c6's idle asserts its rejection flags");
        for (mnemonic, pattern) in &asserted {
            assert!(!pattern.is_empty(), "{mnemonic} has an empty conjunction");
            assert!(pattern.iter().all(|(_, bit)| *bit <= 1), "{mnemonic} expects a bit");
        }
        assert!(
            asserted
                .iter()
                .any(|(_, pattern)| pattern.iter().any(|(flag, bit)| flag == "reject" && *bit == 0)),
            "got {asserted:?}"
        );
    });
}

/// `format` is what the artifact declared; `effective_format` is what was
/// actually used, so a consumer need not re-implement the inference.
#[test]
fn a_circuit_reports_the_format_it_was_parsed_with() {
    with_view(REPETITION3, |view| {
        let logical = &layers(view)[0];

        let idle = find(logical, "idle");
        assert_eq!(string_at(idle.circuit.format).as_deref(), Some("stim"));
        assert_eq!(string_at(idle.circuit.effective_format).as_deref(), Some("stim"));

        // rotate_z omits `format:` and is an inline-YAML sequence.
        let rotate = find(logical, "rotate_z");
        assert!(rotate.circuit.format.is_null(), "the artifact declares no format");
        assert_eq!(string_at(rotate.circuit.effective_format).as_deref(), Some("yaml"));
    });
}

/// A source that does not parse is reported on its own circuit; the rest of the
/// qodec stays readable, where failing the open would lose it.
#[test]
fn an_unparsable_body_is_reported_without_failing_the_open() {
    with_view("tests/fixtures/unparsable-body/unparsable.qodec.yaml", |view| {
        assert_eq!(string_at(view.name).as_deref(), Some("unparsable-body"));

        let gadget = find(&layers(view)[0], "noop");
        assert_eq!(gadget.circuit.calls.count, 0);
        assert_eq!(
            string_at(gadget.circuit.error).as_deref(),
            Some("No source parser registered for '.openqasm'"),
            "the failure is reported rather than swallowed"
        );

        // The rest of the tree is intact.
        assert_eq!(string_at(gadget.circuit.effective_format).as_deref(), Some("openqasm"));
        assert_eq!(
            string_at(layers(view)[1].instruction_set_name).as_deref(),
            Some("bottom")
        );
        assert_eq!(instructions(&layers(view)[0].instructions).len(), 1);
    });
}

/// Every argument shape the model can bind, projected with its tag intact.
///
/// The Python bindings render these as native Python values, which collapses
/// `QUBIT` onto `INTEGER` and `READOUT` onto `TEXT`; the tag is what lets a C
/// consumer tell them apart without consulting the instruction set.
#[test]
fn every_argument_shape_survives_the_boundary() {
    with_view("tests/fixtures/argument-shapes/argument-shapes.qodec.yaml", |view| {
        assert!(view.has_schema_version);
        assert_eq!(view.schema_version, 1);
        let gadget = find(&layers(view)[0], "noop");
        assert!(gadget.circuit.error.is_null(), "{:?}", string_at(gadget.circuit.error));

        let program = calls(&gadget.circuit.calls);
        assert_eq!(
            program.len(),
            6,
            "M, probe, then full and shorthand select and Boolean calls"
        );
        assert_probe_argument_values(&program[1]);
        assert_selection_arguments(&layers(view)[1], &program[2..4]);
        assert_boolean_arguments(&layers(view)[1], &program[4..]);
    });
}

fn argument_value(call: &QodecInstructionCall, name: &str) -> QodecArgumentValue {
    arguments(&call.arguments)
        .iter()
        .find(|argument| string_at(argument.name).as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no argument {name:?}"))
        .value
}

fn assert_probe_argument_values(probe: &QodecInstructionCall) {
    let operands = arguments(&probe.operands);
    assert_eq!(operands.len(), 1);
    assert!(matches!(operands[0].value, QodecArgumentValue::Qubit { index: 0 }));

    let by_name = |name| argument_value(probe, name);

    let QodecArgumentValue::Number { value } = by_name("angle") else {
        panic!("angle is a real literal");
    };
    assert!((value - 0.5).abs() < f64::EPSILON);

    assert!(matches!(by_name("count"), QodecArgumentValue::Integer { value: -1 }));

    let QodecArgumentValue::Text { value } = by_name("label") else {
        panic!("label is a string literal");
    };
    assert_eq!(string_at(value).as_deref(), Some("tag"));

    // Python renders this as the string "circuit.readouts[0]"; the tag
    // keeps it a measurement-record position.
    assert!(matches!(by_name("gated"), QodecArgumentValue::Readout { index: 0 }));

    let QodecArgumentValue::StringList { strings: names } = by_name("names") else {
        panic!("names is a string list");
    };
    assert_eq!(strings(names), ["alpha", "beta"]);

    let QodecArgumentValue::QubitList { qubits } = by_name("targets") else {
        panic!("targets is a qubit list");
    };
    assert_eq!(indices(&qubits), [1, 2]);
}

fn assert_selection_arguments(layer: &QodecLayer, program: &[QodecInstructionCall]) {
    let declared = instruction(layer, "select");
    assert_eq!(strings(declared.flags), ["select"]);
    let declared_parameters = parameters(&declared.parameters);
    assert_eq!(declared_parameters.len(), 1);
    assert_eq!(string_at(declared_parameters[0].name).as_deref(), Some("select"));
    assert_eq!(declared_parameters[0].kind, QODEC_PARAMETER_INTEGER);

    for (call, expected_value) in program.iter().zip([-2, -3]) {
        assert_eq!(string_at(call.mnemonic).as_deref(), Some("select"));
        let operands = arguments(&call.operands);
        assert_eq!(operands.len(), 1);
        assert!(operands[0].name.is_null());
        assert!(matches!(operands[0].value, QodecArgumentValue::Qubit { index: 0 }));
        let bound = arguments(&call.arguments);
        assert_eq!(bound.len(), 1);
        assert_eq!(string_at(bound[0].name).as_deref(), Some("select"));
        let QodecArgumentValue::Integer { value } = bound[0].value else {
            panic!("select is an ordinary integer argument");
        };
        assert_eq!(value, expected_value);
    }
    assert_eq!(select_patterns(program[0].select), vec![vec![("select".to_owned(), 0)]]);
    assert!(select_patterns(program[1].select).is_empty());
}

fn assert_boolean_parameter_kinds(layer: &QodecLayer) {
    let declared = instruction(layer, "boolean");
    for parameter in parameters(&declared.parameters) {
        let expected_kind = match string_at(parameter.name).as_deref() {
            Some("select" | "disabled") => QODEC_PARAMETER_BOOLEAN,
            Some("one" | "zero") => QODEC_PARAMETER_INTEGER,
            name => panic!("unexpected Boolean fixture parameter {name:?}"),
        };
        assert_eq!(parameter.kind, expected_kind);
    }
}

fn assert_boolean_arguments(layer: &QodecLayer, program: &[QodecInstructionCall]) {
    assert_boolean_parameter_kinds(layer);
    for (call, expected) in program.iter().zip([true, false]) {
        assert_eq!(string_at(call.mnemonic).as_deref(), Some("boolean"));
        assert!(matches!(
            arguments(&call.operands)[0].value,
            QodecArgumentValue::Qubit { index: 0 }
        ));
        let bound = arguments(&call.arguments);
        assert_eq!(bound.len(), 4);
        for argument in bound {
            match string_at(argument.name).as_deref() {
                Some("select" | "disabled") => {
                    let QodecArgumentValue::Boolean { value } = argument.value else {
                        panic!("Boolean literals must retain their tag");
                    };
                    assert_eq!(
                        value,
                        expected == (string_at(argument.name).as_deref() == Some("select"))
                    );
                }
                // A parameter argument crosses as an integer whatever its sign.
                Some("one") => assert!(matches!(argument.value, QodecArgumentValue::Integer { value: 1 })),
                Some("zero") => assert!(matches!(argument.value, QodecArgumentValue::Integer { value: 0 })),
                name => panic!("unexpected Boolean fixture argument {name:?}"),
            }
        }
    }
    assert_eq!(select_patterns(program[0].select), vec![vec![("select".to_owned(), 1)]]);
    assert!(select_patterns(program[1].select).is_empty());
}

/// Free-form annotations cross as JSON object text, since C has no shape for
/// arbitrary JSON. The examples carry none, so this writes its own qodec.
fn write_metadata_fixture() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("qodec-c-metadata-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    std::fs::write(
        dir.join("base.isa.yaml"),
        "name: base\ndescription: probe instruction set\nblocks: {qubit: 1}\ninstructions:\n  - mnemonic: idle\n    description: no-op\n    in: [qubit]\n    out: [qubit]\n    action: []\nmetadata:\n  vendor: acme\n",
    )
    .expect("write instruction_set");
    std::fs::write(
        dir.join("qodec.yaml"),
        "name: probe\nmetadata:\n  owner: qec-team\n  revision: 3\nlayers:\n  - instruction_set: base.isa.yaml\n  - instruction_set: phys.isa.yaml\n",
    )
    .expect("write manifest");
    std::fs::write(
        dir.join("phys.isa.yaml"),
        "name: phys\ndescription: physical\nblocks: {qubit: 1}\ninstructions:\n  - mnemonic: idle\n    description: no-op\n    in: [qubit]\n    out: [qubit]\n    action: []\n",
    )
    .expect("write physical instruction_set");
    dir
}

#[test]
fn metadata_crosses_as_json_text() {
    let dir = write_metadata_fixture();
    with_view(dir.join("qodec.yaml").to_str().expect("utf-8 path"), |view| {
        let manifest = string_at(view.metadata_json).expect("manifest metadata");
        assert!(manifest.contains("\"owner\":\"qec-team\""), "got: {manifest}");
        assert!(manifest.contains("\"revision\":3"), "got: {manifest}");

        let layer = unsafe { &*view.layers.items };
        let instruction_set = string_at(layer.instruction_set_metadata_json).expect("instruction_set metadata");
        assert!(
            instruction_set.contains("\"vendor\":\"acme\""),
            "got: {instruction_set}"
        );
    });

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn absent_metadata_is_an_empty_json_object() {
    with_view(REPETITION3, |view| {
        let mut fields = vec![view.metadata_json];
        for layer in layers(view) {
            fields.push(layer.instruction_set_metadata_json);
            fields.extend(instructions(&layer.instructions).iter().map(|item| item.metadata_json));
            for gadget in gadgets(layer) {
                fields.extend([gadget.metadata_json, gadget.implements.metadata_json]);
                if gadget.inputs.count > 0 {
                    fields.push(unsafe { &*gadget.inputs.items }.code.metadata_json);
                }
            }
        }
        for field in fields {
            assert_eq!(string_at(field).as_deref(), Some("{}"));
        }
    });
}
