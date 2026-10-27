// Only macOS needs link arguments here; the interpreter resolves Python symbols at import.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
