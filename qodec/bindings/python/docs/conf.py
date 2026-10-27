from pathlib import Path

project = "qodec"
extensions = ["autoapi.extension", "sphinx.ext.doctest"]
root_doc = "index"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["qodec.css"]
html_show_copyright = False
exclude_patterns = ["_build"]

autoapi_dirs = [str(Path(__file__).resolve().parents[1] / "python" / "qodec")]
autoapi_file_patterns = ["*.pyi"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
    "special-members",
]
autoapi_python_class_content = "class"
autoapi_add_toctree_entry = False

doctest_global_setup = "import qodec"


def _select_members(app, what, name, obj, skip, options):
    """Document type-only aliases and canonical classes, not duplicate re-exports."""
    if obj.imported and not obj.obj["original_path"].startswith("qodec._"):
        return True
    if name in {"qodec.Action", "qodec.Metadata", "qodec.PauliLike", "qodec.PauliString"}:
        return False
    if what == "method" and name.endswith(".__new__"):
        return False
    return None


def setup(app):
    app.connect("autoapi-skip-member", _select_members)