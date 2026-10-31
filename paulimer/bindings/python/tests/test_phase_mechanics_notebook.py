import json
from pathlib import Path


def test_phase_mechanics_notebook():
    notebook = Path(__file__).resolve().parents[1] / "examples" / "phased-outcome-complete-simulation.ipynb"
    cells = [
        "".join(cell["source"])
        for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    ]
    assert len(cells) == 5
    namespace = {"__name__": "__main__"}
    for index, source in enumerate(cells):
        exec(compile(source, f"{notebook}:cell-{index}", "exec"), namespace)  # noqa: S102
