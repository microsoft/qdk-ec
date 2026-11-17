"""Check that the published circuits share one layout and compile like the evaluated source."""

from pathlib import Path

from generate_circuits import expand, generate
from lattice_surgery import circuit_source
from run import Settings


def test_generated_circuits_share_the_interleaved_layout(tmp_path: Path):
    original, expanded = generate(
        6, tmp_path, Settings(threads=1, distance=3, layers=3)
    )
    original_text = original.read_text()
    expanded_text = expanded.read_text()
    assert "    CNOT 0 2\n" in original_text and "    CNOT 2 4\n" in original_text
    assert "    MeasureZ 10\n" in original_text
    assert "PROGRAM CNOT" not in expanded_text
    for control in (0, 2, 4, 6, 8):
        assert f"    PrepareXNonFT {control + 1}\n" in expanded_text
    assert expanded_text.count("    MZZ ") == 8
    for text in (original_text, expanded_text):
        assert "IN(" not in text and "OUT(" not in text
    assert "    MZZ 0 2\n" in original_text


def test_source_and_expansion_use_positional_calls_directly():
    from deqagram import GadgetApplication, Instruction

    source = circuit_source(3, 0.001, 4, 2, "Z", 42, wire_stride=2)
    assert "IN(" not in source and "OUT(" not in source
    statements = expand(source, {0, 2, 4, 6})
    assert not any(isinstance(statement, GadgetApplication) for statement in statements)
    assert any(
        isinstance(statement, Instruction) and statement.name == "MZZ"
        and [target.index for target in statement.targets] == [0, 1]
        for statement in statements
    )


def test_circuit_diagrams_are_png_only(tmp_path: Path):
    from matplotlib import image
    from draw_circuits import draw

    diagrams = draw(tmp_path)
    assert {path.name for path in diagrams} == {
        "brickwall-circuit.png",
        "lattice-surgery-cnot.png",
    }
    for path in diagrams:
        pixels = image.imread(path)
        assert pixels.shape[0] > 300 and pixels.shape[1] > 1000
        assert pixels.min() < 0.5
    assert not list(tmp_path.glob("*.svg"))
