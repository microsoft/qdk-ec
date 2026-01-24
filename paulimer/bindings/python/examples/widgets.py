"""Interactive widgets for paulimer demos."""

import ipywidgets
from IPython.display import HTML

import paulimer
from paulimer import PauliGroup, SparsePauli as Pauli

STYLES = """
<style>
    .pauli-explorer .group-card {
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d3f 100%);
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #3d3d5c;
        color: #cdd6f4;
        font-size: 14px;
        line-height: 1.6;
    }
    .pauli-explorer .section-title {
        color: #89b4fa;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .pauli-explorer .section {
        margin-bottom: 14px;
    }
    .pauli-explorer .section:last-child { margin-bottom: 0; }
    .pauli-explorer .row {
        margin-bottom: 6px;
        display: flex;
        align-items: baseline;
        gap: 8px;
    }
    .pauli-explorer .row:last-child { margin-bottom: 0; }
    .pauli-explorer .label {
        color: #7f849c;
        min-width: 120px;
        font-size: 14px;
    }
    .pauli-explorer .value { color: #cdd6f4; flex: 1; }
    .pauli-explorer .pauli-gen {
        background: #45475a;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 4px;
        color: #f5c2e7;
        font-weight: 500;
    }
    .pauli-explorer .pauli-el {
        background: #313244;
        padding: 2px 6px;
        border-radius: 3px;
        margin: 1px;
        display: inline-block;
        color: #a6e3a1;
        font-size: 13px;
    }
    .pauli-explorer .pauli-centralizer {
        background: #313244;
        padding: 2px 6px;
        border-radius: 3px;
        margin: 1px;
        display: inline-block;
        color: #94e2d5;
        font-size: 13px;
    }
    .pauli-explorer .badge {
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 500;
        margin-right: 6px;
    }
    .pauli-explorer .badge.good { background: #1e3a2f; color: #a6e3a1; }
    .pauli-explorer .badge.bad { background: #3a1e2a; color: #f38ba8; }
    .pauli-explorer .badge.neutral { background: #3a351e; color: #f9e2af; }
    .pauli-explorer .size-num { color: #89b4fa; font-weight: 600; font-size: 16px; }
    .pauli-explorer .more { color: #6c7086; font-style: italic; }
    .pauli-explorer .dim { color: #6c7086; }
    .pauli-explorer .divider { border-top: 1px solid #3d3d5c; margin: 14px 0; }
</style>
"""


def pauli_group_explorer():
    """Return an interactive Pauli group explorer widget."""
    generators = []
    html_output = ipywidgets.HTML()

    def format_elements(group, max_display=16):
        elements = list(group.elements)
        if len(elements) <= max_display:
            return " ".join(f'<span class="pauli-el">{e}</span>' for e in elements)
        shown = " ".join(
            f'<span class="pauli-el">{e}</span>' for e in elements[:max_display]
        )
        return f'{shown} <span class="more">+{len(elements) - max_display} more</span>'

    def format_symplectic(gens):
        form = list(paulimer.symplectic_form_of(gens))
        if not form:
            return '<span class="dim">none</span>'
        return " ".join(f'<span class="pauli-el">{p}</span>' for p in form)

    def format_generators(group):
        gens = list(group.generators)
        if not gens:
            return '<span class="dim">none</span>'
        return " ".join(f'<span class="pauli-gen">{g}</span>' for g in gens)

    def format_standard_gens(group):
        gens = list(group.standard_generators)
        if not gens:
            return '<span class="dim">none</span>'
        return " ".join(f'<span class="pauli-el">{g}</span>' for g in gens)

    def format_centralizer(centralizer):
        gens = list(centralizer.generators)
        if not gens:
            return '<span class="dim">trivial</span>'
        return " ".join(f'<span class="pauli-centralizer">{g}</span>' for g in gens)

    def normalize_pauli_string(s):
        """Uppercase Pauli operators but preserve phase prefix (i, -i, -, +)."""
        s = s.strip()
        if s.startswith(("-i", "+i")):
            return s[:2] + s[2:].upper()
        if s.startswith(("i", "-", "+")):
            return s[0] + s[1:].upper()
        return s.upper()

    def add_generator(_):
        pauli_str = normalize_pauli_string(text_input.value)
        if pauli_str:
            try:
                generators.append(Pauli(pauli_str))
                update_display()
            except Exception:
                pass
            text_input.value = ""

    def clear_generators(_):
        generators.clear()
        update_display()

    def update_display():
        group = PauliGroup(generators)
        support = list(group.support)
        centralizer = (
            paulimer.centralizer_of(group, supported_by=support)
            if support
            else group
        )

        stabilizer_badge = (
            '<span class="badge good">✓ Stabilizer</span>'
            if group.is_stabilizer_group
            else '<span class="badge bad">✗ Not stabilizer</span>'
        )
        abelian_badge = (
            '<span class="badge good">✓ Abelian</span>'
            if group.is_abelian
            else '<span class="badge neutral">Non-abelian</span>'
        )

        html_output.value = STYLES + f'''
<div class="pauli-explorer">
    <div class="group-card">
        <div class="section">
            <div class="section-title">Group</div>
            <div class="row">
                <span class="label">Generators</span>
                <span class="value">{format_generators(group)}</span>
            </div>
            <div class="row">
                <span class="label">Size</span>
                <span class="value"><span class="size-num">2<sup>{group.log2_size}</sup> = {2**group.log2_size}</span></span>
            </div>
            <div class="row">
                <span class="label">Properties</span>
                <span class="value">{abelian_badge} {stabilizer_badge}</span>
            </div>
        </div>
        
        <div class="divider"></div>
        
        <div class="section">
            <div class="section-title">Structure</div>
            <div class="row">
                <span class="label">Standard form</span>
                <span class="value">{format_standard_gens(group)}</span>
            </div>
            <div class="row">
                <span class="label">Symplectic</span>
                <span class="value">{format_symplectic(group.generators)}</span>
            </div>
            <div class="row">
                <span class="label">Centralizer</span>
                <span class="value">{format_centralizer(centralizer)}</span>
            </div>
        </div>
        
        <div class="divider"></div>
        
        <div class="section">
            <div class="section-title">Elements</div>
            <div class="value">{format_elements(group)}</div>
        </div>
    </div>
</div>
'''

    text_input = ipywidgets.Text(placeholder="Enter Pauli (e.g., ZZI, XXI)")
    add_button = ipywidgets.Button(description="Add", button_style="primary")
    clear_button = ipywidgets.Button(description="Clear")

    add_button.on_click(add_generator)
    clear_button.on_click(clear_generators)
    text_input.on_submit(add_generator)

    update_display()
    return ipywidgets.VBox([ipywidgets.HBox([text_input, add_button, clear_button]), html_output])
