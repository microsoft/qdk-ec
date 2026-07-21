# deq Language Basics

The deq language is a domain-specific language (DSL) for defining quantum error correction
codes, logical gadgets, and logical programs. Its key promise: **you describe the physical
circuit, and the system automatically derives the decoding hypergraph** — no manual check
(detector) annotation required.

If you've used [Stim](https://github.com/quantumlib/Stim), you know that constructing a
detector error model requires manually annotating every `DETECTOR` in the circuit — a task
that demands deep understanding of stabilizer propagation, round boundaries, and check
structure. The deq language eliminates this burden entirely. You write small physical
gadgets using Stim-compatible circuit syntax, and the JIT transpiler automatically derives
checks and analyzes error effects via Clifford simulation — all offline, so the expensive
stabilizer propagation never needs to happen at runtime.

> **Note on Stim's automatic detector discovery:**
> Recent versions of Stim include experimental detector-finding features
> (`stim.Circuit.missing_detectors`). However, Stim's own documentation explicitly warns:
> *"It's not recommended to use this method to solve for the detectors of a circuit. The
> returned detectors are not guaranteed to be stable across versions, and aren't optimized
> to be 'good' (e.g. form a low weight basis or be matchable if possible)."*
>
> This limitation is fundamental: Stim only allows describing **global circuits**, so once
> the circuit involves multiple syndrome extraction rounds, the auto-generated checks span
> the full 3D spacetime volume — producing an unstructured decoding hypergraph that is
> extremely difficult to optimize globally.
>
> deq addresses this by letting users define **small gadgets** (each consisting of a single
> syndrome extraction round) and then compose them together. The COMPOSE mechanism invokes
> the [JIT compiler](jit-basics.md) to expand checks: since the finished checks and
> unfinished checks are structurally identical across multiple syndrome extraction rounds,
> the generated checks are **always well-structured by construction** — no global
> optimization needed. Although the JIT compiler was originally designed for runtime stream
> decoding, it turns out to be equally valuable for offline processing: it guarantees time-invariant check
> structure for any number of rounds as long as one round of check structure is defined.

This chapter walks through the language from the simplest example to the full feature set.

---

## Setting Up: Syntax Highlighting

Before writing `.deq` files, install a syntax-highlighting extension for your editor.
It makes `.deq` files much more readable — keywords, Pauli operators, measurement
references, and code parameters are all color-coded.

### VS Code

Install via the top-level Makefile:

```sh
make install-extension
```

After installation, any `.deq` file opened in VS Code will have syntax highlighting
automatically.

### Emacs

Install by:

```elisp
(use-package deq-mode
  :load-path "/path/to/deq/deq/circuit/emacs-deq"
  :mode ("\\.deq\\'" . deq-mode))
```

Any file ending in `.deq` will then open in `deq-mode` automatically.

---

## Defining a QEC Code

Every `.deq` file starts with a `CODE` definition — the mathematical specification of a
quantum error correction code:

[A minimal repetition code definition](../examples/language/01_prepare_measure.deq)
<!-- deq-highlight-begin: ../examples/language/01_prepare_measure.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># A minimal example: prepare and measure a repetition code</span></span>
<span class="line"><span style="color:#008000"># No noise, no syndrome extraction — just the simplest possible circuit</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/language/01_prepare_measure.deq -->

The `CODE` block has three parts:

| Keyword                         | Purpose                                                                                                                                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `CODE RepetitionCode [[3,1,1]]` | Declares a code named `RepetitionCode` with parameters $[[n, k, d]] = [[3, 1, 1]]$. The distance $d$ is optional and not used by the system — it's for documentation                       |
| `LOGICAL X0*X1*X2 Z0*Z1*Z2`     | Specifies the logical operators. Each space-separated entry is one logical operator pair (X and Z for each logical qubit). Pauli products use `*` notation: `X0*X1*X2` means $X_0 X_1 X_2$ |
| `STABILIZER Z0*Z1 Z1*Z2`        | Lists the stabilizer generators, space-separated. `Z0*Z1` means $Z_0 Z_1$                                                                                                                  |

The `CODE` definition is pure mathematics — no physical circuit, no noise, no
implementation details. It tells the transpiler what stabilizers to track at port boundaries
(these become the virtual stabilizer measurements described in the
[JIT chapter](jit-basics.md)).

---

## Your First Gadgets: PrepareZ and MeasureZ

A `GADGET` defines the physical circuit for a logical operation. The circuit syntax is
**Stim-compatible** — if you have existing Stim circuits, you can paste them directly into
a `GADGET` block.

The file shown above contains both gadgets and a program. Let's break down each part:

### PrepareZ

[PrepareZ gadget](../examples/language/snippet_prepare.deq)
<!-- deq-highlight-begin: ../examples/language/snippet_prepare.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/language/snippet_prepare.deq -->

| Line                          | Meaning                                                                                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `GADGET PrepareZ {`           | Declares a gadget named `PrepareZ`                                                                                                                                 |
| `R 0 1 2`                     | Reset qubits 0, 1, 2 to $\lvert 0 \rangle$ (Stim instruction)                                                                                                      |
| `OUTPUT RepetitionCode 0 1 2` | Declares an output port of type `RepetitionCode`, mapping physical qubits 0, 1, 2 to the code's logical qubits. This is how the gadget "exports" the encoded state |

PrepareZ has **no input ports** (it creates a fresh state) and **no measurements** (reset
is not a measurement). It has one output port.

### MeasureZ

[MeasureZ gadget](../examples/language/snippet_measure.deq)
<!-- deq-highlight-begin: ../examples/language/snippet_measure.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/language/snippet_measure.deq -->

| Line                              | Meaning                                                                                                                                |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `INPUT RepetitionCode 0 1 2`      | Declares an input port: the gadget receives an encoded state on physical qubits 0, 1, 2                                                |
| `M 0 1 2`                         | Measure all 3 data qubits (Stim instruction)                                                                                           |
| `READOUT rec[-3] rec[-2] rec[-1]` | Defines a logical readout as the XOR of the 3 most recent measurements. `rec[-k]` references the $k$-th most recent measurement record |

MeasureZ has **one input port** and **no output ports** (the encoded state is consumed by
measurement).

### The PROGRAM Block

[Program block](../examples/language/snippet_program.deq)
<!-- deq-highlight-begin: ../examples/language/snippet_program.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/language/snippet_program.deq -->

A `PROGRAM` defines a sequence of gadget applications at the **logical level**. The
numbers after gadget names are code block indices (not physical qubit indices) — each
code block is an instance of a `CODE` type that may contain multiple logical qubits
(this distinction is irrelevant for $k = 1$ codes but essential for qLDPC codes with
$k > 1$; see [codes with multiple logical qubits](codes-multi-logical.md)):

| Line                  | Meaning                                                                                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PrepareZ 0`          | Instantiate PrepareZ on code block 0                                                                                                                                       |
| `MeasureZ 0`          | Instantiate MeasureZ on code block 0, connecting its input to PrepareZ's output                                                                                            |
| `ASSERT_EQ rec[-1] 0` | Assert that the most recent logical readout equals 0 (preparing $\lvert 0 \rangle$ and measuring Z should give 0). This defines the logical error criterion for simulation |

### What the Transpiler Produces

Run the JIT transpiler:
```sh
deq transpile 01_prepare_measure.deq --out 01_prepare_measure.deq.jit --program Simulation
```

The transpiler produces a `.deq.jit` binary (for the runtime) and a `.deq.jit.txt`
human-readable dump. The rest of this section walks through the output format — if you
haven't read the [JIT chapter](jit-basics.md) yet, feel free to skip ahead to
[Adding Noise](#adding-noise) and come back later.

**Port type** — the transpiler extracted the stabilizers from the `CODE` definition:
```protobuf
port_types {
  base { ptype: 1, name: "RepetitionCode"
    observables { tag: "X0*X1*X2" }
    observables { tag: "Z0*Z1*Z2" }
  }
  stabilizers { tag: "Z0*Z1" }
  stabilizers { tag: "Z1*Z2" }
}
```

**PrepareZ** — no measurements, no finished checks, but 2 unfinished checks (one per
stabilizer). No errors because there's no noise:
```protobuf
gadget_types {
  base { gtype: 1, name: "PrepareZ"
    outputs { ptype: 1 }
    # ... matrices omitted ...
  }
  unfinished_checks { base { tag: "CHECK m0" } }
  unfinished_checks { base { tag: "CHECK m1" } }
  # no errors — no noise in this example
}
```

**MeasureZ** — 3 physical measurements, 2 finished checks (auto-derived from the
stabilizers), no unfinished checks (no output port):
```protobuf
gadget_types {
  base { gtype: 2, name: "MeasureZ"
    measurements { tag: "M 0" }
    measurements { tag: "M 1" }
    measurements { tag: "M 2" }
    inputs { ptype: 1 }
    readouts { tag: "READOUT m0 m1 m2"
      measurement_indices: 0  measurement_indices: 1  measurement_indices: 2
    }
    # ... matrices omitted ...
  }
  finished_checks {
    base { tag: "CHECK m0 m2 m3" }
    measurements { input_port: 0 }                    # v₀ⁱⁿ (virtual)
    measurements { }                                  # physical measurement 0
    measurements { measurement_index: 1 }             # physical measurement 1
  }
  finished_checks {
    base { tag: "CHECK m1 m3 m4" }
    measurements { input_port: 0  measurement_index: 1 }  # v₁ⁱⁿ (virtual)
    measurements { measurement_index: 1 }             # physical measurement 1
    measurements { measurement_index: 2 }             # physical measurement 2
  }
  # no errors — no noise
}
```

> **Reading protobuf text format:** In protobuf's text format, fields with default values
> (0 for integers, empty for strings) are omitted. For example,
> `measurements { }` means `measurement_index: 0` and no `input_port` — i.e., physical
> measurement 0. Similarly, `measurements { measurement_index: 1 }` means physical
> measurement 1. When you see `measurements { input_port: 0 }`, it means input port 0,
> stabilizer 0 (since `measurement_index` defaults to 0).

The transpiler **automatically discovered** the checks $c_0 = v_0^{\text{in}} \oplus m_0
\oplus m_1$ and $c_1 = v_1^{\text{in}} \oplus m_1 \oplus m_2$ — these are exactly the
stabilizer parity checks from the [deq-bin chapter](bin-basics.md). You didn't have
to write a single `DETECTOR` annotation.

---

## Adding Noise

To see how the transpiler analyzes error effects, add `X_ERROR` instructions to the
circuit. The transpiler propagates each error through the Clifford circuit offline to
determine which checks it triggers and which observables it flips — so the runtime never
needs to repeat this expensive analysis:

[Prepare and measure with noise](../examples/language/02_noisy.deq)
<!-- deq-highlight-begin: ../examples/language/02_noisy.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># Adding noise to see how error effects are analyzed offline by the transpiler</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/language/02_noisy.deq -->

The only change is adding `X_ERROR(0.01)` after the resets. The transpiler now analyzes
each error's effect — propagating it through the Clifford circuit to determine which
checks it triggers and which observables it flips. Here's what appears in the PrepareZ
output:

```protobuf
# PrepareZ errors — effects analyzed from X_ERROR(0.01) 0 1 2
errors {
  base { tag: "X_ERROR X0"  residual: 1  probability: 0.01 }
  unfinished_checks: 0
}
errors {
  base { tag: "X_ERROR X1"  residual: 1  probability: 0.01 }
  unfinished_checks: 0  unfinished_checks: 1
}
errors {
  base { tag: "X_ERROR X2"  residual: 1  probability: 0.01 }
  unfinished_checks: 1
}
```

The transpiler determined that:
- An X error on qubit 0 anti-commutes with stabilizer $Z_0 Z_1$ → triggers unfinished
  check 0 and flips the $Z_0Z_1Z_2$ logical observable (`residual: 1`)
- An X error on qubit 1 anti-commutes with **both** stabilizers → triggers unfinished
  checks 0 and 1, and flips the logical observable
- An X error on qubit 2 anti-commutes with stabilizer $Z_1 Z_2$ → triggers unfinished
  check 1 and flips the logical observable

And in MeasureZ:
```protobuf
errors {
  base { tag: "X_ERROR X0"  readout_flips: 0  probability: 0.01 }
  finished_checks: 0
}
errors {
  base { tag: "X_ERROR X1"  readout_flips: 0  probability: 0.01 }
  finished_checks: 0  finished_checks: 1
}
errors {
  base { tag: "X_ERROR X2"  readout_flips: 0  probability: 0.01 }
  finished_checks: 1
}
```

Notice the difference: PrepareZ errors have `residual` (they propagate through the Pauli
frame to downstream gadgets), while MeasureZ errors have `readout_flips` (they directly
affect the final measurement). This matches exactly the discussion of `residual` vs
`readout_flips` in the [deq-bin chapter](bin-basics.md).

All of this was analyzed **automatically** and **offline** from the Clifford circuit — you
only wrote the physical gates and noise instructions, and the transpiler determined every
error's effect on checks and observables so the runtime doesn't have to.

---

## Syndrome Extraction: The Idle Gadget

Now let's add a syndrome extraction round between preparation and measurement:

[Full example with Idle gadget](../examples/language/03_with_idle.deq)
<!-- deq-highlight-begin: ../examples/language/03_with_idle.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># Full example with syndrome extraction (Idle gadget)</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span><span style="color:#008000">  # data qubit error</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">1</span><span style="color:#098658"> 3</span><span style="color:#008000">  # measurement error</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/language/03_with_idle.deq -->

The Idle gadget is the most interesting:

[Idle gadget](../examples/language/snippet_idle.deq)
<!-- deq-highlight-begin: ../examples/language/snippet_idle.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span><span style="color:#008000">  # data qubit error</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">1</span><span style="color:#098658"> 3</span><span style="color:#008000">  # measurement error</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/language/snippet_idle.deq -->

Note the physical qubit layout: data qubits are at indices 0, 2, 4 and ancilla qubits at
1, 3. The `INPUT` and `OUTPUT` statements both specify `RepetitionCode 0 2 4` — the same
data qubits carry the code through the gadget. The ancillas (1, 3) are internal to the
gadget — they're reset, used for stabilizer measurement, and measured.

The transpiler output for Idle is:

```protobuf
gadget_types {
  base { gtype: 2, name: "Idle"
    measurements { tag: "M 1" }
    measurements { tag: "M 3" }
    inputs { ptype: 1 }
    outputs { ptype: 1 }
    # Pauli frame matrices: both logical observables pass through unchanged
    correction_propagation { rows: 4, cols: 5, i: [0, 1], j: [0, 1] }  # logical obs pass through
    readout_propagation { cols: 5 }                                      # 0×5 — no readouts
    logical_correction { rows: 4 }                                       # 4×0 — no readouts
    physical_correction { rows: 4, cols: 2, i: [2, 3], j: [0, 1] }      # syndrome → stabilizer frame
  }
  # Finished checks: time-like (compare with predecessor)
  finished_checks {
    base { tag: "CHECK m0 m2" }
    measurements { input_port: 0 }           # v₀ⁱⁿ
    measurements { }                         # physical measurement 0 (M 1)
  }
  finished_checks {
    base { tag: "CHECK m1 m3" }
    measurements { input_port: 0  measurement_index: 1 }  # v₁ⁱⁿ
    measurements { measurement_index: 1 }    # physical measurement 1 (M 3)
  }
  # Unfinished checks: record output stabilizer
  unfinished_checks {
    base { tag: "CHECK m2 m4" }
    measurements { }                         # physical measurement 0
  }
  unfinished_checks {
    base { tag: "CHECK m3 m5" }
    measurements { measurement_index: 1 }    # physical measurement 1
  }
  # Errors
  errors {
    # Data qubit errors: trigger finished checks only (+ residual)
    base { tag: "X_ERROR X0"  residual: 1  probability: 0.01 }
    finished_checks: 0
  }
  errors {
    base { tag: "X_ERROR X2"  residual: 1  probability: 0.01 }
    finished_checks: 0  finished_checks: 1
  }
  errors {
    base { tag: "X_ERROR X4"  residual: 1  probability: 0.01 }
    finished_checks: 1
  }
  errors {
    # Measurement errors: trigger both finished AND unfinished checks
    base { tag: "X_ERROR X1"  probability: 0.01 }
    finished_checks: 0  unfinished_checks: 0
  }
  errors {
    base { tag: "X_ERROR X3"  probability: 0.01 }
    finished_checks: 1  unfinished_checks: 1
  }
}
```

Note the `correction_propagation` matrix: it is a $4 \times 5$ matrix (4 output observables
$\times$ (4 input observables + 1 constant column)) with entries only at rows 0 and 1
(the logical observables). The stabilizer-tracking rows (2, 3) are zero in
`correction_propagation` — their updates come from `physical_correction` instead, which
maps syndrome measurements to the stabilizer frame columns. This means both logical
observable corrections pass through the Idle gadget unchanged — exactly what you'd expect
for a syndrome extraction round that doesn't apply any logical gate.

The transpiler automatically derived:

1. **2 finished checks** — time-like checks comparing the current syndrome measurement with
   the input virtual stabilizer: $s_0 \oplus v_0^{\text{in}}$ and $s_1 \oplus v_1^{\text{in}}$

2. **2 unfinished checks** — recording the output stabilizer equals the current measurement:
   $s_0 \oplus v_0^{\text{out}}$ and $s_1 \oplus v_1^{\text{out}}$

3. **5 errors** with their effects analyzed offline:
   - Data qubit errors trigger **only finished checks** (not unfinished — because both the
     physical measurement and the output virtual stabilizer flip, canceling in the
     unfinished check)
   - Measurement errors trigger **both finished and unfinished checks** (only the recorded
     measurement flips, not the actual stabilizer)

This matches the analysis in the [JIT chapter](jit-basics.md).

---

## Port Ordering Rule

deq enforces a strict ordering of `INPUT`, circuit instructions, and `OUTPUT` within a
gadget body:

```
GADGET Example {
    INPUT  ...     ← all INPUT ports first
    R 0            ← then all circuit instructions
    CX 0 1         ← (including REPEAT blocks and noise)
    M 1
    OUTPUT ...     ← all OUTPUT ports last
}
```

This ordering is **required** — the parser will reject any gadget that violates it.
Gate instructions, noise instructions (like `DEPOLARIZE1`, `X_ERROR`), and `REPEAT` blocks
must all appear between `INPUT` and `OUTPUT`.

**Why?** The transpiler assigns measurement indices in a flat global layout:
`[input-virtual | internal | output-virtual]`. Input virtual measurements (one per
stabilizer of each INPUT port) occupy the lowest indices, physical measurements come next,
and output virtual measurements occupy the highest indices. If an OUTPUT appeared before a
physical measurement, the indices would be interleaved and the transpiler would produce
incorrect check and error models.

There's also a semantic reason: once a qubit is declared in an `OUTPUT` port, it's part of
the output code state. Any gate applied to it afterward would be ambiguous — is it modifying
the output state, or is it an ancilla operation? By enforcing OUTPUT-last ordering, the
circuit's intent is always clear.

> **Note:** `CHECK`, `READOUT`, `ERROR`, `CONDITIONAL`, and `VIRTUAL` statements are
> declarations, not circuit operations. They can appear after `OUTPUT`.
> Use `ERROR` statements after `OUTPUT` to directly associate error probabilities
> with checks.

---

## IMPORT: Splitting Definitions Across Files

For larger projects, you can split `CODE` and `GADGET` definitions into a library file and
reference them with `IMPORT`. Such a library file is, in effect, a reusable [qodec](../README.md#the-quantum-codec-qodec) — a 
self-contained encode/decode description that programs can share.

[Import example](../examples/language/05_import.deq)
<!-- deq-highlight-begin: ../examples/language/05_import.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">IMPORT</span><span style="color:#A31515"> "05_library.deq"</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/language/05_import.deq -->

`IMPORT` brings all `CODE`, `GADGET`, and `COMPOSE` definitions from the referenced file
into scope. The imported file's path is relative to the importing file.

---

## Taking Control: Manual Checks

By default, the transpiler uses `@CHECKS("auto")` mode: it derives checks automatically
via paulimer simulation. If you also write `CHECK` statements in the gadget body, the
transpiler validates them against the auto-derived row space and emits them first, then
fills in any remaining independent checks. For full control, use `@CHECKS("manual")` —
here the transpiler uses **only** your annotations and generates no checks of its own:

[Manual Idle gadget](../examples/language/manual_idle.deq)
<!-- deq-highlight-begin: ../examples/language/manual_idle.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#008000">    # finished checks: compare current syndrome with input virtual stabilizer</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-4]</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> rec[-1]</span><span style="color:#001080"> rec[-3]</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: record output stabilizer = current measurement</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-4]</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> rec[-1]</span><span style="color:#001080"> rec[-3]</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/language/manual_idle.deq -->

**How `CHECK` works with virtual measurements:**

Each `INPUT` and `OUTPUT` statement introduces virtual stabilizer measurements (one per
stabilizer in the code). These virtual measurements can be referenced with `rec[-k]` just
like physical measurements — they are simply appended to the measurement record at the
point where `INPUT` or `OUTPUT` appears.

Checks must be declared after all of its measurements, because they refer to measurements using relative indices like `rec[-i]`.
- Each unfinished check must involve **exactly one** output virtual measurement
- In manual mode, every output virtual measurement must appear in **exactly one** unfinished
  check (see the [JIT chapter](jit-basics.md) for why)

In this example, the first `rec[-2]` and `rec[-4]` reference the physical measurement `M 1` and the
input virtual stabilizer for $Z_0 Z_1$ respectively.

The transpiler produces **identical output** for manual and auto mode in this case — which
is a good verification that the manual checks are correct.

### Naturally Flipped Checks (`FLIP`)

Some checks have an expected parity of 1 rather than 0 in the noiseless case — these are
called **naturally flipped** checks. This happens when:

- A measurement uses the inverted syntax `M !q` (the result is negated)
- The circuit's stabilizer algebra produces an odd-parity correlation by construction

[Naturally flipped check example](../examples/language/flip_check.deq)
<!-- deq-highlight-begin: ../examples/language/flip_check.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> Trivial</span><span style="color:#000000"> [[</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#0000FF"> Z0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareOne</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    X</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000"> !</span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> rec[-1]</span><span style="color:#0000FF"> FLIP</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Trivial</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/language/flip_check.deq -->

The `M !0` instruction negates the measurement result, so its noiseless value is 1. The
`CHECK rec[-1] FLIP` declares that this measurement is expected to be 1, not 0. Without
`FLIP`, the transpiler would reject this manual check.

### When to Use Each Mode

| Mode                     | Behavior                                                                                |
| ------------------------ | --------------------------------------------------------------------------------------- |
| `@CHECKS("auto")`        | **Default.** Auto-derives checks; your `CHECK` annotations are validated and kept first |
| `@CHECKS("manual")`      | Only your `CHECK` annotations are used; the transpiler adds nothing                     |
| `@CHECKS("syndrome")`    | Per-stabilizer minimal-weight checks; excludes metachecks                               |
| `@CHECKS("transversal")` | Optimized for transversal gates; accepts `max_weight` parameter                         |

Use cases:
- **Non-Clifford gadgets:** The automatic check and error derivation only works for
  Clifford circuits. Gadgets containing T gates (e.g., magic state distillation) require
  the user to manually decide how to decode them by declaring `CHECK` and `ERROR` directly
  with `@CHECKS("manual")`
- **Syndrome extraction:** Use `@CHECKS("syndrome")` for clean weight-2 time-like checks
  without metachecks (see [codes with redundant stabilizers](codes-redundant-stabilizers.md)
  for what metachecks are and when to exclude them)
- **Transversal and automorphism gates:** Use `@CHECKS("transversal", max_weight=3)` for
  optimal low-weight checks on transversal gates and automorphism gates that don't involve
  syndrome extraction
- **Verification:** Verify if your checks are correct and complete: write `CHECK` 
statements in the default mode (equivalent to adding `@CHECK("auto")` at the top), and 
then run `deq annotate` to see if it reports any error or add new checks — if not, then 
your checks are valid and complete.

---

## Running a Simulation

The full pipeline from `.deq` to logical error rates:

```sh
# Step 1: Transpile to JIT format
deq transpile 03_with_idle.deq \
    --out 03_with_idle.deq.jit \
    --program Simulation

# Step 2: Run the decoder server with simulation
deq server \
    --decoder black-box-relay-bp \
    --coordinator window \
    --coordinator-config '{"buffer_radius":3}' \
    --controller jit \
    --controller-config '{"filepath":"03_with_idle.deq.jit"}' \
    --simulator jit-static \
    --simulator-config '{
        "filepath":"03_with_idle.stim",
        "jit_library_filepath":"03_with_idle.deq.jit",
        "shots": 100000,
        "seed": 123
    }'
```

The transpiler produces two files:
- `.deq.jit` — the JIT library (loaded by the JIT compiler at runtime)
- `.stim` — the physical circuit (used by the simulator to sample errors)

The server loads both, runs the simulation, and reports logical error rates.

---

## Reading the .deq.jit.txt Output

The `.deq.jit.txt` file is a human-readable dump of the `JitLibrary` protobuf. It has
three sections that map directly to the concepts from the
[JIT chapter](jit-basics.md):

| Section        | JIT Concept             | Content                                                                                                                                                               |
| -------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `port_types`   | `JitPortType`           | Code definitions with stabilizer lists. The `base` carries the `PortType` (observables), and `stabilizers` lists the virtual stabilizer measurements                  |
| `gadget_types` | `JitGadgetType`         | Offline analysis results for each gadget. Contains `base` (the `GadgetType` with measurements, ports, matrices), `finished_checks`, `unfinished_checks`, and `errors` |
| `program`      | `JitInstruction` stream | The sequence of gadget instantiations with port connectivity (`connectors`). Each entry specifies `gtype` and `gid`                                                   |

The relationship between the `.deq` source and the `.deq.jit.txt` output:

| `.deq` source                                             | `.deq.jit.txt` output                                                                                    |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `CODE RepetitionCode [[3,1,1]] { STABILIZER Z0*Z1 Z1*Z2 }` | `port_types { stabilizers { tag: "Z0*Z1" } stabilizers { tag: "Z1*Z2" } }`                                |
| `GADGET Idle { ... M 1 3 ... }`                            | `gadget_types { base { measurements { tag: "M 1" } measurements { tag: "M 3" } } ... }`                   |
| `INPUT RepetitionCode 0 2 4`                               | `inputs { ptype: 1 }` in the gadget's base                                                                |
| `OUTPUT RepetitionCode 0 2 4`                              | `outputs { ptype: 1 }` in the gadget's base                                                               |
| `READOUT rec[-3] rec[-2] rec[-1]`                          | `readouts { measurement_indices: 0 measurement_indices: 1 measurement_indices: 2 }`                       |
| `X_ERROR(0.01) 0 2 4`                                      | `errors { base { tag: "X_ERROR X0" probability: 0.01 } ... }` (one per error source)                      |
| `PROGRAM Simulation { PrepareZ 0  Idle 0  MeasureZ 0 }`    | `program { gadget { gtype: 1 gid: 1 } } program { gadget { gtype: 2 connectors { gid: 1 } gid: 2 } } ...` |

---

## Summary

The deq language provides three definition types covered in this chapter:

1. **`CODE`** — Pure mathematical specification of a QEC code (stabilizers, logical
   operators). Becomes a `JitPortType` with virtual stabilizer measurements.
2. **`GADGET`** — Physical circuit using Stim-compatible syntax, with `INPUT`/`OUTPUT` port
   bindings. The transpiler automatically derives checks and errors from Clifford circuit
   analysis. Use `@CHECKS("manual")` for full control.
3. **`PROGRAM`** — Logical circuit at the gadget level, with `ASSERT_EQ` for error
   criteria. Becomes a `JitInstruction` stream.

Taken together, a file's `CODE` and `GADGET` definitions form a
[qodec](../README.md#the-quantum-codec-qodec) — the encode/decode description
introduced in the tutorial overview. A `PROGRAM` is then a logical circuit
expressed in terms of that qodec.

The transpiler (`transpile`) converts `.deq` → `.deq.jit` (the JIT intermediate
data), performing all the expensive stabilizer simulation and error propagation offline. At
runtime, the JIT compiler uses this data to dynamically construct the deq-bin decoding
hypergraph.

For hierarchical composition of gadgets (`COMPOSE`) and the `REPEAT` construct — which
enable well-structured decoding hypergraphs for multi-round circuits without global
optimization — see the [COMPOSE chapter](compose-gadgets.md).