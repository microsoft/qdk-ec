# Debugging Your .deq Program

Sometimes we need to debug and improve our `.deq` program — see if the checks are as
expected, verify how errors trigger those checks, or trace why a decoder is misbehaving.

There are four escalating levels of debugging. Higher levels are more intuitive and relate
directly to the source code, but they may not surface all properties. Lower levels are more
powerful but require deeper understanding of how the decoding pipeline works.

We'll use `03_with_idle.deq` from the
[language chapter](language-basics.md) as our running example throughout.

---

## Level 1: Source-Level Annotation with `annotate`

The `annotate` tool rewrites your `.deq` file into an annotated form that mirrors the
compiled JIT structure. It shows you **exactly what the transpiler derived** from your
circuit — checks, propagated errors, measurement counts — all at the source level.

```sh
deq annotate 03_with_idle.deq
```

Output:

[Annotated 03_with_idle.deq](../examples/debug/03_with_idle.annotated.deq)
<!-- deq-highlight-begin: ../examples/debug/03_with_idle.annotated.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@PTYPE</span><span style="color:#000000">(</span><span style="color:#098658">1</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">3</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#008000">  # generator S0, destabilizer DS0=X1*X2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#008000">  # generator S1, destabilizer DS1=X0*X1</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">1</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 1:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 3</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:2, 2:1 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">2</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 2 4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#008000">    # M(0.01) 1 3</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C3</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span><span style="color:#267F99"> IN0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#267F99"> IN0.S1</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S0</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S1</span><span style="color:#001080"> M1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:2 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 5</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:2, 2:3 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">3</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # M(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#001080"> R0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#001080"> R0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#001080"> R0</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-1]</span><span style="color:#008000">  # flipped by: IN0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#001080"> M0</span><span style="color:#267F99"> IN0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M2</span><span style="color:#001080"> M1</span><span style="color:#267F99"> IN0.S1</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 3:2 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # errors: 3</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:2, 2:1 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/debug/03_with_idle.annotated.deq -->

### What to look for

**Noise instructions are commented out** — because the annotator expands each noise
channel into concrete `ERROR(p) C... LX...` statements with explicit check triggers
and logical residuals. Keeping the original noise instructions would cause duplicate
errors during re-transpilation. The annotated file is guaranteed to produce exactly the
same `.deq.jit` as the original (up to tag differences).

**Auto-derived checks** — the transpiler-derived `CHECK` statements. In the Idle gadget:
- `CHECK M0 IN0.S0` — finished check (before `OUTPUT`): physical measurement 0
  (`M0`) XOR'd with input virtual stabilizer 0 (`IN0.S0`)
- `CHECK OUT0.S0 M0` — unfinished check (after `OUTPUT`): output virtual stabilizer 0
  (`OUT0.S0`) XOR'd with physical measurement 0 (`M0`)

**Propagated errors** — `ERROR(0.01) C0 OUT0.LX0` means this error triggers check 0 and has
an X-type logical residual on output-port-0's logical qubit 0. `ERROR(0.01) C0 C2` means
it triggers checks 0 and 2 (a measurement error that affects both the finished and
unfinished check).

**All gadgets are forced to `@CHECKS("manual", verify=0)`** — the annotated output is a
valid `.deq` file that you can feed back to the transpiler. This is useful for incremental
debugging: modify the annotated checks and re-transpile.

---

## Level 2: Inspecting the `.deq.jit.txt` Output

Although the annotate tool presents the same information as a `.deq` source file, you may
want to inspect the **raw protobuf data** — especially when debugging type-level issues
(port type mismatches, incorrect matrix dimensions, missing stabilizers).

The `.deq.jit.txt` file is auto-generated alongside the `.deq.jit` binary:

```sh
deq transpile 03_with_idle.deq \
    --out 03_with_idle.deq.jit --program Simulation
cat 03_with_idle.deq.jit.txt
```

The [deq language chapter](language-basics.md) already covers how to read this output
in detail (port types, gadget types, finished/unfinished checks, errors). Here we highlight
the debugging-specific patterns:

### Verifying port type stabilizers

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

### Verifying check measurement references

Each check lists its participating measurements. Verify that `input_port` references point
to the correct input port index and that `measurement_index` values map to the correct
physical measurements:

```protobuf
finished_checks {
  base { tag: "CHECK m0 m2" }
  measurements { input_port: 0 }           # v₀ⁱⁿ: input port 0, stabilizer 0
  measurements { }                         # physical measurement 0 (M 1)
}
```

### Verifying error triggering

Each error lists which finished and unfinished checks it triggers. Data qubit errors should
trigger **only finished checks** (plus residual), while measurement errors should trigger
**both finished and unfinished checks**:

```protobuf
errors {
  base { tag: "X_ERROR X0"  residual: 1  probability: 0.01 }
  finished_checks: 0          # data error: only finished
}
errors {
  base { tag: "X_ERROR X1"  probability: 0.01 }
  finished_checks: 0  unfinished_checks: 0  # measurement error: both
}
```

If an error is triggering unexpected checks, the issue is likely in the circuit — check
the CNOT ordering and ancilla connectivity.

---

## Level 3: Inspecting the Concrete Decoding Hypergraph

If the problem is not visible at the JIT type level, you may need to inspect the **concrete
decoding hypergraph** after JIT compilation — the actual `.deq.bin` that the decoder
receives. This is useful when you suspect a bug in the JIT compiler itself, or when you
need to verify that remote measurement references resolved correctly.

### Step 1: Compile to `.deq.bin`

```sh
deq compile 03_with_idle.deq.jit \
    --out 03_with_idle.deq.bin
```

This produces `03_with_idle.deq.bin` (binary) and `03_with_idle.deq.bin.txt` (human-readable
protobuf text). The program is already embedded in the `.deq.jit` file (produced by
`transpile --program Simulation`). The text file shows the full Library with gadget
types, check model types, error model types, and the instruction stream.

### Step 2: Canonicalize

The raw `.deq.bin` uses **remote references** — each check model reaches into predecessor
gadgets to fetch measurements. This is efficient for the decoder but hard to read. The
`canonicalize` tool flattens everything into a single global namespace:

```sh
deq canonicalize 03_with_idle.deq.bin
```

Output:
```
Canonicalized: 03_with_idle.canonical.deq.bin
  Measurements: 5
  Checks: 4
  Errors: 11
  Readouts: 1
```

The canonical form resolves all remote references and produces a library with:
- **1 gadget type** — all 5 physical measurements concatenated in program order
- **1 check model** — all 4 checks with absolute global measurement indices
- **1 error model** — all 11 errors with absolute global check indices

### Step 3: Read the canonical `.deq.bin.txt`

```protobuf
gadget_types {
  gtype: 1
  measurements { }       # [0] Idle: M 1   (ancilla for Z0*Z1)
  measurements { }       # [1] Idle: M 3   (ancilla for Z1*Z2)
  measurements { }       # [2] MeasureZ: M 0
  measurements { }       # [3] MeasureZ: M 1
  measurements { }       # [4] MeasureZ: M 2
  readouts {
    tag: "READOUT m0 m1 m2"
    measurement_indices: 2  measurement_indices: 3  measurement_indices: 4
  }
}
```

The 5 global measurements are assigned in program order: Idle's 2 measurements first
(indices 0–1), then MeasureZ's 3 measurements (indices 2–4). PrepareZ has no physical
measurements.

```protobuf
check_model_types {
  ctype: 1
  checks {
    measurements { }                          # [0] = m0 only
  }
  checks {
    measurements { measurement_index: 1 }     # [1] = m1 only
  }
  checks {
    measurements { }                          # [2] = m0 ⊕ m2 ⊕ m3
    measurements { measurement_index: 2 }
    measurements { measurement_index: 3 }
  }
  checks {
    measurements { measurement_index: 1 }     # [3] = m1 ⊕ m3 ⊕ m4
    measurements { measurement_index: 3 }
    measurements { measurement_index: 4 }
  }
}
```

The 4 checks in program order:

| Check | Formula                     | Origin                                                                                                                                                                  |
| ----- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| c0    | $m_0$                       | PrepareZ unfinished → Idle finished: $v_0^{\text{in}} \oplus s_0$ simplified because $v_0^{\text{in}}$ was resolved to the PrepareZ unfinished check which equals $s_0$ |
| c1    | $m_1$                       | PrepareZ unfinished → Idle finished: same for stabilizer 1                                                                                                              |
| c2    | $m_0 \oplus m_2 \oplus m_3$ | Idle unfinished → MeasureZ finished: $v_0^{\text{out}} \oplus m_0^{\text{MZ}} \oplus m_1^{\text{MZ}}$                                                                   |
| c3    | $m_1 \oplus m_3 \oplus m_4$ | Idle unfinished → MeasureZ finished: $v_1^{\text{out}} \oplus m_1^{\text{MZ}} \oplus m_2^{\text{MZ}}$                                                                   |

> **Why are checks c0 and c1 so simple?** The JIT compiler resolves the XOR cancellation
> chain described in the [JIT chapter](jit-basics.md): PrepareZ's unfinished check is
> just the virtual stabilizer $v_0^{\text{out}}$. When Idle's finished check XORs
> $v_0^{\text{in}} \oplus s_0$, and $v_0^{\text{in}}$ is resolved to PrepareZ's
> $v_0^{\text{out}}$, the chain simplifies. The net effect is that c0 and c1 just test
> whether the Idle syndrome measurements are 0 — which is correct for the first round after
> preparation, where the initial stabilizer eigenvalues are deterministic.

---

## Level 4: Interpreting Syndromes from Concrete Measurements

Without running the decoder, you can verify that the syndrome calculation and logical
readouts are correct by providing concrete measurement values to the `interpret` tool.

### Sampling measurements from a Stim circuit

The `transpile` command produces a `.stim` file alongside the `.deq.jit`. You can
sample measurement outcomes from this circuit using `sample`:

```sh
deq sample 03_with_idle.stim --shots 10 --seed 1
```

```
0x00
0x00
0x00
0x00
0x48
0xd0
0x00
0x00
0x00
0x48
```

Each line is a hex string encoding one shot's measurement outcomes in BitVector convention
(MSB-first: bit 0 = `0x80`, bit 1 = `0x40`, bit 2 = `0x20`, etc.). Most shots are `0x00`
(no errors), but shots 4 and 9 show `0x48` (bits 1 and 4 flipped) and shot 5 shows `0xd0`
(bits 0, 1, and 2 flipped — a multi-error event). You can copy any of these hex strings
directly into `interpret`.

### Interpreting with `interpret`

```sh
deq interpret 03_with_idle.deq.bin --measurements <hex>
```

The tool accepts any `.deq.bin` file (canonical or not) — it internally canonicalizes to
resolve remote references, then groups the output by original gadget.

### No-error case: all measurements zero

```sh
deq interpret 03_with_idle.deq.bin --measurements 0x00
```

```
Measurements (5 total, hex=0x00):
  PrepareZ: (none)
  Idle: 0b00
  MeasureZ: 0b000

Checks (syndrome hex=0x00):
  Idle: 0b00
  MeasureZ: 0b00

Readouts:
  MeasureZ: READOUT m0 m1 m2 = 0
```

All syndromes are 0 and the logical readout is 0 — correct for a noiseless
PrepareZ → Idle → MeasureZ circuit. Note how measurements and checks are grouped by their
originating gadget.

### Ancilla measurement error: bit 0 flipped

Flip Idle's first ancilla measurement ($m_0$, the $Z_0 Z_1$ syndrome). Bit 0 = `0x80`:

```sh
deq interpret 03_with_idle.deq.bin --measurements 0x80
```

```
Measurements (5 total, hex=0x80):
  PrepareZ: (none)
  Idle: 0b10
  MeasureZ: 0b000

Checks (syndrome hex=0xa0):
  Idle: 0b10
  MeasureZ: 0b10

Readouts:
  MeasureZ: READOUT m0 m1 m2 = 0
```

Idle's first check fires (`0b10`) and MeasureZ's first check fires (`0b10`) — syndrome = `0xa0`.
This is consistent with a measurement error on the $Z_0 Z_1$ ancilla: it triggers both the
Idle check (comparing with preparation) and the MeasureZ check (comparing Idle's syndrome
with the final data measurements). The logical readout remains 0 because measurement
errors don't flip data qubits.

Use `--verbose` to see the per-check measurement breakdown:

```sh
deq interpret 03_with_idle.deq.bin --measurements 0x80 --verbose
```

### Data qubit error during measurement: bit 2 flipped

Flip MeasureZ's first data measurement ($m_2$, qubit 0). Bit 2 = `0x20`:

```sh
deq interpret 03_with_idle.deq.bin --measurements 0x20
```

```
Measurements (5 total, hex=0x20):
  PrepareZ: (none)
  Idle: 0b00
  MeasureZ: 0b100

Checks (syndrome hex=0x20):
  Idle: 0b00
  MeasureZ: 0b10

Readouts:
  MeasureZ: READOUT m0 m1 m2 = 1
```

Only MeasureZ's first check fires (syndrome = `0x20`), and the logical readout flips to 1.
This is consistent with an X error on data qubit 0 right before measurement — it
anti-commutes with stabilizer $Z_0 Z_1$ (triggering the MeasureZ check) and flips the
$Z_0 Z_1 Z_2$ logical observable.

### Debugging workflow

1. **Start with all-zero measurements** — verify all syndromes are 0 and readouts match
   the expected noiseless values
2. **Flip individual measurement bits** — verify each flip triggers the expected checks
   (cross-reference with the error table from Level 2)
3. **Combine multiple flips** — verify that combined errors produce the expected XOR of
   individual syndromes

### Convenience shortcut: `sample --noiseless --interpret`

The `sample` + `interpret` workflow above is useful for understanding each step,
but for quick correctness checks you can use `sample --noiseless --interpret` which
combines the entire pipeline into a single invocation.

Given a `.deq` source file and a `PROGRAM` name, it compiles through JIT, strips noise,
samples, and evaluates, all in a temporary directory so no intermediate files are left
behind:

```sh
deq sample circuit.deq --program MyProg --noiseless --interpret --shots 5 --seed 42
```

Each line shows the syndrome and readout bit strings for one shot. Use `--verbose` for
the full per-gadget breakdown (same detail level as `interpret --verbose`).

If you already have a `.deq.jit` file (e.g. from a previous `transpile` run), pass
`--jit` to skip the expensive gadget-type construction and only recompile the PROGRAM
block — this is much faster when iterating on program structure:

```sh
deq sample circuit.deq --program MyProg --jit lib.deq.jit --noiseless --interpret --shots 5
```

For `.deq` files that use Mako templates (e.g. `${d}`, `% for i in range(n):`), pass
`--mako` with semicolon-separated variable definitions:

```sh
deq sample template.deq --program Main --noiseless --interpret --mako d=3 --mako p=0.01 --shots 5
```

This is the fastest way to sanity-check a new program. It skips decoding entirely and
evaluates check values and logical readouts assuming ideal measurements.

---

## Summary

| Level | Tool                             | What it shows                                                             | When to use                                                      |
| ----- | -------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 1     | `annotate`                       | Source-level: commented circuit + auto-derived checks + propagated errors | First pass — verify checks match expectations                    |
| 2     | `.deq.jit.txt`                  | Protobuf types: port types, gadget types, measurement references          | Type-level issues — wrong stabilizers, incorrect port bindings   |
| 3     | `canonicalize`                   | Canonical decoding hypergraph: all checks/errors with global indices      | JIT compiler issues — verify XOR cancellation, remote resolution |
| 4     | `sample`                         | Sampled measurement outcomes from a Stim circuit as hex strings           | Generate realistic test inputs for `interpret`                   |
| 4     | `interpret`                      | Concrete syndrome/readout values from measurement bits                    | Correctness verification — trace specific error scenarios        |
| 4     | `sample --noiseless --interpret` | End-to-end: compile + strip noise + sample + interpret from `.deq`       | Fastest sanity check — no intermediate files                     |
