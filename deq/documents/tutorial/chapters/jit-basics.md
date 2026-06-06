# deq-JIT Basics

## Why a JIT Compiler?

The decoding hypergraph depends on circuit context: the same gadget type needs different
check model types and error model types depending on what gadgets surround it. For example,
a transversal CNOT gate alters the decoding hypergraph structure without introducing new
checks itself — but the checks on the surrounding gadgets change depending on whether a
CNOT (or other logical unitaries) appears before or after them. Enumerating all possible
contexts offline would be combinatorially expensive.
(The [deq-bin basics](bin-basics.md) chapter covers these types in detail.)

The **deq-JIT compiler** solves this problem. It accepts a stream of logical operations
(gadget instantiations with port connectivity) and **dynamically generates** the appropriate
deq-bin `CheckModelType` and `ErrorModelType` definitions at runtime, based on the actual
circuit context.

This is essential for two reasons:

1. **Dynamic circuits.** Many essential QEC operations produce dynamic circuits. For
   example, magic state injection conditions on a mid-circuit measurement outcome to decide
   whether a Clifford gate is applied — the circuit topology depends on a runtime
   measurement result. The full decoding hypergraph cannot be compiled offline because the
   full circuit doesn't exist yet.

2. **Context-dependent types.** Even for static circuits, the number of distinct
   check/error model types grows with the variety of gadget neighborhoods. A JIT compiler
   generates exactly the types needed for the actual circuit, avoiding combinatorial
   explosion of pre-compiled type definitions.

To keep the online compilation as fast as possible, we pre-process as much information as
possible offline — especially the expensive stabilizer simulation and error propagation
through Clifford circuits. The workflow has two stages:

- **Offline: JIT transpiler.** Each gadget type is analyzed independently — stabilizer
  propagation, check construction, error analysis — and the results are serialized into an
  intermediate binary file (`.deq.jit`). This file contains all `JitGadgetType` and
  `JitPortType` definitions: a compact summary of each gadget's finished checks, unfinished
  checks, errors, and virtual stabilizer measurements. This analysis is done once per gadget
  type, regardless of how many circuits use it.
- **Online: JIT compiler.** At runtime, the JIT compiler loads the `.deq.jit` file and
  accepts a stream of `JitInstruction` messages (each instantiating a gadget with port
  connectivity). It dynamically generates the deq-bin types and instructions (`.deq.bin`)
  based on the actual circuit context, and feeds them directly to a deq-bin-compatible
  decoder. The online phase performs only lightweight operations — XOR cancellation and index
  bookkeeping — because all the heavy analysis was done offline.

The key innovation that makes independent offline gadget analysis possible is **virtual
stabilizer measurements**.

---

## The Problem: Cross-Gadget Dependencies

Recall from the deq-bin chapter that checks often span gadget boundaries. For example, in
the repetition code, the check $c = s^2 \oplus s^1$ (the difference between consecutive
syndrome extraction rounds) references measurements from two different gadgets. In the
deq-bin format, this is expressed via remote references — the check model of the second
Idle gadget uses a `RemoteGadget` to reach the first Idle's measurements.

But when we analyze a gadget type **offline** — before knowing what circuit it will appear
in — we don't know which gadgets will connect to its ports. We don't know their measurement
counts, their IDs, or even their types. How can we describe a check that references
measurements from an unknown neighbor?

More critically, **errors in one gadget can trigger checks in other gadgets**. An error in
an Idle gadget might flip a syndrome measurement that participates in a check belonging to
the *next* gadget in the circuit — a gadget that hasn't been instantiated yet. The error
model can't be finalized until those future gadgets exist.

These cross-gadget dependencies seem to require global analysis of the full circuit. Virtual
stabilizer measurements eliminate this requirement.

---

## Virtual Stabilizer Measurements

The core idea is simple: at each port boundary, we introduce **virtual measurements** —
synthetic measurements that represent the stabilizer measurements of the quantum code at that
boundary. These virtual measurements are not physically performed; they are placeholders
that enable each gadget to be analyzed as if it had access to its neighbors' stabilizer
outcomes.

### JitPortType

Each port type in the JIT format extends the base deq-bin `PortType` with stabilizer
information:

```protobuf
message JitPortType {
  deq.bin.PortType base = 1;

  message Stabilizer {
    string tag = 1;       // e.g., "Z0Z1" or "Z1Z2"
  }
  repeated Stabilizer stabilizers = 3;
}
```

| Field         | Purpose                                                                                            |
| ------------- | -------------------------------------------------------------------------------------------------- |
| `base`        | The underlying deq-bin `PortType` (carries logical observable definitions and `ptype`)            |
| `stabilizers` | One entry per stabilizer generator of the quantum code at this port. Each has a `tag` for labeling |

**Example — RepetitionCode port with 2 stabilizers:**
```protobuf
JitPortType {
  base: { ptype: 1, name: "RepetitionCode", observables: [{ tag: "Z" }] }
  stabilizers: [{ tag: "Z0Z1" }, { tag: "Z1Z2" }]
}
```

The number of virtual stabilizer measurements per port equals the number of stabilizers
listed. For a $[[n, k]]$ code, this is $n - k$ (or more, if redundant stabilizers are included — see
[codes with redundant stabilizers](codes-redundant-stabilizers.md)).

### How Virtual Measurements Enable Independent Analysis

With virtual measurements, a gadget's checks can be written entirely in terms of:
- Its own **physical measurements** (the real measurements it performs)
- **Input virtual measurements** (stabilizer measurements arriving from predecessor gadgets)
- **Output virtual measurements** (stabilizer measurements leaving to successor gadgets)

The gadget doesn't need to know what specific gadgets connect at its ports — it only needs
to know the stabilizer structure of the code at each port.

**Example — Idle gadget for the repetition code:**

The Idle gadget performs 2 physical measurements ($s_0, s_1$) — the syndrome extraction for
$Z_0 Z_1$ and $Z_1 Z_2$. Its input port carries 2 virtual measurements ($v_0^{\text{in}},
v_1^{\text{in}}$) and its output port carries 2 virtual measurements ($v_0^{\text{out}},
v_1^{\text{out}}$). The gadget defines:

- **A finished check:** $s_0 \oplus v_0^{\text{in}}$ — compares the current syndrome
  measurement with the incoming stabilizer measurement. This is the time-like check.
- **An unfinished check:** $s_0 \oplus v_0^{\text{out}}$ — records that the outgoing
  stabilizer measurement equals the current measurement. This will be consumed by the next
  gadget.

(And similarly for the second stabilizer.)

When two gadgets connect at runtime, the output virtual measurement of the upstream gadget
and the input virtual measurement of the downstream gadget represent the **same** stabilizer
eigenvalue at the boundary. The JIT compiler **cancels** them via XOR (symmetric
difference), replacing virtual measurements with concrete physical measurement references.

---

## Finished vs. Unfinished Checks

Within a `JitGadgetType`, checks are classified into two categories:

### Finished Checks

A **finished check** involves only:
- Physical measurements of the current gadget
- Input virtual measurements (from predecessor gadgets via input ports)

Finished checks can be **resolved immediately** when the gadget is instantiated, because all
referenced measurements are already known (the input virtual measurements are resolved by
looking up the predecessor's cached output checks).

Each finished check produces exactly one **resolved check** in the deq-bin output — a check
involving only concrete physical measurements.

### Unfinished Checks

An **unfinished check** involves:
- Physical measurements of the current gadget
- Input virtual measurements
- Exactly **one** output virtual measurement

Unfinished checks **cannot be resolved independently**. They represent the stabilizer
information flowing out of the gadget through an output port. They are consumed by
downstream gadgets' finished checks via XOR cancellation.

The number of unfinished checks equals the total number of output stabilizers across all
output ports.

### The Unique-Output-Check Constraint

Each unfinished check depends on **exactly one** output virtual stabilizer, and each output
virtual stabilizer appears in **exactly one** unfinished check. This one-to-one
correspondence is a deliberate design choice for runtime efficiency.

**Why this constraint?** Without it, canceling virtual measurements at gadget connection
time would require solving a system of $\mathrm{GF}(2)$ equations to express virtual
measurements in terms of concrete measurements. With the constraint, cancellation is a
simple XOR substitution: whenever a downstream check references an input virtual stabilizer,
the compiler looks up the unique upstream unfinished check for that stabilizer and XORs the
two measurement sets. No equation solving required.

**Is the constraint always achievable?** Yes. By definition, a well-defined logical gadget
maps input code states to output code states: each output stabilizer's eigenvalue is
deterministically determined by the gadget's physical measurements and input stabilizer
eigenvalues. Therefore, for each output virtual stabilizer, there exists a subset of
physical and input virtual measurements whose XOR with that output virtual measurement is a
constant — this is precisely an unfinished check with exactly one output virtual stabilizer.
Any set of checks not already in this form can be brought into it by $\mathrm{GF}(2)$ row
reduction on the output virtual stabilizer columns.

### Examples

**Syndrome extraction round** (the Idle gadget):
- Has $s$ physical stabilizer measurements → produces $s$ finished checks + $s$ unfinished
  checks
- Each finished check is a weight-2 time-like check: $s_i \oplus v_i^{\text{in}}$
- Each unfinished check is weight-2: $s_i \oplus v_i^{\text{out}}$ (records that the output
  stabilizer equals the current measurement)

**Transversal gate** (e.g., transversal CNOT):
- Has **zero** physical measurements → **zero** finished checks → contributes **zero
  resolved checks** to the decoding hypergraph
- But has nontrivial unfinished checks that **reshape** the hypergraph structure. For a
  transversal CNOT, one type of Pauli error propagates from control to target through each
  physical CNOT gate, causing an input virtual stabilizer on one logical qubit to map to a
  product of output virtual stabilizers on *both* logical qubits. This produces **weight-3
  unfinished checks** — each involving two input virtual measurements (one per logical qubit)
  and one output virtual measurement. When downstream syndrome extraction rounds consume
  these unfinished checks, the resulting resolved checks span both logical qubits,
  introducing cross-logical-qubit connections in the decoding hypergraph.

---

## The JIT Format

### JitGadgetType: The Offline Analysis Result

A `JitGadgetType` is the product of offline analysis. It extends a base `GadgetType` with
the information the JIT compiler needs to dynamically construct check models and error
models at runtime:

```protobuf
message JitGadgetType {
  deq.bin.GadgetType base = 1;

  message PresentMeasurement {
    optional uint64 input_port = 1;
    uint64 measurement_index = 2;
  }

  message Check {
    deq.bin.CheckModelType.Check base = 1;
    repeated PresentMeasurement measurements = 2;
  }

  repeated Check finished_checks = 2;
  repeated Check unfinished_checks = 3;

  message Error {
    deq.bin.ErrorModelType.Error base = 1;
    repeated uint64 finished_checks = 2;
    repeated uint64 unfinished_checks = 3;
  }
  repeated Error errors = 4;
}
```

Let's walk through each component.

#### PresentMeasurement

A `PresentMeasurement` identifies a single measurement — either a physical measurement of
the current gadget or a virtual measurement from an input port:

| Field               | Purpose                                                                                                                                                                                       |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_port`        | If set, this measurement is a **virtual** measurement from the specified input port. If absent, it's a **physical** measurement of the current gadget                                         |
| `measurement_index` | When `input_port` is absent: index into the gadget's physical measurement list. When `input_port` is set: index into the port's stabilizer list (the virtual measurement for that stabilizer) |

**Examples:**
```protobuf
PresentMeasurement { measurement_index: 0 }               // physical measurement 0 (s₀)
PresentMeasurement { measurement_index: 1 }               // physical measurement 1 (s₁)
PresentMeasurement { input_port: 0, measurement_index: 0 } // input port 0, stabilizer 0 (v₀ⁱⁿ)
PresentMeasurement { input_port: 0, measurement_index: 1 } // input port 0, stabilizer 1 (v₁ⁱⁿ)
```

#### Check

A `Check` in the JIT format pairs a deq-bin `Check` base (for metadata like `tag` and
`naturally_flipped`) with a list of `PresentMeasurement` entries describing the measurements
to XOR:

| Field          | Purpose                                                                     |
| -------------- | --------------------------------------------------------------------------- |
| `base`         | The deq-bin `CheckModelType.Check` — carries `tag` and `naturally_flipped` |
| `measurements` | The list of measurements (physical and/or virtual) that this check XORs     |

#### finished_checks and unfinished_checks

| Field               | Purpose                                                                                                                                                                                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `finished_checks`   | Checks involving only physical measurements and input virtual measurements. Resolved immediately at instantiation                                                                                                                                                          |
| `unfinished_checks` | Checks involving physical + input virtual + exactly one output virtual measurement. The output virtual measurement is **implicit** — identified by the check's index position in the array (index $i$ corresponds to the $i$-th output stabilizer). This avoids redundancy |

**Example — Idle gadget (repetition code, 2 stabilizers):**

```protobuf
JitGadgetType {
  base: {
    gtype: 2, name: "Idle"
    measurements: [{ tag: "s0" }, { tag: "s1" }]
    inputs: [{ ptype: 1 }]
    outputs: [{ ptype: 1 }]
    correction_propagation: { rows: 1, cols: 2, i: [0], j: [0] }
    readout_propagation: { rows: 0, cols: 2 }
    logical_correction: { rows: 1, cols: 0 }
    physical_correction: { rows: 1, cols: 2 }
  }

  // Finished checks: s₀ ⊕ v₀ⁱⁿ and s₁ ⊕ v₁ⁱⁿ
  finished_checks: [
    {
      base: { tag: "time_like_0" }
      measurements: [
        { measurement_index: 0 },                    // s₀
        { input_port: 0, measurement_index: 0 }      // v₀ⁱⁿ
      ]
    },
    {
      base: { tag: "time_like_1" }
      measurements: [
        { measurement_index: 1 },                    // s₁
        { input_port: 0, measurement_index: 1 }      // v₁ⁱⁿ
      ]
    }
  ]

  // Unfinished checks: s₀ ⊕ v₀ᵒᵘᵗ and s₁ ⊕ v₁ᵒᵘᵗ
  // (output virtual measurement is implicit — index 0 → stabilizer 0, index 1 → stabilizer 1)
  unfinished_checks: [
    {
      base: { tag: "output_0" }
      measurements: [{ measurement_index: 0 }]       // s₀ (v₀ᵒᵘᵗ is implicit)
    },
    {
      base: { tag: "output_1" }
      measurements: [{ measurement_index: 1 }]       // s₁ (v₁ᵒᵘᵗ is implicit)
    }
  ]

  errors: [/* ... see below ... */]
}
```

Notice that each unfinished check lists only the physical measurement — the output virtual
stabilizer measurement is implied by the check's position in the array.

#### Error

An `Error` in the JIT format describes which finished and unfinished checks an error
mechanism triggers:

| Field               | Purpose                                                                                                                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `base`              | The deq-bin `ErrorModelType.Error` — carries `tag`, `residual`, `readout_flips`, and `probability`                                                                                                                      |
| `finished_checks`   | List of indices into `finished_checks` that this error triggers                                                                                                                                                          |
| `unfinished_checks` | List of indices into `unfinished_checks` that this error triggers. If empty, the error is resolved immediately. If non-empty, the error model must wait for these unfinished checks to be consumed by downstream gadgets |

**Example — Idle gadget errors:**

```protobuf
  errors: [
    // Measurement error on s₀: triggers finished check 0 and unfinished check 0
    {
      base: { tag: "meas_s0", probability: 0.03 }
      finished_checks: [0]
      unfinished_checks: [0]
    },
    // Measurement error on s₁
    {
      base: { tag: "meas_s1", probability: 0.03 }
      finished_checks: [1]
      unfinished_checks: [1]
    },
    // Data qubit error on q₀: triggers only finished check 0 + residual
    {
      base: { tag: "data_q0", residual: [0], probability: 0.03 }
      finished_checks: [0]
    },
    // Data qubit error on q₁: triggers both finished checks + residual
    {
      base: { tag: "data_q1", residual: [0], probability: 0.03 }
      finished_checks: [0, 1]
    },
    // Data qubit error on q₂
    {
      base: { tag: "data_q2", residual: [0], probability: 0.03 }
      finished_checks: [1]
    }
  ]
```

Note the difference between measurement errors and data qubit errors:
- A **measurement error** on $s_i$ flips only the recorded measurement outcome, not the
  actual stabilizer eigenvalue. This triggers **both** the finished check
  $s_i \oplus v_i^{\text{in}}$ and the unfinished check $s_i \oplus v_i^{\text{out}}$,
  because $s_i$ flips but neither virtual measurement does.
- A **data qubit error** that anti-commutes with a stabilizer flips the physical measurement
  $s_i$ **and** the actual output stabilizer eigenvalue $v_i^{\text{out}}$ simultaneously.
  The unfinished check $s_i \oplus v_i^{\text{out}}$ is **not** triggered (both terms flip,
  canceling out). Only the finished check $s_i \oplus v_i^{\text{in}}$ is triggered (the
  input virtual measurement is unaffected). The data qubit error does not trigger unfinished
  checks — it only has a `residual` on the output observable.

### JitLibrary and JitInstruction

The top-level container and instruction format:

```protobuf
message JitLibrary {
  string description = 1;
  repeated JitGadgetType gadget_types = 2;
  repeated JitPortType port_types = 3;
  repeated JitInstruction program = 5;
}

message JitInstruction {
  deq.bin.Gadget gadget = 1;
  deq.bin.ProbabilityModifier probability_modifier = 2;
}
```

| Field                  | Purpose                                                                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **JitLibrary**         |                                                                                                                                           |
| `description`          | Human-readable description                                                                                                                |
| `gadget_types`         | All `JitGadgetType` definitions (offline analysis results)                                                                                |
| `port_types`           | All `JitPortType` definitions (codes with stabilizer info)                                                                                |
| `program`              | Stream of `JitInstruction` messages                                                                                                       |
| **JitInstruction**     |                                                                                                                                           |
| `gadget`               | A deq-bin `Gadget` message — specifies `gtype`, `connectors` (port connectivity), `tag`, etc. Same format as in deq-bin                 |
| `probability_modifier` | Optional per-instance probability modification (same `ProbabilityModifier` as in deq-bin). Applied to all errors of this gadget instance |

Unlike deq-bin instructions (which can create gadgets, check models, or error models
separately), a `JitInstruction` only instantiates a **gadget**. The JIT compiler
automatically generates the corresponding check model and error model — that's the whole
point.

There is also an `UnloadJitLibrary` message for freeing resources associated with gadget
types that are no longer needed.

---

## How JIT Compilation Works

The JIT compiler processes each `JitInstruction` in two phases: a **synchronous** phase
that returns immediately with the check model, and an **asynchronous** phase that produces
the error model later.

### Synchronous Phase: Immediate Check Model Construction

When a `JitInstruction` arrives:

1. **Assign gadget ID.** The new gadget gets a unique `gid`.

2. **Fetch upstream cached output checks.** For each input port connector, the compiler
   looks up the upstream gadget's **cached output checks** — fully expanded concrete
   measurement sets that the upstream gadget computed and stored when it was instantiated.
   These contain only concrete physical measurement references; all virtual measurements
   have already been eliminated.

3. **Expand finished checks via XOR cancellation.** For each finished check:
   - Physical measurements are recorded directly as
     `(gadget_id, measurement_index)` references
   - Each input virtual measurement is replaced by the corresponding upstream cached output
     check via **symmetric difference** (XOR): measurements appearing in both sets cancel;
     measurements in only one set are retained
   - The result is a set of concrete `(gadget_id, measurement_index)` references — no
     virtual measurements remain

4. **Cache output checks.** For each unfinished check, the compiler performs the same XOR
   expansion on its input virtual measurements, producing a fully resolved concrete
   measurement set. This set is stored as the gadget's **cached output check** for that
   stabilizer. Downstream gadgets will retrieve it in step 2.

5. **Emit CheckModelType + CheckModel.** The expanded finished checks are assembled into
   a deq-bin `CheckModelType` (with `RemoteGadget` references using absolute `gid`s and
   appropriate `measurement_bias` values) and a `CheckModel` instance attached to the
   gadget. These are returned **immediately**.

### Asynchronous Phase: Deferred Error Model Resolution

The compiler also spawns an asynchronous **future** for the error model:

1. **Wait for unfinished checks to be consumed.** Each error may trigger some unfinished
   checks. An unfinished check is **consumed** when a downstream gadget's finished check
   uses it in XOR cancellation. The error model future waits until all unfinished checks
   triggered by *any* error in this gadget have been fully consumed and will never be
   referenced by future gadgets.

2. **Expand errors.** Once all triggered unfinished checks are consumed:
   - Finished check references → explicit `(cid, check_index)` pairs (known from the
     synchronous phase)
   - Unfinished check references → replaced by the now-resolved downstream check
     references. **Critically**, the degree of the error's hyperedge is not known until this
     step: an error triggering a single unfinished check may end up triggering **multiple**
     resolved checks after expansion — for example, when a transversal CNOT propagates an
     error to multiple logical qubits

3. **Emit ErrorModelType + ErrorModel.** The expanded errors are assembled into a deq-bin
   `ErrorModelType` and `ErrorModel` instance.

### Why the Asymmetry?

| Aspect         | Check Models                       | Error Models                             |
| -------------- | ---------------------------------- | ---------------------------------------- |
| Time reference | Only current and past measurements | May reference future checks              |
| Resolution     | Immediate (synchronous)            | Deferred (asynchronous)                  |
| Dependencies   | Input ports (already connected)    | Output ports (may connect in the future) |

This asymmetry is inherent to QEC: a check is a parity constraint on measurements already
performed, while an error may trigger checks in future gadgets not yet instantiated.

---

## Memoization: Amortized Constant-Time Check Expansion

A naïve implementation of XOR cancellation would recursively traverse the full upstream
gadget chain for every new gadget — with cost proportional to the circuit depth. The JIT
compiler avoids this via **memoization** (dynamic programming):

Each gadget, upon instantiation, computes and **caches** the fully expanded concrete
measurement set for each of its output virtual stabilizers (step 4 in the synchronous
phase). This cached set already has all input virtual measurements eliminated. When a downstream
gadget needs to expand its own checks, it performs a **single lookup** per input virtual
measurement into the immediately upstream gadget's cache — no recursive traversal needed.

This ensures:
- Per-gadget compilation cost depends only on the gadget type's check complexity (number of
  checks and measurements per check), **not** on the circuit depth
- Total compilation cost for an $N$-gadget circuit: $O(N \cdot c)$ where $c$ is the average
  per-gadget check complexity — the same as if each gadget were compiled in complete
  isolation

---

## Why JIT Works: Mathematical Foundations

The correctness of JIT compilation rests on three properties: check closure under XOR,
equivalence of augmented and original circuits, and error locality. Together, they guarantee
that recursive XOR cancellation produces correct checks and that error models are complete
once all unfinished checks are consumed.

### Notation

Let $G_1, G_2, \ldots$ be gadgets ordered topologically. Each $G_i$ has:
- Physical measurements $\mathcal{P}_i$
- Input virtual measurements $\mathcal{V}_i^{\text{in}}$
- Output virtual measurements $\mathcal{V}_i^{\text{out}}$
- Unfinished checks $\mathcal{U}_i$

A *check* is a subset $C$ of measurements such that $\bigoplus_{m \in C} m$ is
deterministic (constant for all valid quantum states). We write $A \triangle B$ for the
symmetric difference (XOR) of measurement sets.

**Critical assumption:** When analyzing gadget $G_i$ in isolation, the input quantum state
is **not** assumed to be in the code space. Instead, the input virtual stabilizer
measurements project an arbitrary input state into a definite stabilizer eigenstate — but
the eigenvalues themselves are unknown. Checks must be deterministic for **every** possible
combination of $\pm 1$ input stabilizer measurement outcomes. This is what enables independent
gadget analysis.

### Property 1: Check Closure Under XOR

If $A$ and $B$ are both checks, then $A \triangle B$ is also a check:

$$\bigoplus_{m \in A \triangle B} m = \bigoplus_{m \in A} m \oplus \bigoplus_{m \in B} m = c_A \oplus c_B = \text{const}$$

Measurements appearing in both $A$ and $B$ cancel; measurements in only one set contribute
once.

### Theorem 1: Recursive Check Construction Produces Valid Checks

Consider a finished check $C$ of gadget $G_i$ that contains an input virtual stabilizer
measurement $v \in \mathcal{V}_i^{\text{in}}$. By port connectivity, $v$ corresponds to an
output virtual stabilizer $v' \in \mathcal{V}_j^{\text{out}}$ of some upstream gadget $G_j$.
By the unique-output-check constraint, there is a unique unfinished check $U \in
\mathcal{U}_j$ containing $v'$.

Forming $C' = C \triangle U$ eliminates $v$ (which appears in both). By Property 1, $C'$
is a valid check. If $C'$ still contains input virtual measurements from $G_j$'s input
ports, we recurse: each is expanded by looking up the corresponding upstream unfinished
check and XOR-canceling again.

Since the circuit connectivity is a DAG with no unconnected input ports, the recursion
terminates, and the final result contains only physical measurements. At each step,
Property 1 guarantees validity. $\square$

### Theorem 2: Virtual Stabilizer Measurements Are Removable

The checks derived above are valid in the *augmented* circuit (including virtual
measurements). We now show they are equally valid in the *original* circuit.

An output virtual stabilizer $v'$ appears in exactly one unfinished check
$U = \{v'\} \cup S$, where $S \subseteq \mathcal{P}_j \cup \mathcal{V}_j^{\text{in}}$.
The check being valid means:

$$v' \oplus \bigoplus_{m \in S} m = \text{const}$$

Thus $v'$ is **deterministically determined** by the measurements in $S$: the quantum state
before $v'$ is measured is already an eigenstate of that stabilizer. Performing or omitting
$v'$ does not disturb the quantum state, does not affect subsequent measurements, and its
outcome is predetermined. The circuit with virtual measurements is physically equivalent to
the circuit without them. $\square$

**Corollary:** All virtual measurements across the full circuit are simultaneously removable,
by induction in reverse topological order of the DAG.

### Lemma 1: Error Locality

An error $e$ in gadget $G_j$ does not trigger any check — finished or unfinished, prior to
XOR expansion — belonging to a different gadget $G_i$ ($G_i \neq G_j$).

*Proof.* When analyzing $G_i$, no assumptions are made about the input quantum state. A
check $C$ of $G_i$ satisfies $\bigoplus_{m \in C} m = \text{const}$ for **any** input state.
An error $e$ in upstream gadget $G_j$ modifies the state arriving at $G_i$'s input, but
since $C$ is deterministic for any input, modifying the input cannot change $C$'s value.
$\square$

### Lemma 2: Error Triggering Under XOR

An error $e$ triggers a check $C = A \triangle B$ if and only if $e$ triggers exactly one
of $A$ or $B$ (but not both).

*Proof.* $\bigoplus_{m \in C} m = \bigoplus_{m \in A} m \oplus \bigoplus_{m \in B} m$, so
$e$ flips $C$ iff $e$ changes the parity of exactly one of $A$ and $B$. $\square$

### Theorem 3: Determining Which Resolved Checks an Error Triggers

Let $C_{\text{resolved}}$ be a resolved check produced by XOR-expanding a finished check
$C_{G_i}$ against upstream unfinished checks:
$C_{\text{resolved}} = C_{G_i} \triangle U_1 \triangle U_2 \triangle \cdots \triangle U_m$.

An error $e$ in gadget $G_j$ triggers $C_{\text{resolved}}$ iff $e$ triggers an **odd**
number of the constituent checks $\{C_{G_i}, U_1, \ldots, U_m\}$.

By Lemma 1 (error locality), $e$ can only trigger checks belonging to $G_j$. Two cases:

- **Same-gadget** ($j = i$): The only constituent belonging to $G_j$ is $C_{G_j}$ itself.
  So $e$ triggers the resolved check iff $e$ triggers the finished check — known from
  offline analysis.
- **Upstream-gadget** ($j \neq i$): $C_{G_i}$ belongs to $G_i \neq G_j$ and contributes
  zero. The constituents belonging to $G_j$ are the unfinished checks of $G_j$ that
  participated in the expansion. So $e$ triggers the resolved check iff $e$ triggers an odd
  number of those unfinished checks.

$\square$

### Corollary: Error Model Completeness

An error $e$ in $G_j$ triggers some subset of $G_j$'s finished checks (known immediately)
and some subset of $G_j$'s unfinished checks $\{U_1, \ldots, U_k\}$. By Theorem 3, $e$
triggers each resulting resolved check iff $e$ triggers the corresponding $U_\ell$.

When all unfinished checks $\{U_1, \ldots, U_k\}$ have been fully consumed by downstream
gadgets and no future gadget will reference them, the complete set of resolved checks
triggered by $e$ is determined. No future resolved check can be triggered by $e$ beyond
those already accounted for — because any such check would need an XOR with one of the
$U_\ell$, and none of them will participate in further cancellation. $\square$

This establishes the correctness of asynchronous error model resolution: the error model is
complete at the moment all triggered unfinished checks are fully consumed.

---

## Concurrency and Termination

### Why Concurrent Execution Is Required

All error model futures must execute **concurrently**. An error model for gadget A may wait
for a resolved check from gadget C, while C's error model waits for a resolved check from
gadget E. If these futures ran sequentially (waiting for A before starting C), deadlock
would occur.

This concurrency requirement mirrors the decode concurrency rule described in the
[deq-bin basics](bin-basics.md): decode calls must not depend on prior decode results.
Both constraints arise from a property unique to QEC: decoding inherently requires
information from the future. The JIT compiler needs future gadget instantiations to
determine which checks an error triggers; the decoder needs future syndrome data for window
decoding. In both cases, blocking on a future result while that result depends on
yet-further-future data creates circular waiting.

### Termination Guarantee

Every error model future is guaranteed to resolve, provided the circuit is well-formed (all
output ports are eventually connected):

- **Base case:** A terminal gadget (e.g., logical measurement) has no output ports and
  therefore zero unfinished checks. Its finished checks consume upstream unfinished checks
  without propagating further dependencies, resolving upstream error model futures
  immediately.
- **Inductive case:** Each gadget's resolution depends on a finite set of downstream
  gadgets in the DAG. Since the DAG is acyclic, the dependency chain is finite and
  terminates.

If compilation is aborted, all pending futures are canceled via a shared cancellation token.
Partial results are discarded without publishing incomplete checks.

### Practical Resolution: Insert Idle Cycles

In practice, inserting a **single syndrome extraction gadget** (e.g., an Idle round) is
sufficient to resolve all pending error model futures from preceding gadgets. This is
because a full syndrome extraction measures every stabilizer, so its finished checks consume
all upstream unfinished checks without exposing any input virtual stabilizer measurement to
any output virtual stabilizer measurement — guaranteeing that the upstream error models
resolve.

However, the JIT compiler is not the only component that needs future gadgets. The
**deq-bin decoder** itself also requires future syndrome data for **window decoding**: the
decoder cannot commit a decoding decision for a region until it has accumulated enough
syndrome rounds in its decoding window. Depending on the window size, multiple additional
syndrome extraction rounds may be needed before the decoder actually produces an
error-corrected logical readout.

To avoid excessive engineering effort in determining exactly how many syndrome extraction
rounds to insert in various situations, we recommend a simple rule:

> **Whenever new logical instructions are not yet available, insert idle cycles.**

This rule is sufficient to guarantee that the deq system terminates and eventually produces
error-corrected logical readouts. If the decoder's throughput is high enough relative to the
rate of incoming measurements, this achieves **constant latency** stream decoding — the idle
cycles keep the qubits alive while the decoder catches up, and the system never stalls.

---

## XOR Cancellation: A Worked Example

Let's trace through the JIT compilation of **PrepareZ → Idle → MeasureZ** from the
[deq-bin chapter](bin-basics.md).

### Step 1: Instantiate PrepareZ (gid=1)

PrepareZ has no input ports, no physical measurements, and 2 unfinished checks (one per
output stabilizer):
- Unfinished check 0: empty measurement set (the output virtual $v_0^{\text{out}}$ is
  implicit, and the check is simply $v_0^{\text{out}} = 0$, meaning the stabilizer
  eigenvalue starts at $+1$)
- Unfinished check 1: empty measurement set ($v_1^{\text{out}} = 0$)

**Cached output checks:** `[{}, {}]` — each is an empty set of concrete measurements
(meaning the virtual stabilizer equals a known constant).

### Step 2: Instantiate Idle (gid=2, input connected to PrepareZ output)

The compiler fetches PrepareZ's cached output checks: `[{}, {}]`.

**Expand finished checks:**
- Finished check 0: $s_0 \oplus v_0^{\text{in}}$. Expand $v_0^{\text{in}}$ by XOR with
  upstream cache `{}` → result: $\{(2, 0)\}$ — i.e., `(gid=2, measurement_index=0)`,
  just the local measurement $s_0$.
  This matches the `IdleBoundaryChecks` from the bin chapter: $c_0 = s_0$.
- Finished check 1: similarly → $\{(2, 1)\}$ — just $s_1$.

**Cache output checks:**
- Unfinished check 0: $s_0$ (plus implicit $v_0^{\text{out}}$). Expand input virtuals (none
  in this check) → cached: $\{(2, 0)\}$
- Unfinished check 1: → cached: $\{(2, 1)\}$

**Emit:** `CheckModelType` with 2 checks, each referencing one local measurement. This is
the `IdleBoundaryChecks` type from the bin chapter.

### Step 3: Instantiate MeasureZ (gid=3, input connected to Idle output)

Fetch Idle's cached output checks: `[{(2, 0)}, {(2, 1)}]`.

**Expand finished checks:**
- Finished check 0: $m_0 \oplus m_1 \oplus v_0^{\text{in}}$. Expand $v_0^{\text{in}}$
  by XOR with upstream cache $\{(2, 0)\}$ → result: $\{(3, 0), (3, 1), (2, 0)\}$
  — that is, $m_0 \oplus m_1 \oplus s_0$. This is $c_0$ from the bin chapter's
  `MeasureChecks`.
- Finished check 1: → $\{(3, 1), (3, 2), (2, 1)\}$ — $m_1 \oplus m_2 \oplus s_1$.

MeasureZ has no output ports, so no unfinished checks and no caching needed.

**Emit:** `CheckModelType` with 2 checks, each referencing local measurements + one remote
measurement from gadget 2 (the Idle). This is exactly the `MeasureChecks` type from the
bin chapter.

### Error Model Resolution

- PrepareZ's error model future: PrepareZ has no physical measurements and no errors that
  trigger unfinished checks, so it resolves immediately.
- Idle's error model future: Idle's errors trigger unfinished checks 0 and 1. These are
  consumed when MeasureZ's finished checks expand them via XOR. After MeasureZ is
  instantiated, all of Idle's unfinished checks are consumed → the error model future
  resolves, producing the `IdleErrors` type from the bin chapter.
- MeasureZ's error model future: MeasureZ has no unfinished checks → resolves immediately,
  producing `MeasureErrors`.

The JIT compiler has produced the same deq-bin output as the hand-written example in the
bin chapter — but dynamically, from the JIT intermediate data, without pre-enumerating all
possible type definitions.

---

## Summary

The deq-JIT compiler bridges offline gadget analysis and online circuit execution:

1. **Virtual stabilizer measurements** enable each gadget type to be analyzed independently,
   describing its checks and errors in terms of physical measurements + virtual measurements
   at port boundaries
2. **Finished/unfinished check classification** with the unique-output-check constraint
   enables efficient XOR cancellation without GF(2) equation solving
3. **Synchronous check model construction** produces check models immediately as gadgets are
   instantiated, enabling the decoder to start decoding early regions while later regions are
   still being constructed
4. **Asynchronous error model resolution** handles the fundamental asymmetry of QEC: errors
   may trigger future checks, so error models are deferred until all dependencies resolve
5. **Memoization** ensures per-gadget compilation cost is independent of circuit depth,
   achieving $O(N \cdot c)$ total cost
6. **Mathematical foundations** (check closure under XOR, virtual measurement removability,
   error locality) guarantee correctness

The JIT compiler produces deq-bin output that can be consumed by any compatible decoder
(software or hardware), enabling asymptotically constant latency stream decoding of
arbitrary dynamic logical circuits.

For constructing well-structured decoding hypergraphs from user-friendly textual
definitions — including hierarchical composition (COMPOSE) and automatic check
derivation — see the [deq language basics](language-basics.md) chapter.

