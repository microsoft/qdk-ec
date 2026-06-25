# deq-bin Basics

## Why deq-bin?

Every quantum error correction (QEC) decoder — whether running in software on a CPU/GPU or
in dedicated hardware on an FPGA/ASIC — fundamentally operates on a **decoding hypergraph**.
This hypergraph encodes the relationships between syndrome checks, error mechanisms, and
logical observables. Given a syndrome (which checks fired), the decoder infers the most
likely error pattern and returns a correction (which logical observables flipped).

The challenge is: **the decoding hypergraph depends on the logical circuit being executed.**
Different sequences of logical operations produce structurally different hypergraphs.
Flat formats (like Stim's detector error model) describe one specific, complete hypergraph
for one specific circuit. This creates two problems:

1. **No incremental construction.** If the circuit changes — or if you simply need to decode
   a different logical program using the same gadgets — the entire hypergraph must be
   regenerated from scratch.
2. **No efficient stream decoding.** In a real fault-tolerant computation, you often need to
   decode logical readouts **in the middle** of the circuit (e.g., for feed-forward or
   hybrid quantum-classical algorithms). With a flat format, producing a mid-circuit readout
   requires regenerating the decoding hypergraph up to that point. As the circuit grows
   deeper, this cost grows proportionally — making stream decoding with asymptotically
   constant latency impossible. You cannot use existing tools like Stim to efficiently decode
   a stream of measurements arriving incrementally.

**deq-bin** solves both problems. It is primarily a **standard for stream decoding**: the
decoding hypergraph is constructed incrementally as instructions arrive, and previously
committed regions can be decoded and freed while new regions are still being constructed.
If the decoder backend is fast enough, this enables **asymptotically constant latency**
decoding regardless of circuit depth.

deq-bin achieves this by separating:

- **Type definitions** (templates): compiled once, describing the physical structure of each
  gadget (logical operation), its syndrome checks, and its error mechanisms. Types are loaded
  once and cached — software decoders can precompute and cache data structures for each type
  at initialization, and hardware decoders (which typically require the decoding hypergraph
  structure to be known offline) can load types into on-chip memory or synthesize them
  directly into logic.
- **Instances** (runtime instructions): a stream of lightweight messages that reference
  types and specify how gadget instances connect to each other. These are streamed
  incrementally as the circuit executes.

Any decoder that implements deq-bin can decode **any** logical circuit composed from the
defined types, without regenerating the hypergraph from scratch. The types are compiled
offline (by the deq compiler) and loaded once; the program is streamed as the circuit
executes. This makes deq-bin the universal interchange format between compilers (which
understand the physics) and decoders (which solve the inference problem).

---

## Decoding Hypergraphs 101

A decoding hypergraph has two kinds of elements:

- **Vertices (checks):** A check is a parity constraint on measurement outcomes — the XOR
  of a specific subset of measurements. Under no errors, each check has a deterministic
  expected value (usually 0). When an error occurs, it may flip one or more checks, causing
  them to deviate from their expected values. The set of flipped checks is the **syndrome**.

- **Hyperedges (errors):** Each error mechanism connects the set of checks it would flip,
  carries a probability, and may flip one or more logical observables. For example, a
  hyperedge touching checks $\{c_0, c_2\}$ with probability $p$ means: with probability $p$,
  both $c_0$ and $c_2$ are flipped simultaneously.

The decoder receives the syndrome and infers the most likely combination of errors that
explains it. The logical readout is correct if the combined effect of the actual errors and
the decoder's corrections is trivial on the logical observables.

<!-- TODO: figure — a simple decoding hypergraph showing vertices (checks) and hyperedges (errors) -->

---

## Why Circuit Context Matters

To see why circuit context changes the decoding hypergraph nontrivially, consider a
**distance-3 repetition code** with:

- 3 data qubits: $q_0, q_1, q_2$
- 2 stabilizers: $Z_0 Z_1$ and $Z_1 Z_2$
- 1 logical observable: $Z_0 Z_1 Z_2$

We define three gadget types:

| Gadget       | What it does                                                                    | Measurements        | Ports             |
| ------------ | ------------------------------------------------------------------------------- | ------------------- | ----------------- |
| **PrepareZ** | Reset all 3 data qubits to $\lvert 0 \rangle$                                   | 0                   | 1 output          |
| **Idle**     | One round of syndrome extraction (measure $Z_0Z_1$ and $Z_1Z_2$ using ancillas) | 2 ($s_0, s_1$)      | 1 input, 1 output |
| **MeasureZ** | Measure all 3 data qubits individually                                          | 3 ($m_0, m_1, m_2$) | 1 input           |

Now compare three circuits built from these same gadgets:

### Scenario A: PrepareZ → MeasureZ

The simplest circuit — prepare and immediately measure. Only MeasureZ produces measurements
($m_0, m_1, m_2$).

**Checks:**
- $c_0 = m_0 \oplus m_1$ — the $Z_0 Z_1$ parity from data measurements (expected 0)
- $c_1 = m_1 \oplus m_2$ — the $Z_1 Z_2$ parity from data measurements (expected 0)
- note that although $m_0 \oplus m_1 \oplus m_2$ is also a check by definition, we will have to exclude it because otherwise the logical error rate is always 0

**Errors** (each is a hyperedge in the decoding hypergraph):
- Data error on $q_0$: flips $c_0$, flips logical $Z$ → hyperedge $\{c_0\}$ + observable
- Data error on $q_1$: flips $c_0$, $c_1$ and logical $Z$ → hyperedge $\{c_0, c_1\}$ + observable
- Data error on $q_2$: flips $c_1$, flips logical $Z$ → hyperedge $\{c_1\}$ + observable

**Result:** A 2-vertex hypergraph with 3 hyperedges.

<!-- TODO: figure — Scenario A decoding hypergraph (2 vertices, 3 edges) -->

### Scenario B: PrepareZ → Idle → MeasureZ

Now we have syndrome measurements from Idle ($s_0, s_1$) plus data measurements from
MeasureZ ($m_0, m_1, m_2$).

**Checks:**
- $c_0 = s_0$ — raw syndrome for $Z_0 Z_1$ (expected 0, since state was $\lvert 000 \rangle$)
- $c_1 = s_1$ — raw syndrome for $Z_1 Z_2$ (expected 0)
- $c_2 = m_0 \oplus m_1 \oplus s_0$ — data parity minus last syndrome
- $c_3 = m_1 \oplus m_2 \oplus s_1$

**New error types** (in addition to data errors):
- Measurement error on $s_0$: flips $c_0$ and $c_2$ → hyperedge $\{c_0, c_2\}$
- Measurement error on $s_1$: flips $c_1$ and $c_3$ → hyperedge $\{c_1, c_3\}$

**Result:** A 4-vertex hypergraph. Measurement errors create **temporal** edges connecting
checks across gadgets.

<!-- TODO: figure — Scenario B decoding hypergraph (4 vertices, temporal edges) -->

### Scenario C: PrepareZ → Idle → Idle → MeasureZ

Two rounds of syndrome extraction: $(s_0^1, s_1^1)$ from the first Idle, $(s_0^2, s_1^2)$
from the second, plus data measurements $(m_0, m_1, m_2)$.

**Checks:**
- $c_0 = s_0^1$ — first Idle, boundary (raw measurement)
- $c_1 = s_1^1$ — first Idle, boundary
- $c_2 = s_0^2 \oplus s_0^1$ — second Idle, **difference** with first
- $c_3 = s_1^2 \oplus s_1^1$ — second Idle, **difference** with first
- $c_4 = m_0 \oplus m_1 \oplus s_0^2$
- $c_5 = m_1 \oplus m_2 \oplus s_1^2$

**Result:** A 6-vertex hypergraph.

<!-- TODO: figure — Scenario C decoding hypergraph (6 vertices, showing how the same Idle gadget produces different check structures) -->

### The Key Insight

The **same Idle gadget type** appears in two different contexts in Scenario C, but produces
**structurally different checks:**

| Idle instance                  | Predecessor                       | Check structure                                                |
| ------------------------------ | --------------------------------- | -------------------------------------------------------------- |
| First Idle (after PrepareZ)    | PrepareZ has **0 measurements**   | Checks are raw local measurements: $c = s$                     |
| Second Idle (after first Idle) | First Idle has **2 measurements** | Checks XOR local and remote measurements: $c = s^2 \oplus s^1$ |

The first Idle's checks reference only local measurements. The second Idle's checks must
reach across the gadget boundary to access the predecessor's measurements. This is a
**nontrivial structural difference** — not just a parameter change, but a different number
of measurements per check.

Moreover, the "predecessor" is not necessarily the gadget immediately before the current
one. In realistic circuits, there may be multiple **transversal gadgets** (e.g., logical
CNOT, logical S) between syndrome extraction rounds. These transversal gadgets have no
syndrome measurements of their own, so a check model must traverse *through* them — hopping
across multiple gadgets via the port connectivity graph — to reach the last gadget that
actually produced syndrome measurements. This is why remote references support
**chaining** via `previous_remote_gadget`: multi-hop paths are essential, not just a
convenience.

This is exactly the problem deq-bin solves. In deq-bin:

1. There is one `GadgetType` for Idle (same physical circuit regardless of context).
2. There are **multiple `CheckModelTypes`** for the same gadget — one per distinct context:
   - **IdleBoundaryChecks** — for an Idle after a gadget with no syndrome measurements.
     Each check references only local measurements.
   - **IdleBulkChecks** — for an Idle after another Idle. Each check uses a **remote
     reference** to reach the predecessor's measurements via the input port.
   - In general, the space of possible contexts can be very large (different predecessor
     gadget types, different port configurations, etc.). It may not be practical to
     enumerate all possible `CheckModelTypes` offline. This is one reason the
     [deq JIT compiler](jit-basics.md) exists — it can generate the appropriate
     `CheckModelType` at runtime, efficiently, given the actual circuit context.
3. **`ErrorModelTypes`** face a similar context-dependence. Depending on what gadget comes
   *next* in the circuit, an error mechanism may flip different checks in the successor —
   so different contexts may require different error model types. The JIT compiler handles
   this as well.
4. The **program stream** decides which `CheckModelType` and `ErrorModelType` to attach to
   each instance based on its circuit context.

The type-instance separation and remote reference mechanism are the core of deq-bin. The
rest of this tutorial explains how every piece works.

---

## The deq-bin Format

deq-bin is defined as a [Protocol Buffers](https://protobuf.dev/) schema. All the types
and messages below correspond directly to the definitions in `deq_bin.proto`.

### Library: The Top-Level Container

A `Library` is the top-level message that packages everything together:

```protobuf
message Library {
  string description = 1;
  repeated GadgetType gadget_types = 2;
  repeated PortType port_types = 3;
  repeated CheckModelType check_model_types = 4;
  repeated ErrorModelType error_model_types = 5;
  repeated Instruction program = 6;
}
```

| Field               | Purpose                                                                   |
| ------------------- | ------------------------------------------------------------------------- |
| `description`       | Human-readable description of the library                                 |
| `gadget_types`      | All gadget type definitions (templates for physical operations)           |
| `port_types`        | All port type definitions (code interfaces between gadgets)               |
| `check_model_types` | All check model type definitions (syndrome check templates)               |
| `error_model_types` | All error model type definitions (error mechanism templates)              |
| `program`           | A reference program showing how to use the types (list of `Instruction`s) |

The `program` field serves as an example — it is the instruction stream that constructs the
decoding hypergraph for one specific circuit. Different circuits using the same types would
have different programs but share the same type definitions.

### Utility Type: BitMatrix

Before diving into the types, we need to understand `BitMatrix`, the representation used for
all binary matrices over $\mathrm{GF}(2)$:

```protobuf
message BitMatrix {
  uint64 rows = 1;
  uint64 cols = 2;
  repeated uint64 i = 3;  // row indices of 1-entries
  repeated uint64 j = 4;  // column indices of 1-entries
}
```

This is a **sparse representation**: only the positions of 1s are stored. The arrays `i` and
`j` must have equal length, and each pair `(i[k], j[k])` is the position of a 1-entry.

**Example:** A $2 \times 3$ identity-like matrix:

$$
\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}
$$

is encoded as:
```protobuf
BitMatrix { rows: 2, cols: 3, i: [0, 1], j: [0, 1] }
```

An all-zero matrix of size $1 \times 2$ is simply:
```protobuf
BitMatrix { rows: 1, cols: 2 }
```

### PortType: Code Interfaces

Ports are the interfaces through which gadgets connect. A `PortType` defines a QEC code
interface — specifically, the logical observables it carries:

```protobuf
message PortType {
  uint64 ptype = 1;      // globally unique ID
  string name = 2;
  string description = 3;
  repeated Observable observables = 4;

  message Observable {
    string tag = 1;       // e.g., "Z" or "X"
  }
}
```

| Field         | Purpose                                                                                |
| ------------- | -------------------------------------------------------------------------------------- |
| `ptype`       | Unique identifier for this port type (starts from 1)                                   |
| `name`        | Human-readable name (e.g., `"RepetitionCode"`)                                         |
| `description` | Optional description                                                                   |
| `observables` | The logical observables carried by this port. Each observable has a `tag` for labeling |

**Example:** A repetition code port with one logical Z observable:
```protobuf
PortType {
  ptype: 1
  name: "RepetitionCode"
  observables: [{ tag: "Z" }]
}
```

Two gadgets can only be connected through ports of the **same `PortType`** — connecting an
output port of one gadget to an input port of another means they share the same code
interface and the same set of logical observables.

### GadgetType: Physical Building Blocks

A `GadgetType` defines the physical structure of a logical operation:

```protobuf
message GadgetType {
  uint64 gtype = 1;
  string name = 2;
  string description = 3;
  repeated Measurement measurements = 4;
  repeated Port inputs = 5;
  repeated Port outputs = 6;
  BitMatrix correction_propagation = 7;
  repeated Readout readouts = 8;
  BitMatrix readout_propagation = 9;
  BitMatrix logical_correction = 10;
  BitMatrix physical_correction = 15;
  optional bool is_free_hop = 14;

  message Measurement { string tag = 1; }
  message Port { uint64 ptype = 1; string tag = 2; }
  message Readout { string tag = 1; repeated uint64 measurement_indices = 2; }
}
```

Let's walk through each field using the repetition code gadgets:

#### Basic structure

| Field                 | Purpose                                                                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `gtype`               | Unique identifier for this gadget type                                                                                                     |
| `name`, `description` | Human-readable labels                                                                                                                      |
| `measurements`        | List of physical measurements this gadget performs. The decoder doesn't need details — just how many to expect. Each has an optional `tag` |
| `inputs`              | Input ports. Each references a `ptype` and has an optional `tag`                                                                           |
| `outputs`             | Output ports. Same structure as inputs                                                                                                     |

**Example — Idle gadget type:**
```protobuf
GadgetType {
  gtype: 2
  name: "Idle"
  measurements: [{ tag: "s0" }, { tag: "s1" }]   // 2 syndrome measurements
  inputs: [{ ptype: 1 }]                         // 1 input port (RepetitionCode)
  outputs: [{ ptype: 1 }]                        // 1 output port (RepetitionCode)
}
```

#### Readouts

A `Readout` defines a classical output bit as the XOR of a subset of local measurements:

| Field                 | Purpose                                                                               |
| --------------------- | ------------------------------------------------------------------------------------- |
| `tag`                 | Label for the readout                                                                 |
| `measurement_indices` | List of local measurement indices. The readout value is the XOR of these measurements |

**Example — MeasureZ with one logical readout:**
```protobuf
GadgetType {
  gtype: 3
  name: "MeasureZ"
  measurements: [{ tag: "m0" }, { tag: "m1" }, { tag: "m2" }]
  inputs: [{ ptype: 1 }]
  readouts: [{ tag: "logical_Z", measurement_indices: [0, 1, 2] }]  // readout = m₀ ⊕ m₁ ⊕ m₂
}
```

#### Pauli frame matrices

Four $\mathrm{GF}(2)$ matrices define how the Pauli frame propagates through a gadget. All
operations are over $\mathrm{GF}(2)$ (bitwise XOR).

**`correction_propagation`** — dimensions: $|\text{output observables}| \times (|\text{input observables}| + 1)$

Maps input observable corrections to output observable corrections. The extra column is a
**constant term** — used for gadgets that apply a virtual logical Pauli gate (no physical
gate, but the correction changes).

*Example — Idle:* 1 output observable, 1 input observable → $1 \times 2$ matrix.
Input Z correction passes through to output:
```protobuf
correction_propagation: { rows: 1, cols: 2, i: [0], j: [0] }
// Matrix: [1, 0] — input correction maps to output, constant term is 0
```

*Example — PrepareZ:* 1 output observable, 0 input observables → $1 \times 1$ matrix.
No input, constant term is 0 (fresh preparation has no correction):
```protobuf
correction_propagation: { rows: 1, cols: 1 }
// Matrix: [0] — constant term is 0
```

**`readout_propagation`** — dimensions: $|\text{readouts}| \times (|\text{input observables}| + 1)$

Maps input observable corrections to their effect on readout values.

*Example — MeasureZ:* 1 readout, 1 input observable → $1 \times 2$ matrix.
If the input has a Z correction, the readout must be flipped:
```protobuf
readout_propagation: { rows: 1, cols: 2, i: [0], j: [0] }
// Matrix: [1, 0] — input Z correction flips the readout
```

**`logical_correction`** — dimensions: $|\text{output observables}| \times |\text{readouts}|$

Maps local readout values back to output observable corrections — the local Pauli feed-forward
mechanism. Used when a readout from this gadget should conditionally modify the Pauli frame.

Most gadgets have an empty matrix (no local feed-forward).

**`physical_correction`** — dimensions: $|\text{output observables}| \times |\text{measurements}|$

Maps physical measurement outcomes to output observable corrections. This matrix encodes how
measurement results update the Pauli frame — capturing two distinct physical effects:

- **Destabilizer corrections in preparation gadgets:** When a code state is prepared (e.g.,
  via stabilizer measurements), the measurement outcomes determine which Pauli frame
  corrections are needed to project into the desired $+1$ eigenstate. The `physical_correction`
  matrix maps these measurement outcomes to the appropriate frame updates.
- **Logical corrections based on physical measurements:** When a gadget's raw (non-error-corrected)
  measurement outcomes determine how the Pauli frame should be updated — for example, ejection gates.

*Example — $[\![4,2,2]\!]$ code PrepareXX gadget:* This gadget prepares the logical $|{+}{+}\rangle$
state by resetting all 4 data qubits to $|{+}\rangle$ and then measuring the $ZZZZ$ stabilizer. It has 1 measurement and 6 output observables —
4 logical ($LX_0, LZ_0, LX_1, LZ_1$) plus 2 stabilizer-tracking columns ($S_0{:}XXXX$,
$S_1{:}ZZZZ$). When the $ZZZZ$ measurement returns $-1$, the Pauli frame must record this
in the stabilizer column $S_1$:

```protobuf
physical_correction: { rows: 6, cols: 1, i: [5], j: [0] }
// Column 0 (ZZZZ measurement): flips S₁ (row 5)
```

The logical observable columns are unaffected — in this case, `physical_correction` updates only the
stabilizer-tracking part of the frame.

#### is_free_hop

```protobuf
optional bool is_free_hop = 14;
```

For **window decoding**: the decoder uses a sliding window measured in "hops" (gadgets). A
free-hop gadget (e.g., a transversal logical gate with no physical measurements) contributes 0 to
the hop distance. By default, a gadget with no measurements is a free hop. Set this field
explicitly to override the default.

### CheckModelType: Syndrome Checks

The `CheckModelType` defines which measurements are XORed to form each syndrome check. This
is where **remote references** come in — the core mechanism that makes deq-bin work.

```protobuf
message CheckModelType {
  uint64 ctype = 1;
  string name = 2;
  string description = 3;
  uint64 gtype = 4;                        // which gadget type this attaches to (0 = any)
  repeated RemoteGadget remote_gadgets = 5;
  repeated Check checks = 7;
}
```

| Field                 | Purpose                                                                         |
| --------------------- | ------------------------------------------------------------------------------- |
| `ctype`               | Unique identifier for this check model type                                     |
| `name`, `description` | Human-readable labels                                                           |
| `gtype`               | Constrains which gadget type this check model can attach to. Set to 0 for "any" |
| `remote_gadgets`      | Declarations of how to reach neighboring gadgets (see below)                    |
| `checks`              | The syndrome checks, each defined as a XOR of measurements                      |

#### Remote References (RemoteGadget)

A `RemoteGadget` describes how to locate a neighboring gadget relative to the current one:

```protobuf
message RemoteGadget {
  optional uint64 previous_remote_gadget = 1;
  oneof port {
    uint64 input = 2;
    uint64 output = 3;
  }
  uint64 expecting_gtype = 4;
  uint64 measurement_bias = 5;
  string tag = 6;
  optional uint64 absolute_gid = 7;
}
```

| Field                    | Purpose                                                                                                                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input` / `output`       | Port traversal direction. `input: 0` means "follow the gadget's input port 0 to the connected gadget." `output: 0` means "follow output port 0"                                               |
| `previous_remote_gadget` | Optional chaining. Instead of starting from the current gadget, start from a previously resolved remote gadget. Enables multi-hop paths                                                       |
| `expecting_gtype`        | Sanity check — assert the target gadget has this type. Set to 0 for "any" (wildcard)                                                                                                          |
| `measurement_bias`       | An unsigned integer offset added to all measurement indices when accessing this remote gadget's measurements. Enables **template reuse** across gadgets with multiple ports (explained below) |
| `tag`                    | Debugging label                                                                                                                                                                               |
| `absolute_gid`           | Direct gadget ID for $O(1)$ lookup (see [Dual Addressing](#dual-addressing-relative-vs-absolute))                                                                                             |

**Why `measurement_bias`?** Consider a gadget type with two output ports, where measurements
0–3 correspond to port 0 and measurements 4–7 correspond to port 1. A check model template
that references "measurement 0 of the remote gadget" can be reused for both ports: when
connected via port 0, set `measurement_bias = 0`; via port 1, set `measurement_bias = 4`.
The actual measurement accessed is:

$$\text{actual\_index} = \text{measurement\_index} + \text{measurement\_bias}$$

Without bias, you would need a separate check model type for each port — a combinatorial
explosion.

#### Checks and RemoteMeasurement

Each `Check` is the XOR of a list of measurements, which may be local or from remote
gadgets:

```protobuf
message Check {
  string tag = 1;
  repeated RemoteMeasurement measurements = 2;
  bool naturally_flipped = 3;
}

message RemoteMeasurement {
  optional uint64 remote_gadget = 1;   // index into remote_gadgets list
  uint64 measurement_index = 2;        // index within the gadget
}
```

| Field               | Purpose                                                                                                                                                                                                                                                                                                           |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tag`               | Label for the check                                                                                                                                                                                                                                                                                               |
| `measurements`      | List of measurements to XOR. Each `RemoteMeasurement` specifies a measurement by its index within a gadget. If `remote_gadget` is omitted, the measurement is **local** (from the current gadget). If `remote_gadget` is specified, it indexes into the `remote_gadgets` list to determine which gadget to access |
| `naturally_flipped` | Set to `true` when the expected value of the check is 1 (not 0) under no errors. This can occur, e.g., when preparing a $-1$ eigenstate of a stabilizer                                                                                                                                                           |

For remote measurements, the actual index is shifted by the bias:

$$\text{actual\_index} = \text{measurement\_index} + \text{measurement\_bias}[\text{remote\_gadget}]$$

#### Example: IdleBoundaryChecks and IdleBulkChecks

Here are the two check model types for the Idle gadget in our repetition code:

**IdleBoundaryChecks** — for Idle after PrepareZ (no predecessor measurements):
```protobuf
CheckModelType {
  ctype: 1
  name: "IdleBoundaryChecks"
  gtype: 2                                          // attaches to Idle
  // no remote_gadgets — all measurements are local
  checks: [
    { tag: "c0", measurements: [{ measurement_index: 0 }] },   // c₀ = s₀
    { tag: "c1", measurements: [{ measurement_index: 1 }] }    // c₁ = s₁
  ]
}
```

Each check references a single local measurement. No remote references needed because
PrepareZ has no syndrome measurements to diff against.

**IdleBulkChecks** — for Idle after another Idle (difference of consecutive rounds):
```protobuf
CheckModelType {
  ctype: 2
  name: "IdleBulkChecks"
  gtype: 2                                          // attaches to Idle
  remote_gadgets: [{ input: 0 }]                    // hop via input port → predecessor
  checks: [
    {
      tag: "c0",
      measurements: [
        { measurement_index: 0 },                   // local s₀
        { remote_gadget: 0, measurement_index: 0 }  // predecessor's s₀
      ]
    },
    {
      tag: "c1",
      measurements: [
        { measurement_index: 1 },                   // local s₁
        { remote_gadget: 0, measurement_index: 1 }  // predecessor's s₁
      ]
    }
  ]
}
```

Each check XORs a local measurement with a remote measurement from the predecessor. The
`remote_gadget: 0` means "use remote gadget entry 0," which hops via the input port to
find the predecessor.

**MeasureChecks** — for MeasureZ after Idle:
```protobuf
CheckModelType {
  ctype: 3
  name: "MeasureChecks"
  gtype: 3                                          // attaches to MeasureZ
  remote_gadgets: [{ input: 0 }]                    // hop via input port → predecessor
  checks: [
    {
      tag: "c0",
      measurements: [
        { measurement_index: 0 },                   // local m₀
        { measurement_index: 1 },                   // local m₁
        { remote_gadget: 0, measurement_index: 0 }  // predecessor's s₀
      ]
    },
    {
      tag: "c1",
      measurements: [
        { measurement_index: 1 },                   // local m₁
        { measurement_index: 2 },                   // local m₂
        { remote_gadget: 0, measurement_index: 1 }  // predecessor's s₁
      ]
    }
  ]
}
```

The check $c_0 = m_0 \oplus m_1 \oplus s_0$ combines two local measurements with one
remote measurement. This verifies that the $Z_0 Z_1$ parity from data measurements matches
the last syndrome extraction.

### ErrorModelType: Error Mechanisms

An `ErrorModelType` defines the error mechanisms (hyperedges) in the decoding hypergraph.
Its structure mirrors `CheckModelType`: it declares remote check model references, then
defines errors that connect checks across check models.

```protobuf
message ErrorModelType {
  uint64 etype = 1;
  string name = 2;
  string description = 3;
  uint64 ctype = 4;                      // which check model type to attach to (0 = any)
  repeated RemoteCheckModel remote_check_models = 5;
  repeated Error errors = 6;
}
```

| Field                 | Purpose                                                                 |
| --------------------- | ----------------------------------------------------------------------- |
| `etype`               | Unique identifier                                                       |
| `name`, `description` | Human-readable labels                                                   |
| `ctype`               | Constrains which check model type this error model attaches to. 0 = any |
| `remote_check_models` | How to reach neighboring check models (analogous to `remote_gadgets`)   |
| `errors`              | The error mechanisms                                                    |

#### Remote Check Model References

```protobuf
message RemoteCheckModel {
  optional uint64 previous_remote_check_model = 1;
  oneof port {
    uint64 input = 2;
    uint64 output = 3;
  }
  uint64 expecting_ctype = 4;
  uint64 check_bias = 5;
  string tag = 6;
  optional uint64 absolute_cid = 7;
}
```

This mirrors `RemoteGadget` exactly, but for check models:
- Port traversal follows the **owning gadget's** ports (the gadget the check model is
  attached to)
- `check_bias` is the check-level analog of `measurement_bias`:
  $\text{actual\_index} = \text{check\_index} + \text{check\_bias}$
- `absolute_cid` enables $O(1)$ lookup (see [Dual Addressing](#dual-addressing-relative-vs-absolute))

#### Errors

```protobuf
message Error {
  string tag = 1;
  repeated RemoteCheck checks = 2;
  repeated uint64 residual = 3;
  repeated uint64 readout_flips = 4;
  double probability = 5;
}

message RemoteCheck {
  optional uint64 remote_check_model = 1;  // index into remote_check_models
  uint64 check_index = 2;
}
```

| Field           | Purpose                                                                                                                                                    |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tag`           | Label                                                                                                                                                      |
| `checks`        | Which checks this error flips. Each `RemoteCheck` specifies a check by index, optionally from a remote check model                                         |
| `residual`      | List of **output observable indices** that this error flips. Indices refer to the concatenated output observables of all output ports of the owning gadget |
| `readout_flips` | List of **readout indices** that this error flips. Used when an error directly affects a gadget's readout value                                            |
| `probability`   | Probability of this error occurring, in $[0, 1]$                                                                                                           |

**`residual` vs `readout_flips`:** `residual` tracks the effect on logical observables
propagated through output ports (relevant for mid-circuit corrections). `readout_flips`
tracks the effect on classical readout bits (relevant for final measurements). An error can
have both, either, or neither.

#### Example: IdleErrors

```protobuf
ErrorModelType {
  etype: 1
  name: "IdleErrors"
  ctype: 0                                          // attaches to any check model type
  remote_check_models: [{ output: 0 }]              // hop via gadget's output port → successor's check model
  errors: [
    // Measurement error on s₀: flips local check and successor's check
    {
      tag: "meas_s0",
      checks: [
        { check_index: 0 },                         // local c₀
        { remote_check_model: 0, check_index: 0 }   // successor's c₀
      ],
      probability: 0.03
    },
    // Measurement error on s₁
    {
      tag: "meas_s1",
      checks: [
        { check_index: 1 },
        { remote_check_model: 0, check_index: 1 }
      ],
      probability: 0.03
    },
    // Data qubit error on q₀: flips one spatial check + logical observable
    {
      tag: "data_q0",
      checks: [{ check_index: 0 }],
      residual: [0],                                 // flips logical Z
      probability: 0.03
    },
    // Data qubit error on q₁: flips two spatial checks
    {
      tag: "data_q1",
      checks: [{ check_index: 0 }, { check_index: 1 }],
      residual: [0],
      probability: 0.03
    },
    // Data qubit error on q₂
    {
      tag: "data_q2",
      checks: [{ check_index: 1 }],
      residual: [0],
      probability: 0.03
    }
  ]
}
```

Notice that `ctype: 0` (any check model) allows the same error model type to be reused for
both `IdleBoundaryChecks` and `IdleBulkChecks`. The remote check model reference
`{ output: 0 }` resolves to whichever check model is on the successor gadget — this is the
power of **relative addressing**: the same type template works regardless of what comes next
in the circuit.

<!-- TODO: figure — error model showing temporal edges (measurement errors) and spatial edges (data errors) -->

#### Example: MeasureErrors

```protobuf
ErrorModelType {
  etype: 2
  name: "MeasureErrors"
  ctype: 3
  // no remote_check_models — MeasureZ is the last gadget
  errors: [
    // Data qubit error on q₀: flips check and readout
    {
      tag: "data_q0",
      checks: [{ check_index: 0 }],
      readout_flips: [0],                            // flips the logical readout
      probability: 0.03
    },
    // Data qubit error on q₁
    {
      tag: "data_q1",
      checks: [{ check_index: 0 }, { check_index: 1 }],
      readout_flips: [0],
      probability: 0.03
    },
    // Data qubit error on q₂
    {
      tag: "data_q2",
      checks: [{ check_index: 1 }],
      readout_flips: [0],
      probability: 0.03
    }
  ]
}
```

Since MeasureZ has no output ports, there are no output observables — so `residual` is
always empty. Instead, errors that flip the logical value use `readout_flips` to indicate
which readout is affected.

**Why both `residual` and `readout_flips` exist:** In QEC, we don't actually apply physical
corrections to the circuit. Instead, the decoder maintains a **Pauli frame** — a classical
record of what logical correction would need to be applied. When the decoder determines that
a data qubit error occurred in, say, the Idle gadget in the middle of the circuit, that
error has a `residual` flip on the output observable. This residual propagates through the
`correction_propagation` matrices of all successor gadgets in the chain until it reaches the
input port of the MeasureZ gadget. There, the `readout_propagation` matrix says that the
readout is sensitive to this Pauli frame bit, so the decoder knows to flip the logical
readout value accordingly. This is the normal path: errors in the middle of the circuit
affect the final readout **indirectly**, through Pauli frame propagation.

On the other hand, an error in the final MeasureZ gadget — such as a bit-flip on data qubit
$q_0$ right before measurement — directly corrupts the measurement outcome. There is no
successor gadget to propagate through; the error immediately affects the readout. This is
why MeasureZ errors use `readout_flips` instead of `residual`: they bypass the Pauli frame
propagation path entirely.

In more complex circuits — such as those using **lattice surgery** — a single error can have
**both** `residual` and `readout_flips` simultaneously: it flips a readout in the current
gadget *and* propagates a residual correction to successor gadgets. This is why the decoder
must decode every gadget in the circuit, not just those at the final logical readout. The
full space-time structure of the decoding hypergraph matters: errors at any point can affect
both local readouts and downstream Pauli frame propagation, and the decoder must consider
all possible combinations.

### The Instruction Stream (Program)

The second half of a deq-bin `Library` is the **program**: an ordered sequence of
`Instruction` messages. Each instruction creates exactly one instance:

```protobuf
message Instruction {
  oneof create {
    Gadget gadget = 1;
    CheckModel check_model = 2;
    ErrorModel error_model = 3;
  }
}
```

#### Gadget Instances

```protobuf
message Gadget {
  uint64 gtype = 1;                    // which gadget type to instantiate
  string tag = 2;
  repeated Connector connectors = 3;   // connect input ports to predecessor output ports
  uint64 gid = 5;                      // auto-assigned sequentially from 1
  optional GadgetModifier modifier = 6;

  message Connector {
    uint64 gid = 1;                    // the predecessor gadget's ID
    uint64 port = 2;                   // which of its output ports to connect
  }
}
```

The `connectors` list must match the gadget type's `inputs` in length. Each `Connector`
specifies which previously created gadget (by `gid`) and which of its output ports to
connect to.

The `gid` field is auto-assigned: the first `Gadget` instruction gets `gid = 1`, the
second gets `gid = 2`, and so on. Subsequent instructions reference gadgets by these
sequential IDs.

#### CheckModel Instances

```protobuf
message CheckModel {
  uint64 ctype = 1;                    // which check model type to instantiate
  uint64 gid = 2;                      // the gadget to attach to
  string tag = 3;
  optional CheckModelModifier modifier = 4;
  uint64 cid = 5;                      // auto-assigned sequentially from 1
}
```

A check model is attached to a specific gadget (by `gid`). Each gadget can have **at most
one** check model attached to it — the check model fully describes all syndrome checks for
that gadget. The `cid` is auto-assigned independently from gadget IDs.

#### ErrorModel Instances

```protobuf
message ErrorModel {
  uint64 etype = 1;                    // which error model type to instantiate
  uint64 cid = 2;                      // the check model to attach to
  string tag = 3;
  optional ErrorModelModifier modifier = 4;
  uint64 eid = 5;                      // auto-assigned sequentially from 1
}
```

An error model is attached to a specific check model (by `cid`). Unlike the one-to-one
gadget-to-check-model binding, a single check model can have **multiple** error models
attached to it. Each error model is added independently, and their errors are combined
additively in the decoding hypergraph. This is useful, for example, in **erasure decoding**:
the baseline error model captures the standard noise, and a second error model can be
attached dynamically (with modified probabilities) to represent erasure events detected in a
particular shot. The `eid` is auto-assigned independently.

---

## A Complete Example

Here is the full `Library` for **Scenario B: PrepareZ → Idle → MeasureZ**, combining all
the type definitions and program instructions from above:

```json
Library {
  description: "d=3 repetition code: PrepareZ → Idle → MeasureZ"

  port_types: [{
    ptype: 1
    name: "RepetitionCode"
    observables: [{ tag: "LX" }, { tag: "LZ" }, { tag: "S0:Z0Z1" }, { tag: "S1:Z1Z2" }]
  }]

  gadget_types: [
    {  // PrepareZ: 0 inputs, 1 output (4 obs), 0 measurements, 0 readouts
      gtype: 1, name: "PrepareZ"
      outputs: [{ ptype: 1 }]
      correction_propagation: { rows: 4, cols: 1 }           // 4×1 zero — no input obs, constant=0
      readout_propagation: { rows: 0, cols: 1 }              // 0×1 — no readouts
      logical_correction: { rows: 4, cols: 0 }               // 4×0 — no readouts
      physical_correction: { rows: 4, cols: 0 }              // 4×0 — no measurements
    },
    {  // Idle: 1 input (4 obs), 1 output (4 obs), 2 measurements, 0 readouts
      gtype: 2, name: "Idle"
      measurements: [{ tag: "s0" }, { tag: "s1" }]
      inputs: [{ ptype: 1 }]
      outputs: [{ ptype: 1 }]
      correction_propagation: { rows: 4, cols: 5,
        i: [0, 1], j: [0, 1] }                               // logical obs pass through; stabilizer rows are zero
      readout_propagation: { rows: 0, cols: 5 }              // 0×5 — no readouts
      logical_correction: { rows: 4, cols: 0 }               // 4×0 — no readouts
      physical_correction: { rows: 4, cols: 2,
        i: [2, 3], j: [0, 1] }                               // s₀ flips S0, s₁ flips S1
    },
    {  // MeasureZ: 1 input (4 obs), 0 outputs, 1 readout
      gtype: 3, name: "MeasureZ"
      measurements: [{ tag: "m0" }, { tag: "m1" }, { tag: "m2" }]
      inputs: [{ ptype: 1 }]
      readouts: [{ tag: "logical_Z", measurement_indices: [0, 1, 2] }]
      correction_propagation: { rows: 0, cols: 5 }           // 0×5 — no output observables
      readout_propagation: { rows: 1, cols: 5, i: [0], j: [1] }  // input LZ correction flips readout
      logical_correction: { rows: 0, cols: 1 }               // 0×1 — no output observables
      physical_correction: { rows: 0, cols: 3 }              // 0×3 — no output observables
    }
  ]

  check_model_types: [
    {  // IdleBoundaryChecks (ctype 1)
      ctype: 1, name: "IdleBoundaryChecks", gtype: 2
      checks: [
        { tag: "c0", measurements: [{ measurement_index: 0 }] },
        { tag: "c1", measurements: [{ measurement_index: 1 }] }
      ]
    },
    {  // MeasureChecks (ctype 2)
      ctype: 2, name: "MeasureChecks", gtype: 3
      remote_gadgets: [{ input: 0 }]
      checks: [
        { tag: "c0", measurements: [
            { measurement_index: 0 }, { measurement_index: 1 },
            { remote_gadget: 0, measurement_index: 0 }
        ]},
        { tag: "c1", measurements: [
            { measurement_index: 1 }, { measurement_index: 2 },
            { remote_gadget: 0, measurement_index: 1 }
        ]}
      ]
    }
  ]

  error_model_types: [
    {  // IdleErrors (etype 1) — see detailed definition above
      etype: 1, name: "IdleErrors", ctype: 0
      remote_check_models: [{ output: 0 }]
      errors: [
        { tag: "meas_s0", checks: [{ check_index: 0 }, { remote_check_model: 0, check_index: 0 }], probability: 0.03 },
        { tag: "meas_s1", checks: [{ check_index: 1 }, { remote_check_model: 0, check_index: 1 }], probability: 0.03 },
        { tag: "data_q0", checks: [{ check_index: 0 }], residual: [0], probability: 0.03 },
        { tag: "data_q1", checks: [{ check_index: 0 }, { check_index: 1 }], residual: [0], probability: 0.03 },
        { tag: "data_q2", checks: [{ check_index: 1 }], residual: [0], probability: 0.03 }
      ]
    },
    {  // MeasureErrors (etype 2)
      etype: 2, name: "MeasureErrors", ctype: 2
      errors: [
        { tag: "data_q0", checks: [{ check_index: 0 }], readout_flips: [0], probability: 0.03 },
        { tag: "data_q1", checks: [{ check_index: 0 }, { check_index: 1 }], readout_flips: [0], probability: 0.03 },
        { tag: "data_q2", checks: [{ check_index: 1 }], readout_flips: [0], probability: 0.03 }
      ]
    }
  ]

  program: [
    // Step 1: Create gadgets with connectivity
    { gadget: { gtype: 1 } },                                       // gid=1 (PrepareZ)
    { gadget: { gtype: 2, connectors: [{ gid: 1, port: 0 }] } },    // gid=2 (Idle)
    { gadget: { gtype: 3, connectors: [{ gid: 2, port: 0 }] } },    // gid=3 (MeasureZ)

    // Step 2: Attach check models to gadgets
    { check_model: { ctype: 1, gid: 2 } },                          // cid=1 (IdleBoundaryChecks on Idle)
    { check_model: { ctype: 2, gid: 3 } },                          // cid=2 (MeasureChecks on MeasureZ)

    // Step 3: Attach error models to check models
    { error_model: { etype: 1, cid: 1 } },                          // eid=1 (IdleErrors on IdleBoundaryChecks)
    { error_model: { etype: 2, cid: 2 } }                           // eid=2 (MeasureErrors on MeasureChecks)
  ]
}
```

To extend this to **Scenario C** (PrepareZ → Idle → Idle → MeasureZ), we would add an
`IdleBulkChecks` check model type (see [the earlier example](#example-idleboundarychecks-and-idlebulkchecks))
and extend the program:

```json
  // Additional type definition:
  // CheckModelType { ctype: 3, name: "IdleBulkChecks", gtype: 2, ... }

  program: [
    { gadget: { gtype: 1 } },                                       // gid=1 (PrepareZ)
    { gadget: { gtype: 2, connectors: [{ gid: 1, port: 0 }] } },    // gid=2 (first Idle)
    { gadget: { gtype: 2, connectors: [{ gid: 2, port: 0 }] } },    // gid=3 (second Idle)
    { gadget: { gtype: 3, connectors: [{ gid: 3, port: 0 }] } },    // gid=4 (MeasureZ)

    { check_model: { ctype: 1, gid: 2 } },                          // IdleBoundaryChecks on first Idle
    { check_model: { ctype: 3, gid: 3 } },                          // IdleBulkChecks on second Idle
    { check_model: { ctype: 2, gid: 4 } },                          // MeasureChecks on MeasureZ

    { error_model: { etype: 1, cid: 1 } },
    { error_model: { etype: 1, cid: 2 } },                          // same error model type, different context!
    { error_model: { etype: 2, cid: 3 } }
  ]
```

Notice how `IdleErrors` (etype 1) is reused for both Idle instances — the relative
addressing in its `RemoteCheckModel { output: 0 }` automatically resolves to the correct
successor.

---

## Advanced Features

### Modifiers: Runtime Reconfiguration

Modifiers allow per-instance customization without creating new types. They are optional
fields on instance messages.

#### GadgetModifier and BitMatrixModifier

A `GadgetModifier` can modify any of the four Pauli frame matrices at instance creation:

```protobuf
message GadgetModifier {
  optional BitMatrixModifier correction_propagation_mod = 1;
  optional BitMatrixModifier readout_propagation_mod = 2;
  optional BitMatrixModifier logical_correction_mod = 3;
  optional RemoteConditionalCorrection remote_conditional_correction = 4;
  optional BitMatrixModifier physical_correction_mod = 5;
}
```

Each matrix modification uses a `BitMatrixModifier`:

```protobuf
message BitMatrixModifier {
  optional BitMatrix toggle = 1;    // XOR sparse bits into the existing matrix
  optional BitMatrix overwrite = 2; // replace the entire matrix
}
```

The two modes are applied **sequentially**: toggle first, then overwrite. In practice, you
use one or the other — not both — since an overwrite would discard any preceding toggle:
- **Small targeted changes:** use `toggle` to flip specific bits in the type's existing
  matrix (e.g., insert a virtual Pauli gate by flipping the constant column)
- **Full replacement:** use `overwrite` to completely replace the matrix with a new one

**Use case:** A classical control program reads a mid-circuit measurement, decides to apply
a logical Pauli correction, and modifies the next gadget's `correction_propagation` matrix
via a `GadgetModifier` — without pre-defining all possible configurations as separate
gadget types.

#### CheckModelModifier: Rerouting Remote Gadgets

```protobuf
message CheckModelModifier {
  message RerouteRemoteGadget {
    uint64 remote_gadget_index = 1;
    CheckModelType.RemoteGadget value = 2;
  }
  repeated RerouteRemoteGadget reroute_remote_gadgets = 4;
}
```

This overrides specific entries in the check model type's `remote_gadgets` list. Each
`RerouteRemoteGadget` specifies an index and a replacement `RemoteGadget` definition.

The list is **dynamically extensible** — if the index exceeds the type's original list
length, new entries are added. Only changed entries need to be specified (sparse
representation).

**Use case:** A check model type defines a remote gadget with `measurement_bias = 0`,
suitable for connecting to port 0 of the predecessor. If at runtime the gadget is actually
connected to port 1 (where measurements start at offset 4), a modifier can reroute that
remote gadget entry to use `measurement_bias = 4` — reusing the same check model type
without creating a new one.

#### ErrorModelModifier: Rerouting and Probability

```protobuf
message ErrorModelModifier {
  ProbabilityModifier probability_modifier = 1;

  message RerouteRemoteCheckModel {
    uint64 remote_check_model_index = 1;
    ErrorModelType.RemoteCheckModel value = 2;
  }
  repeated RerouteRemoteCheckModel reroute_remote_check_models = 2;
}
```

Analogous to `CheckModelModifier`, but also supports probability modification (see
[Probability Modifier](#probability-modifier-dynamic-noise-models)).

### Remote Conditional Correction: Non-Blocking Feed-Forward

The `RemoteConditionalCorrection` mechanism enables the decoder to handle feed-forward
dependencies **internally**, without blocking the quantum computer:

```protobuf
message RemoteConditionalCorrection {
  message RemoteReadout {
    uint64 gid = 1;            // which gadget
    uint64 readout_index = 2;  // which readout in that gadget
  }
  repeated RemoteReadout remote_readouts = 1;
  BitMatrix correction = 2;   // |output_observables| × |remote_readouts|
}
```

| Field             | Purpose                                                                                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `remote_readouts` | Ordered list of readouts from remote (previously executed) gadgets, each identified by `(gid, readout_index)`. The order defines column indices in the correction matrix |
| `correction`      | A $\mathrm{GF}(2)$ matrix mapping remote readout values to corrections on the current gadget's output observables                                                        |

At runtime, the decoder:
1. Looks up each remote gadget's error-corrected readout by `gid` and `readout_index`
2. Assembles the readout values into a vector $\mathbf{m}$
3. Computes $R \cdot \mathbf{m}$ (matrix-vector multiply over $\mathrm{GF}(2)$)
4. XORs the result into the output observable residual

**Why this enables non-blocking execution:** Logical Pauli gates are purely virtual — they
affect only the classical Pauli frame, not the physical qubits. The correction can be
computed retroactively when the remote readout becomes available. The quantum computer
continues executing without waiting for the decoder. No idle syndrome extraction rounds are
needed.

### Probability Modifier: Dynamic Noise Models

Error probabilities defined in `ErrorModelType` represent the baseline noise model. At
runtime, they can be modified per instance:

```protobuf
message ProbabilityModifier {
  repeated double probabilities = 1;        // dense overwrite
  repeated uint64 sparse_indices = 2;       // sparse overwrite indices
  repeated double sparse_probabilities = 3; // sparse overwrite values
}
```

The two modes are applied **sequentially**:

1. **Dense overwrite** (`probabilities`): A vector replacing all error probabilities. Length
   must be 0 (no change) or exactly equal to the number of errors.
2. **Sparse overwrite** (`sparse_indices` + `sparse_probabilities`): A list of
   `(error_index, new_probability)` pairs applied after the dense overwrite. The two arrays
   must have equal length.

**Use cases:**

- **Soft-information decoding:** Quantum hardware provides analog measurement confidence
  levels. Convert these to modified error probabilities and pass via the dense vector.
- **Erasure decoding:** When a qubit is detected as leaked and reset, set associated error
  probabilities to 0.5 (maximum entropy — the measurement carries no information). Use
  dense mode for bulk marking, sparse mode for targeted adjustments.
- **Per-shot noise calibration:** Physical noise drifts over time. Update probabilities
  every shot without redefining error model types. The sparse mode is efficient when only a
  few probabilities change.

### Dual Addressing: Relative vs. Absolute

Remote references (`RemoteGadget` and `RemoteCheckModel`) support two addressing modes that
**coexist** in the same message:

**Relative addressing** (port traversal):
```protobuf
RemoteGadget { input: 0 }                                       // one hop via input port
RemoteGadget { output: 0, previous_remote_gadget: 0 }           // chain: start from a previous remote, then hop via output
```
- Portable: does not depend on runtime-assigned IDs
- Self-documenting: explicitly encodes topology relationships
- Generated by **offline/static compilers**

**Absolute addressing** (direct ID):
```protobuf
RemoteGadget { absolute_gid: 42 }
RemoteCheckModel { absolute_cid: 17 }
```
- $O(1)$ resolution: direct indexed lookup, no traversal
- Context-dependent: requires knowledge of runtime-assigned IDs
- Generated by **JIT compilers** that operate during circuit execution

**Resolution algorithm:** The runtime checks for the absolute field first. If present, it
uses the direct ID. Otherwise, it falls back to relative port traversal. This allows:

- **Offline-compiled libraries** to use relative addressing for portability
- **JIT-compiled programs** to use absolute addressing for maximum performance
- **Mixed-mode programs** where some references are relative and others are absolute

---

## Summary

deq-bin is a three-layer type-instance system for describing QEC decoding hypergraphs:

1. **Types** (Library): `GadgetType` → `CheckModelType` → `ErrorModelType`, compiled once
2. **Instances** (Program): a stream of `Instruction` messages assembling types into a
   specific circuit, with optional modifiers for runtime customization
3. **Remote references**: parameterized cross-piece connections using port traversal (or
   direct IDs), with bias offsets for template reuse

This separation means:
- The same types can be reused across arbitrarily different circuits
- The decoding hypergraph is constructed incrementally as instructions arrive
- Any decoder implementing deq-bin can decode any compatible circuit
- The format is simple enough for hardware implementation (integer parameters, indexed
  lookups) and portable enough for software decoders across all quantum platforms

### Toward Hardware Decoder Accelerators

deq-bin is the first standard specification that enables **systematic development of
hardware decoder accelerators** capable of supporting dynamic logical circuits. Under
certain constraints — for example, if every logical instruction includes syndrome extraction
at its beginning — the decoding hypergraph structure does not change across instances, up to `check_bias` values and check definitions (which are simple XOR
computations compared to the decoding itself). In this regime, the deq-bin library can be
generated offline, and a hardware design can be **generated and optimized directly from the
library** using hardware description frameworks like
[SpinalHDL](https://github.com/SpinalHDL/SpinalHDL). For example,
[Micro Blossom](https://github.com/yuewuo/micro-blossom) demonstrates this approach: it
generates MWPM hardware decoders for arbitrary decoding graphs from an external
specification.
Yet, Micro Blossom has not supported deq-bin to allow decoding across multiple gadgets.

For the first time, hardware decoder developers have a **concrete, formal specification** to
build against — rather than relying on ad-hoc knowledge about specific codes and circuits.
This is the path from software prototyping to production hardware: the same deq-bin library
that drives simulation today can drive FPGA synthesis tomorrow.

