# The YAML representation

Current manifests use `schema_version: 1`. An explicit version must match the
loader; omission means the loader's current version. See the
[compatibility contract](../../CHANGELOG.md#compatibility-contract).

This page describes how to write a qodec in YAML. The [model](../concepts/model.md)
is independent of the file format; see the
[language APIs](../concepts/model.md#representations) for working with it in memory.
The same documents can also be written as JSON. Their structure is defined by the
[artifact schemas](../../schemas/).

Use the schemas from the same source revision as the package. They describe
individual artifact shapes; they are not a substitute for loading referenced
files or auditing protocol behavior. The loader does not run a JSON Schema
validator. See [Validation](../concepts/validation.md) for that boundary.

A qodec can be stored in either form:

- **Referenced files**: a YAML manifest and its artifacts, linked by relative path.
- **Single-file bundle**: several YAML documents in one file, each stored under
  the path used to reference it.

## Loading

Pass a manifest or bundle file to `Qodec.load`. Filenames and folders are your
choice. Each path is relative to the document containing it, including that
document's path inside a bundle. Only referenced artifacts are loaded.

`Qodec.save` takes a destination directory and writes the manifest there under
`manifest_filename`.

Directory saves reuse unchanged files outside the original manifest's directory,
using absolute references to the same files. Edited external artifacts are copied
locally, and files inside the original directory are copied normally. A reused
external file must still match its loaded text; otherwise saving fails before
writing. External files are never overwritten. Bundles copy current values without
reusing external files.

Compatible local artifact paths are preserved. Moving the manifest above the
destination (for example, to `../entry`) relocates local artifacts alongside it.

### Multi-level chains

The manifest lists layers in order, each with its ISA and (except for the bottom
layer) its codes and gadgets. For a three-level chain:

```yaml
# qodec.yaml
layers:
  - instruction_set: logical.isa.yaml
    codes:   {patch: surface.code.yaml}
    gadgets: {idle: logical_idle.gadget.yaml, measure_z: measure_z.gadget.yaml}
  - instruction_set: routing.isa.yaml
    codes:   {patch: routing.code.yaml}
    gadgets: {idle: routing_idle.gadget.yaml}
  - instruction_set: physical.isa.yaml          # floor: an ISA, no gadgets
```

Each layer lists the gadgets that implement its instructions in the layer below.
The logical-to-routing gadgets belong to the first layer; the routing-to-physical
gadgets belong to the second. See [the model](../concepts/model.md#the-lowering-chain).

### The single-file bundle

The first document contains one entry: its key is the manifest's path within the
bundle, and its value is the manifest. Later documents hold the referenced
artifacts, each under its own path. For example:

```yaml
protocol: ...             # manifest, first document
---
repetition3.isa.yaml: ...  # logical ISA
---
stim+rz.isa.yaml: ...      # physical ISA
---
repetition3.code.yaml: ... # the [[3,1,1]] code
---
measure_z.gadget.yaml: ... # a gadget
```

An entry referenced as circuit source can hold raw text in a YAML block scalar:

```yaml
measure_z.stim: |
  M 0 1 2
```

This is convenient for sharing a whole protocol as a single file. The
[`repetition3` example](../../examples/repetition3/) is authored this way.

## Shorthands and alternative forms

Several qodec fields accept shorter forms.

- **Bare circuit path.** When the circuit needs only a source:

  Shorthand:

  ```yaml
  circuit: ./idle.stim
  ```

  Equivalent standard form:

  ```yaml
  circuit:
    source: ./idle.stim
  ```

- **Bare circuit call list.** The `source` wrapper can also be omitted for
  inline calls:

  Shorthand:

  ```yaml
  circuit:
    - M: [0]
  ```

  Equivalent standard form:

  ```yaml
  circuit:
    source:
      - M: [0]
  ```

  Both circuit shorthands are written back on save when only `source` is set.
  Use the [object form](../concepts/gadget.md#circuit-object) for additional fields.

- **Inline YAML call.** A call with operands and arguments can use a list:

  Shorthand:

  ```yaml
  - rotate_z: [0, theta: 1.5708]
  ```

  Equivalent standard form:

  ```yaml
  - rotate_z:
      operands: [0]
      arguments: {theta: 1.5708}
  ```

  Saving preserves the authored source; it does not convert calls between these
  forms. See [Inline YAML calls](#inline-yaml-calls) for the call fields.

- **Single-observable `observe`.** One observable can be written without a list:

  Shorthand:

  ```yaml
  observe: Z_0
  ```

  Equivalent standard form:

  ```yaml
  observe: [Z_0]
  ```

  A single observable is written in shorthand on save.

- **Single-operator `stabilize`.** One operator can be written without a list:

  Shorthand:

  ```yaml
  stabilize: Z_0
  ```

  Equivalent standard form:

  ```yaml
  stabilize: [Z_0]
  ```

  This is always written as a list on save.

## Circuit sources

A [gadget](../concepts/gadget.md)'s circuit calls instructions in the target ISA.

### Source formats

A circuit source can use these formats:

| Format       | File extension | Notes                                                              |
| ------------ | -------------- | ------------------------------------------------------------------ |
| Stim         | `*.stim`       | Referenced by path, or inlined with `format: stim`. |
| OpenQASM 3   | `*.qasm`       | Referenced by path, or inlined with `format: openqasm`. |
| Inline YAML  | (inline)       | A list of calls embedded in the gadget. |

The value of `circuit.source` determines how it is read:

- A **list** contains inline YAML calls.
- A **string without `format`** is a relative file path.
- A **string with an explicit `format`** is the circuit text itself. The tag may
  name `yaml`, `stim`, `openqasm`, or a language without a qodec parser.

Loading and saving preserve circuit text and inline lists without interpreting
their calls. Unknown instructions and malformed circuits remain available for
inspection and repair. Audit checks call validity when a parser is available.

qodec loads and saves OpenQASM source without parsing it. Its calls require a
registered parser. Loading does not check calls in any source language.

`Circuit.calls()` uses the shared Rust registry and resolves its instructions
on request. YAML is registered directly in Rust.
Python Stim support is registered on import when the optional dependency is
available through `qodec[parsers]`. Native Rust Stim support is deferred. See
[parser registration](source-formats.md#registering-a-parser).
Accessors requiring parsed calls can fail even after a successful load. See
[Validation](../concepts/validation.md#circuit-sources).

The circuit's calls append their observe outcomes and then their flags to an
ordered record. Hidden reset outcomes do not enter it. `circuit.readouts[i]`
refers to bit `i`, counting from zero across all calls. Gadget equations use
these references in
[parity equations](../concepts/gadget.md#parity-equations-property-path-references).

### Parity bits and frames

Checks, readouts, and frames accept quoted references and integer literals `0`
or `1`. Each list is an XOR expression: `1` flips its value and `0` leaves it
unchanged. Booleans, floats, and quoted `"1"` are not literal parity bits.

```yaml
frames:
  "out[0].z[0]": ["circuit.readouts[0]", 1]
```

This flips the output's logical Z sign when the recorded bit is zero. It is an
additional correction, not a declaration of the total output frame. Missing
entries and `[]` add no correction. Frame values may use circuit readouts,
literal bits, and acyclic readout aliases that resolve to those terms. Unlike
checks and readouts, they may not use `in[...]` or `out[...]` signs, even through
aliases. See [Frames](../concepts/gadget.md#frames).

### Inline YAML calls

An inline YAML source is a list of calls into the target ISA. Each call has one
key, the instruction name. The standard form puts `operands`, `arguments`, and
`select` in a call object:

```yaml
circuit:
  instruction_set: ./stim.isa.yaml
  source:
    - R: {operands: [0]}
    - R: {operands: [1]}
    - CX: {operands: [0, 1]}
    - M: {operands: [0]}
```

`operands` lists the blocks the instruction acts on, in the order declared by
`in:`/`out:`. Each is a non-negative integer or a string name, never a Boolean.
`arguments` maps parameter names to values. `select` describes the expected flag
values, as explained [below](#per-call-select). All three fields are optional and
default to empty; unknown call fields are rejected.

For example, `- rotate_z: {operands: [0], arguments: {theta: 1.5708}}` acts on
block `0`, using `1.5708` as the rotation angle. The block is the operand.
The instruction declares a parameter named `theta`, with type `number`.
The call supplies the argument `1.5708`, the value to use for that angle.

Argument values can be booleans (`true` or `false`), numbers, strings, lists of
non-negative integers, or lists of strings. Empty lists are accepted. For example,
`- configure: {arguments: {enabled: true}}` and
`- configure: [enabled: true]` both supply `true` for a parameter named `enabled`
declared with type `boolean`.

Booleans are distinct from integers: `true` is not `1`, and `false` is not `0`.
Quoted `'true'` and `'false'` remain strings. Boolean lists, nulls, maps, and
mixed-type lists are not argument values. The instruction declares the type
of each parameter.

The [list shorthand](#shorthands-and-alternative-forms) puts operands first,
followed by named arguments. An argument map may contain several pairs. Every
named pair is an argument, with no reserved names, including `operands`,
`arguments`, and `select`. Duplicate argument names and operands after arguments
are errors. Per-call selection requires the call object's `select` field.

A null call, such as `- tick:` or `- tick: null`, is an empty shorthand, equivalent
to `- tick: []` or `- tick: {}`.

### Threading readouts into `bit` parameters

To use an earlier measurement result in a later call, pass its reference to a
parameter declared as `bit`. `circuit.readouts[0]` means the first recorded bit,
`circuit.readouts[1]` the second, and so on. Counting continues across calls; it
does not restart for each instruction.

A slice selecting exactly one position, such as `circuit.readouts[0:1]`, is
also accepted. Parsed calls expose it as `circuit.readouts[0]`; the source
retains its authored spelling. Empty or multi-position selectors are rejected,
including unions that repeat the same position.

A called gadget's flags are bits in this same record and can be passed the same
way. There is no separate flag record. A consumer must support the called
instructions and their bit arguments to interpret this flow; loading preserves
it without executing or verifying it.

```yaml
circuit:
  source:
    - measure_z: [1]                             # emits circuit.readouts[0]
    - correct_z: [0, c: "circuit.readouts[0]"]   # argument for bit parameter c
```

Inside a bracketed YAML list, quote the reference as shown above so its brackets
are read as part of the reference, not as another YAML list.

Only `bit` parameters accept readout references as arguments. Supplying one for
a parameter of another type is an error.

### Per-call `select`

An inline YAML call object may include `select:` to describe its expected flag
values without noise. It applies to that call, not to the gadget as a whole.

The value is always a list of patterns. All flag values within one pattern must match;
at least one pattern in the list must match. A key names a flag declared by the
called instruction, such as `reject`, or refers to it by position as `flags[i]`.
An undeclared flag is an error. Omitting `select` or using `[]` imposes no
selection constraint. A bare map, such as `select: {reject: 0}`, is invalid.

Downstream tools, such as decoders and samplers, may use this hint for selection
without deriving the allowed values from the circuit.

YAML lets a single-flag map omit its braces inside a list: `[reject: 0]` means
`[{reject: 0}]`. A pattern with several flags needs braces to keep them together:

```yaml
select: [reject: 0]             # one single-flag pattern
select: [reject: 0, leak: 0]    # two single-flag patterns OR-ed together
select: [{reject: 0, leak: 1}]  # one AND pattern (explicit braces required)
```

Putting it together:

```yaml
circuit:
  source:
    - prepare_x_all:
        operands: [3]
        select: [reject: 0]
```

This says that the `prepare_x_all` call's `reject` flag is 0 without noise. Flag
values must be the integers `0` or `1`, not booleans. Inside `arguments`, `select`
is an ordinary parameter name, independent of this selection field.

### Forwarded parameters

A gadget can pass a value from its instruction into its circuit. For example,
`parameter_bindings: {theta: circuit.source.angle}` makes the value supplied for
the instruction's `theta` available as `angle` in the source. The
`circuit.source.` prefix belongs to the on-disk form; in Rust and Python the same
binding reads `{"theta": "angle"}`.

Use that source name as a bare name in an inline YAML call:
`- rotate_z: [0, theta: angle]`. The angle now comes from the original instruction
call. Writing `theta: 1.5708` would use that fixed value instead.
See [Parameterized gadgets](../concepts/gadget.md#parameterized-gadgets).

Use inline YAML for the forwarding shown here; a Stim or OpenQASM source can
instead contain fixed values. Those languages also support measurement-based
control: Stim uses `rec[-i]` references, and OpenQASM can condition gates on
classical bits such as `c[i]`.

### Stim and OpenQASM sources

Stim and OpenQASM circuits can be stored in separate files or written inline.
For their syntax, see the [Stim documentation](https://github.com/quantumlib/Stim)
and the [OpenQASM 3 specification](https://openqasm.com/).

To write one inline, put the circuit text in `source:` and set `format: stim`
or `format: openqasm`:

```yaml
circuit:
  instruction_set: ./stim.isa.yaml
  format: stim
  source: |
    R 3 4
    CX 0 3 1 3
    M 3 4
```

The `format` tag is required for inline text; without it, the string is read as a
file path. Neither language has qodec's per-call `select` or named flag declarations.
A gadget can still refer to their measurement results through
`circuit.readouts[i]` in its `checks` and `readouts`. See
[Circuit object](../concepts/gadget.md#circuit-object) for details.
