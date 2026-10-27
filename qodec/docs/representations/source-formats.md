# Source formats

A gadget can store a Stim circuit as text:

```yaml
circuit: {format: stim, source: "M 0 1 2"}
```

Or it can reference a file beside the gadget document:

```yaml
circuit: ./measure.stim
```

In the first form, `format` identifies the language and marks the string as
inline source. In the second, the file extension identifies the language.
Source language and artifact type are separate: a manifest's `gadgets` entries
always reference YAML gadget documents, not raw circuit files.

## Storage and interpretation

Loading and saving preserve source without parsing its calls. Any explicit
format tag can be stored, including a language for which qodec has no parser:

```yaml
circuit: {format: custom-ir, source: "custom circuit text"}
```

Calling `Circuit.calls()` or reading `Circuit.blocks` or `Circuit.readouts` requires
interpretation and may fail on a successfully loaded circuit. The C ABI tries
to parse each circuit while building its projection and records failures in
`circuit.error`; the rest of the qodec remains readable.

| Source | Stored by load/save | Parsed by qodec |
| --- | --- | --- |
| Inline YAML call list | Yes, including unfinished calls | Yes, subject to the call syntax and target ISA |
| Stim text | Yes, including unsupported syntax | Optional Python adapter via `qodec[parsers]`; no default Rust/C adapter |
| Other tagged text, such as OpenQASM or a custom language | Yes | An explicitly registered parser is required |

A tag does not install a parser or certify that the text is valid in that
language. Nor does accepting text provide object-level interoperation with
Cirq, QIR, or deq. See [tool connections](../ecosystem.md) for those distinctions.

## Inline YAML

Inline YAML represents an ordered list of calls into the circuit's instruction
set. A list can be written directly, without a `format` tag. A YAML source
*string* needs `format: yaml`; otherwise the string is a path.

The call object separates `operands`, `arguments`, and per-call `select`
patterns. Its list shorthand contains operands followed by named arguments;
it has no selection field. Arguments can forward parameters or refer to
preceding circuit readouts. See [Inline YAML calls](yaml.md#inline-yaml-calls)
for the syntax and value types.

## Registering a parser

In Python, `qodec.register(custom_parser, format="my-format")` supplies the parser
used by `circuit.calls()`, `blocks`, and `readouts` for that `effective_format`.
Use `circuit.calls(parser=custom_parser)` to override one call without changing
registration.

The callback receives `(source, instruction_set)` with a snapshot of the target
ISA, and returns a sequence of `InstructionCall` values. It must preserve
execution and readout positions or raise an error. The binding converts complete
call values to Rust, including argument types and selection patterns. Calls are
unconditional; `if`/`unless` guards belong to the instruction's action steps.
Rust checks mnemonic lookup; audit checks argument uses, selection, and record bounds
for every successfully parsed format. Neither proves a custom translation correct.

Each access parses the current source again. Parsers should return the same
result for the same source and declarations, without mutating the ISA. Configure
registrations before starting an analysis. Editing returned calls does not edit
source; parser configuration is not serialized or included in equality.

Rust holds the only registry. Python registration wraps the callable in a native
parser closure. Registrations from either language replace the same entry, so
the latest wins. Calls already in progress keep their selected parser; callbacks
and replaced callback destructors run outside the registry lock.

YAML is registered directly in Rust. The supplied Stim adapter registers through
the binding at Python module load when Stim is available. An unknown format fails
interpretation. Python callbacks acquire interpreter access when invoked, including
from native threads. They require the registering interpreter to remain running.
Shutdown releases the callable, and later invocations return an error. Exceptions
retain their Python type on Python-originated calls; native callers receive the
error description. A separately loaded native library can still have its own
registry; sharing a process does not merge separately linked Rust instances.

Rust uses `qodec::register(parser, "my-format")`; its callback takes
`(&str, &InstructionSet)` and returns `Result<Vec<InstructionCall>, String>` and
must be `Send + Sync + 'static`. The `calls_with`, `blocks_with`, and
`readouts_with` methods provide one-use overrides. The pure C ABI has no
callback-registration function, but a C projection using the same Rust instance
uses that registry too.

## Stim parser limits

When building from source, run `python -m maturin develop --release --extras parsers`
from `qodec/bindings/python` in the repository. For a published release, use
`python -m pip install "qodec[parsers]"` before importing qodec. The extra
currently installs Stim; YAML needs no extra. The adapter is registered at module load
when Stim is available; it imports
the official Stim parser when interpreting source. An already installed compatible
Stim package works too; extras install dependencies, not runtime feature flags.
Without Stim at module load, there is no Stim registration and interpretation
reports a missing parser. Loading and saving still work. Restart the Python
process or notebook kernel after installing Stim to initialize its registration.

- One- and two-qubit gates with plain integer targets become individual calls.
  `CX 0 3 1 3` becomes `CX 0 3` and `CX 1 3`.
- Fixed `REPEAT` blocks are expanded, including nested repeats, up to one million
  calls. Larger expansions fail; repeats containing only annotations add no calls.
  Nesting beyond Python's recursion limit reports a parser error.
- Stim canonical gate names select matching ISA instructions. If only an alias
  is declared, it must be unique. Target block widths and measurement counts
  must match; extra declared flags would change the record and are rejected.
  Ordinary gate actions are taken from the ISA; the adapter does not prove that
  a declaration implements the standard Stim gate bearing that name.
- Noise-free `MPAD` and `MPAD(0)` append constant bits through declared
  `MPAD0` and `MPAD1` instructions, as described below. Padding counts toward
  the expansion limit and occupies ordinary readout positions.
- Nonzero MPAD noise, record-controlled gates, inverted targets, Pauli-product
  targets, and other gate arguments are rejected. The latter includes
  measurement-noise arguments even when zero; they are not interpreted as
  ordinary argument-free gates.
- `DETECTOR`, `OBSERVABLE_INCLUDE`, `TICK`, and coordinate annotations do not
  become calls. They are retained in source, not converted into gadget equations.

Parsing Stim successfully does not mean its operations are supported by this
adapter or a downstream analysis. Native Rust Stim support is deferred. The
original source remains available to another parser or Stim itself.

### Measurement padding

For `M 3; MPAD 0 1; M 5`, the record contains the first measurement, zero, one,
and the second measurement. Padding adds no qubit operands. The adapter translates
each constant into a call to `MPAD0` or `MPAD1`; it never drops the bit or creates
an undeclared instruction. Add these entries under the target ISA's `instructions`:

```yaml
- mnemonic: MPAD0
  description: Append a constant zero without using qubits.
  action: [observe: "I"]
- mnemonic: MPAD1
  description: Append a constant one without using qubits.
  action: [observe: "-I"]
```

The identity observable has eigenvalue +1, so it reports zero; negative identity
reports one. Each declaration must contain exactly that one observation, with
no operands, parameters, or flags. Only the constants used by the source need
declarations. Missing or incompatible declarations fail interpretation without
changing the ISA. The [shared physical ISA](../../examples/stim.isa.yaml) includes
both instructions.

## What belongs in the gadget

The instruction's action states the operation the gadget must implement.
The gadget's encodings place the code blocks on circuit operands. Its `checks`
and `readouts` relate measurement bits to input and output encoding signs.
Those declarations remain necessary even when a source language has related
annotations.

For example, a Stim `DETECTOR` can combine measurement-record bits, but it does
not identify a qodec input encoding's stabilizer. An adapter must account for
that sign when translating a gadget check. deq also has gadget and decoding
declarations; translating them requires agreement on their meanings, not just
copying a circuit string. See the [gadget guide](../concepts/gadget.md) and
[qodec with deq](../ecosystem.md#deq-the-decoding-surface).
