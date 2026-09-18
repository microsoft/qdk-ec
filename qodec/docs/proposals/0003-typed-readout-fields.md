# 0003 - Typed readout fields

**Status:** deferred

## Context

A measurement may return a bit and a loss indication; a decoder may return a
bit and a confidence score. qodec has bit-valued readouts and flags, but no
declaration attaching such information to a particular readout.

A decoder can treat a known error location differently from an unknown one.
If that indication is lost between decoding layers, the next decoder cannot
use it; if attached to the wrong bit, it can guide the wrong correction.
Declaring the association in qodec would let a protocol say which information
must accompany each output. Fields used only inside one runtime need no such
declaration, so the proposal needs a case where tools exchange or reference them.

## Benefits

- Let upper-layer decoders use loss and confidence information without guessing
  which readout it belongs to.

## Drawbacks

- The fields may depend on the chosen noise model or decoder, not the protocol.
- Tools need rules for producing, passing, and interpreting the extra values.
- Ignoring a required field can invalidate the decoding result.
- Boolean fields overlap with existing flags.

## Proposal

Explore an optional instruction-set `readout` schema: a sparse map from authored
field names to types, excluding the implicit hard bit. This is proposed syntax:

```yaml
readout:
  lost: bool
  score: float
```

An omitted schema declares no auxiliary fields. It retains bit-only behavior.
Reserve `bit` so a field cannot redefine the unsuffixed readout. Whether this
schema applies to an entire ISA or varies by instruction or outcome remains open.

### Addressing and roles

- `circuit.readouts[i]` continues to address the hard bit, regardless of how
  many auxiliary fields are declared.
- A proposed suffix such as `circuit.readouts[i].lost` identifies attached
  information, not a second record entry that shifts existing indices.
- Hard-bit checks and readout equations retain their existing XOR semantics.
  Real-valued fields are not parity terms.
- Do not implicitly add truth tables to flags. Current flags are parity
  readouts. A Boolean field used by a flag would need a separately specified
  interpretation and an ideal-zero justification; its Boolean type alone does
  not establish that it is a valid flag.

### Static declarations and runtime values

qodec stores field declarations and references, not sampled values. Runtimes
produce and pass those values; decoders interpret them. A model `Readout`
therefore does not gain Boolean coercion to a measurement result.

## Alternatives

- Keep the fields in runtime configuration when no protocol artifact needs them.
- Use flags for ideally zero Boolean signals. This needs no new type, but does
  not attach the flag to an outcome or represent a real-valued score.
- Declare `bit` explicitly: `readout: {bit: bool, lost: bool}`. This alternative
  needs a rule preserving the meaning of bare references when fields are added.
- Use metadata: adequate for experiments, but without a shared type contract.

## Discussion

### Names and scope

The sketch adds one field name, two type labels, and one reserved name:

- `readout` describes one readout's structure; `readouts` would suggest a list.
- `bool` names a Boolean value; `flag` would imply ideal-zero behavior.
- `float` names a real value; `confidence` would prescribe its meaning.
- `bit` reserves the hard outcome; `value` would not distinguish it from extras.

`lost` and `score` are example names, not built-ins. Binding APIs remain open.

A runtime can carry side information without qodec declaring it. The case for
this proposal needs an artifact or protocol requirement that names the field,
not merely a decoder that uses extra data.

Boolean values may use extra bit records; real values need another representation.
Neither may shift the existing readout indices. Field references may require
explicit circuit interpretation, while loading must still preserve unparsed drafts.
Rust, schemas, bindings, and consumer support need to agree on the chosen contract.

## Open Questions

- Which concrete protocol requires outcome-associated fields rather than
  runtime configuration or existing flags?
- Should the schema be ISA-wide, per instruction, or per outcome?
- Should the first version support only Boolean heralds or also real values?
- Which artifact contexts may reference fields, and which Boolean uses can be
  expressed without expanding the linear parity language?
- How are missing fields, unsupported consumers, and required payloads reported?
- Who defines the units and calibration semantics of confidence scores?
- How do upper-layer decoders declare the side information they produce, and
  how does that interact with the fault-model interchange in 0002?