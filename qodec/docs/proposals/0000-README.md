# qodec Proposals

Proposals describe possible changes to qodec, using the Architecture Decision
Record (ADR) structure. The [concept guides](../concepts/model.md), schemas,
and language APIs describe current behavior; unaccepted proposals do not.

## Writing a Proposal

Start from [TEMPLATE.md](TEMPLATE.md). Name the file `NNNN-short-title.md`, use
the same number in its title, and add it to the index. The next number is **0014**.

Keep these seven sections, in order:

1. **Context:** what is missing, why it matters, and why the workaround is insufficient.
2. **Benefits:** the distinct things the change lets us do.
3. **Drawbacks:** costs, limitations, and risks.
4. **Proposal:** the behavior, scope, and contract.
5. **Alternatives:** other approaches, including no change, and their tradeoffs.
6. **Discussion:** naming, interactions, implementation, and compatibility.
7. **Open Questions:** what remains unresolved; write "None" when settled.

Say what changes, not that it improves "portability" or "flexibility." State
each benefit or cost once. Keep technical detail where it defines behavior,
not where it merely repeats a claim. Use US English and established model terms.

Be brief, but keep the argument: show what goes wrong or remains impractical
without the change, and why qodec is the right place for it.

Label proposed syntax. Define index spaces, mapping shapes, defaults, errors,
and significant costs. Explain new public names and their alternatives briefly;
count them when specified. Prefer existing APIs. Do not include unpublished history.

## Status

| Status | Meaning |
| --- | --- |
| proposed | Open for discussion; not adopted. |
| deferred | Still a candidate, but waiting for a use case or prerequisite. |
| accepted | The decision is approved; implementation may follow separately. |
| rejected | Considered and declined; explain the reason. |
| superseded | Replaced by another proposal; link to it. |

Keep all proposals in this directory and index, regardless of status. For
deferred proposals, say what would justify reconsidering them. Keep IDs stable;
record decisions through statuses and links.

All proposals follow the [compatibility contract](../../CHANGELOG.md#compatibility-contract)
and [validation boundary](../concepts/validation.md): qodec preserves declarations;
consumer audit checks protocol correctness. Describe exceptions or new work in
the proposal rather than repeating these rules in every section.

## Index

There are **13 proposals**, including deferred proposals. Existing numbers are
kept when a proposal is removed.

| # | Proposal | Status |
| --- | --- | --- |
| 0001 | [`mix`: a stochastic action step](0001-mix-stochastic-action-step.md) | proposed |
| 0002 | [Decode-side fault models](0002-fault-models.md) | deferred |
| 0003 | [Typed readout fields](0003-typed-readout-fields.md) | deferred |
| 0004 | [Subsystem-code gauge operators](0004-subsystem-code-gauge-operators.md) | proposed |
| 0005 | [Lateral decomposition](0005-lateral-decomposition.md) | deferred |
| 0006 | [Vertical gadget reach](0006-vertical-gadget-reach.md) | deferred |
| 0007 | [Ergonomic readout addressing](0007-ergonomic-readout-addressing.md) | proposed |
| 0008 | [Triggers for adaptive syndrome extraction and error correction](0008-optional-or-of-parities-flags.md) | proposed |
| 0009 | [Continuous-variable (GKP) codes](0009-cv-gkp-codes.md) | deferred |
| 0010 | [Choi-stabilizer instruction actions](0010-choi-stabilizer-actions.md) | proposed |
| 0011 | [Parameterized Pauli indices in instruction actions](0011-parameterized-pauli-indices.md) | proposed |
| 0012 | [Transversal action steps over variadic block groups](0012-transversal-variadic-steps.md) | proposed |
| 0013 | [URI references to qodec artifacts](0013-uri-artifact-references.md) | proposed |
