# 0013 - URI references to qodec artifacts

**Status:** proposed

## Context

A protocol using a shared instruction set should be able to reference its
published definition instead of copying it from another repository. For example,
an author could name a particular version of the standard Stim instruction set.
The same need applies to shared codes and gadgets.

Today references name local files or entries in a bundle. Authors must download
dependencies and arrange local paths themselves. URI references would keep the
published location with the protocol and let the caller choose how to retrieve
it. This belongs in artifact loading, not in a special loader for standard ISAs.

## Benefits

- Use published artifacts without manually downloading dependencies and arranging
  local paths for each project.

## Drawbacks

- Remote content can change or disappear, making results hard to reproduce.
- Fetching an untrusted document's dependencies can access unintended resources.
- Loading and saving need rules beyond filesystem path handling, including
  redirects, caching, and offline use.

## Proposal

Allow artifact-reference fields to contain URIs as well as existing paths.
This is proposed syntax; current loading does not fetch this example URL:

```yaml
layers:
  - instruction_set: https://example.org/qodec-isas/v1/stim.isa.yaml
```

Use the same location rules for instruction sets, codes, gadgets, external
checks/readouts, and circuit-source files. The referencing field still determines
the artifact type. URI support does not make a raw circuit file a gadget or
install a parser for its source language.

Apply those rules to manifest and standalone artifact loading too. The public
loader arguments and retrieval interface remain to be designed.

### Relative references

Resolve a relative reference against the document containing it. For a gadget
retrieved from `https://example.org/protocols/v1/idle.gadget.yaml`:

```yaml
circuit: ./idle.stim
```

the circuit source is `https://example.org/protocols/v1/idle.stim`, not a file
relative to the process's working directory. Locally loaded documents and
bundle entries keep their existing path rules. Preserve Windows drive and UNC
paths; do not mistake a drive letter for a URI scheme.

### Retrieval is caller-controlled

A URI identifies content; it does not grant permission to fetch it. Keep network
access disabled unless the caller enables it. An approved retrieval provider
could obtain content from a local mapping, cache, or remote service.

Apply the caller's policy to every dependency and redirect, not just the entry
document. Credentials stay in caller configuration, not artifact URLs. Report
unsupported schemes, denied requests, missing content, and integrity failures;
do not silently skip an artifact or load a different version.

Start with a defined set of supported schemes, with HTTPS as the remote use
case. A general URI model does not require built-in clients for every scheme.
Existing local loading should work without network dependencies or configuration.

### Reproducible content

A path containing `v1` is not proof that its contents are immutable. Support
pinning the dependency content, for example through immutable locations or
expected content hashes. The exact pinning representation is open. A cache
can improve availability, but is not itself a version guarantee.

### Saving

Retain the loaded content so saving does not fetch a potentially different
revision. A bundle must contain the current content of every referenced
dependency and rewrite references to its own entries. It must load offline
without the original server or a populated cache.

For directory saves, copy remotely loaded artifacts to local files and rewrite
their references. Do not upload edits or overwrite remote content. Keep the
existing reuse rules for unchanged external local files. Preserving remote
links on save could be a separate explicit option if a workflow requires it.

## Alternatives

- Download or vendor files first: current loading works, but dependency retrieval
  and provenance remain outside the protocol.
- Distribute standard ISAs in a package: useful for curated content, but not a
  general way to reference third-party artifacts.
- Add `builtin:` names: convenient for a fixed catalog, but creates a special
  namespace instead of using the artifact's published location.

## Discussion

### Names and scope

No new serialized field names or URI schemes are proposed. Existing reference
fields accept more location forms. Loader or provider API names are unspecified;
prefer options on existing loading operations over parallel URI-only loaders.

Artifact URIs identify documents. They are distinct from the `Reference` paths
inside a loaded qodec. Fetching a document does not evaluate model paths or parse
its circuit calls.

### Integration

Filesystem joining cannot resolve remote references. Loading needs a document
base location and retained content; saving needs a mapping from those locations
to output files or bundle entries. Diagnostic origins may also be URIs rather
than filesystem paths. Authentication, allowed destinations, and resource limits
belong to retrieval policy, not protocol data.

The existing `implements: <instruction_set>#<mnemonic>` spelling also uses `#`.
Specify its interaction with URI fragments and escaping before adoption. Review
other URI/path ambiguities under the compatibility contract so existing valid
local references do not silently acquire a new meaning.

URI support provides access to published gate definitions; it does not establish
their correctness.

## Open Questions

- Which schemes are supported initially, and how does a caller supply retrieval
  policy or a custom provider across Rust, Python, and C?
- Where are content pins recorded, and how are transitive dependencies pinned?
- Which document base applies after a redirect, and how are cycles and repeated
  references handled without unbounded fetching?
- How do URI fragments interact with `implements`, and how are literal local
  filenames that resemble URIs distinguished?
- How should diagnostics and source locations represent remote documents?
- Does any workflow need an explicit save option that retains remote links?