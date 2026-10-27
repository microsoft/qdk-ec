# Releasing qodec

## Version Ownership

Each qodec crate declares its own `[package].version` in
[Cargo.toml](Cargo.toml), [bindings/python/Cargo.toml](bindings/python/Cargo.toml),
or [bindings/c/Cargo.toml](bindings/c/Cargo.toml).

- The three crates use `shared-version = "qodec"` in their
  `[package.metadata.release]` tables. This named group keeps their versions
  aligned without coupling other qdk-ec packages.
- The Python package declares a dynamic version in
  [pyproject.toml](bindings/python/pyproject.toml); maturin reads the binding
  crate's version.
- `qodec.__version__` comes from installed package metadata. A checkout without
  installed metadata reports `0+unknown`.

Use the scoped `cargo release version` commands below instead of editing each
version manually. The packaging tests check that the three versions agree.
Schema and ABI revisions are independent of the package version:

| Version | Source |
| --- | --- |
| Rust package | [Cargo.toml](Cargo.toml) |
| Python package | [bindings/python/Cargo.toml](bindings/python/Cargo.toml) |
| C binding crate | [bindings/c/Cargo.toml](bindings/c/Cargo.toml) |
| On-disk schema | [CURRENT_SCHEMA_VERSION](src/manifest.rs) |
| C ABI | [QODEC_ABI_VERSION](bindings/c/src/lib.rs) |

Follow the [compatibility contract](CHANGELOG.md#compatibility-contract) when
changing a schema or ABI revision. Update affected examples, fixtures, and
documentation, and regenerate the C header after an ABI change. From the qdk-ec
repository root:

```bash
bash qodec/bindings/c/regenerate.sh
```

## Bump Versions

qodec is pre-1.0: each `0.x` minor may break the API or on-disk format. Use a
patch release only for compatible changes. Consumers should pin an exact version.

Install the version driver with `cargo install cargo-release --locked`. From
the qdk-ec repository root, preview the next bump before executing it:

```bash
cargo release version minor -p qodec -p qodec-python -p qodec-c
cargo release version minor -p qodec -p qodec-python -p qodec-c --execute
```

The first command is a dry run. The second updates the selected package versions
and any affected lockfile entries. Neither command commits, tags, pushes, or
publishes. Start from a clean worktree before using `--execute`, review the diff,
and commit the changes through the normal pull-request process. A `patch` bump
or an explicit version can be supplied instead. Update the changelog with the
release's user-visible changes and applicable schema and ABI revisions.

For prerelease versions, replace `minor` with `alpha` in the same scoped command
to produce `X.Y.Z-alpha.N`, or with `release` to drop the suffix. Do not use
`patch`, `minor`, or `major` to advance an alpha: those remove the suffix and
advance the base version.
Maturin maps Rust prerelease suffixes to PEP 440 wheel versions, such as
`-alpha.1` to `a1`.

## Verification and Publication

From the qdk-ec repository root, run the
[verification gates](../.github/instructions/qodec-checks.instructions.md) with
the selected Python environment:

```bash
python qodec/tools/check.py all
```

The gates cover Rust, C, Python, documentation, coverage, example audits, and
source-distribution rebuilding. Example audits require a compatible QDK build
with its `ec` dependencies. Check the release artifacts outside the source
checkout and require CI to pass for the release commit.

Publication uses the parent [release pipeline](../.ado/publish.yaml), not the
version-bump command. Keep the binding crates unpublished as Rust packages.
Create release tags only on reviewed, merged commits.