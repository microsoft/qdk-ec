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

The gates cover Rust, C, the CMake package, Python, documentation, coverage,
example audits, and source-distribution rebuilding. Example audits require a
compatible QDK build with its `ec` dependencies. Check the release artifacts
outside the source checkout and require CI to pass for the release commit.

Publication uses the parent [release pipeline](../.ado/publish.yaml), not the
version-bump command. Keep the binding crates unpublished as Rust packages.
Create release tags only on reviewed, merged commits.

## vcpkg Port

The `ports/qodec` port is also a Git registry in this repository. It builds the
[C bindings' CMake package](bindings/c/README.md#cmake-package) for native x64
Linux, macOS, and Windows/MSVC, with static and shared libraries. To consume the
current branch from a project manifest, add `"qodec"` to `dependencies` and add
the registry to the `configuration` field in its `vcpkg.json`:

```json
{
  "dependencies": ["qodec"],
  "configuration": {
    "registries": [
      {
        "kind": "git",
        "repository": "https://github.com/microsoft/qdk-ec",
        "reference": "apaetz/qodec-vcpkg",
        "baseline": "<registry-commit-sha>",
        "packages": ["qodec"]
      }
    ]
  }
}
```

The baseline is the commit containing the registry version metadata; pin it to
the commit you want to use. `reference` names the branch that contains it. For
reproducible builds, keep the baseline commit fixed. vcpkg builds offline from
checksummed downloads, so the port pins:

- The qdk-ec source commit used by the port.
- `crates.cmake`: every crates.io package qodec-c needs, with archive SHA-512s.
  The port vendors these and trims the workspace to `qodec` and `qodec/bindings/c`
  without their dev-dependencies, because Cargo resolves every workspace member.
  Both manifests must end with their `[dev-dependencies]` section.
- An official Rust toolchain (rustc, cargo, rust-std) for each host, with SHA-512s.
  It must be at least the CMake layer's minimum Rust version.

The release tarball has no `Cargo.lock`, so `crates.cmake` is the port's lockfile.
To update the port for a new qodec source commit:

```bash
cargo generate-lockfile   # or reuse the lockfile CI tested
python qodec/tools/vcpkg_crates.py > <vcpkg>/ports/qodec/crates.cmake
```

Update `REF` in `ports/qodec/portfile.cmake` to the new source commit. Then,
from the qdk-ec root, regenerate the registry version metadata:

```bash
<vcpkg>/vcpkg x-add-version \
  --x-builtin-ports-root=ports \
  --x-builtin-registry-versions-dir=versions \
  qodec
```

When qodec's version changes, update `version` in `ports/qodec/vcpkg.json` and
reset `port-version`. Port-only fixes increase `port-version` instead. Commit
`ports/qodec` and `versions/` together. The upstream microsoft/vcpkg submission
can later copy the port and version files into its curated registry.
