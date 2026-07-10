# DEQ Language Support for Emacs

A major mode for editing `.deq` quantum error correction files in GNU Emacs,
derived from `prog-mode`. Mirrors the feature set of the VS Code extension
in [`../vscode-deq`](../vscode-deq/).

## Features

- Syntax highlighting for all DEQ constructs:
  - `CODE`, `GADGET`, `COMPOSE`, `PROGRAM` blocks
  - `LOGICAL`, `STABILIZER` declarations with Pauli products
  - `INPUT`, `OUTPUT`, `CHECK`, `READOUT`, `OBSERVABLE_INCLUDE`,
    `DETECTOR`, `ERROR(prob)`, `CONDITIONAL`, `PRESELECT`, `VIRTUAL`,
    `PROPAGATE`, `FROM`, `FLIP`, `ASSERT_EQ`
  - Gadget applications with `IN(…)` / `OUT(…)` port bindings
  - Embedded Stim instructions with all target types
  - Decorators (`@Name(…)`)
- Distinct, themable faces for each target class:
  - `deq-pauli-face`    — `[IXYZ]N`   (e.g. `X0`)
  - `deq-logical-face`  — `L[XYZ]N`   (e.g. `LZ3`)
  - `deq-check-face`    — `CN`        (e.g. `C12`)
  - `deq-readout-face`  — `RN`, `DSN`
  - `deq-record-face`   — `rec[-N]`, `sweep[N]`
  - `deq-decorator-face`
- `{`-aware indentation via `syntax-ppss`; customizable through
  `deq-indent-offset` (default `4`).
- Imenu navigation by `CODE` / `GADGET` / `COMPOSE` / `PROGRAM` name.
- Comment toggling (`#`), bracket matching, and `show-paren-mode` support.
- Inherits everything from `prog-mode`: `prettify-symbols-mode`,
  `flymake-mode`, `display-line-numbers-mode`, xref context menu, etc.

## Installation

### Option 1 — load directly from this directory

```elisp
(add-to-list 'load-path "/path/to/deq/deq/circuit/emacs-deq")
(require 'deq-mode)
```

Files ending in `.deq` will automatically open in `deq-mode` thanks to the
`auto-mode-alist` entry installed by the autoload cookie.

### Option 2 — `use-package`

```elisp
(use-package deq-mode
  :load-path "/path/to/deq/deq/circuit/emacs-deq"
  :mode ("\\.deq\\'" . deq-mode))
```

### Optional — byte-compile

```bash
emacs -Q --batch -f batch-byte-compile deq-mode.el
```

### Optional — run the ERT test suite

```bash
make test               # or: make EMACS=/path/to/emacs test
```

24 tests cover mode activation, syntax-table parsing, font-lock face
assignment for every target class, indentation (with custom offset),
imenu indexing, and comment commands.

## Customization

| Variable             | Default | Purpose                              |
|----------------------|---------|--------------------------------------|
| `deq-indent-offset`  | `4`     | Number of spaces per indent level.   |

All faces (`deq-pauli-face`, `deq-logical-face`, `deq-check-face`,
`deq-readout-face`, `deq-record-face`, `deq-decorator-face`) inherit from
standard `font-lock-*` faces and can be remapped per-theme via
`custom-set-faces` or `face-remap-add-relative`.
