# DEQ Language Support for Emacs

A `prog-mode`-derived major mode for `.deq` files, mirroring the
[VS Code extension](../vscode-deq/). It follows the grammar in
[`deqagram/src/deq.pest`](../../../deqagram/src/deq.pest) and requires
Emacs 30.1 or later.

## Features

- Highlighting of all DEQ keywords, targets, Stim instructions, tags,
  decorators and Mako template syntax. Python inside Mako `<% ... %>` blocks
  is highlighted by `python-mode`.
- Warnings for `PRESELECT` forms that the parser rejects.
- Bracket-aware indentation that leaves Python in Mako blocks untouched.
- Imenu navigation of definitions and Mako `<%def>` blocks.

## Installation

Use `use-package`, which ships with Emacs:

```elisp
(use-package deq-mode
  :load-path "/path/to/deq/deq/circuit/emacs-deq"
  :mode ("\\.deq\\'" . deq-mode))
```

This loads the mode only when you first open a `.deq` file.

## Testing

```bash
make test               # or: make EMACS=/path/to/emacs test
```

Face assertions live in annotated files under `resources/`, in
[`ert-font-lock`](https://www.gnu.org/software/emacs/manual/html_node/ert/Syntax-Highlighting-Tests.html)
format.

## Customization

Set `deq-indent-offset` (default `4`) to change the indentation width, and
`deq-mako-fontify-natively` to `nil` to stop highlighting Mako Python with
`deq-mako-python-mode` (default `python-mode`).
Each target class has its own face (`deq-pauli-face`, `deq-logical-face`,
`deq-check-face`, `deq-readout-face`, `deq-record-face`, `deq-loss-face`,
`deq-decorator-face`, `deq-mako-face`); see `M-x customize-group RET deq`.
