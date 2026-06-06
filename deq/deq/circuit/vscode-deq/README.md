# DEQ Language Support for VS Code

Syntax highlighting for `.deq` quantum error correction files.

## Features

- Syntax highlighting for all DEQ constructs:
  - `CODE`, `GADGET`, `COMPOSE`, `PROGRAM` blocks
  - `LOGICAL`, `STABILIZER` declarations with Pauli products
  - `INPUT`, `OUTPUT`, `CHECK`, `READOUT` statements
  - `ERROR(prob)` and `MEASURE(count)` statements
  - Distinct colors for check (`C0`), Pauli (`X0`/`Y0`/`Z0`), readout (`R0`), and logical Pauli shortcut (`LX0`/`LY0`/`LZ0`) targets
  - `ASSERT_EQ` assertions
  - Gadget applications with `IN(...)` / `OUT(...)` port bindings
  - Embedded Stim instructions with all target types
- Comment toggling (`#`)
- Bracket matching and auto-closing
- Code folding for `{ }` blocks

## Installation

### Option 1: Copy to extensions directory

```bash
cp -r vscode-deq ~/.vscode/extensions/vscode-deq-0.1.0
```

Restart VS Code, then open any `.deq` file.

### Option 2: Package as .vsix

```bash
cd vscode-deq
npx @vscode/vsce package
code --install-extension vscode-deq-0.1.0.vsix
```

### Option 3: Development mode

1. Open the `vscode-deq/` folder in VS Code
2. Press **F5** to launch an Extension Development Host
3. Open a `.deq` file in the new window
