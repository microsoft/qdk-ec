# Parametrization with Mako

When evaluating a QEC code across multiple distances ($d = 3, 5, 7, \ldots$) or error
rates, maintaining a separate `.deq` file for each parameter combination is tedious and
error-prone. Every time the circuit structure changes, you must update every copy — and
the copies inevitably diverge.

[Mako](https://www.makotemplates.org/) is a Python template engine that solves this
problem: you write **one** `.deq` file with embedded Python expressions, and the template
engine renders it into valid `.deq` code for any parameter values. The deq CLI has
built-in Mako support — no external tools required.

---

## The Problem: Hardcoded Parameters

Here is a simple repetition code memory experiment hardcoded at $d = 3$ and $p = 0.05$:

[Fixed d=3 repetition code](../examples/mako/01_fixed_d3.deq)
<!-- deq-highlight-begin: ../examples/mako/01_fixed_d3.deq -->
```deq
# A repetition code memory experiment — hardcoded at d=3, p=0.05

CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}

GADGET PrepareZ {
    R 0 1 2
    X_ERROR(0.05) 0 1 2
    OUTPUT RepetitionCode 0 1 2
}

GADGET Syndrome {
    INPUT RepetitionCode 0 2 4
    X_ERROR(0.05) 0 2 4
    R 1 3
    CX 0 1 2 3
    CX 2 1 4 3
    X_ERROR(0.05) 1 3
    M 1 3
    OUTPUT RepetitionCode 0 2 4
}

GADGET MeasureZ {
    INPUT RepetitionCode 0 1 2
    X_ERROR(0.05) 0 1 2
    M 0 1 2
    READOUT rec[-3] rec[-2] rec[-1]
}

# d=3 rounds of syndrome extraction
COMPOSE FTSyndrome {
    INPUT RepetitionCode 0
    REPEAT 3 {
        Syndrome 0
    }
    OUTPUT RepetitionCode 0
}

PROGRAM MemoryExperiment {
    PrepareZ 0
    FTSyndrome 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
```
<!-- deq-highlight-end: ../examples/mako/01_fixed_d3.deq -->

To switch to $d = 5$, you would need to change:
- The `CODE` block: `[[3,1,1]]` → `[[5,1,1]]`, add `X3*X4` / `Z3*Z4` to the logicals,
  add two more stabilizers
- Every `GADGET`: update qubit indices (`R 0 1 2` → `R 0 1 2 3 4`), add more `CX`
  pairs, more ancillae, more measurements
- The `COMPOSE` block: `REPEAT 3` → `REPEAT 5`

That's a lot of changes for a single parameter, and it's easy to make mistakes. What if
you also want to sweep over error rates? The combinatorial explosion makes manual
duplication unworkable.

---

## Mako Syntax Basics

Mako provides three constructs that can be embedded in `.deq` files:

### 1. Block declarations: `<%...%>`

A `<%...%>` block defines Python variables that are available throughout the rest of the
file. This is where you declare parameters with defaults:

[Parameter block](../examples/mako/snippet_mako_header.deq)
<!-- deq-highlight-begin: ../examples/mako/snippet_mako_header.deq -->
```deq
<%
# parameters
d = int(context.get('d', 3))
p = float(context.get('p', 0.05))
%>
```
<!-- deq-highlight-end: ../examples/mako/snippet_mako_header.deq -->

Parameters arrive as **strings** from the CLI (e.g., `--mako d=5` passes `"5"`), so you
must cast them explicitly with `int()` or `float()`. The `context.get('key', default)`
pattern provides a fallback when no value is supplied — matching the behavior of
`mako-render --var`.

### 2. Inline expressions: `${...}`

A `${...}` expression evaluates arbitrary Python and inserts the result as text. This is
used for computed values:

[Parametrized CODE block](../examples/mako/snippet_mako_code.deq)
<!-- deq-highlight-begin: ../examples/mako/snippet_mako_code.deq -->
```deq
CODE RepetitionCode [[${d},1,1]] {
    LOGICAL ${"*".join(f"X{i}" for i in range(d))} ${"*".join(f"Z{i}" for i in range(d))}
    STABILIZER ${" ".join(f"Z{i}*Z{i+1}" for i in range(d-1))}
}
```
<!-- deq-highlight-end: ../examples/mako/snippet_mako_code.deq -->

The expression `${"*".join(f"X{i}" for i in range(d))}` generates `X0*X1*X2` for $d = 3$,
or `X0*X1*X2*X3*X4` for $d = 5$. Similarly, `${" ".join(...)}` produces space-separated
qubit lists of the correct length.

### 3. Control lines: `% for`, `% if`

Lines starting with `%` followed by a Python keyword are Mako control lines:

```text
% for i in range(d):
PROGRAM Test${i} { ... }
% endfor
```

These are useful when you need to **repeat entire blocks** of `.deq` code. For the
repetition code examples in this chapter, inline expressions with Python comprehensions
are sufficient, so we won't use control lines here.

---

## The Full Parametrized Example

Here is the same repetition code, fully parametrized with Mako:

[Parametrized repetition code](../examples/mako/02_parametrized.deq)
<!-- deq-highlight-begin: ../examples/mako/02_parametrized.deq -->
```deq
<%
# parameters
d = int(context.get('d', 3))
p = float(context.get('p', 0.05))
%>
CODE RepetitionCode [[${d},1,1]] {
    LOGICAL ${"*".join(f"X{i}" for i in range(d))} ${"*".join(f"Z{i}" for i in range(d))}
    STABILIZER ${" ".join(f"Z{i}*Z{i+1}" for i in range(d-1))}
}

GADGET PrepareZ {
    R ${" ".join(str(i) for i in range(d))}
    X_ERROR(${p}) ${" ".join(str(i) for i in range(d))}
    OUTPUT RepetitionCode ${" ".join(str(i) for i in range(d))}
}

GADGET Syndrome {
    INPUT RepetitionCode ${" ".join(str(2*i) for i in range(d))}
    X_ERROR(${p}) ${" ".join(f"{2*i}" for i in range(d))}
    R ${" ".join(f"{2*i+1}" for i in range(d-1))}
    CX ${" ".join(f"{2*i} {2*i+1}" for i in range(d-1))}
    CX ${" ".join(f"{2*i+2} {2*i+1}" for i in range(d-1))}
    X_ERROR(${p}) ${" ".join(f"{2*i+1}" for i in range(d-1))}
    M ${" ".join(f"{2*i+1}" for i in range(d-1))}
    OUTPUT RepetitionCode ${" ".join(str(2*i) for i in range(d))}
}

GADGET MeasureZ {
    INPUT RepetitionCode ${" ".join(str(i) for i in range(d))}
    X_ERROR(${p}) ${" ".join(str(i) for i in range(d))}
    M ${" ".join(str(i) for i in range(d))}
    READOUT ${" ".join(f"rec[-{i}]" for i in range(1,d+1))}
}

# d rounds of syndrome extraction for fault tolerance
COMPOSE FTSyndrome {
    INPUT RepetitionCode 0
    REPEAT ${d} {
        Syndrome 0
    }
    OUTPUT RepetitionCode 0
}

PROGRAM MemoryExperiment {
    PrepareZ 0
    FTSyndrome 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
```
<!-- deq-highlight-end: ../examples/mako/02_parametrized.deq -->

### Walkthrough

Let's compare key lines between the fixed and parametrized versions:

| Element        | Fixed ($d = 3$)                  | Parametrized                                                       |
| -------------- | -------------------------------- | ------------------------------------------------------------------ |
| CODE header    | `[[3,1,1]]`                      | `[[${d},1,1]]`                                                  |
| Logical X      | `X0*X1*X2`                       | `${"*".join(f"X{i}" for i in range(d))}`                           |
| Stabilizers    | `Z0*Z1 Z1*Z2`                   | `${" ".join(f"Z{i}*Z{i+1}" for i in range(d-1))}`                 |
| Data qubit list | `0 1 2`                         | `${" ".join(str(i) for i in range(d))}`                            |
| Error rate     | `0.05`                           | `${p}`                                                             |
| REPEAT count   | `3`                              | `${d}`                                                             |

The Syndrome gadget is the most involved, because it uses interleaved data/ancilla qubit
indices ($0, 2, 4, \ldots$ for data; $1, 3, 5, \ldots$ for ancillae):

[Parametrized Syndrome gadget](../examples/mako/snippet_mako_gadget.deq)
<!-- deq-highlight-begin: ../examples/mako/snippet_mako_gadget.deq -->
```deq
GADGET Syndrome {
    INPUT RepetitionCode ${" ".join(str(2*i) for i in range(d))}
    X_ERROR(${p}) ${" ".join(f"{2*i}" for i in range(d))}
    R ${" ".join(f"{2*i+1}" for i in range(d-1))}
    CX ${" ".join(f"{2*i} {2*i+1}" for i in range(d-1))}
    CX ${" ".join(f"{2*i+2} {2*i+1}" for i in range(d-1))}
    X_ERROR(${p}) ${" ".join(f"{2*i+1}" for i in range(d-1))}
    M ${" ".join(f"{2*i+1}" for i in range(d-1))}
    OUTPUT RepetitionCode ${" ".join(str(2*i) for i in range(d))}
}
```
<!-- deq-highlight-end: ../examples/mako/snippet_mako_gadget.deq -->

Each `${...}` expression generates the correct qubit list for any distance. For example,
at $d = 5$ the `CX` line `${" ".join(f"{2*i} {2*i+1}" for i in range(d-1))}` produces
`0 1 2 3 4 5 6 7` — four CNOT pairs connecting each data qubit to its ancilla.

---

## What the Template Produces

You can render the template yourself to see the expanded output:

```sh
mako-render 02_parametrized.deq --var d=5 --var p=0.05
```

Here is the result with $d = 5$:

[Rendered output at d=5](../examples/mako/02_parametrized_d5.deq)
<!-- deq-highlight-begin: ../examples/mako/02_parametrized_d5.deq -->
```deq

CODE RepetitionCode [[5,1,1]] {
    LOGICAL X0*X1*X2*X3*X4 Z0*Z1*Z2*Z3*Z4
    STABILIZER Z0*Z1 Z1*Z2 Z2*Z3 Z3*Z4
}

GADGET PrepareZ {
    R 0 1 2 3 4
    X_ERROR(0.05) 0 1 2 3 4
    OUTPUT RepetitionCode 0 1 2 3 4
}

GADGET Syndrome {
    INPUT RepetitionCode 0 2 4 6 8
    X_ERROR(0.05) 0 2 4 6 8
    R 1 3 5 7
    CX 0 1 2 3 4 5 6 7
    CX 2 1 4 3 6 5 8 7
    X_ERROR(0.05) 1 3 5 7
    M 1 3 5 7
    OUTPUT RepetitionCode 0 2 4 6 8
}

GADGET MeasureZ {
    INPUT RepetitionCode 0 1 2 3 4
    X_ERROR(0.05) 0 1 2 3 4
    M 0 1 2 3 4
    READOUT rec[-1] rec[-2] rec[-3] rec[-4] rec[-5]
}

# d rounds of syndrome extraction for fault tolerance
COMPOSE FTSyndrome {
    INPUT RepetitionCode 0
    REPEAT 5 {
        Syndrome 0
    }
    OUTPUT RepetitionCode 0
}

PROGRAM MemoryExperiment {
    PrepareZ 0
    FTSyndrome 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
```
<!-- deq-highlight-end: ../examples/mako/02_parametrized_d5.deq -->

Note how the `<%...%>` block is gone (it was consumed during rendering) and all `${...}`
expressions have been replaced with their computed values. The result is a valid `.deq`
file that can be parsed and transpiled normally.

---

## Using Mako from the CLI

### Approach 1: Integrated `--mako` flag (recommended)

All major deq CLI commands accept `--mako key=value` to pass parameters directly:

```sh
# Transpile at d=5 with p=0.01
deq transpile 02_parametrized.deq --mako d=5 --mako p=0.01 --program MemoryExperiment

# Annotate at d=3 (uses default p=0.05)
deq annotate 02_parametrized.deq --mako d=3

# Simulate logical error rate at d=7
deq simulate ler 02_parametrized.deq --mako d=7 --program MemoryExperiment
```

The `--mako` flag can be repeated for multiple parameters. Passing `--mako` implies
consent to execute Mako templates (see [Security Note](#security-note) below).

### Approach 2: External `mako-render` tool

You can also render the template externally and pipe the result to `deq`:

```sh
# Render to a standalone .deq file, then transpile
mako-render 02_parametrized.deq --var d=5 > 02_parametrized_d5.deq
deq transpile 02_parametrized_d5.deq --program MemoryExperiment
```

This approach is useful when you want to inspect the rendered output before
transpiling it.

### Supported CLI commands

The following commands accept `--mako` and `--skip-mako-warning`:

| Command              | Purpose                                  |
| -------------------- | ---------------------------------------- |
| `deq transpile`     | Transpile to `.deq.jit`                 |
| `deq annotate`      | Annotate with derived checks and errors  |
| `deq simulate ler`  | Logical error rate simulation            |
| `deq inject si1000` | Inject SI1000 noise model                |
| `deq inject biased` | Inject biased noise model                |
| `deq strip-noise`   | Remove noise from a `.deq` file         |

---

## Including External Files

Mako's `<%include>` directive lets you inline an external file into a `.deq` template.
This is especially useful for reusing existing Stim circuit fragments — you can keep
the circuit body in a standalone `.stim` file and include it in a gadget without
copy-pasting:

[Include example](../examples/mako/03_include.deq)
<!-- deq-highlight-begin: ../examples/mako/03_include.deq -->
```deq
# Demonstrates Mako's include directive to inline an existing stim
# circuit file into a gadget body, avoiding copy-paste

CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}

GADGET Syndrome {
    INPUT RepetitionCode 0 2 4
    X_ERROR(0.05) 0 2 4
    <%include file="syndrome_body.stim"/>
    OUTPUT RepetitionCode 0 2 4
}
```
<!-- deq-highlight-end: ../examples/mako/03_include.deq -->

The included file [syndrome_body.stim](../examples/mako/syndrome_body.stim) contains
just the raw circuit instructions:

```text
R 0 1 2
CX 0 1 2 3
CX 2 1 4 3
M 1 3
```

During Mako rendering, the `<%include>` directive is replaced with the file's contents,
producing a valid gadget body. The file path is relative to the `.deq` file's directory.

Since `<%include>` is Mako syntax, the CLI will prompt for confirmation (or require
`--skip-mako-warning`):

```sh
deq transpile 03_include.deq --skip-mako-warning --program MinimalExperiment
```

---

## Security Note

Mako templates execute **arbitrary Python code**. When a `.deq` file contains Mako
syntax and no `--mako` flag is passed, the CLI prompts for confirmation before rendering:

```text
WARNING: A .deq file contains Mako template syntax which can execute
arbitrary Python code. ...
Proceed? [y/N]
```

Passing `--mako` (with any variable) or `--skip-mako-warning` suppresses this prompt.
In non-interactive environments (e.g., piped stdin), the CLI exits with an error unless
one of these flags is provided.

---

## Summary

| Aspect            | Fixed file                             | Mako template                                            |
| ----------------- | -------------------------------------- | -------------------------------------------------------- |
| Files needed      | One per $(d, p)$ combination           | One file for all combinations                            |
| Maintenance       | Every copy must be updated separately  | Single source of truth                                   |
| CLI usage         | `deq transpile fixed.deq`            | `deq transpile template.deq --mako d=5 --mako p=0.01` |
| Readability       | Straightforward `.deq` syntax         | `.deq` + embedded Python expressions                    |
| Error-proneness   | High (manual copy-paste)               | Low (parameters computed automatically)                  |
| Supports sweeps   | No (need external scripting)           | Yes (loop over `--mako` values in a shell script)        |

For any code family where you need to evaluate performance across distances or noise
parameters, Mako parametrization eliminates the duplication and keeps your `.deq`
definitions maintainable.
