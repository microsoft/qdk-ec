# deq: dynamic and generic QEC decoding system

deq is a quantum error correction (QEC) decoding system that provides an automated workflow from declarative definitions of QEC codes and logical instructions to a runtime decoder for arbitrary dynamic logical circuits.

Key features:

- **Declarative `.deq` language** — define QEC codes and their physical gate realizations using a stim-compatible DSL; deq automatically discovers checks (detectors) from the Clifford circuit
- **Dynamic circuit decoding** — decode logical circuits whose instructions stream in at runtime, not just static offline-known circuits
- **Simulation & deployment** — run logical error rate simulations, latency benchmarks, and deploy on real hardware with the same compiled library

See the **[Tutorial](https://github.com/microsoft/qdk-ec/blob/main/deq/documents/tutorial/README.md)** for a full introduction, language reference, and worked examples.

## Installation

```sh
pip install deq deq-runtime
```

See [Install from source](#install-from-source) below if you want a development
build or to hack on the Rust runtime.

## Quick start

Here is an example deq program:

<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># define a QEC code of [[n,k,d]] (d is optional)</span></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">3</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.03</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.03</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span><span style="color:#008000">  # data qubit error</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000">(</span><span style="color:#098658">0.03</span><span style="color:#000000">) </span><span style="color:#098658">1</span><span style="color:#098658"> 3</span><span style="color:#008000">  # measurement error</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000">(</span><span style="color:#098658">0.03</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#008000">  # measurement error</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-3]</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000"># a logical circuit with criteria of logical error</span></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>

```sh
# Transpile a .deq definition into a JIT library
deq transpile example.deq --out example.deq.jit --program Simulation

# Run a logical error rate simulation
deq server --decoder black-box-relay-bp --coordinator window \
    --controller jit --controller-config '{"filepath":"example.deq.jit"}' \
    --simulator jit-static --simulator-config '{"filepath":"example.stim","jit_library_filepath":"example.deq.jit","shots":100000}'
```

## Install from source

### Prerequisites

- Python ≥ 3.10
- Rust toolchain (for building `deq_runtime`)
- [maturin](https://github.com/PyO3/maturin) (`pip install maturin`)
- protobuf compiler (`apt install protobuf-compiler` on Ubuntu, `brew install protobuf` on macOS)

### Steps

```sh
# 1. Build and install the Rust runtime (deq_runtime)
cd deq_runtime
maturin develop --release
cd ..

# 2. Generate protobuf Python bindings
python deq/proto/compile.py

# 3. Install the deq Python package
pip install -e .
```
