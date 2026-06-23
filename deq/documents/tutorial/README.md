# A Tutorial to the deq Decoding System

deq provides an automated workflow from declarative definitions of any QEC subsystem code and logical instructions (with their physical realizations) to a runtime system that can decode arbitrary dynamic logical circuits composed of these codes and instructions.
deq has the following features:

- **Easy Migration**: physical realizations in deq are fully compatible with existing [stim](https://github.com/quantumlib/Stim) circuits
- **Automatic Checks**: deq automatically finds checks (aka detectors) from the Clifford circuit, removing the need to manually annotate them in most situations
- **Dynamic Circuit**: you can decode a dynamic logical circuit by instantiating these user-defined logical instructions at runtime

[A minimal CODE + GADGET definition](examples/intro/small_example.deq)
<!-- deq-highlight-begin: examples/intro/small_example.deq -->
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
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: examples/intro/small_example.deq -->

## The quantum codec (qodec)

Taken together, the codes and gadgets of a `.deq` file describe the encoding and
decoding of a QEC protocol. We call this a quantum codec, or **qodec**. The name
is an analogy to the term [codec](https://en.wikipedia.org/wiki/Codec) for
encoding and decoding digital streams. A qodec is a
self-contained description of how logical information is *encoded* into physical
qubits and how the resulting syndromes are *decoded* back into logical outcomes.

A qodec is an abstract object, not a file format. The `.deq` source you write
and the compiled `.deq.jit`/`.deq.bin` protobuf libraries are both
*representations* of the same underlying qodec. Loading a `.deq` file yields a
qodec; compiling, simulating, and decoding are operations on it. A qodec could
be expressed in other representations too, such as Python objects, or YAML.

## Capabilities

What you can do with your qodec and the deq decoding system:

- **Simulation**: you can easily run logical error rate simulations of any logical circuit with your own logical error criteria, and/or timing emulation to test latency of various decoder backends on different logical circuits
- **Deployment**: After simulation, you can run the actual circuit on quantum hardware. deq compiles your qodec (written in the .deq language) offline; at runtime, you stream in logical instructions and physical measurements, then wait for error-corrected logical readout

For example, if you want to evaluate the logical error rate performance, just write a small .deq file to construct the logical circuit, and then compile and simulate using the commands shown after the circuit definition

[A small logical circuit for logical error rate evaluation](examples/intro/small_example_evaluation.deq)
<!-- deq-highlight-begin: examples/intro/small_example_evaluation.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">IMPORT</span><span style="color:#A31515"> "small_example.deq"</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000"># a logical circuit with criteria of logical error</span></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: examples/intro/small_example_evaluation.deq -->


```sh
# generate small_example.deq.jit and small_example.stim
deq transpile small_example_evaluation.deq --out small_example.deq.jit --program Simulation
# run logical error rate simulation from these files
# NOTE: on Windows (cmd/PowerShell), escape inner double quotes, e.g. '{\"buffer_radius\":3}' instead of '{"buffer_radius":3}'
deq server \
    --decoder black-box-relay-bp \
    --coordinator window \
    --coordinator-config '{"buffer_radius":3}' \
    --controller jit \
    --controller-config '{"filepath":"small_example.deq.jit"}' \
    --simulator jit-static \
    --simulator-config '{
        "filepath":"small_example.stim",
        "jit_library_filepath":"small_example.deq.jit",
        "shots": 100000,
        "seed": 123
    }'
# server running on "http://[::1]:50051" (port=50051)
# === Simulation Complete ===
#   Logical errors: 2130/4000
#   Error rate: 2.130000e-2 ± 8.95e-4
#   Decoding time: 34.120s (3.412e-4s per shot)
#   Last-batch latency: 34.027s (3.403e-4s per shot)
# *notice that the logical error rate is below physical error rates in the circuit
```

To dive deeper, read the following chapters:
- [deq language basics](chapters/language-basics.md)
- [Composing Gadgets with COMPOSE](chapters/compose-gadgets.md)
- [deq JIT basics](chapters/jit-basics.md)
- [deq bin basics](chapters/bin-basics.md)

Once you become comfortable with the basics, let's look at some advanced topics:
- Use cases
  - [Codes with multiple logical qubits ($k>1$)](chapters/codes-multi-logical.md)
  - [Codes with redundant stabilizers](chapters/codes-redundant-stabilizers.md)
  - [Logical operation with multiple inputs and outputs](chapters/multi-port-gadgets.md)
  - [Floquet codes and dynamically generated logical qubits](chapters/floquet-code.md)
  - [Logical Teleportation in COMPOSE: the `@REPROPAGATE` Decorator](chapters/compose-repropagate.md)
- [Parametrization with Mako](chapters/mako-parametrization.md)
- [Plug in your own decoder in Python](chapters/python-decoder.md)
- [Driving the runtime from Python](chapters/python-runtime.md)
- [Debugging your .deq program](chapters/debug-deq-program.md)
- [Steane-style syndrome extraction](chapters/steane-style-ec.md)
- [Speed-accuracy trade-off with .deq program]
- [Noise models]
- [Pre-Selection and post-selection]
- [Window decoding](../examples/window_decoding_tutorial.ipynb)
- [Deployment Notes] # mention GTYPE decorator here

## Research Story

### The Problem

Most prior research on fast QEC decoders — for example [Fusion Blossom](https://arxiv.org/abs/2305.08307), [Micro Blossom](https://arxiv.org/abs/2502.14787), [DecoNet](https://arxiv.org/abs/2504.11805v1) and [LEGO](https://arxiv.org/abs/2410.03073) proposals — targets an offline-known, static logical circuit.
But if the entire logical circuit is known upfront, there is no need for fast decoding at all: everything can be processed offline.
The real challenge is decoding **dynamic** logical circuits, where logical instructions stream in at runtime and the decoding system must keep up in real time.
deq is built around this challenge — not as an optimization of any single point solution, but as a complete decoding *system*.

### What We Discovered

Building a dynamic decoding system that is general — rather than tied to a specific code such as the surface code with a limited set of lattice surgery operations — requires cross-disciplinary expertise in both QEC and classical software/hardware, and the resulting systems tend to be hard to maintain and extend.
deq addresses this by defining standard specifications so that hardware acceleration can be automated, rather than re-engineered for every new code or protocol.

In the course of developing deq, we identified a deeper structural problem: **decoding graphs are context-dependent**.
Even when executing the same logical gate, the decoding problem changes depending on what gates come before and after it.
Outside of narrow cases like surface code with lattice surgery, you cannot precompute a single "decoding block" per gate type — blocks must be assembled incrementally as the circuit unfolds.
deq solves this through a type-instance architecture: gadget *types* are compiled offline with parameterized remote references, while the actual decoding graph is assembled at runtime as gadget *instances* connect to their neighbors.

A second insight is the need for a **standard protocol between QEC researchers and system developers**.
The people who discover new QEC codes and logical gates are experts in quantum information, not decoding system internals; conversely, system developers may not understand every new code and protocol.
In traditional engineering, specifications come first and implementations follow — but in QEC, stable standards may not emerge for years, and we cannot wait.
deq addresses this by defining a declarative language (`.deq`) that both parties can speak: QEC researchers specify their codes and physical realizations, and the decoding system handles the rest automatically.

### Where We Are Going

The first generation of deq is a software system with GPU-accelerated decoder backends. We chose software first for three reasons:

- **Near-term utility**: A software solution is already practical for neutral-atom and trapped-ion quantum computers with millisecond-level logical cycle times.
- **System-level understanding before hardware commitment**: We want to understand every aspect of the system design before committing to hardware. Bottom-up approaches that build dedicated ASICs for surface code memory experiment are impressive as concrete engineering achievements, but have not yet proven practical beyond demonstration; general-purpose decoding involves substantially more challenges than any single point solution can address.
- **Parametrized, automated design**: We design the system to be parametric from the start, so that the same architecture applies to as broad a range of QEC codes and protocols as possible.

The long-term goal is dedicated hardware acceleration — FPGAs for reconfigurable deployments and ultimately cryo-ASICs for direct integration with quantum processors.
By building the software system first, we ensure the hardware architecture is grounded in a thorough understanding of the full decoding problem, not just the easiest special case.
