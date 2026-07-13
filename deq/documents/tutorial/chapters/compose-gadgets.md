# Composing Gadgets with COMPOSE

The [language chapter](language-basics.md) showed how to define small gadgets — such as
a single syndrome extraction round — and how the transpiler automatically derives checks
and errors for each one. But what happens when you need **multiple rounds**?

You have two options:

1. **Write a single large gadget** with multiple rounds of syndrome extraction inlined.
   The transpiler auto-derives checks for the entire circuit, but the resulting error
   model can have errors that span **all rounds simultaneously** — producing a dense,
   unstructured decoding hypergraph.

2. **Use `COMPOSE`** to chain small per-round gadgets. Each sub-gadget is analyzed
   independently, then the COMPOSE mechanism chains their matrices and accumulates
   their checks. The result: errors are **always local to individual sub-gadgets**,
   producing a sparse, well-structured hypergraph — by construction.

This chapter demonstrates the difference with our running d=3 repetition code example.

---

## The Problem: Flat Multi-Round Gadgets

Let's write a single gadget with 3 rounds of syndrome extraction:

[Flat 3-round gadget](../examples/compose/01_flat_3idle.deq)
<!-- deq-highlight-begin: ../examples/compose/01_flat_3idle.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># Flat 3-round syndrome extraction: all 3 rounds in a single gadget</span></span>
<span class="line"><span style="color:#008000"># This demonstrates the problem with auto-derived checks spanning all rounds</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000"># A single gadget with 3 rounds of syndrome extraction inlined</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Flat3Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#AF00DB">    REPEAT</span><span style="color:#098658"> 3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">        X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">        R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">        CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">        CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">        X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">        M</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#000000">    }</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Flat3Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose/01_flat_3idle.deq -->

The circuit is physically identical to running the Idle gadget 3 times. Running
`annotate` reveals the derived checks and errors:

[Annotated flat 3-round gadget](../examples/compose/01_flat_3idle.annotated.deq)
<!-- deq-highlight-begin: ../examples/compose/01_flat_3idle.annotated.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@PTYPE</span><span style="color:#000000">(</span><span style="color:#098658">1</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#008000">  # generator S0, destabilizer DS0=X1*X2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#008000">  # generator S1, destabilizer DS1=X0*X1</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">1</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 1:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 3</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:2, 2:1 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">2</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Flat3Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 2 4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 1 3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C3</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 2 4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#267F99"> C2</span><span style="color:#267F99"> C3</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C3</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 1 3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C5</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 2 4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span><span style="color:#267F99"> C4</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#267F99"> C2</span><span style="color:#267F99"> C3</span><span style="color:#267F99"> C4</span><span style="color:#267F99"> C5</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C3</span><span style="color:#267F99"> C5</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 1 3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span><span style="color:#267F99"> C4</span><span style="color:#267F99"> C6</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C3</span><span style="color:#267F99"> C5</span><span style="color:#267F99"> C7</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M4</span><span style="color:#267F99"> IN0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M5</span><span style="color:#267F99"> IN0.S1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M4</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M5</span><span style="color:#001080"> M1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M4</span><span style="color:#001080"> M2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M5</span><span style="color:#001080"> M3</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S0</span><span style="color:#001080"> M4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S1</span><span style="color:#001080"> M5</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 6</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:6 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 15</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:6, 2:3, 3:2, 4:3, 6:1 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">3</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # M(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#001080"> R0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#001080"> R0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#001080"> R0</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-1]</span><span style="color:#008000">  # IN0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#001080"> M0</span><span style="color:#267F99"> IN0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M2</span><span style="color:#001080"> M1</span><span style="color:#267F99"> IN0.S1</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 3:2 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # errors: 3</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:2, 2:1 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Flat3Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose/01_flat_3idle.annotated.deq -->

Look at the **checks** in the Flat3Idle gadget — they all involve `M4` or `M5`
(the measurements from **round 3**):

```
CHECK M4 IN0.S0   # round 3 vs input virtual
CHECK M4 M0       # round 3 vs round 1
CHECK M4 M2       # round 3 vs round 2
```

Although these checks are weight-2 (each references only 2 measurements), they are
**not** the natural round-to-round comparisons. Instead, every check involves the last
round's measurement — the auto-derivation chose a valid but non-local basis. Minimizing
the check weight alone does not guarantee a good decoding hypergraph structure.

Now look at the **errors**:

```
ERROR(0.01) C0 C2 C4 OUT0.LX0              # data error in round 1: triggers 3 checks + logical
ERROR(0.01) C0 C1 C2 C3 C4 C5 OUT0.LX0     # data error in round 1 (qubit 1): triggers 6 checks + logical
ERROR(0.01) C0 C2 C4 C6                    # measurement error in round 3: triggers 4 checks
```

A single data qubit error in round 1 triggers checks from **all 3 rounds** simultaneously
(C0, C1, C2, C3, C4, C5) plus the logical observable — a weight-6 hyperedge in the
decoding graph. This happens because the flat gadget's error analysis propagates each
error through the **entire** remaining circuit: a data error in round 1 propagates
through the round-2 and round-3 CNOT gates, accumulating check triggers along the way.

The checks themselves are weight-2, but because they are non-locally defined (all
involving the last round), the errors that trigger them span multiple rounds. The
combination of non-local checks and high-weight errors makes the decoding hypergraph
dense and difficult to decode efficiently.

---

## The Solution: COMPOSE

The `COMPOSE` block chains small gadgets at the **logical level**, preserving each
sub-gadget's error locality:

[COMPOSE 3-round gadget](../examples/compose/02_compose_3idle.deq)
<!-- deq-highlight-begin: ../examples/compose/02_compose_3idle.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># COMPOSE version: 3 rounds of syndrome extraction via composition</span></span>
<span class="line"><span style="color:#008000"># Demonstrates well-structured checks by construction</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> Idle3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#AF00DB">    REPEAT</span><span style="color:#098658"> 3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">        Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">    }</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle3</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose/02_compose_3idle.deq -->

The key part is the `COMPOSE` block:

[COMPOSE Idle3 block](../examples/compose/snippet_compose_idle3.deq)
<!-- deq-highlight-begin: ../examples/compose/snippet_compose_idle3.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> Idle3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#AF00DB">    REPEAT</span><span style="color:#098658"> 3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">        Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">    }</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose/snippet_compose_idle3.deq -->

| Element                   | Meaning                                |
| ------------------------- | -------------------------------------- |
| `INPUT RepetitionCode 0`  | Declare an input port on code block 0  |
| `REPEAT 3 { ... }`        | Unroll 3 copies of the body            |
| `Idle 0`                  | Apply the Idle gadget on code block 0  |
| `OUTPUT RepetitionCode 0` | Declare an output port on code block 0 |

Note: inside a `COMPOSE` block, the numbers are **code block indices** (not physical
qubit indices). Each code block is an instance of a `CODE` type and may contain multiple
logical qubits when $k > 1$.

Running `annotate` on the COMPOSE version produces a flattened `GADGET` block:

[Annotated COMPOSE 3-round gadget](../examples/compose/02_compose_3idle.annotated.deq)
<!-- deq-highlight-begin: ../examples/compose/02_compose_3idle.annotated.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@PTYPE</span><span style="color:#000000">(</span><span style="color:#098658">1</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#008000">  # generator S0, destabilizer DS0=X1*X2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#008000">  # generator S1, destabilizer DS1=X0*X1</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">1</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 1:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 3</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:2, 2:1 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">2</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 2 4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 1 3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C3</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span><span style="color:#267F99"> IN0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#267F99"> IN0.S1</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S0</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S1</span><span style="color:#001080"> M1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:2 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 5</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:2, 2:3 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">3</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # M(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#001080"> R0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#001080"> R0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#001080"> R0</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-1]</span><span style="color:#008000">  # IN0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#001080"> M0</span><span style="color:#267F99"> IN0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M2</span><span style="color:#001080"> M1</span><span style="color:#267F99"> IN0.S1</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 3:2 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # errors: 3</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:2, 2:1 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">4</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 3 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 3 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 3 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S0</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S1</span><span style="color:#001080"> M1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span><span style="color:#001080"> M2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#001080"> M3</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M2</span><span style="color:#001080"> M4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M3</span><span style="color:#001080"> M5</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M4</span><span style="color:#267F99"> OUT0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M5</span><span style="color:#267F99"> OUT0.S1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#267F99"> C3</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C3</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#267F99"> C4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C3</span><span style="color:#267F99"> C5</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#267F99"> C5</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C5</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#267F99"> C6</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C5</span><span style="color:#267F99"> C7</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 6</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:6 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 1:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 15</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:6, 2:9 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle3</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose/02_compose_3idle.annotated.deq -->

The composed `Idle3` gadget appears as:

[Composed Idle3 gadget](../examples/compose/snippet_idle3_annotated.deq)
<!-- deq-highlight-begin: ../examples/compose/snippet_idle3_annotated.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">4</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 3 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 3 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 3 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S0</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S1</span><span style="color:#001080"> M1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span><span style="color:#001080"> M2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#001080"> M3</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M2</span><span style="color:#001080"> M4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M3</span><span style="color:#001080"> M5</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M4</span><span style="color:#267F99"> OUT0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M5</span><span style="color:#267F99"> OUT0.S1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#267F99"> C3</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C3</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#267F99"> C4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C3</span><span style="color:#267F99"> C5</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#267F99"> C5</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C5</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#267F99"> C6</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C5</span><span style="color:#267F99"> C7</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 6</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:6 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 1:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 15</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:6, 2:9 }</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose/snippet_idle3_annotated.deq -->

Every check references only **adjacent measurement pairs** — each spans exactly 2
positions in the measurement record. And crucially, the Idle gadget's errors remain
unchanged:

```
ERROR(0.01) C0 LX0    # data error: 1 check + logical
ERROR(0.01) C0 C2     # measurement error: 2 checks
```

Each error triggers at most **2 checks** — weight-2 edges, perfectly matchable. The
COMPOSE mechanism achieves this because it processes each sub-gadget independently:
errors are resolved within their originating gadget and never propagate through subsequent
sub-gadgets' circuits.

### Why the errors differ

In the flat gadget, a data qubit error in round 1 propagates through the round-2 and
round-3 CNOT gates. The transpiler sees this full propagation and reports the error as
triggering checks from all subsequent rounds.

In the COMPOSE version, the same data qubit error is analyzed within a single Idle
instance. It triggers one finished check and has a residual (it flips the output
observable). When the next Idle instance runs, the residual is carried through the
correction propagation matrix — but this is handled by the **Pauli frame**, not by
additional check triggers. The error's effect on the decoding hypergraph remains local.

---

## REPEAT Inside COMPOSE

The `REPEAT N { ... }` block inside COMPOSE unrolls N copies of the body at transpile
time. It is equivalent to writing the body N times:

[REPEAT equivalence](../examples/compose/repeat_equivalent.deq)
<!-- deq-highlight-begin: ../examples/compose/repeat_equivalent.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># These are equivalent:</span></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> Idle3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#AF00DB">    REPEAT</span><span style="color:#098658"> 3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">        Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">    }</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> Idle3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose/repeat_equivalent.deq -->

The `REPEAT` form is preferred for clarity and to avoid repetitive code, especially when
the number of rounds is large.

---

## Nested COMPOSE

A COMPOSE gadget is itself a gadget — it can be used inside other COMPOSE blocks. This
enables hierarchical composition:

[Nested COMPOSE: Idle4 = Idle3 + Idle](../examples/compose/03_nested_compose.deq)
<!-- deq-highlight-begin: ../examples/compose/03_nested_compose.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># Nested COMPOSE: Idle4 = Idle3 + Idle</span></span>
<span class="line"><span style="color:#008000"># Demonstrates composing composed gadgets</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> Idle3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#AF00DB">    REPEAT</span><span style="color:#098658"> 3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">        Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">    }</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> Idle4</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle3</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle4</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose/03_nested_compose.deq -->

The key definitions:

[Nested COMPOSE definitions](../examples/compose/snippet_nested_compose.deq)
<!-- deq-highlight-begin: ../examples/compose/snippet_nested_compose.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> Idle3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#AF00DB">    REPEAT</span><span style="color:#098658"> 3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">        Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">    }</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> Idle4</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle3</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose/snippet_nested_compose.deq -->

`Idle4` uses `Idle3` (a COMPOSE) and `Idle` (a GADGET) as sub-gadgets. This works because
COMPOSEs are processed in declaration order — by the time `Idle4` is processed, `Idle3`
is already available as a `JitGadgetType`.

The annotated output for `Idle4` shows 8 measurements (6 from Idle3 + 2 from Idle) and
10 checks (8 from Idle3 + 2 from Idle):

[Annotated nested COMPOSE](../examples/compose/03_nested_compose.annotated.deq)
<!-- deq-highlight-begin: ../examples/compose/03_nested_compose.annotated.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@PTYPE</span><span style="color:#000000">(</span><span style="color:#098658">1</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#008000">  # generator S0, destabilizer DS0=X1*X2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#008000">  # generator S1, destabilizer DS1=X0*X1</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">1</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 1:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 3</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:2, 2:1 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">2</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 2 4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#008000">    # M(0.01) 1 3</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C3</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span><span style="color:#267F99"> IN0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#267F99"> IN0.S1</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S0</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S1</span><span style="color:#001080"> M1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:2 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 5</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:2, 2:3 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">3</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # M(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#001080"> R0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#001080"> R0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#001080"> R0</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-1]</span><span style="color:#008000">  # IN0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#001080"> M0</span><span style="color:#267F99"> IN0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M2</span><span style="color:#001080"> M1</span><span style="color:#267F99"> IN0.S1</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 3:2 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # errors: 3</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:2, 2:1 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">4</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S0</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S1</span><span style="color:#001080"> M1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span><span style="color:#001080"> M2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#001080"> M3</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M2</span><span style="color:#001080"> M4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M3</span><span style="color:#001080"> M5</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M4</span><span style="color:#267F99"> OUT0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M5</span><span style="color:#267F99"> OUT0.S1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#267F99"> C3</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C3</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#267F99"> C4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C3</span><span style="color:#267F99"> C5</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#267F99"> C5</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C5</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#267F99"> C6</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C5</span><span style="color:#267F99"> C7</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 6</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:6 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 1:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 15</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:6, 2:9 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">5</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle4</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S0</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S1</span><span style="color:#001080"> M1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span><span style="color:#001080"> M2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#001080"> M3</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M2</span><span style="color:#001080"> M4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M3</span><span style="color:#001080"> M5</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M4</span><span style="color:#001080"> M6</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M5</span><span style="color:#001080"> M7</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M6</span><span style="color:#267F99"> OUT0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M7</span><span style="color:#267F99"> OUT0.S1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#267F99"> C3</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C3</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#267F99"> C4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C3</span><span style="color:#267F99"> C5</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#267F99"> C5</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C5</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#267F99"> C6</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C5</span><span style="color:#267F99"> C7</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C6</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C6</span><span style="color:#267F99"> C7</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C7</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C6</span><span style="color:#267F99"> C8</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C7</span><span style="color:#267F99"> C9</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 8</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:8 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 1:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 20</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:8, 2:12 }</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Idle4</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose/03_nested_compose.annotated.deq -->

[Composed Idle4 gadget](../examples/compose/snippet_idle4_annotated.deq)
<!-- deq-highlight-begin: ../examples/compose/snippet_idle4_annotated.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">5</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle4</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 1 2</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S0</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S1</span><span style="color:#001080"> M1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span><span style="color:#001080"> M2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#001080"> M3</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M2</span><span style="color:#001080"> M4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M3</span><span style="color:#001080"> M5</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M4</span><span style="color:#001080"> M6</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M5</span><span style="color:#001080"> M7</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M6</span><span style="color:#267F99"> OUT0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M7</span><span style="color:#267F99"> OUT0.S1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#267F99"> C3</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C3</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#267F99"> C4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C3</span><span style="color:#267F99"> C5</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#267F99"> C5</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C5</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C4</span><span style="color:#267F99"> C6</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C5</span><span style="color:#267F99"> C7</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C6</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C6</span><span style="color:#267F99"> C7</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C7</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C6</span><span style="color:#267F99"> C8</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C7</span><span style="color:#267F99"> C9</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 8</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:8 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 1:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 20</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:8, 2:12 }</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose/snippet_idle4_annotated.deq -->

Every check still spans exactly 2 adjacent measurements — the hierarchical composition
preserves the per-round-pair structure at every level.

---

## How It Works

The COMPOSE mechanism performs three key operations for each sub-gadget application:

1. **Measurement offsetting**: Each sub-gadget's measurements are assigned global indices
   in sequence. Idle instance 0 gets measurements 0–1, instance 1 gets 2–3, instance 2
   gets 4–5. This prevents index collisions and preserves round identity.

2. **Matrix chaining**: The sub-gadget's `correction_propagation` matrix is multiplied
   with the accumulated state to propagate observable corrections through the chain. This
   is how the Pauli frame tracks logical observable transformations.

3. **Check accumulation**: Each sub-gadget's finished checks are added to a global list
   with their measurement indices offset. Errors reference these global check indices.
   Because each sub-gadget's errors only reference its own checks, the resulting error
   model is inherently structured.

The result is a single `JitGadgetType` — indistinguishable from a hand-written gadget in
the `.deq.jit` format. The decoder sees it as one gadget with well-structured checks.

---

## COMPOSE vs PROGRAM

Both `COMPOSE` and `PROGRAM` can chain gadgets, but they serve different purposes:

| Aspect                | `COMPOSE`                                  | `PROGRAM`                              |
| --------------------- | ------------------------------------------ | -------------------------------------- |
| **Output**            | One `JitGadgetType` (flattened)            | Separate `JitInstruction`s (preserved) |
| **When processed**    | Transpile time                             | Runtime (JIT compiler)                 |
| **Checks**            | Merged into one check model                | Separate per-gadget check models       |
| **Windowed decoding** | Not applicable (single gadget)             | Enables parallel windowed decoding     |
| **Use case**          | Structuring checks for a logical operation | Defining the execution plan            |

**Rule of thumb**: Use `COMPOSE` to build structured multi-round gadgets (e.g., fault-tolerant
operations). Use `PROGRAM` to define the sequence of these composed gadgets for decoding.

A typical pattern:

[COMPOSE vs PROGRAM pattern](../examples/compose/compose_vs_program.deq)
<!-- deq-highlight-begin: ../examples/compose/compose_vs_program.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> FTIdle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#AF00DB">    REPEAT</span><span style="color:#098658"> 10</span><span style="color:#000000"> { </span><span style="color:#795E26">SyndromeExtraction</span><span style="color:#098658"> 0</span><span style="color:#000000"> }</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">PROGRAM</span><span style="color:#000000"> MemoryExperiment {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#0000FF"> OUT</span><span style="color:#000000">(</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">    FTIdle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    FTIdle</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#0000FF"> IN</span><span style="color:#000000">(</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose/compose_vs_program.deq -->

The `FTIdle` gadget has well-structured checks (thanks to COMPOSE). The `PROGRAM` keeps
each `FTIdle` instance separate, enabling the windowed decoder to process them in parallel.

---

## Summary

| Concept                | Purpose                                                           |
| ---------------------- | ----------------------------------------------------------------- |
| `COMPOSE Name { ... }` | Chain sub-gadgets into a single gadget with structured checks     |
| `REPEAT N { ... }`     | Unroll N copies inside COMPOSE (syntactic sugar)                  |
| `Idle 0`               | Apply sub-gadget on code block 0                                  |
| Nested COMPOSE         | Use a COMPOSE gadget inside another COMPOSE                       |
| Error locality         | COMPOSE preserves per-sub-gadget error locality by construction   |
| vs PROGRAM             | COMPOSE flattens at transpile time; PROGRAM preserves for runtime |
