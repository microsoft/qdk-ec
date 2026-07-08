# Steane-style Syndrome Extraction

Traditional syndrome extraction uses stabilizer measurements (e.g., `MPP`) that
produce checks spanning two consecutive rounds — the current measurement and
the previous one. This creates cross-round dependencies that force the window
decoder to look at multiple gadgets simultaneously (`buffer_radius ≥ 1`).

Steane-style syndrome extraction eliminates these cross-round dependencies
entirely. Each syndrome gadget prepares fresh ancilla blocks, entangles them
with the data, and measures destructively. The **output is a fresh logical
block** (the logical ancilla), so the checks depend only on local measurements —
enabling fully time-isolated decoding with `buffer_radius = 0`.

This chapter walks through a concrete implementation on the Steane [[7,1,3]]
code and demonstrates how `@DECODE_ONLY` phantom noise restores full
distance scaling under buffer=0 window decoding.

## The Steane [[7,1,3]] code

The Steane code encodes 1 logical qubit in 7 physical qubits with distance 3.
It has 6 stabilizer generators — 3 Z-type and 3 X-type:

[Steane code definition](../examples/steane-ec/snippet_code_definition.deq)
<!-- deq-highlight-begin: ../examples/steane-ec/snippet_code_definition.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> SteaneCode</span><span style="color:#000000"> [[</span><span style="color:#098658">7</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">3</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z3</span><span style="color:#000000">*</span><span style="color:#0000FF">Z4</span><span style="color:#000000">*</span><span style="color:#0000FF">Z5</span><span style="color:#000000">*</span><span style="color:#0000FF">Z6</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#000000">*</span><span style="color:#0000FF">Z5</span><span style="color:#000000">*</span><span style="color:#0000FF">Z6</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#000000">*</span><span style="color:#0000FF">Z4</span><span style="color:#000000">*</span><span style="color:#0000FF">Z6</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> X3</span><span style="color:#000000">*</span><span style="color:#0000FF">X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X5</span><span style="color:#000000">*</span><span style="color:#0000FF">X6</span><span style="color:#0000FF"> X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#000000">*</span><span style="color:#0000FF">X5</span><span style="color:#000000">*</span><span style="color:#0000FF">X6</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#000000">*</span><span style="color:#0000FF">X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X6</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/steane-ec/snippet_code_definition.deq -->

The logical operators are $\bar{X} = X_0 X_1 X_2$ and $\bar{Z} = Z_0 Z_1 Z_2$.

## The `SteaneSyndrome` gadget

The syndrome extraction gadget uses a teleportation-based approach:

[SteaneSyndrome gadget](../examples/steane-ec/snippet_syndrome_gadget.deq)
<!-- deq-highlight-begin: ../examples/steane-ec/snippet_syndrome_gadget.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> SteaneSyndrome</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SteaneCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # prepare 1st ancilla logical block in |0_L> state</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X10</span><span style="color:#000000">*</span><span style="color:#0000FF">X11</span><span style="color:#000000">*</span><span style="color:#0000FF">X12</span><span style="color:#000000">*</span><span style="color:#0000FF">X13</span><span style="color:#0000FF"> X8</span><span style="color:#000000">*</span><span style="color:#0000FF">X9</span><span style="color:#000000">*</span><span style="color:#0000FF">X12</span><span style="color:#000000">*</span><span style="color:#0000FF">X13</span><span style="color:#0000FF"> X7</span><span style="color:#000000">*</span><span style="color:#0000FF">X9</span><span style="color:#000000">*</span><span style="color:#0000FF">X11</span><span style="color:#000000">*</span><span style="color:#0000FF">X13</span></span>
<span class="line"><span style="color:#795E26">    CZ</span><span style="color:#001080"> rec[-3]</span><span style="color:#098658"> 8</span><span style="color:#001080"> rec[-3]</span><span style="color:#098658"> 9</span><span style="color:#001080"> rec[-3]</span><span style="color:#098658"> 11</span></span>
<span class="line"><span style="color:#795E26">    CZ</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 8</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 9</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 11</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 12</span></span>
<span class="line"><span style="color:#795E26">    CZ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 8</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 9</span></span>
<span class="line"><span style="color:#795E26">    DEPOLARIZE1</span><span style="color:#000000">(${p}) </span><span style="color:#098658">7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # prepare 2nd ancilla logical block in |+_L> state</span></span>
<span class="line"><span style="color:#795E26">    RX</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z17</span><span style="color:#000000">*</span><span style="color:#0000FF">Z18</span><span style="color:#000000">*</span><span style="color:#0000FF">Z19</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span><span style="color:#0000FF"> Z15</span><span style="color:#000000">*</span><span style="color:#0000FF">Z16</span><span style="color:#000000">*</span><span style="color:#0000FF">Z19</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span><span style="color:#0000FF"> Z14</span><span style="color:#000000">*</span><span style="color:#0000FF">Z16</span><span style="color:#000000">*</span><span style="color:#0000FF">Z18</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#001080"> rec[-3]</span><span style="color:#098658"> 17</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 15</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 16</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 17</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 20</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 15</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 16</span></span>
<span class="line"><span style="color:#795E26">    DEPOLARIZE1</span><span style="color:#000000">(${p}) </span><span style="color:#098658">14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # CNOT data to 1st ancilla</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 7</span><span style="color:#098658"> 1</span><span style="color:#098658"> 8</span><span style="color:#098658"> 2</span><span style="color:#098658"> 9</span><span style="color:#098658"> 3</span><span style="color:#098658"> 10</span><span style="color:#098658"> 4</span><span style="color:#098658"> 11</span><span style="color:#098658"> 5</span><span style="color:#098658"> 12</span><span style="color:#098658"> 6</span><span style="color:#098658"> 13</span></span>
<span class="line"><span style="color:#795E26">    DEPOLARIZE2</span><span style="color:#000000">(${p}) </span><span style="color:#098658">0</span><span style="color:#098658"> 7</span><span style="color:#098658"> 1</span><span style="color:#098658"> 8</span><span style="color:#098658"> 2</span><span style="color:#098658"> 9</span><span style="color:#098658"> 3</span><span style="color:#098658"> 10</span><span style="color:#098658"> 4</span><span style="color:#098658"> 11</span><span style="color:#098658"> 5</span><span style="color:#098658"> 12</span><span style="color:#098658"> 6</span><span style="color:#098658"> 13</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # CNOT 2nd ancilla to 1st ancilla</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 14</span><span style="color:#098658"> 7</span><span style="color:#098658"> 15</span><span style="color:#098658"> 8</span><span style="color:#098658"> 16</span><span style="color:#098658"> 9</span><span style="color:#098658"> 17</span><span style="color:#098658"> 10</span><span style="color:#098658"> 18</span><span style="color:#098658"> 11</span><span style="color:#098658"> 19</span><span style="color:#098658"> 12</span><span style="color:#098658"> 20</span><span style="color:#098658"> 13</span></span>
<span class="line"><span style="color:#795E26">    DEPOLARIZE2</span><span style="color:#000000">(${p}) </span><span style="color:#098658">14</span><span style="color:#098658"> 7</span><span style="color:#098658"> 15</span><span style="color:#098658"> 8</span><span style="color:#098658"> 16</span><span style="color:#098658"> 9</span><span style="color:#098658"> 17</span><span style="color:#098658"> 10</span><span style="color:#098658"> 18</span><span style="color:#098658"> 11</span><span style="color:#098658"> 19</span><span style="color:#098658"> 12</span><span style="color:#098658"> 20</span><span style="color:#098658"> 13</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # measure data in X basis, measure 1st ancilla in Z basis</span></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#000000">(${pm}) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span></span>
<span class="line"><span style="color:#795E26">    MZ</span><span style="color:#000000">(${pm}) </span><span style="color:#098658">7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SteaneCode</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/steane-ec/snippet_syndrome_gadget.deq -->

The circuit proceeds in five stages:

| Step | Operations                                | Purpose                                           |
| ---- | ----------------------------------------- | ------------------------------------------------- |
| 1    | `R 7–13`, `MPP X`-stab, `CZ` correction   | Prepare 1st ancilla in $\lvert 0_L \rangle$       |
| 2    | `RX 14–20`, `MPP Z`-stab, `CX` correction | Prepare 2nd ancilla in $\lvert +_L \rangle$       |
| 3    | `CX data→anc1`                            | Propagate X errors from data into ancilla 1       |
| 4    | `CX anc2→anc1`                            | Entangle ancilla 2 with ancilla 1 (teleportation) |
| 5    | `MX data`, `MZ anc1`                      | Destructive measurements yield syndrome           |
| 6    | `OUTPUT anc2`                             | Fresh ancilla becomes the new data block          |

The key insight: because ancilla 2 is prepared **fresh** inside this gadget,
the output block carries no history from the previous round. Any errors on the
output come only from the current gadget's noise.

## Check structure: time-isolated syndrome

Running `deq annotate` on this gadget reveals the check structure:

[Annotated SteaneSyndrome](../examples/steane-ec/snippet_syndrome_annotated.deq)
<!-- deq-highlight-begin: ../examples/steane-ec/snippet_syndrome_annotated.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">3</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> SteaneSyndrome</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SteaneCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X10</span><span style="color:#000000">*</span><span style="color:#0000FF">X11</span><span style="color:#000000">*</span><span style="color:#0000FF">X12</span><span style="color:#000000">*</span><span style="color:#0000FF">X13</span><span style="color:#0000FF"> X8</span><span style="color:#000000">*</span><span style="color:#0000FF">X9</span><span style="color:#000000">*</span><span style="color:#0000FF">X12</span><span style="color:#000000">*</span><span style="color:#0000FF">X13</span><span style="color:#0000FF"> X7</span><span style="color:#000000">*</span><span style="color:#0000FF">X9</span><span style="color:#000000">*</span><span style="color:#0000FF">X11</span><span style="color:#000000">*</span><span style="color:#0000FF">X13</span></span>
<span class="line"><span style="color:#795E26">    CZ</span><span style="color:#001080"> rec[-3]</span><span style="color:#098658"> 8</span><span style="color:#001080"> rec[-3]</span><span style="color:#098658"> 9</span><span style="color:#001080"> rec[-3]</span><span style="color:#098658"> 11</span></span>
<span class="line"><span style="color:#795E26">    CZ</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 8</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 9</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 11</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 12</span></span>
<span class="line"><span style="color:#795E26">    CZ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 8</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 9</span></span>
<span class="line"><span style="color:#008000">    # DEPOLARIZE1(0) 7 8 9 10 11 12 13</span></span>
<span class="line"><span style="color:#795E26">    RX</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z17</span><span style="color:#000000">*</span><span style="color:#0000FF">Z18</span><span style="color:#000000">*</span><span style="color:#0000FF">Z19</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span><span style="color:#0000FF"> Z15</span><span style="color:#000000">*</span><span style="color:#0000FF">Z16</span><span style="color:#000000">*</span><span style="color:#0000FF">Z19</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span><span style="color:#0000FF"> Z14</span><span style="color:#000000">*</span><span style="color:#0000FF">Z16</span><span style="color:#000000">*</span><span style="color:#0000FF">Z18</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#001080"> rec[-3]</span><span style="color:#098658"> 17</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 15</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 16</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 17</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 20</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 15</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 16</span></span>
<span class="line"><span style="color:#008000">    # DEPOLARIZE1(0) 14 15 16 17 18 19 20</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 7</span><span style="color:#098658"> 1</span><span style="color:#098658"> 8</span><span style="color:#098658"> 2</span><span style="color:#098658"> 9</span><span style="color:#098658"> 3</span><span style="color:#098658"> 10</span><span style="color:#098658"> 4</span><span style="color:#098658"> 11</span><span style="color:#098658"> 5</span><span style="color:#098658"> 12</span><span style="color:#098658"> 6</span><span style="color:#098658"> 13</span></span>
<span class="line"><span style="color:#008000">    # DEPOLARIZE2(0) 0 7 1 8 2 9 3 10 4 11 5 12 6 13</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 14</span><span style="color:#098658"> 7</span><span style="color:#098658"> 15</span><span style="color:#098658"> 8</span><span style="color:#098658"> 16</span><span style="color:#098658"> 9</span><span style="color:#098658"> 17</span><span style="color:#098658"> 10</span><span style="color:#098658"> 18</span><span style="color:#098658"> 11</span><span style="color:#098658"> 19</span><span style="color:#098658"> 12</span><span style="color:#098658"> 20</span><span style="color:#098658"> 13</span></span>
<span class="line"><span style="color:#008000">    # DEPOLARIZE2(0) 14 7 15 8 16 9 17 10 18 11 19 12 20 13</span></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#000000">(</span><span style="color:#098658">0</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span></span>
<span class="line"><span style="color:#795E26">    MZ</span><span style="color:#000000">(</span><span style="color:#098658">0</span><span style="color:#000000">) </span><span style="color:#098658">7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M19</span><span style="color:#001080"> M18</span><span style="color:#001080"> M17</span><span style="color:#001080"> M16</span><span style="color:#267F99"> IN0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M19</span><span style="color:#001080"> M18</span><span style="color:#001080"> M15</span><span style="color:#001080"> M14</span><span style="color:#267F99"> IN0.S1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M19</span><span style="color:#001080"> M17</span><span style="color:#001080"> M15</span><span style="color:#001080"> M13</span><span style="color:#267F99"> IN0.S2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M12</span><span style="color:#001080"> M11</span><span style="color:#001080"> M10</span><span style="color:#001080"> M9</span><span style="color:#267F99"> IN0.S3</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M12</span><span style="color:#001080"> M11</span><span style="color:#001080"> M8</span><span style="color:#001080"> M7</span><span style="color:#267F99"> IN0.S4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M12</span><span style="color:#001080"> M10</span><span style="color:#001080"> M8</span><span style="color:#001080"> M6</span><span style="color:#267F99"> IN0.S5</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SteaneCode</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S3</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S5</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span><span style="color:#001080"> M6</span><span style="color:#001080"> M7</span><span style="color:#001080"> M8</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span><span style="color:#001080"> M13</span><span style="color:#001080"> M14</span><span style="color:#001080"> M15</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 6</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 5:6 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 6</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 1:6 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/steane-ec/snippet_syndrome_annotated.deq -->

The gadget has **6 finished checks** (syndrome) and **6 unfinished checks**
(output stabilizer tracking). Let us examine them in detail.

### Unfinished checks: output is a +1 eigenstate

The 6 unfinished checks appear after the `OUTPUT` statement:

```
CHECK OUT0.S0
CHECK OUT0.S1
CHECK OUT0.S2
CHECK OUT0.S3
CHECK OUT0.S4
CHECK OUT0.S5
```

Each is a **weight-1** check referencing a single output-virtual stabilizer
(`OUT0.S<s>`). This means the output state is a $+1$ eigenstate of every
stabilizer, **regardless of the input state**. The fresh ancilla block's
stabilizer values are deterministically 0.

This is a striking contrast with traditional repeated-round syndrome
extraction, where unfinished checks have weight $\geq 2$ because the output
stabilizer value must be expressed as a combination of the current round's
measurement and the previous round's value.

Because the unfinished checks are weight-1 and do not expand into additional
physical measurements, the previous gadget's output stabilizer values are
simply constants. This is the key to what happens next.

### Finished checks: purely local syndrome

The 6 finished checks appear before the `OUTPUT` statement:

```
CHECK M19 M18 M17 M16 IN0.S0
CHECK M19 M18 M15 M14 IN0.S1
CHECK M19 M17 M15 M13 IN0.S2
CHECK M12 M11 M10 M9 IN0.S3
CHECK M12 M11 M8 M7 IN0.S4
CHECK M12 M10 M8 M6 IN0.S5
```

Each finished check combines **one input-virtual stabilizer** (`IN0.S<s>`)
with **four physical measurements** (`M<i>`) from the current gadget. Since
the previous gadget's unfinished checks are weight-1 (the output is always a
$+1$ eigenstate), the input-virtual stabilizer contribution is a known
constant — it does not expand into measurements from a previous round.

This means the finished checks depend **only on the current gadget's local
physical measurements**. The decoder can compute the full syndrome without
consulting any neighboring gadgets. This is the defining property of
Steane-style EC that enables `buffer_radius = 0`.

## Gate noise and phantom errors

### The problem

When the `CX` gates between ancilla 2 and ancilla 1 have noise
(`DEPOLARIZE2`), a Z error on an ancilla-2 qubit:

1. Stays on ancilla 2 (which becomes the output)
2. Does **not** affect ancilla 1 (Z does not propagate through CX on the
   control side)
3. Has **no local check** in the current gadget (ancilla 2 is not measured)

The window coordinator correctly drops this error from the decoder — it is a
commit-region error that triggers external checks.
Such errors should never be selected to avoid syndrome conflicts.
But the error propagates to
the next gadget's input, and the next gadget (e.g., `MeasureZ`) has no error
models to explain it.

### The solution: `@DECODE_ONLY` phantom noise

The `@DECODE_ONLY` decorator marks an instruction as visible only to the
decoder (not included in the simulation). By adding phantom noise to the
receiving gadget's input, we give the decoder error models for carried-over
errors:

[MeasureZ with phantom noise](../examples/steane-ec/snippet_measure_phantom.deq)
<!-- deq-highlight-begin: ../examples/steane-ec/snippet_measure_phantom.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SteaneCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span></span>
<span class="line"></span>
<span class="line"><span style="color:#000000">%</span><span style="color:#795E26">if</span><span style="color:#000000"> has_phantom:</span></span>
<span class="line"><span style="color:#795E26">    @DECODE_ONLY</span></span>
<span class="line"><span style="color:#795E26">    DEPOLARIZE1</span><span style="color:#000000">(${p}) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span></span>
<span class="line"><span style="color:#000000">%</span><span style="color:#795E26">endif</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-7]</span><span style="color:#001080"> rec[-6]</span><span style="color:#001080"> rec[-5]</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/steane-ec/snippet_measure_phantom.deq -->

The `@DECODE_ONLY DEPOLARIZE1` instruction:
- Is **included** in the JIT library (decoder sees error models for input errors)
- Is **excluded** from the `.stim` file (simulation does not double-count noise)
- Gives the window decoder the ability to explain syndrome defects caused by
  the previous gadget's output errors

Note that `@SIMULATE_ONLY` is **not** needed on the `DEPOLARIZE2` after the CX
gates. The window coordinator automatically drops commit-region errors with
external checks, so these error models are harmless — they simply get discarded.

## Simulation results

We compare logical error rates for the `Minimal` program
(`PrepareZ → SteaneSyndrome → MeasureZ`) at physical error rate $p = 10^{-3}$:

```text
Steane [[7,1,3]] code — LER sweep (100 target errors per data point)
================================================================================

                                          p = 1e-3
monolithic                                3.0e-4
buffer=0 (no phantom)                     3.3e-3
buffer=0 (with @DECODE_ONLY phantom)      2.4e-4
```

|                         | LER at p = 10⁻³ | Ratio to monolithic   |
| ----------------------- | --------------- | --------------------- |
| monolithic              | 3.0 × 10⁻⁴      | 1×                    |
| buffer=0 (no phantom)   | 3.3 × 10⁻³      | **11× worse**         |
| buffer=0 (with phantom) | 2.4 × 10⁻⁴      | **0.8×** (comparable) |

Without phantom noise, `buffer=0` produces a logical error rate **11×
worse** than monolithic — the decoder has no error model for errors carried
over from the previous gadget, so it misattributes syndrome defects and
applies wrong corrections.

With `@DECODE_ONLY` phantom noise, `buffer=0` performance matches
monolithic. The phantom `DEPOLARIZE1` gives the receiving gadget's decoder
error models for carried-over errors, restoring full decoding accuracy.

## Summary

| Property                 | Traditional MPP syndrome      | Steane-style EC                           |
| ------------------------ | ----------------------------- | ----------------------------------------- |
| Finished checks span     | 2 rounds (current + previous) | 1 round (current only)                    |
| Unfinished checks weight | ≥ 2                           | 1                                         |
| Minimum `buffer_radius`  | ≥ 1                           | 0                                         |
| Cross-round dependencies | Yes                           | No                                        |
| Requires phantom noise   | No                            | Yes (`@DECODE_ONLY` on receiving gadgets) |

Steane-style syndrome extraction enables **fully time-isolated window
decoding**: each gadget is decoded independently with `buffer_radius = 0`.
The only annotation needed is `@DECODE_ONLY` phantom noise on gadgets that
receive teleported data, giving their decoders error models for carried-over
errors. This dramatically simplifies the decoding system — no overlapping
windows, no cross-round syndrome dependencies, and no temporal buffering.
