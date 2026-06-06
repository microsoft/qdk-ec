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
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># A repetition code memory experiment — hardcoded at d=3, p=0.05</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">3</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.05</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Syndrome</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.05</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.05</span><span style="color:#000000">) </span><span style="color:#098658">1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.05</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000"># d=3 rounds of syndrome extraction</span></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> FTSyndrome</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#AF00DB">    REPEAT</span><span style="color:#098658"> 3</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">        Syndrome</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">    }</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> MemoryExperiment</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    FTSyndrome</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/mako/01_fixed_d3.deq -->

To switch to $d = 5$, you would need to change:
- The `CODE` block: `[[3,1,3]]` → `[[5,1,5]]`, add `X3*X4` / `Z3*Z4` to the logicals,
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
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#0000FF">&#x3C;%</span></span>
<span class="line"><span style="color:#008000"># parameters</span></span>
<span class="line"><span style="color:#000000FF">d </span><span style="color:#000000">=</span><span style="color:#267F99"> int</span><span style="color:#000000FF">(context.get(</span><span style="color:#A31515">'d'</span><span style="color:#000000FF">, </span><span style="color:#098658">3</span><span style="color:#000000FF">))</span></span>
<span class="line"><span style="color:#000000FF">p </span><span style="color:#000000">=</span><span style="color:#267F99"> float</span><span style="color:#000000FF">(context.get(</span><span style="color:#A31515">'p'</span><span style="color:#000000FF">, </span><span style="color:#098658">0.05</span><span style="color:#000000FF">))</span></span>
<span class="line"><span style="color:#0000FF">%></span></span></code></pre>
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
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">${d}</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">${d}</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">"*"</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"X</span><span style="color:#0000FF">{</span><span style="color:#000000FF">i</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">"*"</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"Z</span><span style="color:#0000FF">{</span><span style="color:#000000FF">i</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"Z</span><span style="color:#0000FF">{</span><span style="color:#000000FF">i</span><span style="color:#0000FF">}</span><span style="color:#A31515">*Z</span><span style="color:#0000FF">{</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d</span><span style="color:#000000">-</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
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
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#0000FF">&#x3C;%</span></span>
<span class="line"><span style="color:#008000"># parameters</span></span>
<span class="line"><span style="color:#000000FF">d </span><span style="color:#000000">=</span><span style="color:#267F99"> int</span><span style="color:#000000FF">(context.get(</span><span style="color:#A31515">'d'</span><span style="color:#000000FF">, </span><span style="color:#098658">3</span><span style="color:#000000FF">))</span></span>
<span class="line"><span style="color:#000000FF">p </span><span style="color:#000000">=</span><span style="color:#267F99"> float</span><span style="color:#000000FF">(context.get(</span><span style="color:#A31515">'p'</span><span style="color:#000000FF">, </span><span style="color:#098658">0.05</span><span style="color:#000000FF">))</span></span>
<span class="line"><span style="color:#0000FF">%></span></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">${d}</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">${d}</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">"*"</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"X</span><span style="color:#0000FF">{</span><span style="color:#000000FF">i</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">"*"</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"Z</span><span style="color:#0000FF">{</span><span style="color:#000000FF">i</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"Z</span><span style="color:#0000FF">{</span><span style="color:#000000FF">i</span><span style="color:#0000FF">}</span><span style="color:#A31515">*Z</span><span style="color:#0000FF">{</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d</span><span style="color:#000000">-</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(i) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(${p}) </span><span style="color:#0000FF">${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(i) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(i) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Syndrome</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(${p}) </span><span style="color:#0000FF">${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d</span><span style="color:#000000">-</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#0000FF">}</span><span style="color:#0000FF"> {</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d</span><span style="color:#000000">-</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">2</span><span style="color:#0000FF">}</span><span style="color:#0000FF"> {</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d</span><span style="color:#000000">-</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(${p}) </span><span style="color:#0000FF">${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d</span><span style="color:#000000">-</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d</span><span style="color:#000000">-</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(i) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(${p}) </span><span style="color:#0000FF">${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(i) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(i) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"rec[-</span><span style="color:#0000FF">{</span><span style="color:#000000FF">i</span><span style="color:#0000FF">}</span><span style="color:#A31515">]"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(</span><span style="color:#098658">1</span><span style="color:#000000FF">,d</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000"># d rounds of syndrome extraction for fault tolerance</span></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> FTSyndrome</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#AF00DB">    REPEAT</span><span style="color:#098658"> ${d}</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">        Syndrome</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">    }</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> MemoryExperiment</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    FTSyndrome</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/mako/02_parametrized.deq -->

### Walkthrough

Let's compare key lines between the fixed and parametrized versions:

| Element        | Fixed ($d = 3$)                  | Parametrized                                                       |
| -------------- | -------------------------------- | ------------------------------------------------------------------ |
| CODE header    | `[[3,1,3]]`                      | `[[${d},1,${d}]]`                                                  |
| Logical X      | `X0*X1*X2`                       | `${"*".join(f"X{i}" for i in range(d))}`                           |
| Stabilizers    | `Z0*Z1 Z1*Z2`                   | `${" ".join(f"Z{i}*Z{i+1}" for i in range(d-1))}`                 |
| Data qubit list | `0 1 2`                         | `${" ".join(str(i) for i in range(d))}`                            |
| Error rate     | `0.05`                           | `${p}`                                                             |
| REPEAT count   | `3`                              | `${d}`                                                             |

The Syndrome gadget is the most involved, because it uses interleaved data/ancilla qubit
indices ($0, 2, 4, \ldots$ for data; $1, 3, 5, \ldots$ for ancillae):

[Parametrized Syndrome gadget](../examples/mako/snippet_mako_gadget.deq)
<!-- deq-highlight-begin: ../examples/mako/snippet_mako_gadget.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Syndrome</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(${p}) </span><span style="color:#0000FF">${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d</span><span style="color:#000000">-</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#0000FF">}</span><span style="color:#0000FF"> {</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d</span><span style="color:#000000">-</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">2</span><span style="color:#0000FF">}</span><span style="color:#0000FF"> {</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d</span><span style="color:#000000">-</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(${p}) </span><span style="color:#0000FF">${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d</span><span style="color:#000000">-</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d</span><span style="color:#000000">-</span><span style="color:#098658">1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(</span><span style="color:#098658">2</span><span style="color:#000000">*</span><span style="color:#000000FF">i) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
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
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">5</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">5</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#000000">*</span><span style="color:#0000FF">X3</span><span style="color:#000000">*</span><span style="color:#0000FF">X4</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#000000">*</span><span style="color:#0000FF">Z3</span><span style="color:#000000">*</span><span style="color:#0000FF">Z4</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#0000FF"> Z2</span><span style="color:#000000">*</span><span style="color:#0000FF">Z3</span><span style="color:#0000FF"> Z3</span><span style="color:#000000">*</span><span style="color:#0000FF">Z4</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.05</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Syndrome</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span><span style="color:#098658"> 6</span><span style="color:#098658"> 8</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.05</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span><span style="color:#098658"> 6</span><span style="color:#098658"> 8</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 5</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span><span style="color:#098658"> 6</span><span style="color:#098658"> 5</span><span style="color:#098658"> 8</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.05</span><span style="color:#000000">) </span><span style="color:#098658">1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 5</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 5</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span><span style="color:#098658"> 6</span><span style="color:#098658"> 8</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.05</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span><span style="color:#001080"> rec[-2]</span><span style="color:#001080"> rec[-3]</span><span style="color:#001080"> rec[-4]</span><span style="color:#001080"> rec[-5]</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000"># d rounds of syndrome extraction for fault tolerance</span></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> FTSyndrome</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#AF00DB">    REPEAT</span><span style="color:#098658"> 5</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">        Syndrome</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">    }</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> MemoryExperiment</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    FTSyndrome</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
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
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># Demonstrates Mako's include directive to inline an existing stim</span></span>
<span class="line"><span style="color:#008000"># circuit file into a gadget body, avoiding copy-paste</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">3</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Syndrome</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.05</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#000000">    &#x3C;%</span><span style="color:#795E26">include</span><span style="color:#000000"> file="syndrome_body.stim"/></span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
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
