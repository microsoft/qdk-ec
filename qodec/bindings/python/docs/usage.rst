Using qodec
===========

You can explore an existing protocol or build one from Python objects. We'll
start with the three-qubit repetition code: three physical measurement results
produce one logical result, with parity checks to reveal disagreements.

The Python package requires Python 3.11 or newer. The example YAML files are
part of the source repository, not installed package data. Run these examples
from the qodec directory containing ``examples/``, or obtain the repetition3
bundle and supply its path to ``load``.
The parsed-circuit examples also require the optional ``parsers`` extra, which
currently installs Stim. When building from source, run
``python -m maturin develop --release --extras parsers`` from
``qodec/bindings/python`` in the repository. For a published release, install
``qodec[parsers]`` instead of the base package.

Loading and saving
------------------

Open the example from the qodec directory containing ``examples/``, then follow
its logical measurement down to the physical circuit:

.. doctest::

   >>> import qodec
   >>> repetition3 = qodec.Qodec.load("examples/repetition3/repetition3.qodec.yaml")
   >>> logical, physical = repetition3.layers
   >>> measure_z = logical.gadgets["measure_z"]
   >>> len(measure_z.circuit.readouts)
   3
   >>> len(measure_z.readouts)
   1

The two readout lists answer different questions. ``circuit.readouts`` describes
the bits produced inside the circuit. ``measure_z.readouts`` describes the
logical result that the gadget exposes to its caller. The gadget's checks and
readout equations connect these two levels.

Instruction definitions and circuit calls are separate:

.. doctest::

   >>> instruction_set = logical.instruction_set
   >>> instruction_set.instructions["measure_z"].mnemonic
   'measure_z'
   >>> len(instruction_set.blocks)
   1
   >>> call = measure_z.circuit.calls()[0]
   >>> measure_z.circuit.instruction_set.instructions[call.mnemonic].mnemonic
   'M'

``instruction_set`` contains both block declarations and instruction
definitions. ``calls()`` returns a circuit's ordered invocations, with their
operands and arguments. Calling it interprets the source and can fail
even when the protocol loaded successfully.

To interpret another source language, supply a parser returning
``Sequence[InstructionCall]``. Register it explicitly or use it for one call:

.. doctest::

   >>> from qodec.gadgets import Circuit
   >>> from qodec.instructions import InstructionCall
   >>> def parse_measurements(source, instruction_set):
   ...     return [InstructionCall("M", operands=[int(label)]) for label in source.split()]
   >>> custom = Circuit(physical.instruction_set, "0 1 2", format="example-measurements")
   >>> len(custom.calls(parser=parse_measurements))
   3
   >>> qodec.register(parse_measurements, format="example-measurements")
   >>> [call.operands for call in custom.calls()]
   [[0], [1], [2]]
   >>> len(custom.readouts)
   3

The callback receives source and a snapshot of the target ISA as positional arguments.
It must preserve the source's behavior and readout order or raise an exception.
YAML is registered in Rust. The supplied Stim adapter is registered in Python
on import if Stim is available. The latest registration replaces the earlier
parser for that format, including native and supplied parsers. Calls already
in progress keep their selected parser.
``calls(parser=...)`` changes only that invocation. Python and Rust share the
native registry; registrations from either language replace the existing entry.
Python callbacks require a live interpreter, including when invoked from native
threads. Registration is not saved with the protocol.

To pass the protocol around as text, use ``dumps`` and ``loads``:

.. doctest::

   >>> qodec.Qodec.loads(repetition3.dumps()) == repetition3
   True

``dumps`` produces a self-contained YAML bundle. Loading that text needs no
files on disk. A bundle you write by hand can also refer to external files;
``loads`` resolves those paths from the current working directory.

To save to disk, give ``save`` a directory. It returns the written manifest as
a ``pathlib.Path`` that you can pass directly to ``load``:

.. doctest::

   >>> from pathlib import Path
   >>> from tempfile import TemporaryDirectory
   >>> with TemporaryDirectory() as destination:
   ...     manifest = repetition3.save(destination, single_file=True)
   ...     assert isinstance(manifest, Path)
   ...     reloaded = qodec.Qodec.load(manifest)
   >>> reloaded == repetition3
   True

The temporary directory is removed on leaving the block. Use a persistent
directory such as ``Path("out")`` to keep the saved files. Saving can overwrite
files at generated paths; it does not remove unrelated existing files or
promise rollback after an I/O failure. Use a fresh directory when you need to
keep an earlier revision intact.

An unchanged file outside the original manifest's directory stays external. For
example, ``../shared/physical.yaml`` can remain a shared instruction set after a
directory save. The saved reference points to that same file, using its absolute
path. Editing the instruction set creates a local copy and updates the saved
reference; it never overwrites the shared file. A gadget also needs a local copy
when one of its file references changes.

Before writing, qodec checks that every reused external file still has its loaded
text. A missing file or changed text raises ``QodecSaveError``. This is not a
filesystem lock: symlinks and concurrent changes are outside this protection.
The saved directory remains dependent on external files, so later changes to
those files affect what loads. Files inside the original directory are copied
to the destination as usual, even if their input references were absolute.

``single_file=True`` groups the YAML artifacts in one bundle, including circuit
sources where possible. Sources that cannot be inlined are written beside it.
It copies current values without rereading or linking the original external files.
``dumps`` also includes all values and fails if a source cannot be inlined.
Leave the option out to write separate artifact files.

Saving keeps the manifest filename and reuses compatible loaded artifact paths.
Referenced codes unused by gadgets remain while their layer and block declaration
remain. Edits use current objects, and YAML formatting may change.
``layer.codes`` is a dictionary of shared code definitions keyed by the layer's
block type, not the code name. Editing one of these code objects also updates
encodings that share it. Replacing a code object requires updating the layer
binding and its encodings together. A slice keeps the retained layers' code
bindings even when it removes their gadgets.
A bundle has its own internal manifest filename: this example was
loaded from ``repetition3.qodec.yaml``, but its manifest is named ``qodec.yaml``.
The returned path is ``Path(destination) / repetition3.manifest_filename``.
Relative destinations remain relative, including any ``..`` components;
``save`` does not resolve the path to an absolute location.

How files are found
~~~~~~~~~~~~~~~~~~~

``load`` needs a manifest or bundle **file**, not a directory. Filenames and
folder layouts are your choice. A directory or a missing file raises
:class:`qodec.QodecLoadError`.

The manifest's references tell qodec what to read. A layer's ``instruction_set`` points to
an instruction set; ``codes`` and ``gadgets`` point to code and gadget documents.
Only referenced content is loaded. A gadget document can be as short as
``circuit: ./idle.stim``, but the manifest must point to that document, not
directly to the Stim file. Circuit-source extensions and inline ``format``
still identify the source language.

In a bundle, the first YAML document is ``{key: manifest}``: the key gives the
manifest's path inside the bundle. Later entries are found by their referenced
paths. Paths are relative to the document containing them, including its path
inside a bundle; leading ``..`` components are preserved. Circuit sources can
be bundled or stored separately, and external circuit, check, and readout paths
are relative to their gadget. An explicit ``schema_version`` must be 1; other
versions are rejected.

Inspecting objects
------------------

Use ``print(artifact)`` to inspect a ``Code``, ``InstructionSet``,
``Instruction``, ``Circuit``, or ``Gadget`` as YAML. A notebook cell ending
with that object shows the same text. ``repr(artifact)`` remains compact.

.. doctest::

    >>> print(measure_z.circuit)
    source: M 0 1 2
    format: stim
    <BLANKLINE>
    >>> repr(measure_z)
    'Gadget("measure_z")'

Gadget and circuit displays are **layer-relative snippets**, not complete
bundles. They omit the implemented-instruction and instruction-set file
references; the containing layer supplies those and the code bindings.
They do not expand code definitions or instruction sets. Use
``print(gadget.implements)`` to inspect the declared instruction separately.
Explicit physical block types are included in the circuit boundary maps.
An unmatched gadget encoding in an unfinished draft uses its code name as
the block label.

Display preserves source text with an explicit effective format, even for
malformed or unknown languages. It does not parse calls, validate the model,
run an audit, or write files. Displays and saved files use compact flow lists
for single-line scalar sequences, such as ``[0, 1, 2]`` and parity equations.
Lists containing mappings or other lists remain in block form, as do lists
with multi-line scalar values. Multi-line circuit source keeps its block-scalar
form. The emitter chooses quoting; exact whitespace is not an API contract.
Blocks, block operands, parameters, conditions, and actions display their
YAML fragments, including action guards. ``Qodec`` and ``Layer``
display summaries rather than expanding every artifact. References, readouts,
and Pauli expressions display their text forms, including in notebooks.
Encodings and instruction calls have compact representations:
their standalone objects lack the surrounding context for an on-disk fragment.

No IPython dependency is required for display.

Imports
-------

The main objects are available directly from ``qodec``: ``Qodec``, ``Layer``,
``Code``, ``Gadget``, ``InstructionSet``, and ``Instruction``. For the pieces
used to build them, import from the module that owns the concept:

.. code-block:: python

   from qodec.actions import Observe, Stabilize
   from qodec.gadgets import Circuit, Encoding
   from qodec.instructions import Block, BlockOperand, Parameter

The :doc:`API reference <autoapi/qodec/index>` lists these modules and their
members. The construction example below shows how they fit together.

Inline YAML calls
-----------------

You can write a circuit as a list of instruction calls. Here is a rotation
of block ``0`` through an angle of ``1.5708``:

.. code-block:: yaml

   - rotate_z:
       operands: [0]
       arguments: {theta: 1.5708}

The block is the **operand**: the thing the instruction acts on. The instruction
declares the angle it needs as a **parameter** named ``theta``. This call supplies
``1.5708`` as the **argument**, the value to use for that angle.

The list shorthand puts blocks first and named values after them. Parse that
shorter spelling with the physical ISA from the example we loaded above:

.. doctest::

   >>> from qodec.gadgets import Circuit
   >>> rotation = Circuit(physical.instruction_set, "- rotate_z: [0, theta: 1.5708]", format="yaml")
   >>> call = rotation.calls()[0]
   >>> call.operands
   [0]
   >>> call.arguments
   {'theta': 1.5708}

The call object also has an optional ``select`` field for expected flag values:

.. code-block:: yaml

   - prepare_x_all:
       operands: [3]
       select: [reject: 0]

This says the call's ``reject`` flag is zero without noise. ``select`` is always
a list of patterns: any pattern may match, but all values within one pattern
must match. YAML reads ``[reject: 0]`` as ``[{reject: 0}]``. Omitting ``select``
or using ``[]`` adds no constraint.

For a parameter declared as ``boolean``, use YAML ``true`` or ``false``.
Python receives ``True`` or ``False``, not the integers ``1`` or ``0``:

.. doctest::

   >>> from qodec.instructions import Block, Parameter
   >>> settings = qodec.InstructionSet(
   ...     "settings", blocks=[Block("qubit", encodes=1)],
   ...     instructions=[qodec.Instruction(
   ...         "configure", parameters=[Parameter("enabled", Parameter.Kind.BOOLEAN)]
   ...     )],
   ... )
   >>> configure = Circuit(settings, "- configure: [enabled: true]", format="yaml")
   >>> enabled = configure.calls()[0].arguments["enabled"]
   >>> enabled, type(enabled) is bool
   (True, True)

Booleans are single argument values, not block labels or selection bits.
Selection patterns still use the integers ``0`` and ``1``.

All three call fields are optional and default to empty; unknown fields are
rejected. In the list shorthand, every named value is an instruction argument.
For example, ``- choose: [0, select: -3]`` supplies an argument to a parameter
named ``select``; it does not select on flags. Use the call object for selection
metadata. :class:`qodec.instructions.InstructionCall` keeps arguments and
selection in separate attributes.

Build a qodec from scratch
--------------------------

Now build a small protocol yourself: prepare a logical qubit, then measure it
using the distance-3 bit-flip repetition code. We need to describe what the
logical operations mean, which physical operations are available, and how the
gadgets connect them.

1. Define the logical ISA
~~~~~~~~~~~~~~~~~~~~~~~~~

The logical instruction set architecture (ISA) has one block type and two
instructions: prepare and measure a logical qubit in the Z basis.

``encodes=1`` says that a block holds one logical qubit. The instruction actions
describe that qubit; they do not yet say which physical circuit to use.

.. testcode:: build-qodec

   import qodec
   from qodec.actions import Observe, Stabilize
   from qodec.codes import Code
   from qodec.gadgets import Circuit, Encoding
   from qodec.instructions import Block, BlockOperand, Instruction

   logical_isa = qodec.InstructionSet(
      name="repetition3",
      blocks=[Block("repetition3", encodes=1)],
      instructions=[
         Instruction(
            mnemonic="prepare_z",
            outputs=[BlockOperand("repetition3")],
            action=[Stabilize(["Z_0"])],
         ),
         Instruction(
            mnemonic="measure_z",
            inputs=[BlockOperand("repetition3")],
            action=[Observe(["Z_0"])],
         ),
      ],
   )

2. Define the physical ISA
~~~~~~~~~~~~~~~~~~~~~~~~~~

The physical layer needs only reset and measurement of individual qubits.
These will be the operations used by the gadget circuits: ``R`` prepares a
qubit in the Z-basis zero state, and ``M`` measures it in the Z basis.

.. testcode:: build-qodec

   physical_isa = qodec.InstructionSet(
      name="stim",
      blocks=[Block("qubit", encodes=1)],
      instructions=[
         Instruction(
            mnemonic="R",
            outputs=[BlockOperand("qubit")],
            action=[Stabilize(["Z_0"])],
         ),
         Instruction(
            mnemonic="M",
            inputs=[BlockOperand("qubit")],
            action=[Observe(["Z_0"])],
         ),
      ],
   )

3. Define the code and encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code stores one logical qubit in three physical qubits. Its stabilizers
compare neighboring qubits: ``Z_0 Z_1`` and ``Z_1 Z_2`` both have sign +1 in
the code space. The logical X acts on all three qubits; the logical Z can be
read from qubit 0.

An encoding places this code in a circuit. Here ``support=["0", "1", "2"]``
puts code qubits 0, 1, and 2 on circuit qubits with those same labels.

.. testcode:: build-qodec

   code = Code(
      name="repetition3",
      stabilizers=["Z_0 Z_1", "Z_1 Z_2"],
      x=["X_0 X_1 X_2"],
      z=["Z_0"],
   )
   encoding = Encoding(code=code, support=["0", "1", "2"])

4. Define the gadgets
~~~~~~~~~~~~~~~~~~~~~

The preparation gadget resets all three qubits. The measurement gadget measures
them and connects those three results to the code's stabilizers and logical Z.

Each equation below is a list of bit references to XOR together. The first
check compares measurement bits 0 and 1 with the input's first stabilizer sign;
the second does the same for bits 1 and 2. ``in[0]`` means the gadget's first
input encoding. The logical readout relates measurement bit 0 to that input's
logical Z sign.

``readouts`` lists the instruction's ``Observe`` outcomes first, then any flags.
This instruction has one outcome and no flags.

.. testcode:: build-qodec

   prepare = qodec.Gadget(
      implements=logical_isa.instructions["prepare_z"],
      circuit=Circuit(instruction_set=physical_isa, source="R 0 1 2", format="stim"),
      outputs=[encoding],
   )
   measure = qodec.Gadget(
      implements=logical_isa.instructions["measure_z"],
      circuit=Circuit(instruction_set=physical_isa, source="M 0 1 2", format="stim"),
      inputs=[encoding],
      checks=[
         ["circuit.readouts[0:2]", "in[0].stabilizers[0]"],
         ["circuit.readouts[1:3]", "in[0].stabilizers[1]"],
      ],
      readouts=[["circuit.readouts[0]", "in[0].z[0]"]],
   )

5. Assemble the layers
~~~~~~~~~~~~~~~~~~~~~~

Put the logical layer first, with its two gadgets. The physical layer comes
last and has no gadgets: there is no lower layer to implement its operations.

.. testcode:: build-qodec

   codec = qodec.Qodec(
      layers=[
         qodec.Layer(logical_isa, gadgets=[prepare, measure]),
         qodec.Layer(physical_isa),
      ],
      name="repetition3",
   )
   assert len(measure.circuit.readouts) == 3
   assert len(measure.readouts) == 1
   assert qodec.Qodec.loads(codec.dumps()).name == "repetition3"

The protocol is ready to inspect or save. As in the loaded example, the
measurement circuit produces three bits, while the gadget exposes one logical
result.

Readouts can be reused directly, without extracting equations or losing names:

.. testcode:: build-qodec

   draft = qodec.Gadget(
      measure.implements,
      measure.circuit,
      inputs=measure.inputs,
      checks=[],
      readouts=measure.readouts,
   )
   assert draft.readouts == measure.readouts
   assert str(draft.readouts[0]) == '["circuit.readouts[0]", "in[0].z[0]"]'

``str(readout)`` prints the authored parity list, or a named dictionary when
the readout has a name. ``repr(readout)`` also shows its position and flag role.
When copying a readout into another gadget, its name and equation survive;
its position and flag role come from the destination instruction and list order.

References
----------

The strings supplied to those equations are paths into the gadget. Construction
and loading parse them into immutable :class:`qodec.Reference` values,
retaining their spelling and caching parsed fields for reuse. For example, ``circuit.readouts[0:2]`` selects
bits 0 and 1, just like a Python slice:

.. doctest::

   >>> from qodec import Reference
   >>> reference = Reference("circuit.readouts[0:2]")
   >>> reference.path
   'circuit.readouts[0:2]'
   >>> [term.path for term in reference.expand()]
   ['circuit.readouts[0]', 'circuit.readouts[1]']
   >>> Reference("in[0].stabilizers[01]") == Reference("in[0].stabilizers[1]")
   True
   >>> Reference("in[0].stabilizers[1]") == "in[0].stabilizers[1]"
   False
   >>> Reference("out[00].code.z[01]").expand()[0].path
   'out[0].z[1]'

``path`` keeps the original spelling, while equality and hashing compare
normalized reference addresses. Strings must be converted for equality.
Expansion returns canonical references, including for single-index paths.
Slices stay compact until expanded. Invalid paths and empty selectors raise
``ValueError`` when supplied to ``Reference`` or a gadget; invalid files raise
``QodecLoadError`` during loading. A failed equation setter leaves its previous
value unchanged.
Syntax checking does not establish that the referenced bit or encoding exists
in a particular gadget.

``Reference`` also accepts general model addresses for ``resolve``, such as
``metadata["description"]``. General model addresses cannot be
used in equations or frame keys. See :doc:`nodes` for model lookup and selections.

``segments`` lists the path's fields, literal mapping keys, indices, slices, and unions:

.. doctest::

   >>> Reference("in[0].z[0]").segments
   (Reference.Field(name='in'), Reference.Index(value=0), Reference.Field(name='z'), Reference.Index(value=0))
   >>> Reference('layers[0].gadgets["measure_z"]').segments[-1]
   Reference.Key(value='measure_z')
   >>> Reference(None)
   Traceback (most recent call last):
      ...
   TypeError: ...

The constructor accepts only strings and existing references. A malformed
model path and a valid model address supplied as a parity term have different
error messages.

Mutation
--------

Owned collections are live views. Item assignment, append, deletion, and clear
write through to the owner. Whole-property assignment replaces the contents;
previously obtained views see the replacement. Individual equations and readout
descriptors are immutable: replace an entry to change an equation.

.. testcode:: build-qodec

   snapshot = tuple(draft.checks)
   draft.checks = measure.checks
   assert snapshot == ()
   assert draft.checks == measure.checks
   draft.readouts = measure.readouts

Collection getters use standard ``MutableSequence`` and ``MutableMapping``
annotations. Typed item edits use the same values those collections return:
strings for code operators, ``Check`` tuples for checks and frame equations,
and ``Readout`` values for gadget readouts. Frame keys are authored strings.

Whole-property assignment accepts shorthand inputs, such as lists of reference
strings or ``PauliExpression`` values. For example:

.. testcode:: build-qodec

   from qodec import Reference

   draft.checks = [["circuit.readouts[0]"]]
   draft.checks.append((Reference("circuit.readouts[1]"),))
   assert draft.checks[-1] == (Reference("circuit.readouts[1]"),)

Broader shorthand item writes still work at runtime, but the standard collection
annotations deliberately do not model those conversions. Code operator strings
such as ``X_0 Z_1`` define Paulis; they are not ``Reference`` addresses.

Metadata is live too, including its nested dictionaries and lists:

.. doctest::

   >>> repetition3.metadata["note"] = "Ready for review"
   >>> repetition3.metadata["note"]
   'Ready for review'

An encoding's support is a live sequence. The encoding itself is shared:
changing it takes effect on every gadget referencing it.

.. testcode:: build-qodec

   boundary = measure.inputs[0]
   original_support = list(boundary.support)
   boundary.support = ["3", "4", "5"]
   assert measure.inputs[0].support == ["3", "4", "5"]
   assert prepare.outputs[0].support == ["3", "4", "5"]
   boundary.support = original_support

Layers, gadgets, instructions, encodings, and codes retain object identity.
Instructions are mutable except for their mnemonic. A loaded gadget's
``implements`` is the same instruction object as its layer's declaration.
Editing that instruction changes both views; replacing a mapping entry changes
only that entry, not other references to the old object. Keys must match
instruction and gadget mnemonics. ``Layer.codes`` keys instead name block types.

Removing an entry from a live mapping immediately changes the model:

.. doctest::

   >>> gadgets = logical.gadgets
   >>> idle = gadgets.pop("idle")
   >>> "idle" in logical.gadgets
   False
   >>> gadgets["idle"] = idle

A change to a shared ``Circuit.instruction_set``, ``Layer.instruction_set``, or ``Encoding.code`` is
visible wherever that object is used.

:meth:`qodec.Qodec.slice` also shares its retained layers, except for the new
bottom layer, which has no gadgets and shares only its ISA.

Derived indexes, ``Qodec.codes`` and ``Qodec.instruction_sets``, are read-only
live mappings. Parsed ``Circuit.calls()`` results are standalone call objects;
editing them does not rewrite source. Action and condition collections are
immutable tuples or mappings. Use ``list(view)`` and ``dict(view)`` for explicit
container snapshots; mutable objects inside them remain shared.

Copying
-------

``copy.copy`` returns a new outer object while sharing its model children.
``copy.deepcopy`` detaches mutable children and preserves repeated references
inside the new graph, including across objects copied together. Both preserve
drafts and loaded layout history without parsing source or accessing files.
They do not copy external files or change save destinations.

.. doctest::

   >>> from copy import copy, deepcopy
   >>> shallow = copy(repetition3)
   >>> shallow.layers[0] is repetition3.layers[0]
   True
   >>> detached = deepcopy(repetition3)
   >>> detached == repetition3
   True
   >>> detached.layers[0] is repetition3.layers[0]
   False
   >>> original = qodec.Instruction("prepare", description="Retained")
   >>> flagged = original.__replace__(flags=["reject"])
   >>> flagged.description, list(original.flags), list(flagged.flags)
   ('Retained', [], ['reject'])

On Python 3.13+, use ``copy.replace(original, flags=["reject"])``. The
``__replace__`` protocol also works directly on older supported Python versions.
It returns a new outer object, preserves unspecified fields and shared children,
and accepts constructor keyword names. Unknown fields raise ``TypeError``;
explicit ``None`` clears nullable fields. Replacement applies constructor
guards, not protocol auditing, and does not install the result into any owner.
There are no direct ``copy`` or ``replace`` methods on model objects.

Rust uses ``Clone`` and struct-update syntax for shallow copies and replacements.

Checking preservation
---------------------

Call :meth:`qodec.Qodec.validate` after constructing or editing a protocol:

.. doctest::

   >>> repetition3.validate()

It checks whether current resolved values can be preserved without guessing or
discarding information. Ambiguous names, conflicting layer code bindings, and
unaligned encodings raise ``ValueError``. The bottom layer cannot have gadgets
because there is no next layer to supply their target. These guards also run
on load and before writing. They do not certify protocol correctness.

Incomplete drafts can be saved:

.. doctest::

   >>> draft = qodec.Qodec([])
   >>> draft.validate()
   >>> qodec.Qodec.loads(draft.dumps()).layers
   []

Use ``qdk.ec.audit`` for protocol completeness, matching code-list lengths,
capacities, reference bounds, parameter uses, circuit validity, and algebra.
Malformed circuit text and incomplete readout lists can be preserved for
that audit. Interpreting accessors and analysis routines check their own
preconditions; successful loading does not guarantee they will succeed.

Audit requires a compatible QDK build exposing ``qdk.ec.audit`` and its ``ec``
dependencies. It is not part of the qodec package:

.. code-block:: python

   import qdk.ec

   report = qdk.ec.audit(repetition3, promote_warnings=True)
   print(report)

``promote_warnings=True`` makes unsupported verification and other warnings
fail ``report.ok`` as well as errors. Informational diagnostics do not fail it.
The repetition3 example currently reports one unsupported rotation-action
warning. Its conserved output-stabilizer signs are verified, but its logical
rotation is not. An unsupported check is not a proven mismatch, and a clean report
would not establish distance or fault tolerance.

Explicit output frames
----------------------

``Gadget.frames`` is a sparse dictionary of additional output logical-sign
corrections. The values accept references and integer bits, as checks and
readouts do. A literal ``1`` complements the XOR; booleans and other numbers
are rejected. Missing entries and empty equations apply no additional correction.
They do not reset incoming frames. The mapping is live; its term tuples are
immutable. Assign a new equation to a mapping entry to edit it.

Frame values may reference ``circuit.readouts[...]`` and ``readouts[...]``
aliases resolving entirely to circuit readouts and literal bits. Incoming and
output encoding signs are not permitted in frame values, even through aliases.
Audit reports them as invalid frame declarations, not unsupported analysis.
Checks and readout equations accept input and output encoding-sign references.

.. doctest::

   >>> from qodec import Gadget, Instruction, InstructionSet
   >>> from qodec.gadgets import Circuit
   >>> draft = Gadget(Instruction("draft"), Circuit(InstructionSet("physical"), "[]", format="yaml"))
   >>> draft.frames = {"out[0].z[0]": ["circuit.readouts[0]", 1]}
   >>> draft.frames["out[0].z[0]"][1]
   1
   >>> snapshot = dict(draft.frames)
   >>> snapshot.clear()
   >>> len(draft.frames)
   1

Frame construction and whole-property assignment accept strings or parsed
references as keys. Iteration returns authored strings, so typed live edits use
string keys and ``Check`` tuple values:

.. doctest::

   >>> target = Reference("out[0].x[0]")
   >>> draft.frames[target.path] = (0,)
   >>> draft.frames[target.path]
   (0,)
   >>> target.path in list(draft.frames)
   True
   >>> del draft.frames[target.path]

Runtime mapping operations also accept parsed-reference keys and shorthand
equations. Use whole-property assignment when those forms need to type-check.

This is a preservable draft, not a valid implementation: it has no output
encoding or measurement. Audit checks those bounds and the interpreted action.
In a complete gadget, a Z-sign correction represents a logical X Pauli, and
conversely. The consuming tool must apply the declared map when interpreting
outputs and when calculating fault effects.

Errors
------

Catch :class:`qodec.QodecLoadError` for loading failures and
:class:`qodec.QodecSaveError` for saving failures. Both inherit from
:class:`qodec.QodecError` when you want to handle them together.

For example, a missing file is a loading error:

.. doctest::

   >>> with TemporaryDirectory() as directory:
   ...     try:
   ...         qodec.Qodec.load(Path(directory) / "missing.yaml")
   ...     except qodec.QodecLoadError:
   ...         print("Protocol file could not be loaded")
   Protocol file could not be loaded

``Code`` checks Pauli syntax and ``InstructionSet`` checks map-name uniqueness
at construction, load, and save. Code X/Z lists can be edited independently.
Circuit text is parsed only when an operation needs it, such as ``calls``,
``qubits``, or ``readouts``. An unknown instruction or malformed source raises
``ValueError`` from those accessors, not ``QodecLoadError`` during loading.
Unknown languages can be stored with an explicit ``format`` tag.
For example, this circuit retains its source but has no parser:

.. doctest::

   >>> opaque = Circuit(physical.instruction_set, "custom circuit text", format="custom-ir")
   >>> opaque.source
   'custom circuit text'
   >>> try:
   ...     opaque.calls()
   ... except ValueError:
   ...     print("Circuit source cannot be interpreted")
   Circuit source cannot be interpreted

Argument conversion can also raise
``TypeError`` or ``OverflowError``; each method's reference entry lists its
exceptions.