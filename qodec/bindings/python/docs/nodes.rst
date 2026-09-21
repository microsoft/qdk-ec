Model paths
===========

Resolve one declared value without parsing circuit source or running analysis:

.. doctest::

   >>> import qodec
   >>> protocol = qodec.Qodec([], metadata={"enabled": True})
   >>> node = protocol.resolve('metadata["enabled"]')
   >>> node.value(bool)
   True
   >>> str(node)
   'metadata["enabled"]'
   >>> node == protocol.resolve('metadata["enabled"]')
   True
   >>> node.source_location is None
   True
   >>> from qodec import Reference
   >>> node == protocol.resolve(Reference('metadata["enabled"]'))
   True
   >>> reference = Reference('metadata["enabled"]')
   >>> reference.segments
   (Reference.Field(name='metadata'), Reference.Key(value='enabled'))
   >>> Reference("layers[1:5:2]").segments[-1]
   Reference.Slice(start=1, stop=5, step=2)
   >>> protocol.metadata["values"] = [10, 20, 30]
   >>> selected = protocol.resolve('metadata["values"][2,0,2]')
   >>> [member.value(int) for member in selected.sequence_nodes()]
   [30, 10, 30]
   >>> selected.value(tuple)
   (30, 10, 30)
   >>> from collections.abc import MutableSequence
   >>> values = protocol.resolve('metadata["values"]').value(MutableSequence)
   >>> values[0] = 40
   >>> protocol.metadata["values"][0]
   40
   >>> selected.value()
   (30, 40, 30)

An empty path selects the root. Fields use dots, literal mapping keys use
JSON-quoted brackets, and sequence indices use nonnegative brackets.
``node.resolve(path)`` navigates relative to a node. ``sequence_nodes()`` returns
a tuple of child nodes; ``mapping_nodes()`` returns a read-only mapping of string
keys to child nodes. Nodes themselves are not iterable or subscriptable.
``Reference`` stores an owner-independent address; ``Node`` selects an occurrence
in one owner. A reference's immutable ``segments`` support structural inspection
and pattern matching through ``Reference.Field``, ``Key``, ``Index``, ``Slice``,
and ``Union``.
``value()`` returns what ordinary field access returns: scalars, model objects,
immutable equations, and live collection views. For example,
``gadget.resolve("checks[1]").value(tuple)`` returns the same check tuple as
``gadget.checks[1]``. Mutable values retain their normal getter's ownership rules.

An optional positional type checks the result with ``isinstance``. It never
converts, copies, consumes an iterator, or checks element types. Use runtime types
such as ``tuple`` or ``collections.abc.Sequence``, not ``Sequence[Reference]``.
Python's normal subclass rules apply: ``True`` satisfies ``int``. Without an
expected type, static checking treats the result as ``object``; supplying a type
narrows the result to that type. Runtime-checkable protocols are also accepted.

Strings and parsed ``Reference`` values use the same grammar. Slices and unions
return a selection node. Its ``value()`` is a tuple of selected values, preserving
order and duplicates; ``sequence_nodes()`` retains individual member paths.
The outer tuple is a snapshot, not a writable selection view. Its elements
retain their normal ownership rules. Even a one-entry slice remains a selection.
Every member must exist; no partial or truncated result is returned.

Gadget boundary paths use ``in`` and ``out``. Python object properties are
``inputs`` and ``outputs``. For example,
``gadget_node.resolve("in[0].stabilizers[1]").value(str)`` returns an operator
declaration in code-local coordinates, not a measured sign or placed operator.
``gadget.resolve(reference)`` works on a gadget directly. Its nodes have
gadget-relative paths, gadget-owner identity, and no source locations.
``circuit.readouts`` requires explicit circuit interpretation and is not
part of model lookup. Gadget parity fields accept only gadget-local readout
and encoding-sign paths.

``value(expected)`` checks the requested type; ``sequence_nodes()`` and
``mapping_nodes()`` require the corresponding collection kind. Mismatches
raise ``TypeError``. Malformed
paths raise ``ValueError``; missing targets raise ``LookupError``. Use
``node.value() is None`` for an absent optional value. Getter errors propagate.
Truth tests on a node raise
``TypeError`` because existence, nullness, and a stored boolean are different.

A node follows a path within its Qodec or standalone Gadget owner.
Reordering or replacing model entries changes its target; removing the path
makes value access fail. Equality and hashing use owner identity and canonical path,
so handles to the same location compare equal even after edits.
Previously returned values keep their ordinary behavior: a saved collection view
stays bound to its original owner, while another ``node.value()`` follows the
current path. There is no assignment-through-path API; use normal model setters
or mutate a returned live view.

Loaded nodes may expose ``source_location.path`` and ``source_location.line``.
The path names the actual file, including the outer file for bundles; lines
are one-based. Locations refer to the loaded revision and are suppressed if
the model differs from its loaded snapshot. Constructed models, slices, and
in-memory bundle text have no file location. Saving does not update locations.

Source locations are optional and do not participate in protocol equality.
The complete model-path field table is in the repository's concept guide,
``docs/concepts/paths.md``.