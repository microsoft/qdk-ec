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

An empty path selects the root. Fields use dots, literal mapping keys use
JSON-quoted brackets, and sequence indices use nonnegative brackets.
``node.resolve(path)`` navigates relative to a node. ``as_sequence()`` returns
a tuple of child nodes; ``as_mapping()`` returns a read-only mapping of string
keys to child nodes. Nodes themselves are not iterable or subscriptable.

Typed ``as_*`` accessors raise ``TypeError`` rather than converting. Malformed
paths raise ``ValueError``; missing targets raise ``LookupError``. Use
``is_none`` for an absent optional value. Truth tests on a node raise
``TypeError`` because existence, nullness, and a stored boolean are different.

A node is a live path in one protocol, not a permanent object identity.
Reordering or replacing model entries changes its target; removing the path
makes value access fail. Equality and hashing use protocol identity and path,
so handles to the same location compare equal even after edits.

Loaded nodes may expose ``source_location.path`` and ``source_location.line``.
The path names the actual file, including the outer file for bundles; lines
are one-based. Locations refer to the loaded revision and are suppressed if
the model differs from its loaded snapshot. Constructed models, slices, and
in-memory bundle text have no file location. Saving does not update locations.

Source locations are optional and do not participate in protocol equality.
The complete model-path field table is in the repository's concept guide,
``docs/concepts/paths.md``.