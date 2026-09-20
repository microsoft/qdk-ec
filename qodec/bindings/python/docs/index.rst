qodec Python API
================

QEC tools provide codes, circuits, and decoders. qodec organizes these parts and
records how they fit together: which instruction a circuit implements, how its
qubits are encoded, and what its measurements mean.

A logical instruction says what to do. A gadget supplies its implementation in
the next layer down. Keeping the instruction separate lets you compare or
replace gadgets without redefining the operation. You choose the physical ISA
and the encoding levels above it; consuming tools supply analysis and execution.

Start with :doc:`usage` to open a protocol, inspect its measurements, and build
a small example yourself. The :doc:`API reference <autoapi/qodec/index>` covers
the objects used along the way.

.. toctree::
   :maxdepth: 2

   usage
   nodes
   autoapi/qodec/index