# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
"""
The JIT compiler for static circuits.

Note that this is a simple wrapper that calls the asynchronous JIT compiler implemented
in Rust. The purpose of this static JIT compiler is primarily for debugging purposes:
for the testing circuits, check if the generated library is equivalent to the reference.

For the real-time JIT compiler, call the Rust function directly. The JIT compiler
compiles a JIT instruction to gadgets and check models immediately, but the error model
is generated asynchronously because it has to wait for all the output ports to be connected
before it can determine the error model.
"""

import deq_runtime
import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb


def static_jit_compiler(jit_library: jit_pb.JitLibrary) -> pb.Library:
    jit_library_bin = jit_library.SerializeToString()
    library_bin = deq_runtime.static_jit_compile(jit_library_bin)
    library = pb.Library.FromString(library_bin)
    return library
