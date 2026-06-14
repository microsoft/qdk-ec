import os
from deq.cli.jit import transpile

this_dir = os.path.dirname(__file__)

transpile(
    os.path.join(this_dir, "small_example_evaluation.deq"),
    out=os.path.join(this_dir, "small_example.deq.jit"),
    program="Simulation",
)
