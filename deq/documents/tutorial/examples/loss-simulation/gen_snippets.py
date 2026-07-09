"""Generate Mako-source snippets for the QDK loss-simulation chapter.

Extracts the ``GADGET Syndrome`` block out of ``repetition_code.deq`` so
the tutorial's ``## What's in the .deq`` section can link to it via the
standard ``highlight_deq.py`` pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snippet_utils import extract_block, write_snippet

this_dir = os.path.dirname(os.path.abspath(__file__))
source = os.path.join(this_dir, "repetition_code.deq")

with open(source, encoding="utf-8") as f:
    text = f.read()

write_snippet(
    os.path.join(this_dir, "snippet_syndrome.deq"),
    extract_block(text, "GADGET", "Syndrome"),
)

write_snippet(
    os.path.join(this_dir, "snippet_prepareone.deq"),
    extract_block(text, "GADGET", "PrepareOne"),
)
