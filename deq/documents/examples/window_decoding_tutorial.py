"""Helper functions for the window decoding tutorial notebook.

This module is imported by ``window_decoding_tutorial.ipynb`` to keep the
notebook concise.  All heavy utility functions live here.
"""

# pylint: disable=no-member
# (protobuf-generated modules lack static member info)


import atexit
import json
import random
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import grpc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, Image, display

import deq.proto.deq_bin_pb2 as pb2
import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.visualizer_pb2 as vis_pb
from deq.cli.jit import parse_jit_program
from deq.proto import coordinator_pb2, jit_controller_pb2_grpc, util_pb2
from deq.visual.render import render_to_png
from deq.visual.svg_player import SVGFrame, SVGPlayer  # pylint: disable=unused-import
from deq.visual.window_trace import STATE_COLORS, STATE_LABELS, WindowTraceVisualizer

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def show(png_data: bytes, label: str = "") -> None:
    """Display a PNG image inline with an optional label."""
    if label:
        display(HTML(f"<b>{label}</b>"))
    display(Image(data=png_data))


# ---------------------------------------------------------------------------
# Gadget library helpers
# ---------------------------------------------------------------------------


def build_single_gadget_library(
    jit_lib: jit_pb.JitLibrary,
    gtype_index: int,
) -> pb2.Library:
    """Build a Library proto showing a single gadget type."""
    jit_gt = jit_lib.gadget_types[gtype_index]
    lib = pb2.Library()
    gt = lib.gadget_types.add()
    gt.CopyFrom(jit_gt.base)
    gt.gtype = 1
    for pt in jit_lib.port_types:
        lib_pt = lib.port_types.add()
        lib_pt.CopyFrom(pt.base)
    instr = lib.program.add()
    instr.gadget.CopyFrom(
        pb2.Gadget(gtype=1, gid=1, position=vis_pb.Position(t=0, i=0, j=0))
    )
    return lib


def render_gadget_types(jit_library: jit_pb.JitLibrary) -> None:
    """Render every gadget type in *jit_library* in realization view."""
    for i, gt in enumerate(jit_library.gadget_types):
        name = gt.base.name
        lib = build_single_gadget_library(jit_library, i)
        png = render_to_png(
            lib,
            width=400,
            height=300,
            display_mode={"1": {"showBlock": False, "showRealization": True}},
            background="white",
            gate_style="front",
            camera_type="orthographic",
            camera_position={"x": 5.13, "y": 3.36, "z": 10},
            orbit_target={"x": 5.13, "y": 3.36, "z": 0},
        )
        show(png, label=name)


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

_server_process: Optional[subprocess.Popen[str]] = None
_server_port: Optional[int] = None


def _read_line_with_timeout(
    process: subprocess.Popen[str],
    timeout: int = 30,
) -> Optional[str]:
    result: list[Optional[str]] = [None]

    def reader() -> None:
        assert process.stdout is not None
        result[0] = process.stdout.readline()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        raise TimeoutError(f"Timeout after {timeout}s")
    return result[0]


def cleanup_server() -> None:
    """Terminate the running deq server (if any)."""
    global _server_process, _server_port  # pylint: disable=global-statement
    if _server_process is not None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_process.kill()
        _server_process = None
    _server_port = None


atexit.register(cleanup_server)


def start_server(
    jit_path: str | Path,
    coordinator: str = "window",
    buffer_radius: int = 1,
    lookahead_radius: int | None = 0,
    trace_path: Optional[str | Path] = None,
    decoder: str = "mock",
    decoder_config: Optional[Dict[str, Any]] = None,
) -> int:
    """Start a deq server and return the port number."""
    global _server_process, _server_port  # pylint: disable=global-statement
    cleanup_server()

    coord_config: Dict[str, Any] = {"buffer_radius": buffer_radius}
    if lookahead_radius is not None:
        coord_config["lookahead_radius"] = lookahead_radius
    if trace_path:
        coord_config["trace_filepath"] = str(trace_path)

    if decoder_config is None:
        decoder_config = {"decode_delay_ms": 100}

    cmd = [
        sys.executable,
        "-m",
        "deq.runtime",
        "server",
        "--addr",
        "[::]:0",
        "--coordinator",
        coordinator,
        "--coordinator-config",
        json.dumps(coord_config),
        "--controller",
        "jit",
        "--controller-config",
        json.dumps({"filepath": str(jit_path)}),
        "--decoder",
        decoder,
        "--decoder-config",
        json.dumps(decoder_config),
    ]

    _server_process = subprocess.Popen(  # pylint: disable=consider-using-with
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    port_pattern = re.compile(r"port=(\d+)")
    for _ in range(20):
        line = _read_line_with_timeout(_server_process, timeout=30)
        if line:
            match = port_pattern.search(line)
            if match:
                _server_port = int(match.group(1))
                break

    if _server_port is None:
        cleanup_server()
        raise RuntimeError("Failed to start server")

    channel = grpc.insecure_channel(f"localhost:{_server_port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=10)
    finally:
        channel.close()
    print(f"Server started on port {_server_port}")
    return _server_port


# ---------------------------------------------------------------------------
# Circuit execution
# ---------------------------------------------------------------------------


def build_gtype_info(
    jit_library: jit_pb.JitLibrary,
) -> Dict[int, Dict[str, Any]]:
    """Build a ``{gtype: {"name": ..., "measurements": ...}}`` mapping."""
    info: Dict[int, Dict[str, Any]] = {}
    for gt in jit_library.gadget_types:
        info[gt.base.gtype] = {
            "name": gt.base.name,
            "measurements": len(gt.base.measurements),
        }
    return info


def run_circuit(
    jit_lib: jit_pb.JitLibrary,
    program_str: str,
    port: int,
    gtype_to_info: Dict[int, Dict[str, Any]],
) -> None:
    """Execute a circuit and decode all gadgets concurrently (batch mode)."""
    channel = grpc.insecure_channel(f"localhost:{port}")
    client = jit_controller_pb2_grpc.JitControllerStub(channel)

    instructions = parse_jit_program(jit_lib, program_str)

    for instr in instructions:
        client.Execute(instr)

    def decode_one(instr: Any) -> Any:
        gtype = instr.gadget.gtype
        num_meas = gtype_to_info[gtype]["measurements"]
        num_bytes = (num_meas + 7) // 8
        outcomes = coordinator_pb2.Outcomes(
            gid=instr.gadget.gid,
            outcomes=util_pb2.BitVector(
                data=bytes(num_bytes),
                size=num_meas,
            ),
        )
        return client.Decode(outcomes)

    with ThreadPoolExecutor(max_workers=len(instructions)) as executor:
        futures = {executor.submit(decode_one, instr): instr for instr in instructions}
        for future in as_completed(futures):
            instr = futures[future]
            future.result()
            name = gtype_to_info[instr.gadget.gtype]["name"]
            print(f"  gid={instr.gadget.gid} ({name}): decoded")

    client.Reset(coordinator_pb2.ResetRequest())
    channel.close()
    print("Circuit completed and trace flushed.")


def run_circuit_streaming(
    jit_lib: jit_pb.JitLibrary,
    program_str: str,
    port: int,
    gtype_to_info: Dict[int, Dict[str, Any]],
    delay_seconds: float = 0.15,
) -> None:
    """Execute a circuit with streaming: execute and decode each gadget
    with a delay between them, simulating incremental measurement arrival."""
    import time as _time

    channel = grpc.insecure_channel(f"localhost:{port}")
    client = jit_controller_pb2_grpc.JitControllerStub(channel)

    instructions = parse_jit_program(jit_lib, program_str)

    def decode_one(instr: Any) -> Any:
        gtype = instr.gadget.gtype
        num_meas = gtype_to_info[gtype]["measurements"]
        num_bytes = (num_meas + 7) // 8
        outcomes = coordinator_pb2.Outcomes(
            gid=instr.gadget.gid,
            outcomes=util_pb2.BitVector(
                data=bytes(num_bytes),
                size=num_meas,
            ),
        )
        return client.Decode(outcomes)

    # Submit execute + decode for each gadget with a delay between them.
    # Decodes are submitted to a thread pool so they run concurrently
    # while we continue executing the next gadgets.
    with ThreadPoolExecutor(max_workers=len(instructions)) as executor:
        futures = {}
        for i, instr in enumerate(instructions):
            if i > 0:
                _time.sleep(delay_seconds)
            client.Execute(instr)
            futures[executor.submit(decode_one, instr)] = instr
            name = gtype_to_info[instr.gadget.gtype]["name"]
            print(f"  gid={instr.gadget.gid} ({name}): executed & decode submitted")

        for future in as_completed(futures):
            instr = futures[future]
            future.result()
            name = gtype_to_info[instr.gadget.gtype]["name"]
            print(f"  gid={instr.gadget.gid} ({name}): decoded")

    client.Reset(coordinator_pb2.ResetRequest())
    channel.close()
    print("Circuit completed and trace flushed.")


# ---------------------------------------------------------------------------
# Timeline plot
# ---------------------------------------------------------------------------


def plot_timeline(
    timeline_data: Dict[str, Any],
    title: str = "Window Decode Timeline",
) -> None:
    """Plot a Gantt-style timeline of decode windows."""
    windows = timeline_data["windows"]
    if not windows:
        print("No windows to plot")
        return

    t0 = timeline_data["min_time_ns"]
    _, ax = plt.subplots(figsize=(12, max(3, len(windows) * 0.4)))
    cmap = plt.colormaps["Set3"]
    colors = cmap(np.linspace(0, 1, max(len(windows), 1)))

    for i, w in enumerate(windows):
        start_ms = (w["start_ns"] - t0) / 1e6
        end_ms = ((w["end_ns"] or w["start_ns"]) - t0) / 1e6
        duration = max(end_ms - start_ms, 0.01)

        ax.barh(
            i,
            duration,
            left=start_ms,
            height=0.6,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.5,
        )
        buffer = sorted(set(w.get("window", [])) - set(w["committing_gids"]))
        label = f"L{w['leader_gid']}: commit={w['committing_gids']}, buf={buffer}"
        ax.text(start_ms + duration / 2, i, label, ha="center", va="center", fontsize=7)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Window")
    ax.set_title(title)
    ax.set_yticks(range(len(windows)))
    ax.set_yticklabels([f"W{i}" for i in range(len(windows))])
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Random circuit generator
# ---------------------------------------------------------------------------


def generate_random_program(
    n_qubits: int,
    n_gates: int,
    seed: int = 42,
) -> Tuple[str, Dict[str, Any]]:
    """Generate a random circuit program string and 2D layout.

    Maintains a logical clock cycle per qubit:
    - PrepareZ and Idle advance the clock by 1.
    - TransversalCNOT and MeasureZ do not advance the clock.

    Constraints:
    - CNOTs only between adjacent qubits (q_i, q_{i+1}).
    - Before a CNOT, idle cycles are inserted so both qubits
      reach the same clock.  Padding idles do not count toward *n_gates*.
    - Each qubit must have at least one idle between consecutive CNOTs.
      Padding idles are inserted automatically if needed.

    Returns:
        ``(program_str, layout)`` where *layout* has keys ``n_qubits``,
        ``max_cycle``, and ``gadgets`` (a dict mapping gid to position info).
    """
    rng = random.Random(seed)
    parts: List[str] = []
    # Each qubit gets a wire id; wires are reused in-place.
    wire_of_qubit: List[int] = []
    clock: List[int] = []
    needs_idle: List[bool] = []  # True if qubit must idle before next CX
    gadget_layout: Dict[int, Dict[str, Any]] = {}
    gid = 1

    for q in range(n_qubits):
        parts.append(f"PrepareZ {q}")
        gadget_layout[gid] = {"qubit": q, "cycle": 0, "kind": "prepare"}
        gid += 1
        wire_of_qubit.append(q)
        clock.append(1)
        needs_idle.append(False)

    for _ in range(n_gates):
        gate_type = (
            rng.choices(["idle", "cnot"], weights=[80, 20])[0]
            if n_qubits > 1
            else "idle"
        )
        if gate_type == "idle":
            q = rng.randint(0, n_qubits - 1)
            w = wire_of_qubit[q]
            parts.append(f"Idle {w}")
            gadget_layout[gid] = {"qubit": q, "cycle": clock[q], "kind": "idle"}
            gid += 1
            clock[q] += 1
            needs_idle[q] = False
        elif gate_type == "cnot":
            q1 = rng.randint(0, n_qubits - 2)
            q2 = q1 + 1
            # Ensure at least one idle separates consecutive CX gates
            for q in (q1, q2):
                if needs_idle[q]:
                    w = wire_of_qubit[q]
                    parts.append(f"Idle {w}")
                    gadget_layout[gid] = {
                        "qubit": q,
                        "cycle": clock[q],
                        "kind": "idle",
                    }
                    gid += 1
                    clock[q] += 1
                    needs_idle[q] = False
            target_clock = max(clock[q1], clock[q2])
            for q in (q1, q2):
                while clock[q] < target_clock:
                    w = wire_of_qubit[q]
                    parts.append(f"Idle {w}")
                    gadget_layout[gid] = {
                        "qubit": q,
                        "cycle": clock[q],
                        "kind": "idle",
                    }
                    gid += 1
                    clock[q] += 1
            w1, w2 = wire_of_qubit[q1], wire_of_qubit[q2]
            parts.append(f"TransversalCNOT {w1} {w2}")
            gadget_layout[gid] = {
                "qubit": q1,
                "qubit2": q2,
                "cycle": clock[q1],
                "kind": "cnot",
            }
            gid += 1
            needs_idle[q1] = True
            needs_idle[q2] = True

    max_cycle = max(clock)
    for q in range(n_qubits):
        while clock[q] < max_cycle:
            w = wire_of_qubit[q]
            parts.append(f"Idle {w}")
            gadget_layout[gid] = {"qubit": q, "cycle": clock[q], "kind": "idle"}
            gid += 1
            clock[q] += 1
        w = wire_of_qubit[q]
        parts.append(f"MeasureZ {w}")
        gadget_layout[gid] = {"qubit": q, "cycle": max_cycle, "kind": "measure"}
        gid += 1

    layout: Dict[str, Any] = {
        "n_qubits": n_qubits,
        "max_cycle": max_cycle,
        "gadgets": gadget_layout,
    }
    return "\n".join(parts), layout


def build_layout_from_program(program: str) -> Dict[str, Any]:
    """Build a 2D layout dict from a ``.deq`` PROGRAM body string.

    Parses statements in shortcut form (``PrepareZ 0``, ``Idle 0``,
    ``TransversalCNOT 0 1``, ``MeasureZ 0``) or explicit form
    (``PrepareZ OUT(0)``, ``Idle IN(0) OUT(0)``) and returns a layout
    compatible with :func:`plot_2d_snapshot`.
    """
    lines = [s.strip() for s in program.splitlines() if s.strip()]
    wire_to_qubit: Dict[int, int] = {}
    clock: Dict[int, int] = {}
    gadget_layout: Dict[int, Dict[str, Any]] = {}
    next_qubit = 0
    gid = 1

    def _parse_ints(s: str) -> List[int]:
        return [int(x) for x in s.split() if x.isdigit()]

    for line in lines:
        if line.startswith("VIRTUAL"):
            continue
        parts = line.split()
        name = parts[0]

        in_match = re.search(r"IN\(([^)]+)\)", line)
        out_match = re.search(r"OUT\(([^)]+)\)", line)
        in_wires = _parse_ints(in_match.group(1)) if in_match else []
        out_wires = _parse_ints(out_match.group(1)) if out_match else []

        # Shortcut form: bare integers after gadget name (no IN/OUT)
        if not in_wires and not out_wires:
            bare_ints = [int(p) for p in parts[1:] if p.isdigit()]
        else:
            bare_ints = []

        # Merge: use explicit wires if present, otherwise bare ints
        wires = bare_ints if (not in_wires and not out_wires) else []

        if name == "PrepareZ" or name.startswith("Prepare"):
            for w in (out_wires or wires):
                if w not in wire_to_qubit:
                    wire_to_qubit[w] = next_qubit
                    next_qubit += 1
                    clock[wire_to_qubit[w]] = 1
            first_wire = (out_wires or wires or [0])[0]
            gadget_layout[gid] = {
                "qubit": wire_to_qubit.get(first_wire, 0),
                "cycle": 0,
                "kind": "prepare",
            }
        elif name == "MeasureZ" or name.startswith("Measure"):
            first_wire = (in_wires or wires or [0])[0]
            q = wire_to_qubit.get(first_wire, 0)
            gadget_layout[gid] = {
                "qubit": q,
                "cycle": clock.get(q, 0),
                "kind": "measure",
            }
        elif name == "Idle":
            first_wire = (in_wires or wires or [0])[0]
            q = wire_to_qubit.get(first_wire, 0)
            gadget_layout[gid] = {
                "qubit": q,
                "cycle": clock.get(q, 0),
                "kind": "idle",
            }
            clock[q] = clock.get(q, 0) + 1
        elif name == "TransversalCNOT" or name.startswith("Cx"):
            all_wires = in_wires or wires
            w1 = all_wires[0] if len(all_wires) > 0 else 0
            w2 = all_wires[1] if len(all_wires) > 1 else 1
            q1 = wire_to_qubit.get(w1, 0)
            q2 = wire_to_qubit.get(w2, 1)
            gadget_layout[gid] = {
                "qubit": q1,
                "qubit2": q2,
                "cycle": max(clock.get(q1, 0), clock.get(q2, 0)),
                "kind": "cnot",
            }
        else:
            gadget_layout[gid] = {"qubit": 0, "cycle": 0, "kind": "unknown"}

        gid += 1

    max_cycle = max(clock.values()) if clock else 0
    return {
        "n_qubits": next_qubit,
        "max_cycle": max_cycle,
        "gadgets": gadget_layout,
    }


# ---------------------------------------------------------------------------
# 2D snapshot renderer (matplotlib)
# ---------------------------------------------------------------------------


def compute_gadget_roles(
    viz: WindowTraceVisualizer,
    timestamp_ns: int,
    shot_index: int = 0,
) -> Dict[int, str]:
    """Classify each gadget at *timestamp_ns* into a rendering role."""
    gadgets = viz._parse_shot(shot_index)
    shot = viz.trace.shots[shot_index]

    leader_windows: Dict[int, Dict[str, Any]] = {}
    for event in shot.events:
        ev = event.WhichOneof("event")
        if ev == "decode":
            de = event.decode
            if de.is_leader:
                leader_windows[de.leader_gid] = {
                    "window": set(de.window),
                    "committing_gids": set(de.committing_gids),
                    "start_ns": event.timestamp_ns,
                    "end_ns": None,
                }
        elif ev == "decode_finished":
            df = event.decode_finished
            if df.leader_gid in leader_windows:
                leader_windows[df.leader_gid]["end_ns"] = event.timestamp_ns

    committed_gids: set[int] = set()
    active_committing: set[int] = set()
    active_buffer: set[int] = set()
    for info in leader_windows.values():
        if timestamp_ns >= info["start_ns"]:
            if info["end_ns"] is not None and timestamp_ns >= info["end_ns"]:
                committed_gids.update(info["committing_gids"])
            else:
                active_committing.update(info["committing_gids"])
                active_buffer.update(info["window"] - info["committing_gids"])

    gid_to_role: Dict[int, str] = {}
    for g in gadgets:
        if timestamp_ns < g.execute_time:
            gid_to_role[g.gid] = "not_executed"
        elif g.gid in active_committing:
            gid_to_role[g.gid] = "committing"
        elif g.gid in committed_gids and g.gid in active_buffer:
            gid_to_role[g.gid] = "committed_buffer"
        elif g.gid in committed_gids:
            gid_to_role[g.gid] = "committed"
        elif g.gid in active_buffer:
            gid_to_role[g.gid] = "buffer"
        else:
            gid_to_role[g.gid] = "executed"
    return gid_to_role


def plot_2d_snapshot(
    layout: Dict[str, Any],
    gid_to_role: Dict[int, str],
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Draw a 2D grid: qubits (rows) x clock cycles (cols), colored by role."""
    n_q = layout["n_qubits"]
    max_c = layout["max_cycle"]
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, (max_c + 1) * 0.8), max(2, n_q * 0.7)))
    ax.set_xlim(-0.5, max_c + 0.5)
    ax.set_ylim(-0.5, n_q - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(max_c + 1))
    ax.set_xticklabels([str(c) for c in range(max_c + 1)], fontsize=7)
    ax.set_yticks(range(n_q))
    ax.set_yticklabels([f"Q{q}" for q in range(n_q)], fontsize=8)
    ax.set_xlabel("Clock cycle", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9, pad=4)

    for gid, info in layout["gadgets"].items():
        role = gid_to_role.get(gid, "not_executed")
        color = STATE_COLORS.get(role, "#FFFFFF")
        ec = "#888888" if role == "not_executed" else "black"
        lw = 0.5 if role == "not_executed" else 1.0
        c = info["cycle"]
        if info["kind"] == "cnot":
            q_top = info["qubit"]
            q_bot = info.get("qubit2", q_top + 1)
            cx = c - 0.5
            hw = 0.12
            rect = mpatches.FancyBboxPatch(
                (cx - hw, q_top - 0.48),
                hw * 2,
                (q_bot - q_top) + 0.96,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor=ec,
                linewidth=lw,
            )
            ax.add_patch(rect)
            ax.text(
                cx,
                (q_top + q_bot) / 2,
                f"CX\n{gid}",
                ha="center",
                va="center",
                fontsize=4.5,
                weight="bold",
            )
        else:
            q = info["qubit"]
            rect = mpatches.FancyBboxPatch(
                (c - 0.4, q - 0.4),
                0.8,
                0.8,
                boxstyle="round,pad=0.03",
                facecolor=color,
                edgecolor=ec,
                linewidth=lw,
            )
            ax.add_patch(rect)
            lbl = {"prepare": "P", "idle": "I", "measure": "M"}.get(
                info["kind"],
                "?",
            )
            ax.text(c, q, f"{lbl}\n{gid}", ha="center", va="center", fontsize=5.5)

    for q in range(n_q):
        ax.axhline(q - 0.5, color="#DDDDDD", linewidth=0.5)
    ax.axhline(n_q - 0.5, color="#DDDDDD", linewidth=0.5)
    for c in range(max_c + 2):
        ax.axvline(c - 0.5, color="#DDDDDD", linewidth=0.5)
    return ax


def add_role_legend(fig: plt.Figure) -> None:
    """Add a shared color legend to the figure."""
    legend_order = [
        "committed",
        "committing",
        "buffer",
        "committed_buffer",
        "executed",
        "not_executed",
    ]
    handles = [
        mpatches.Patch(
            facecolor=STATE_COLORS[r],
            edgecolor="black",
            label=STATE_LABELS[r],
        )
        for r in legend_order
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(legend_order),
        fontsize=7,
        frameon=True,
    )


# ---------------------------------------------------------------------------
# SVG trace frame helpers
# ---------------------------------------------------------------------------


def precompute_trace_frames(
    viz: WindowTraceVisualizer,
    shot_index: int = 0,
) -> List[Dict[str, Any]]:
    """Pre-compute gadget roles at each key timestamp."""
    timestamps = viz.get_snapshot_timestamps(shot_index)
    gadgets = viz._parse_shot(shot_index)
    t0 = min(g.execute_time for g in gadgets) if gadgets else 0
    frames: List[Dict[str, Any]] = []
    for ts in timestamps:
        roles = compute_gadget_roles(viz, ts, shot_index)
        rel_us = max(0.0, (ts - t0) / 1e3)
        frames.append(
            {
                "timestamp_ns": ts,
                "label": f"t = +{rel_us:.0f} \u00b5s",
                "roles": roles,
            }
        )
    return frames


def render_trace_svg_frames(
    frames: Sequence[Dict[str, Any]],
    layout: Dict[str, Any],
) -> List[SVGFrame]:
    """Convert precomputed trace frames into :class:`SVGFrame` objects."""
    svg_frames: List[SVGFrame] = []
    for frame in frames:
        fig, ax = plt.subplots(
            figsize=(
                max(6, (layout["max_cycle"] + 1) * 0.8),
                max(2, layout["n_qubits"] * 0.7),
            ),
        )
        plot_2d_snapshot(layout, frame["roles"], title=frame["label"], ax=ax)
        add_role_legend(fig)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.15)
        buf = BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg_frames.append(
            SVGFrame(
                svg=buf.getvalue().decode("utf-8"),
                timestamp_ns=frame["timestamp_ns"],
                label=frame["label"],
            )
        )
    return svg_frames


def export_trace_svgs(
    frames: Sequence[Dict[str, Any]],
    layout: Dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Export each trace frame as a publication-quality SVG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        fig, ax = plt.subplots(
            figsize=(
                max(6, (layout["max_cycle"] + 1) * 0.8),
                max(2, layout["n_qubits"] * 0.7),
            ),
        )
        plot_2d_snapshot(layout, frame["roles"], title=frame["label"], ax=ax)
        add_role_legend(fig)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.15)
        path = out / f"frame_{i:03d}.svg"
        fig.savefig(str(path), format="svg", bbox_inches="tight")
        plt.close(fig)
    print(f"Exported {len(frames)} SVGs to {out}")
