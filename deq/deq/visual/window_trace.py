# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
"""
Window coordinator trace visualization.

Parses a WindowCoordinatorTrace protobuf trace file and extracts
timing data for visualization.

Usage::

    from deq.visual.window_trace import WindowTraceVisualizer

    viz = WindowTraceVisualizer("trace.pb", "library.deq.jit")
    timeline = viz.get_timeline_data(shot_index=0)
"""


import enum
import pathlib
from typing import Any

import deq.proto.deq_jit_pb2 as jit_pb
import deq.proto.window_coordinator_pb2 as trace_pb


class GadgetState(enum.Enum):
    """State of a gadget at a given point in time."""

    NOT_EXECUTED = "not_executed"
    EXECUTED = "executed"
    COMMITTING = "committing"
    COMMITTED = "committed"
    BUFFER = "buffer"
    COMMITTED_BUFFER = "committed_buffer"


# Color scheme for gadget states
STATE_COLORS: dict[str, str] = {
    GadgetState.NOT_EXECUTED.value: "#FFFFFF",
    GadgetState.EXECUTED.value: "#AAAAAA",
    GadgetState.COMMITTING.value: "#9B59B6",
    GadgetState.COMMITTED.value: "#7FBA00",
    GadgetState.BUFFER.value: "#FFD700",
    GadgetState.COMMITTED_BUFFER.value: "#FFE082",
}

# Human-readable labels for gadget states
STATE_LABELS: dict[str, str] = {
    GadgetState.NOT_EXECUTED.value: "Not executed",
    GadgetState.EXECUTED.value: "Executed",
    GadgetState.COMMITTING.value: "Committing",
    GadgetState.COMMITTED.value: "Committed",
    GadgetState.BUFFER.value: "Buffer",
    GadgetState.COMMITTED_BUFFER.value: "Committed + Buffer",
}


class _GadgetInfo:
    """Tracked info for a single gadget instance."""

    __slots__ = ("gid", "gtype", "execute_time", "commit_start", "commit_end")

    def __init__(self, gid: int, gtype: int, execute_time: int) -> None:
        self.gid = gid
        self.gtype = gtype
        self.execute_time = execute_time
        self.commit_start: int | None = None
        self.commit_end: int | None = None

    def state_at(self, timestamp_ns: int) -> GadgetState:
        """Determine the gadget state at a given timestamp."""
        if timestamp_ns < self.execute_time:
            return GadgetState.NOT_EXECUTED
        if self.commit_end is not None and timestamp_ns >= self.commit_end:
            return GadgetState.COMMITTED
        if self.commit_start is not None and timestamp_ns >= self.commit_start:
            return GadgetState.COMMITTING
        return GadgetState.EXECUTED


class WindowTraceVisualizer:
    """Visualize window decoding progression from a trace file.

    Parses the trace to extract gadget execution and decode events,
    then generates snapshot Library protos at key timestamps showing
    each gadget's state via color coding.

    Args:
        trace_path: Path to the WindowCoordinatorTrace protobuf file.
        jit_library_path: Path to the .deq.jit library (for gadget type info).
    """

    def __init__(
        self,
        trace_path: str | pathlib.Path,
        jit_library_path: str | pathlib.Path | None = None,
    ) -> None:
        # Load trace
        trace_path = pathlib.Path(trace_path)
        with open(trace_path, "rb") as f:
            self.trace = trace_pb.WindowCoordinatorTrace.FromString(f.read())

        # Load JIT library if provided (for gadget type names/structure)
        self.jit_library: jit_pb.JitLibrary | None = None
        if jit_library_path is not None:
            jit_library_path = pathlib.Path(jit_library_path)
            with open(jit_library_path, "rb") as f:
                self.jit_library = jit_pb.JitLibrary.FromString(f.read())

    def _parse_shot(self, shot_index: int = 0) -> list[_GadgetInfo]:
        """Parse a shot's events into GadgetInfo objects."""
        shot = self.trace.shots[shot_index]
        gadgets: dict[int, _GadgetInfo] = {}

        # Track which leader_gid is decoding which gids
        leader_commit_gids: dict[int, list[int]] = {}

        for event in shot.events:
            ev = event.WhichOneof("event")
            ts = event.timestamp_ns

            if ev == "execute_gadget":
                eg = event.execute_gadget
                gadgets[eg.gadget.gid] = _GadgetInfo(eg.gadget.gid, eg.gadget.gtype, ts)

            elif ev == "decode":
                de = event.decode
                if de.is_leader:
                    # Leader starts decoding — mark all committing_gids
                    leader_commit_gids[de.leader_gid] = list(de.committing_gids)
                    for gid in de.committing_gids:
                        if gid in gadgets:
                            gadgets[gid].commit_start = ts

            elif ev == "decode_finished":
                df = event.decode_finished
                leader_gid = df.leader_gid
                if leader_gid in leader_commit_gids:
                    for gid in leader_commit_gids[leader_gid]:
                        if gid in gadgets:
                            gadgets[gid].commit_end = ts

        # Return sorted by gid
        return sorted(gadgets.values(), key=lambda g: g.gid)

    def get_snapshot_timestamps(self, shot_index: int = 0) -> list[int]:
        """Return 2N+1 key timestamps for snapshot rendering.

        Returns timestamps just before the first window, then just after
        each window start and end event.
        """
        gadgets = self._parse_shot(shot_index)
        timestamps: set[int] = set()

        # Find the earliest execute time as the "before" timestamp
        if gadgets:
            min_exec = min(g.execute_time for g in gadgets)
            timestamps.add(min_exec - 1)  # before anything happens

        for g in gadgets:
            if g.commit_start is not None:
                timestamps.add(g.commit_start + 1)  # just after start
            if g.commit_end is not None:
                timestamps.add(g.commit_end + 1)  # just after end

        return sorted(timestamps)

    def get_timeline_data(self, shot_index: int = 0) -> dict[str, Any]:
        """Extract timing data for matplotlib timeline plots.

        Returns a dict with:
            - "gadgets": list of {gid, gtype, execute_time, commit_start, commit_end}
            - "windows": list of {leader_gid, start_ns, end_ns, committing_gids}
            - "min_time_ns": earliest timestamp
            - "max_time_ns": latest timestamp
        """
        shot = self.trace.shots[shot_index]
        gadgets_info = self._parse_shot(shot_index)

        windows: list[dict[str, Any]] = []
        leader_info: dict[int, dict[str, Any]] = {}

        for event in shot.events:
            ev = event.WhichOneof("event")
            ts = event.timestamp_ns

            if ev == "decode":
                de = event.decode
                if de.is_leader:
                    leader_info[de.leader_gid] = {
                        "leader_gid": de.leader_gid,
                        "start_ns": ts,
                        "end_ns": None,
                        "committing_gids": list(de.committing_gids),
                        "window": list(de.window),
                    }

            elif ev == "decode_finished":
                df = event.decode_finished
                if df.leader_gid in leader_info:
                    leader_info[df.leader_gid]["end_ns"] = ts

        windows = sorted(leader_info.values(), key=lambda w: w["start_ns"])

        all_times = [g.execute_time for g in gadgets_info]
        for w in windows:
            all_times.append(w["start_ns"])
            if w["end_ns"] is not None:
                all_times.append(w["end_ns"])

        return {
            "gadgets": [
                {
                    "gid": g.gid,
                    "gtype": g.gtype,
                    "execute_time": g.execute_time,
                    "commit_start": g.commit_start,
                    "commit_end": g.commit_end,
                }
                for g in gadgets_info
            ],
            "windows": windows,
            "min_time_ns": min(all_times) if all_times else 0,
            "max_time_ns": max(all_times) if all_times else 0,
        }
