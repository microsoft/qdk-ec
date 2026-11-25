"""Cyclic, validated raw-byte sampler for native PROGRAM-timed replay."""

import hashlib
import json
import os
from pathlib import Path
import sys
import time


class Sampler:
    def __init__(self, circuit_text: str, config: dict):
        if circuit_text:
            raise ValueError("Binary replay uses PROGRAM metadata, not a Stim source")
        measurements = config["num_measurements"]
        if (
            isinstance(measurements, bool)
            or not isinstance(measurements, int)
            or measurements < 0
        ):
            raise ValueError("num_measurements must be a nonnegative integer")
        self.size = (measurements + 7) // 8
        self.remainder = measurements % 8
        if self.size > 10_000_000_000:
            raise ValueError("Raw replay shot exceeds the 10 GB file guard")
        files = config["files"]
        if (
            not isinstance(files, list)
            or not files
            or any(not isinstance(path, str) for path in files)
        ):
            raise ValueError("files must be a nonempty list of paths")
        self.files = [Path(path) for path in files]
        self.hashes = config.get("sha256")
        if self.hashes is not None and (
            not isinstance(self.hashes, list)
            or len(self.hashes) != len(self.files)
            or any(
                not isinstance(value, str) or len(value) != 64 for value in self.hashes
            )
        ):
            raise ValueError("sha256 must contain one digest per input file")
        for path in self.files:
            if not path.is_file() or path.stat().st_size != self.size:
                raise ValueError(f"Raw replay file has the wrong size: {path}")
        offset = config.get("offset", 0)
        skipped = config.get("skip_shots", 0)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (offset, skipped)
        ):
            raise ValueError("offset and skip_shots must be nonnegative integers")
        self.index = (offset + skipped) % len(self.files)
        self._sample_calls = 0
        self._trim_heap = None
        self._memory_log = None
        requested_trim = config.get("trim_heap", False)
        if not isinstance(requested_trim, bool):
            raise ValueError("trim_heap must be a boolean")
        if requested_trim:
            import ctypes

            path = config.get("memory_log")
            if not isinstance(path, str) or not path:
                raise ValueError("trim_heap requires a memory_log path")
            if sys.platform != "linux":
                raise ValueError("Heap trimming requires glibc on Linux")
            library = ctypes.CDLL(None)
            try:
                self._trim_heap = library.malloc_trim
            except AttributeError as error:
                raise ValueError("Heap trimming requires glibc malloc_trim") from error
            self._trim_heap.argtypes = [ctypes.c_size_t]
            self._trim_heap.restype = ctypes.c_int
            self._memory_log = Path(path)

    def _prepare_memory(self) -> None:
        """Reclaim free allocator pages during sampling, before the latency clock."""
        if self._trim_heap is None:
            return

        def resident_bytes() -> int:
            return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
                "SC_PAGE_SIZE"
            )

        before = resident_bytes()
        started = time.monotonic()
        released = self._trim_heap(0)
        record = {
            "sample_call": self._sample_calls,
            "file_index": self.index,
            "rss_before_bytes": before,
            "rss_after_bytes": resident_bytes(),
            "trim_seconds": time.monotonic() - started,
            "released_free_pages": bool(released),
        }
        encoded = (json.dumps(record) + "\n").encode()
        assert self._memory_log is not None
        with self._memory_log.open("ab") as stream:
            if stream.tell() + len(encoded) > 10_000_000_000:
                raise ValueError("Allocator log exceeds the 10 GB file guard")
            stream.write(encoded)

    def sample(self) -> bytes:
        self._prepare_memory()
        path = self.files[self.index]
        with path.open("rb") as stream:
            data = stream.read(self.size + 1)
        if len(data) != self.size:
            raise ValueError(f"Raw replay file changed size: {path}")
        if self.remainder and data[-1] >> self.remainder:
            raise ValueError(f"Raw replay file has nonzero padding: {path}")
        if (
            self.hashes is not None
            and hashlib.sha256(data).hexdigest() != self.hashes[self.index]
        ):
            raise ValueError(f"Raw replay file failed its SHA-256 check: {path}")
        self.index = (self.index + 1) % len(self.files)
        self._sample_calls += 1
        return data
