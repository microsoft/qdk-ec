# pylint: disable=no-member
#   no-member: compiled Rust extension module members not detected by pylint
"""High-level Python wrapper around the in-process `deq_runtime` runtime.

Instantiating :class:`Runtime` brings up a complete deq runtime (decoder,
coordinator, optional controller) inside the current Python process. Services
are exposed through namespaced sub-objects:

* :attr:`Runtime.coordinator` (:class:`Coordinator`) — the lower-level
  ``deq.bin`` interface (``load_library`` / ``execute`` / ``decode`` /
  ``reset``). Always present.
* :attr:`Runtime.jit_controller` (:class:`JitController`) — the dynamic-circuit
  interface (``load_library`` / ``execute`` / ``batch_execute`` / ``decode`` /
  ``batch_decode`` / ``reset``). Present only when constructed with
  ``controller="jit"``; this is the surface most online-decoding callers want
  because it compiles ``deq.jit`` types into ``deq.bin`` on the fly.

All RPC-style methods are coroutines. Each one accepts both raw
protobuf-serialized ``bytes`` and the corresponding ``*_pb2`` message object,
returning a typed proto by default (use ``raw=True`` on the ``decode`` /
``batch_decode`` methods to opt out of parsing).

The Rust extension classes are re-exported as :class:`RawRuntime`,
:class:`RawCoordinator` and :class:`RawJitController` for callers that want to
bypass the Python-side proto conversion.

Example — JIT controller for dynamic circuits::

    import asyncio
    from deq.runtime import Runtime
    from deq.proto import deq_jit_pb2 as jit_pb

    async def main() -> None:
        async with Runtime(
            decoder="black-box-relay-bp",
            coordinator="monolithic",
            controller="jit",
        ) as runtime:
            jit = runtime.jit_controller
            await jit.load_library(jit_pb.JitLibrary(...))
            gid = await jit.execute(jit_pb.JitInstruction(...))
            readouts = await jit.decode(outcomes)

    asyncio.run(main())
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Mapping, Optional, Union, overload

import deq_runtime
from deq.proto import coordinator_pb2 as _coord_pb
from deq.proto import deq_bin_pb2 as _bin_pb
from deq.proto import deq_jit_pb2 as _jit_pb
from deq.proto import simulator_pb2 as _sim_pb
from deq.proto import util_pb2 as _util_pb

# Re-export raw Rust pyclasses for callers that want to bypass the wrapper.
RawRuntime = deq_runtime.Runtime
RawCoordinator = deq_runtime.Coordinator
RawJitController = deq_runtime.JitController
RawSampler = deq_runtime.Sampler

__all__ = [
    "Coordinator",
    "JitController",
    "RawCoordinator",
    "RawJitController",
    "RawRuntime",
    "RawSampler",
    "Runtime",
    "Sampler",
]


_ConfigLike = Optional[Union[str, Mapping[str, Any]]]


def _normalize_config(value: _ConfigLike) -> Optional[str]:
    """Accept a JSON string or a Python mapping; emit a JSON string."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


def _split_bit_vector(
    bit_vector: _util_pb.BitVector,
    partition: List[int],
) -> List[_util_pb.BitVector]:
    """Slice a ``BitVector`` into chunks sized by ``partition``.

    Matches the runtime's MSB-first packing — bit ``i`` of the input lives
    in byte ``i // 8`` at bit position ``7 - (i % 8)`` — so each returned
    chunk is byte-identical to the per-gadget ``BitVector``s the
    coordinator's ``decode`` expects.
    """
    total = sum(partition)
    if total != bit_vector.size:
        raise ValueError(
            f"BitVector size {bit_vector.size} does not match partition "
            f"sum {total}; check that the BitVector came from this sampler"
        )
    data = bit_vector.data
    chunks: List[_util_pb.BitVector] = []
    offset = 0
    for count in partition:
        chunk_bytes = bytearray((count + 7) // 8)
        for i in range(count):
            source_index = offset + i
            if (data[source_index >> 3] >> (7 - (source_index & 7))) & 1:
                chunk_bytes[i >> 3] |= 1 << (7 - (i & 7))
        chunks.append(_util_pb.BitVector(size=count, data=bytes(chunk_bytes)))
        offset += count
    return chunks


# ── Coordinator wrapper ─────────────────────────────────────────────────────


class Coordinator:
    """Typed wrapper around :class:`deq_runtime.Coordinator`.

    Accepts both raw ``bytes`` and ``*_pb2`` proto objects on input; returns
    parsed proto messages by default.
    """

    def __init__(self, raw: deq_runtime.Coordinator) -> None:
        self._raw = raw

    @property
    def raw(self) -> deq_runtime.Coordinator:
        """The underlying Rust pyclass (raw-bytes interface)."""
        return self._raw

    async def load_library(self, library: Union[bytes, _bin_pb.Library]) -> None:
        """Load a :class:`deq.proto.deq_bin_pb2.Library` into the coordinator.

        Accepts either a parsed proto message or raw bytes.
        """
        payload = (
            library.SerializeToString() if isinstance(library, _bin_pb.Library) else library
        )
        await self._raw.load_library(payload)

    async def execute(self, instruction: Union[bytes, _bin_pb.Instruction]) -> int:
        """Submit an :class:`Instruction` and return the assigned id."""
        payload = (
            instruction.SerializeToString()
            if isinstance(instruction, _bin_pb.Instruction)
            else instruction
        )
        return await self._raw.execute(payload)

    @overload
    async def decode(
        self,
        outcomes: Union[bytes, _coord_pb.Outcomes],
        *,
        raw: bool = False,
    ) -> _coord_pb.Readouts: ...

    @overload
    async def decode(
        self,
        outcomes: Union[bytes, _coord_pb.Outcomes],
        *,
        raw: bool = True,
    ) -> bytes: ...

    async def decode(
        self,
        outcomes: Union[bytes, _coord_pb.Outcomes],
        *,
        raw: bool = False,
    ) -> Union[_coord_pb.Readouts, bytes]:
        """Submit :class:`Outcomes` and await the corresponding readouts.

        Returns a parsed :class:`Readouts` by default; pass ``raw=True`` to
        get protobuf-serialized bytes instead.
        """
        payload = (
            outcomes.SerializeToString()
            if isinstance(outcomes, _coord_pb.Outcomes)
            else outcomes
        )
        result = await self._raw.decode(payload)
        if raw:
            return result
        readouts = _coord_pb.Readouts()
        readouts.ParseFromString(result)
        return readouts

    async def reset(
        self,
        *,
        reset_library: bool = False,
        reset_decoder_service: bool = False,
        custom: str = "",
    ) -> None:
        """Reset the coordinator state."""
        await self._raw.reset(reset_library, reset_decoder_service, custom)

    def __repr__(self) -> str:
        return "Coordinator()"


# ── JIT controller wrapper ──────────────────────────────────────────────────


class JitController:
    """Typed wrapper around :class:`deq_runtime.JitController`.

    Accepts both raw ``bytes`` and ``*_pb2`` proto objects on input; returns
    parsed proto messages by default. Use the ``raw=True`` flag on
    :meth:`decode` / :meth:`batch_decode` to opt out of parsing.
    """

    def __init__(self, raw: deq_runtime.JitController) -> None:
        self._raw = raw

    @property
    def raw(self) -> deq_runtime.JitController:
        """The underlying Rust pyclass (raw-bytes interface)."""
        return self._raw

    async def load_library(self, library: Union[bytes, _jit_pb.JitLibrary]) -> None:
        """Load a :class:`deq.proto.deq_jit_pb2.JitLibrary`."""
        payload = (
            library.SerializeToString() if isinstance(library, _jit_pb.JitLibrary) else library
        )
        await self._raw.load_library(payload)

    async def execute(self, instruction: Union[bytes, _jit_pb.JitInstruction]) -> int:
        """Compile and execute one JIT instruction. Returns the assigned gid."""
        payload = (
            instruction.SerializeToString()
            if isinstance(instruction, _jit_pb.JitInstruction)
            else instruction
        )
        return await self._raw.execute(payload)

    async def batch_execute(
        self,
        instructions: List[Union[bytes, _jit_pb.JitInstruction]],
    ) -> List[int]:
        """Compile and execute a batch, respecting connector dependencies.

        All instructions must specify a non-zero ``gid``. Returns the gids
        in input order.
        """
        payloads = [
            i.SerializeToString() if isinstance(i, _jit_pb.JitInstruction) else i
            for i in instructions
        ]
        return await self._raw.batch_execute(payloads)

    @overload
    async def decode(
        self,
        outcomes: Union[bytes, _coord_pb.Outcomes],
        *,
        raw: bool = False,
    ) -> _coord_pb.Readouts: ...

    @overload
    async def decode(
        self,
        outcomes: Union[bytes, _coord_pb.Outcomes],
        *,
        raw: bool = True,
    ) -> bytes: ...

    async def decode(
        self,
        outcomes: Union[bytes, _coord_pb.Outcomes],
        *,
        raw: bool = False,
    ) -> Union[_coord_pb.Readouts, bytes]:
        """Submit outcomes for a previously-executed gadget.

        Returns parsed :class:`Readouts` by default; ``raw=True`` returns bytes.
        """
        payload = (
            outcomes.SerializeToString()
            if isinstance(outcomes, _coord_pb.Outcomes)
            else outcomes
        )
        result = await self._raw.decode(payload)
        if raw:
            return result
        readouts = _coord_pb.Readouts()
        readouts.ParseFromString(result)
        return readouts

    @overload
    async def batch_decode(
        self,
        outcomes_list: List[Union[bytes, _coord_pb.Outcomes]],
        *,
        raw: bool = False,
    ) -> List[_coord_pb.Readouts]: ...

    @overload
    async def batch_decode(
        self,
        outcomes_list: List[Union[bytes, _coord_pb.Outcomes]],
        *,
        raw: bool = True,
    ) -> List[bytes]: ...

    async def batch_decode(
        self,
        outcomes_list: List[Union[bytes, _coord_pb.Outcomes]],
        *,
        raw: bool = False,
    ) -> Union[List[_coord_pb.Readouts], List[bytes]]:
        """Decode multiple gadgets concurrently. Results are in input order."""
        payloads = [
            o.SerializeToString() if isinstance(o, _coord_pb.Outcomes) else o
            for o in outcomes_list
        ]
        results = await self._raw.batch_decode(payloads)
        if raw:
            return list(results)
        parsed = []
        for blob in results:
            readouts = _coord_pb.Readouts()
            readouts.ParseFromString(blob)
            parsed.append(readouts)
        return parsed

    async def reset(
        self,
        *,
        reset_library: bool = False,
        reset_decoder_service: bool = False,
        custom: str = "",
    ) -> None:
        """Reset the JIT controller (cancels pending tasks, clears caches)."""
        await self._raw.reset(reset_library, reset_decoder_service, custom)

    def __repr__(self) -> str:
        return "JitController()"


# ── Runtime ─────────────────────────────────────────────────────────────────


class Runtime:
    """In-process deq runtime with a typed, async Python API.

    Args:
        decoder: Decoder algorithm name (e.g. ``"black-box-naive"``,
            ``"black-box-relay-bp"``, ``"black-box-tesseract"``, ``"mock"``).
            Defaults to the Rust CLI default.
        decoder_config: Decoder-specific configuration. May be a JSON string,
            a Python mapping (serialized via :mod:`json`), or ``None``.
        coordinator: Coordinator name (``"naive"``, ``"monolithic"``,
            ``"window"``).
        coordinator_config: Coordinator configuration (see ``decoder_config``).
        controller: Optional controller name (``"none"``, ``"static"``,
            ``"jit"``). Pass ``"jit"`` to enable :attr:`jit_controller` (the
            dynamic-circuit interface).
        controller_config: Controller configuration (see ``decoder_config``).

    Use :attr:`coordinator` for low-level ``deq.bin`` access (always
    available) and :attr:`jit_controller` for the dynamic-circuit interface
    (only when ``controller="jit"``).

    Most methods are coroutines and must be awaited on an asyncio event loop.
    The runtime can be used as an async context manager so :meth:`shutdown`
    is called automatically on exit.
    """

    def __init__(
        self,
        *,
        decoder: Optional[str] = None,
        decoder_config: _ConfigLike = None,
        coordinator: Optional[str] = None,
        coordinator_config: _ConfigLike = None,
        controller: Optional[str] = None,
        controller_config: _ConfigLike = None,
    ) -> None:
        self._raw: deq_runtime.Runtime = deq_runtime.Runtime(
            decoder=decoder,
            decoder_config=_normalize_config(decoder_config),
            coordinator=coordinator,
            coordinator_config=_normalize_config(coordinator_config),
            controller=controller,
            controller_config=_normalize_config(controller_config),
        )
        # Cache wrappers so getters return the same instance every time.
        self._coordinator: Coordinator = Coordinator(self._raw.coordinator)
        self._jit_controller: Optional[JitController] = (
            JitController(self._raw.jit_controller) if self._raw.has_jit_controller() else None
        )

    # ── Lifecycle ─────────────────────────────────────────────────────

    @property
    def raw(self) -> deq_runtime.Runtime:
        """Access the underlying Rust extension runtime."""
        return self._raw

    async def bind(self, addr: str = "[::]:0") -> str:
        """Bind a gRPC server on ``addr`` so external clients can connect.

        Returns the URL clients should use (with the unspecified-address
        rewrite applied).
        """
        return await self._raw.bind(addr)

    def bound_port(self) -> Optional[int]:
        """Return the bound gRPC port, or ``None`` if not bound."""
        return self._raw.bound_port()

    async def shutdown(self) -> None:
        """Shut down the optional gRPC server and wait for it to finish."""
        await self._raw.shutdown()

    async def __aenter__(self) -> "Runtime":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.shutdown()

    def __repr__(self) -> str:
        port = self.bound_port()
        jit = ", controller=jit" if self._jit_controller is not None else ""
        if port is None:
            return f"Runtime(unbound{jit})"
        return f"Runtime(bound_port={port}{jit})"

    # ── Services ──────────────────────────────────────────────────────

    @property
    def coordinator(self) -> Coordinator:
        """Coordinator (``deq.bin``) interface. Always available."""
        return self._coordinator

    @property
    def jit_controller(self) -> JitController:
        """JIT controller (``deq.jit``) interface for dynamic circuits.

        Raises :class:`AttributeError` if the runtime was not constructed
        with ``controller="jit"``.
        """
        if self._jit_controller is None:
            raise AttributeError(
                'jit_controller is not configured; '
                'pass controller="jit" to Runtime(...) to enable it'
            )
        return self._jit_controller

    def has_jit_controller(self) -> bool:
        """True iff a JIT controller is configured (no exception thrown)."""
        return self._jit_controller is not None


# ── Sampler ─────────────────────────────────────────────────────────────────


class Sampler:
    """deq-native measurement sampler.

    Takes a ``.deq`` file plus the name of one of its ``PROGRAM`` blocks,
    transpiles the program to a Stim circuit internally, and produces
    per-gadget-partitioned shots ready to feed straight into a runtime's
    coordinator or JIT controller. The noise model is whatever is baked
    into the gadget bodies (``X_ERROR``, ``DEPOLARIZE1``, ``M(p)``, ..) —
    no separate noise configuration is needed.

    Args:
        deq_path: Filesystem path to a ``.deq`` file. ``IMPORT`` statements
            are resolved relative to the file.
        program: Name of the ``PROGRAM`` block to sample. Raises
            :class:`KeyError` if no program by that name exists in the file.
        simulator: Backend name. ``"stim"`` (default) uses Stim's compiled
            measurement sampler, auto-wrapping with resample-on-failure
            when the circuit has ``PREPARE { ... REQUIRE ... }`` blocks
            (QDK v1.30+ preselection syntax).  ``"preselect"`` uses a
            tableau-based sampler that natively runs each PREPARE block
            with retry-from-checkpoint semantics.
        simulator_config: Optional JSON string or mapping with backend
            options (currently ``preselect_max_attempts``).
        seed: Optional deterministic seed. When omitted, a random seed is
            drawn from the OS.
        skip_shots: Number of initial shots to consume and discard. Useful
            for resuming a deterministic run from a known offset.

    For in-memory ``.deq`` source text (no file on disk), use
    :meth:`from_source`.

    Example::

        from deq.runtime import Runtime, Sampler
        from deq.proto import coordinator_pb2 as coord_pb

        sampler = Sampler("program.deq", program="Simulation", seed=42)
        instructions = list(sampler.instructions)

        async with Runtime(
            decoder="black-box-relay-bp",
            coordinator="monolithic",
            controller="jit",
        ) as runtime:
            jit = runtime.jit_controller
            await jit.load_library(sampler.library)
            for shot in sampler.sample(100):
                # Each shot needs fresh gadget instances: once a gid is
                # decoded the coordinator refuses to decode it again.
                await jit.reset()
                await jit.batch_execute(instructions)
                chunks = sampler.split_outcomes(shot.outcomes)
                readouts = await jit.batch_decode([
                    coord_pb.Outcomes(gid=instr.gadget.gid, outcomes=chunk)
                    for instr, chunk in zip(instructions, chunks)
                ])

    Sampling is synchronous and CPU-bound; the call releases the GIL while
    the background thread produces the next batch.

    Attributes
    ----------
    program : str
        The program name that was selected.
    library : deq.proto.deq_jit_pb2.JitLibrary
        The compiled JIT library. Its ``program`` field is populated with
        the instructions for the selected program — pass it straight to
        ``runtime.jit_controller.load_library`` (or to
        :func:`deq.compiler.jit_compiler.static_jit_compiler` for the
        coordinator path).
    instructions : list[deq.proto.deq_jit_pb2.JitInstruction]
        The compiled JIT instructions for the selected program, in
        execution order. Use these to drive
        ``runtime.jit_controller.execute`` / ``batch_execute`` and to
        pair gids with the per-gadget measurement chunks.
    partition : list[int]
        Per-gadget measurement counts in :attr:`instructions` order — the
        shape the runtime uses to chunk each shot's flat outcome record
        into one ``Outcomes`` per gadget. See :meth:`split_outcomes`.
    circuit : str
        The Stim circuit string that the sampler is driving. Exposed for
        debugging / inspection.
    """

    def __init__(
        self,
        deq_path: Union[str, "os.PathLike[str]"],
        program: str,
        *,
        simulator: Optional[str] = None,
        simulator_config: _ConfigLike = None,
        seed: Optional[int] = None,
        skip_shots: int = 0,
    ) -> None:
        from deq.circuit.parser import parse_file
        self._init_from_deq_file(
            parse_file(deq_path), program, simulator, simulator_config, seed, skip_shots
        )

    @classmethod
    def from_source(
        cls,
        deq_source: str,
        program: str,
        *,
        simulator: Optional[str] = None,
        simulator_config: _ConfigLike = None,
        seed: Optional[int] = None,
        skip_shots: int = 0,
    ) -> "Sampler":
        """Construct a :class:`Sampler` from in-memory ``.deq`` source text.

        Useful when the program is generated on the fly (e.g. via Mako
        templates) or when you don't want to hit the filesystem in tests.
        ``IMPORT`` statements in the source are **not** resolved — pass a
        file path to :meth:`__init__` if you need import resolution.
        """
        from deq.circuit.parser import parse
        self = cls.__new__(cls)
        self._init_from_deq_file(
            parse(deq_source), program, simulator, simulator_config, seed, skip_shots
        )
        return self

    def _init_from_deq_file(
        self,
        deq_file: Any,
        program_name: str,
        simulator: Optional[str],
        simulator_config: _ConfigLike,
        seed: Optional[int],
        skip_shots: int,
    ) -> None:
        from deq.transpiler.program_artifacts import transpile_program

        artifacts = transpile_program(deq_file, program_name, decoder_data=False)

        self._program: str = program_name
        self._deq_file: Any = deq_file
        self._instructions: List[_jit_pb.JitInstruction] = artifacts.instructions
        self._circuit: str = artifacts.circuit
        self._partition: List[int] = artifacts.partition
        self._library: Optional[_jit_pb.JitLibrary] = None

        kwargs: dict[str, Any] = {"skip_shots": skip_shots}
        if seed is not None:
            kwargs["seed"] = seed
        if simulator is not None:
            kwargs["simulator"] = simulator
        normalized_config = _normalize_config(simulator_config)
        if normalized_config is not None:
            kwargs["simulator_config"] = normalized_config
        self._raw: deq_runtime.Sampler = deq_runtime.Sampler(
            artifacts.circuit, **kwargs
        )

    @property
    def raw(self) -> deq_runtime.Sampler:
        """The underlying Rust pyclass (raw-bytes interface)."""
        return self._raw

    @property
    def program(self) -> str:
        """The name of the program this sampler is sampling."""
        return self._program

    @property
    def library(self) -> _jit_pb.JitLibrary:
        """The full decoder-capable JIT library, with the program's
        instructions appended to its ``program`` field.

        Built lazily on first access: the Sampler itself only needs
        the lightweight program-only library (see
        :func:`deq.transpiler.jit_library_builder.build_jit_program`),
        so most callers never pay for the per-gadget Clifford
        analysis, noise propagation, and check resolution that the
        full :func:`~deq.transpiler.jit_library_builder.build_jit_library`
        performs. The first ``sampler.library`` access does pay for it;
        subsequent accesses return the cached result.
        """
        if self._library is None:
            from deq.transpiler.jit_library_builder import build_jit_library

            full = build_jit_library(self._deq_file)
            for instruction in self._instructions:
                full.program.append(instruction)
            self._library = full
        return self._library

    @property
    def instructions(self) -> List[_jit_pb.JitInstruction]:
        """The compiled JIT instructions for the selected program, in
        execution order."""
        return self._instructions

    @property
    def partition(self) -> List[int]:
        """Per-gadget measurement counts in :attr:`instructions` order.

        Each entry is the number of physical measurement bits the
        corresponding gadget contributes to the flat ``shot.outcomes``
        record. Pass it to :meth:`split_outcomes` (or slice manually)
        to recover the per-gadget chunks the coordinator's ``decode``
        expects.
        """
        return list(self._partition)

    @property
    def circuit(self) -> str:
        """The transpiled Stim circuit string."""
        return self._circuit

    def split_outcomes(
        self, outcomes: _util_pb.BitVector
    ) -> List[_util_pb.BitVector]:
        """Slice a flat ``outcomes`` BitVector into per-gadget chunks.

        Returns one :class:`~deq.proto.util_pb2.BitVector` per gadget in
        :attr:`instructions` order, each byte-identical to what
        ``coordinator.decode(Outcomes(outcomes=chunk))`` expects.

        Raises :class:`ValueError` if ``outcomes.size`` does not match
        the sum of :attr:`partition` — typically a sign that the
        ``BitVector`` came from a different circuit than this sampler.
        """
        return _split_bit_vector(outcomes, self._partition)

    @overload
    def sample(self, num_shots: int, *, raw: bool = False) -> List[_sim_pb.ShotSample]: ...

    @overload
    def sample(self, num_shots: int, *, raw: bool = True) -> List[bytes]: ...

    def sample(
        self,
        num_shots: int,
        *,
        raw: bool = False,
    ) -> Union[List[_sim_pb.ShotSample], List[bytes]]:
        """Sample ``num_shots`` shots from the circuit.

        Returns a list of :class:`deq.proto.simulator_pb2.ShotSample` proto
        messages by default; pass ``raw=True`` to get the protobuf-serialized
        bytes instead.

        Each ``ShotSample`` carries the flat physical-measurement record
        in its ``outcomes`` field. Use :meth:`split_outcomes` (with
        :attr:`partition`) to slice it into per-gadget chunks suitable
        for ``coordinator.decode`` / ``jit_controller.decode``.
        """
        blobs = self._raw.sample(num_shots)
        if raw:
            return list(blobs)
        parsed = []
        for blob in blobs:
            shot = _sim_pb.ShotSample()
            shot.ParseFromString(blob)
            parsed.append(shot)
        return parsed

    def __repr__(self) -> str:
        return f"Sampler(program={self._program!r}, gadgets={len(self._instructions)})"
