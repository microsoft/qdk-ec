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
from typing import Any, List, Mapping, Optional, Union, overload

import deq_runtime
from deq.proto import coordinator_pb2 as _coord_pb
from deq.proto import deq_bin_pb2 as _bin_pb
from deq.proto import deq_jit_pb2 as _jit_pb

# Re-export raw Rust pyclasses for callers that want to bypass the wrapper.
RawRuntime = deq_runtime.Runtime
RawCoordinator = deq_runtime.Coordinator
RawJitController = deq_runtime.JitController

__all__ = [
    "Coordinator",
    "JitController",
    "RawCoordinator",
    "RawJitController",
    "RawRuntime",
    "Runtime",
]


_ConfigLike = Optional[Union[str, Mapping[str, Any]]]


def _normalize_config(value: _ConfigLike) -> Optional[str]:
    """Accept a JSON string or a Python mapping; emit a JSON string."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


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
