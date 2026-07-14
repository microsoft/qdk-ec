"""Type stubs for the `deq_runtime` Rust extension module.

The `Runtime` class brings up the configured decoder, coordinator and
(optional) controller services in process. Services are exposed through
namespaced sub-objects (`runtime.coordinator`, `runtime.jit_controller`).
All RPC-style methods are coroutines and accept raw protobuf-serialized
bytes; the higher-level wrapper in `deq.runtime` adds typed conversions on
top.
"""

from typing import Optional


def static_jit_compile(jit_library: bytes) -> bytes: ...


def cli_run(*args: str, **kwargs: str) -> None: ...


class DecodingHypergraph:
    vertex_num: int
    hyperedges: list["Hyperedge"]


class Hyperedge:
    vertices: list[int]
    probability: float


class Coordinator:
    """Coordinator (`deq.bin`) interface for the in-process runtime.

    Obtain instances via :attr:`Runtime.coordinator`.
    """

    async def load_library(self, library: bytes) -> None:
        """Load a `deq.bin.Library` (protobuf-serialized) into the coordinator."""
        ...

    async def execute(self, instruction: bytes) -> int:
        """Submit a `deq.bin.Instruction` (protobuf-serialized).

        Returns the assigned id (gid / cid / eid depending on the
        instruction kind).
        """
        ...

    async def decode(self, outcomes: bytes) -> bytes:
        """Submit `deq.coordinator.Outcomes` (protobuf-serialized).

        Returns `deq.coordinator.Readouts` as protobuf-serialized bytes;
        parse with `coordinator_pb2.Readouts.FromString`.
        """
        ...

    async def reset(
        self,
        reset_library: bool = False,
        reset_decoder_service: bool = False,
        custom: str = "",
    ) -> None:
        """Reset the coordinator state."""
        ...


class JitController:
    """JIT controller (`deq.jit`) interface for dynamic-circuit decoding.

    Obtain instances via :attr:`Runtime.jit_controller` (which raises
    ``AttributeError`` if the runtime was not constructed with
    ``controller="jit"``).
    """

    async def load_library(self, library: bytes) -> None:
        """Load a `deq.jit.JitLibrary` (protobuf-serialized)."""
        ...

    async def execute(self, instruction: bytes) -> int:
        """Compile and execute a `deq.jit.JitInstruction`. Returns the gid."""
        ...

    async def batch_execute(self, instructions: list[bytes]) -> list[int]:
        """Compile and execute a batch of JIT instructions, respecting
        connector-based dependencies. All instructions must specify a
        non-zero gid. Returns gids in input order.
        """
        ...

    async def decode(self, outcomes: bytes) -> bytes:
        """Submit outcomes for a previously-executed gadget; returns Readouts bytes."""
        ...

    async def batch_decode(self, outcomes_list: list[bytes]) -> list[bytes]:
        """Decode multiple gadgets concurrently. Returns Readouts bytes in input order."""
        ...

    async def reset(
        self,
        reset_library: bool = False,
        reset_decoder_service: bool = False,
        custom: str = "",
    ) -> None:
        """Reset the JIT controller (cancels pending tasks, clears caches)."""
        ...


class Runtime:
    """In-process deq runtime.

    Parameters mirror the `deq server` CLI; pass `None` (or omit) to accept
    defaults.

    Args:
        decoder: Decoder algorithm name. Examples: `"black-box-naive"`,
            `"black-box-relay-bp"`, `"black-box-tesseract"`, `"mock"`.
        decoder_config: JSON-encoded decoder configuration.
        coordinator: Coordinator name (`"naive"`, `"monolithic"`, `"window"`).
        coordinator_config: JSON-encoded coordinator configuration.
        controller: Optional controller name (`"none"`, `"static"`, `"jit"`).
        controller_config: JSON-encoded controller configuration.
    """

    def __init__(
        self,
        *,
        decoder: Optional[str] = None,
        decoder_config: Optional[str] = None,
        coordinator: Optional[str] = None,
        coordinator_config: Optional[str] = None,
        controller: Optional[str] = None,
        controller_config: Optional[str] = None,
    ) -> None: ...

    @property
    def coordinator(self) -> Coordinator:
        """Coordinator (`deq.bin`) interface. Always available."""
        ...

    @property
    def jit_controller(self) -> JitController:
        """JIT controller (`deq.jit`) interface for dynamic circuits.

        Raises `AttributeError` if `controller="jit"` was not passed at
        construction.
        """
        ...

    def has_jit_controller(self) -> bool:
        """True iff a JIT controller is configured."""
        ...

    async def bind(self, addr: str = "[::]:0") -> str:
        """Optionally bind a gRPC server so external clients can connect.

        Returns the URL clients should use to connect.
        """
        ...

    def bound_port(self) -> Optional[int]:
        """Return the bound gRPC port, or None if not bound."""
        ...

    async def shutdown(self) -> None:
        """Shut down the optional gRPC server and wait for it to finish."""
        ...


class Sampler:
    """Low-level Rust binding for the measurement sampler.

    Accepts a Stim circuit source string; the concrete backend is chosen
    via the ``simulator`` argument and configured via ``simulator_config``,
    mirroring the decoder/coordinator/controller selection on
    :class:`Runtime`. Recognized backends:

    * ``"stim"`` (default) — Stim's compiled measurement sampler,
      auto-wrapped with resample-on-failure when the circuit has
      ``PREPARE { ... REQUIRE ... }`` blocks (QDK v1.30+).
    * ``"preselect"`` — Tableau-based sampler with retry-from-checkpoint
      semantics; runs each PREPARE block natively.

    The high-level :class:`deq.runtime.Sampler` wraps this class — it
    transpiles ``.deq`` programs to Stim before delegating sampling here.
    Use the high-level wrapper unless you have a Stim circuit string in
    hand and want to bypass the deq toolchain.
    """

    def __init__(
        self,
        circuit: str,
        *,
        simulator: Optional[str] = None,
        simulator_config: Optional[str] = None,
        seed: Optional[int] = None,
        skip_shots: int = 0,
    ) -> None: ...

    def sample(self, num_shots: int) -> list[bytes]:
        """Sample ``num_shots`` shots. Returns a list of protobuf-serialized
        :class:`deq.proto.simulator_pb2.ShotSample` byte strings, each
        carrying only the flat ``outcomes`` field. Callers that want
        per-gadget chunks should slice it themselves (see
        :meth:`deq.runtime.Sampler.split_outcomes`).
        """
        ...
