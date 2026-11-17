"""Resolve PROGRAM-local clocks and preserve references through completion ordering."""

from dataclasses import dataclass, field
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

from deqagram import Decorator, GadgetApplication, Instruction, KeywordArg, RepeatBlock
from deq.proto.deq_bin_pb2 import Gadget
from deq.proto.deq_jit_pb2 import JitInstruction

MAX_TIME_NS = (1 << 63) - 1
TIMING_METADATA_KEY = "deq.simulation.timing"


def has_timing(body: Sequence[object]) -> bool:
    return any(
        any(
            decorator.name in ("START", "DELAY")
            for decorator in getattr(item, "decorators", ())
        )
        or (isinstance(item, RepeatBlock) and has_timing(item.body))
        for item in body
    )


def nanoseconds(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("timing values must be numeric seconds")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("invalid timing value") from error
    if not decimal.is_finite() or decimal < 0:
        raise ValueError("timing must be finite and nonnegative")
    _, digits, exponent = decimal.as_tuple()
    if not any(digits):
        return 0
    assert isinstance(exponent, int)
    shift = exponent + 9
    if shift < 0:
        if -shift >= len(digits) or any(digits[shift:]):
            raise ValueError("timing must be representable in integer nanoseconds")
        digits = digits[:shift]
        shift = 0
    if len(digits) + shift > 19:
        raise ValueError("PROGRAM timing exceeds the supported time range")
    result = int("".join(map(str, digits))) * 10**shift
    if result > MAX_TIME_NS:
        raise ValueError(
            "timing must be finite, nonnegative, and representable in integer nanoseconds"
        )
    return result


@dataclass(frozen=True)
class Timing:
    start: int
    duration: int = 0
    declared_duration: bool = False
    deadline: int | None = None

    @property
    def end(self) -> int:
        result = self.start + self.duration
        if result > MAX_TIME_NS:
            raise ValueError("PROGRAM timing exceeds the supported time range")
        if self.deadline is not None and result > self.deadline:
            raise ValueError(
                "PROGRAM operation exceeds its enclosing invocation duration"
            )
        return result


def gadget_timing(gadget: Gadget) -> Timing | None:
    """Read the simulator's metadata convention without interpreting other keys."""
    value = gadget.metadata.fields.get(TIMING_METADATA_KEY)
    if value is None:
        return None
    if value.WhichOneof("kind") != "struct_value":
        raise ValueError(f"{TIMING_METADATA_KEY} must be an object")
    fields = value.struct_value.fields
    if set(fields) - {"start_ns", "duration_ns"}:
        raise ValueError(f"unknown {TIMING_METADATA_KEY} field")
    times = []
    for name in ("start_ns", "duration_ns"):
        field = fields.get(name)
        if field is None and name == "duration_ns":
            times.append(0)
            continue
        if field is None or field.WhichOneof("kind") != "string_value":
            raise ValueError(f"{TIMING_METADATA_KEY}.{name} must be a decimal string")
        text = field.string_value
        if not text.isascii() or not text.isdecimal() or len(text) > 19 or int(text) > MAX_TIME_NS:
            raise ValueError(f"{TIMING_METADATA_KEY}.{name} must fit nonnegative signed 63-bit nanoseconds")
        times.append(int(text))
    timing = Timing(*times)
    timing.end
    return timing


def set_gadget_timing(gadget: Gadget, timing: Timing) -> None:
    """Store exact nanoseconds while leaving unrelated user metadata untouched."""
    timing.end
    gadget.metadata[TIMING_METADATA_KEY] = {
        "start_ns": str(timing.start),
        "duration_ns": str(timing.duration),
    }


@dataclass
class Schedule:
    used: bool = False
    statements: dict[int, Timing] = field(default_factory=dict)


@dataclass
class Clock:
    schedule: Schedule
    origin: int = 0
    cursor: int = 0
    deadline: int | None = None

    def resolve(self, statement: object) -> Timing:
        annotations = [
            decorator
            for decorator in getattr(statement, "decorators", ())
            if decorator.name in ("START", "DELAY")
        ]
        if len(annotations) > 1:
            raise ValueError("use at most one START or DELAY annotation per statement")
        duration = 0
        declared_duration = False
        if annotations:
            if not isinstance(statement, (GadgetApplication, Instruction, RepeatBlock)):
                raise ValueError(
                    "START and DELAY apply only to PROGRAM calls and REPEAT blocks"
                )
            self.schedule.used = True
            annotation: Decorator = annotations[0]
            if annotation.name == "DELAY":
                if len(annotation.arguments) != 1 or isinstance(
                    annotation.arguments[0], KeywordArg
                ):
                    raise ValueError(
                        "DELAY requires one positional duration in seconds"
                    )
                self.cursor += nanoseconds(annotation.arguments[0])
            else:
                keywords = {}
                for argument in annotation.arguments:
                    if not isinstance(argument, KeywordArg) or argument.key not in (
                        "at",
                        "duration",
                    ):
                        raise ValueError(
                            "START accepts only at= and duration= in seconds"
                        )
                    if argument.key in keywords:
                        raise ValueError(f"duplicate START argument {argument.key}")
                    keywords[argument.key] = nanoseconds(argument.value)
                if "at" in keywords:
                    self.cursor = keywords["at"]
                duration = keywords.get("duration", 0)
                declared_duration = "duration" in keywords
        timing = Timing(
            self.origin + self.cursor, duration, declared_duration, self.deadline
        )
        timing.end
        return timing


def order_program(
    instructions: list[tuple[JitInstruction, GadgetApplication, int]],
    assertions: list[tuple[list[int], bool, str]],
    schedule: Schedule,
) -> tuple[
    list[tuple[JitInstruction, GadgetApplication]], list[tuple[list[int], bool, str]]
]:
    """Bind in source order first; only then remap IDs and logical readout indices."""
    old_readouts: list[tuple[int, int]] = []
    by_gid: dict[int, Timing] = {}
    for instruction, application, count in instructions:
        gid = instruction.gadget.gid
        timing = schedule.statements[id(application)]
        by_gid[gid] = timing
        old_readouts.extend((gid, index) for index in range(count))
    for instruction, application, _ in instructions:
        gadget = instruction.gadget
        timing = by_gid[gadget.gid]
        predecessors = [connector.gid for connector in gadget.connectors]
        predecessors.extend(
            readout.gid
            for readout in gadget.modifier.remote_conditional_correction.remote_readouts
        )
        if application.gadget_name.startswith("__identity_"):
            timing = Timing(
                max([timing.start, *(by_gid[gid].end for gid in predecessors)]),
                deadline=timing.deadline,
            )
            by_gid[gadget.gid] = timing
        timing.end
        for gid in predecessors:
            if by_gid[gid].end > timing.start:
                raise ValueError(
                    f"invalid schedule: gadget {gadget.gid} ({application.gadget_name}) starts at "
                    f"{timing.start} ns before predecessor {gid} finishes at {by_gid[gid].end} ns"
                )
        set_gadget_timing(gadget, timing)
    ordered = sorted(instructions, key=lambda item: by_gid[item[0].gadget.gid].end)
    mapping = {
        instruction.gadget.gid: index
        for index, (instruction, _, _) in enumerate(ordered, 1)
    }
    new_readouts = {
        source: index
        for index, source in enumerate(
            (instruction.gadget.gid, local)
            for instruction, _, count in ordered
            for local in range(count)
        )
    }
    for instruction, _, _ in ordered:
        gadget = instruction.gadget
        old_gid = gadget.gid
        for connector in gadget.connectors:
            if mapping[connector.gid] >= mapping[old_gid]:
                raise ValueError(
                    "completion-time/source-order tie break violates a gadget dependency"
                )
            connector.gid = mapping[connector.gid]
        for readout in gadget.modifier.remote_conditional_correction.remote_readouts:
            if mapping[readout.gid] >= mapping[old_gid]:
                raise ValueError(
                    "completion order violates a conditional-readout dependency"
                )
            readout.gid = mapping[readout.gid]
        gadget.gid = mapping[old_gid]
    assertions = [
        ([new_readouts[old_readouts[index]] for index in indices], expected, source)
        for indices, expected, source in assertions
    ]
    return [
        (instruction, application) for instruction, application, _ in ordered
    ], assertions
