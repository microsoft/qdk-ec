from dataclasses import dataclass, field


@dataclass
class Violations:
    violations: list[str] = field(default_factory=list)

    def __init__(self, violations: str | list[str] | None = None) -> None:
        if violations is None:
            self.violations = []
        elif isinstance(violations, str):
            self.violations = [violations]
        elif isinstance(violations, list):
            self.violations = violations

    def __bool__(self) -> bool:
        return len(self.violations) == 0

    def __add__(self, message: "str | Violations") -> "Violations":
        if isinstance(message, str):
            return Violations(self.violations + [message])
        return Violations(self.violations + message.violations)

    def __contains__(self, sub_message: str | tuple[str, ...]) -> bool:
        if isinstance(sub_message, tuple):
            for sm in sub_message:
                if sm not in self:
                    return False
            return True
        for message in self.violations:
            if sub_message in message:
                return True
        return False
