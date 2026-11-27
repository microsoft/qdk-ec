"""Build, relocate, and consume the qodec CMake package on this host."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
C_BINDINGS = ROOT / "bindings/c"
CONSUMER = C_BINDINGS / "tests/cmake-consumer"
FIXTURE = ROOT / "examples/repetition3/repetition3.qodec.yaml"
STATIC_CRT = "MultiThreaded$<$<CONFIG:Debug>:Debug>"


@dataclass(frozen=True)
class Case:
    shared: bool
    configuration: str
    static_crt: bool = False

    @property
    def name(self) -> str:
        linkage = "shared" if self.shared else "static"
        crt = "-static-crt" if self.static_crt else ""
        return f"{linkage}-{self.configuration.lower()}{crt}"

    def options(self) -> list[str]:
        options = [f"-DCMAKE_BUILD_TYPE={self.configuration}"]
        options.append(f"-DBUILD_SHARED_LIBS={'ON' if self.shared else 'OFF'}")
        if self.static_crt:
            options.append(f"-DCMAKE_MSVC_RUNTIME_LIBRARY={STATIC_CRT}")
        return options


def cases_for(platform: str) -> list[Case]:
    cases = [
        Case(shared, configuration)
        for shared in (False, True)
        for configuration in ("Debug", "Release")
    ]
    if platform == "win32":
        cases.extend(Case(False, configuration, static_crt=True) for configuration in ("Debug", "Release"))
    return cases


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def build(source: Path, directory: Path, configuration: str, *options: str) -> None:
    run(["cmake", "-S", str(source), "-B", str(directory), *options])
    run(["cmake", "--build", str(directory), "--config", configuration])


def executable(build_directory: Path, case: Case, name: str) -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    for path in (build_directory / f"{name}{suffix}",
                 build_directory / case.configuration / f"{name}{suffix}"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"{name} was not built in {build_directory}")


def check_case(case: Case, work: Path) -> None:
    print(f"== {case.name}", flush=True)
    installed = work / "install"
    relocated = work / "relocated"
    build_directory = work / "build"

    build(C_BINDINGS, build_directory, case.configuration, *case.options())
    run(["cmake", "--install", str(build_directory), "--config", case.configuration,
         "--prefix", str(installed)])
    shutil.rmtree(build_directory)
    shutil.move(installed, relocated)

    consumer_build = work / "consumer"
    build(CONSUMER, consumer_build, case.configuration,
          *case.options(), f"-DCMAKE_PREFIX_PATH={relocated}")
    for name in ("consumer_c", "consumer_cxx"):
        run([str(executable(consumer_build, case, name)), str(FIXTURE),
             str(work / "missing.qodec.yaml")])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Run only the named case; repeatable")
    arguments = parser.parse_args()

    cases = cases_for(sys.platform)
    if arguments.case:
        unknown = set(arguments.case) - {case.name for case in cases}
        if unknown:
            parser.error(f"unknown cases {sorted(unknown)}; choose from {[case.name for case in cases]}")
        cases = [case for case in cases if case.name in arguments.case]

    for case in cases:
        with tempfile.TemporaryDirectory(prefix=f"qodec-cmake-{case.name}-") as directory:
            check_case(case, Path(directory))
    print(f"PASS: {len(cases)} CMake package cases", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
