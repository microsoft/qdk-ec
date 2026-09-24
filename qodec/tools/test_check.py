import contextlib
import importlib.util
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import yaml


spec = importlib.util.spec_from_file_location("qodec_checks", Path(__file__).with_name("check.py"))
checks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checks
spec.loader.exec_module(checks)


class CheckRunnerTests(unittest.TestCase):
    def test_maturin_package_and_executable_versions_agree(self):
        expected = (checks.ROOT.parent / "requirements-build.txt").read_text().splitlines()[0].split("==")[1]
        self.assertEqual(version("maturin"), expected)
        result = subprocess.run([sys.executable, "-m", "maturin", "--version"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), f"maturin {expected}")

    def test_native_build_tools_use_shared_pins(self):
        root = checks.ROOT.parent
        self.assertEqual((root / "requirements-build.txt").read_text().splitlines(), [
            "maturin==1.15.0", "uv==0.11.32", 'ziglang==0.14.1; sys_platform == "linux"',
        ])
        for filename in (".ado/stages/build.yaml", ".ado/templates/build-wheels-steps.yaml",
                         ".ado/templates/build-python-bindings-steps.yaml", ".github/workflows/build.yaml",
                         ".github/workflows/qodec-wheels.yaml"):
            with self.subTest(filename=filename):
                source = (root / filename).read_text()
                yaml.safe_load(source)
                installs = [line for line in source.splitlines() if "pip install" in line]
                self.assertTrue(any("-r requirements-build.txt" in line for line in installs))
                self.assertFalse(any("maturin" in line for line in installs))
                if "--zig" in source:
                    self.assertIn('export CARGO_ZIGBUILD_PYTHON_PATH="$PWD/tools/zig.py"', source)

    @unittest.skipUnless(sys.platform == "linux", "Zig is used for Linux native wheels")
    def test_zig_adapter_preserves_other_compiler_arguments(self):
        spec = importlib.util.spec_from_file_location("zig_adapter", checks.ROOT.parent / "tools/zig.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        for compiler in ("cc", "c++"):
            arguments = [compiler, "-O3", "-Wl,-O1", "-Wl,--as-needed", "source with spaces.c"]
            self.assertEqual(adapter.zig_arguments(arguments), [compiler, "-O3", "-Wl,--as-needed", "source with spaces.c"])
            self.assertEqual(adapter.zig_arguments(["-m", "ziglang", *arguments]), adapter.zig_arguments(arguments))
            self.assertIn("-Wl,-O1", arguments)
        self.assertEqual(adapter.zig_arguments(["version"]), ["version"])
        self.assertEqual(adapter.zig_arguments(["ar", "-Wl,-O1"]), ["ar", "-Wl,-O1"])

    @unittest.skipUnless(sys.platform == "linux", "Zig is used for Linux native wheels")
    def test_zig_adapter_links_and_reports_invalid_flags(self):
        adapter = checks.ROOT.parent / "tools/zig.py"
        environment = dict(os.environ, CARGO_ZIGBUILD_PYTHON_PATH=str(adapter),
                           PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-m", "maturin", "zig", "cc", "--", "-x", "c", "-", "-shared", "-O3", "-Wl,-O1",
                       "-o", str(Path(directory) / "probe.so")]
            result = subprocess.run(command, input="int probe(void) { return 0; }\n", env=environment, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            invalid = subprocess.run(command + ["-Wl,--qdk-unknown-flag"], input="", env=environment, text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("--qdk-unknown-flag", invalid.stderr)

    def test_qodec_uses_native_publishing_and_forces_builds(self):
        pipeline = yaml.safe_load((checks.ROOT.parent / ".ado/publish.yaml").read_text())
        stages = pipeline["extends"]["parameters"]["stages"]
        build = next(stage for stage in stages if stage["template"] == "stages/build.yaml@self")
        publisher = next(stage for stage in stages if stage["template"] == "stages/publish_python.yaml@self")
        self.assertEqual(build["parameters"]["buildAndTest"], "${{ or(parameters.buildAndTest, parameters.publishDeqagramPython, parameters.publishQodecPython) }}")
        self.assertEqual(publisher["parameters"]["publishQodecPython"], "${{ parameters.publishQodecPython }}")
        self.assertFalse(any(stage["parameters"].get("packageName") == "qodec" for stage in stages))
        template = yaml.safe_load((checks.ROOT.parent / ".ado/stages/publish_python.yaml").read_text())
        stage = template["stages"][0]
        self.assertEqual(stage["dependsOn"], "build")
        self.assertIn("eq(variables['Build.Reason'], 'Manual')", stage["condition"])
        self.assertIn("parameters.publishQodecPython", stage["condition"])
        steps = stage["jobs"][0]["steps"]
        collect = next(step for step in steps if step["displayName"] == "Collect qodec wheels")
        self.assertEqual(collect["condition"], "eq(${{ parameters.publishQodecPython }}, true)")
        self.assertIn("qodec-*.whl", collect["script"])
        self.assertEqual(sum(step.get("task") == "EsrpRelease@9" for step in steps), 1)

    def test_native_builds_probe_qodec_wheels_before_upload(self):
        pipeline = yaml.safe_load((checks.ROOT.parent / ".ado/stages/build.yaml").read_text())
        jobs = pipeline["stages"][0]["jobs"][0]["${{ each platform in parameters.platforms }}"]
        job = jobs[0]
        steps = job["steps"]
        commands = [step.get("bash", step.get("pwsh", "")) for step in steps]
        builds = [command for command in commands if "--out \"target/qodec-wheels/$version\"" in command]
        self.assertEqual(len(builds), 2)
        for command in builds:
            self.assertIn("--manifest-path qodec/bindings/python/Cargo.toml", command)
            self.assertIn("--interpreter", command)
            self.assertIn("uv python install 3.11 3.14t 3.15.0b4 3.15.0b4+freethreaded", command)
            self.assertLess(command.index("maturin build"), command.index("qodec/tools/check_wheel.py"))
        source = next(step for step in steps if step.get("displayName") == "Build qodec source distribution")
        self.assertIn("eq(variables['arch'], 'x86_64')", source["condition"])
        self.assertIn("eq(variables['Agent.OS'], 'Linux')", source["condition"])
        self.assertIn("maturin sdist --manifest-path qodec/bindings/python/Cargo.toml --out target/wheels", source["bash"])

    @unittest.skipUnless(sys.platform == "linux", "release collection runs in a Linux job")
    def test_release_collection_requires_complete_qodec_artifacts(self):
        template = yaml.safe_load((checks.ROOT.parent / ".ado/stages/publish_python.yaml").read_text())
        steps = template["stages"][0]["jobs"][0]["steps"]
        validation = next(step["bash"] for step in steps if step["displayName"] == "Validate all three native ABI families")
        script = validation + "\n" + next(step["script"] for step in steps if step["displayName"] == "Collect qodec wheels")
        for package in ("Binar", "Paulimer", "Deqagram", "Qodec"):
            script = script.replace("${{ parameters.publish" + package + "Python }}", str(package == "Qodec"))
        cases = ("complete", "missing wheel", "extra wheel", "duplicate wheel", "missing sdist", "extra sdist", "no platforms")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                expected = self.make_release_artifacts(root, case)
                command = script.replace("$(System.DefaultWorkingDirectory)", str(root))
                result = subprocess.run(["bash"], input=command, cwd=root, text=True, capture_output=True)
                self.assertEqual(result.returncode == 0, case == "complete", result.stderr)
                if case == "complete":
                    self.assertEqual({path.name for path in (root / "target/wheels").iterdir()}, expected)

    def make_release_artifacts(self, root, case):
        (root / "target/wheels").mkdir(parents=True, exist_ok=True)
        expected = set()
        if case == "no platforms":
            return expected
        for platform in ("linux_x86_64", "windows_aarch64"):
            directory = root / "artifacts" / platform
            directory.mkdir(parents=True)
            for tag in ("cp311-abi3", "cp314-cp314t", "cp315-abi3.abi3t"):
                wheel_platform = "linux_x86_64" if case == "duplicate wheel" else platform
                wheel = f"qodec-0.1.0-{tag}-{wheel_platform}.whl"
                if not (case == "missing wheel" and platform == "windows_aarch64" and tag == "cp315-abi3.abi3t"):
                    (directory / wheel).write_text("fixture")
                    expected.add(wheel)
            (directory / f"binar-0.1.0-cp311-abi3-{platform}.whl").write_text("not selected")
        source = root / "artifacts/linux_x86_64"
        if case != "missing sdist":
            (source / "qodec-0.1.0.tar.gz").write_text("fixture")
            expected.add("qodec-0.1.0.tar.gz")
        if case == "extra wheel":
            (source / "qodec-0.0.1-cp311-abi3-linux_x86_64.whl").write_text("stale")
        if case == "extra sdist":
            (source / "qodec-0.0.1.tar.gz").write_text("stale")
        return expected

    def test_native_wheels_share_the_import_check_and_keep_release_targets(self):
        azure = yaml.safe_load((checks.ROOT.parent / ".ado/publish.yaml").read_text())
        github = yaml.safe_load((checks.ROOT.parent / ".github/workflows/qodec-wheels.yaml").read_text())
        platforms = next(parameter["default"] for parameter in azure["parameters"] if parameter["name"] == "platforms")
        self.assertEqual({(platform["os"], platform["arch"]) for platform in platforms}, {
            (system, arch) for system in ("linux", "macOS", "windows") for arch in ("x86_64", "aarch64")
        })
        targets = {entry["target"] for entry in github["jobs"]["wheel"]["strategy"]["matrix"]["include"]}
        self.assertEqual(targets, {"x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc", "universal2-apple-darwin"})
        self.assertIn("python qodec/tools/check_wheel.py target/wheels", [step.get("run") for step in github["jobs"]["wheel"]["steps"]])
        build = next(step for step in github["jobs"]["wheel"]["steps"] if step.get("name") == "Build wheel")
        self.assertEqual(build["working-directory"], "qodec/bindings/python")
        self.assertIn("--out ../../../target/wheels", build["run"])

    def test_ci_uses_the_same_runner_scopes(self):
        expected = {"rust", "python", "docs", "coverage", "packaging"}
        workflow = yaml.safe_load((checks.ROOT.parent / ".github/workflows/build.yaml").read_text())
        job = workflow["jobs"]["qodec"]
        scopes = {}
        for step in job["steps"]:
            for line in step.get("run", "").splitlines():
                command = shlex.split(line, comments=True)
                if command[:2] == ["python", "qodec/tools/check.py"]:
                    self.assertNotIn(command[2], scopes)
                    scopes[command[2]] = step.get("if")
        self.assertEqual(set(scopes), expected)
        self.assertTrue(expected <= set(checks.SCOPES))
        self.assertEqual(job["strategy"]["matrix"]["python-version"], ["3.11", "3.12"])
        self.assertEqual(job["strategy"]["matrix"]["include"], [{"python-version": "3.12", "full": True}])
        self.assertEqual({scope for scope, condition in scopes.items() if condition == "matrix.full"}, {"docs", "coverage", "packaging"})

    def test_ci_cached_qodec_builds_use_portable_cpu_flags(self):
        workflow = yaml.safe_load((checks.ROOT.parent / ".github/workflows/build.yaml").read_text())
        job = workflow["jobs"]["qodec"]
        self.assertEqual(job.get("env", {}).get("RUSTFLAGS"), "-C target-cpu=x86-64-v3")
        cache = next(step["with"] for step in job["steps"] if step.get("name") == "Cache cargo build")
        self.assertTrue(cache["key"].startswith("v2-"))
        self.assertIn("${{ env.RUSTFLAGS }}", cache["key"])
        self.assertIn("${{ matrix.python-version }}", cache["key"])
        self.assertNotIn("restore-keys", cache)

    def test_ci_test_profile_only_disables_lto(self):
        manifest = tomllib.loads((checks.ROOT.parent / "Cargo.toml").read_text())
        self.assertEqual(manifest["profile"]["ci-test"], {"inherits": "release", "lto": "off"})
        self.assertEqual(manifest["profile"]["release"], {"lto": True, "codegen-units": 1})

    def test_ci_native_wheels_keep_the_release_profile(self):
        root = checks.ROOT.parent
        for filename in (".github/workflows/build.yaml", ".github/workflows/qodec-wheels.yaml", ".ado/stages/build.yaml"):
            source = (root / filename).read_text()
            workflow = yaml.safe_load(source)
            if "jobs" in workflow:
                jobs = workflow["jobs"].values()
            else:
                jobs = workflow["stages"][0]["jobs"][0]["${{ each platform in parameters.platforms }}"]
            scripts = [step[key] for job in jobs for step in job.get("steps", []) for key in ("run", "bash", "pwsh", "script") if key in step]
            commands = [line for script in scripts for line in script.splitlines() if line.lstrip().startswith(("maturin build ", "maturin develop "))]
            self.assertTrue(commands, filename)
            for command in commands:
                with self.subTest(filename=filename, command=command):
                    self.assertNotIn("--profile", command)
                    self.assertTrue("--release" in command or '"${maturin_args[@]}"' in command)
            if '"${maturin_args[@]}"' in source:
                self.assertIn("maturin_args=(--release --strip)", source)

    def test_parent_workspace_tests_build_libraries_with_the_same_profile(self):
        github = yaml.safe_load((checks.ROOT.parent / ".github/workflows/build.yaml").read_text())
        steps = github["jobs"]["test"]["steps"]
        callers = [step["run"] for step in steps if "cargo test --workspace" in step.get("run", "")]
        self.assertEqual(len(callers), 2)
        for command in callers:
            for package in ("qodec-c", "deq-decoder-reference-plugin"):
                self.assertLess(command.index(f"cargo build --profile ci-test -p {package}"), command.index("cargo test --workspace"))
            tests = [shlex.split(line) for line in command.splitlines() if line.startswith("cargo test ")]
            self.assertEqual(len(tests), 2)
            for arguments in tests:
                self.assertEqual(arguments[arguments.index("--profile") + 1], "ci-test")
                self.assertNotIn("--release", arguments)

    def test_azure_workspace_tests_build_libraries_with_the_same_profile(self):
        pipeline = yaml.safe_load((checks.ROOT.parent / ".ado/stages/build.yaml").read_text())
        job = pipeline["stages"][0]["jobs"][0]["${{ each platform in parameters.platforms }}"][0]
        commands = [step["script"] for step in job["steps"] if step.get("script", "").startswith("cargo ")]
        test = next(command for command in commands if command.startswith("cargo test --workspace"))
        for package in ("qodec-c", "deq-decoder-reference-plugin"):
            self.assertLess(commands.index(f"cargo build -p {package} --profile ci-test"), commands.index(test))
        arguments = shlex.split(test)
        self.assertEqual(arguments[arguments.index("--profile") + 1], "ci-test")
        self.assertNotIn("--release", arguments)

    def test_azure_retains_rust_timings_after_test_failures(self):
        pipeline = yaml.safe_load((checks.ROOT.parent / ".ado/stages/build.yaml").read_text())
        job = pipeline["stages"][0]["jobs"][0]["${{ each platform in parameters.platforms }}"][0]
        command = next(step["script"] for step in job["steps"] if step.get("script", "").startswith("cargo test --workspace"))
        self.assertIn("--timings", shlex.split(command))
        artifacts = job["templateContext"]["outputs"]
        timings = next(artifact for artifact in artifacts if artifact["artifactName"] == "${{ platform.name }}-rust-timings")
        self.assertEqual(timings["output"], "pipelineArtifact")
        self.assertEqual(timings["condition"], "succeededOrFailed()")
        self.assertEqual(timings["targetPath"], "$(System.DefaultWorkingDirectory)/target/cargo-timings")

    def test_supported_scopes(self):
        self.assertEqual(set(checks.SCOPES), {"rust", "python", "docs", "coverage", "examples", "packaging", "all"})

    def test_all_includes_all_gates_once(self):
        groups = [checks.steps_for(scope, sys.executable) for scope in checks.SCOPES if scope != "all"]
        steps = checks.steps_for("all", sys.executable)
        self.assertEqual(steps, sum(groups, []))
        self.assertEqual(len(steps), 16)
        self.assertEqual(len({step.command for step in steps}), len(steps))

    def test_all_requires_example_audits_and_packaging(self):
        examples = [checks.Step((sys.executable, "-m", "pytest", "examples/tests", "-q"))]
        packaging = [checks.Step((sys.executable, "-m", "unittest", "discover", "-s", "tools", "-p", "test_*.py"))]
        self.assertEqual(checks.steps_for("examples", sys.executable), examples)
        self.assertEqual(checks.steps_for("packaging", sys.executable), packaging)
        for step in examples + packaging:
            self.assertIn(step, checks.steps_for("all", sys.executable))

    def test_python_tools_and_directories(self):
        python = "/selected interpreter/bin/python"
        steps = checks.steps_for("python", python)
        self.assertTrue(all(step.command[0] == python for step in steps))
        self.assertEqual(steps[0].command[1:], ("-m", "maturin", "develop", "--release"))
        self.assertTrue(all(step.cwd == checks.BINDINGS for step in steps if "-m" in step.command))
        self.assertIn("tests", steps[2].command)
        self.assertEqual(steps[3].command, (python, "-m", "mypy.stubtest", "qodec", "--allowlist", "stubtest-allowlist.txt"))
        self.assertEqual(steps[-1].command[-2:], ("pytest", "-q"))

    def test_rust_gates_select_only_qodec_packages(self):
        package_args = ("-p", "qodec", "-p", "qodec-python", "-p", "qodec-c")
        commands = [step.command for step in checks.steps_for("rust", sys.executable)]
        self.assertEqual(commands, [
            ("cargo", "fmt", *package_args, "--", "--check"),
            ("cargo", "clippy", *package_args, "--all-targets", "--all-features", "--", "-D", "clippy::pedantic"),
            ("cargo", "build", "-p", "qodec-c"),
            ("cargo", "test", *package_args, "--all-features"),
        ])

    def test_documentation_and_coverage_exclude_sibling_packages(self):
        rustdoc = checks.steps_for("docs", sys.executable)[0]
        self.assertEqual(rustdoc.command, (
            "cargo", "doc", "-p", "qodec", "-p", "qodec-python", "-p", "qodec-c", "--no-deps",
        ))
        self.assertEqual(rustdoc.extra_env, {"RUSTDOCFLAGS": "-D warnings"})
        self.assertEqual(checks.steps_for("coverage", sys.executable), [
            checks.Step(("cargo", "llvm-cov", "-p", "qodec", "--summary-only", "--fail-under-lines", "88")),
        ])

    def test_environment_does_not_mutate_parent(self):
        inherited = {"CONDA_PREFIX": "/old", "VIRTUAL_ENV": "/older", "PATH": "/compiler tools", "CC": "cc"}
        original = inherited.copy()
        selected = checks.selected_environment("/chosen/bin/python", Path("/chosen"), Path("/base"), inherited)
        self.assertEqual(inherited, original)
        self.assertEqual(Path(selected["VIRTUAL_ENV"]), Path("/chosen"))
        self.assertNotIn("CONDA_PREFIX", selected)
        self.assertEqual(selected["CC"], "cc")
        self.assertTrue(selected["PATH"].endswith("/compiler tools"))
        self.assertEqual(selected["PYO3_PYTHON"], "/chosen/bin/python")

    def test_coverage_toolchain_changes_only_the_coverage_step(self):
        default = checks.steps_for("all", sys.executable)
        selected = checks.steps_for("all", sys.executable, coverage_toolchain="stable")
        changes = [(before, after) for before, after in zip(default, selected) if before != after]
        self.assertEqual(len(changes), 1)
        before, after = changes[0]
        self.assertEqual(after.command, ("rustup", "run", "stable", *before.command))
        self.assertEqual(after.cwd, before.cwd)
        self.assertEqual(after.extra_env, {})

    def test_coverage_rejects_utc_before_running_tests(self):
        result = subprocess.CompletedProcess([], 0, stdout="rustc 1.97.1\nx86_64-utc-builder-path: amd64/r2c2.dll\n")
        with patch.object(checks.subprocess, "run", return_value=result):
            message = checks.coverage_compiler_error({}, None)
        self.assertIn("ignores -Cinstrument-coverage", message)
        self.assertIn("--coverage-toolchain", message)

    def test_coverage_inspects_the_explicit_toolchain(self):
        environment = {"PATH": "/tools"}
        result = subprocess.CompletedProcess([], 0, stdout="rustc 1.98.1\nLLVM version: 22.1.8\n")
        with patch.object(checks.subprocess, "run", return_value=result) as run:
            self.assertIsNone(checks.coverage_compiler_error(environment, "stable"))
        run.assert_called_once_with(("rustup", "run", "stable", "rustc", "-vV"), cwd=checks.ROOT,
                                    env=environment, text=True, capture_output=True, check=False)
        self.assertEqual(environment, {"PATH": "/tools"})

    def test_coverage_reports_unavailable_toolchains(self):
        result = subprocess.CompletedProcess([], 1, stderr="toolchain not installed")
        with patch.object(checks.subprocess, "run", return_value=result):
            self.assertIn("toolchain not installed", checks.coverage_compiler_error({}, "missing"))

    def test_conda_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / "conda-meta").mkdir()
            selected = checks.selected_environment(str(prefix / "bin/python"), prefix, prefix, {"VIRTUAL_ENV": "/wrong"})
            self.assertEqual(selected["CONDA_PREFIX"], str(prefix))
            self.assertNotIn("VIRTUAL_ENV", selected)

    def test_failure_stops_remaining_gates(self):
        steps = checks.steps_for("all", sys.executable)
        with patch.object(checks.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as run:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(checks.execute(steps, {}, dry_run=False), 7)
            self.assertEqual(run.call_count, 1)

    def test_dry_run_never_executes(self):
        with patch.object(checks.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(checks.execute(checks.steps_for("all", sys.executable), {}, dry_run=True), 0)
            run.assert_not_called()
            self.assertIn("DRY RUN: no checks executed", output.getvalue())

    def test_real_subprocess_failure_and_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            steps = [checks.Step((sys.executable, "-c", "import os, sys; sys.exit(7 if os.getcwd() == sys.argv[1] else 8)", str(cwd)), cwd)]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(checks.execute(steps, dict(checks.os.environ), dry_run=False), 7)

    def test_import_guard_accepts_checkout_rejects_other_package(self):
        guard = checks.steps_for("python", sys.executable)[1].command[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "qodec"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            for target, succeeds in ((package, True), (root / "other", False)):
                result = subprocess.run([sys.executable, "-c", guard, str(target)], cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, succeeds)

    def test_runner_dry_run_from_workspace_root(self):
        result = subprocess.run(
            [sys.executable, "qodec/tools/check.py", "all", "--dry-run"],
            cwd=checks.ROOT.parent, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Python: {sys.executable}", result.stdout)
        self.assertIn(f"cwd={checks.ROOT}", result.stdout)
        self.assertIn(f"cwd={checks.BINDINGS}", result.stdout)
        self.assertIn("DRY RUN: no checks executed", result.stdout)

    def test_root_tasks_use_selected_interpreter(self):
        config = json.loads((checks.ROOT.parent / ".vscode/tasks.json").read_text())
        tasks = [task for task in config["tasks"] if task["label"].startswith("qodec: ")]
        self.assertEqual(len(tasks), len(checks.SCOPES))
        self.assertEqual({task["args"][-1] for task in tasks}, set(checks.SCOPES))
        for task in tasks:
            self.assertEqual(task["type"], "process")
            self.assertEqual(task["command"], "${command:python.interpreterPath}")
            self.assertEqual(task["args"][0], "${workspaceFolder}/qodec/tools/check.py")
            self.assertEqual(task["options"]["cwd"], "${workspaceFolder}")

    def test_qodec_instructions_are_scoped_to_the_package(self):
        directory = checks.ROOT.parent / ".github/instructions"
        expected = {
            "qodec.instructions.md": "qodec/**",
            "qodec-python.instructions.md": "qodec/bindings/python/**",
            "qodec-model.instructions.md": "qodec/src/**,qodec/schemas/**,qodec/tests/**,qodec/examples/**,qodec/docs/concepts/**",
            "qodec-checks.instructions.md": None,
        }
        for filename, scope in expected.items():
            with self.subTest(filename=filename):
                metadata = yaml.safe_load((directory / filename).read_text().split("---", 2)[1])
                self.assertTrue(metadata["description"])
                self.assertEqual(metadata.get("applyTo"), scope)


if __name__ == "__main__":
    unittest.main()