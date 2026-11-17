"""Exercise process limits and cleanup with real, bounded subprocesses."""

from pathlib import Path
import sys

import psutil
import pytest

from study_utils import run_process, sampling_plan


def test_guards_report_success_and_nonzero_exit(tmp_path: Path):
    log = tmp_path / "worker.log"
    result = run_process(
        [sys.executable, "-c", "print('done')"], log, memory_gib=1, timeout_seconds=5
    )
    assert log.read_text().strip() == "done"
    assert result["wall_seconds"] > 0
    with pytest.raises(RuntimeError, match="exited with 7"):
        run_process(
            [sys.executable, "-c", "raise SystemExit(7)"],
            log,
            memory_gib=1,
            timeout_seconds=5,
        )


def test_subprocess_working_directory_does_not_change_the_caller(tmp_path: Path):
    previous = Path.cwd()
    log = tmp_path / "cwd.log"
    run_process(
        [sys.executable, "-c", "from pathlib import Path; print(Path.cwd())"],
        log,
        memory_gib=1,
        timeout_seconds=5,
        cwd=tmp_path,
    )
    assert Path(log.read_text().strip()) == tmp_path
    assert Path.cwd() == previous


@pytest.mark.parametrize(
    "memory,timeout,error", [(1, 0.6, TimeoutError), (0.000001, 5, MemoryError)]
)
def test_resource_guards_stop_the_worker(
    tmp_path: Path, memory: float, timeout: float, error: type[Exception]
):
    log = tmp_path / "worker.log"
    code = "import os,time; print(os.getpid(), flush=True); time.sleep(20)"
    with pytest.raises(error):
        run_process(
            [sys.executable, "-c", code],
            log,
            memory_gib=memory,
            timeout_seconds=timeout,
        )
    if log.read_text().strip():
        assert not psutil.pid_exists(int(log.read_text().strip()))


def test_timeout_stops_descendants(tmp_path: Path):
    log = tmp_path / "worker.log"
    code = (
        "import os,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); "
        "print(os.getpid(),child.pid,flush=True); time.sleep(20)"
    )
    with pytest.raises(TimeoutError):
        run_process(
            [sys.executable, "-c", code], log, memory_gib=1, timeout_seconds=0.6
        )
    parent, child = map(int, log.read_text().split())
    assert not psutil.pid_exists(parent)
    assert (
        not psutil.pid_exists(child)
        or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
    )


def test_quota_counts_and_invalid_limits(tmp_path: Path):
    assert sampling_plan([1, 4], 2) == [
        {"width": 1, "shots": 100, "intermediate": 0, "final": 100},
        {"width": 4, "shots": 112, "intermediate": 1008, "final": 448},
    ]
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            run_process(
                [sys.executable],
                tmp_path / "never.log",
                memory_gib=value,
                timeout_seconds=1,
            )
