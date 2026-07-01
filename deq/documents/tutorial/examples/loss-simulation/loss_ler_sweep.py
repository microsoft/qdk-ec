"""Loss-tolerance LER sweep for the QDK loss-simulation tutorial.

For each ``(d, variant, p_loss)`` point the driver transpiles
``repetition_code.deq`` with the appropriate Mako parameters, then
delegates each shot batch to :func:`deq.cli.simulate._run_batch` (the
same helper that powers ``deq simulate ler``) with ``simulator="qdk"``.
Logical errors accumulate until ``--target-errors`` is reached or
``--max-shots`` is spent.  Batches from all points are round-robin
dispatched across ``--workers`` parallel subprocesses so the plot
fills in all curves at once.  JSON + PNG are rewritten after every
batch.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import sys
import time
from typing import Sequence

import deq_runtime  # noqa: F401  (ensure the extension is importable)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(THIS_DIR, "repetition_code.deq")


@dataclasses.dataclass
class Point:
    """One ``(d, variant, p_loss)`` datapoint.  This is the JSON schema."""

    variant: str       # "baseline" or "replenish"
    d: int
    p: float
    p_loss: float
    rounds: int
    shots: int
    logical_errors: int
    wall_time: float

    @property
    def ler(self) -> float:
        return self.logical_errors / self.shots if self.shots else float("nan")

    @property
    def ler_se(self) -> float:
        # Wilson-style standard error for a binomial proportion.
        if self.shots == 0:
            return float("nan")
        p = self.ler
        return max((p * (1 - p) / self.shots) ** 0.5, 0.5 / self.shots)


def build_artifacts(
    *,
    d: int,
    p: float,
    p_loss: float,
    rounds: int,
    replenish: bool,
    out_dir: str,
    program: str = "Memory",
) -> tuple[str, str]:
    """Transpile the template with Mako params + static-compile; return (stim, bin) paths."""
    from deq.cli.jit import transpile
    from deq.compiler.jit_compiler import static_jit_compiler
    import deq.proto.deq_jit_pb2 as jit_pb

    jit_path = os.path.join(out_dir, f"{program}.deq.jit")
    stim_path = os.path.join(out_dir, f"{program}.stim")
    bin_path = os.path.join(out_dir, f"{program}.deq.bin")
    transpile(
        TEMPLATE_PATH,
        out=jit_path,
        program=program,
        jobs=1,
        mako=[
            f"d={d}", f"p={p}", f"p_loss={p_loss}", f"rounds={rounds}",
            f"replenish={'1' if replenish else '0'}",
        ],
        skip_mako_warning=True,
    )
    with open(jit_path, "rb") as f:
        lib = jit_pb.JitLibrary.FromString(f.read())
    with open(bin_path, "wb") as f:
        f.write(static_jit_compiler(lib).SerializeToString())
    return stim_path, bin_path


def run_server_batch(
    *,
    bin_path: str,
    stim_path: str,
    shots: int,
    seed: int,
    decoder: str,
) -> tuple[int, int]:
    """Run one deq.runtime server subprocess; return (shots, logical_errors).

    Delegates to :func:`deq.cli.simulate._run_batch` — the same helper
    that powers ``deq simulate ler``.  We ask for ``simulator="qdk"`` so
    the wrapper plumbs the ``@qdk_sampler`` builtin adapter.
    """
    from deq.cli.simulate import _run_batch

    result = _run_batch(
        bin_path=bin_path,
        stim_path=stim_path,
        jit_path="",
        batch_size=shots,
        max_errors=shots,  # never stop early on errors — collect the full batch
        decoder=decoder,
        decoder_config=None,
        coordinator="monolithic",
        coordinator_config=json.dumps({"loss_random_imputation_seed": seed}),
        seed=seed,
        debug_dir=None,
        simulator="qdk",
    )
    return int(result.get("shots", 0)), int(result.get("logical_errors", 0))


@dataclasses.dataclass
class _PointState:
    """Mutable per-point scheduling state; wraps a Point + build artifacts."""

    point: Point
    bin_path: str
    stim_path: str
    target_errors: int
    max_shots: int
    shots_per_batch: int
    decoder: str
    base_seed: int
    next_batch_idx: int = 0

    def is_done(self) -> bool:
        return (
            self.point.logical_errors >= self.target_errors
            or self.point.shots >= self.max_shots
        )

    def label(self) -> str:
        p = self.point
        return f"{p.variant} d={p.d} p_loss={p.p_loss:g}"


def _run_one_batch(state: _PointState, seed: int) -> tuple[int, int, float]:
    """Pure worker: run one batch; caller accumulates into state."""
    t0 = time.time()
    got_shots, got_errors = run_server_batch(
        bin_path=state.bin_path,
        stim_path=state.stim_path,
        shots=min(state.shots_per_batch, state.max_shots - state.point.shots),
        seed=seed,
        decoder=state.decoder,
    )
    return got_shots, got_errors, time.time() - t0


def sweep(
    *,
    distances: Sequence[int],
    loss_rates: Sequence[float],
    max_shots: int,
    target_errors: int,
    shots_per_batch: int,
    p: float | None,
    rounds_factor: int,
    decoder: str,
    seed_base: int,
    workdir: str,
    workers: int = 1,
    out_json: str | None = None,
    out_png: str | None = None,
) -> list[Point]:
    """Parallel round-robin sweep.  Every point gets one batch before any
    point starts a second one, so the plot fills in all curves at once.
    After each batch we rewrite JSON + PNG so anyone tailing the files
    sees gradual refinement.
    """
    # Pre-build all points sequentially (transpile is cheap).
    seed_step = max(1, max_shots // shots_per_batch) + 1
    states: list[_PointState] = []
    seed = seed_base
    for d in distances:
        rounds = rounds_factor * d
        for variant in ("baseline", "replenish"):
            for p_loss in loss_rates:
                p_eff = p if p is not None else p_loss / 10.0
                point_dir = os.path.join(workdir, f"{variant}_d{d}_pl{p_loss:g}")
                os.makedirs(point_dir, exist_ok=True)
                stim_path, bin_path = build_artifacts(
                    d=d, p=p_eff, p_loss=p_loss, rounds=rounds,
                    replenish=(variant == "replenish"), out_dir=point_dir,
                )
                states.append(_PointState(
                    point=Point(
                        variant=variant, d=d, p=p_eff, p_loss=p_loss,
                        rounds=rounds, shots=0, logical_errors=0, wall_time=0.0,
                    ),
                    bin_path=bin_path, stim_path=stim_path,
                    target_errors=target_errors, max_shots=max_shots,
                    shots_per_batch=shots_per_batch, decoder=decoder,
                    base_seed=seed,
                ))
                seed += seed_step
    print(f"built {len(states)} points; running with {workers} parallel worker(s)")

    def _refresh() -> None:
        snap = [s.point for s in states]
        if out_json is not None:
            _write_json(snap, out_json)
        if out_png is not None:
            plot_sweep(snap, out_png)

    rr_cursor = 0

    def _pick(busy: set[int]) -> int | None:
        """Round-robin: next undone, not-already-in-flight state."""
        nonlocal rr_cursor
        n = len(states)
        for offset in range(n):
            idx = (rr_cursor + offset) % n
            if idx not in busy and not states[idx].is_done():
                rr_cursor = (idx + 1) % n
                return idx
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        in_flight: dict[concurrent.futures.Future, int] = {}

        def _submit(idx: int) -> None:
            st = states[idx]
            fut = pool.submit(_run_one_batch, st, st.base_seed + st.next_batch_idx)
            st.next_batch_idx += 1
            in_flight[fut] = idx

        def _refill() -> None:
            while len(in_flight) < workers:
                idx = _pick(set(in_flight.values()))
                if idx is None:
                    return
                _submit(idx)

        _refill()
        while in_flight:
            done, _pending = concurrent.futures.wait(
                in_flight, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                idx = in_flight.pop(fut)
                st = states[idx]
                try:
                    got_shots, got_errors, elapsed = fut.result()
                except Exception as exc:  # pragma: no cover — surface and skip
                    print(f"  [{st.label()}] FAILED: {exc}")
                    st.point.shots = st.max_shots
                    st.point.logical_errors = st.target_errors
                else:
                    st.point.shots += got_shots
                    st.point.logical_errors += got_errors
                    st.point.wall_time += elapsed
                    ler = st.point.logical_errors / max(1, st.point.shots)
                    print(
                        f"  [{st.label()}] batch: {got_errors}/{got_shots} "
                        f"\u2192 cum {st.point.logical_errors}/{st.point.shots} "
                        f"(LER \u2248 {ler:.3e}) {elapsed:.1f}s"
                    )
                _refresh()
            _refill()

    return [s.point for s in states]


def _write_json(points: list[Point], out_json: str) -> None:
    """Atomically rewrite the sweep JSON so partial runs stay valid."""
    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(pt) for pt in points], f, indent=2)
    os.replace(tmp, out_json)


def plot_sweep(points: list[Point], out_png: str) -> None:
    """LER vs p_loss curves.  Zero-error points render as 95 % Wilson upper limits."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return

    def wilson_upper_95(k: int, n: int) -> float:
        if n == 0:
            return 1.0
        z = 1.96
        denom = 1.0 + z * z / n
        center = (k + z * z / 2.0) / n
        rad = z / n * ((k * (n - k) / n + z * z / 4.0) ** 0.5)
        return min(1.0, (center + rad) / denom)

    by_curve: dict[tuple[str, int], list[Point]] = {}
    for pt in points:
        by_curve.setdefault((pt.variant, pt.d), []).append(pt)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {3: "#1f77b4", 5: "#d62728", 7: "#2ca02c", 9: "#9467bd"}
    styles = {"baseline": ":", "replenish": "-"}
    markers = {"baseline": "x", "replenish": "o"}
    y_floor = 1e-5

    for (variant, d), pts in sorted(by_curve.items()):
        pts = sorted(pts, key=lambda x: x.p_loss)
        measured = [p for p in pts if p.logical_errors > 0]
        upper = [p for p in pts if p.logical_errors == 0]
        color = colors.get(d, "k")
        label = f"d={d}, {variant}"
        if measured:
            ax.errorbar(
                [p.p_loss for p in measured], [p.ler for p in measured],
                yerr=[p.ler_se for p in measured],
                color=color, linestyle=styles.get(variant, "-"),
                marker=markers.get(variant, "."), label=label, capsize=3,
            )
        if upper:
            ubs = [wilson_upper_95(0, p.shots) for p in upper]
            # Cap the visible down-arrow at one decade / the y-axis floor.
            downs = [max(ub - max(ub * 0.1, y_floor), 0.0) for ub in ubs]
            ax.errorbar(
                [p.p_loss for p in upper], ubs,
                yerr=[downs, [0] * len(upper)],
                color=color, linestyle="", marker="v", uplims=True,
                label=None if measured else label,
            )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.95, 0.5, " random guess (50%)",
            color="grey", fontsize=8, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(y_floor, 1.0)
    ax.set_xlabel("per-cycle loss probability p_loss")
    ax.set_ylabel("logical error rate")
    ax.set_title(
        "Repetition-code memory experiment over 3d rounds\n"
        "baseline (no replenish) vs loss-aware (teleportation replenish)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--loss-rates", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--max-shots", type=int, default=1_000_000,
                    help="hard upper bound on total shots per (variant, d, p_loss) point.")
    ap.add_argument("--target-errors", type=int, default=20,
                    help="stop a point once this many logical errors have accumulated.")
    ap.add_argument("--shots-per-batch", type=int, default=10_000,
                    help="shots per server subprocess; outer loop runs batches until stopping rule.")
    ap.add_argument("--p", type=float, default=None,
                    help="per-instruction Pauli noise rate; defaults to p_loss/10.")
    ap.add_argument("--rounds-factor", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel server subprocesses; batches are dispatched round-robin "
                         "so the figure refines all curves at the same time.")
    ap.add_argument("--decoder", type=str, default="black-box-relay-bp")
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--workdir", type=str, default=os.path.join(THIS_DIR, "workdir"),
                    help="directory for per-point .deq.jit / .stim / .deq.bin artifacts.")
    ap.add_argument("--out-png", type=str,
                    default=os.path.join(THIS_DIR, "loss_ler_sweep.png"))
    ap.add_argument("--out-json", type=str,
                    default=os.path.join(THIS_DIR, "loss_ler_sweep.json"))
    args = ap.parse_args(argv)

    os.makedirs(args.workdir, exist_ok=True)
    points = sweep(
        distances=args.distances, loss_rates=args.loss_rates,
        max_shots=args.max_shots, target_errors=args.target_errors,
        shots_per_batch=args.shots_per_batch, p=args.p,
        rounds_factor=args.rounds_factor,
        decoder=args.decoder, seed_base=args.seed_base,
        workdir=args.workdir, workers=args.workers,
        out_json=args.out_json, out_png=args.out_png,
    )
    _write_json(points, args.out_json)
    print(f"wrote {args.out_json}")
    plot_sweep(points, args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
