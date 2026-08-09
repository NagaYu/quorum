"""Run the four-condition benchmark and write tidy CSVs to ``results/``.

Three experiments, matching the metrics the project claims:

  main      cost to a correct ranking for A/B/C/D, plus their error rates. A is swept
            over ``n_per_cell`` so we can quote the smallest uniform budget whose error
            rate reaches the nominal level -- otherwise "A is cheap" compares a method
            with a guarantee against one without.
  shift     accuracy as a scaffold shift grows. B is expected to degrade; C and D are
            not, because their estimand is a scaffold-marginalised paired contrast.
  ablation  the Quorum contributions switched on one at a time, plus the inner-truncation
            audit-rate sweep that produces the compute-vs-width trade-off curve.

Usage::

    python -m benchmarks.run --quick        # ~1 minute
    python -m benchmarks.run --reps 30      # full

Claim: low-cost
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from benchmarks.methods import naive_uniform, quorum, sequential_only, subset_irt
from quorum.presets import PresetSpec, QuorumConfig, build_test, build_truth
from quorum.simulator import apply_scaffold_shift, make_truth
from quorum.types import GridSpec

__all__ = ["main", "run_main", "run_shift", "run_ablation"]

logger = logging.getLogger(__name__)

_A_SWEEP = (1, 2, 4, 8, 16)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a list of uniform dicts as CSV.

    Claim: infrastructure
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        logger.warning("no rows to write for %s", path)
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("wrote %s (%d rows)", path, len(rows))


def _grid_for(quick: bool) -> tuple[GridSpec, float, int]:
    """Grid shape, ability gap and rollout cap for quick vs full mode.

    Claim: infrastructure
    """
    if quick:
        return GridSpec(5, 2, 3, 3), 0.55, 30_000
    return GridSpec(6, 3, 3, 4), 0.45, 80_000


def _truth(spec: GridSpec, gap: float, seed: int, *, shift: float = 0.0,
           distortion: float = 0.0) -> Any:
    """Build one ground-truth world, optionally scaffold-shifted.

    Claim: infrastructure
    """
    truth = make_truth(spec, seed=seed, ability_gap=gap, rank_distortion=0.0)
    if shift or distortion:
        truth = apply_scaffold_shift(truth, shift=shift,
                                     rank_distortion=distortion or None, seed=seed + 1)
    return truth


def run_main(reps: int, quick: bool, alpha: float, base_seed: int) -> list[dict[str, Any]]:
    """Experiment 1: cost to a correct ranking, and the error rate that cost buys.

    Claim: low-cost
    """
    spec, gap, cap = _grid_for(quick)
    rows: list[dict[str, Any]] = []
    for rep in range(reps):
        seed = base_seed + rep
        truth = _truth(spec, gap, seed)
        true_order = list(truth.true_order)

        for n_per_cell in _A_SWEEP:
            d = naive_uniform(truth, n_per_cell=n_per_cell, seed=seed)
            rows.append(dict(experiment="main", condition="A_naive", variant=f"n{n_per_cell}",
                             rep=rep, seed=seed, n_rollouts=d.n_rollouts,
                             cost_usd=round(d.cost_usd, 4), cost_sec=round(d.cost_sec, 2),
                             correct=int(d.is_correct(true_order)), resolved=int(d.resolved),
                             steps_executed=d.n_steps_executed, steps_full=d.n_steps_full,
                             cuped_variance_ratio=""))

        d = subset_irt(truth, seed=seed, n_per_anchor=6 if quick else 8)
        rows.append(dict(experiment="main", condition="B_subset_irt", variant="anchor",
                         rep=rep, seed=seed, n_rollouts=d.n_rollouts,
                         cost_usd=round(d.cost_usd, 4), cost_sec=round(d.cost_sec, 2),
                         correct=int(d.is_correct(true_order)), resolved=int(d.resolved),
                         steps_executed=d.n_steps_executed, steps_full=d.n_steps_full,
                         cuped_variance_ratio=""))

        for name, fn in (("C_sequential", sequential_only), ("D_quorum", quorum)):
            d = fn(truth, alpha=alpha, seed=seed, max_rollouts=cap)
            rows.append(dict(experiment="main", condition=name, variant="default",
                             rep=rep, seed=seed, n_rollouts=d.n_rollouts,
                             cost_usd=round(d.cost_usd, 4), cost_sec=round(d.cost_sec, 2),
                             correct=int(d.is_correct(true_order)), resolved=int(d.resolved),
                             steps_executed=d.n_steps_executed, steps_full=d.n_steps_full,
                             cuped_variance_ratio=d.meta.get("cuped_variance_ratio", "")))
        logger.info("main rep %d/%d done", rep + 1, reps)
    return rows


def run_shift(reps: int, quick: bool, alpha: float, base_seed: int) -> list[dict[str, Any]]:
    """Experiment 2: accuracy under a growing scaffold shift.

    The shift moves absolute scores hard while leaving the ranking intact (the
    interaction term stays at zero), so any method whose estimand is the ranking should
    be flat. A method that estimates absolute ability from a frozen anchor set should not.

    Claim: shift-robust
    """
    spec, gap, cap = _grid_for(quick)
    shifts = (0.0, 0.8, 1.6, 2.4) if quick else (0.0, 0.6, 1.2, 1.8, 2.4, 3.0)
    rows: list[dict[str, Any]] = []
    for rep in range(reps):
        seed = base_seed + 500 + rep
        base = _truth(spec, gap, seed)  # the world the anchor set was published on
        for shift in shifts:
            truth = _truth(spec, gap, seed, shift=shift)
            true_order = list(truth.true_order)
            for name, fn, kw in (
                ("A_naive", naive_uniform, dict(n_per_cell=4)),
                ("B_subset_irt", subset_irt,
                 dict(n_per_anchor=6 if quick else 8, anchor_truth=base)),
                ("C_sequential", sequential_only, dict(alpha=alpha, max_rollouts=cap)),
                ("D_quorum", quorum, dict(alpha=alpha, max_rollouts=cap)),
            ):
                d = fn(truth, seed=seed, **kw)
                rows.append(dict(experiment="shift", condition=name, shift=shift,
                                 rep=rep, seed=seed, n_rollouts=d.n_rollouts,
                                 cost_usd=round(d.cost_usd, 4),
                                 correct=int(d.is_correct(true_order)),
                                 resolved=int(d.resolved)))
        logger.info("shift rep %d/%d done", rep + 1, reps)
    return rows


def run_ablation(reps: int, quick: bool, alpha: float, base_seed: int) -> list[dict[str, Any]]:
    """Experiment 3: contribution ablation and the inner-truncation trade-off curve.

    Claim: unbiased-truncation
    """
    spec, gap, cap = _grid_for(quick)
    rows: list[dict[str, Any]] = []
    configs: list[tuple[str, QuorumConfig]] = [
        ("uniform (C)", QuorumConfig(allocator=False, cuped=False)),
        ("+allocator", QuorumConfig(allocator=True, cuped=False)),
        ("+allocator+cuped (D)", QuorumConfig(allocator=True, cuped=True)),
        ("neyman+cuped", QuorumConfig(allocator=True, cuped=True, strategy="neyman")),
        ("thompson+cuped", QuorumConfig(allocator=True, cuped=True, strategy="thompson")),
        ("unpaired, no cuped", QuorumConfig(allocator=True, cuped=False, paired=False)),
        ("unpaired + cuped", QuorumConfig(allocator=True, cuped=True, paired=False)),
    ]
    audit_rates = (0.3, 0.5, 0.7, 0.9)
    for rep in range(reps):
        seed = base_seed + 1000 + rep
        truth = _truth(spec, gap, seed)
        true_order = list(truth.true_order)

        for label, cfg in configs:
            test = build_test(truth, cfg, alpha=alpha, seed=seed, max_rollouts=cap)
            d = test.run()
            ratio = ""
            if test.cuped is not None:
                vals = [test.cuped.variance_ratio(p) for p, ch in test.channels.items()
                        if ch.n > 100]
                if vals:
                    ratio = round(float(np.mean(vals)), 4)
            rows.append(dict(experiment="ablation", config=label, audit_rate="",
                             rep=rep, seed=seed, n_rollouts=d.n_rollouts,
                             cost_usd=round(d.cost_usd, 4), cost_sec=round(d.cost_sec, 2),
                             correct=int(d.is_correct(true_order)), resolved=int(d.resolved),
                             steps_executed=d.n_steps_executed, steps_full=d.n_steps_full,
                             compute_saved=round(d.compute_saved, 4),
                             cuped_variance_ratio=ratio))

        for rho in audit_rates:
            cfg = QuorumConfig(allocator=True, cuped=True, truncation=True, audit_rate=rho)
            test = build_test(truth, cfg, alpha=alpha, seed=seed, max_rollouts=cap)
            d = test.run()
            rows.append(dict(experiment="ablation", config="truncation", audit_rate=rho,
                             rep=rep, seed=seed, n_rollouts=d.n_rollouts,
                             cost_usd=round(d.cost_usd, 4), cost_sec=round(d.cost_sec, 2),
                             correct=int(d.is_correct(true_order)), resolved=int(d.resolved),
                             steps_executed=d.n_steps_executed, steps_full=d.n_steps_full,
                             compute_saved=round(d.compute_saved, 4),
                             cuped_variance_ratio=""))
        logger.info("ablation rep %d/%d done", rep + 1, reps)
    return rows


def run_coverage(reps: int, quick: bool, base_seed: int) -> list[dict[str, Any]]:
    """Experiment 4: empirical error rate on a NULL pair versus the nominal level.

    Two models with identical ability, so ``Delta = 0`` exactly and *any* rejection is an
    error. Run under adaptive allocation and the control variate, because the
    non-adaptive case is the easy one -- the whole reason the estimator is built around
    Horvitz-Thompson weighting is to keep this honest when the design adapts.

    Claim: coverage
    """
    rows: list[dict[str, Any]] = []
    spec = GridSpec(4, 2, 2, 2)
    cap = 600 if quick else 1200
    for alpha in ((0.05, 0.10, 0.20) if not quick else (0.10, 0.20)):
        for label, cfg in (
            ("uniform", QuorumConfig(allocator=False, cuped=False)),
            ("adaptive", QuorumConfig(allocator=True, cuped=False)),
            ("adaptive+cuped", QuorumConfig(allocator=True, cuped=True)),
        ):
            errors = 0
            for rep in range(reps):
                seed = base_seed + 2000 + rep
                truth = make_truth(spec, seed=seed, ability=np.zeros(2))
                test = build_test(truth, cfg, alpha=alpha, seed=seed, max_rollouts=cap)
                test.alpha_split = "none"
                for ch in test.channels.values():
                    ch.alpha = alpha
                ever = False
                while test.n_rollouts < cap:
                    test.step()
                    if any(c.excludes_zero for c in test.channels.values()):
                        ever = True
                        break
                errors += int(ever)
            rows.append(dict(experiment="coverage", config=label, alpha=alpha,
                             reps=reps, errors=errors, error_rate=round(errors / reps, 4)))
            logger.info("coverage %s alpha=%.2f -> %d/%d", label, alpha, errors, reps)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run every experiment and write the CSVs.

    Claim: low-cost
    """
    p = argparse.ArgumentParser(description="Quorum four-condition benchmark")
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true", help="small grid, few reps (~1 min)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default=None,
                   help="comma-separated subset of main,shift,ablation,coverage")
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("quorum").setLevel(logging.ERROR)

    reps = args.reps if args.reps is not None else (3 if args.quick else 30)
    out = Path(args.out)
    wanted = set((args.only or "main,shift,ablation,coverage").split(","))
    t0 = time.time()

    if "main" in wanted:
        _write_csv(out / "main.csv", run_main(reps, args.quick, args.alpha, args.seed))
    if "shift" in wanted:
        _write_csv(out / "shift.csv", run_shift(reps, args.quick, args.alpha, args.seed))
    if "ablation" in wanted:
        _write_csv(out / "ablation.csv", run_ablation(reps, args.quick, args.alpha, args.seed))
    if "coverage" in wanted:
        cov_reps = 120 if args.quick else 300
        _write_csv(out / "coverage.csv", run_coverage(cov_reps, args.quick, args.seed))

    logger.info("benchmark finished in %.1fs; CSVs in %s/", time.time() - t0, out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
