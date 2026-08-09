"""Command-line interface for Quorum.

Output discipline is a design constraint here, not a nicety. When a pair of models is not
separated by the data, this tool says so and stops. It never reports that one model is
better than another on the strength of an interval that contains zero, and it never
attaches a verdict to a real named system: the labels a user passes in are report
captions attached to synthetic indices. Making indistinguishability visible is the whole
point of the tool.

Claim: infrastructure
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from quorum.nested_seq import rollouts_to_resolve
from quorum.presets import PRESETS, QuorumConfig, build_test, build_truth, list_presets
from quorum.types import RankDecision

__all__ = ["main", "parse_budget", "format_decision"]

logger = logging.getLogger(__name__)

_BUDGET_RE = re.compile(
    r"^\s*\$?\s*(?P<num>[0-9]*\.?[0-9]+)\s*(?P<unit>usd|dollars?|\$|s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?)?\s*$",
    re.IGNORECASE,
)

_SECOND_UNITS = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
}


def parse_budget(text: str | None) -> tuple[float | None, float | None]:
    """Parse a budget string into ``(usd, seconds)``, exactly one of which is set.

    Accepts ``"$50"``, ``"50"``, ``"50usd"``, ``"120s"``, ``"30m"``, ``"1h"``. A bare
    number means dollars, because that is the unit the brief's invocation uses.

    Claim: infrastructure
    """
    if text is None:
        return (None, None)
    match = _BUDGET_RE.match(str(text))
    if match is None:
        raise ValueError(
            f"could not parse budget {text!r}; try forms like '$50', '50usd', '120s', '1h'"
        )
    value = float(match.group("num"))
    if value < 0:
        raise ValueError("budget must be non-negative")
    unit = (match.group("unit") or "").lower()
    if unit in _SECOND_UNITS:
        return (None, value * _SECOND_UNITS[unit])
    return (value, None)


def _config_from_args(args: argparse.Namespace) -> QuorumConfig:
    """Build the run configuration from the parsed flags.

    Claim: infrastructure
    """
    return QuorumConfig(
        allocator=not args.no_allocator,
        truncation=bool(getattr(args, "truncation", False)),
        cuped=not args.no_cuped,
        paired=not getattr(args, "unpaired", False),
        audit_rate=float(getattr(args, "audit_rate", 0.3)),
    )


def _load_truth(args: argparse.Namespace) -> Any:
    """Resolve ``--benchmark`` to a ground-truth world.

    ``file:<path>`` replays real logged rollouts through :mod:`quorum.loaders`; every
    other value names a synthetic preset.

    Claim: infrastructure
    """
    name = args.benchmark
    if isinstance(name, str) and name.startswith("file:"):
        from quorum.loaders import load_rollouts_csv, load_rollouts_jsonl, to_frozen_grid

        path = Path(name[len("file:"):]).expanduser()
        if not path.exists():
            raise SystemExit(f"quorum: no such rollout file: {path}")
        loader = load_rollouts_jsonl if path.suffix in (".jsonl", ".ndjson") else load_rollouts_csv
        rollouts, spec, _meta = loader(path)
        return ("replay", rollouts, spec, to_frozen_grid(rollouts, spec, seed=args.seed))
    if name not in PRESETS:
        raise SystemExit(
            f"quorum: unknown benchmark {name!r}. Available: {', '.join(list_presets())}, "
            "or file:<path> to replay logged rollouts."
        )
    return ("synthetic", build_truth(name, seed=args.seed))


def format_decision(
    decision: RankDecision,
    labels: Sequence[str],
    alpha: float,
    *,
    focus: tuple[int, int] | None = None,
) -> str:
    """Render a decision as text that never overstates what the data supports.

    Every unresolved pair is reported as "indistinguishable at this budget", together
    with a clearly-labelled *projection* of how many more rollouts would be needed. The
    projection extrapolates the observed ``1/sqrt(n)`` shrinkage and is not a guarantee.

    Claim: coverage
    """
    def name(i: int) -> str:
        return labels[i] if 0 <= i < len(labels) else f"model-{i}"

    lines: list[str] = []
    lines.append(f"rollouts: {decision.n_rollouts}   spend: ${decision.cost_usd:,.2f}"
                 f"   wall: {decision.cost_sec:,.0f}s")
    if decision.n_steps_full:
        lines.append(f"trajectory steps executed: {decision.n_steps_executed:,} of "
                     f"{decision.n_steps_full:,} ({decision.compute_saved:.1%} saved)")
    lines.append("")

    if focus is not None:
        pairs = [focus]
    else:
        # `intervals` stores both orientations of each pair; report each pair once.
        seen: set[tuple[int, int]] = set()
        pairs = []
        for a, c in sorted(decision.intervals):
            canon = (a, c) if a < c else (c, a)
            if canon not in seen:
                seen.add(canon)
                pairs.append(canon)

    unevaluated: list[tuple[int, int]] = []
    for pair in pairs:
        if pair is None:
            continue
        key = pair if pair in decision.intervals else (pair[1], pair[0])
        if key not in decision.intervals:
            continue
        lo, hi = decision.intervals[key]
        a, c = key
        flip = key != pair
        if flip:
            lo, hi = -hi, -lo
            a, c = pair
        # Delta is a difference of two success *probabilities*, so it lies in [-1, 1] by
        # construction. Intersecting is free information and keeps a channel that was
        # never funded from being displayed as if it had been measured.
        if lo <= -1.0 and hi >= 1.0 and focus is None:
            # A channel still at its initial full width was never funded: the design
            # buys adjacent pairs only. The explicitly requested pair is never treated
            # this way -- there, a full-width interval means "we looked and could not
            # tell", which is a different and important thing to say.
            unevaluated.append((a, c))
            continue
        lo, hi = max(lo, -1.0), min(hi, 1.0)
        width = (hi - lo) / 2.0
        if lo > 0.0 or hi < 0.0:
            lead, trail = (a, c) if lo > 0.0 else (c, a)
            lines.append(
                f"[separated]  {name(lead)} vs {name(trail)}: the interval for the "
                f"success-rate difference excludes 0 "
                f"(Delta in [{lo:+.4f}, {hi:+.4f}], {1-alpha:.0%} anytime-valid)."
            )
            lines.append(
                f"             On this synthetic benchmark the data orders "
                f"{name(lead)} above {name(trail)}."
            )
        else:
            extra = decision.extra_rollouts_needed.get(key)
            if extra is None:
                centre = (lo + hi) / 2.0
                extra = rollouts_to_resolve(
                    max(1, decision.n_rollouts // 2), width, centre
                )
            lines.append(
                f"[indistinguishable at this budget]  {name(a)} vs {name(c)}: "
                f"Delta in [{lo:+.4f}, {hi:+.4f}] contains 0."
            )
            if extra is None or extra > 5_000_000 or not math.isfinite(float(extra)):
                lines.append(
                    "             The data does not support ordering these two. The "
                    "observed difference is too close to zero to project a finishing "
                    "budget from -- if the two are genuinely equal, no budget resolves it."
                )
            else:
                lines.append(
                    f"             The data does not support ordering these two. "
                    f"Projection (not a guarantee): ~{extra:,} more paired rollouts "
                    f"at the current shrinkage rate."
                )
    if unevaluated:
        shown = ", ".join(f"{name(a)}/{name(c)}" for a, c in unevaluated[:6])
        more = "" if len(unevaluated) <= 6 else f" (+{len(unevaluated)-6} more)"
        lines.append(
            f"[not directly compared]  {shown}{more}: no rollouts were spent on these "
            "pairs. The design funds adjacent pairs and gets the rest by transitivity, "
            "so they carry no interval of their own."
        )
    lines.append("")
    if focus is not None:
        # A two-model comparison certifies exactly one ordering. Printing the driver's
        # full model order here would show orderings this run never bought evidence for.
        return "\n".join(lines).rstrip()
    if decision.resolved:
        order = " > ".join(name(i) for i in decision.order)
        lines.append(f"ranking (all adjacent pairs separated): {order}")
    else:
        lines.append(
            "ranking not established at this budget; "
            f"{len(decision.undecided_pairs)} pair(s) remain indistinguishable."
        )
    return "\n".join(lines)


def _decision_to_json(decision: RankDecision, labels: Sequence[str]) -> dict[str, Any]:
    """Machine-readable dump of a decision.

    Claim: infrastructure
    """
    return {
        "resolved": bool(decision.resolved),
        "order": [int(i) for i in decision.order],
        "order_labels": [labels[i] if i < len(labels) else f"model-{i}" for i in decision.order],
        "n_rollouts": int(decision.n_rollouts),
        "cost_usd": float(decision.cost_usd),
        "cost_sec": float(decision.cost_sec),
        "steps_executed": int(decision.n_steps_executed),
        "steps_full": int(decision.n_steps_full),
        "compute_saved": float(decision.compute_saved),
        "intervals": {f"{a}-{c}": [float(lo), float(hi)]
                      for (a, c), (lo, hi) in decision.intervals.items()},
        "undecided_pairs": [[int(a), int(c)] for a, c in decision.undecided_pairs],
        "extra_rollouts_needed": {f"{a}-{c}": int(v)
                                  for (a, c), v in decision.extra_rollouts_needed.items()},
        "note": "Synthetic benchmark. Labels are captions, not claims about real systems.",
    }


def _pair_separated(decision: RankDecision, pair: tuple[int, int]) -> bool:
    """Whether this specific pair's interval excludes zero.

    Claim: coverage
    """
    for key in (pair, (pair[1], pair[0])):
        interval = decision.intervals.get(key)
        if interval is not None:
            lo, hi = interval
            return bool(lo > 0.0 or hi < 0.0)
    return False


def _cmd_compare(args: argparse.Namespace) -> int:
    """`quorum compare A B` -- settle (or decline to settle) one pairwise ordering.

    Claim: low-cost
    """
    usd, sec = parse_budget(args.budget)
    loaded = _load_truth(args)
    labels = [args.model_a, args.model_b]

    if loaded[0] == "replay":
        _, _rollouts, spec, sim = loaded
        from quorum.simulator import make_truth
        truth = make_truth(spec, seed=args.seed)
        test = build_test(truth, _config_from_args(args), alpha=args.alpha, seed=args.seed,
                          budget_usd=usd, budget_sec=sec, max_rollouts=args.max_rollouts,
                          candidate_pairs=[(0, 1)], simulator=sim)
    else:
        truth = loaded[1]
        if truth.spec.n_models < 2:
            raise SystemExit("quorum: benchmark must have at least two models")
        test = build_test(truth, _config_from_args(args), alpha=args.alpha, seed=args.seed,
                          budget_usd=usd, budget_sec=sec, max_rollouts=args.max_rollouts,
                          candidate_pairs=[(0, 1)])

    decision = test.run()
    print(f"quorum compare  --benchmark {args.benchmark}  alpha={args.alpha}"
          f"  budget={args.budget or 'unbounded'}")
    print(f"  (synthetic benchmark; '{args.model_a}' and '{args.model_b}' are labels "
          "for report readability, not measurements of real systems)")
    print()
    print(format_decision(decision, labels, args.alpha, focus=(0, 1)))
    if args.json:
        Path(args.json).write_text(json.dumps(_decision_to_json(decision, labels), indent=2))
        print(f"\nwrote {args.json}")
    # `compare` was asked about ONE pair, so its exit status reports that pair. The
    # decision's `resolved` flag covers the whole model order, which this invocation
    # never bought evidence for -- using it here would report failure on a comparison
    # that actually succeeded.
    return 0 if _pair_separated(decision, (0, 1)) else 2


def _cmd_rank(args: argparse.Namespace) -> int:
    """`quorum rank` -- settle a full ordering over several models.

    Claim: low-cost
    """
    usd, sec = parse_budget(args.budget)
    loaded = _load_truth(args)
    truth = loaded[1] if loaded[0] == "synthetic" else None
    if truth is None:
        raise SystemExit("quorum rank: file: benchmarks are supported via `compare` only")
    labels = [s.strip() for s in args.models.split(",")] if args.models else [
        f"model-{i}" for i in range(truth.spec.n_models)
    ]
    if len(labels) != truth.spec.n_models:
        raise SystemExit(
            f"quorum: benchmark {args.benchmark!r} has {truth.spec.n_models} models but "
            f"{len(labels)} label(s) were given"
        )
    test = build_test(truth, _config_from_args(args), alpha=args.alpha, seed=args.seed,
                      budget_usd=usd, budget_sec=sec, max_rollouts=args.max_rollouts)
    decision = test.run()
    print(f"quorum rank  --benchmark {args.benchmark}  alpha={args.alpha}"
          f"  budget={args.budget or 'unbounded'}")
    print("  (synthetic benchmark; labels are captions, not claims about real systems)")
    print()
    print(format_decision(decision, labels, args.alpha))
    if args.json:
        Path(args.json).write_text(json.dumps(_decision_to_json(decision, labels), indent=2))
        print(f"\nwrote {args.json}")
    return 0 if decision.resolved else 2


def _cmd_simulate(args: argparse.Namespace) -> int:
    """`quorum simulate` -- emit rollout-level records for the Space and the Hub.

    Claim: infrastructure
    """
    from quorum.simulator import RolloutSimulator

    truth = build_truth(args.benchmark, seed=args.seed)
    sim = RolloutSimulator(truth, seed=args.seed)
    rng = np.random.default_rng(args.seed + 5)
    spec = truth.spec
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w") as fh:
        while written < args.n:
            t = int(rng.integers(spec.n_tasks))
            k = int(rng.integers(spec.n_seeds))
            s = int(rng.integers(spec.n_scaffolds))
            shock = sim.draw_shock(rng)
            for m in range(spec.n_models):
                r = sim.rollout(t, k, s, m, rng=rng, shock=shock)
                fh.write(json.dumps({
                    "task": t, "seed": k, "scaffold": s, "model": f"model-{m}",
                    "success": int(r.success), "completed": bool(r.completed),
                    "n_steps": int(r.n_steps), "cost_usd": round(float(r.cost_usd), 6),
                    "cost_sec": round(float(r.cost_sec), 4),
                    "truncated_at": int(r.truncated_at), "audited": bool(r.audited),
                }) + "\n")
                written += 1
                if written >= args.n:
                    break
    print(f"wrote {written} rollout records to {out}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    """`quorum bench` -- run the four-condition benchmark.

    Claim: low-cost
    """
    try:
        from benchmarks.run import main as bench_main
    except ImportError as exc:
        raise SystemExit(
            "quorum bench needs the benchmarks package (and pandas/matplotlib): "
            f"{exc}. Install with `pip install -e '.[bench]'`."
        ) from exc
    argv = ["--reps", str(args.reps), "--out", args.out]
    if args.quick:
        argv.append("--quick")
    return int(bench_main(argv) or 0)


def _cmd_figures(args: argparse.Namespace) -> int:
    """`quorum figures` -- rebuild the figures from existing benchmark CSVs.

    Claim: infrastructure
    """
    results = Path(args.results)
    if not results.exists() or not any(results.glob("*.csv")):
        raise SystemExit(
            f"quorum figures: no CSVs in {results}/. Run `python -m benchmarks.run --quick` "
            "first to produce them."
        )
    try:
        from benchmarks.figures import main as fig_main
    except ImportError as exc:
        raise SystemExit(f"quorum figures needs matplotlib/pandas: {exc}") from exc
    return int(fig_main(["--results", str(results), "--out", args.out]) or 0)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Claim: infrastructure
    """
    p = argparse.ArgumentParser(
        prog="quorum",
        description="Rank-targeted nested sequential design for LLM evaluation. "
                    "Reports which orderings the data supports -- and, just as "
                    "importantly, which it does not.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--benchmark", default="synthetic-easy",
                        help=f"preset ({', '.join(list_presets())}) or file:<path>")
        sp.add_argument("--budget", default=None, help="e.g. $50, 50usd, 120s, 1h")
        sp.add_argument("--alpha", type=float, default=0.05)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--max-rollouts", type=int, default=200_000, dest="max_rollouts")
        sp.add_argument("--json", default=None, help="write a machine-readable dump here")
        sp.add_argument("--no-allocator", action="store_true",
                        help="ablate decision-targeted allocation (uniform design)")
        sp.add_argument("--no-cuped", action="store_true",
                        help="ablate the control variate")
        sp.add_argument("--truncation", action="store_true",
                        help="enable inner truncation (saves steps, widens the interval)")
        sp.add_argument("--audit-rate", type=float, default=0.3, dest="audit_rate")
        sp.add_argument("--unpaired", action="store_true",
                        help="evaluate the two models on independently drawn instances")

    sp = sub.add_parser("compare", help="compare exactly two models")
    sp.add_argument("model_a", metavar="MODEL_A")
    sp.add_argument("model_b", metavar="MODEL_B")
    common(sp)
    sp.set_defaults(func=_cmd_compare)

    sp = sub.add_parser("rank", help="rank several models")
    sp.add_argument("--models", default=None, help="comma-separated labels")
    common(sp)
    sp.set_defaults(func=_cmd_rank)

    sp = sub.add_parser("simulate", help="write rollout-level records")
    sp.add_argument("--out", default="data/rollouts.jsonl")
    sp.add_argument("--n", type=int, default=5000)
    sp.add_argument("--benchmark", default="synthetic-easy")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=_cmd_simulate)

    sp = sub.add_parser("bench", help="run the four-condition benchmark")
    sp.add_argument("--reps", type=int, default=None)
    sp.add_argument("--out", default="results")
    sp.add_argument("--quick", action="store_true")
    sp.set_defaults(func=_cmd_bench)

    sp = sub.add_parser("figures", help="rebuild figures from benchmark CSVs")
    sp.add_argument("--results", default="results")
    sp.add_argument("--out", default="figures")
    sp.set_defaults(func=_cmd_figures)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Exit code 0 when the decision resolved, 2 when the budget ran out.

    Claim: infrastructure
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except SystemExit:
        raise
    except (ValueError, KeyError) as exc:
        print(f"quorum: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
