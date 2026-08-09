"""Render the five benchmark figures from the CSVs in ``results/``.

Figures regenerate from the CSVs alone, so they can be rebuilt without re-running the
experiment: ``python -m benchmarks.figures``.

Claim: low-cost
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = ["main", "fig_cost_to_decision", "fig_scaffold_shift", "fig_coverage",
           "fig_variance_reduction", "fig_truncation"]

logger = logging.getLogger(__name__)

PALETTE = {
    "A_naive": "#7f8c9b",
    "B_subset_irt": "#d1743a",
    "C_sequential": "#4a7fb5",
    "D_quorum": "#2e8b6f",
}
LABELS = {
    "A_naive": "(A) Naive uniform",
    "B_subset_irt": "(B) Subset-IRT",
    "C_sequential": "(C) Sequential only",
    "D_quorum": "(D) Quorum",
}
DPI = 150


def _save(fig: plt.Figure, path: Path) -> None:
    """Write a figure and log where it went.

    Claim: infrastructure
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)


def _read(results: Path, name: str) -> pd.DataFrame | None:
    """Read one results CSV, returning ``None`` when it is absent.

    Claim: infrastructure
    """
    path = results / name
    if not path.exists():
        logger.warning("missing %s; skipping the figures that need it", path)
        return None
    return pd.read_csv(path)


def fig_cost_to_decision(results: Path, out: Path) -> bool:
    """Figure 1: what each method pays to reach a correct ranking.

    For (A) we quote the *smallest* uniform sweep whose accuracy over replications
    reaches the level the sequential methods guarantee -- comparing a method that
    controls its error rate against one that merely happened to be right would flatter
    the latter.

    Claim: low-cost
    """
    df = _read(results, "main.csv")
    if df is None:
        return False

    rows = []
    a = df[df.condition == "A_naive"]
    chosen = None
    for variant, grp in sorted(a.groupby("variant"), key=lambda kv: int(kv[0][1:])):
        if grp.correct.mean() >= 0.95:
            chosen = (variant, grp)
            break
    if chosen is None and len(a):
        variant = max(a.variant.unique(), key=lambda v: int(v[1:]))
        chosen = (variant, a[a.variant == variant])
    if chosen is not None:
        variant, grp = chosen
        rows.append(("A_naive", f"(A) Naive uniform\n(smallest sweep at >=95%: {variant})",
                     grp.n_rollouts, grp.cost_usd, grp.correct.mean()))
    for cond in ("B_subset_irt", "C_sequential", "D_quorum"):
        grp = df[df.condition == cond]
        if len(grp):
            rows.append((cond, LABELS[cond], grp.n_rollouts, grp.cost_usd, grp.correct.mean()))
    if not rows:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    x = np.arange(len(rows))
    for ax, idx, title, unit in (
        (axes[0], 2, "Rollouts to a correct ranking", "rollouts"),
        (axes[1], 3, "Spend to a correct ranking", "US$"),
    ):
        vals = [r[idx].mean() for r in rows]
        errs = [r[idx].std() / max(1, np.sqrt(len(r[idx]))) for r in rows]
        colours = [PALETTE[r[0]] for r in rows]
        ax.bar(x, vals, yerr=errs, capsize=4, color=colours, edgecolor="white", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([r[1] for r in rows], fontsize=8, rotation=18, ha="right")
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        for xi, v in zip(x, vals):
            ax.text(xi, v * 1.12, f"{v:,.0f}", ha="center", fontsize=8)

    acc = "   ".join(f"{LABELS[r[0]].split(chr(10))[0]}: {r[4]:.0%}" for r in rows)
    fig.suptitle("Cost to the same conclusion (log scale). Accuracy over replications -- "
                 + acc, fontsize=9, y=1.04)
    fig.text(0.5, -0.16,
             "(A) and (B) return a point estimate with no error guarantee; (C) and (D) "
             "certify the ranking at a family-wise level.\nThe honest comparison is "
             "within those two groups.",
             ha="center", fontsize=8, style="italic")
    _save(fig, out / "fig1_cost_to_decision.png")
    return True


def fig_scaffold_shift(results: Path, out: Path) -> bool:
    """Figure 2: accuracy as a scaffold shift moves absolute scores but not the ranking.

    Claim: shift-robust
    """
    df = _read(results, "shift.csv")
    if df is None:
        return False
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for cond, grp in df.groupby("condition"):
        agg = grp.groupby("shift").correct.agg(["mean", "count"]).reset_index()
        n = agg["count"].clip(lower=1)
        se = np.sqrt(agg["mean"] * (1 - agg["mean"]) / n)
        ax.errorbar(agg.shift_ if hasattr(agg, "shift_") else agg["shift"], agg["mean"],
                    yerr=se, marker="o", capsize=3, linewidth=2,
                    color=PALETTE.get(cond, "#888"), label=LABELS.get(cond, cond))
    ax.set_xlabel("scaffold shift (logit units added to the scaffold main effect)")
    ax.set_ylabel("P(correct ranking)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("A shift that moves absolute scores but preserves the true ranking",
                 fontsize=11)
    fig.text(0.5, -0.09,
             "(B)'s anchor set is fitted once and reused, as published anchor sets are. "
             "Its estimand is absolute ability,\nso the shift invalidates it. (C) and (D) "
             "estimate a scaffold-marginalised paired contrast and are unaffected.",
             ha="center", fontsize=8, style="italic")
    _save(fig, out / "fig2_scaffold_shift.png")
    return True


def fig_coverage(results: Path, out: Path) -> bool:
    """Figure 3: empirical error rate against the nominal level, on a null pair.

    Claim: coverage
    """
    df = _read(results, "coverage.csv")
    if df is None:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    lim = float(max(df.alpha.max() * 1.25, df.error_rate.max() * 1.25, 0.05))
    ax.plot([0, lim], [0, lim], "--", color="#444", linewidth=1, label="nominal (y = x)")
    markers = {"uniform": "o", "adaptive": "s", "adaptive+cuped": "^"}
    for config, grp in df.groupby("config"):
        n = grp.reps.clip(lower=1)
        se = np.sqrt(grp.error_rate * (1 - grp.error_rate) / n)
        ax.errorbar(grp.alpha, grp.error_rate, yerr=2 * se, marker=markers.get(config, "o"),
                    linestyle="none", capsize=3, markersize=7, label=config)
    ax.set_xlabel("nominal alpha")
    ax.set_ylabel("empirical error rate on a null pair")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8)
    ax.set_title("Anytime-valid coverage holds under adaptive allocation", fontsize=11)
    fig.text(0.5, -0.08,
             "Points on or below the dashed line respect the nominal level. The adaptive "
             "rows are the load-bearing ones:\nHorvitz-Thompson weighting is what keeps a "
             "data-dependent design honest. Error bars are 2 s.e.",
             ha="center", fontsize=8, style="italic")
    _save(fig, out / "fig3_coverage.png")
    return True


def fig_variance_reduction(results: Path, out: Path) -> bool:
    """Figure 4: what each contribution buys, and where CUPED does and does not help.

    Claim: variance-reduction
    """
    df = _read(results, "ablation.csv")
    if df is None:
        return False
    df = df[df.config != "truncation"]
    if not len(df):
        return False
    order = ["uniform (C)", "+allocator", "+allocator+cuped (D)", "neyman+cuped",
             "thompson+cuped", "unpaired, no cuped", "unpaired + cuped"]
    present = [c for c in order if c in set(df.config)]
    agg = df.groupby("config").agg(
        n=("n_rollouts", "mean"), n_se=("n_rollouts", "sem"),
        usd=("cost_usd", "mean"), usd_se=("cost_usd", "sem"),
        vr=("cuped_variance_ratio", "mean")).reindex(present)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(present))
    colours = ["#4a7fb5" if "uniform (C)" == c else
               "#b57f4a" if c.startswith("unpaired") else "#2e8b6f" for c in present]
    axes[0].bar(x, agg.usd, yerr=agg.usd_se.fillna(0), capsize=4, color=colours,
                edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(present, fontsize=7.5, rotation=22, ha="right")
    axes[0].set_ylabel("US$ to certify the ranking")
    axes[0].set_title("Ablation: spend by configuration", fontsize=11)
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)

    vr = agg.vr.astype(float)
    mask = vr.notna()
    axes[1].bar(x[mask.values], vr[mask], color="#2e8b6f", edgecolor="white")
    axes[1].axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.2,
                    label="no reduction")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(present, fontsize=7.5, rotation=22, ha="right")
    axes[1].set_ylabel("Var(adjusted) / Var(raw)")
    axes[1].set_ylim(0, 1.25)
    axes[1].legend(fontsize=8)
    axes[1].set_title("CUPED variance ratio (lower is better)", fontsize=11)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)

    fig.text(0.5, -0.14,
             "The control variate does real work only when the design cannot pair the two "
             "models on the same instance.\nUnder exact pairing the shared instance "
             "difficulty has already cancelled, so there is almost nothing left to remove "
             "-- a structural result, not a tuning failure.",
             ha="center", fontsize=8, style="italic")
    _save(fig, out / "fig4_variance_reduction.png")
    return True


def fig_truncation(results: Path, out: Path) -> bool:
    """Figure 5: the inner-truncation trade-off -- steps saved against total spend.

    Claim: unbiased-truncation
    """
    df = _read(results, "ablation.csv")
    if df is None:
        return False
    trunc = df[df.config == "truncation"].copy()
    if not len(trunc):
        return False
    trunc["audit_rate"] = trunc["audit_rate"].astype(float)
    agg = trunc.groupby("audit_rate").agg(
        saved=("compute_saved", "mean"), usd=("cost_usd", "mean"),
        usd_se=("cost_usd", "sem"), n=("n_rollouts", "mean")).reset_index()

    baseline = df[df.config == "+allocator+cuped (D)"]
    base_usd = float(baseline.cost_usd.mean()) if len(baseline) else float("nan")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(agg.audit_rate, agg.saved * 100, marker="o", color="#2e8b6f",
            linewidth=2, label="trajectory steps saved (%)")
    ax.set_xlabel("audit rate rho  (fraction of fired rollouts run to completion)")
    ax.set_ylabel("steps saved (%)", color="#2e8b6f")
    ax.tick_params(axis="y", labelcolor="#2e8b6f")
    ax.grid(alpha=0.25, linewidth=0.6)

    ax2 = ax.twinx()
    ax2.errorbar(agg.audit_rate, agg.usd, yerr=agg.usd_se.fillna(0), marker="s",
                 color="#c0392b", linewidth=2, capsize=3, label="total spend")
    if np.isfinite(base_usd):
        ax2.axhline(base_usd, color="#444", linestyle="--", linewidth=1.2)
        ax2.text(agg.audit_rate.max(), base_usd, "  no truncation", va="bottom",
                 ha="right", fontsize=8, color="#444")
    ax2.set_ylabel("US$ to certify the ranking", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    ax.set_title("Inner truncation: compute saved versus what the correction costs",
                 fontsize=11)
    fig.text(0.5, -0.11,
             "A low audit rate saves the most steps but reweights surviving successes by "
             "1/rho, widening the corrected-outcome\nrange to [-q_max(1-rho)/rho, 1/rho] "
             "-- and an empirical-Bernstein interval pays for range directly. The "
             "estimator stays\nunbiased at every rho; only the price changes.",
             ha="center", fontsize=8, style="italic")
    _save(fig, out / "fig5_truncation.png")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Rebuild every figure that has data behind it.

    Claim: infrastructure
    """
    p = argparse.ArgumentParser(description="Render Quorum benchmark figures")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    args = p.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    results, out = Path(args.results), Path(args.out)
    made = [
        fig_cost_to_decision(results, out),
        fig_scaffold_shift(results, out),
        fig_coverage(results, out),
        fig_variance_reduction(results, out),
        fig_truncation(results, out),
    ]
    logger.info("rendered %d/5 figures into %s/", sum(made), out)
    return 0 if any(made) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
