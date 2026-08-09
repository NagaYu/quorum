"""Quorum leaderboard viewer -- a Gradio Space that shows what a score table cannot.

A leaderboard column of numbers invites the reader to order them. This app recomputes
anytime-valid confidence sequences from rollout-level data and marks, in red, every pair
the data does not actually separate -- together with a projection of how many more
rollouts would settle it.

It runs entirely offline against a bundled demo file. The demo models are anonymous
("model-A", ...) and synthetic; nothing here measures or ranks a real system, and the UI
says so.

Claim: coverage
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd

from quorum.nested_seq import EmpiricalBernsteinCS, rollouts_to_resolve

logger = logging.getLogger(__name__)

DEMO_CSV = Path(__file__).parent / "data" / "demo_leaderboard.csv"

NORM_NOTE = (
    "This tool reports which orderings a dataset supports. Pairs shown in red are "
    "**not statistically distinguishable** at the chosen level -- the correct reading is "
    "\"this data cannot tell them apart\", not \"they are equal\" and certainly not that "
    "either one is deficient. The bundled demo is synthetic and its model labels are "
    "placeholders."
)


def _ensure_demo() -> Path:
    """Generate the bundled demo CSV on first use, without touching the network.

    Claim: infrastructure
    """
    if DEMO_CSV.exists():
        return DEMO_CSV
    from quorum.simulator import RolloutSimulator, make_truth
    from quorum.types import GridSpec

    spec = GridSpec(20, 4, 3, 5)
    # Tuned so the demo shows *both* outcomes at the default alpha: about half the
    # pairs separate and half do not. A demo where nothing ever resolves teaches only
    # half the lesson, and one where everything resolves teaches the wrong one.
    # The binding constraint is the number of blocks, not rollouts per block: the
    # confidence sequence runs over block-level differences, so adding repeats inside a
    # block barely moves it while adding tasks does.
    truth = make_truth(spec, seed=7, ability_gap=0.42)
    sim = RolloutSimulator(truth, seed=7)
    rng = np.random.default_rng(7)
    rows = []
    for t in range(spec.n_tasks):
        for k in range(spec.n_seeds):
            for s in range(spec.n_scaffolds):
                for _ in range(6):
                    shock = sim.draw_shock(rng)
                    for m in range(spec.n_models):
                        r = sim.rollout(t, k, s, m, rng=rng, shock=shock)
                        rows.append({
                            "task": t, "seed": k, "scaffold": s,
                            "model": f"model-{chr(ord('A') + m)}",
                            "success": int(r.success),
                            "cost_usd": round(float(r.cost_usd), 6),
                            "cost_sec": round(float(r.cost_sec), 3),
                        })
    DEMO_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DEMO_CSV, index=False)
    logger.info("generated demo leaderboard at %s", DEMO_CSV)
    return DEMO_CSV


def _load(file_obj: Any) -> pd.DataFrame:
    """Load rollout-level data from an upload, falling back to the bundled demo.

    Claim: infrastructure
    """
    if file_obj is None:
        return pd.read_csv(_ensure_demo())
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path:
        return pd.read_csv(_ensure_demo())
    return pd.read_csv(path)


def _paired_matrix(df: pd.DataFrame) -> tuple[list[str], dict[tuple[str, str], np.ndarray]]:
    """Build per-pair paired difference streams, blocking on (task, seed, scaffold).

    Pairing on the shared block is what makes the contrast scaffold-marginalised, which
    is what keeps the comparison stable when a harness changes its scaffolding.

    Claim: shift-robust
    """
    key = [c for c in ("task", "seed", "scaffold") if c in df.columns]
    models = sorted(df["model"].unique())
    wide = (df.groupby(key + ["model"])["success"].mean().unstack("model"))
    streams: dict[tuple[str, str], np.ndarray] = {}
    for i, a in enumerate(models):
        for c in models[i + 1:]:
            if a in wide.columns and c in wide.columns:
                d = (wide[a] - wide[c]).dropna().to_numpy(dtype=float)
                if d.size:
                    streams[(a, c)] = d
    return models, streams


def analyse(file_obj: Any, alpha: float) -> tuple[pd.DataFrame, str, str]:
    """Recompute scores, intervals and the pairwise separation matrix.

    Claim: coverage
    """
    try:
        df = _load(file_obj)
    except Exception as exc:  # pragma: no cover - user upload
        return pd.DataFrame(), f"Could not read that file: {exc}", ""
    missing = {"model", "success"} - set(df.columns)
    if missing:
        return pd.DataFrame(), f"Missing required column(s): {sorted(missing)}", ""

    models, streams = _paired_matrix(df)
    n_pairs = max(1, len(models) * (len(models) - 1) // 2)
    alpha_pair = alpha / n_pairs  # Bonferroni over the family of candidate pairs

    summary = (df.groupby("model")
                 .agg(score=("success", "mean"), rollouts=("success", "size"))
                 .reset_index()
                 .sort_values("score", ascending=False))

    results: dict[tuple[str, str], dict[str, Any]] = {}
    for (a, c), stream in streams.items():
        cs = EmpiricalBernsteinCS(alpha_pair, -1.0, 1.0)
        lo, hi = cs.update(stream)
        half = (hi - lo) / 2.0
        delta_hat = float(stream.mean())
        separated = bool(lo > 0.0 or hi < 0.0)
        extra = 0 if separated else rollouts_to_resolve(stream.size, half, delta_hat)
        results[(a, c)] = {"lo": lo, "hi": hi, "sep": separated, "extra": extra,
                           "n": int(stream.size)}

    rows = []
    for _, r in summary.iterrows():
        model = r["model"]
        widths = [v for (a, c), v in results.items() if model in (a, c)]
        rows.append({
            "model": model,
            "observed score": round(float(r["score"]), 4),
            "rollouts": int(r["rollouts"]),
            "pairs not separated": sum(1 for v in widths if not v["sep"]),
        })
    table = pd.DataFrame(rows)

    order = list(summary["model"])
    head = "<table style='border-collapse:collapse;font-size:13px'><tr><th></th>"
    head += "".join(f"<th style='padding:6px'>{m}</th>" for m in order) + "</tr>"
    body = ""
    for a in order:
        body += f"<tr><th style='padding:6px;text-align:left'>{a}</th>"
        for c in order:
            if a == c:
                body += "<td style='padding:6px;text-align:center;opacity:.35'>-</td>"
                continue
            res = results.get((a, c)) or results.get((c, a))
            if res is None:
                body += "<td style='padding:6px;text-align:center;opacity:.35'>n/a</td>"
                continue
            if res["sep"]:
                body += ("<td style='padding:6px;text-align:center;background:#e8f5ec;"
                         f"color:#1d6b45'>separated<br><small>[{res['lo']:+.3f}, "
                         f"{res['hi']:+.3f}]</small></td>")
            else:
                extra = res["extra"]
                extra_txt = (f"+~{extra:,} rollouts" if 0 < extra <= 5_000_000
                             else "not projectable")
                body += ("<td style='padding:6px;text-align:center;background:#fdeaea;"
                         f"color:#a32222'><b>not distinguishable</b><br>"
                         f"<small>[{res['lo']:+.3f}, {res['hi']:+.3f}]<br>"
                         f"{extra_txt}</small></td>")
        body += "</tr>"
    matrix = head + body + "</table>"

    n_undecided = sum(1 for v in results.values() if not v["sep"])
    note = (
        f"**{n_undecided} of {len(results)} pairs are not distinguishable** at "
        f"alpha = {alpha:g} (Bonferroni-split to {alpha_pair:.4g} per pair, "
        "anytime-valid empirical-Bernstein). Red cells show the projected additional "
        "paired rollouts needed -- a projection from the observed shrinkage rate, "
        "**not a guarantee**."
    )
    return table, note, matrix


def build_interface() -> gr.Blocks:
    """Construct the Gradio app without launching it.

    Kept separate from ``launch`` so importing this module is side-effect free and the
    tests can build the interface offline.

    Claim: infrastructure
    """
    with gr.Blocks(title="Quorum -- which leaderboard gaps are real?") as demo:
        gr.Markdown("# Quorum\n### Which leaderboard gaps does the data actually support?")
        gr.Markdown(NORM_NOTE)
        with gr.Row():
            upload = gr.File(label="Rollout-level CSV (task, seed, scaffold, model, success)",
                             file_types=[".csv"])
            alpha = gr.Slider(0.01, 0.25, value=0.05, step=0.01,
                              label="alpha (family-wise)")
        run = gr.Button("Recompute with confidence sequences", variant="primary")
        note = gr.Markdown()
        table = gr.Dataframe(label="Models", interactive=False)
        matrix = gr.HTML(label="Pairwise separation")
        gr.Markdown(
            "Red means the interval for the paired success-rate difference contains "
            "zero. The estimand is a **scaffold-marginalised paired contrast**, so it is "
            "unaffected by a scaffold change that moves every model's absolute score."
        )
        run.click(analyse, inputs=[upload, alpha], outputs=[table, note, matrix])
        demo.load(analyse, inputs=[upload, alpha], outputs=[table, note, matrix])
    return demo


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    build_interface().launch()
