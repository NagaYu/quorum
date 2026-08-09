"""The four benchmark conditions, run on identical ground truth.

Every condition answers the same question -- what is the ranking of the models? -- and is
scored on whether it got the ranking right and what it paid to get there.

  A  naive_uniform     every cell, N rollouts each, rank by grand mean. No stopping rule.
  B  subset_irt        tinyBenchmarks-style: fit IRT on a pilot, keep an informative
                       anchor subset, estimate ability from the anchors only.
  C  sequential_only   anytime-valid sequential test on a uniform design. The honest
                       "existing method" baseline.
  D  quorum            C plus decision-targeted allocation and a control variate.

B is implemented to *win* in the unshifted regime -- it is the strongest baseline here by
rollout count, and that is the point. Its failure mode is not inefficiency, it is that its
anchor set encodes an assumption (item difficulty is a property of the item) which a
scaffold shift falsifies. A strawman B would prove nothing.

Claim: low-cost
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit

from quorum.presets import QuorumConfig, build_test
from quorum.simulator import GridTruth, RolloutSimulator
from quorum.types import RankDecision, uniform_weights

__all__ = [
    "naive_uniform",
    "subset_irt",
    "sequential_only",
    "quorum",
    "CONDITIONS",
    "run_condition",
]

logger = logging.getLogger(__name__)


def _decision_from_scores(
    scores: np.ndarray,
    sim: RolloutSimulator,
    *,
    resolved: bool = True,
) -> RankDecision:
    """Package a point-estimate ranking with the cost that produced it.

    Fixed-sample methods have no interval to report, so ``intervals`` stays empty and
    ``resolved`` reflects only that the method returned an answer -- not that the answer
    carries a guarantee. The benchmark scores them on whether the answer was *right*,
    which is the fair comparison.

    Claim: low-cost
    """
    order = [int(i) for i in np.argsort(-np.asarray(scores, dtype=float))]
    return RankDecision(
        order=order,
        resolved=bool(resolved),
        n_rollouts=int(sim.n_rollouts),
        cost_usd=float(sim.total_usd),
        cost_sec=float(sim.total_sec),
        n_steps_executed=int(getattr(sim, "total_steps", 0)),
        n_steps_full=int(getattr(sim, "total_steps_full", getattr(sim, "total_steps", 0))),
        meta={"scores": np.asarray(scores, dtype=float).tolist()},
    )


# ---------------------------------------------------------------------------- (A) --


def naive_uniform(
    truth: GridTruth,
    *,
    n_per_cell: int = 4,
    seed: int = 0,
    **_: Any,
) -> RankDecision:
    """(A) Uniform sweep: ``n_per_cell`` rollouts in every cell, rank by grand mean.

    This is what most evaluation harnesses do. It has no stopping rule, so its cost is
    whatever the sweep costs, and its error rate is whatever the sweep buys -- the
    benchmark sweeps ``n_per_cell`` to find the smallest budget at which its error rate
    falls to the nominal level, which is the honest statement of what it costs to be
    right this often.

    Claim: low-cost
    """
    spec = truth.spec
    sim = RolloutSimulator(truth, seed=seed)
    rng = np.random.default_rng(seed + 1)
    totals = np.zeros(spec.n_models)
    counts = np.zeros(spec.n_models)
    for t in range(spec.n_tasks):
        for k in range(spec.n_seeds):
            for s in range(spec.n_scaffolds):
                for _ in range(int(n_per_cell)):
                    shock = sim.draw_shock(rng)
                    for m in range(spec.n_models):
                        r = sim.rollout(t, k, s, m, rng=rng, shock=shock)
                        totals[m] += r.success
                        counts[m] += 1
    return _decision_from_scores(totals / np.maximum(counts, 1), sim)


# ---------------------------------------------------------------------------- (B) --


def _fit_2pl(
    successes: np.ndarray,
    trials: np.ndarray,
    *,
    n_iter: int = 200,
    lr: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MAP fit of a 2PL IRT model ``p[i, m] = sigmoid(disc[i] * (ability[m] - diff[i]))``.

    Plain projected gradient ascent on the penalised likelihood -- adequate and stable at
    this scale, and deliberately not something exotic, because condition B has to be a
    *fair* representative of the tinyBenchmarks family.

    Claim: shift-robust
    """
    n_items, n_models = successes.shape
    ability = np.zeros(n_models)
    diff = np.zeros(n_items)
    disc = np.ones(n_items)
    for _ in range(n_iter):
        z = disc[:, None] * (ability[None, :] - diff[:, None])
        p = expit(z)
        resid = successes - trials * p
        g_ability = (disc[:, None] * resid).sum(axis=0) - 0.1 * ability
        g_diff = -(disc * resid.sum(axis=1)) - 0.1 * diff
        g_disc = ((ability[None, :] - diff[:, None]) * resid).sum(axis=1) - 0.5 * (disc - 1.0)
        scale = max(1.0, float(trials.sum()) / max(1, n_items))
        ability += lr * g_ability / scale
        diff += lr * g_diff / scale
        disc = np.clip(disc + lr * g_disc / scale, 0.2, 4.0)
        ability -= ability.mean()
    return ability, diff, disc


def subset_irt(
    truth: GridTruth,
    *,
    n_pilot: int = 1,
    n_anchor: int | None = None,
    n_per_anchor: int = 6,
    seed: int = 0,
    anchor_truth: GridTruth | None = None,
    **_: Any,
) -> RankDecision:
    """(B) tinyBenchmarks-style anchor-subset estimation.

    Pilot every item once, fit a 2PL, choose the anchor subset that maximises Fisher
    information at the current ability estimates (with a difficulty-spread term so the
    anchors are not all clustered), then spend the remaining budget on the anchors only
    and rank by the ability estimated from them.

    The model has **no scaffold term**. That is not an oversight -- it is the assumption
    the method family actually makes: that an item's difficulty is a property of the item.

    ``anchor_truth`` is what makes metric (3) a fair test. Passing it selects the anchors
    from a pilot on *that* world and then spends the evaluation budget on ``truth``,
    which is how these methods are really used: an anchor set is fitted once, published,
    and reused after the harness has moved on. Re-piloting on the shifted world every
    time would repair the anchors and hide the failure mode entirely -- and would also
    throw away the sample-efficiency that is the method's entire reason for existing.
    With ``anchor_truth=None`` the anchors are fitted in-place, which is the optimistic
    case and the one where B is at its strongest.

    Claim: shift-robust
    """
    spec = truth.spec
    sim = RolloutSimulator(truth, seed=seed)
    rng = np.random.default_rng(seed + 2)
    pilot_sim = RolloutSimulator(anchor_truth, seed=seed + 91) if anchor_truth is not None else sim
    pilot_rng = np.random.default_rng(seed + 92) if anchor_truth is not None else rng
    n_items = spec.n_blocks
    if n_anchor is None:
        n_anchor = max(4, n_items // 4)
    n_anchor = int(min(n_anchor, n_items))

    successes = np.zeros((n_items, spec.n_models))
    trials = np.zeros((n_items, spec.n_models))

    def run_item(item: int, reps: int, *, pilot: bool = False) -> None:
        t, k, s = np.unravel_index(item, spec.block_shape)
        use_sim, use_rng = (pilot_sim, pilot_rng) if pilot else (sim, rng)
        for _ in range(reps):
            shock = use_sim.draw_shock(use_rng)
            for m in range(spec.n_models):
                r = use_sim.rollout(int(t), int(k), int(s), m, rng=use_rng, shock=shock)
                if pilot:
                    pilot_succ[item, m] += r.success
                    pilot_trial[item, m] += 1
                else:
                    successes[item, m] += r.success
                    trials[item, m] += 1

    pilot_succ = np.zeros((n_items, spec.n_models))
    pilot_trial = np.zeros((n_items, spec.n_models))
    for item in range(n_items):
        run_item(item, int(n_pilot), pilot=True)

    ability, diff, disc = _fit_2pl(pilot_succ, np.maximum(pilot_trial, 1))
    if anchor_truth is None:
        # In-place piloting: the pilot rollouts were bought on the evaluation world, so
        # they count towards the estimate as well as towards the bill.
        successes += pilot_succ
        trials += pilot_trial

    # Fisher information of item i at the current abilities, plus a spread bonus so the
    # anchors cover the difficulty range rather than piling up at the median.
    p = expit(disc[:, None] * (ability[None, :] - diff[:, None]))
    info = (disc[:, None] ** 2 * p * (1 - p)).sum(axis=1)
    spread = np.abs(diff - np.median(diff))
    score = info + 0.15 * spread / (spread.std() + 1e-9)
    anchors = np.argsort(-score)[:n_anchor]

    for item in anchors:
        run_item(int(item), int(n_per_anchor))

    a_succ = successes[anchors]
    a_trial = np.maximum(trials[anchors], 1)
    ability_hat, _, _ = _fit_2pl(a_succ, a_trial)
    return _decision_from_scores(ability_hat, sim)


# ------------------------------------------------------------------------ (C), (D) --


def sequential_only(
    truth: GridTruth,
    *,
    alpha: float = 0.05,
    seed: int = 0,
    max_rollouts: int = 200_000,
    budget_usd: float | None = None,
    **_: Any,
) -> RankDecision:
    """(C) Anytime-valid sequential test on a uniform design -- no Quorum contributions.

    Same code path as (D) with allocation, truncation and the control variate switched
    off, so the ablation is exact rather than approximate.

    Claim: low-cost
    """
    cfg = QuorumConfig(allocator=False, truncation=False, cuped=False)
    test = build_test(truth, cfg, alpha=alpha, seed=seed, max_rollouts=max_rollouts,
                      budget_usd=budget_usd)
    return test.run()


def quorum(
    truth: GridTruth,
    *,
    alpha: float = 0.05,
    seed: int = 0,
    max_rollouts: int = 200_000,
    budget_usd: float | None = None,
    truncation: bool = False,
    paired: bool = True,
    **_: Any,
) -> RankDecision:
    """(D) Quorum: decision-targeted allocation plus a known-mean control variate.

    ``truncation`` defaults to off; see :class:`quorum.presets.QuorumConfig` for the
    measured reason (it saves trajectory steps but widens the confidence sequence enough
    to cost more overall on this simulator).

    Claim: low-cost
    """
    cfg = QuorumConfig(allocator=True, truncation=truncation, cuped=True, paired=paired)
    test = build_test(truth, cfg, alpha=alpha, seed=seed, max_rollouts=max_rollouts,
                      budget_usd=budget_usd)
    decision = test.run()
    ratios = []
    if test.cuped is not None:
        for pair, ch in test.channels.items():
            if ch.n > 100:
                try:
                    ratios.append(float(test.cuped.variance_ratio(pair)))
                except Exception:  # pragma: no cover - defensive
                    pass
    decision.meta["cuped_variance_ratio"] = float(np.mean(ratios)) if ratios else float("nan")
    decision.meta["n_truncated"] = int(getattr(test, "n_truncated", 0))
    return decision


CONDITIONS: dict[str, Any] = {
    "A_naive": naive_uniform,
    "B_subset_irt": subset_irt,
    "C_sequential": sequential_only,
    "D_quorum": quorum,
}


def run_condition(name: str, truth: GridTruth, **kwargs: Any) -> RankDecision:
    """Dispatch to a named condition.

    Claim: low-cost
    """
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; have {sorted(CONDITIONS)}")
    return CONDITIONS[name](truth, **kwargs)
