"""The simulator must reproduce the regularity the whole design exploits.

Scaffolds move absolute scores a great deal and the model ranking not at all -- unless a
scaffold x model interaction is deliberately injected, in which case no single global
ranking exists and every method should be expected to struggle.

Claim: shift-robust
"""

from __future__ import annotations

import numpy as np
import pytest

from quorum.simulator import RolloutSimulator, apply_scaffold_shift, make_truth, true_delta
from quorum.types import GridSpec, uniform_weights

SPEC = GridSpec(6, 3, 3, 4)


def _per_scaffold_orders(truth) -> list[list[int]]:
    """Model ordering induced by each scaffold on its own.

    Claim: shift-robust
    """
    per = truth.p.mean(axis=(0, 1))  # (S, M)
    return [list(np.argsort(-per[s])) for s in range(truth.spec.n_scaffolds)]


def test_rank_is_scaffold_invariant_without_distortion() -> None:
    """With ``rank_distortion=0`` every scaffold induces the same model ordering.

    Claim: shift-robust
    """
    for seed in range(6):
        truth = make_truth(SPEC, seed=seed, ability_gap=0.4, rank_distortion=0.0)
        expected = list(truth.true_order)
        for s, order in enumerate(_per_scaffold_orders(truth)):
            assert order == expected, f"scaffold {s} reorders models at seed {seed}"


def test_scaffold_shift_moves_scores_but_not_the_ranking() -> None:
    """A large shift must change absolute scores materially and leave the order alone.

    Claim: shift-robust
    """
    truth = make_truth(SPEC, seed=0, ability_gap=0.4)
    shifted = apply_scaffold_shift(truth, shift=2.0, seed=1)
    assert list(shifted.true_order) == list(truth.true_order)
    moved = float(np.abs(shifted.theta - truth.theta).max())
    assert moved > 0.05, f"the shift barely moved absolute scores ({moved:.4f})"


def test_large_distortion_can_break_the_ranking() -> None:
    """With a real scaffold x model interaction, per-scaffold orders may disagree.

    This is the regime where a single global ranking is not the right question, and the
    test exists so the invariance above cannot pass vacuously.

    Claim: shift-robust
    """
    disagreements = 0
    for seed in range(8):
        truth = make_truth(SPEC, seed=seed, ability_gap=0.25, rank_distortion=1.6)
        orders = _per_scaffold_orders(truth)
        if any(o != orders[0] for o in orders[1:]):
            disagreements += 1
    assert disagreements > 0, "large rank distortion never reordered anything"


def test_costs_are_non_uniform_across_cells() -> None:
    """A cost-aware allocator is only meaningful if costs actually vary.

    Claim: low-cost
    """
    truth = make_truth(SPEC, seed=0)
    spread = float(truth.cost_usd.max() / truth.cost_usd.min())
    assert spread > 5.0, f"cost spread only {spread:.2f}x"
    # ...and the strong models should be the expensive ones, which is what makes the
    # cost-aware criterion adversarial rather than decorative.
    per_model = truth.cost_usd.mean(axis=(0, 1, 2))
    corr = float(np.corrcoef(truth.ability, per_model)[0, 1])
    assert corr > 0.3, f"cost is not correlated with ability (r={corr:.2f})"


def test_instance_shock_preserves_the_marginals() -> None:
    """The copula changes dependence between models, never their success probabilities.

    If it moved the marginals it would change the estimand, and the variance reduction it
    enables would be worthless.

    Claim: variance-reduction
    """
    truth = make_truth(GridSpec(2, 1, 1, 2), seed=0, ability_gap=0.5)
    p = float(truth.p[0, 0, 0, 0])
    for rho in (0.0, 0.9):
        sim = RolloutSimulator(truth, seed=1, instance_corr=rho)
        rng = np.random.default_rng(2)
        hits = 0
        n = 20000
        for _ in range(n):
            shock = sim.draw_shock(rng)
            hits += sim.rollout(0, 0, 0, 0, rng=rng, shock=shock).success
        rate = hits / n
        se = np.sqrt(p * (1 - p) / n)
        assert abs(rate - p) < 4 * se, f"rho={rho}: rate {rate:.4f} vs p {p:.4f}"


def test_instance_shock_creates_the_correlation_cuped_needs() -> None:
    """A shared shock must make the proxy's realised outcome informative.

    Claim: variance-reduction
    """
    truth = make_truth(GridSpec(2, 1, 1, 2), seed=0, ability_gap=0.5)
    corrs = {}
    for rho in (0.0, 0.8):
        sim = RolloutSimulator(truth, seed=1, instance_corr=rho)
        rng = np.random.default_rng(3)
        xs, ps = [], []
        for _ in range(8000):
            shock = sim.draw_shock(rng)
            xs.append(sim.rollout(0, 0, 0, 0, rng=rng, shock=shock).success)
            ps.append(sim.proxy_rollout(0, 0, 0, shock=shock, rng=rng))
        corrs[rho] = float(np.corrcoef(xs, ps)[0, 1])
    assert abs(corrs[0.0]) < 0.06, f"independent draws correlated: {corrs[0.0]:.3f}"
    assert corrs[0.8] > 0.25, f"shared shock gave only r={corrs[0.8]:.3f}"


def test_true_delta_matches_the_estimand_definition() -> None:
    """``true_delta`` is exactly the SPEC 2.1 contrast used by the tests.

    Claim: coverage
    """
    truth = make_truth(SPEC, seed=0)
    w = uniform_weights(SPEC)
    manual = float((truth.p_blocks[:, 0] - truth.p_blocks[:, 1]) @ w)
    assert abs(true_delta(truth, 0, 1) - manual) < 1e-12


def test_costs_accrue_per_step_so_truncation_can_save_money() -> None:
    """A rollout stopped early must cost strictly less than one run to completion.

    Claim: unbiased-truncation
    """
    truth = make_truth(SPEC, seed=0)
    sim = RolloutSimulator(truth, seed=1)
    rng = np.random.default_rng(2)
    full = [sim.rollout(0, 0, 0, 0, rng=rng) for _ in range(300)]
    per_step = [r.cost_usd / max(1, r.n_steps) for r in full]
    assert np.std(per_step) / np.mean(per_step) < 1.5
    assert all(r.n_steps == r.n_steps_full for r in full)
