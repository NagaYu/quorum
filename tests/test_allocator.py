"""Contribution (b): the allocator's invariants and its closed-form information gain.

The exploration floor is not a heuristic here -- it is what bounds the Horvitz-Thompson
weight and therefore what keeps the outer confidence sequence finite-width. It is tested
as a hard invariant, not a preference.

Claim: coverage
"""

from __future__ import annotations

import numpy as np
import pytest

from quorum.allocator import STRATEGIES, DecisionTargetedAllocator, mix_to_floor
from quorum.cell_model import HierarchicalCellModel
from quorum.simulator import RolloutSimulator, make_truth
from quorum.types import GridSpec, uniform_weights

SPEC = GridSpec(4, 2, 2, 3)


@pytest.fixture
def allocator_factory():
    """Build an allocator with a cell model that has seen some data.

    Claim: infrastructure
    """
    def make(strategy: str = "info_cost", pi_min_frac: float = 0.2, seed: int = 0):
        truth = make_truth(SPEC, seed=seed, ability_gap=0.5)
        model = HierarchicalCellModel(SPEC)
        sim = RolloutSimulator(truth, seed=seed)
        rng = np.random.default_rng(seed + 1)
        for t in range(SPEC.n_tasks):
            for s in range(SPEC.n_scaffolds):
                for m in range(SPEC.n_models):
                    r = sim.rollout(t, 0, s, m, rng=rng)
                    model.add(t, 0, s, m, r.success)
        model.fit()
        alloc = DecisionTargetedAllocator(
            SPEC, model, truth.cost_usd, truth.cost_sec,
            w=uniform_weights(SPEC), strategy=strategy, pi_min_frac=pi_min_frac,
            rng=np.random.default_rng(seed + 2),
        )
        return alloc, model, truth
    return make


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_pi_is_a_distribution_respecting_the_floor(allocator_factory, strategy) -> None:
    """``pi`` sums to 1 and never drops below ``pi_min_frac * w`` for any block.

    Claim: coverage
    """
    eps = 0.2
    alloc, _model, _truth = allocator_factory(strategy, pi_min_frac=eps)
    pairs = [(0, 1), (1, 2)]
    for _ in range(25):
        pair, pi, block, pi_b = alloc.select(pairs, None)
        assert pair in pairs
        assert abs(float(pi.sum()) - 1.0) < 1e-8
        assert np.all(pi >= eps * alloc.w - 1e-12), "exploration floor violated"
        assert 0 <= block < SPEC.n_blocks
        assert abs(pi_b - float(pi[block])) < 1e-12


def test_ht_bound_matches_the_realised_weights(allocator_factory) -> None:
    """The declared HT bound must actually dominate every realisable ``w/pi``.

    If it did not, the confidence sequence would be sized against a range its own
    observations can escape.

    Claim: coverage
    """
    eps = 0.2
    alloc, _m, _t = allocator_factory("info_cost", pi_min_frac=eps)
    declared = alloc.ht_bound()
    assert abs(declared - 1.0 / eps) < 1e-9
    worst = 0.0
    for _ in range(40):
        _pair, pi, _block, _pi_b = alloc.select([(0, 1), (1, 2)], None)
        worst = max(worst, float(np.max(alloc.w / pi)))
    assert worst <= declared + 1e-9, f"realised HT weight {worst} exceeds declared {declared}"


def test_mix_to_floor_is_a_valid_mixture() -> None:
    """The floor mixture keeps ``pi`` normalised and bounded away from zero.

    Claim: coverage
    """
    rng = np.random.default_rng(0)
    w = np.full(12, 1 / 12)
    raw = rng.dirichlet(np.full(12, 0.1))
    pi = mix_to_floor(raw, w, 0.25)
    assert abs(pi.sum() - 1.0) < 1e-12
    assert np.all(pi >= 0.25 * w - 1e-12)
    # A degenerate raw distribution must not produce zeros.
    degenerate = np.zeros(12)
    degenerate[3] = 1.0
    pi2 = mix_to_floor(degenerate, w, 0.1)
    assert np.all(pi2 > 0)


def test_info_gain_matches_a_brute_force_rank_one_update(allocator_factory) -> None:
    """The closed-form gain equals the actual drop in contrast variance.

    ``gain = u (v'Sx)^2 / (1 + u x'Sx)`` should equal ``v'Sv - v'S'v`` for the exact
    rank-1 posterior update, which is the whole justification for using it.

    Claim: low-cost
    """
    _alloc, model, _truth = allocator_factory()
    a, c = 0, 2
    Sigma = np.asarray(model.Sigma, dtype=float)
    v = np.asarray(model.contrast_vector(a, c), dtype=float)
    before = float(v @ Sigma @ v)

    for (t, k, s, m) in ((0, 0, 0, a), (1, 0, 1, c), (2, 0, 0, a)):
        x = np.zeros(Sigma.shape[0])
        x[list(model.cell_design_indices(t, k, s, m))] = 1.0
        # Use the model's own Fisher weight, so this checks the *identity* rather than
        # rediscovering which point estimate of p the model evaluates it at.
        u = float(model._fisher_weights_rows()[model._row_of(t, s, m)])
        Sx = Sigma @ x
        after_sigma = Sigma - u * np.outer(Sx, Sx) / (1.0 + u * float(x @ Sx))
        brute = before - float(v @ after_sigma @ v)
        closed = float(model.info_gain_for_cell(t, k, s, m, a, c))
        assert abs(brute - closed) < 1e-8 * max(1.0, abs(brute)), (
            f"cell ({t},{k},{s},{m}): closed form {closed:.6g} vs brute force {brute:.6g}"
        )


def test_vectorised_gain_matches_the_scalar_form(allocator_factory) -> None:
    """``info_gain_all_cells`` must agree with the per-cell computation exactly.

    Claim: low-cost
    """
    _alloc, model, _truth = allocator_factory()
    a, c = 0, 1
    grid = np.asarray(model.info_gain_all_cells(a, c), dtype=float).reshape(SPEC.grid_shape)
    for (t, k, s, m) in ((0, 0, 0, 0), (1, 1, 1, 2), (3, 0, 1, 1)):
        expected = float(model.info_gain_for_cell(t, k, s, m, a, c))
        assert abs(grid[t, k, s, m] - expected) < 1e-9


def test_resolved_pairs_stop_attracting_budget(allocator_factory) -> None:
    """A pair whose channel already excludes zero must lose priority.

    Claim: low-cost
    """
    alloc, _m, _t = allocator_factory()

    class _Channel:
        def __init__(self, done: bool) -> None:
            self.excludes_zero = done
            self.n = 500

    channels = {(0, 1): _Channel(True), (1, 2): _Channel(False)}
    picks = [alloc.select([(0, 1), (1, 2)], channels)[0] for _ in range(60)]
    unresolved = sum(1 for p in picks if p == (1, 2))
    assert unresolved > 45, f"only {unresolved}/60 picks went to the unresolved pair"


def test_allocator_is_deterministic_given_a_seed(allocator_factory) -> None:
    """Same seed, same sequence of decisions.

    Claim: infrastructure
    """
    picks = []
    for _ in range(2):
        alloc, _m, _t = allocator_factory(seed=5)
        picks.append([alloc.select([(0, 1), (1, 2)], None)[2] for _ in range(20)])
    assert picks[0] == picks[1]
