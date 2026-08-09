"""The hierarchical cell model steers allocation; it never enters the validity argument.

So the bar here is that it is *useful* -- recovers the ability ordering, pools partially,
predicts unvisited blocks -- not that it is exactly right. A wrong model costs efficiency,
which is why the coverage tests do not depend on any of this.

Claim: low-cost
"""

from __future__ import annotations

import numpy as np

from quorum.cell_model import HierarchicalCellModel
from quorum.simulator import RolloutSimulator, make_truth
from quorum.types import GridSpec

SPEC = GridSpec(6, 2, 2, 4)


def _fitted(seed: int = 0, reps: int = 4) -> tuple[HierarchicalCellModel, object]:
    """Fit the model on a uniform sweep of the grid.

    Claim: infrastructure
    """
    truth = make_truth(SPEC, seed=seed, ability_gap=0.6)
    model = HierarchicalCellModel(SPEC)
    sim = RolloutSimulator(truth, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    for t in range(SPEC.n_tasks):
        for k in range(SPEC.n_seeds):
            for s in range(SPEC.n_scaffolds):
                for m in range(SPEC.n_models):
                    for _ in range(reps):
                        model.add(t, k, s, m, sim.rollout(t, k, s, m, rng=rng).success)
    model.fit()
    return model, truth


def test_recovers_the_ability_ordering() -> None:
    """The fitted ability ordering matches the truth on most seeds.

    Claim: low-cost
    """
    hits = 0
    for seed in range(5):
        model, truth = _fitted(seed)
        recovered = list(np.argsort(-np.asarray(model.beta)))
        hits += int(recovered == list(truth.true_order))
    assert hits >= 4, f"ability ordering recovered on only {hits}/5 seeds"


def test_partial_pooling_predicts_unvisited_cells() -> None:
    """A cell with no data still gets a sensible prediction from the pooled structure.

    This is what makes the allocator able to score blocks it has never sampled, which is
    the whole point of using a hierarchical model rather than per-cell means.

    Claim: low-cost
    """
    truth = make_truth(SPEC, seed=0, ability_gap=0.6)
    model = HierarchicalCellModel(SPEC)
    sim = RolloutSimulator(truth, seed=1)
    rng = np.random.default_rng(2)
    # Deliberately leave scaffold 1 of task 0 completely unobserved.
    for t in range(SPEC.n_tasks):
        for s in range(SPEC.n_scaffolds):
            if (t, s) == (0, 1):
                continue
            for m in range(SPEC.n_models):
                for _ in range(6):
                    model.add(t, 0, s, m, sim.rollout(t, 0, s, m, rng=rng).success)
    model.fit()
    p = model.predict_p()
    held_out = p[0, 0, 1, :]
    assert np.all((held_out > 0.0) & (held_out < 1.0))
    # The pooled prediction should still rank the models in ability order there.
    assert list(np.argsort(-held_out)) == list(np.argsort(-np.asarray(model.beta)))


def test_contrast_variance_shrinks_with_data() -> None:
    """More observations must reduce the posterior variance of the decision contrast.

    Claim: low-cost
    """
    sparse, _ = _fitted(seed=3, reps=1)
    dense, _ = _fitted(seed=3, reps=8)
    assert dense.contrast_variance(0, 1) < sparse.contrast_variance(0, 1)


def test_rank_posterior_is_a_probability_matrix() -> None:
    """``rank_posterior`` returns coherent pairwise probabilities.

    Claim: low-cost
    """
    model, _truth = _fitted(seed=0)
    P = np.asarray(model.rank_posterior(n_draws=256, rng=np.random.default_rng(0)))
    assert P.shape == (SPEC.n_models, SPEC.n_models)
    assert np.all((P >= 0) & (P <= 1))
    off = ~np.eye(SPEC.n_models, dtype=bool)
    assert np.allclose((P + P.T)[off], 1.0, atol=0.02)


def test_fit_is_deterministic() -> None:
    """Same data, same fit.

    Claim: infrastructure
    """
    a, _ = _fitted(seed=1)
    b, _ = _fitted(seed=1)
    assert np.allclose(np.asarray(a.beta), np.asarray(b.beta))
