"""Ground-truth generator and rollout simulator for the Quorum benchmark.

This module manufactures the *world* that every Quorum method is scored against. It has
three jobs, and each one exists to make a specific claim testable:

1. :func:`make_truth` builds a `(task, seed, scaffold, model)` grid of true success
   probabilities from the additive latent predictor of SPEC 4.1. The only term that can
   reorder models across scaffolds is ``delta[s, m]``; with ``rank_distortion=0.0`` the
   induced model ranking is *identical for every scaffold and every scaffold subset*.
   That empirical regularity -- scaffolds move absolute scores a lot and the ranking not
   at all -- is the thing the whole rank-targeted design exploits.
2. Costs are deliberately **non-uniform across cells** and positively correlated with
   both model strength and scaffold index, so the most informative cells tend to be the
   most expensive ones. A cost-aware allocator is only interesting in a world like that.
3. :class:`RolloutSimulator` emits trajectories whose running mean separates eventual
   successes from eventual failures *progressively*, and charges money and seconds
   **per executed step**. That is what makes inner truncation genuinely save compute --
   and what makes the audit correction of SPEC 5.4 necessary rather than decorative.

The simulator never imports :mod:`quorum.nested_seq`: the inner truncation rule is
consumed structurally (``should_stop`` / ``audit`` / ``q``), which keeps the dependency
edge one-way and lets real-data replay in :mod:`quorum.loaders` reuse the same protocol.

Claim: infrastructure
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.special import expit, ndtr as _std_normal_cdf

from quorum.types import GridSpec, Rollout, uniform_weights

__all__ = [
    "GridTruth",
    "RolloutSimulator",
    "Truncator",
    "apply_scaffold_shift",
    "default_ability",
    "make_truth",
    "true_delta",
]

logger = logging.getLogger(__name__)

# Cost-structure constants. These are *not* free knobs of the statistical procedure --
# they only shape the economics of the grid, and they are fixed so that every benchmark
# replication faces the same adversarial correlation between value and price.
_COST_ABILITY_SLOPE: float = 1.2
"""Log-cost spread between the weakest and the strongest model (stronger => pricier)."""

_COST_SCAFFOLD_SLOPE: float = 1.0
"""Log-cost spread between the cheapest and the most elaborate scaffold."""

_COST_RHO: float = 0.7
"""Correlation between the dollar and second lognormal shocks of the same cell."""

_CROSSING_SAFETY: float = 0.9
"""Fraction of the ability gap that ``zeta`` is allowed to eat (keeps block ranks strict)."""


@runtime_checkable
class Truncator(Protocol):
    """Structural type of the inner truncation rule consulted during a rollout.

    Deliberately a :class:`typing.Protocol` rather than an import of
    ``quorum.nested_seq.InnerTruncator``: the simulator must not depend on the estimator
    package (that would be a cycle), and real-data replay in :mod:`quorum.loaders` wants
    the same duck-typed contract. Any object with these three methods works.

    ``should_stop`` reads a trajectory prefix, ``audit`` flips the *independent* audit
    coin at the moment the rule fires, and ``q`` reports the calibrated
    ``P(success | fired)`` used by the outer correction (the simulator never reads ``q``
    -- it belongs to the estimator -- but it is part of the contract).

    Claim: unbiased-truncation
    """

    def should_stop(self, traj_prefix: np.ndarray, step: int) -> bool:
        """Whether to abandon the rollout after observing ``traj_prefix``.

        Claim: unbiased-truncation
        """
        ...

    def audit(self, rng: np.random.Generator) -> bool:
        """Flip the audit coin, independently of the (unseen) outcome.

        Claim: unbiased-truncation
        """
        ...

    def q(self, traj_prefix: np.ndarray, step: int) -> float:
        """Calibrated ``P(success | rule fired here)``; may be arbitrarily wrong.

        Claim: unbiased-truncation
        """
        ...


def default_ability(n_models: int, ability_gap: float) -> np.ndarray:
    """Evenly spaced latent abilities, centred at zero, adjacent gap ``ability_gap``.

    Model ``0`` is the strongest, so the default ``true_order`` is ``[0, 1, ..., M-1]``.
    ``ability_gap`` is the hardness knob for the entire benchmark: it is the logit-scale
    separation the ranking procedure has to certify, so halving it roughly quadruples
    the number of rollouts any honest method needs.

    Claim: low-cost
    """
    if n_models < 1:
        raise ValueError("n_models must be >= 1")
    centred = (np.arange(n_models, dtype=float)[::-1] - (n_models - 1) / 2.0)
    return ability_gap * centred


@dataclass
class GridTruth:
    """Ground truth for one benchmark world: probabilities, ranking, costs, proxy.

    ``p`` has shape ``(T, K, S, M)``; ``theta[m] = sum_b w[b] p[b, m]`` under uniform
    ``w`` and ``true_order`` is ``argsort(-theta)``. Because the latent predictor is
    additive and monotone in ability *at fixed block* (see :func:`make_truth`), the
    ordering of models is the same in every block whenever ``scaffold_model`` is zero --
    hence the same for every scaffold subset, which is exactly the invariance SPEC 2.1
    relies on.

    The fields after ``proxy_p`` are the latent components; they are retained so that
    :func:`apply_scaffold_shift` can rebuild a *coherent* alternative world (same tasks,
    same seeds, same models, moved scaffolds) instead of resampling an unrelated one.

    Claim: shift-robust
    """

    spec: GridSpec
    p: np.ndarray
    theta: np.ndarray
    true_order: np.ndarray
    ability: np.ndarray
    difficulty: np.ndarray
    scaffold_effect: np.ndarray
    cost_usd: np.ndarray
    cost_sec: np.ndarray
    proxy_p: np.ndarray
    mu: float = 0.0
    seed_jitter: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    task_model: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    scaffold_model: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    beta_proxy: float = -0.6
    rank_distortion: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def p_blocks(self) -> np.ndarray:
        """``p`` reshaped to ``(B, M)`` with the row-major block flattening of SPEC 1.

        Claim: infrastructure
        """
        return self.p.reshape(self.spec.n_blocks, self.spec.n_models)

    @property
    def proxy_blocks(self) -> np.ndarray:
        """``proxy_p`` reshaped to ``(B,)`` with the same block flattening.

        Claim: variance-reduction
        """
        return self.proxy_p.reshape(self.spec.n_blocks)

    def theta_for(
        self,
        scaffolds: Sequence[int] | int | None = None,
        w: np.ndarray | None = None,
    ) -> np.ndarray:
        """``w``-weighted mean success per model, restricted to a scaffold subset.

        Restricting to a subset renormalises ``w`` over the surviving blocks, so the
        result is the estimand a practitioner would target if they had only run those
        scaffolds. Comparing orders across subsets is how the scaffold-invariance claim
        is checked.

        Claim: shift-robust
        """
        w_full = uniform_weights(self.spec) if w is None else np.asarray(w, dtype=float)
        if w_full.shape != (self.spec.n_blocks,):
            raise ValueError(
                f"w must have shape ({self.spec.n_blocks},), got {w_full.shape}"
            )
        mask = np.zeros(self.spec.block_shape, dtype=bool)
        if scaffolds is None:
            mask[:] = True
        else:
            idx = [scaffolds] if isinstance(scaffolds, (int, np.integer)) else list(scaffolds)
            mask[:, :, list(idx)] = True
        wm = w_full * mask.reshape(-1)
        total = float(wm.sum())
        if total <= 0.0:
            raise ValueError("scaffold subset has zero total weight")
        return (wm / total) @ self.p_blocks

    def order_for(
        self,
        scaffolds: Sequence[int] | int | None = None,
        w: np.ndarray | None = None,
    ) -> np.ndarray:
        """Model order (best first) induced by a scaffold subset.

        Claim: shift-robust
        """
        return np.argsort(-self.theta_for(scaffolds, w), kind="stable")

    def scaffold_orders(self, w: np.ndarray | None = None) -> np.ndarray:
        """``(S, M)`` array whose row ``s`` is the order induced by scaffold ``s`` alone.

        With ``rank_distortion == 0`` every row equals :attr:`true_order`; this is the
        regularity the rank-targeted estimand exploits, and it is asserted in the tests.

        Claim: shift-robust
        """
        return np.stack(
            [self.order_for(s, w) for s in range(self.spec.n_scaffolds)], axis=0
        )

    def proxy_mean(self, w: np.ndarray | None = None) -> float:
        """Closed-form ``w``-weighted mean of the proxy covariate over the whole grid.

        This number is what makes the proxy a *known-mean* control variate: CUPED may
        subtract ``phi * (C - E[C])`` without estimating ``E[C]`` from the same data,
        which would have quietly biased the confidence sequence.

        Claim: variance-reduction
        """
        w_full = uniform_weights(self.spec) if w is None else np.asarray(w, dtype=float)
        return float(w_full @ self.proxy_blocks)

    def rank_is_scaffold_invariant(self, w: np.ndarray | None = None) -> bool:
        """Whether every single-scaffold order equals the global :attr:`true_order`.

        Claim: shift-robust
        """
        orders = self.scaffold_orders(w)
        return bool(np.all(orders == np.asarray(self.true_order)[None, :]))


def _shrink_rank_crossing(ability: np.ndarray, zeta: np.ndarray) -> float:
    """Largest scale on ``zeta``'s model-contrast part that keeps block ranks intact.

    ``zeta[t, m]`` is a task x model interaction, so it can flip two models *within a
    task*. If it does, the model ordering of the block-averaged probabilities can depend
    on which scaffolds are averaged over (the logistic link re-weights tasks differently
    at different scaffold levels), and ``delta[s, m]`` would no longer be the only
    rank-distorting term. Scaling the model-contrast part of ``zeta`` by this factor
    restores "monotone in ``beta[m]`` at fixed block" exactly as SPEC 4.1 asserts.

    Exactly tied abilities are skipped: tied models are made exchangeable elsewhere, and
    a tied pair has no rank to preserve.

    Claim: shift-robust
    """
    n_models = ability.shape[0]
    if n_models < 2 or zeta.size == 0:
        return 1.0
    order = np.argsort(-ability, kind="stable")
    scale = 1.0
    for i in range(n_models - 1):
        hi, lo = int(order[i]), int(order[i + 1])
        gap = float(ability[hi] - ability[lo])
        if gap <= 0.0:
            continue
        d = zeta[:, hi] - zeta[:, lo]
        worst = float(-np.min(d))
        if worst > 0.0:
            scale = min(scale, _CROSSING_SAFETY * gap / worst)
    return float(max(scale, 0.0))


def _tie_groups(ability: np.ndarray) -> list[list[int]]:
    """Groups of model indices sharing *exactly* the same ability.

    Claim: coverage
    """
    groups: dict[float, list[int]] = {}
    for m, value in enumerate(np.asarray(ability, dtype=float)):
        groups.setdefault(float(value), []).append(int(m))
    return [g for g in groups.values() if len(g) > 1]


def _make_exchangeable(arrays: Sequence[np.ndarray], groups: Sequence[Sequence[int]]) -> None:
    """Copy each tied group's first column over its siblings, in place.

    Two models with identical ability must be *exactly* exchangeable, otherwise the
    "null pair" used by the coverage test would have ``Delta != 0`` and the nominal
    error rate would not be the quantity being measured.

    Claim: coverage
    """
    for group in groups:
        head = group[0]
        for other in group[1:]:
            for arr in arrays:
                if arr.size:
                    arr[:, other] = arr[:, head]


def _cost_multipliers(ability: np.ndarray, n_scaffolds: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-model and per-scaffold cost multipliers (stronger and later => pricier).

    The correlation is adversarial on purpose: the cells that carry the most information
    about the decision are also the ones that drain the budget fastest, which is the
    only regime in which "information per dollar" beats "information".

    Claim: low-cost
    """
    ability = np.asarray(ability, dtype=float)
    spread = float(np.ptp(ability))
    if spread <= 0.0:
        model_mult = np.ones_like(ability)
    else:
        centred = (ability - float(ability.mean())) / spread
        model_mult = np.exp(_COST_ABILITY_SLOPE * centred)
    if n_scaffolds <= 1:
        scaffold_mult = np.ones(max(n_scaffolds, 1), dtype=float)
    else:
        s_idx = np.arange(n_scaffolds, dtype=float)
        centred_s = (s_idx - (n_scaffolds - 1) / 2.0) / (n_scaffolds - 1)
        scaffold_mult = np.exp(_COST_SCAFFOLD_SLOPE * centred_s)
    return model_mult, scaffold_mult


def _assemble_p(
    *,
    mu: float,
    alpha: np.ndarray,
    eta: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    zeta: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Evaluate the SPEC 4.1 latent predictor and squash it to probabilities.

    Claim: infrastructure
    """
    lin = (
        float(mu)
        + alpha[:, None, None, None]
        + eta[:, :, None, None]
        + gamma[None, None, :, None]
        + beta[None, None, None, :]
        + zeta[:, None, None, :]
        + delta[None, None, :, :]
    )
    return expit(lin)


def _assemble_proxy(
    *,
    mu: float,
    alpha: np.ndarray,
    eta: np.ndarray,
    gamma: np.ndarray,
    beta_proxy: float,
) -> np.ndarray:
    """Block-level success probability of the cheap proxy model, shape ``(T, K, S)``.

    Deliberately has **no** model index: it is correlated with block difficulty (which
    is what drives outcome variance) and carries no information about which of the two
    compared models is stronger, so subtracting it cannot move ``Delta``.

    Claim: variance-reduction
    """
    lin = (
        float(mu)
        + alpha[:, None, None]
        + eta[:, :, None]
        + gamma[None, None, :]
        + float(beta_proxy)
    )
    return expit(lin)


def _finalise(
    *,
    spec: GridSpec,
    p: np.ndarray,
    proxy_p: np.ndarray,
    ability: np.ndarray,
    alpha: np.ndarray,
    gamma: np.ndarray,
    eta: np.ndarray,
    zeta: np.ndarray,
    delta: np.ndarray,
    cost_usd: np.ndarray,
    cost_sec: np.ndarray,
    mu: float,
    beta_proxy: float,
    rank_distortion: float,
    meta: dict[str, Any],
) -> GridTruth:
    """Bundle latent parts into a :class:`GridTruth`, computing ``theta``/``true_order``.

    Claim: infrastructure
    """
    theta = p.reshape(spec.n_blocks, spec.n_models).mean(axis=0)
    true_order = np.argsort(-theta, kind="stable")
    truth = GridTruth(
        spec=spec,
        p=p,
        theta=theta,
        true_order=true_order,
        ability=ability,
        difficulty=alpha,
        scaffold_effect=gamma,
        cost_usd=cost_usd,
        cost_sec=cost_sec,
        proxy_p=proxy_p,
        mu=float(mu),
        seed_jitter=eta,
        task_model=zeta,
        scaffold_model=delta,
        beta_proxy=float(beta_proxy),
        rank_distortion=float(rank_distortion),
        meta=meta,
    )
    if float(rank_distortion) == 0.0 and not truth.rank_is_scaffold_invariant():
        logger.warning(
            "scaffold-invariance of the model ranking failed with rank_distortion=0; "
            "orders=%s true_order=%s",
            truth.scaffold_orders().tolist(),
            np.asarray(true_order).tolist(),
        )
    return truth


def make_truth(
    spec: GridSpec,
    *,
    seed: int = 0,
    mu: float = 0.0,
    sd_task: float = 1.0,
    sd_seed: float = 0.35,
    sd_scaffold: float = 0.8,
    sd_task_model: float = 0.4,
    rank_distortion: float = 0.0,
    ability_gap: float = 0.35,
    ability: np.ndarray | None = None,
    cost_usd_base: float = 0.02,
    cost_sec_base: float = 8.0,
    cost_dispersion: float = 0.6,
    beta_proxy: float = -0.6,
) -> GridTruth:
    """Draw one ground-truth world on the ``spec`` grid.

    The latent predictor is exactly SPEC 4.1::

        logit p[t,k,s,m] = mu + alpha[t] + beta[m] + gamma[s]
                         + eta[t,k] + zeta[t,m] + delta[s,m]

    with ``alpha ~ N(0, sd_task^2)``, ``gamma ~ N(0, sd_scaffold^2)``,
    ``eta ~ N(0, sd_seed^2)``, ``zeta ~ N(0, sd_task_model^2)`` and
    ``delta ~ N(0, rank_distortion^2)``.

    ``delta[s, m]`` is the **only** rank-distorting term. To make that literally true --
    and to honour SPEC 4.1's parenthetical "monotone in ``beta[m]`` at fixed block" --
    the model-contrast part of ``zeta`` is scaled by the largest factor <= 1 that keeps
    ``beta[m] + zeta[t, m]`` in ability order for every task (see
    :func:`_shrink_rank_crossing`). Without that, a task x model interaction large
    enough to flip two models inside a task could, through the curvature of the logistic
    link, make the block-averaged order depend on which scaffolds were averaged -- and
    the scaffold-invariance the estimand exploits would be an accident rather than a
    property. With ``rank_distortion=0`` the induced order is then identical for every
    scaffold and, since block weights are balanced across scaffolds, for every scaffold
    *subset*.

    Models with exactly equal ability are made exchangeable (identical ``zeta`` and
    ``delta`` rows), so ``ability=[0, 0, ...]`` yields an exact null pair for coverage
    testing rather than an almost-null one.

    Costs follow SPEC 4.1: ``cost_usd = cost_usd_base * exp(N(0, cost_dispersion^2))
    * scaffold_cost_mult[s] * model_cost_mult[m]``, with the multipliers increasing in
    model strength and scaffold index, and with the dollar and second shocks positively
    correlated. Costs therefore vary by an order of magnitude across cells and the
    expensive cells are the informative ones.

    Claim: infrastructure
    """
    if sd_task < 0 or sd_seed < 0 or sd_scaffold < 0 or sd_task_model < 0:
        raise ValueError("standard deviations must be non-negative")
    if rank_distortion < 0:
        raise ValueError("rank_distortion must be non-negative")
    if cost_usd_base <= 0 or cost_sec_base <= 0:
        raise ValueError("cost bases must be positive")
    if cost_dispersion < 0:
        raise ValueError("cost_dispersion must be non-negative")

    t_n, k_n, s_n, m_n = spec.grid_shape
    if min(t_n, k_n, s_n, m_n) < 1:
        raise ValueError(f"every grid dimension must be >= 1, got {spec.grid_shape}")

    rng = np.random.default_rng(seed)

    if ability is None:
        beta = default_ability(m_n, ability_gap)
    else:
        beta = np.asarray(ability, dtype=float).reshape(-1).copy()
        if beta.shape[0] != m_n:
            raise ValueError(f"ability must have length {m_n}, got {beta.shape[0]}")

    alpha = rng.normal(0.0, sd_task, size=t_n) if sd_task > 0 else np.zeros(t_n)
    gamma = rng.normal(0.0, sd_scaffold, size=s_n) if sd_scaffold > 0 else np.zeros(s_n)
    eta = rng.normal(0.0, sd_seed, size=(t_n, k_n)) if sd_seed > 0 else np.zeros((t_n, k_n))
    zeta = (
        rng.normal(0.0, sd_task_model, size=(t_n, m_n))
        if sd_task_model > 0
        else np.zeros((t_n, m_n))
    )
    delta = (
        rng.normal(0.0, rank_distortion, size=(s_n, m_n))
        if rank_distortion > 0
        else np.zeros((s_n, m_n))
    )

    groups = _tie_groups(beta)
    if groups:
        _make_exchangeable([zeta, delta], groups)
        logger.debug("tied abilities made exchangeable for groups %s", groups)

    crossing_scale = _shrink_rank_crossing(beta, zeta)
    if crossing_scale < 1.0:
        row_mean = zeta.mean(axis=1, keepdims=True)
        zeta = row_mean + crossing_scale * (zeta - row_mean)
        logger.debug(
            "scaled task x model contrasts by %.4f to keep block ranks strict",
            crossing_scale,
        )

    p = _assemble_p(mu=mu, alpha=alpha, eta=eta, gamma=gamma, beta=beta, zeta=zeta, delta=delta)
    proxy_p = _assemble_proxy(mu=mu, alpha=alpha, eta=eta, gamma=gamma, beta_proxy=beta_proxy)

    model_mult, scaffold_mult = _cost_multipliers(beta, s_n)
    shape = spec.grid_shape
    z_shared = rng.normal(size=shape)
    z_usd = rng.normal(size=shape)
    z_sec = rng.normal(size=shape)
    tail = float(np.sqrt(max(1.0 - _COST_RHO**2, 0.0)))
    eps_usd = cost_dispersion * (_COST_RHO * z_shared + tail * z_usd)
    eps_sec = cost_dispersion * (_COST_RHO * z_shared + tail * z_sec)
    mult = scaffold_mult[None, None, :, None] * model_mult[None, None, None, :]
    cost_usd = cost_usd_base * np.exp(eps_usd) * mult
    cost_sec = cost_sec_base * np.exp(eps_sec) * mult

    meta: dict[str, Any] = {
        "seed": int(seed),
        "mu": float(mu),
        "sd_task": float(sd_task),
        "sd_seed": float(sd_seed),
        "sd_scaffold": float(sd_scaffold),
        "sd_task_model": float(sd_task_model),
        "rank_distortion": float(rank_distortion),
        "ability_gap": float(ability_gap),
        "cost_usd_base": float(cost_usd_base),
        "cost_sec_base": float(cost_sec_base),
        "cost_dispersion": float(cost_dispersion),
        "crossing_scale": float(crossing_scale),
        "tied_ability_groups": [list(g) for g in groups],
        "shift": 0.0,
        "shifted_scaffolds": [],
    }
    return _finalise(
        spec=spec,
        p=p,
        proxy_p=proxy_p,
        ability=beta,
        alpha=alpha,
        gamma=gamma,
        eta=eta,
        zeta=zeta,
        delta=delta,
        cost_usd=cost_usd,
        cost_sec=cost_sec,
        mu=mu,
        beta_proxy=beta_proxy,
        rank_distortion=rank_distortion,
        meta=meta,
    )


def apply_scaffold_shift(
    truth: GridTruth,
    *,
    shift: float,
    scaffolds: Sequence[int] | int | None = None,
    rank_distortion: float | None = None,
    seed: int = 0,
) -> GridTruth:
    """Move scaffold main effects (and optionally inject rank distortion).

    Adds ``shift`` to ``gamma[s]`` for the selected scaffolds -- ``None`` means all of
    them -- keeping every other latent component fixed, so the returned world is the
    *same* tasks, seeds and models seen through a differently-behaved harness. A large
    ``shift`` moves absolute scores a great deal, and through the curvature of the
    logistic link it changes **which items are hard**, which is precisely what
    invalidates a fixed anchor set chosen by a scaffold-free IRT pilot.

    With ``rank_distortion=None`` the existing ``delta`` is kept, so with a
    rank-distortion-free base world the ranking survives the shift untouched: absolute
    scores move, the estimand does not. Passing a positive ``rank_distortion`` redraws
    ``delta ~ N(0, rank_distortion^2)`` and genuinely reorders models per scaffold; the
    returned ``true_order`` is recomputed from the shifted world, because that is the
    ground truth a method run *after* the shift is responsible for recovering.

    Claim: shift-robust
    """
    spec = truth.spec
    s_n, m_n = spec.n_scaffolds, spec.n_models
    if scaffolds is None:
        idx = list(range(s_n))
    elif isinstance(scaffolds, (int, np.integer)):
        idx = [int(scaffolds)]
    else:
        idx = [int(s) for s in scaffolds]
    for s in idx:
        if not 0 <= s < s_n:
            raise IndexError(f"scaffold {s} out of range for S={s_n}")

    gamma = np.asarray(truth.scaffold_effect, dtype=float).copy()
    gamma[idx] += float(shift)

    if rank_distortion is None:
        delta = np.asarray(truth.scaffold_model, dtype=float).copy()
        new_distortion = float(truth.rank_distortion)
    else:
        if rank_distortion < 0:
            raise ValueError("rank_distortion must be non-negative")
        rng = np.random.default_rng(seed)
        delta = (
            rng.normal(0.0, float(rank_distortion), size=(s_n, m_n))
            if rank_distortion > 0
            else np.zeros((s_n, m_n))
        )
        groups = _tie_groups(truth.ability)
        if groups:
            _make_exchangeable([delta], groups)
        new_distortion = float(rank_distortion)

    if delta.shape != (s_n, m_n):
        raise ValueError(f"scaffold_model must have shape {(s_n, m_n)}, got {delta.shape}")

    alpha = np.asarray(truth.difficulty, dtype=float)
    eta = np.asarray(truth.seed_jitter, dtype=float)
    zeta = np.asarray(truth.task_model, dtype=float)
    beta = np.asarray(truth.ability, dtype=float)
    if eta.shape != (spec.n_tasks, spec.n_seeds) or zeta.shape != (spec.n_tasks, m_n):
        raise ValueError(
            "truth is missing its latent components; apply_scaffold_shift needs a "
            "GridTruth produced by make_truth"
        )

    p = _assemble_p(mu=truth.mu, alpha=alpha, eta=eta, gamma=gamma, beta=beta, zeta=zeta, delta=delta)
    proxy_p = _assemble_proxy(
        mu=truth.mu, alpha=alpha, eta=eta, gamma=gamma, beta_proxy=truth.beta_proxy
    )

    meta = dict(truth.meta)
    meta.update(
        {
            "shift": float(shift),
            "shifted_scaffolds": list(idx),
            "shift_seed": int(seed),
            "rank_distortion": new_distortion,
            "base_true_order": np.asarray(truth.true_order).tolist(),
        }
    )
    shifted = _finalise(
        spec=spec,
        p=p,
        proxy_p=proxy_p,
        ability=beta,
        alpha=alpha,
        gamma=gamma,
        eta=eta,
        zeta=zeta,
        delta=delta,
        cost_usd=np.asarray(truth.cost_usd, dtype=float).copy(),
        cost_sec=np.asarray(truth.cost_sec, dtype=float).copy(),
        mu=truth.mu,
        beta_proxy=truth.beta_proxy,
        rank_distortion=new_distortion,
        meta=meta,
    )
    logger.debug(
        "applied scaffold shift %.3f to scaffolds %s; order %s -> %s",
        shift,
        idx,
        np.asarray(truth.true_order).tolist(),
        np.asarray(shifted.true_order).tolist(),
    )
    return shifted


def true_delta(truth: GridTruth, a: int, c: int, w: np.ndarray | None = None) -> float:
    """Exact SPEC 2.1 contrast ``Delta[a,c] = sum_b w[b] (p[b,a] - p[b,c])``.

    Ground truth for the outer estimand: tests compare a confidence sequence against
    this number, never against a re-estimate. Note ``Delta[a,c] = theta[a] - theta[c]``,
    which is why certifying adjacent pairs certifies the whole order by transitivity.

    Claim: coverage
    """
    spec = truth.spec
    if not (0 <= a < spec.n_models and 0 <= c < spec.n_models):
        raise IndexError(f"model indices ({a}, {c}) out of range for M={spec.n_models}")
    w_full = uniform_weights(spec) if w is None else np.asarray(w, dtype=float)
    if w_full.shape != (spec.n_blocks,):
        raise ValueError(f"w must have shape ({spec.n_blocks},), got {w_full.shape}")
    pb = truth.p_blocks
    return float(w_full @ (pb[:, a] - pb[:, c]))


class RolloutSimulator:
    """Executes rollouts on a :class:`GridTruth`, charging money and seconds per step.

    The trajectory model of SPEC 4.3 is what makes inner truncation meaningful. The true
    outcome is drawn **first**; the trajectory is then a random walk whose drift sign is
    that outcome's, so the running mean of a prefix separates eventual successes from
    eventual failures *progressively* -- early steps are ambiguous, late steps are
    informative. This is the "failure becomes visible partway through" structure that a
    per-rollout truncation rule can exploit, and it is also why a naive rule is
    optimistic on early stops and needs the audit correction of SPEC 5.4.

    Cost accrues per executed step: a rollout stopped after ``j+1`` of ``n_steps_full``
    steps costs that fraction of the cell's cost (times lognormal noise). Truncation
    therefore saves real dollars and seconds, which is what metric (4) measures --
    except on audited rollouts, which run to completion and save nothing. That is the
    honest price of unbiasedness.

    ``Rollout.success`` on a truncated, non-audited rollout is the rule's guess (``0``)
    and ``completed`` is ``False``; ``true_success`` records ground truth for tests and
    diagnostics only. No estimator may read ``true_success``, and none may read
    ``success`` for a truncated rollout without going through
    ``nested_seq.corrected_outcome``.

    Claim: unbiased-truncation
    """

    def __init__(
        self,
        truth: GridTruth,
        *,
        seed: int = 0,
        max_steps: int = 24,
        signal_strength: float = 2.2,
        cost_noise: float = 0.25,
        instance_corr: float = 0.6,
    ) -> None:
        """Bind a simulator to a truth grid with its own reproducible RNG stream.

        Claim: infrastructure
        """
        if max_steps < 3:
            raise ValueError("max_steps must be >= 3")
        if cost_noise < 0:
            raise ValueError("cost_noise must be non-negative")
        self.truth = truth
        self.spec: GridSpec = truth.spec
        self.max_steps = int(max_steps)
        self.signal_strength = float(signal_strength)
        self.cost_noise = float(cost_noise)
        instance_corr = float(instance_corr)
        if not (0.0 <= instance_corr <= 1.0):
            raise ValueError('instance_corr must be in [0, 1]')
        self.instance_corr = instance_corr
        self.seed = int(seed)
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self.n_rollouts: int = 0
        self.n_proxy_rollouts: int = 0
        self.total_usd: float = 0.0
        self.total_sec: float = 0.0
        self.total_steps: int = 0
        self.total_steps_full: int = 0
        self.n_truncated: int = 0
        self.n_audited: int = 0

    def reset_counters(self) -> None:
        """Zero the cost/step accounting (the RNG stream is untouched).

        Claim: infrastructure
        """
        self.n_rollouts = 0
        self.n_proxy_rollouts = 0
        self.total_usd = 0.0
        self.total_sec = 0.0
        self.total_steps = 0
        self.total_steps_full = 0
        self.n_truncated = 0
        self.n_audited = 0

    def reseed(self, seed: int) -> None:
        """Restart the internal RNG stream, for replication-level reproducibility.

        Claim: infrastructure
        """
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)

    @property
    def counters(self) -> dict[str, float]:
        """Snapshot of the accounting counters, for benchmark bookkeeping.

        Claim: low-cost
        """
        return {
            "n_rollouts": float(self.n_rollouts),
            "total_usd": self.total_usd,
            "total_sec": self.total_sec,
            "total_steps": float(self.total_steps),
            "total_steps_full": float(self.total_steps_full),
            "n_truncated": float(self.n_truncated),
            "n_audited": float(self.n_audited),
        }

    @property
    def compute_saved(self) -> float:
        """Fraction of trajectory steps avoided so far, in ``[0, 1)``.

        Claim: unbiased-truncation
        """
        if self.total_steps_full <= 0:
            return 0.0
        return 1.0 - (self.total_steps / self.total_steps_full)

    def mean_costs(self) -> tuple[np.ndarray, np.ndarray]:
        """``(cost_usd, cost_sec)`` mean-cost grids, copies, for the cost-aware allocator.

        Claim: low-cost
        """
        return (
            np.asarray(self.truth.cost_usd, dtype=float).copy(),
            np.asarray(self.truth.cost_sec, dtype=float).copy(),
        )

    def _check_cell(self, t: int, k: int, s: int, m: int) -> None:
        """Validate a cell index against the grid shape.

        Claim: infrastructure
        """
        shape = self.spec.grid_shape
        for name, value, size in zip(("task", "seed", "scaffold", "model"), (t, k, s, m), shape):
            if not 0 <= int(value) < int(size):
                raise IndexError(f"{name} index {value} out of range [0, {size})")

    def _draw_trajectory(
        self, success: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, int]:
        """Draw ``n_steps_full`` and the outcome-conditioned random walk of SPEC 4.3.

        The whole walk is generated up front even when the rule is going to truncate it,
        so that a rollout's outcome and trajectory do not depend on *where* the rule
        fires inside it. Truncation then only decides how much of an already-determined
        rollout is paid for and observed, which is the clean version of the counterfactual
        the compute-saving metric reports.

        Claim: unbiased-truncation
        """
        n_full = int(np.clip(6 + rng.poisson(self.max_steps / 3.0), 3, self.max_steps))
        drift = self.signal_strength / n_full
        if not success:
            drift = -drift
        x0 = float(rng.normal(0.0, 1.0))
        if n_full == 1:
            return np.array([x0], dtype=float), n_full
        increments = rng.normal(drift, 1.0, size=n_full - 1)
        traj = np.empty(n_full, dtype=float)
        traj[0] = x0
        traj[1:] = x0 + np.cumsum(increments)
        return traj, n_full

    def _cost_factors(self, rng: np.random.Generator) -> tuple[float, float]:
        """Correlated mean-one lognormal shocks for this rollout's dollars and seconds.

        Claim: low-cost
        """
        if self.cost_noise <= 0.0:
            return 1.0, 1.0
        z_shared = float(rng.normal())
        z_usd = float(rng.normal())
        z_sec = float(rng.normal())
        tail = float(np.sqrt(max(1.0 - _COST_RHO**2, 0.0)))
        half = 0.5 * self.cost_noise**2
        f_usd = float(np.exp(self.cost_noise * (_COST_RHO * z_shared + tail * z_usd) - half))
        f_sec = float(np.exp(self.cost_noise * (_COST_RHO * z_shared + tail * z_sec) - half))
        return f_usd, f_sec

    def _draw_uniform(self, shock: float | None, gen: np.random.Generator) -> float:
        """Draw the instance's uniform, optionally coupled to a shared instance shock.

        With ``shock=None`` this is a plain ``U(0, 1)``.  With a shock ``g`` (a standard
        normal drawn once per problem instance and shared by every model attempting it),
        the uniform is a Gaussian copula draw::

            z = rho * g + sqrt(1 - rho^2) * eps,   eps ~ N(0, 1)
            u = Phi(z)

        ``u`` is **exactly** ``U(0, 1)`` marginally, so ``P(success) = p`` is untouched:
        the copula changes the *dependence* between models on the same instance and
        nothing else.  ``p``, ``theta`` and ``true_order`` are therefore identical with
        and without it -- the estimand does not move, which is what makes this a fair
        variance-reduction device rather than a change of problem.

        Why it belongs here: with independent draws, two models' outcomes on the same
        instance share nothing, and a cheap proxy's realised success carries no
        information about them.  Real evaluations are not like that -- a genuinely hard
        instance is hard for everything -- and it is precisely that shared difficulty
        that a control variate exists to remove.  ``instance_corr = 0`` recovers the
        independent case for ablations.

        Claim: variance-reduction
        """
        if shock is None or self.instance_corr <= 0.0:
            return float(gen.random())
        rho = self.instance_corr
        eps = float(gen.standard_normal())
        z = rho * float(shock) + math.sqrt(max(0.0, 1.0 - rho * rho)) * eps
        return float(_std_normal_cdf(z))

    def draw_shock(self, rng: np.random.Generator | None = None) -> float:
        """Draw one shared instance shock, to be passed to every model on that instance.

        The driver draws this once per matched-pair observation so both models face the
        same instance.  This is common-random-numbers pairing, and it is legitimate for
        the same reason the copula is: it changes only the joint law, never the marginals
        that define the estimand.

        Claim: variance-reduction
        """
        gen = self._rng if rng is None else rng
        return float(gen.standard_normal())

    def proxy_rollout(
        self,
        t: int,
        k: int,
        s: int,
        *,
        shock: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Realised success of the cheap proxy model on this instance.

        This is the covariate the brief asks for -- "a small proxy's success or failure on
        the same task" -- and it is deliberately the **realised outcome**, not the
        probability.  A probability is a function of the block, so it can only remove
        *between-block* variance; for a paired Bernoulli contrast almost all the variance
        is within a block, so a block-level covariate has essentially nothing to remove
        (measured: variance ratio 1.00).  The realised proxy outcome shares the instance
        shock with the models under test, so it removes within-block variance too.

        Its ``w``-weighted mean is still exactly known -- ``sum_b w[b] proxy_p[b]``, a
        closed-form sum over the whole grid -- so it remains an admissible known-mean
        control variate under :mod:`quorum.cuped`.

        The proxy is charged no budget here: it stands in for a model orders of magnitude
        cheaper than those under test.  Benchmarks that want to charge it should add
        ``proxy_cost_frac * cost`` themselves.

        Claim: variance-reduction
        """
        self._check_cell(t, k, s, 0)
        gen = self._rng if rng is None else rng
        p_proxy = float(self.truth.proxy_p[int(t), int(k), int(s)])
        self.n_proxy_rollouts += 1
        return int(self._draw_uniform(shock, gen) < p_proxy)

    def rollout(
        self,
        t: int,
        k: int,
        s: int,
        m: int,
        *,
        truncator: Truncator | None = None,
        rng: np.random.Generator | None = None,
        shock: float | None = None,
    ) -> Rollout:
        """Run one rollout of model ``m`` on block ``(t, k, s)``, optionally truncated.

        With a ``truncator``, the rule is consulted after every step ``j >= 1`` on the
        prefix ``x[:j+1]``. When it fires the audit coin is flipped **once**, using this
        rollout's RNG and independently of the (unseen) outcome:

        * audited -> the rollout runs to completion, the true outcome is reported,
          ``completed=False`` and ``audited=True``. No compute is saved; this subsample
          is what the SPEC 5.4 correction spends to buy back unbiasedness.
        * not audited -> execution stops immediately, ``success=0`` (the rule's guess),
          ``completed=False``, ``n_steps=j+1``. Compute is saved and the reported
          outcome is, on its own, biased low -- which is exactly why estimators must
          consume it through ``nested_seq.corrected_outcome``.

        ``true_success`` always carries ground truth, for tests and diagnostics only.

        Claim: unbiased-truncation
        """
        self._check_cell(t, k, s, m)
        t, k, s, m = int(t), int(k), int(s), int(m)
        gen = self._rng if rng is None else rng

        p_cell = float(self.truth.p[t, k, s, m])
        true_success = int(self._draw_uniform(shock, gen) < p_cell)
        traj, n_full = self._draw_trajectory(true_success, gen)

        truncated_at = -1
        audited = False
        if truncator is not None:
            for j in range(1, n_full):
                prefix = traj[: j + 1]
                prefix.flags.writeable = False  # the rule observes, it does not edit
                if bool(truncator.should_stop(prefix, j)):
                    truncated_at = j
                    audited = bool(truncator.audit(gen))
                    break

        if truncated_at >= 0:
            completed = False
            if audited:
                n_steps = n_full
                observed = true_success
            else:
                n_steps = truncated_at + 1
                observed = 0
        else:
            completed = True
            n_steps = n_full
            observed = true_success

        f_usd, f_sec = self._cost_factors(gen)
        frac = n_steps / n_full
        cost_usd = float(self.truth.cost_usd[t, k, s, m]) * frac * f_usd
        cost_sec = float(self.truth.cost_sec[t, k, s, m]) * frac * f_sec

        self.n_rollouts += 1
        self.total_usd += cost_usd
        self.total_sec += cost_sec
        self.total_steps += n_steps
        self.total_steps_full += n_full
        if truncated_at >= 0:
            self.n_truncated += 1
            if audited:
                self.n_audited += 1

        return Rollout(
            task=t,
            seed=k,
            scaffold=s,
            model=m,
            success=int(observed),
            completed=completed,
            n_steps=int(n_steps),
            n_steps_full=int(n_full),
            cost_usd=cost_usd,
            cost_sec=cost_sec,
            trajectory=traj[:n_steps].copy(),
            audited=audited,
            truncated_at=int(truncated_at),
            true_success=int(true_success),
        )
