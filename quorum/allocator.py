"""Decision-targeted, cost-aware allocation of the next rollout (SPEC section 7).

This module answers the question *"which rollout should I buy next?"* by pricing every
candidate cell in units of **reduction in the variance of the contrast that decides the
ranking, per dollar spent**.  Three ideas are stacked:

1. **c-optimality (decision-targeted).**  Under the Laplace-Gaussian posterior of
   :class:`quorum.cell_model.HierarchicalCellModel` with covariance ``Sigma``, buying one
   Bernoulli observation at a cell with one-hot design row ``x`` and Fisher weight
   ``u = p (1 - p)`` performs the exact rank-1 update
   ``Sigma' = Sigma - u Sigma x x^T Sigma / (1 + u x^T Sigma x)``, so the variance of the
   decision contrast ``v = e_a - e_c`` drops by exactly
   ``gain = u (v^T Sigma x)^2 / (1 + u x^T Sigma x) >= 0``.  We score cells by *that*,
   not by how much they shrink uncertainty overall.
2. **Cost awareness.**  ``score = gain / (cost_usd + lambda_sec * cost_sec + eps)``.  A
   cell that is informative but ruinously expensive loses to one that is slightly less
   informative and ten times cheaper.
3. **Decision focus across pairs.**  Each candidate pair is weighted by its probability
   of *sign flip* under the rank posterior, so pairs that are already resolved stop
   attracting budget and the remaining money flows to the contested frontier.

The allocator is **steering only**.  Validity of the outer confidence sequence comes from
the Horvitz-Thompson identity of SPEC 5.2, which holds for *any* predictable ``pi``; a
badly fit cell model costs efficiency, never coverage.  The one thing the allocator owes
the estimator is the exploration floor of :func:`mix_to_floor`, which is a statistical
requirement rather than a heuristic.

Claim: low-cost
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from quorum.types import GridSpec, Rollout, uniform_weights, unblock_index

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the runtime import graph acyclic
    from quorum.cell_model import HierarchicalCellModel

__all__ = [
    "DecisionTargetedAllocator",
    "mix_to_floor",
    "STRATEGIES",
]

logger = logging.getLogger(__name__)

#: Strategies accepted by :class:`DecisionTargetedAllocator`.
STRATEGIES: tuple[str, ...] = ("info_cost", "thompson", "uniform", "neyman")

#: Additive guard in the cost denominator so a free cell does not divide by zero.
_COST_EPS: float = 1e-9

#: Floor on a pair's flip weight, so a fully resolved family still yields a valid draw.
_WEIGHT_FLOOR: float = 1e-6

#: Multiplicative damping applied to pairs whose channel already excludes zero.
_RESOLVED_DAMPING: float = 1e-3


def mix_to_floor(pi_raw: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Mix a raw allocation towards the target weights: ``pi = (1 - eps) pi_raw + eps w``.

    This is the **mandatory exploration floor** of SPEC 7.2, and it is a statistical
    requirement, not a heuristic.  The outer estimator is Horvitz-Thompson:
    ``Z_i = (w[b_i] / pi_i[b_i]) (Ytil_a - Ytil_c)``.  Its unbiasedness holds for any
    predictable ``pi``, but the empirical-Bernstein confidence sequence needs *bounded*
    observations, and the bound is driven by the HT weight.  Mixing guarantees
    ``pi[b] >= eps * w[b]`` and therefore

        ``w[b] / pi[b] <= w[b] / (eps * w[b]) = 1 / eps``   for every block with ``w[b] > 0``

    (blocks with ``w[b] = 0`` contribute an HT weight of exactly 0, so the bound
    ``G = 1/eps`` holds uniformly).  Without the floor an adaptive allocator can drive
    ``pi[b]`` to zero, ``G`` to infinity, and the confidence sequence to infinite width --
    the interval would stop shrinking no matter how much budget was spent.  The floor is
    what keeps the outer CS finite-width, and it is the reason adaptivity is free.

    Args:
        pi_raw: Non-negative raw allocation over blocks; it is normalised internally.
        w: Fixed target weights over blocks (sum 1); these define the estimand.
        eps: Floor fraction ``pi_min_frac`` in ``(0, 1]``.

    Returns:
        A probability vector over blocks satisfying ``pi >= eps * w`` and ``sum(pi) == 1``.

    Claim: coverage
    """
    if not 0.0 < eps <= 1.0:
        raise ValueError(f"pi_min_frac must lie in (0, 1], got {eps!r}")
    raw = np.asarray(pi_raw, dtype=float).ravel()
    weights = np.asarray(w, dtype=float).ravel()
    if raw.shape != weights.shape:
        raise ValueError(f"pi_raw shape {raw.shape} does not match w shape {weights.shape}")
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.clip(raw, 0.0, None)
    total = float(raw.sum())
    if total <= 0.0:
        logger.debug("mix_to_floor: degenerate raw allocation, falling back to w")
        raw = weights.copy()
        total = float(raw.sum())
    raw = raw / total
    pi = (1.0 - eps) * raw + eps * weights
    pi = np.clip(pi, 0.0, None)
    pi_sum = float(pi.sum())
    if pi_sum <= 0.0:  # pragma: no cover - impossible for a valid w
        raise ValueError("mix_to_floor produced an all-zero allocation")
    return pi / pi_sum


class DecisionTargetedAllocator:
    """Chooses the next (pair, block) to fund by expected rank-information per dollar.

    This is contribution (b) of the thesis.  At every step it produces a *predictable*
    distribution ``pi`` over blocks -- predictable in the martingale sense: it is a
    function of what was observed strictly before the current draw, which is exactly the
    condition under which the Horvitz-Thompson channel of SPEC 5.2 stays unbiased.  The
    returned ``pi`` always satisfies the exploration floor of :func:`mix_to_floor`.

    Strategies:

    ``"info_cost"`` (default)
        The c-optimal criterion of SPEC 7.1.  For the pair under consideration the
        per-cell gains come from ``cell_model.info_gain_all_cells(a, c)`` (vectorised) or,
        if that helper is absent, from a loop over ``info_gain_for_cell``.  Because a pair
        is executed as a *matched pair* on the same block -- one rollout of ``a`` and one
        of ``c`` at the same ``b`` -- the relevant unit is the block, so the block score
        aggregates both of the pair's cells::

            gain_block[b] = gain[b, a] + gain[b, c]
            cost_block[b] = cost_usd[b, a] + cost_usd[b, c]
                          + lambda_sec * (cost_sec[b, a] + cost_sec[b, c])
            score[b]      = gain_block[b] / (cost_block[b] + eps)

        i.e. **information about the deciding contrast per dollar**.  ``pi_raw`` is
        ``score`` normalised.  The pair itself is chosen from the utility
        ``u[pair] = flip_weight[pair] * max_b score[b]`` (how close the pair is to
        flipping, times the best purchase available for it) -- see
        :meth:`_choose_index` for the exact, rng-deterministic rule.

    ``"thompson"``
        Two ability vectors are drawn from the posterior (two independent halves of the
        same draw matrix); the funded pair is the one whose contrast sign disagrees most
        often between the halves.  Blocks are then sampled proportionally to the
        model-predicted contrast variance divided by ``sqrt(cost)``.

    ``"uniform"``
        The genuinely non-adaptive (C) *Sequential-only* baseline: ``pi_raw = w`` exactly
        (so after mixing ``pi == w``), and the pair cycles round-robin through the
        supplied adjacent pairs.  It never reads the cell model or the channels, which is
        what makes it an honest control.

    ``"neyman"``
        The classical optimum as an ablation: ``pi_raw ∝ w * sd_b / sqrt(cost_b)`` with
        ``sd_b`` the model-predicted standard deviation of the block contrast.  It is
        variance-targeted rather than decision-targeted, which is precisely the
        comparison the benchmark wants.

    Cost control: the cell model is refit only every ``refit_every`` observations, and all
    Sigma-derived quantities (per-cell gains, predicted ``p``, rank posterior, ability
    draws) are cached between refits, so per-step allocation overhead is negligible next
    to the cost of a rollout.

    Claim: low-cost
    """

    def __init__(
        self,
        spec: GridSpec,
        cell_model: "HierarchicalCellModel",
        costs_usd: np.ndarray | float,
        costs_sec: np.ndarray | float,
        *,
        w: np.ndarray | None = None,
        pi_min_frac: float = 0.10,
        strategy: str = "info_cost",
        lambda_sec: float = 0.0,
        refit_every: int = 25,
        n_draws: int = 256,
        rng: np.random.Generator | None = None,
        temperature: float = 1.0,
    ) -> None:
        """Build an allocator over ``spec``'s blocks with per-cell costs.

        Args:
            spec: Grid shape; ``spec.n_blocks`` blocks, ``spec.n_models`` models.
            cell_model: Fitted-or-fittable hierarchical cell model used purely to steer.
            costs_usd: Per-cell dollar cost, shape ``(T, K, S, M)``, ``(B, M)``, ``(M,)``
                or a scalar.
            costs_sec: Per-cell wall-clock cost, same accepted shapes.
            w: Fixed target weights over blocks (default uniform).  Never adapts.
            pi_min_frac: Exploration floor ``eps`` of :func:`mix_to_floor`; it bounds the
                Horvitz-Thompson weight by ``1 / pi_min_frac``.
            strategy: One of :data:`STRATEGIES`.
            lambda_sec: Dollars-per-second exchange rate folded into the cost denominator.
            refit_every: Refit the cell model only every this many observations.
            n_draws: Posterior draws used for flip weights and Thompson sampling.
            rng: Explicit generator; all randomness in this object flows through it.
            temperature: Pair-selection temperature; ``<= 0`` means deterministic argmax.

        Claim: low-cost
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        if not 0.0 < pi_min_frac <= 1.0:
            raise ValueError(f"pi_min_frac must lie in (0, 1], got {pi_min_frac!r}")
        if refit_every < 1:
            raise ValueError(f"refit_every must be >= 1, got {refit_every!r}")
        if n_draws < 2:
            raise ValueError(f"n_draws must be >= 2, got {n_draws!r}")

        self.spec: GridSpec = spec
        self.cell_model = cell_model
        self.strategy: str = strategy
        self.lambda_sec: float = float(lambda_sec)
        self.pi_min_frac: float = float(pi_min_frac)
        self.refit_every: int = int(refit_every)
        self.n_draws: int = int(n_draws)
        self.temperature: float = float(temperature)
        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

        self.n_blocks: int = spec.n_blocks
        self.n_models: int = spec.n_models

        self.w: np.ndarray = (
            uniform_weights(spec) if w is None else np.asarray(w, dtype=float).ravel()
        )
        if self.w.shape != (self.n_blocks,):
            raise ValueError(
                f"w must have shape ({self.n_blocks},), got {self.w.shape}"
            )
        if np.any(self.w < 0.0):
            raise ValueError("w must be non-negative")
        w_sum = float(self.w.sum())
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError("w must be finite and have positive mass")
        self.w = self.w / w_sum

        self.costs_usd: np.ndarray = self._as_cell_array(costs_usd, "costs_usd")
        self.costs_sec: np.ndarray = self._as_cell_array(costs_sec, "costs_sec")
        #: Per-cell cost denominator, dollars-equivalent (never zero).
        self._cell_cost: np.ndarray = (
            self.costs_usd + self.lambda_sec * self.costs_sec
        )

        # --- bookkeeping -------------------------------------------------------------
        self.n_observed: int = 0
        self.n_fits: int = 0
        self._since_fit: int = 0
        self._fitted: bool = False
        self._rr_index: int = 0
        # Posterior-dependent cache, cleared on every refit.
        self._cache: dict[Any, Any] = {}
        # Cost-only cache: costs are fixed inputs, so these survive refits.
        self._cache_static: dict[Any, Any] = {}
        #: Last returned allocation, kept for diagnostics and tests.
        self.last_pi: np.ndarray | None = None
        self.last_pair: tuple[int, int] | None = None

    # ------------------------------------------------------------------ properties --

    @property
    def pi_min(self) -> float:
        """Smallest allocation probability guaranteed for any block with ``w[b] > 0``.

        Equal to ``pi_min_frac * min_{w[b] > 0} w[b]``; under uniform ``w`` this is
        ``pi_min_frac / B``, exactly the ``pi_min`` that :class:`PairChannel` asserts.

        Claim: coverage
        """
        positive = self.w[self.w > 0.0]
        if positive.size == 0:  # pragma: no cover - excluded by the constructor
            return 0.0
        return float(self.pi_min_frac * positive.min())

    def ht_bound(self) -> float:
        """The Horvitz-Thompson ratio bound ``max_b w[b]/pi[b]`` this design guarantees.

        Because :func:`mix_to_floor` returns ``pi = (1-eps)*pi_raw + eps*w`` with
        ``eps = pi_min_frac``, every block satisfies ``pi[b] >= eps * w[b]`` and therefore
        ``w[b]/pi[b] <= 1/eps`` *pointwise*.  That is strictly tighter than the generic
        ``w_max / pi_min`` bound whenever ``w`` is non-uniform, and it is the quantity the
        outer confidence sequence should be sized against: the width of an
        empirical-Bernstein CS is linear in the declared range once the betting fraction
        saturates, so every factor of slack here is paid for in rollouts.

        This is the exact statistical price of adaptivity.  Concentrating the design buys
        a lower variance in the ``1/sqrt(n)`` term and costs ``1/eps`` in the ``1/n`` term;
        ``pi_min_frac`` is where that trade is set.

        Claim: coverage
        """
        return float(1.0 / self.pi_min_frac)

    # ------------------------------------------------------------------- public API --

    def select(
        self,
        pairs: Sequence[tuple[int, int]],
        channels: Mapping[tuple[int, int], Any] | Sequence[Any] | None = None,
    ) -> tuple[tuple[int, int], np.ndarray, int, float]:
        """Pick the pair to fund and the block distribution, then sample a block from it.

        The block is sampled with **this allocator's** rng from the very vector that is
        returned, so the sampling probability handed to :class:`PairChannel` is exactly
        the predictable ``pi[b]`` that was in force at the draw -- which is what makes the
        Horvitz-Thompson ratio ``w[b]/pi[b]`` an unbiased reweighting rather than a
        post-hoc guess.

        Args:
            pairs: Candidate (usually adjacent) ordered model pairs ``(a, c)``.
            channels: Optional per-pair channels, either a mapping keyed by ``(a, c)``
                (either orientation) or a sequence aligned with ``pairs``.  Used only to
                stop funding pairs whose interval already excludes zero.

        Returns:
            ``(pair, pi, block, pi[block])`` with ``pi`` of length ``B``, summing to 1 and
            satisfying ``pi >= pi_min_frac * w`` elementwise.

        Claim: low-cost
        """
        pair_list = [(int(a), int(c)) for a, c in pairs]
        if not pair_list:
            raise ValueError("select() needs at least one candidate pair")
        for a, c in pair_list:
            if not (0 <= a < self.n_models and 0 <= c < self.n_models):
                raise ValueError(f"pair ({a}, {c}) out of range for {self.n_models} models")
            if a == c:
                raise ValueError(f"pair ({a}, {c}) compares a model with itself")

        if self.strategy != "uniform":
            self._maybe_refit()

        if self.strategy == "uniform":
            pair, pi_raw = self._select_uniform(pair_list)
        elif self.strategy == "info_cost":
            pair, pi_raw = self._select_info_cost(pair_list, channels)
        elif self.strategy == "thompson":
            pair, pi_raw = self._select_thompson(pair_list, channels)
        elif self.strategy == "neyman":
            pair, pi_raw = self._select_neyman(pair_list, channels)
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"unknown strategy {self.strategy!r}")

        pi = mix_to_floor(pi_raw, self.w, self.pi_min_frac)

        # The floor and the normalisation are load-bearing for validity, so assert them.
        assert abs(float(pi.sum()) - 1.0) <= 1e-9, "allocation does not sum to 1"
        floor = self.pi_min_frac * self.w
        assert np.all(pi >= floor - 1e-12), "allocation violates the exploration floor"

        block = int(self.rng.choice(self.n_blocks, p=pi))
        self.last_pi = pi
        self.last_pair = pair
        return pair, pi, block, float(pi[block])

    def observe(self, rollout: Rollout) -> None:
        """Forward one executed rollout into the cell model and advance the refit clock.

        The cell model is fed the *observed* ``success`` (which for a truncated,
        non-audited rollout is the truncation rule's guess).  That is deliberate and
        harmless: the cell model only steers allocation, so a slightly pessimistic input
        costs efficiency, never validity -- the unbiased, audit-corrected outcome is what
        goes into the estimator, never here.

        Claim: low-cost
        """
        self.n_observed += 1
        self._since_fit += 1
        added = False
        add = getattr(self.cell_model, "add", None)
        if callable(add):
            try:
                add(
                    int(rollout.task),
                    int(rollout.seed),
                    int(rollout.scaffold),
                    int(rollout.model),
                    int(rollout.success),
                )
                added = True
            except Exception:  # pragma: no cover - defensive
                logger.debug("cell_model.add failed; falling back to update", exc_info=True)
        if not added:
            update = getattr(self.cell_model, "update", None)
            if callable(update):
                update([rollout])
            else:  # pragma: no cover - a model with neither hook cannot be steered
                logger.warning("cell model exposes neither add() nor update(); not learning")

    # ------------------------------------------------------------------- strategies --

    def _select_uniform(
        self, pairs: Sequence[tuple[int, int]]
    ) -> tuple[tuple[int, int], np.ndarray]:
        """Non-adaptive control: ``pi_raw = w`` and round-robin over the supplied pairs.

        Reads neither the cell model nor the channels, so condition (C) really is
        "sequential testing without any of Quorum's design contributions".

        Claim: low-cost
        """
        pair = pairs[self._rr_index % len(pairs)]
        self._rr_index += 1
        return pair, self.w.copy()

    def _select_info_cost(
        self,
        pairs: Sequence[tuple[int, int]],
        channels: Mapping[tuple[int, int], Any] | Sequence[Any] | None,
    ) -> tuple[tuple[int, int], np.ndarray]:
        """c-optimal information-per-dollar allocation (SPEC 7.1).

        Claim: low-cost
        """
        weights = self._flip_weights(pairs, channels)
        scores = [self._pair_block_score(a, c) for (a, c) in pairs]
        utility = np.array(
            [float(w_p * (s.max() if s.size else 0.0)) for w_p, s in zip(weights, scores)],
            dtype=float,
        )
        idx = self._choose_index(utility)
        return pairs[idx], scores[idx]

    def _select_thompson(
        self,
        pairs: Sequence[tuple[int, int]],
        channels: Mapping[tuple[int, int], Any] | Sequence[Any] | None,
    ) -> tuple[tuple[int, int], np.ndarray]:
        """Posterior-sampling allocation: fund the pair whose sign is least settled.

        Two ability vectors are taken as the two halves of one posterior draw matrix; the
        disagreement rate of ``sign(theta_a - theta_c)`` between the halves is a direct
        Monte-Carlo estimate of how unsettled the pair's ordering is.  Blocks are then
        drawn proportionally to predicted contrast variance over ``sqrt(cost)``.

        Claim: low-cost
        """
        draws = self._ability_draws()
        weights = self._flip_weights(pairs, channels)
        disagree = np.empty(len(pairs), dtype=float)
        if draws is None or draws.shape[0] < 2:
            disagree[:] = weights
        else:
            half = draws.shape[0] // 2
            first, second = draws[:half], draws[half : 2 * half]
            for i, (a, c) in enumerate(pairs):
                s1 = np.sign(first[:, a] - first[:, c])
                s2 = np.sign(second[:, a] - second[:, c])
                disagree[i] = float(np.mean(s1 != s2))
            disagree = disagree * weights
        idx = int(np.argmax(disagree)) if np.any(disagree > 0.0) else self._choose_index(
            np.ones(len(pairs))
        )
        a, c = pairs[idx]
        var_b = self._block_contrast_variance(a, c)
        cost_b = self._pair_block_cost(a, c)
        raw = var_b / np.sqrt(cost_b)
        return pairs[idx], raw

    def _select_neyman(
        self,
        pairs: Sequence[tuple[int, int]],
        channels: Mapping[tuple[int, int], Any] | Sequence[Any] | None,
    ) -> tuple[tuple[int, int], np.ndarray]:
        """Classical cost-aware Neyman allocation ``pi ∝ w * sd_b / sqrt(cost_b)``.

        Included as an ablation: it minimises the variance of the *estimator* but is blind
        to which contrast is actually deciding the ranking, which is the gap ``info_cost``
        is meant to demonstrate.

        Claim: low-cost
        """
        weights = self._flip_weights(pairs, channels)
        raws: list[np.ndarray] = []
        utility = np.empty(len(pairs), dtype=float)
        for i, (a, c) in enumerate(pairs):
            sd_b = np.sqrt(self._block_contrast_variance(a, c))
            cost_b = self._pair_block_cost(a, c)
            raw = self.w * sd_b / np.sqrt(cost_b)
            raws.append(raw)
            utility[i] = float(weights[i] * raw.sum())
        idx = self._choose_index(utility)
        return pairs[idx], raws[idx]

    # ------------------------------------------------------------- scoring internals --

    def _pair_block_score(self, a: int, c: int) -> np.ndarray:
        """Information-per-dollar of buying the matched pair ``(a, c)`` at each block.

        ``score[b] = (gain[b, a] + gain[b, c]) / (cost_block[b] + eps)``.  The two cells
        are summed because the pair is executed as a matched pair on a shared block: the
        purchase is indivisible, so the cost and the value must both be block-level.
        (Summing the two rank-1 gains is the first-order value of the joint purchase; the
        exact joint update differs only by a second-order term of the same sign.)

        Claim: low-cost
        """
        key = ("score", a, c)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        gain = self._pair_block_gain(a, c)
        cost = self._pair_block_cost(a, c)
        score = gain / cost
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score = np.clip(score, 0.0, None)
        self._cache[key] = score
        return score

    def _pair_block_gain(self, a: int, c: int) -> np.ndarray:
        """Per-block c-optimal information gain for buying cells ``(b, a)`` and ``(b, c)``.

        Uses the vectorised ``cell_model.info_gain_all_cells(a, c)`` when the cell model
        provides it, and otherwise falls back to a loop over ``info_gain_for_cell``.

        Claim: low-cost
        """
        key = ("gain", a, c)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        gain: np.ndarray | None = None
        vec = getattr(self.cell_model, "info_gain_all_cells", None)
        if callable(vec):
            try:
                gain = self._coerce_gain(np.asarray(vec(a, c), dtype=float), a, c)
            except Exception:
                logger.debug("info_gain_all_cells failed; using per-cell loop", exc_info=True)
                gain = None
        if gain is None:
            gain = self._loop_gain(a, c)

        gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
        gain = np.clip(gain, 0.0, None)
        self._cache[key] = gain
        return gain

    def _coerce_gain(self, arr: np.ndarray, a: int, c: int) -> np.ndarray:
        """Reduce whatever shape the cell model returned to a per-block ``(B,)`` gain.

        Accepts ``(T, K, S, M)``, ``(B, M)``, ``(T, K, S)``, ``(B,)``, ``(B, 2)`` and the
        flattened ``(B * M,)`` forms.

        Claim: infrastructure
        """
        B, M = self.n_blocks, self.n_models
        if arr.ndim == 4:
            flat = arr.reshape(B, M)
            return flat[:, a] + flat[:, c]
        if arr.ndim == 3 and arr.size == B:
            return arr.reshape(B)
        if arr.ndim == 2:
            if arr.shape == (B, M):
                return arr[:, a] + arr[:, c]
            if arr.shape[0] == B and arr.shape[1] == 2:
                return arr[:, 0] + arr[:, 1]
            if arr.shape[0] == B:
                return arr.sum(axis=1)
        if arr.ndim == 1:
            if arr.size == B:
                return arr.astype(float, copy=True)
            if arr.size == B * M:
                flat = arr.reshape(B, M)
                return flat[:, a] + flat[:, c]
        raise ValueError(f"cannot interpret info_gain_all_cells output of shape {arr.shape}")

    def _loop_gain(self, a: int, c: int) -> np.ndarray:
        """Fallback per-block gain via ``cell_model.info_gain_for_cell`` (2B calls).

        Claim: low-cost
        """
        per_cell = getattr(self.cell_model, "info_gain_for_cell", None)
        if not callable(per_cell):
            logger.warning("cell model exposes no information-gain hook; scoring by cost only")
            return np.ones(self.n_blocks, dtype=float)
        K, S = self.spec.n_seeds, self.spec.n_scaffolds
        out = np.zeros(self.n_blocks, dtype=float)
        for b in range(self.n_blocks):
            t, k, s = unblock_index(b, K, S)
            try:
                out[b] = float(per_cell(t, k, s, a, a, c)) + float(per_cell(t, k, s, c, a, c))
            except Exception:  # pragma: no cover - defensive
                logger.debug("info_gain_for_cell failed at block %d", b, exc_info=True)
                out[b] = 0.0
        return out

    def _pair_block_cost(self, a: int, c: int) -> np.ndarray:
        """Dollar-equivalent cost of running the matched pair ``(a, c)`` at each block.

        Claim: low-cost
        """
        key = ("cost", a, c)
        cached = self._cache_static.get(key)
        if cached is not None:
            return cached
        cost = self._cell_cost[:, a] + self._cell_cost[:, c] + _COST_EPS
        cost = np.clip(np.nan_to_num(cost, nan=_COST_EPS), _COST_EPS, None)
        self._cache_static[key] = cost
        return cost

    def _block_contrast_variance(self, a: int, c: int) -> np.ndarray:
        """Model-predicted variance of the per-block contrast ``Y_a - Y_c``.

        With independent Bernoulli draws at the two cells of a block this is
        ``p_a (1 - p_a) + p_c (1 - p_c)`` -- the quantity Neyman allocation wants and the
        one Thompson sampling uses to spread blocks.

        Claim: low-cost
        """
        key = ("bvar", a, c)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        p = self._predicted_p()
        var = p[:, a] * (1.0 - p[:, a]) + p[:, c] * (1.0 - p[:, c])
        var = np.clip(np.nan_to_num(var, nan=0.25), 1e-6, None)
        self._cache[key] = var
        return var

    def _predicted_p(self) -> np.ndarray:
        """Posterior-mean success probability per (block, model), shape ``(B, M)``.

        Claim: low-cost
        """
        cached = self._cache.get("pred_p")
        if cached is not None:
            return cached
        p = np.full((self.n_blocks, self.n_models), 0.5, dtype=float)
        predict = getattr(self.cell_model, "predict_p", None)
        if callable(predict):
            try:
                arr = np.asarray(predict(), dtype=float)
                if arr.size == self.n_blocks * self.n_models:
                    p = arr.reshape(self.n_blocks, self.n_models)
                else:  # pragma: no cover - defensive
                    logger.debug("predict_p returned unexpected size %d", arr.size)
            except Exception:  # pragma: no cover - defensive
                logger.debug("predict_p failed; using 0.5 everywhere", exc_info=True)
        p = np.clip(np.nan_to_num(p, nan=0.5), 1e-6, 1.0 - 1e-6)
        self._cache["pred_p"] = p
        return p

    def _ability_draws(self) -> np.ndarray | None:
        """Cached ``(n_draws, M)`` posterior ability draws, or ``None`` if unavailable.

        Claim: low-cost
        """
        if "ability" in self._cache:
            return self._cache["ability"]
        draws: np.ndarray | None = None
        hook = getattr(self.cell_model, "ability_posterior", None)
        if callable(hook):
            try:
                arr = np.asarray(hook(self.n_draws, rng=self.rng), dtype=float)
                if arr.ndim == 2 and arr.shape[1] == self.n_models:
                    draws = arr
            except Exception:  # pragma: no cover - defensive
                logger.debug("ability_posterior failed", exc_info=True)
        self._cache["ability"] = draws
        return draws

    def _rank_posterior(self) -> np.ndarray | None:
        """Cached ``(M, M)`` matrix of ``P(theta_a > theta_c)``, or ``None``.

        Claim: low-cost
        """
        if "rankpost" in self._cache:
            return self._cache["rankpost"]
        post: np.ndarray | None = None
        hook = getattr(self.cell_model, "rank_posterior", None)
        if callable(hook):
            try:
                arr = np.asarray(hook(self.n_draws, rng=self.rng), dtype=float)
                if arr.shape == (self.n_models, self.n_models):
                    post = arr
            except Exception:  # pragma: no cover - defensive
                logger.debug("rank_posterior failed", exc_info=True)
        if post is None:
            draws = self._ability_draws()
            if draws is not None:
                diff = draws[:, :, None] - draws[:, None, :]
                post = (diff > 0.0).mean(axis=0)
        self._cache["rankpost"] = post
        return post

    def _flip_weights(
        self,
        pairs: Sequence[tuple[int, int]],
        channels: Mapping[tuple[int, int], Any] | Sequence[Any] | None,
    ) -> np.ndarray:
        """How close each pair is to flipping: ``min(P(a > c), P(c > a))`` in ``[0, 0.5]``.

        A pair whose posterior is already decisive gets a weight near zero, and a pair
        whose channel interval already excludes zero is damped by a further factor of
        1e-3, so **resolved pairs stop attracting budget** and the money moves to the
        contested frontier.  A small floor is retained so a fully resolved family still
        yields a well-defined draw.

        Claim: low-cost
        """
        post = self._rank_posterior()
        out = np.empty(len(pairs), dtype=float)
        for i, (a, c) in enumerate(pairs):
            if post is None:
                flip = 0.5
            else:
                p_ac = float(post[a, c])
                flip = float(min(max(p_ac, 0.0), max(1.0 - p_ac, 0.0)))
            if self._is_resolved(channels, pairs, i):
                flip *= _RESOLVED_DAMPING
            out[i] = max(flip, _WEIGHT_FLOOR)
        return out

    @staticmethod
    def _is_resolved(
        channels: Mapping[tuple[int, int], Any] | Sequence[Any] | None,
        pairs: Sequence[tuple[int, int]],
        index: int,
    ) -> bool:
        """Whether the channel for ``pairs[index]`` already excludes zero.

        Tolerates a mapping keyed by either pair orientation, a sequence aligned with
        ``pairs``, or no channels at all.

        Claim: infrastructure
        """
        if channels is None:
            return False
        pair = pairs[index]
        chan: Any = None
        if isinstance(channels, Mapping):
            chan = channels.get(pair)
            if chan is None:
                chan = channels.get((pair[1], pair[0]))
        else:
            try:
                if index < len(channels):
                    chan = channels[index]
            except TypeError:  # pragma: no cover - non-indexable container
                chan = None
        if chan is None:
            return False
        flag = getattr(chan, "excludes_zero", None)
        if flag is None:
            return False
        try:
            value = flag() if callable(flag) else flag
            return bool(value)
        except Exception:  # pragma: no cover - defensive
            return False

    def _choose_index(self, utility: np.ndarray) -> int:
        """Pick a pair index from non-negative utilities, deterministically given the rng.

        The rule is explicit: with ``temperature <= 0`` the choice is ``argmax`` (ties go
        to the lowest index); otherwise the index is sampled with probability
        proportional to ``(u_i / max_j u_j) ** (1 / temperature)``, so ``temperature == 1``
        is plain proportional sampling and ``temperature -> 0`` tends to the argmax.  With
        all-zero utilities the choice is uniform.  Every draw uses ``self.rng``, so a run
        is reproducible from its seed.

        Claim: low-cost
        """
        u = np.asarray(utility, dtype=float).ravel()
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        u = np.clip(u, 0.0, None)
        n = u.size
        if n == 1:
            return 0
        top = float(u.max())
        if top <= 0.0:
            return int(self.rng.integers(n))
        if self.temperature <= 0.0:
            return int(np.argmax(u))
        power = 1.0 / self.temperature
        scaled = np.power(u / top, power)
        total = float(scaled.sum())
        if not np.isfinite(total) or total <= 0.0:  # pragma: no cover - defensive
            return int(np.argmax(u))
        return int(self.rng.choice(n, p=scaled / total))

    # ------------------------------------------------------------------ maintenance --

    def _maybe_refit(self) -> None:
        """Refit the cell model at most every ``refit_every`` observations.

        Between refits every Sigma-derived quantity (gains, predicted ``p``, rank
        posterior, ability draws) is served from cache, so per-step allocation overhead
        stays small relative to running a rollout -- which is the point: an allocator that
        costs as much as the thing it is allocating saves nothing.

        Claim: low-cost
        """
        if self._fitted and self._since_fit < self.refit_every:
            return
        fit = getattr(self.cell_model, "fit", None)
        if callable(fit):
            try:
                fit()
                self.n_fits += 1
            except Exception:  # pragma: no cover - a failed fit must not stop the run
                logger.warning("cell model fit failed; reusing previous posterior", exc_info=True)
        self._fitted = True
        self._since_fit = 0
        self._cache.clear()

    def _as_cell_array(self, value: np.ndarray | float, name: str) -> np.ndarray:
        """Broadcast a cost specification to a ``(B, M)`` array of non-negative floats.

        Claim: infrastructure
        """
        B, M = self.n_blocks, self.n_models
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            out = np.full((B, M), float(arr))
        elif arr.shape == (B, M):
            out = arr.astype(float, copy=True)
        elif arr.ndim == 4 and arr.size == B * M:
            out = arr.reshape(B, M).astype(float, copy=True)
        elif arr.shape == (M,):
            out = np.tile(arr.astype(float), (B, 1))
        elif arr.size == B * M:
            out = arr.reshape(B, M).astype(float, copy=True)
        else:
            raise ValueError(
                f"{name} must broadcast to ({B}, {M}); got shape {arr.shape}"
            )
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, 0.0, None)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        """Compact description of the allocator's configuration.

        Claim: infrastructure
        """
        return (
            f"DecisionTargetedAllocator(strategy={self.strategy!r}, B={self.n_blocks}, "
            f"M={self.n_models}, pi_min_frac={self.pi_min_frac}, "
            f"lambda_sec={self.lambda_sec}, refit_every={self.refit_every})"
        )
