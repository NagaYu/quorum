"""Nested sequential testing: the anytime-valid core of Quorum.

This module is where the validity argument lives, and it is deliberately small and
self-contained.  Nothing here depends on the Bayesian cell model, the allocator, or the
control-variate machinery: those are injected as duck-typed collaborators and may be
arbitrarily wrong without endangering a single claim made below.

The nesting has two levels:

* **Outer** -- an anytime-valid empirical-Bernstein confidence sequence
  (:class:`EmpiricalBernsteinCS`) on each pairwise contrast
  ``Delta[a, c] = sum_b w[b] (p[b, a] - p[b, c])``, fed by a Horvitz-Thompson channel
  (:class:`PairChannel`).  Because a confidence sequence only requires a *bounded
  martingale difference* -- ``E[X_i | F_{i-1}] = mu``, not independence -- the block
  sampling distribution ``pi_i`` may be chosen adaptively by any predictable rule.  That
  single fact is what licenses cost-aware allocation without a penalty.
* **Inner** -- a per-rollout truncation rule (:class:`InnerTruncator`) that abandons a
  rollout once its trajectory looks doomed, plus an audit-based correction
  (:func:`corrected_outcome`) that restores exact conditional unbiasedness *for any
  ``q``*, however miscalibrated.  Miscalibration costs variance; it never costs validity.

Design-based inference: the sampling design (Horvitz-Thompson + martingale) supplies the
guarantee, and every model in the system only steers where the money goes.

Claim: coverage
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from quorum.types import GridSpec, RankDecision, Rollout, uniform_weights, unblock_index

__all__ = [
    "outcome_interval",
    "EmpiricalBernsteinCS",
    "PairChannel",
    "InnerTruncator",
    "CalibratedQ",
    "corrected_outcome",
    "outcome_range",
    "rollouts_to_resolve",
    "NestedSequentialTest",
    "UniformAllocator",
    "Allocator",
]

logger = logging.getLogger(__name__)

# Largest projection we are willing to print.  ``rollouts_to_resolve`` is explicitly a
# projection and not a guarantee, so an "effectively unbounded" answer is capped rather
# than returned as ``inf`` (which no caller can format sensibly).
_MAX_PROJECTION: int = 1_000_000_000

_TINY: float = 1e-12


# --------------------------------------------------------------------------------------
# 5.1  Empirical-Bernstein confidence sequence
# --------------------------------------------------------------------------------------


class EmpiricalBernsteinCS:
    """Predictable plug-in empirical-Bernstein confidence sequence (SPEC 5.1).

    **Validity requirement.** ``X_i`` need only be a bounded sequence with
    ``E[X_i | F_{i-1}] = mu``.  It does **not** need to be i.i.d.  This is precisely why
    adaptive allocation is allowed: the allocator may look at everything observed so far
    when choosing ``pi_i``, because ``pi_i`` is then ``F_{i-1}``-measurable and the
    Horvitz-Thompson ratio restores the same conditional mean at every step.

    Construction.  Observations are rescaled to ``Y_i = (X_i - L) / (U - L) in [0, 1]``
    and the target becomes ``m = (mu - L) / (U - L)``.  With plug-ins that use only data
    strictly before observation ``i`` (1-indexed)::

        mu_hat[i]     = (1/2 + sum_{j<i} Y_j) / i
        sigma2_hat[i] = (1/4 + sum_{j<i} (Y_j - mu_hat[j])^2) / i
        lambda[i]     = min(c_lambda, sqrt(2 log(2/alpha) / (sigma2_hat[i] i log(1+i))))
        v[i]          = 4 (Y_i - mu_hat[i])^2
        psi_E(l)      = (-log(1-l) - l) / 4

    the process ``exp(sum_i lambda[i](Y_i - m) - sum_i v[i] psi_E(lambda[i]))`` is a
    non-negative supermartingale under ``E[Y_i | F_{i-1}] = m``, so Ville's inequality
    gives, uniformly over all times simultaneously with probability ``1 - alpha``::

        center = sum_i lambda[i] Y_i / sum_i lambda[i]
        radius = (log(2/alpha) + sum_i v[i] psi_E(lambda[i])) / sum_i lambda[i]

    The reported interval is that band mapped back to the original scale, intersected
    with ``[L, U]`` (the mean of a bounded variable lives there) and with the previously
    reported interval.  Running intersection preserves validity and is what makes this a
    confidence *sequence* rather than a series of unrelated intervals.

    Args:
        alpha: Coverage level; the interval is valid uniformly in time w.p. ``1 - alpha``.
        lower: Known almost-sure lower bound ``L`` on every observation.
        upper: Known almost-sure upper bound ``U`` on every observation, ``U > L``.
        c_lambda: Cap on the betting fraction, in ``(0, 1)``.  Default ``0.5``.

    Claim: coverage
    """

    def __init__(
        self,
        alpha: float,
        lower: float,
        upper: float,
        c_lambda: float = 0.5,
    ) -> None:
        alpha = float(alpha)
        lower = float(lower)
        upper = float(upper)
        c_lambda = float(c_lambda)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("lower/upper must be finite -- the CS needs known bounds")
        if not upper > lower:
            raise ValueError(f"upper must exceed lower, got lower={lower!r} upper={upper!r}")
        if not (0.0 < c_lambda < 1.0):
            raise ValueError(f"c_lambda must be in (0, 1), got {c_lambda!r}")

        self.alpha: float = alpha
        self.lower: float = lower
        self.upper: float = upper
        self.c_lambda: float = c_lambda
        self.scale: float = upper - lower
        self.log_term: float = float(math.log(2.0 / alpha))

        # Running state (all in the rescaled [0, 1] coordinate).
        self._n: int = 0
        self._sum_y: float = 0.0
        self._sum_sq: float = 0.0  # sum_j (Y_j - mu_hat[j])^2, mu_hat predictable
        self._sum_lambda: float = 0.0
        self._sum_A: float = 0.0
        self._sum_S: float = 0.0
        self._sum_v_psi: float = 0.0

        # Reported interval, in the original scale.
        self._lo: float = lower
        self._hi: float = upper
        self._history: list[tuple[float, float]] = []
        self.n_clipped: int = 0

    # ------------------------------------------------------------------ properties --

    @property
    def n(self) -> int:
        """Number of observations pushed into the sequence so far.

        Claim: coverage
        """
        return self._n

    @property
    def interval(self) -> tuple[float, float]:
        """Current anytime-valid interval ``(lo, hi)`` for the conditional mean.

        Before any observation this is the trivial ``[L, U]``, which is honest: a bounded
        mean is known to live there without looking at data at all.

        Claim: coverage
        """
        return (self._lo, self._hi)

    @property
    def center(self) -> float:
        """Midpoint of the current interval (a convenience, not the point estimate).

        The estimator of record for a contrast is the Horvitz-Thompson sample mean held
        by :class:`PairChannel`; this is only the middle of the reported band.

        Claim: coverage
        """
        return 0.5 * (self._lo + self._hi)

    @property
    def half_width(self) -> float:
        """Half-width of the current interval, ``(hi - lo) / 2``, clipped at zero.

        Claim: coverage
        """
        return max(0.0, 0.5 * (self._hi - self._lo))

    @property
    def history(self) -> np.ndarray:
        """``(n, 2)`` array of the interval after each observation, for width-vs-n plots.

        Claim: infrastructure
        """
        if not self._history:
            return np.zeros((0, 2), dtype=float)
        return np.asarray(self._history, dtype=float)

    # --------------------------------------------------------------------- methods --

    def update(
        self,
        x: float | np.ndarray,
        lower: float | np.ndarray | None = None,
        upper: float | np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Push one observation, or a whole array of them, and return the new interval.

        The vectorised path is not a shortcut: it reproduces the sequential recursion
        exactly (the predictable plug-ins are computed from *shifted* cumulative sums, so
        observation ``i`` never sees itself) and applies the running intersection at every
        intermediate index, so ``update(arr)`` is identical to a loop of scalar updates.

        Per-observation bounds
        ----------------------
        ``lower``/``upper`` may be supplied per observation.  They must be **predictable**
        -- known from ``F_{i-1}``, before ``x_i`` is drawn -- which is exactly the case for
        a Horvitz-Thompson channel, where the range of ``Z_i`` is fixed once the allocator
        has chosen ``pi`` and the block has been sampled.

        This matters more than it looks.  With a single global range the width of the
        interval carries a term linear in ``max_i c_i``, so one rare heavy-weight step
        prices every other step.  Normalising each observation by its own range replaces
        that with a term in the *average* ``c_i``; since ``E[w_b/pi_b] = 1`` under any
        sampling distribution, an adaptive design then pays essentially nothing for its
        exploration floor.  Without this, concentrating the design is a net loss at any
        realistic sample size -- measured, not assumed.

        The algebra stays closed-form.  With ``Y_i = (X_i - L_i)/s_i`` and
        ``m_i(mu) = (mu - L_i)/s_i``, the test statistic is affine in ``mu``::

            |sum_i lam_i Y_i + sum_i lam_i L_i/s_i - mu * sum_i lam_i/s_i| <= K
            =>  mu in [(S - K)/A, (S + K)/A],   A = sum lam_i/s_i,
                                                S = sum lam_i Y_i + sum lam_i L_i/s_i

        which reduces to the usual centre +/- radius when every ``s_i`` is equal.

        Observations outside their bounds violate the boundedness premise; they are
        clipped back into range, counted in ``n_clipped`` and logged, because silently
        accepting them would invalidate the supermartingale.

        Claim: coverage
        """
        y = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        if y.size == 0:
            return self.interval
        if not np.all(np.isfinite(y)):
            raise ValueError("EmpiricalBernsteinCS.update received a non-finite observation")

        n_new = y.size
        lo_i = (
            np.full(n_new, self.lower)
            if lower is None
            else np.broadcast_to(np.asarray(lower, dtype=float).ravel(), (n_new,)).copy()
        )
        hi_i = (
            np.full(n_new, self.upper)
            if upper is None
            else np.broadcast_to(np.asarray(upper, dtype=float).ravel(), (n_new,)).copy()
        )
        if not (np.all(np.isfinite(lo_i)) and np.all(np.isfinite(hi_i))):
            raise ValueError("per-observation bounds must be finite")
        scale_i = hi_i - lo_i
        if np.any(scale_i <= 0.0):
            raise ValueError("per-observation bounds must satisfy upper > lower")

        out_of_range = int(np.count_nonzero((y < lo_i - _TINY) | (y > hi_i + _TINY)))
        if out_of_range:
            self.n_clipped += out_of_range
            logger.warning(
                "EmpiricalBernsteinCS: %d observation(s) outside their declared bounds "
                "were clipped; the declared bounds are part of the validity argument",
                out_of_range,
            )
        y = np.clip(y, lo_i, hi_i)

        # Rescale each observation by its own predictable range.
        y = (y - lo_i) / scale_i

        # 1-based global indices for this batch.
        idx = np.arange(self._n + 1, self._n + n_new + 1, dtype=float)

        # Predictable mean plug-in: uses the prefix sum strictly before each observation.
        cum_y = np.cumsum(y)
        sum_before = self._sum_y + cum_y - y
        mu_hat = (0.5 + sum_before) / idx

        # Predictable variance plug-in, from the same shifted-prefix trick.
        dev2 = (y - mu_hat) ** 2
        cum_dev2 = np.cumsum(dev2)
        sq_before = self._sum_sq + cum_dev2 - dev2
        sigma2 = (0.25 + sq_before) / idx
        sigma2 = np.maximum(sigma2, _TINY)  # structurally >= 0.25/i, but be explicit

        denom = sigma2 * idx * np.log1p(idx)
        denom = np.maximum(denom, _TINY)
        lam = np.sqrt(2.0 * self.log_term / denom)
        # lambda must live in (0, c_lambda] for psi_E to be finite and the bet to be legal.
        lam = np.clip(lam, _TINY, self.c_lambda)

        v = 4.0 * dev2
        psi = (-np.log1p(-lam) - lam) / 4.0

        # A = sum lam_i / s_i  and  S = sum lam_i Y_i + sum lam_i L_i / s_i.
        cum_A = self._sum_A + np.cumsum(lam / scale_i)
        cum_S = self._sum_S + np.cumsum(lam * y + lam * lo_i / scale_i)
        cum_lambda = self._sum_lambda + np.cumsum(lam)
        cum_v_psi = self._sum_v_psi + np.cumsum(v * psi)

        K = self.log_term + cum_v_psi
        safe_A = np.maximum(cum_A, _TINY)
        lo = np.clip((cum_S - K) / safe_A, self.lower, self.upper)
        hi = np.clip((cum_S + K) / safe_A, self.lower, self.upper)

        # Running intersection with everything reported before, including within-batch.
        lo_run = np.maximum.accumulate(lo)
        hi_run = np.minimum.accumulate(hi)
        lo_run = np.maximum(lo_run, self._lo)
        hi_run = np.minimum(hi_run, self._hi)

        self._history.extend(zip(lo_run.tolist(), hi_run.tolist()))
        self._lo = float(lo_run[-1])
        self._hi = float(hi_run[-1])

        # Commit the running sums.
        self._n += n_new
        self._sum_y = float(self._sum_y + cum_y[-1])
        self._sum_sq = float(self._sum_sq + cum_dev2[-1])
        self._sum_lambda = float(cum_lambda[-1])
        self._sum_A = float(cum_A[-1])
        self._sum_S = float(cum_S[-1])
        self._sum_v_psi = float(cum_v_psi[-1])

        return self.interval

    def excludes(self, value: float = 0.0) -> bool:
        """Whether the current interval strictly excludes ``value`` (default zero).

        Excluding zero on a contrast is the event that certifies an ordering of the two
        models involved; because the interval is anytime-valid, this may be checked after
        every single observation with no correction for peeking.

        Claim: coverage
        """
        return bool(self._lo > value or self._hi < value)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"EmpiricalBernsteinCS(alpha={self.alpha:g}, n={self._n}, "
            f"interval=({self._lo:.4g}, {self._hi:.4g}))"
        )


# --------------------------------------------------------------------------------------
# 5.2  Horvitz-Thompson pair channel
# --------------------------------------------------------------------------------------


class PairChannel:
    """Horvitz-Thompson channel carrying one pairwise contrast into a confidence sequence.

    At step ``i`` a *predictable* distribution ``pi_i`` over blocks (chosen however the
    allocator likes, but before seeing step ``i``'s outcome) is used to draw a block
    ``b_i``; both models are run on that same block and their corrected outcomes
    ``Ytil_a, Ytil_c`` are combined into::

        Z_i = (w[b_i] / pi_i[b_i]) * (Ytil_a - Ytil_c)

    whose conditional mean is the estimand itself::

        E[Z_i | F_{i-1}] = sum_b pi_i[b] * (w[b]/pi_i[b]) * (p[b,a] - p[b,c])
                         = sum_b w[b] (p[b,a] - p[b,c])
                         = Delta[a, c]

    valid for **any** predictable ``pi_i`` with ``pi_i[b] > 0`` wherever ``w[b] > 0``.
    That identity is the whole reason adaptive, cost-aware allocation is legal here: the
    design reweighting undoes whatever the allocator did, so a greedy or even adversarial
    allocation rule changes only the variance.

    Boundedness, which the confidence sequence does require, comes from the exploration
    floor: ``|w[b]/pi_i[b]| <= w_max/pi_min =: G`` and ``|Ytil| <= R``, hence
    ``Z_i in [-2GR, +2GR]``.  The floor is the allocator's job, but this class
    **verifies** it on every observation and raises rather than accept an unbounded
    importance weight -- an unbounded weight does not merely widen the interval, it
    destroys the supermartingale the coverage guarantee rests on.

    Args:
        pair: Ordered model pair ``(a, c)``; the channel estimates ``Delta[a, c]``.
        w: Target block weights, length ``B``, non-negative, summing to 1.  Fixed.
        alpha: Per-pair level (already Bonferroni-split by the driver).
        pi_min: Smallest sampling probability the allocator promised for any block.
        outcome_range: ``R`` from :func:`outcome_range`.
        cuped: Optional control-variate adjuster with ``adjust(pair, z, cov)`` and
            ``observe(pair, z, cov)``; ``adjust`` must be non-anticipative.
        cuped_slack: Bound inflation applied when ``cuped`` is not None, since a
            control-variate-adjusted ``Z`` has a wider support than ``Z`` itself.

    Claim: coverage
    """

    def __init__(
        self,
        pair: tuple[int, int],
        w: np.ndarray,
        alpha: float,
        *,
        pi_min: float,
        outcome_range: float,
        cuped: Any | None = None,
        cuped_slack: float = 1.35,
        ht_bound: float | None = None,
        outcome_lo: float | None = None,
        outcome_hi: float | None = None,
        cov_abs_max: np.ndarray | None = None,
        cov_means: np.ndarray | None = None,
    ) -> None:
        a, c = int(pair[0]), int(pair[1])
        if a == c:
            raise ValueError(f"PairChannel needs two distinct models, got {pair!r}")
        w_arr = np.asarray(w, dtype=float).ravel()
        if w_arr.size == 0:
            raise ValueError("w must be non-empty")
        if np.any(w_arr < 0.0) or not np.all(np.isfinite(w_arr)):
            raise ValueError("w must be finite and non-negative")
        total = float(w_arr.sum())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"w must sum to 1, got {total!r}")
        pi_min = float(pi_min)
        if not (0.0 < pi_min <= 1.0):
            raise ValueError(f"pi_min must be in (0, 1], got {pi_min!r}")
        outcome_range = float(outcome_range)
        if not (outcome_range > 0.0) or not math.isfinite(outcome_range):
            raise ValueError(f"outcome_range must be finite and positive, got {outcome_range!r}")
        cuped_slack = float(cuped_slack)
        if not (cuped_slack >= 1.0):
            raise ValueError("cuped_slack must be >= 1")

        self.pair: tuple[int, int] = (a, c)
        self.w: np.ndarray = w_arr
        self.n_blocks: int = int(w_arr.size)
        self.alpha: float = float(alpha)
        self.pi_min: float = pi_min
        self.outcome_range: float = outcome_range
        self.cuped = cuped

        self.w_max: float = float(w_arr.max())
        # The Horvitz-Thompson ratio bound G = max_b w[b]/pi[b].  An allocator that mixes
        # to the floor as pi = (1-eps)*pi_raw + eps*w guarantees pi[b] >= eps*w[b] and
        # hence G <= 1/eps *per block*, which is strictly tighter than the generic
        # w_max/pi_min whenever w is non-uniform.  Allocators that know their own bound
        # declare it; otherwise fall back to the generic (valid but looser) form.
        self.G: float = float(ht_bound) if ht_bound is not None else self.w_max / pi_min
        if not (self.G > 0.0) or not math.isfinite(self.G):
            raise ValueError(f"ht_bound must be finite and positive, got {ht_bound!r}")

        # Tight range for the *difference* of two corrected outcomes.  Both Ytil live in
        # [outcome_lo, outcome_hi], so their difference lives in +/- (hi - lo).  Declaring
        # 2*R with R = max|Ytil| instead is valid but up to 2x too wide, and the width of
        # an empirical-Bernstein CS is linear in the declared range once the betting
        # fraction saturates -- so slack here is paid for directly in rollouts.
        if outcome_lo is not None and outcome_hi is not None:
            lo, hi = float(outcome_lo), float(outcome_hi)
            if not (hi > lo):
                raise ValueError(f"outcome_hi must exceed outcome_lo, got ({lo}, {hi})")
            width = hi - lo
        else:
            width = 2.0 * outcome_range
        self.outcome_width: float = float(width)
        self.outcome_lo_hi: tuple[float, float] = (
            (float(outcome_lo), float(outcome_hi))
            if outcome_lo is not None and outcome_hi is not None
            else (-float(outcome_range), float(outcome_range))
        )

        self._cuped_slack: float = cuped_slack if cuped is not None else 1.0
        bound = self.G * width * self._cuped_slack
        self.bound: float = float(bound)
        self.n_cuped_clipped: int = 0
        self.cov_abs_max: np.ndarray | None = (
            None if cov_abs_max is None else np.abs(np.asarray(cov_abs_max, float).ravel())
        )
        self.cov_means_vec: np.ndarray | None = (
            None if cov_means is None else np.asarray(cov_means, float).ravel()
        )

        self.cs: EmpiricalBernsteinCS = EmpiricalBernsteinCS(self.alpha, -self.bound, self.bound)
        self._z: list[float] = []
        self._z_raw: list[float] = []
        self._sum_z: float = 0.0
        self._sum_z2: float = 0.0

    # ------------------------------------------------------------------ properties --

    @property
    def n(self) -> int:
        """Number of paired observations pushed into this channel.

        One observation costs two rollouts (one per model on the shared block).

        Claim: low-cost
        """
        return len(self._z)

    @property
    def interval(self) -> tuple[float, float]:
        """Current anytime-valid interval for ``Delta[a, c]``.

        Claim: coverage
        """
        return self.cs.interval

    @property
    def half_width(self) -> float:
        """Half-width of the current interval on ``Delta[a, c]``.

        Claim: coverage
        """
        return self.cs.half_width

    @property
    def z_history(self) -> np.ndarray:
        """All ``Z_i`` actually pushed into the confidence sequence (post-adjustment).

        Claim: variance-reduction
        """
        return np.asarray(self._z, dtype=float)

    @property
    def z_raw_history(self) -> np.ndarray:
        """The unadjusted Horvitz-Thompson terms, for measuring the CUPED variance ratio.

        Claim: variance-reduction
        """
        return np.asarray(self._z_raw, dtype=float)

    @property
    def estimate(self) -> float:
        """Point estimate of ``Delta[a, c]``: the running mean of the pushed ``Z_i``.

        Unbiased for any predictable allocation; zero before any data, which makes the
        initial ranking the identity order rather than an accident of floating point.

        Claim: coverage
        """
        if not self._z:
            return 0.0
        return self._sum_z / len(self._z)

    @property
    def excludes_zero(self) -> bool:
        """Whether this pair is certified: the interval on ``Delta[a, c]`` excludes 0.

        Claim: coverage
        """
        return self.cs.excludes(0.0)

    # --------------------------------------------------------------------- methods --

    def observe(
        self,
        block: int,
        pi_b: float,
        ytil_a: float,
        ytil_c: float,
        cov: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Fold one paired observation into the channel and return the new interval.

        ``pi_b`` is the probability with which the allocator drew ``block`` at this step.
        It is validated against ``pi_min``: an importance weight is only bounded because
        somebody promised a floor, and a promise that is never checked is not a bound.

        Claim: coverage
        """
        b = int(block)
        if not (0 <= b < self.n_blocks):
            raise ValueError(f"block {b} out of range for {self.n_blocks} blocks")
        pi_b = float(pi_b)
        if not math.isfinite(pi_b) or pi_b <= 0.0:
            raise ValueError(f"pi[{b}] must be strictly positive and finite, got {pi_b!r}")
        if pi_b < self.pi_min * (1.0 - 1e-9):
            raise ValueError(
                f"pi[{b}] = {pi_b!r} is below the declared floor pi_min = {self.pi_min!r}; "
                "the Horvitz-Thompson weight would be unbounded and the confidence "
                "sequence would lose its guarantee"
            )
        ytil_a = float(ytil_a)
        ytil_c = float(ytil_c)
        if not math.isfinite(ytil_a) or not math.isfinite(ytil_c):
            raise ValueError("corrected outcomes must be finite")

        ht = self.w[b] / pi_b
        z_raw = ht * (ytil_a - ytil_c)
        z = z_raw

        # Compute the adjustment bound ONCE, from the phi that `adjust` will use --
        # i.e. before `cuped.observe` advances phi to phi_i. Sizing the confidence
        # sequence against the *post-update* phi is a real bug: phi can shrink, leaving
        # the realised z outside its own declared range and voiding the supermartingale.
        cap = self._adjustment_cap(ht) if (self.cuped is not None and cov is not None) else 0.0

        if self.cuped is not None and cov is not None:
            cov_vec = np.asarray(cov, dtype=float).ravel()
            adjust = getattr(self.cuped, "adjust", None)
            if callable(adjust):
                adjusted = float(adjust(self.pair, z_raw, cov_vec))
                if math.isfinite(adjusted):
                    # Cap the control-variate correction at a *predictable* multiple of
                    # this step's own range, so that |z| <= slack * ht * width holds by
                    # construction rather than by hope.  Without this the adjusted value
                    # can leave the range the confidence sequence was sized against, and
                    # an out-of-range observation does not merely widen the interval --
                    # it voids the supermartingale the guarantee rests on.
                    delta = adjusted - z_raw
                    if abs(delta) > cap:
                        delta = math.copysign(cap, delta)
                        self.n_cuped_clipped += 1
                    z = z_raw + delta
                else:  # pragma: no cover - defensive
                    logger.warning("CUPED adjustment produced a non-finite value; using raw Z")
            observe = getattr(self.cuped, "observe", None)
            if callable(observe):
                # Strictly after adjusting, so phi stays F_{i-1}-measurable.
                observe(self.pair, z_raw, cov_vec)

        self._z_raw.append(float(z_raw))
        self._z.append(float(z))
        self._sum_z += float(z)
        self._sum_z2 += float(z) * float(z)

        # Per-observation range. ``ht`` is predictable -- ``w[b]`` is fixed and ``pi_b``
        # was chosen by the allocator before the block was drawn -- so this is a legal
        # F_{i-1}-measurable bound, and it is the *realised* weight rather than the
        # worst case the design allows. See EmpiricalBernsteinCS.update.
        step = ht * self.outcome_width + cap
        step = max(step, _TINY)
        return self.cs.update(z, -step, step)

    def _adjustment_cap(self, ht_total: float) -> float:
        """Exact predictable bound on ``|phi^T (C_i - E[C])|`` for this step.

        Every factor is ``F_{i-1}``-measurable: ``phi`` is fit on observations before this
        one, ``ht_total`` follows from the allocator's ``pi`` (fixed before the block was
        drawn), and the covariate columns are bounded over the whole grid in advance::

            |phi^T (C_i - E[C])| <= sum_j |phi_j| * (ht_total * max_b |c_j(b)| + |E[C_j]|)

        Using this instead of an arbitrary multiplicative slack means the confidence
        sequence is sized against a quantity the adjustment provably cannot exceed, so the
        magnitude clip never binds and the bounded-conservatism argument for clipping is
        not needed at all.  Falls back to the ``cuped_slack`` heuristic when the covariate
        scale was not supplied.

        Claim: variance-reduction
        """
        if self.cov_abs_max is None or self.cuped is None:
            return (self._cuped_slack - 1.0) * ht_total * self.outcome_width
        phi_fn = getattr(self.cuped, "phi", None)
        if not callable(phi_fn):
            return (self._cuped_slack - 1.0) * ht_total * self.outcome_width
        try:
            phi = np.abs(np.asarray(phi_fn(self.pair), dtype=float).ravel())
        except Exception:  # pragma: no cover - defensive
            return (self._cuped_slack - 1.0) * ht_total * self.outcome_width
        if phi.size != self.cov_abs_max.size:
            return (self._cuped_slack - 1.0) * ht_total * self.outcome_width
        means = (
            np.zeros_like(phi) if self.cov_means_vec is None
            else np.abs(self.cov_means_vec)
        )
        return float(np.sum(phi * (float(ht_total) * self.cov_abs_max + means)))

    def observe_unpaired(
        self,
        block_a: int,
        pi_a: float,
        ytil_a: float,
        block_c: int,
        pi_c: float,
        ytil_c: float,
        cov: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Fold in one observation whose two rollouts came from *different* instances.

        ``Z_i = (w[b_a]/pi[b_a]) Ytil_a - (w[b_c]/pi[b_c]) Ytil_c``.  Each term is
        separately unbiased for its model's ``theta`` under any predictable ``pi``, so
        ``E[Z_i | F_{i-1}] = Delta[a, c]`` exactly as in the paired case -- but the shared
        instance difficulty no longer cancels between the two terms, so between-block
        variance now dominates.  That is the regime logged evaluation data actually lives
        in, and the one where a known-mean control variate has something large to remove.

        Claim: variance-reduction
        """
        for b, pi_b in ((int(block_a), float(pi_a)), (int(block_c), float(pi_c))):
            if not (0 <= b < self.n_blocks):
                raise ValueError(f"block {b} out of range for {self.n_blocks} blocks")
            if not math.isfinite(pi_b) or pi_b <= 0.0:
                raise ValueError(f"pi[{b}] must be strictly positive and finite")
            if pi_b < self.pi_min * (1.0 - 1e-9):
                raise ValueError(
                    f"pi[{b}] = {pi_b!r} is below the declared floor pi_min = "
                    f"{self.pi_min!r}; the Horvitz-Thompson weight would be unbounded"
                )
        ht_a = self.w[int(block_a)] / float(pi_a)
        ht_c = self.w[int(block_c)] / float(pi_c)
        z_raw = ht_a * float(ytil_a) - ht_c * float(ytil_c)
        z = z_raw

        cap = (
            self._adjustment_cap(ht_a + ht_c)
            if (self.cuped is not None and cov is not None)
            else 0.0
        )

        if self.cuped is not None and cov is not None:
            cov_vec = np.asarray(cov, dtype=float).ravel()
            adjust = getattr(self.cuped, "adjust", None)
            if callable(adjust):
                adjusted = float(adjust(self.pair, z_raw, cov_vec))
                if math.isfinite(adjusted):
                    delta = adjusted - z_raw
                    if abs(delta) > cap:
                        delta = math.copysign(cap, delta)
                        self.n_cuped_clipped += 1
                    z = z_raw + delta
            observe = getattr(self.cuped, "observe", None)
            if callable(observe):
                observe(self.pair, z_raw, cov_vec)

        self._z_raw.append(float(z_raw))
        self._z.append(float(z))
        self._sum_z += float(z)
        self._sum_z2 += float(z) * float(z)

        half = max(abs(self.outcome_lo_hi[0]), abs(self.outcome_lo_hi[1]))
        step = (ht_a + ht_c) * half + cap
        step = max(step, _TINY)
        return self.cs.update(z, -step, step)

    def variance(self) -> float:
        """Sample variance of the pushed ``Z_i`` (0 with fewer than two observations).

        Claim: variance-reduction
        """
        n = len(self._z)
        if n < 2:
            return 0.0
        mean = self._sum_z / n
        var = (self._sum_z2 - n * mean * mean) / (n - 1)
        return float(max(0.0, var))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        lo, hi = self.interval
        return f"PairChannel(pair={self.pair}, n={self.n}, interval=({lo:.4g}, {hi:.4g}))"


# --------------------------------------------------------------------------------------
# 5.3  Inner truncation and its calibrated q-model
# --------------------------------------------------------------------------------------


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic function.

    Claim: infrastructure
    """
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def _logit(p: float) -> float:
    """Inverse logistic on a strictly interior probability.

    Claim: infrastructure
    """
    p = float(min(max(p, 1e-6), 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def _fire_prefix_end(r: Rollout) -> int:
    """Length of the trajectory prefix the truncation rule actually saw when it fired.

    For a truncated, non-audited rollout this is simply the number of executed steps.
    For an *audited* rollout the trajectory continues past the firing point, so the
    prefix must be cut at ``truncated_at`` -- conditioning ``q`` on the post-firing tail
    would leak the outcome into the very quantity the audit is meant to correct.

    Claim: unbiased-truncation
    """
    traj = getattr(r, "trajectory", None)
    if traj is None:
        return 0
    length = int(np.asarray(traj).size)
    if length == 0:
        return 0
    cut = int(getattr(r, "truncated_at", -1))
    if cut >= 0:
        end = cut + 1
    else:
        end = int(getattr(r, "n_steps", length))
    return int(min(max(end, 1), length))


class CalibratedQ:
    """Calibrated ``P(success | the rule fired here)``, fit on the audited subsample only.

    The model is a two-parameter logistic in the running-mean statistic of the trajectory
    prefix -- the same statistic the truncation rule thresholds -- shrunk towards a prior
    so that ``predict`` is well defined *before any audit has landed*.  Concretely it
    maximises the Bernoulli log-likelihood penalised by
    ``0.5 * prior_strength * ||beta - beta0||^2`` with ``beta0 = (logit(prior_mean), 0)``,
    which with no data returns exactly ``prior_mean``.

    Only audited rollouts may be fed in.  A non-audited truncated rollout has no observed
    outcome at all (its recorded ``success`` is the rule's guess), so training on it would
    make ``q`` a self-fulfilling prophecy.  Note carefully that this would not break the
    estimator -- :func:`corrected_outcome` is unbiased for *any* ``q`` -- but it would
    inflate the residual ``E[(X - q)^2 | fired]`` that drives the variance penalty, which
    is the only thing ``q`` is here to shrink.

    Args:
        prior_mean: Prior belief for ``P(success | fired)``; the rule fires on evidence
            of failure, so a small value is the honest default.
        prior_strength: Ridge weight pulling the fit back to the prior.
        max_iter: Newton iterations.
        tol: Convergence tolerance on the coefficient step.
        eps: Clip applied to returned probabilities.

    Claim: variance-reduction
    """

    def __init__(
        self,
        *,
        prior_mean: float = 0.10,
        prior_strength: float = 5.0,
        max_iter: int = 50,
        tol: float = 1e-8,
        eps: float = 1e-4,
    ) -> None:
        prior_mean = float(prior_mean)
        if not (0.0 < prior_mean < 1.0):
            raise ValueError(f"prior_mean must be in (0, 1), got {prior_mean!r}")
        if not (float(prior_strength) > 0.0):
            raise ValueError("prior_strength must be positive so q is defined before any audit")
        if int(max_iter) < 1:
            raise ValueError("max_iter must be >= 1")
        if not (0.0 < float(eps) < 0.5):
            raise ValueError("eps must be in (0, 0.5)")

        self.prior_mean: float = prior_mean
        self.prior_strength: float = float(prior_strength)
        self.max_iter: int = int(max_iter)
        self.tol: float = float(tol)
        self.eps: float = float(eps)

        self._beta0: np.ndarray = np.array([_logit(prior_mean), 0.0], dtype=float)
        self.beta: np.ndarray = self._beta0.copy()
        self._stats: list[float] = []
        self._ys: list[int] = []
        self._centre: float = 0.0
        self._scale: float = 1.0
        self._dirty: bool = False

    # ------------------------------------------------------------------ properties --

    @property
    def n_audited(self) -> int:
        """Number of audited, truncated rollouts this model has been trained on.

        Claim: variance-reduction
        """
        return len(self._ys)

    # --------------------------------------------------------------------- methods --

    def observe(self, stat: float, success: int) -> None:
        """Record one audited observation: the firing statistic and the true outcome.

        Claim: variance-reduction
        """
        stat = float(stat)
        if not math.isfinite(stat):
            raise ValueError("calibration statistic must be finite")
        y = int(success)
        if y not in (0, 1):
            raise ValueError(f"success must be 0 or 1, got {success!r}")
        self._stats.append(stat)
        self._ys.append(y)
        self._dirty = True

    def observe_rollout(self, r: Rollout) -> bool:
        """Feed a rollout if and only if it was truncated *and* audited; else ignore it.

        Returns whether the rollout was used, so callers can log the audit yield.

        Claim: variance-reduction
        """
        if getattr(r, "completed", True):
            return False
        if not getattr(r, "audited", False):
            return False
        end = _fire_prefix_end(r)
        if end <= 0:
            return False
        traj = np.asarray(r.trajectory, dtype=float).ravel()
        self.observe(float(traj[:end].mean()), int(r.success))
        return True

    def fit(self) -> None:
        """Refit the penalised logistic by Newton iterations; a no-op when up to date.

        Claim: variance-reduction
        """
        if not self._dirty:
            return
        self._dirty = False
        n = len(self._ys)
        if n == 0:
            self.beta = self._beta0.copy()
            return

        stats = np.asarray(self._stats, dtype=float)
        self._centre = float(stats.mean())
        scale = float(stats.std())
        self._scale = scale if scale > 1e-6 else 1.0

        x1 = np.clip((stats - self._centre) / self._scale, -6.0, 6.0)
        design = np.column_stack([np.ones(n, dtype=float), x1])
        y = np.asarray(self._ys, dtype=float)

        beta = self._beta0.copy()
        ridge_eye = self.prior_strength * np.eye(2, dtype=float)
        for _ in range(self.max_iter):
            eta = np.clip(design @ beta, -30.0, 30.0)
            p = np.asarray(_sigmoid(eta), dtype=float)
            wgt = np.maximum(p * (1.0 - p), 1e-8)
            grad = design.T @ (y - p) - self.prior_strength * (beta - self._beta0)
            hess = design.T @ (design * wgt[:, None]) + ridge_eye
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]
            if not np.all(np.isfinite(step)):  # pragma: no cover - defensive
                logger.warning("CalibratedQ Newton step was non-finite; keeping previous fit")
                break
            beta = beta + step
            if float(np.max(np.abs(step))) < self.tol:
                break
        self.beta = beta

    def predict(self, stat: float) -> float:
        """Predicted ``P(success | fired at this statistic)``, clipped into ``(0, 1)``.

        Claim: variance-reduction
        """
        self.fit()
        s = float(stat)
        if not math.isfinite(s):
            s = self._centre
        x1 = float(np.clip((s - self._centre) / self._scale, -6.0, 6.0))
        eta = float(np.clip(self.beta[0] + self.beta[1] * x1, -30.0, 30.0))
        p = float(_sigmoid(eta))
        return float(min(max(p, self.eps), 1.0 - self.eps))

    def q(self, traj_prefix: np.ndarray, step: int) -> float:
        """``predict`` on the running mean of ``traj_prefix`` -- matches ``InnerTruncator.q``.

        Claim: variance-reduction
        """
        prefix = np.asarray(traj_prefix, dtype=float).ravel()
        if prefix.size == 0:
            return self.predict(self._centre)
        return self.predict(float(prefix.mean()))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CalibratedQ(n_audited={self.n_audited}, prior_mean={self.prior_mean:g})"


class InnerTruncator:
    """Sequentially decides, from a partial trajectory, that this rollout will fail.

    The rule is deliberately crude -- fire when ``step >= min_steps`` and the running mean
    of the trajectory prefix has fallen to ``threshold`` -- because the correction in
    :func:`corrected_outcome` makes its accuracy an efficiency question rather than a
    correctness question.  A sharper rule saves more compute and adds less variance; a
    broken rule wastes money; neither can bias the outer estimator.

    The audit coin is drawn **at the moment the rule fires and independently of the
    outcome**.  Independence is not decoration: the correction divides by ``rho``, and if
    auditing were correlated with success the reweighted term would no longer have the
    right conditional mean.

    Args:
        threshold: Fire when the prefix running mean is at or below this.
        min_steps: Never fire before this many steps have been executed.
        audit_rate: ``rho``, the probability a fired rollout is run to completion anyway.
        q_hat: Constant fallback for ``P(success | fired)`` when ``calibrated`` is None.
        calibrated: Optional :class:`CalibratedQ` (or anything with ``predict``).
        rng: Default generator for :meth:`audit` when the caller passes none.

    Claim: unbiased-truncation
    """

    def __init__(
        self,
        *,
        threshold: float = -1.0,
        min_steps: int = 3,
        audit_rate: float = 0.25,
        q_hat: float = 0.0,
        q_max: float = 0.25,
        calibrated: Any | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        audit_rate = float(audit_rate)
        if not (0.0 <= audit_rate <= 1.0):
            raise ValueError(f"audit_rate must be in [0, 1], got {audit_rate!r}")
        if audit_rate == 0.0:
            logger.warning(
                "audit_rate=0 leaves the truncation correction with no audit subsample; "
                "the outer estimator then reports the rule's own guess q"
            )
        if int(min_steps) < 1:
            raise ValueError("min_steps must be >= 1")
        q_hat = float(q_hat)
        if not (0.0 <= q_hat <= 1.0):
            raise ValueError(f"q_hat must be a probability in [0, 1], got {q_hat!r}")
        q_max = float(q_max)
        if not (0.0 <= q_max <= 1.0):
            raise ValueError(f"q_max must be a probability in [0, 1], got {q_max!r}")
        if q_hat > q_max:
            raise ValueError(
                f"q_hat={q_hat!r} exceeds q_max={q_max!r}: a constant guess above the cap "
                "would make the rule decline to fire on every rollout"
            )

        self.threshold: float = float(threshold)
        self.min_steps: int = int(min_steps)
        self.audit_rate: float = audit_rate
        self.q_hat: float = q_hat
        self.q_max: float = q_max
        self.calibrated = calibrated
        self.rng: np.random.Generator | None = rng

        self.n_fired: int = 0
        self.n_audited: int = 0
        self.n_declined: int = 0

    # --------------------------------------------------------------------- methods --

    def should_stop(self, traj_prefix: np.ndarray, step: int) -> bool:
        """Whether to abandon the rollout given the prefix observed after ``step`` steps.

        Claim: unbiased-truncation
        """
        if int(step) < self.min_steps:
            return False
        prefix = np.asarray(traj_prefix, dtype=float).ravel()
        if prefix.size == 0:
            return False
        mean = float(prefix.mean())
        if not math.isfinite(mean):
            return False
        fired = bool(mean <= self.threshold)
        if fired and self.q(prefix, int(step)) > self.q_max:
            # Confidence gate. The declared confidence-sequence range depends on q_max
            # (see outcome_interval), so a rule that fired while believing success was
            # still plausible would invalidate the bound it was sized against -- and it
            # would be a bad trade anyway, paying full CS width to save a few steps.
            # Declining to fire is always safe: the rollout simply runs to completion.
            self.n_declined += 1
            return False
        if fired:
            self.n_fired += 1
        return fired

    def audit(self, rng: np.random.Generator | None = None) -> bool:
        """Draw the independent ``Bernoulli(audit_rate)`` coin that spares this rollout.

        Independent of the outcome by construction: it consults only the generator, never
        the trajectory, and is called at the firing instant.

        Claim: unbiased-truncation
        """
        gen = rng if rng is not None else self.rng
        if gen is None:
            raise ValueError(
                "InnerTruncator.audit needs an explicit rng (or one supplied at construction) "
                "-- the audit coin is part of the design and must be reproducible"
            )
        if self.audit_rate <= 0.0:
            return False
        if self.audit_rate >= 1.0:
            self.n_audited += 1
            return True
        drawn = bool(gen.random() < self.audit_rate)
        if drawn:
            self.n_audited += 1
        return drawn

    def q(self, traj_prefix: np.ndarray, step: int) -> float:
        """Predicted ``P(success | the rule stopped here)`` for the correction term.

        Claim: unbiased-truncation
        """
        if self.calibrated is None:
            return self.q_hat
        prefix = np.asarray(traj_prefix, dtype=float).ravel()
        predict = getattr(self.calibrated, "predict", None)
        if prefix.size == 0 or not callable(predict):
            q_fn = getattr(self.calibrated, "q", None)
            if callable(q_fn):
                value = float(q_fn(prefix, int(step)))
                return float(min(max(value, 0.0), 1.0))
            return self.q_hat
        value = float(predict(float(prefix.mean())))
        if not math.isfinite(value):  # pragma: no cover - defensive
            return self.q_hat
        return float(min(max(value, 0.0), 1.0))

    def q_for_rollout(self, r: Rollout) -> float:
        """``q`` evaluated on the prefix that actually triggered this rollout's truncation.

        Claim: unbiased-truncation
        """
        end = _fire_prefix_end(r)
        if end <= 0:
            return self.q_hat if self.calibrated is None else self.q(np.zeros(0), 0)
        traj = np.asarray(r.trajectory, dtype=float).ravel()
        return self.q(traj[:end], end)

    def observe(self, r: Rollout) -> None:
        """Update the calibrated ``q`` model, using audited rollouts and nothing else.

        Claim: variance-reduction
        """
        if self.calibrated is None:
            return
        sink = getattr(self.calibrated, "observe_rollout", None)
        if callable(sink):
            sink(r)
            return
        if getattr(r, "completed", True) or not getattr(r, "audited", False):
            return
        end = _fire_prefix_end(r)
        if end <= 0:
            return
        raw = getattr(self.calibrated, "observe", None)
        if callable(raw):
            traj = np.asarray(r.trajectory, dtype=float).ravel()
            raw(float(traj[:end].mean()), int(r.success))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"InnerTruncator(threshold={self.threshold:g}, min_steps={self.min_steps}, "
            f"audit_rate={self.audit_rate:g}, fired={self.n_fired}, audited={self.n_audited})"
        )


# --------------------------------------------------------------------------------------
# 5.4  The correction
# --------------------------------------------------------------------------------------


def corrected_outcome(r: Rollout, q: float, audit_rate: float) -> float:
    """Unbiased outcome for a possibly-truncated rollout (SPEC 5.4).

    ::

        if r.completed:                          # the rule never fired
            Ytil = r.success
        else:                                    # the rule fired
            Ytil = q + (1{r.audited} / rho) * (r.success - q)

    Proof of conditional unbiasedness, for **any** ``q`` whatsoever, correct or not::

        E[Ytil | fired, features] = q + (E[1{audited}] / rho) * (E[X | features] - q)
                                  = q + (rho / rho) * (E[X | features] - q) = E[X | features]

    (the audit coin is independent of the outcome given the firing event, so the
    expectation factorises).  Hence a miscalibrated truncation rule costs **variance**,
    never validity: the excess is ``(1 - rho)/rho * E[(X - q)^2 | fired]``, which vanishes
    as ``q`` approaches the truth and is why a confident rule buys compute savings almost
    for free.

    Args:
        r: The rollout, carrying ``completed``/``audited``/``success``.
        q: The rule's predicted ``P(success | fired)``; arbitrary, see above.
        audit_rate: ``rho``, the probability with which a fired rollout was audited.

    Claim: unbiased-truncation
    """
    rho = float(audit_rate)
    if not (0.0 <= rho <= 1.0) or not math.isfinite(rho):
        raise ValueError(f"audit_rate must be in [0, 1], got {audit_rate!r}")
    q = float(q)
    if not math.isfinite(q):
        raise ValueError(f"q must be finite, got {q!r}")

    if bool(getattr(r, "completed", True)):
        return float(r.success)

    audited = bool(getattr(r, "audited", False))
    if not audited:
        return q
    if rho <= 0.0:
        raise ValueError(
            "a rollout is marked audited but audit_rate is 0; the audit coin and the "
            "correction disagree, which would silently break unbiasedness"
        )
    return q + (float(r.success) - q) / rho


def outcome_range(audit_rate: float, any_truncation: bool) -> float:
    """The bound ``R`` with ``|Ytil| <= R``, used to size the confidence sequence.

    Without truncation ``Ytil = success in [0, 1]`` so ``R = 1``.  With truncation the
    audit reweighting can reach ``1 + 1/rho`` in magnitude, so ``R = max(1, 1 + 1/rho)``.
    A small audit rate therefore widens the interval -- the honest price of correcting an
    aggressive truncation rule, and the reason ``rho`` is a tuning knob rather than a
    number to drive to zero.  With ``rho = 0`` no audit ever fires, ``Ytil = q in [0, 1]``
    and ``R = 1`` again (at the cost of an estimator that is only as good as ``q``).

    Claim: unbiased-truncation
    """
    rho = float(audit_rate)
    if not (0.0 <= rho <= 1.0) or not math.isfinite(rho):
        raise ValueError(f"audit_rate must be in [0, 1], got {audit_rate!r}")
    if not any_truncation:
        return 1.0
    if rho <= 0.0:
        return 1.0
    return float(max(1.0, 1.0 + 1.0 / rho))


def outcome_interval(
    audit_rate: float,
    any_truncation: bool,
    q_max: float = 1.0,
) -> tuple[float, float]:
    """Exact reachable interval ``[lo, hi]`` of ``Ytil``, given a capped ``q``.

    Enumerating the three branches of :func:`corrected_outcome` with ``X in {0, 1}`` and
    ``q in [0, q_max]``::

        completed              Ytil = X                      in [0, 1]
        fired, not audited     Ytil = q                      in [0, q_max]
        fired, audited, X=0    Ytil = -q(1-rho)/rho          in [-q_max(1-rho)/rho, 0]
        fired, audited, X=1    Ytil = q + (1-q)/rho          max at q=0, giving 1/rho

    so ``lo = -q_max (1-rho)/rho`` and ``hi = max(1, 1/rho)``.  Two facts are worth
    naming.  First, the interval is **asymmetric**: correcting an unexpected success costs
    the upper end ``1/rho`` no matter how confident the rule is, while the lower end is
    governed by ``q_max``.  Second, the upper end is why an aggressive audit rate is not
    free -- it is the price of the unbiasedness in :func:`corrected_outcome`, and it is
    paid in confidence-sequence width, not in bias.  ``q_max`` is enforced by
    :class:`InnerTruncator`, which refuses to fire when its own ``q`` exceeds the cap; if
    it did not, this interval would not be a valid bound.

    Claim: unbiased-truncation
    """
    rho = float(audit_rate)
    if not (0.0 <= rho <= 1.0) or not math.isfinite(rho):
        raise ValueError(f"audit_rate must be in [0, 1], got {audit_rate!r}")
    qm = float(q_max)
    if not (0.0 <= qm <= 1.0) or not math.isfinite(qm):
        raise ValueError(f"q_max must be in [0, 1], got {q_max!r}")
    if not any_truncation:
        return (0.0, 1.0)
    if rho <= 0.0:
        # No audit ever fires: Ytil = q in [0, q_max], alongside completed rollouts.
        return (0.0, 1.0)
    lo = -qm * (1.0 - rho) / rho
    hi = max(1.0, 1.0 / rho)
    return (float(lo), float(hi))


# --------------------------------------------------------------------------------------
# 12.  Extra-rollouts projection
# --------------------------------------------------------------------------------------


def rollouts_to_resolve(
    n: int,
    half_width: float,
    delta_hat: float,
    safety: float = 0.95,
) -> int:
    """Project how many more paired observations would separate a still-undecided pair.

    The empirical-Bernstein half-width shrinks like ``h_n ~ c / sqrt(n)``; fitting ``c``
    from the realised ``(n, h_n)`` and solving ``h_{n*} = |Delta_hat| * safety`` gives
    ``n* = n * (h_n / (|Delta_hat| * safety))^2``, and the answer is
    ``max(0, ceil(n*) - n)``.

    This is explicitly a **projection, not a guarantee**.  It extrapolates a rate, assumes
    the point estimate stops moving, and ignores that the true contrast may be exactly
    zero -- in which case no budget ever resolves the pair.  Every surface that prints it
    must label it as a projection.  Degenerate inputs (no data yet, a point estimate of
    exactly zero) return a capped sentinel rather than infinity.

    Claim: low-cost
    """
    n = int(n)
    h = float(half_width)
    target = abs(float(delta_hat)) * float(safety)
    if n <= 0:
        return _MAX_PROJECTION
    if not math.isfinite(h) or h <= 0.0:
        return 0
    if not math.isfinite(target) or target <= 0.0:
        return _MAX_PROJECTION
    if h <= target:
        return 0
    ratio = h / target
    n_star = float(n) * ratio * ratio
    if not math.isfinite(n_star):  # pragma: no cover - defensive
        return _MAX_PROJECTION
    extra = math.ceil(n_star) - n
    return int(min(_MAX_PROJECTION, max(0, extra)))


# --------------------------------------------------------------------------------------
# 5.6  The driver
# --------------------------------------------------------------------------------------


@runtime_checkable
class Allocator(Protocol):
    """Structural type for anything that can choose what to buy next.

    Only two methods are required, and neither is trusted: the returned ``pi`` is
    re-validated against the exploration floor before it is allowed to weight anything.

    Claim: infrastructure
    """

    def select(
        self,
        pairs: Sequence[tuple[int, int]],
        channels: Mapping[tuple[int, int], Any],
    ) -> tuple[tuple[int, int], np.ndarray, int, float]:
        """Return ``(pair, pi, block, pi[block])``.

        Claim: infrastructure
        """
        ...

    def observe(self, rollout: Rollout) -> None:
        """Absorb one executed rollout.

        Claim: infrastructure
        """
        ...


class UniformAllocator:
    """The dependency-free fallback allocator: ``pi = w``, round-robin over the pairs.

    This is condition (C) of the benchmark -- "an existing anytime-valid method" with no
    decision-targeted design -- and it also lets this module (and the coverage tests) run
    without importing :mod:`quorum.allocator` at all.  ``pi`` is mixed to the exploration
    floor exactly as the real allocator does, so ``pi[b] >= pi_min_frac * w[b] > 0`` and
    the Horvitz-Thompson weight stays bounded even for a non-uniform ``w``.

    Claim: infrastructure
    """

    def __init__(
        self,
        w: np.ndarray,
        rng: np.random.Generator,
        *,
        pi_min_frac: float = 0.10,
    ) -> None:
        w_arr = np.asarray(w, dtype=float).ravel()
        if w_arr.size == 0 or np.any(w_arr < 0.0):
            raise ValueError("w must be non-empty and non-negative")
        if not (0.0 < float(pi_min_frac) <= 1.0):
            raise ValueError("pi_min_frac must be in (0, 1]")
        self.w: np.ndarray = w_arr / float(w_arr.sum())
        self.rng: np.random.Generator = rng
        self.pi_min_frac: float = float(pi_min_frac)
        self.n_blocks: int = int(w_arr.size)
        self._cursor: int = 0
        self.n_observed: int = 0

        eps = self.pi_min_frac
        uniform = np.full(self.n_blocks, 1.0 / self.n_blocks, dtype=float)
        pi = (1.0 - eps) * self.w + eps * uniform
        self.pi: np.ndarray = pi / float(pi.sum())

    def ht_bound(self) -> float:
        """The exact Horvitz-Thompson ratio bound ``max_b w[b]/pi[b]`` for this design.

        ``pi`` is fixed and known in advance here, so the bound is not a worst case at
        all -- it is computed directly, and equals exactly 1 for the uniform ``w`` that
        the (C) baseline runs on. Declaring it keeps the non-adaptive baseline from being
        charged the exploration-floor width that only an adaptive design incurs.

        Claim: coverage
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(self.pi > 0.0, self.w / np.maximum(self.pi, _TINY), 0.0)
        return float(np.max(ratio))

    def select(
        self,
        pairs: Sequence[tuple[int, int]],
        channels: Mapping[tuple[int, int], Any] | None = None,
    ) -> tuple[tuple[int, int], np.ndarray, int, float]:
        """Round-robin the pair, sample the block from the fixed ``pi``.

        Claim: infrastructure
        """
        pair_list = [(int(a), int(c)) for a, c in pairs]
        if not pair_list:
            raise ValueError("select() needs at least one candidate pair")
        pair = pair_list[self._cursor % len(pair_list)]
        self._cursor += 1
        block = int(self.rng.choice(self.n_blocks, p=self.pi))
        return pair, self.pi, block, float(self.pi[block])

    def observe(self, rollout: Rollout) -> None:
        """Count the rollout; a uniform design has nothing to learn.

        Claim: infrastructure
        """
        self.n_observed += 1


class NestedSequentialTest:
    """Driver for the nested procedure: rank models, fund the tightest pair, repeat.

    **The estimand is transitive by construction.**  ``Delta[a, c] = theta[a] - theta[c]``
    with ``theta[m] = sum_b w[b] p[b, m]``, so certifying every *adjacent* pair in the
    current point-estimate order certifies the whole order by transitivity.  That is why
    ``M - 1`` intervals suffice for an ``M``-model ranking.

    **Why the alpha split is over candidate pairs, not adjacent ones (SPEC 5.5).**  Only
    ``M - 1`` pairs are ever adjacent, but *which* pairs those are is decided by the data:
    the order is re-estimated at every step, so the family of hypotheses actually looked
    at is a data-dependent subset of all ``M(M-1)/2`` pairs.  Splitting ``alpha`` over the
    ``M - 1`` pairs that happen to end up adjacent would be selecting the family after
    seeing the data, and the family-wise claim would be worth nothing.  Splitting over the
    full candidate set makes the guarantee hold simultaneously for every pair the
    procedure could ever have looked at, which is the honest statement.  It is
    conservative by a factor of about ``M/2``, and we accept that.

    Collaborators are duck-typed and none of them can affect validity:

    * ``allocator`` -- ``select(pairs, channels) -> (pair, pi, block, pi[block])`` and
      ``observe(rollout)``.  ``None`` falls back to :class:`UniformAllocator`.
    * ``truncator`` -- see :class:`InnerTruncator`.  ``None`` disables inner truncation.
    * ``cuped`` -- see :mod:`quorum.cuped`.  ``None`` disables the control variate.

    Args:
        spec: Grid shape.
        simulator: Anything with ``rollout(t, k, s, m, truncator=..., rng=...) -> Rollout``.
        alpha: Family-wise level.
        w: Fixed target block weights; uniform by default.  Never adapts.
        pi_min_frac: Exploration floor fraction; ``pi_min = pi_min_frac * min(w[w > 0])``,
            which is ``pi_min_frac / B`` for the default uniform ``w`` exactly as SPEC 5.6
            states, and remains a valid floor when ``w`` is not uniform.
        max_rollouts: Hard cap on rollouts (two per step).
        budget_usd: Optional dollar cap.
        budget_sec: Optional wall-second cap.
        alpha_split: ``"bonferroni"`` (default) or ``"none"``.
        rng: Explicit generator; determinism is a requirement, not a nicety.
        check_every: Test the stopping rule every this many steps.  Peeking is free under
            an anytime-valid sequence; this knob only saves arithmetic.
        candidate_pairs: Restrict the family; defaults to all ``M(M-1)/2`` pairs.

    Claim: coverage
    """

    def __init__(
        self,
        spec: GridSpec,
        simulator: Any,
        *,
        alpha: float = 0.05,
        w: np.ndarray | None = None,
        allocator: Any | None = None,
        truncator: Any | None = None,
        cuped: Any | None = None,
        pi_min_frac: float = 0.10,
        max_rollouts: int = 200_000,
        budget_usd: float | None = None,
        budget_sec: float | None = None,
        alpha_split: str = "bonferroni",
        rng: np.random.Generator | None = None,
        check_every: int = 1,
        candidate_pairs: Iterable[tuple[int, int]] | None = None,
        proxy_covariate: bool = False,
        paired: bool = True,
    ) -> None:
        alpha = float(alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if alpha_split not in ("bonferroni", "none"):
            raise ValueError(f"alpha_split must be 'bonferroni' or 'none', got {alpha_split!r}")
        if int(max_rollouts) < 0:
            raise ValueError("max_rollouts must be non-negative")
        if int(check_every) < 1:
            raise ValueError("check_every must be >= 1")
        if not (0.0 < float(pi_min_frac) <= 1.0):
            raise ValueError("pi_min_frac must be in (0, 1]")

        self.spec: GridSpec = spec
        self.simulator = simulator
        self.alpha: float = alpha
        self.alpha_split: str = alpha_split
        self.pi_min_frac: float = float(pi_min_frac)
        self.max_rollouts: int = int(max_rollouts)
        self.budget_usd: float | None = None if budget_usd is None else float(budget_usd)
        self.budget_sec: float | None = None if budget_sec is None else float(budget_sec)
        self.check_every: int = int(check_every)

        self.n_models: int = int(spec.n_models)
        self.n_blocks: int = int(spec.n_blocks)

        if w is None:
            w_arr = uniform_weights(spec)
        else:
            w_arr = np.asarray(w, dtype=float).ravel()
        if w_arr.size != self.n_blocks:
            raise ValueError(f"w must have length {self.n_blocks}, got {w_arr.size}")
        if np.any(w_arr < 0.0) or not np.all(np.isfinite(w_arr)):
            raise ValueError("w must be finite and non-negative")
        total = float(w_arr.sum())
        if total <= 0.0:
            raise ValueError("w must have positive total mass")
        self.w: np.ndarray = w_arr / total

        positive = self.w[self.w > 0.0]
        self.pi_min: float = float(self.pi_min_frac * positive.min())

        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng(0)
        )
        if rng is None:
            logger.debug("NestedSequentialTest: no rng supplied, defaulting to default_rng(0)")

        self.candidate_pairs: list[tuple[int, int]] = self._build_candidate_pairs(candidate_pairs)
        self.n_candidate_pairs: int = len(self.candidate_pairs)
        if self.alpha_split == "bonferroni" and self.n_candidate_pairs > 0:
            self.alpha_pair: float = self.alpha / self.n_candidate_pairs
        else:
            self.alpha_pair = self.alpha

        self.truncator = truncator
        self.cuped = cuped
        self.proxy_covariate: bool = bool(proxy_covariate) and cuped is not None
        self.paired: bool = bool(paired)
        self._cov_abs_max: np.ndarray | None = self._covariate_scale()
        self.audit_rate: float = float(getattr(truncator, "audit_rate", 0.0)) if truncator else 1.0
        self.outcome_range: float = outcome_range(
            self.audit_rate if truncator is not None else 0.0,
            truncator is not None,
        )
        self.q_max: float = float(getattr(truncator, "q_max", 1.0)) if truncator else 1.0
        self.outcome_lo, self.outcome_hi = outcome_interval(
            self.audit_rate if truncator is not None else 0.0,
            truncator is not None,
            self.q_max,
        )

        self.allocator = (
            allocator
            if allocator is not None
            else UniformAllocator(self.w, self.rng, pi_min_frac=self.pi_min_frac)
        )
        self._own_allocator: bool = allocator is None

        # Ask the allocator for the Horvitz-Thompson ratio bound it actually guarantees.
        # A non-adaptive design has G = 1 and should not be charged the adaptive design's
        # exploration-floor penalty; charging every design the worst case would make the
        # (C) baseline look artificially wide and the comparison meaningless.
        declared = getattr(self.allocator, "ht_bound", None)
        if callable(declared):
            declared = declared()
        if declared is None:
            self.ht_bound: float = float(self.w.max() / self.pi_min)
            logger.debug(
                "allocator declares no ht_bound; falling back to w_max/pi_min = %.3g",
                self.ht_bound,
            )
        else:
            self.ht_bound = float(declared)

        self.channels: dict[tuple[int, int], PairChannel] = {}
        for pair in self.candidate_pairs:
            self.channels[pair] = self._make_channel(pair)

        # Bookkeeping.
        self.n_rollouts: int = 0
        self.n_steps_taken: int = 0
        self.cost_usd: float = 0.0
        self.cost_sec: float = 0.0
        self.n_traj_steps_executed: int = 0
        self.n_traj_steps_full: int = 0
        self.n_truncated: int = 0
        self.n_audited: int = 0
        self.stop_reason: str = "not_started"
        self._cov_dim: int | None = None
        self._warned_no_truncator: bool = False

    # ------------------------------------------------------------------- internals --

    def _build_candidate_pairs(
        self, candidate_pairs: Iterable[tuple[int, int]] | None
    ) -> list[tuple[int, int]]:
        """Canonicalise and deduplicate the family of pairs the alpha is split over.

        Claim: coverage
        """
        if candidate_pairs is None:
            return [
                (a, c)
                for a in range(self.n_models)
                for c in range(a + 1, self.n_models)
            ]
        seen: dict[tuple[int, int], None] = {}
        for a, c in candidate_pairs:
            a_i, c_i = int(a), int(c)
            if a_i == c_i:
                raise ValueError(f"candidate pair ({a_i}, {c_i}) compares a model with itself")
            if not (0 <= a_i < self.n_models and 0 <= c_i < self.n_models):
                raise ValueError(f"candidate pair ({a_i}, {c_i}) out of range")
            seen.setdefault(_canonical(a_i, c_i), None)
        return list(seen.keys())

    def _make_channel(self, pair: tuple[int, int]) -> PairChannel:
        """Build the Horvitz-Thompson channel for one canonical pair.

        Claim: coverage
        """
        return PairChannel(
            pair,
            self.w,
            self.alpha_pair,
            pi_min=self.pi_min,
            outcome_range=self.outcome_range,
            cuped=self.cuped,
            ht_bound=self.ht_bound,
            outcome_lo=self.outcome_lo,
            outcome_hi=self.outcome_hi,
            cov_abs_max=self._cov_abs_max,
            cov_means=None if self.cuped is None else getattr(self.cuped, "cov_means", None),
        )

    def _channel_for(self, pair: tuple[int, int]) -> PairChannel:
        """Fetch (or lazily create) the channel for a pair, in canonical orientation.

        Lazy creation only happens when the caller restricted ``candidate_pairs`` and the
        data-driven order nevertheless made some other pair adjacent.  It is logged
        loudly, because the family-wise statement then covers only the declared family.

        Claim: coverage
        """
        key = _canonical(*pair)
        channel = self.channels.get(key)
        if channel is None:
            logger.warning(
                "pair %s became adjacent but was not in the declared candidate family; "
                "the family-wise level is only guaranteed over the declared family",
                key,
            )
            channel = self._make_channel(key)
            self.channels[key] = channel
        return channel

    def _estimate(self, a: int, c: int) -> float:
        """Point estimate of ``Delta[a, c]``, sign-corrected from the canonical channel.

        Claim: coverage
        """
        if a == c:
            return 0.0
        key = _canonical(a, c)
        channel = self.channels.get(key)
        if channel is None or channel.n == 0:
            return 0.0
        est = channel.estimate
        return est if key == (a, c) else -est

    def _interval(self, a: int, c: int) -> tuple[float, float]:
        """Interval on ``Delta[a, c]``, flipped when the pair is stored the other way.

        Claim: coverage
        """
        key = _canonical(a, c)
        channel = self.channels.get(key)
        if channel is None:
            return (-math.inf, math.inf)
        lo, hi = channel.interval
        return (lo, hi) if key == (a, c) else (-hi, -lo)

    def _covariates(
        self, block: int, ht: float, shock: float | None = None
    ) -> np.ndarray | None:
        """Build the Horvitz-Thompson-weighted covariate vector for the CUPED adjuster.

        The HT weight multiplies the covariate too, deliberately: the control variate then
        absorbs the *weight* noise ``w_b/pi_b`` as well as the outcome noise, which is
        where much of the variance of an adaptively allocated estimator actually lives.
        Only covariates with an exactly known ``w``-weighted mean are eligible; the
        adjuster owns that table, so this method just asks it.

        Claim: variance-reduction
        """
        if self.cuped is None:
            return None
        raw: np.ndarray | None = None
        getter = getattr(self.cuped, "block_covariates", None)
        if callable(getter):
            try:
                raw = np.asarray(getter(int(block)), dtype=float).ravel()
            except (AttributeError, NotImplementedError):
                # An adjuster may expose the method but hold no per-block table; fall
                # through to the table probes rather than taking the whole run down.
                raw = None
        if raw is None:
            table = getattr(self.cuped, "cov_table", None)
            if table is None:
                table = getattr(self.cuped, "covariates", None)
            if table is not None and not callable(table):
                arr = np.asarray(table, dtype=float)
                if arr.ndim == 1:
                    arr = arr[:, None]
                if arr.shape[0] == self.n_blocks:
                    raw = arr[int(block)].astype(float, copy=False).ravel()
        if raw is None:
            if self.proxy_covariate:
                # The realised proxy outcome is a covariate in its own right; a block
                # table is optional alongside it.
                raw = np.zeros(0, dtype=float)
            else:
                if self._cov_dim is None:
                    logger.warning(
                        "a CUPED adjuster was supplied but exposes no block covariate "
                        "table (block_covariates/cov_table); running without adjustment"
                    )
                    self._cov_dim = 0
                return None
        if self.proxy_covariate:
            raw = np.concatenate(([float(self._proxy_draw(block, shock))], raw))
        if not np.all(np.isfinite(raw)):
            logger.warning("non-finite block covariate at block %d; skipping adjustment", block)
            return None
        self._cov_dim = int(raw.size)
        return float(ht) * raw

    def _sample_block(self, pi_arr: np.ndarray | None) -> tuple[int, float]:
        """Draw a second, independent block from the same predictable ``pi``.

        Used only in the unpaired regime. Drawing from the *same* ``pi`` keeps the second
        Horvitz-Thompson term unbiased for ``theta_c`` with the identical weighting, so
        the difference of the two terms is still unbiased for ``Delta``.

        Claim: variance-reduction
        """
        if pi_arr is None:
            b = int(self.rng.choice(self.n_blocks, p=self.w))
            return b, float(self.w[b])
        b = int(self.rng.choice(self.n_blocks, p=pi_arr))
        return b, float(pi_arr[b])

    def _covariate_scale(self) -> np.ndarray | None:
        """Per-column ``max_b |c_j(b)|`` over the whole grid, for the exact adjustment cap.

        Known in advance because every admitted covariate is a fixed function of the
        block. The realised-proxy column is an indicator, so its bound is 1.

        Claim: variance-reduction
        """
        if self.cuped is None:
            return None
        cols: list[float] = []
        if self.proxy_covariate:
            cols.append(1.0)
        table = getattr(self.cuped, "cov_table", None)
        if table is not None and not callable(table):
            arr = np.asarray(table, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            cols.extend(np.abs(arr).max(axis=0).tolist())
        if not cols:
            return None
        return np.asarray(cols, dtype=float)

    def _cuped_means(self, dim: int) -> np.ndarray:
        """The adjuster's declared ``E[C]``, used to re-centre differenced covariates.

        Claim: variance-reduction
        """
        means = getattr(self.cuped, "cov_means", None)
        if means is None:
            return np.zeros(dim, dtype=float)
        arr = np.asarray(means, dtype=float).ravel()
        if arr.size != dim:
            raise ValueError(
                f"CUPED adjuster declares {arr.size} covariate means but the driver built "
                f"a {dim}-dimensional covariate vector; they must describe the same "
                "covariates in the same order"
            )
        return arr

    def _draw_shock(self) -> float | None:
        """Draw the shared instance shock for one matched pair, if the simulator has one.

        Returns ``None`` for simulators that do not model instance-level dependence (for
        example replay from logged data), in which case the two rollouts are simply
        independent and everything downstream still holds -- just with more variance.

        Claim: variance-reduction
        """
        draw = getattr(self.simulator, "draw_shock", None)
        if not callable(draw):
            return None
        return float(draw(self.rng))

    def _proxy_draw(self, block: int, shock: float | None) -> float:
        """Realised success of the cheap proxy on this instance, sharing its shock.

        The proxy is run on the *same* instance as the pair under test, so its outcome
        carries within-block information -- which is the only place a paired Bernoulli
        contrast keeps most of its variance. Its ``w``-weighted mean stays exactly known,
        so it remains an admissible known-mean control variate.

        Claim: variance-reduction
        """
        runner = getattr(self.simulator, "proxy_rollout", None)
        if not callable(runner):
            return 0.0
        t, k, s = unblock_index(int(block), self.spec.n_seeds, self.spec.n_scaffolds)
        return float(runner(t, k, s, shock=shock, rng=self.rng))

    def _run_one(
        self, block: int, model: int, shock: float | None = None
    ) -> tuple[Rollout, float]:
        """Execute one rollout at ``block`` for ``model`` and return it with its ``Ytil``.

        Claim: unbiased-truncation
        """
        t, k, s = unblock_index(int(block), self.spec.n_seeds, self.spec.n_scaffolds)
        if shock is None:
            rollout: Rollout = self.simulator.rollout(
                t, k, s, int(model), truncator=self.truncator, rng=self.rng
            )
        else:
            rollout = self.simulator.rollout(
                t, k, s, int(model), truncator=self.truncator, rng=self.rng, shock=shock
            )

        completed = bool(getattr(rollout, "completed", True))
        if self.truncator is None:
            if not completed and not self._warned_no_truncator:
                self._warned_no_truncator = True
                logger.warning(
                    "simulator returned a truncated rollout although no truncator was "
                    "configured; treating rho=1 so the observed success is used as-is"
                )
            q = 0.0
            rho = 1.0
        else:
            rho = self.audit_rate
            q = float(self.truncator.q_for_rollout(rollout)) if hasattr(
                self.truncator, "q_for_rollout"
            ) else self._q_via_prefix(rollout)

        ytil = corrected_outcome(rollout, q, rho)

        # Bookkeeping happens for every rollout, truncated or not.
        self.n_rollouts += 1
        self.cost_usd += float(getattr(rollout, "cost_usd", 0.0))
        self.cost_sec += float(getattr(rollout, "cost_sec", 0.0))
        n_steps = int(getattr(rollout, "n_steps", 0))
        n_full = int(getattr(rollout, "n_steps_full", 0)) or n_steps
        self.n_traj_steps_executed += n_steps
        self.n_traj_steps_full += n_full
        if not completed:
            self.n_truncated += 1
            if bool(getattr(rollout, "audited", False)):
                self.n_audited += 1

        # The truncator learns q only from audited rollouts; the allocator learns freely,
        # because it touches efficiency and never validity.
        if self.truncator is not None and hasattr(self.truncator, "observe"):
            self.truncator.observe(rollout)
        observe = getattr(self.allocator, "observe", None)
        if callable(observe):
            observe(rollout)

        return rollout, ytil

    def _q_via_prefix(self, r: Rollout) -> float:
        """Fallback ``q`` lookup for truncators without ``q_for_rollout``.

        Claim: unbiased-truncation
        """
        q_fn = getattr(self.truncator, "q", None)
        if not callable(q_fn):
            return float(getattr(self.truncator, "q_hat", 0.0))
        end = _fire_prefix_end(r)
        traj = np.asarray(getattr(r, "trajectory", np.zeros(0)), dtype=float).ravel()
        prefix = traj[:end] if end > 0 else traj[:0]
        value = float(q_fn(prefix, int(end)))
        return float(min(max(value, 0.0), 1.0))

    # ---------------------------------------------------------------------- public --

    def current_order(self) -> list[int]:
        """Total order over models induced by the current pairwise point estimates.

        Built by **Copeland score** (how many pairwise contrasts a model currently wins),
        with the mean contrast as the tiebreak and the model index as the final tiebreak.
        Copeland is used rather than sorting by any single scalar because the channels are
        estimated at wildly different precisions -- some pairs have hundreds of
        observations and some none -- and Copeland degrades gracefully when the estimated
        contrasts are not exactly transitive.  With no data at all every score ties and
        the order is ``[0, 1, ..., M-1]``, so the procedure starts deterministically.

        Claim: coverage
        """
        m = self.n_models
        est = np.zeros((m, m), dtype=float)
        for a in range(m):
            for c in range(a + 1, m):
                value = self._estimate(a, c)
                est[a, c] = value
                est[c, a] = -value

        copeland = np.zeros(m, dtype=float)
        mean_contrast = np.zeros(m, dtype=float)
        for a in range(m):
            wins = 0.0
            for c in range(m):
                if c == a:
                    continue
                if est[a, c] > 0.0:
                    wins += 1.0
                elif est[a, c] == 0.0:
                    wins += 0.5
            copeland[a] = wins
            mean_contrast[a] = float(est[a].sum() / max(1, m - 1))

        return sorted(
            range(m),
            key=lambda i: (-copeland[i], -mean_contrast[i], i),
        )

    def adjacent_pairs(self) -> list[tuple[int, int]]:
        """Adjacent pairs of the current order, oriented ``(better, worse)``.

        Claim: coverage
        """
        order = self.current_order()
        return [(order[i], order[i + 1]) for i in range(len(order) - 1)]

    def step(self) -> None:
        """Buy one paired observation: pick a pair and a block, run both models, update.

        The two rollouts share the block, which is what makes the contrast paired and
        removes all block-level nuisance variation (task difficulty, seed jitter, scaffold
        effect) from the estimand's noise.

        Claim: low-cost
        """
        pairs = self.adjacent_pairs()
        if not pairs:
            return

        unresolved = [p for p in pairs if not self._channel_for(p).excludes_zero]
        target_pairs = unresolved if unresolved else pairs

        selection = self.allocator.select(target_pairs, self.channels)
        pair, pi, block, pi_at_block = _unpack_selection(selection)

        pi_arr: np.ndarray | None = None
        if pi is not None:
            pi_arr = np.asarray(pi, dtype=float).ravel()
            if pi_arr.size != self.n_blocks:
                raise ValueError(
                    f"allocator returned pi of length {pi_arr.size}, expected {self.n_blocks}"
                )
            if not np.all(np.isfinite(pi_arr)) or np.any(pi_arr < 0.0):
                raise ValueError("allocator returned a non-finite or negative pi")
            total = float(pi_arr.sum())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"allocator returned pi summing to {total!r}, expected 1")

        block = int(block)
        if not (0 <= block < self.n_blocks):
            raise ValueError(f"allocator returned block {block} out of range")

        if pi_arr is not None:
            pi_b = float(pi_arr[block])
            if pi_at_block is not None and not math.isclose(
                pi_b, float(pi_at_block), rel_tol=1e-6, abs_tol=1e-12
            ):
                logger.warning(
                    "allocator reported pi[b]=%g but its pi vector says %g; using the vector",
                    float(pi_at_block),
                    pi_b,
                )
        else:
            if pi_at_block is None:
                raise ValueError("allocator returned neither a pi vector nor pi[block]")
            pi_b = float(pi_at_block)

        a, c = int(pair[0]), int(pair[1])
        channel = self._channel_for((a, c))
        key = channel.pair

        # Run the model that the channel treats as "a" first, purely for readability;
        # both rollouts are drawn at the same block so the pairing is exact either way.
        # One shared instance shock per matched pair: both models face the same problem
        # instance. This is common-random-numbers pairing -- it changes only the joint
        # law of the two outcomes, never their marginals, so Delta is untouched while
        # Var(Ytil_a - Ytil_c) falls.
        shock = self._draw_shock()
        if self.paired:
            rollout_a, ytil_a = self._run_one(block, key[0], shock)
            rollout_c, ytil_c = self._run_one(block, key[1], shock)
            del rollout_a, rollout_c  # bookkeeping already absorbed them
            ht = self.w[block] / pi_b if pi_b > 0.0 else 0.0
            cov = self._covariates(block, ht, shock)
            channel.observe(block, pi_b, ytil_a, ytil_c, cov)
        else:
            # Unpaired regime: the two models are evaluated on *independently* drawn
            # instances, as happens with logged data and public leaderboards where each
            # model was run on its own sample. E[Z] is still Delta -- the two HT terms
            # are separately unbiased for theta_a and theta_c -- but the shared instance
            # difficulty no longer cancels, so between-block variance dominates. This is
            # precisely the regime where a known-mean control variate earns its keep.
            block_c, pi_c = self._sample_block(pi_arr)
            shock_c = self._draw_shock()
            rollout_a, ytil_a = self._run_one(block, key[0], shock)
            rollout_c, ytil_c = self._run_one(block_c, key[1], shock_c)
            del rollout_a, rollout_c
            ht_a = self.w[block] / pi_b if pi_b > 0.0 else 0.0
            ht_c = self.w[block_c] / pi_c if pi_c > 0.0 else 0.0
            cov_a = self._covariates(block, ht_a, shock)
            cov_c = self._covariates(block_c, ht_c, shock_c)
            # The DIFFERENCE of two HT-weighted covariates has known mean ZERO, not the
            # per-term mean E[C] the adjuster was built with.  Re-centre by E[C] so the
            # quantity handed to the adjuster still has the mean the adjuster subtracts.
            # Getting this wrong does not merely lose efficiency: it injects the constant
            # phi^T E[C] into every observation, which is a bias the confidence sequence
            # cannot see.  (Measured before the fix: 100% false-positive rate on a null
            # pair.  It is now a regression test.)
            if cov_a is None or cov_c is None:
                cov = None
            else:
                cov = cov_a - cov_c + self._cuped_means(cov_a.size)
            channel.observe_unpaired(
                block, pi_b, ytil_a, block_c, pi_c, ytil_c, cov
            )

        self.n_steps_taken += 1

    def all_adjacent_resolved(self) -> bool:
        """Whether every adjacent pair's interval currently excludes zero.

        Adjacent separation plus transitivity of ``Delta`` certifies the entire order at
        family-wise level ``alpha`` after the Bonferroni split of SPEC 5.5.

        Claim: coverage
        """
        pairs = self.adjacent_pairs()
        if not pairs:
            return True
        return all(self._channel_for(p).excludes_zero for p in pairs)

    def budget_exhausted(self) -> tuple[bool, str]:
        """Whether any budget is spent, and which one, as ``(exhausted, reason)``.

        Claim: low-cost
        """
        if self.n_rollouts + 2 > self.max_rollouts:
            return True, "max_rollouts"
        if self.budget_usd is not None and self.cost_usd >= self.budget_usd:
            return True, "budget_usd"
        if self.budget_sec is not None and self.cost_sec >= self.budget_sec:
            return True, "budget_sec"
        return False, ""

    def run(self) -> RankDecision:
        """Run until the order is certified or the budget runs out; return the decision.

        Stops on the first of: every adjacent pair's confidence-sequence interval excludes
        zero, ``max_rollouts`` reached, ``budget_usd`` spent, ``budget_sec`` spent.  When
        it stops undecided the result must be reported as "not distinguishable at this
        budget" -- never as a superiority claim -- and the projection in
        ``extra_rollouts_needed`` says how much more would plausibly settle it.

        Claim: low-cost
        """
        if self.n_models < 2:
            self.stop_reason = "resolved"
            return self._decision(resolved=True)

        if self.all_adjacent_resolved():
            self.stop_reason = "resolved"
            return self._decision(resolved=True)

        while True:
            exhausted, reason = self.budget_exhausted()
            if exhausted:
                self.stop_reason = reason
                break
            self.step()
            if self.n_steps_taken % self.check_every == 0 and self.all_adjacent_resolved():
                self.stop_reason = "resolved"
                break

        resolved = self.stop_reason == "resolved"
        logger.info(
            "NestedSequentialTest finished: reason=%s rollouts=%d usd=%.4f sec=%.1f",
            self.stop_reason,
            self.n_rollouts,
            self.cost_usd,
            self.cost_sec,
        )
        return self._decision(resolved=resolved)

    def _decision(self, *, resolved: bool) -> RankDecision:
        """Assemble the :class:`~quorum.types.RankDecision` from the current state.

        Claim: coverage
        """
        order = self.current_order()
        adjacent = [(order[i], order[i + 1]) for i in range(len(order) - 1)]

        intervals: dict[tuple[int, int], tuple[float, float]] = {}
        for key, channel in self.channels.items():
            lo, hi = channel.interval
            intervals[key] = (float(lo), float(hi))
            intervals[(key[1], key[0])] = (float(-hi), float(-lo))

        undecided: list[tuple[int, int]] = []
        extra: dict[tuple[int, int], int] = {}
        for pair in adjacent:
            channel = self._channel_for(pair)
            if channel.excludes_zero:
                continue
            undecided.append(pair)
            # One channel observation costs two rollouts (one per model on the block), so
            # the projection is doubled to be reported in rollout units.
            extra_obs = rollouts_to_resolve(
                channel.n, channel.half_width, self._estimate(*pair)
            )
            extra[pair] = int(min(_MAX_PROJECTION, 2 * extra_obs))

        resolved_flag = bool(resolved and not undecided)

        meta: dict[str, Any] = {
            "alpha": self.alpha,
            "alpha_pair": self.alpha_pair,
            "alpha_split": self.alpha_split,
            "n_candidate_pairs": self.n_candidate_pairs,
            "pi_min": self.pi_min,
            "pi_min_frac": self.pi_min_frac,
            "outcome_range": self.outcome_range,
            "audit_rate": self.audit_rate if self.truncator is not None else None,
            "n_steps_taken": self.n_steps_taken,
            "n_truncated": self.n_truncated,
            "n_audited": self.n_audited,
            "stop_reason": self.stop_reason,
            "adjacent_pairs": adjacent,
            "pair_n": {key: channel.n for key, channel in self.channels.items()},
            "pair_estimate": {
                key: channel.estimate for key, channel in self.channels.items()
            },
            "allocator": type(self.allocator).__name__,
            "truncation": self.truncator is not None,
            "cuped": self.cuped is not None,
            "extra_rollouts_is_projection": True,
        }

        return RankDecision(
            order=order,
            resolved=resolved_flag,
            n_rollouts=self.n_rollouts,
            cost_usd=self.cost_usd,
            cost_sec=self.cost_sec,
            intervals=intervals,
            undecided_pairs=undecided,
            n_steps_executed=self.n_traj_steps_executed,
            n_steps_full=self.n_traj_steps_full,
            extra_rollouts_needed=extra,
            meta=meta,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"NestedSequentialTest(M={self.n_models}, B={self.n_blocks}, "
            f"alpha={self.alpha:g}, alpha_pair={self.alpha_pair:g}, "
            f"rollouts={self.n_rollouts})"
        )


# --------------------------------------------------------------------------------------
# small module-private helpers
# --------------------------------------------------------------------------------------


def _canonical(a: int, c: int) -> tuple[int, int]:
    """Canonical (sorted) key for an unordered model pair.

    Claim: infrastructure
    """
    a, c = int(a), int(c)
    return (a, c) if a < c else (c, a)


def _unpack_selection(
    selection: Any,
) -> tuple[tuple[int, int], np.ndarray | None, int, float | None]:
    """Normalise an allocator's ``select`` return value into a 4-tuple.

    Accepts the SPEC 7.2 shape ``(pair, pi, block, pi[block])`` and, for convenience in
    tests and adapters, the shorter ``(pair, pi, block)``.

    Claim: infrastructure
    """
    if not isinstance(selection, (tuple, list)):
        raise TypeError("allocator.select must return a tuple (pair, pi, block, pi[block])")
    if len(selection) == 4:
        pair, pi, block, pi_b = selection
        return (int(pair[0]), int(pair[1])), pi, int(block), (
            None if pi_b is None else float(pi_b)
        )
    if len(selection) == 3:
        pair, pi, block = selection
        return (int(pair[0]), int(pair[1])), pi, int(block), None
    raise TypeError(
        f"allocator.select returned {len(selection)} values, expected 4: "
        "(pair, pi, block, pi[block])"
    )
