"""Known-mean control variates (CUPED) for the Horvitz-Thompson pair channel.

This module is contribution (d) of the thesis: cut the variance of the outer
Horvitz-Thompson (HT) pair estimator *without* touching its unbiasedness, and hence
without touching the anytime validity of the confidence sequence that wraps it.

The estimator
-------------
The pair channel (SPEC section 5.2) feeds the confidence sequence the raw quantity

    Z_i = (w[b_i] / pi_i[b_i]) * (Ytil_a - Ytil_c),      E[Z_i | F_{i-1}] = Delta[a, c].

We replace it with the control-variate-adjusted

    Ztil_i = Z_i - phi_i^T (C_i - E[C]),

where

* ``C_i`` is a covariate vector measured at the *sampled block* ``b_i``,
* ``E[C]`` is the **exactly known** expectation of ``C_i`` under the sampling design,
* ``phi_i`` is ``F_{i-1}``-measurable: it is fit from observations ``1..i-1`` only.

Because ``E[C_i | F_{i-1}] = E[C]`` holds *identically* (not approximately) and ``phi_i``
is predictable,

    E[Ztil_i | F_{i-1}] = E[Z_i | F_{i-1}] - phi_i^T (E[C_i | F_{i-1}] - E[C]) = Delta,

so the martingale structure the empirical-Bernstein confidence sequence relies on is
untouched. Every design decision in this file exists to protect that one equation.

Why the covariates carry the HT weight
--------------------------------------
Each admitted covariate is carried with the **same** HT weight as the outcome:

    C_i[j] = (w[b_i] / pi_i[b_i]) * c_j[b_i],     E[C[j]] = sum_b w[b] * c_j[b],

the second identity because ``b_i ~ pi_i`` gives
``E[(w_b/pi_b) c_j(b) | F_{i-1}] = sum_b pi_b (w_b/pi_b) c_j(b) = sum_b w_b c_j(b)``,
which does not depend on ``pi_i`` at all -- so the mean stays known *even though the
allocator adapts ``pi`` every step*. This is not cosmetic. Under an adaptive allocator a
large share of ``Var(Z_i)`` is **weight noise**: the ratio ``w_b/pi_b`` swings by orders
of magnitude between an aggressively targeted block and a floor-probability block. A
covariate defined on the raw block scale (``c_j(b)`` alone) can only absorb outcome
noise; carrying the weight lets the control variate absorb the weight noise as well,
which is the dominant term exactly in the regime contribution (b) creates.

Which covariates are admitted (and which are refused)
-----------------------------------------------------
Only covariates with a closed-form ``w``-weighted mean over the *whole* grid:

1. **proxy** -- a cheap proxy model's success probability at the block, ``proxy_p[b]``;
2. **history** -- the block's success rate from a previous, already-completed run;
3. **pilot difficulty** -- block difficulty read off a **frozen** pilot fit of the cell
   model. Frozen means the fit is a fixed function of ``b`` from that point on, so its
   ``w``-weighted mean is a number we can compute exactly.
4. **weight** (optional) -- the constant covariate, i.e. the bare HT ratio
   ``w_b/pi_b``, whose known mean is ``sum_b w_b = 1``.

Trajectory-prefix features of the *same* rollout are **refused**. They have no known
mean (their distribution depends on the rollout we are currently buying), so plugging in
an estimated mean would put an ``O(n^{-1/2})`` bias inside a procedure whose whole selling
point is that its error is design-based. That refusal is a correctness decision, not an
omission -- such features belong in the ``q``-model of SPEC section 5.3, where the audit
correction makes miscalibration cost variance instead of validity.

How much variance can this remove?
----------------------------------
With a single covariate and the optimal coefficient ``phi* = Cov(Z, C) / Var(C)``,

    Var(Ztil) = Var(Z) - Cov(Z, C)^2 / Var(C) = Var(Z) * (1 - corr(Z, C)^2),

so the measured :meth:`CupedAdjuster.variance_ratio` should track ``1 - corr^2``. With
``d`` covariates the same identity holds with ``corr`` replaced by the multiple
correlation ``R``. :func:`variance_reduction_report` reports both sides so the benchmark
can show them tracking.

Claim: variance-reduction
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from scipy import linalg as _linalg
from scipy import stats as _stats

from quorum.types import GridSpec, block_index

__all__ = [
    "CupedAdjuster",
    "build_covariate_vector",
    "known_covariate_means",
    "covariate_names",
    "pilot_difficulty_from_predictions",
    "variance_reduction_report",
]

logger = logging.getLogger(__name__)

_EPS = 1e-12
_SHARED_KEY: tuple[int, int] = (-1, -1)

PairKey = tuple[int, int]


# --------------------------------------------------------------------------------------
# Covariate construction: the *only* covariates whose mean is known in closed form.
# --------------------------------------------------------------------------------------


def _flat_block_array(values: Any, name: str, n_blocks: int | None = None) -> np.ndarray:
    """Flatten a per-block array to shape (B,) and validate it.

    Accepts either a flat ``(B,)`` array or a ``(T, K, S)`` grid array; the latter is
    flattened row-major over ``(t, k, s)``, matching :func:`quorum.types.block_index`.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3:
        arr = arr.reshape(-1)
    elif arr.ndim != 1:
        arr = np.ravel(arr)
    if arr.size == 0:
        raise ValueError(f"{name} must be a non-empty per-block array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values; a covariate must be finite")
    if n_blocks is not None and arr.size != n_blocks:
        raise ValueError(
            f"{name} has {arr.size} blocks but {n_blocks} were expected; every "
            "covariate must be defined on the same block grid as w"
        )
    return arr


def _covariate_stack(
    proxy_p: Any,
    history: Any | None,
    pilot_difficulty: Any | None,
    include_weight: bool,
    n_blocks: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Stack the admitted per-block covariate columns in canonical order.

    Canonical order is ``proxy, history, pilot_difficulty, weight``; optional entries are
    dropped (not zero-filled) when ``None``. Both :func:`build_covariate_vector` and
    :func:`known_covariate_means` route through this helper, which is what guarantees the
    covariate vector and the known-mean vector can never disagree about layout.
    """
    cols: list[np.ndarray] = []
    names: list[str] = []

    proxy = _flat_block_array(proxy_p, "proxy_p", n_blocks)
    cols.append(proxy)
    names.append("proxy")
    n_b = proxy.size

    if history is not None:
        cols.append(_flat_block_array(history, "history", n_b))
        names.append("history")
    if pilot_difficulty is not None:
        cols.append(_flat_block_array(pilot_difficulty, "pilot_difficulty", n_b))
        names.append("pilot_difficulty")
    if include_weight:
        cols.append(np.ones(n_b, dtype=float))
        names.append("weight")

    return np.vstack(cols), names


def _resolve_block(block: Any, n_blocks: int, spec: GridSpec | None) -> int:
    """Turn ``block`` (flat id, or a ``(t, k, s)`` triple plus a spec) into a flat id."""
    if isinstance(block, (tuple, list, np.ndarray)) and np.ndim(block) == 1 and len(block) == 3:
        t, k, s = (int(v) for v in block)  # type: ignore[misc]
        if spec is None:
            raise ValueError(
                "block was given as a (t, k, s) triple; pass spec=GridSpec(...) so it "
                "can be flattened with quorum.types.block_index"
            )
        b = block_index(t, k, s, spec.n_seeds, spec.n_scaffolds)
    else:
        b = int(block)
    if not 0 <= b < n_blocks:
        raise ValueError(f"block index {b} out of range for {n_blocks} blocks")
    return b


def covariate_names(
    history: Any | None = None,
    pilot_difficulty: Any | None = None,
    *,
    include_weight: bool = False,
) -> list[str]:
    """Names of the covariate columns, in the order the vectors are built.

    Useful for logging and for the benchmark's variance-reduction table; the layout is
    exactly the one :func:`build_covariate_vector` and :func:`known_covariate_means` use.

    Claim: variance-reduction
    """
    names = ["proxy"]
    if history is not None:
        names.append("history")
    if pilot_difficulty is not None:
        names.append("pilot_difficulty")
    if include_weight:
        names.append("weight")
    return names


def build_covariate_vector(
    block: Any,
    pi_b: float,
    w_b: float,
    proxy_p: Any,
    history: Any | None = None,
    pilot_difficulty: Any | None = None,
    *,
    include_weight: bool = False,
    spec: GridSpec | None = None,
) -> np.ndarray:
    """Build the HT-weighted control-variate vector ``C_i`` for one sampled block.

    ``C_i[j] = (w_b / pi_b) * c_j[b]`` where ``c_j`` runs over the admitted per-block
    covariates in canonical order (``proxy, history, pilot_difficulty[, weight]``);
    entries passed as ``None`` are dropped rather than zero-filled.

    The HT ratio multiplies the covariate as well as the outcome **on purpose**: it is
    what lets the control variate absorb the ``w_b/pi_b`` weight noise injected by the
    adaptive allocator, not merely the Bernoulli outcome noise. The known mean is
    unaffected -- ``E[(w_b/pi_b) c_j(b)] = sum_b w_b c_j(b)`` for *any* predictable
    ``pi`` with ``pi_b > 0`` -- which is the whole reason this covariate is admissible.

    Parameters
    ----------
    block:
        Flat block id, or a ``(t, k, s)`` triple together with ``spec``.
    pi_b:
        The sampling probability actually used for this block; must be strictly positive
        (the allocator's ``pi_min`` floor guarantees that).
    w_b:
        The fixed target weight of this block. ``w`` defines the estimand and never adapts.
    proxy_p, history, pilot_difficulty:
        Per-block arrays, shape ``(B,)`` or ``(T, K, S)``. ``pilot_difficulty`` must come
        from a **frozen** fit -- a fit that keeps updating is not a fixed function of the
        block and its ``w``-weighted mean would no longer be known.

    Returns
    -------
    np.ndarray
        Shape ``(d,)``, matching :func:`known_covariate_means` called with the same
        optional arguments.

    Claim: variance-reduction
    """
    stack, _ = _covariate_stack(proxy_p, history, pilot_difficulty, include_weight)
    n_blocks = stack.shape[1]
    b = _resolve_block(block, n_blocks, spec)

    pi = float(pi_b)
    if not np.isfinite(pi) or pi <= 0.0:
        raise ValueError(
            f"pi_b must be strictly positive (got {pi_b!r}); the allocator's pi_min floor "
            "is what keeps the HT ratio bounded"
        )
    w = float(w_b)
    if not np.isfinite(w) or w < 0.0:
        raise ValueError(f"w_b must be finite and non-negative (got {w_b!r})")

    return (w / pi) * stack[:, b]


def known_covariate_means(
    w: Any,
    proxy_p: Any,
    history: Any | None = None,
    pilot_difficulty: Any | None = None,
    *,
    include_weight: bool = False,
) -> np.ndarray:
    """Exact ``w``-weighted means ``E[C]`` of the admitted covariates -- no estimation.

    ``E[C[j]] = sum_b w[b] * c_j[b]``. This is a closed-form sum over the *whole* grid,
    including blocks that were never sampled, which is why it is exactly known before a
    single rollout is bought. It is deliberately **not** a function of ``pi``: the HT
    weighting inside :func:`build_covariate_vector` cancels the allocator's choice, so
    the same mean vector stays correct for every step of an adaptive run.

    Using an *estimated* mean here would inject an ``O(n^{-1/2})`` bias into the outer
    martingale and silently void the confidence sequence; that is the failure mode this
    whole module is arranged to make impossible.

    Parameters
    ----------
    w:
        Target weights over blocks, shape ``(B,)`` or ``(T, K, S)``, summing to 1.
    proxy_p, history, pilot_difficulty:
        Per-block covariate arrays, same layout rules as ``w``.

    Returns
    -------
    np.ndarray
        Shape ``(d,)``, aligned with :func:`build_covariate_vector`.

    Claim: variance-reduction
    """
    w_flat = _flat_block_array(w, "w")
    if np.any(w_flat < 0.0):
        raise ValueError("w must be non-negative")
    total = float(w_flat.sum())
    if total <= 0.0:
        raise ValueError("w must have positive total mass")
    if abs(total - 1.0) > 1e-6:
        logger.warning(
            "known_covariate_means: w sums to %.6g, not 1; the estimand is defined by w "
            "so this is almost certainly a bug upstream",
            total,
        )
    stack, _ = _covariate_stack(
        proxy_p, history, pilot_difficulty, include_weight, n_blocks=w_flat.size
    )
    return stack @ w_flat


def pilot_difficulty_from_predictions(
    p_hat: Any,
    *,
    models: Sequence[int] | None = None,
    clip: float = 1e-6,
) -> np.ndarray:
    """Per-block difficulty from a **frozen** pilot fit, as a fixed function of the block.

    Given a pilot cell-model prediction array ``p_hat`` of shape ``(T, K, S, M)``, returns
    ``-mean_m logit(p_hat[b, m])``: high where the block is hard for everyone. Averaging
    over models makes it a pure block statistic, so it carries information about the
    block's difficulty but (to first order) none about which model wins there -- exactly
    the "correlated with the outcome, orthogonal to the treatment contrast" property a
    control variate needs.

    The caller is responsible for **freezing** the fit: pass predictions from a pilot fit
    that is not updated afterwards. A live fit is a function of the data, its ``w``-weighted
    mean would be a random quantity, and the known-mean guarantee would be lost.

    Claim: variance-reduction
    """
    arr = np.asarray(p_hat, dtype=float)
    if arr.ndim == 4:
        if models is not None:
            arr = arr[..., list(models)]
        flat = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim == 2:
        if models is not None:
            arr = arr[:, list(models)]
        flat = arr
    elif arr.ndim in (1, 3):
        flat = arr.reshape(-1, 1)
    else:
        raise ValueError("p_hat must have shape (T,K,S,M), (B,M), (T,K,S) or (B,)")
    if flat.size == 0:
        raise ValueError("p_hat is empty")
    lo = float(clip)
    hi = 1.0 - lo
    q = np.clip(flat, lo, hi)
    logits = np.log(q / (1.0 - q))
    return -np.mean(logits, axis=1)


# --------------------------------------------------------------------------------------
# Per-pair state
# --------------------------------------------------------------------------------------


class _PairState:
    """Sufficient statistics and history for one pair's control-variate fit.

    Held privately by :class:`CupedAdjuster`. The regression is maintained through
    running sums so that refitting after every observation is O(d^2) rather than O(n d),
    which keeps a strictly-predictable (refit-every-step) ``phi`` affordable.
    """

    __slots__ = (
        "d",
        "n_fit",
        "n_total",
        "sum_z",
        "sum_abs_z",
        "sum_zz",
        "sum_c",
        "sum_cc",
        "sum_cz",
        "z_hist",
        "c_hist",
        "ytil_hist",
        "active_hist",
        "n_clipped",
        "_phi",
        "_phi_at_n",
    )

    def __init__(self, d: int) -> None:
        self.d = d
        self.n_fit = 0
        self.n_total = 0
        self.sum_z = 0.0
        self.sum_abs_z = 0.0
        self.sum_zz = 0.0
        self.sum_c = np.zeros(d, dtype=float)
        self.sum_cc = np.zeros((d, d), dtype=float)
        self.sum_cz = np.zeros(d, dtype=float)
        self.z_hist: list[float] = []
        self.c_hist: list[np.ndarray | None] = []
        self.ytil_hist: list[float] = []
        self.active_hist: list[bool] = []
        self.n_clipped = 0
        self._phi = np.zeros(d, dtype=float)
        self._phi_at_n = -1

    # -- predictable summaries (functions of strictly-prior data only) -----------------

    def scale(self) -> float:
        """Running scale of the raw outcome, used only for the clipping rail.

        A function of strictly prior observations, so the rail it feeds stays predictable.

        Claim: variance-reduction
        """
        n = len(self.z_hist)
        if n >= 2:
            mean = self.sum_z / n
            var = max(0.0, (self.sum_zz - n * mean * mean) / (n - 1))
            sd = float(np.sqrt(var))
            if np.isfinite(sd) and sd > _EPS:
                return sd
        if n >= 1:
            mad = self.sum_abs_z / n
            if np.isfinite(mad) and mad > _EPS:
                return float(mad)
        return 1.0

    def running_cov_mean(self) -> np.ndarray:
        """Running sample mean of the covariates (only if estimated means are allowed).

        Never consulted on the default path, where every covariate mean is known in
        closed form; an estimated mean degrades the design-based guarantee.

        Claim: variance-reduction
        """
        if self.n_fit == 0:
            return np.zeros(self.d, dtype=float)
        return self.sum_c / self.n_fit


# --------------------------------------------------------------------------------------
# The adjuster
# --------------------------------------------------------------------------------------


class CupedAdjuster:
    """Known-mean control variates. ``Y~ = Y - phi^T (C - E[C])`` with ``E[C]`` KNOWN.

    The adjuster sits between the pair channel and the confidence sequence. It never sees
    a rollout; it sees the channel's HT-weighted contrast ``z`` and a covariate vector
    ``cov_vec`` built by :func:`build_covariate_vector`, and it returns a lower-variance
    quantity with the *same conditional mean*.

    Non-anticipativity, structurally
    --------------------------------
    ``phi`` must be ``F_{i-1}``-measurable or ``E[Y~_i | F_{i-1}] = Delta`` fails and the
    outer confidence sequence stops being anytime-valid. That is enforced here by
    construction rather than by convention:

    * :meth:`adjust` is a **pure function of stored state**. It appends nothing, updates
      no sufficient statistic and caches nothing derived from its arguments, so the
      current ``(z, cov_vec)`` cannot reach the fit through any code path.
    * :meth:`observe` is the only mutator. It appends the point *after* computing the
      realised adjustment with the pre-existing ``phi``.
    * ``phi`` is therefore always the ridge solution on observations ``1..i-1`` when it is
      applied to observation ``i``. :meth:`adjust_and_observe` performs the two calls in
      that order for callers who would rather not be trusted with it.

    A test asserts this (``tests/test_cuped.py``): replaying a history and then perturbing
    only the current point must leave ``theta(pair)`` bit-identical.

    Numerical rails
    ---------------
    * ``warmup`` observations during which ``phi = 0`` (the adjustment is exactly the
      identity, so no bias and no variance change) -- a fit from three points is noise.
    * Ridge regularisation (relative to the covariate scatter) on the least-squares solve.
    * A **symmetric, predictable** clip of the adjustment magnitude to ``clip`` times the
      running scale of the raw outcome. The limit depends only on prior data, and the map
      ``a -> clip(a, -L, +L)`` is odd, so it cannot introduce a systematic sign. It can
      still introduce a small conservatism: the induced bias is bounded in absolute value
      by ``E[(|a| - L)_+ | F_{i-1}]``, which is exactly zero on every step where the clip
      does not bind. The default ``clip=10.0`` running standard deviations is deliberately
      generous -- the rail exists so that a pathological early fit cannot blow up the CS
      bounds, not as part of the estimator. :meth:`clip_rate` reports how often it bound
      so a run can demonstrate the conservatism was nil.

    The theoretical target
    ----------------------
    With one covariate and the optimal ``phi* = Cov(Z, C)/Var(C)``,
    ``Var(Y~) = Var(Y) (1 - corr(Z, C)^2)``; :meth:`variance_ratio` should track
    ``1 - corr^2`` and :func:`variance_reduction_report` prints both.

    Parameters
    ----------
    cov_means:
        The exactly-known ``E[C]``, from :func:`known_covariate_means`. Entries that are
        ``NaN`` mean "unknown"; those are rejected unless ``allow_estimated_mean=True``.
    warmup:
        Number of usable observations before ``phi`` is allowed to be non-zero.
    ridge:
        Ridge penalty, applied relative to the mean diagonal of the covariate scatter.
    clip:
        Adjustment magnitude cap, in units of the running scale of the raw outcome.
    per_pair:
        Keep a separate ``phi`` per model pair (default). With ``False`` a single ``phi``
        is shared by every pair, which pools data at the price of assuming one common
        covariate/outcome relationship.
    allow_estimated_mean:
        Default ``False``. Setting it ``True`` permits ``NaN`` entries in ``cov_means`` to
        be filled with the running sample mean; this is predictable but **not** an exactly
        known mean, so the design-based guarantee degrades to approximate. Provided for
        ablations only, and it logs a warning.

    Claim: variance-reduction
    """

    def __init__(
        self,
        cov_means: np.ndarray,
        *,
        warmup: int = 30,
        ridge: float = 1e-8,
        clip: float = 10.0,
        per_pair: bool = True,
        allow_estimated_mean: bool = False,
        names: Sequence[str] | None = None,
        cov_table: np.ndarray | None = None,
    ) -> None:
        means = np.atleast_1d(np.asarray(cov_means, dtype=float)).reshape(-1)
        if means.size == 0:
            raise ValueError("cov_means must contain at least one covariate mean")
        if np.any(np.isinf(means)):
            raise ValueError("cov_means contains infinite entries")
        unknown = np.isnan(means)
        if unknown.any() and not allow_estimated_mean:
            raise ValueError(
                "cov_means contains NaN (an unknown mean) while allow_estimated_mean is "
                "False. Only covariates with a closed-form w-weighted mean over the whole "
                "grid are admissible; an estimated mean would bias the outer confidence "
                "sequence. Trajectory-prefix features of the current rollout belong in the "
                "q-model of SPEC 5.3, not here."
            )
        if unknown.any():
            logger.warning(
                "CupedAdjuster: %d/%d covariate means are unknown and will be estimated "
                "from history; the known-mean (design-based) guarantee no longer holds "
                "exactly for those columns",
                int(unknown.sum()),
                means.size,
            )

        if int(warmup) < 0:
            raise ValueError("warmup must be non-negative")
        if float(ridge) < 0.0:
            raise ValueError("ridge must be non-negative")
        if not (float(clip) > 0.0):
            raise ValueError("clip must be strictly positive")

        self._cov_means = means
        self._unknown_mean = unknown
        self.d: int = int(means.size)
        self.warmup = int(warmup)
        self.ridge = float(ridge)
        self.clip = float(clip)
        self.per_pair = bool(per_pair)
        self.allow_estimated_mean = bool(allow_estimated_mean)
        self.names: list[str] = (
            list(names) if names is not None else [f"c{j}" for j in range(self.d)]
        )
        if len(self.names) != self.d:
            raise ValueError("names must have one entry per covariate")

        self._states: dict[PairKey, _PairState] = {}
        self._seen: list[PairKey] = []

        self._cov_table: np.ndarray | None = None
        if cov_table is not None:
            table = np.asarray(cov_table, dtype=float)
            if table.ndim == 1:
                table = table[:, None]
            if table.ndim != 2:
                raise ValueError("cov_table must be 2-D with shape (n_blocks, d)")
            if table.shape[1] > self.d:
                raise ValueError(
                    f"cov_table has {table.shape[1]} covariate columns but cov_means has "
                    f"only {self.d}; they must describe the same covariates in order"
                )
            # Fewer columns than means is legal: the driver may prepend covariates that
            # are not functions of the block alone (the realised proxy outcome), whose
            # known means still occupy the leading entries of cov_means.
            if not np.all(np.isfinite(table)):
                raise ValueError("cov_table contains non-finite entries")
            self._cov_table = table

    # -- basic accessors ---------------------------------------------------------------

    def block_covariates(self, block: int) -> np.ndarray:
        """Unweighted per-block covariate vector ``c[:, b]`` for the driver.

        This is the hand-off point between :class:`quorum.nested_seq.NestedSequentialTest`
        and this module. The driver asks for the *unweighted* covariate row and applies the
        Horvitz-Thompson ratio ``w_b / pi_b`` itself, because it is the only party that
        knows the ``pi`` the block was actually sampled from. The product is exactly the
        ``C_i`` of :func:`build_covariate_vector`, so the known mean
        :func:`known_covariate_means` returned stays correct.

        Requires the adjuster to have been constructed with ``cov_table``. Without it the
        driver logs a warning and runs unadjusted -- correct, but with no variance
        reduction, so a silently missing table would quietly void claim (d).

        Claim: variance-reduction
        """
        if self._cov_table is None:
            raise AttributeError(
                "this CupedAdjuster was built without cov_table, so it cannot serve "
                "per-block covariates; pass cov_table=<(n_blocks, d) array> to enable "
                "the control variate inside NestedSequentialTest"
            )
        return self._cov_table[int(block)].copy()

    @property
    def cov_table(self) -> np.ndarray | None:
        """The per-block covariate table, or ``None`` if this adjuster has no table.

        Claim: variance-reduction
        """
        return None if self._cov_table is None else self._cov_table.copy()

    @property
    def cov_means(self) -> np.ndarray:
        """The known ``E[C]`` this adjuster was constructed with (a copy).

        Claim: variance-reduction
        """
        return self._cov_means.copy()

    @property
    def n_covariates(self) -> int:
        """Number of admitted covariates ``d``.

        Claim: variance-reduction
        """
        return self.d

    def pairs(self) -> list[PairKey]:
        """Pairs that have been observed, in first-seen order.

        Claim: variance-reduction
        """
        return list(self._seen)

    # -- internals ---------------------------------------------------------------------

    def _key(self, pair: Any, *, register: bool = False) -> PairKey:
        """Normalise a pair to a hashable key, honouring ``per_pair``.

        ``register=True`` (only ever passed by :meth:`observe`) also records the pair in
        the first-seen list, so that read-only entry points cannot mutate the adjuster.
        """
        if pair is not None:
            try:
                a, c = (int(v) for v in pair)  # type: ignore[misc]
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise TypeError(f"pair must be a 2-tuple of model indices, got {pair!r}") from exc
            norm: PairKey = (a, c)
            if register and norm not in self._seen:
                self._seen.append(norm)
        else:
            norm = _SHARED_KEY
        return norm if self.per_pair else _SHARED_KEY

    def _state(self, key: PairKey, *, create: bool = True) -> _PairState:
        st = self._states.get(key)
        if st is None:
            st = _PairState(self.d)
            if create:
                self._states[key] = st
        return st

    def _check_cov(self, cov_vec: Any) -> np.ndarray | None:
        if cov_vec is None:
            return None
        c = np.atleast_1d(np.asarray(cov_vec, dtype=float)).reshape(-1)
        if c.size != self.d:
            raise ValueError(
                f"cov_vec has length {c.size} but cov_means has length {self.d}; the "
                "covariate vector and the known-mean vector must be built with the same "
                "optional covariates (see build_covariate_vector / known_covariate_means)"
            )
        if not np.all(np.isfinite(c)):
            raise ValueError("cov_vec contains non-finite values")
        return c

    def _effective_means(self, st: _PairState) -> np.ndarray:
        """``E[C]``, with any NaN entries filled by the running (predictable) mean."""
        if not self._unknown_mean.any():
            return self._cov_means
        mu = self._cov_means.copy()
        mu[self._unknown_mean] = st.running_cov_mean()[self._unknown_mean]
        return mu

    def _fit(self, st: _PairState) -> np.ndarray:
        """Ridge least-squares ``phi`` from the sufficient statistics of *stored* history.

        Cached on ``st.n_fit``; because :meth:`adjust` never touches ``n_fit``, the cache
        can only ever hold a fit over strictly-prior observations.

        The solve is done in the standardised (correlation) basis and each eigendirection
        of the covariate scatter is shrunk by its own signal-to-noise ratio,

            f_k = beta_k^2 / (beta_k^2 + sigma^2 / (n * lambda_k)),

        the posterior-mean factor for a coefficient observed with sampling variance
        ``sigma^2 / (n lambda_k)``. This matters in practice, not just cosmetically: the
        admitted covariates are strongly collinear (they are all functions of the same
        block, all carrying the same HT weight), so the near-degenerate directions have
        essentially unbounded coefficient noise. Feeding that noise into the adjustment
        *increases* ``Var(Y~)`` -- a control variate with a badly estimated ``phi`` is
        worse than none. The shrinkage sends those directions to zero and leaves the
        well-determined ones untouched, so the realised variance ratio degrades gracefully
        to 1 instead of blowing past it. It is computed from strictly-prior data, so it
        is predictable and the unbiasedness argument is unaffected.
        """
        if st._phi_at_n == st.n_fit:
            return st._phi

        n = st.n_fit
        phi = np.zeros(self.d, dtype=float)
        if n >= max(self.warmup, 2) and n > self.d + 1:
            mean_c = st.sum_c / n
            mean_z = st.sum_z / n
            scatter = st.sum_cc - n * np.outer(mean_c, mean_c)
            scatter = 0.5 * (scatter + scatter.T)
            rhs = st.sum_cz - n * mean_c * mean_z
            var_z = max((st.sum_zz - n * mean_z * mean_z) / n, 0.0)

            diag = np.clip(np.diag(scatter) / n, 0.0, None)
            sd = np.sqrt(diag)
            good = sd > 1e-10 * max(1.0, float(sd.max(initial=0.0)))
            if var_z > _EPS and good.any():
                idx = np.flatnonzero(good)
                sdg = sd[idx]
                corr = scatter[np.ix_(idx, idx)] / (n * np.outer(sdg, sdg))
                gvec = rhs[idx] / (n * sdg)
                try:
                    evals, evecs = _linalg.eigh(0.5 * (corr + corr.T))
                except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - defensive
                    evals, evecs = np.linalg.eigh(0.5 * (corr + corr.T))
                evals = np.clip(np.asarray(evals, dtype=float), 0.0, None)
                lam = self.ridge * max(1.0, float(evals.max(initial=1.0)))
                denom = evals + lam
                proj = evecs.T @ gvec
                beta = np.where(denom > _EPS, proj / np.where(denom > _EPS, denom, 1.0), 0.0)
                explained = float(np.sum(proj * beta))
                sigma2 = max(var_z - explained, var_z * 1e-6)
                signal = n * evals * beta * beta
                shrink = np.where(signal > 0.0, signal / (signal + sigma2), 0.0)
                phi_std = evecs @ (shrink * beta)
                phi[idx] = phi_std / sdg

            if not np.all(np.isfinite(phi)):
                logger.warning("CupedAdjuster: non-finite phi from the fit; falling back to 0")
                phi = np.zeros(self.d, dtype=float)

        st._phi = phi
        st._phi_at_n = st.n_fit
        return phi

    def _adjusted(self, st: _PairState, z: float, cov: np.ndarray | None) -> tuple[float, bool]:
        """Apply the current (predictable) ``phi``. Returns ``(ytil, clipped)``."""
        zf = float(z)
        if cov is None:
            return zf, False
        phi = self._fit(st)
        if not np.any(phi):
            return zf, False
        raw_adj = float(phi @ (cov - self._effective_means(st)))
        if not np.isfinite(raw_adj):
            return zf, False
        limit = self.clip * st.scale()
        adj = float(np.clip(raw_adj, -limit, limit))
        return zf - adj, bool(adj != raw_adj)

    # -- public API (SPEC section 8) ----------------------------------------------------

    def adjust(self, pair: Any, z: float, cov_vec: np.ndarray | None) -> float:
        """Return ``Y~ = z - phi^T (cov_vec - E[C])`` using the ``F_{i-1}``-measurable phi.

        **Pure**: this method mutates nothing. The ``phi`` it applies was fit from all
        observations recorded *before* this one, so ``E[Y~_i | F_{i-1}] = E[z_i | F_{i-1}]
        = Delta`` and the outer confidence sequence keeps its anytime coverage. Call
        :meth:`observe` (or use :meth:`adjust_and_observe`) *afterwards* to record the point.

        ``cov_vec=None`` is allowed and returns ``z`` untouched -- useful for the ablation
        conditions that run the same channel with CUPED disabled.

        Claim: variance-reduction
        """
        key = self._key(pair)
        st = self._state(key, create=False)
        cov = self._check_cov(cov_vec)
        ytil, _ = self._adjusted(st, z, cov)
        return ytil

    def observe(self, pair: Any, z: float, cov_vec: np.ndarray | None) -> None:
        """Record one ``(z, cov_vec)`` point; this is the only method that mutates state.

        The realised adjusted value is computed here with the *pre-update* ``phi`` -- the
        identical value :meth:`adjust` returned for this point -- and only then are the
        sufficient statistics advanced. So the point being recorded can influence ``phi``
        for step ``i+1`` onwards and never for step ``i``.

        Claim: variance-reduction
        """
        key = self._key(pair, register=True)
        st = self._state(key)
        cov = self._check_cov(cov_vec)
        zf = float(z)
        if not np.isfinite(zf):
            raise ValueError(f"z must be finite, got {z!r}")

        phi_was_active = bool(np.any(self._fit(st))) and cov is not None
        ytil, clipped = self._adjusted(st, zf, cov)

        st.z_hist.append(zf)
        st.c_hist.append(None if cov is None else cov.copy())
        st.ytil_hist.append(ytil)
        st.active_hist.append(phi_was_active)
        st.n_total += 1
        st.sum_z += zf
        st.sum_abs_z += abs(zf)
        st.sum_zz += zf * zf
        if clipped:
            st.n_clipped += 1
        if cov is not None:
            st.n_fit += 1
            st.sum_c += cov
            st.sum_cc += np.outer(cov, cov)
            st.sum_cz += cov * zf

    def adjust_and_observe(self, pair: Any, z: float, cov_vec: np.ndarray | None) -> float:
        """Convenience: :meth:`adjust` then :meth:`observe`, in the only valid order.

        Callers that use this cannot accidentally leak the current point into ``phi``.

        Claim: variance-reduction
        """
        ytil = self.adjust(pair, z, cov_vec)
        self.observe(pair, z, cov_vec)
        return ytil

    def theta(self, pair: Any) -> np.ndarray:
        """The current control-variate coefficient vector ``phi`` for ``pair`` (a copy).

        Zero during warmup. Fit from every observation recorded so far and therefore
        ``F_{i-1}``-measurable with respect to the next observation.

        Claim: variance-reduction
        """
        st = self._state(self._key(pair), create=False)
        return self._fit(st).copy()

    def phi(self, pair: Any) -> np.ndarray:
        """Alias of :meth:`theta`, named after the symbol used in the docs.

        Claim: variance-reduction
        """
        return self.theta(pair)

    def n_observations(self, pair: Any) -> int:
        """Number of points recorded for ``pair``.

        Claim: variance-reduction
        """
        return self._state(self._key(pair), create=False).n_total

    def clip_rate(self, pair: Any) -> float:
        """Fraction of recorded points whose adjustment hit the clipping rail.

        A run should report this as ~0; that is the evidence that the (bounded, symmetric)
        conservatism the rail can introduce was in fact never exercised.

        Claim: variance-reduction
        """
        st = self._state(self._key(pair), create=False)
        if st.n_total == 0:
            return 0.0
        return st.n_clipped / st.n_total

    def max_adjustment(self, pair: Any) -> float:
        """Current absolute bound on the adjustment, ``clip * running_scale``.

        A channel that wants provably valid bounds for the confidence sequence can widen
        ``[L, U]`` by this amount; it is a predictable quantity.

        Claim: variance-reduction
        """
        st = self._state(self._key(pair), create=False)
        return self.clip * st.scale()

    def variance_ratio(self, pair: Any) -> float:
        """``Var(Y~) / Var(Y)`` measured on the collected history of ``pair``.

        Measured over the **effective** sub-history: the points at which ``phi`` was
        actually non-zero (i.e. after warmup, with a covariate present). Including the
        warmup points, where ``Y~ == Y`` by construction, would dilute the number with a
        stretch during which no adjustment was even attempted; that dilution is reported
        separately as ``ratio_all`` in :func:`variance_reduction_report`.

        Returns ``1.0`` (no measured reduction) when there is not enough history. With a
        single covariate and a near-optimal ``phi`` this should track ``1 - corr^2``.

        Claim: variance-reduction
        """
        st = self._state(self._key(pair), create=False)
        raw, adj = self._effective_arrays(st)
        return _var_ratio(raw, adj)

    def correlation(self, pair: Any) -> float:
        """Achieved correlation between the raw outcome and the covariates.

        Signed Pearson correlation when there is a single covariate; the (non-negative)
        multiple correlation ``R = sqrt(R^2)`` otherwise. The variance ratio should be
        close to ``1 - correlation^2``.

        Claim: variance-reduction
        """
        st = self._state(self._key(pair), create=False)
        return _achieved_correlation(st)

    def reset(self, pair: Any | None = None) -> None:
        """Drop history for one pair, or for all pairs when ``pair is None``.

        Claim: infrastructure
        """
        if pair is None:
            self._states.clear()
            self._seen.clear()
            return
        key = self._key(pair)
        self._states.pop(key, None)
        if key in self._seen:
            self._seen.remove(key)

    # -- helpers used by the report -----------------------------------------------------

    def _effective_arrays(self, st: _PairState) -> tuple[np.ndarray, np.ndarray]:
        """Raw and realised-adjusted outcomes over the points where ``phi`` was active."""
        if st.n_total == 0:
            return np.zeros(0), np.zeros(0)
        mask = np.asarray(st.active_hist, dtype=bool)
        raw = np.asarray(st.z_hist, dtype=float)
        adj = np.asarray(st.ytil_hist, dtype=float)
        if not mask.any():
            return np.zeros(0), np.zeros(0)
        return raw[mask], adj[mask]

    def _all_arrays(self, st: _PairState) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(st.z_hist, dtype=float)
        adj = np.asarray(st.ytil_hist, dtype=float)
        return raw, adj

    def report(self) -> dict[PairKey, dict[str, Any]]:
        """Per-pair variance-reduction diagnostics; see :func:`variance_reduction_report`.

        Claim: variance-reduction
        """
        return variance_reduction_report(self)

    # -- covariate construction, re-exported as static methods for convenience ----------

    build_covariate_vector = staticmethod(build_covariate_vector)
    known_covariate_means = staticmethod(known_covariate_means)
    covariate_names = staticmethod(covariate_names)
    pilot_difficulty_from_predictions = staticmethod(pilot_difficulty_from_predictions)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _sample_var(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.var(x, ddof=1))


def _var_ratio(raw: np.ndarray, adj: np.ndarray) -> float:
    v_raw = _sample_var(raw)
    v_adj = _sample_var(adj)
    if v_raw <= _EPS:
        return 1.0
    return float(v_adj / v_raw)


def _achieved_correlation(st: _PairState) -> float:
    """Multiple correlation between the raw outcome and the covariates, on the fit history."""
    rows = [(z, c) for z, c in zip(st.z_hist, st.c_hist) if c is not None]
    if len(rows) < 3:
        return 0.0
    z = np.asarray([r[0] for r in rows], dtype=float)
    cmat = np.vstack([r[1] for r in rows])
    zc = z - z.mean()
    sstot = float(zc @ zc)
    if sstot <= _EPS:
        return 0.0
    cc = cmat - cmat.mean(axis=0, keepdims=True)
    if cmat.shape[1] == 1:
        col = cc[:, 0]
        denom = float(np.sqrt((col @ col) * sstot))
        if denom <= _EPS:
            return 0.0
        return float((col @ zc) / denom)
    try:
        coef, *_ = np.linalg.lstsq(cc, zc, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return 0.0
    resid = zc - cc @ coef
    ssres = float(resid @ resid)
    r2 = max(0.0, min(1.0, 1.0 - ssres / sstot))
    return float(np.sqrt(r2))


def _pitman_morgan_p(raw: np.ndarray, adj: np.ndarray) -> float:
    """One-sided p-value for ``Var(adj) < Var(raw)`` on *paired* samples.

    Pitman-Morgan: for paired ``(X, Y)`` the difference in variances is testable through
    ``corr(X - Y, X + Y)``; ``t = r sqrt((n-2)/(1-r^2))`` on ``n-2`` degrees of freedom.
    Reported as a diagnostic -- the validity of the procedure does not depend on it.
    """
    n = raw.size
    if n < 4 or adj.size != n:
        return 1.0
    u = raw - adj
    v = raw + adj
    su = float(np.std(u, ddof=1))
    sv = float(np.std(v, ddof=1))
    if su <= _EPS or sv <= _EPS:
        return 1.0
    r = float(np.corrcoef(u, v)[0, 1])
    if not np.isfinite(r):
        return 1.0
    r = float(np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12))
    tstat = r * np.sqrt((n - 2) / (1.0 - r * r))
    return float(_stats.t.sf(tstat, df=n - 2))


def variance_reduction_report(adjuster: CupedAdjuster) -> dict[PairKey, dict[str, Any]]:
    """Per-pair measured variance reduction achieved by the control variates.

    Returns ``{pair: {...}}`` with, for each pair that has been observed:

    ``n``
        total number of points recorded for the pair;
    ``n_effective``
        number of points over which the variances below are measured, i.e. those where
        ``phi`` was actually non-zero (post-warmup, covariate present);
    ``var_raw``
        sample variance of the raw HT contrast ``Y`` on that sub-history;
    ``var_adjusted``
        sample variance of the realised ``Y~`` on the same points (realised: computed with
        the ``phi`` that was in force at the time, not with hindsight);
    ``ratio``
        ``var_adjusted / var_raw`` -- identical to :meth:`CupedAdjuster.variance_ratio`;
    ``correlation``
        achieved correlation between ``Y`` and the covariates. The theory says
        ``ratio ~= 1 - correlation^2`` for a near-optimal ``phi``; ``theoretical_ratio``
        is that value, so a benchmark can plot the two against each other.

    Plus ``ratio_all`` (the same ratio including warmup, which is necessarily closer to 1),
    ``p_value`` (one-sided Pitman-Morgan test for a genuine variance reduction),
    ``clip_rate``, ``phi`` and ``n_covariates``. Everything reported here is measured from
    the collected history -- nothing is assumed.

    With ``per_pair=False`` every pair shares one fit, so all entries carry the same
    (pooled) numbers; the keys still list the pairs that were observed.

    Claim: variance-reduction
    """
    out: dict[PairKey, dict[str, Any]] = {}
    seen = adjuster.pairs()
    if not seen and adjuster._states:
        seen = list(adjuster._states.keys())

    for pair in seen:
        key = adjuster._key(pair)
        st = adjuster._state(key, create=False)
        raw_eff, adj_eff = adjuster._effective_arrays(st)
        raw_all, adj_all = adjuster._all_arrays(st)
        corr = _achieved_correlation(st)
        ratio = _var_ratio(raw_eff, adj_eff)
        out[tuple(pair)] = {
            "pair": tuple(pair),
            "n": int(st.n_total),
            "n_effective": int(raw_eff.size),
            "n_warmup": int(min(adjuster.warmup, st.n_total)),
            "var_raw": _sample_var(raw_eff),
            "var_adjusted": _sample_var(adj_eff),
            "ratio": ratio,
            "correlation": float(corr),
            "theoretical_ratio": float(max(0.0, 1.0 - corr * corr)),
            "ratio_all": _var_ratio(raw_all, adj_all),
            "var_raw_all": _sample_var(raw_all),
            "var_adjusted_all": _sample_var(adj_all),
            "p_value": _pitman_morgan_p(raw_eff, adj_eff),
            "clip_rate": float(st.n_clipped / st.n_total) if st.n_total else 0.0,
            "phi": [float(v) for v in adjuster._fit(st)],
            "n_covariates": int(adjuster.d),
            "covariate_names": list(adjuster.names),
        }
    return out
