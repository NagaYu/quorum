"""Hierarchical partial-pooling cell model, fit by MAP + Laplace (no MCMC).

**This model never enters the validity argument.** Quorum's inference is *design-based
and model-assisted*: the confidence sequences of :mod:`quorum.nested_seq` are valid
because of the sampling design (Horvitz-Thompson reweighting plus a martingale
argument), not because this model is right. The cell model exists purely to *steer* the
allocator -- to say which rollout looks most worth buying next. If the model is badly
misspecified the allocator simply spends money less cleverly; the coverage guarantee,
the unbiasedness of the contrast estimator, and the family-wise error control are all
untouched. **A wrong model costs efficiency, never coverage.**

The linear predictor (SPEC section 6) is::

    logit p[t, k, s, m] = mu + alpha[t] + beta[m] + gamma[s] + zeta[t, m] + delta[s, m]

with Gaussian priors ``alpha ~ N(0, tau_task^2)``, ``beta ~ N(0, tau_model^2)``,
``gamma ~ N(0, tau_scaffold^2)``, ``zeta ~ N(0, tau_tm^2)``, ``delta ~ N(0, tau_sm^2)``
and ``mu ~ N(0, 5^2)``.

**The seed dimension is deliberately pooled.** There is no free per-seed parameter: a
seed's contribution is absorbed by ``alpha[t]`` (plus, in the simulator's truth, an
``eta[t,k]`` jitter that this model treats as noise). That is the partial pooling that
makes *unvisited* blocks predictable -- the whole point of a hierarchical model here is
that buying a rollout at seed 0 tells you something about seed 3 of the same task. Give
every seed its own parameter and the model would fit the visited blocks perfectly and
say nothing at all about the unvisited ones, which is exactly the regime the allocator
operates in. A consequence, stated plainly so nobody is surprised by it: predicted
probabilities and information gains are constant across ``k`` within a
``(task, scaffold, model)`` triple, so seeds are differentiated by *cost* alone in the
allocator's score.

Fitting is Newton / IRLS on the penalised Bernoulli log-likelihood over **aggregated
per-cell (successes, trials) counts**, so an iteration is O(n_cells) no matter how many
rollouts have been observed. The posterior covariance is the Laplace approximation
``Sigma = H^{-1}`` at the mode, which is what makes the closed-form value-of-information
of SPEC section 7.1 a rank-1 update rather than a refit.

Claim: low-cost
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import scipy.linalg as sla
from scipy.special import expit

from quorum.types import GridSpec, Rollout

__all__ = [
    "HierarchicalCellModel",
    "TAU_MIN",
    "TAU_MAX",
    "MU_PRIOR_SD",
    "BLOCK_NAMES",
]

logger = logging.getLogger(__name__)

#: Variance components are clipped to this range (SPEC section 6).
TAU_MIN: float = 0.05
TAU_MAX: float = 5.0

#: Fixed prior sd on the global intercept ``mu`` (never empirical-Bayes updated).
MU_PRIOR_SD: float = 5.0

#: Names of the coefficient blocks, in parameter-vector order after ``mu``.
BLOCK_NAMES: tuple[str, ...] = ("task", "model", "scaffold", "tm", "sm")

#: Number of non-zero entries in a cell's one-hot design row.
_N_DESIGN_TERMS: int = 6

#: Above this many dense design entries we never materialise ``X``.
_DENSE_MAX_ELEMS: int = 2_000_000

_TAU_KEYS: tuple[str, ...] = (
    "tau_task",
    "tau_model",
    "tau_scaffold",
    "tau_tm",
    "tau_sm",
)


class HierarchicalCellModel:
    """Partial-pooling Bayesian model over evaluation cells, fit by MAP + Laplace.

    The model is a steering device for :class:`quorum.allocator.DecisionTargetedAllocator`
    and nothing more -- see the module docstring for why its correctness is irrelevant to
    Quorum's coverage guarantee. What it must be is *cheap* and *smooth*: cheap because
    the allocator refits it repeatedly inside the sequential loop, and smooth because the
    Laplace covariance turns "how much would this rollout help the ranking decision?" into
    a closed-form rank-1 expression instead of a simulation.

    Observations are folded into per-cell ``(successes, trials)`` counts by :meth:`update`
    and :meth:`add`, which are O(1) each; :meth:`fit` does the actual optimisation and
    caches its result until new data arrives (a dirty flag). Every read-only accessor
    fits lazily, so callers never have to remember to call :meth:`fit`.

    Because the seed dimension is pooled (no free per-seed parameter), the internal
    design has one row per ``(task, scaffold, model)`` triple rather than per cell; the
    per-cell counts are summed over seeds before the Newton loop. Outputs are still
    reported on the full ``(T, K, S, M)`` grid.

    Args:
        spec: Grid shape.
        tau_init: Initial variance-component sds. A scalar applies to every block, or a
            mapping over ``tau_task``/``tau_model``/``tau_scaffold``/``tau_tm``/``tau_sm``.
        empirical_bayes: If True, refine ``tau_*`` with EB fixed-point updates.
        max_iter: Hard cap on Newton iterations per inner fit.
        ridge: Ridge added to the Hessian diagonal for numerical stability only.
        eb_rounds: Number of outer EB rounds when ``empirical_bayes`` is True.
        tol: Convergence tolerance on the gradient 2-norm.

    Claim: low-cost
    """

    def __init__(
        self,
        spec: GridSpec,
        *,
        tau_init: float | Mapping[str, float] = 1.0,
        empirical_bayes: bool = True,
        max_iter: int = 50,
        ridge: float = 1e-6,
        eb_rounds: int = 5,
        tol: float = 1e-7,
    ) -> None:
        """Build the parameter layout, the one-hot design and the zeroed count arrays.

        Nothing is fitted here: an unfitted model is already well posed (the mode is the
        prior mean, the covariance the prior covariance), so the allocator can query it
        before a single rollout has been bought.

        See the class docstring for the argument semantics.

        Claim: low-cost
        """
        self.spec: GridSpec = spec
        self.empirical_bayes: bool = bool(empirical_bayes)
        self.max_iter: int = int(max_iter)
        self.ridge: float = float(ridge)
        self.eb_rounds: int = int(eb_rounds)
        self.tol: float = float(tol)

        T, K, S, M = spec.grid_shape
        self._T, self._K, self._S, self._M = T, K, S, M

        # --- parameter layout -------------------------------------------------
        self._o_mu = 0
        self._o_alpha = 1
        self._o_beta = self._o_alpha + T
        self._o_gamma = self._o_beta + M
        self._o_zeta = self._o_gamma + S
        self._o_delta = self._o_zeta + T * M
        self._n_params = self._o_delta + S * M

        self._slices: dict[str, slice] = {
            "mu": slice(self._o_mu, self._o_mu + 1),
            "task": slice(self._o_alpha, self._o_alpha + T),
            "model": slice(self._o_beta, self._o_beta + M),
            "scaffold": slice(self._o_gamma, self._o_gamma + S),
            "tm": slice(self._o_zeta, self._o_zeta + T * M),
            "sm": slice(self._o_delta, self._o_delta + S * M),
        }

        # --- design (index form; one row per (t, s, m), seeds pooled) ---------
        self._n_rows = T * S * M
        self._idx = self._build_design_indices()
        self._pair_flat = (
            self._idx[:, :, None] * self._n_params + self._idx[:, None, :]
        ).ravel()
        self._grad_scatter = self._idx.ravel()

        self._dense_X: np.ndarray | None = None
        if self._n_rows * self._n_params <= _DENSE_MAX_ELEMS:
            X = np.zeros((self._n_rows, self._n_params), dtype=float)
            rows = np.repeat(np.arange(self._n_rows), _N_DESIGN_TERMS)
            np.add.at(X, (rows, self._idx.ravel()), 1.0)
            self._dense_X = X
        else:  # pragma: no cover - only on very large grids
            logger.debug(
                "cell model: using index-form design (%d rows x %d params)",
                self._n_rows,
                self._n_params,
            )

        # --- counts -----------------------------------------------------------
        self._successes = np.zeros(spec.grid_shape, dtype=float)
        self._trials = np.zeros(spec.grid_shape, dtype=float)
        self._n_obs: int = 0

        # --- state ------------------------------------------------------------
        self._tau = self._init_tau(tau_init)
        self._theta = np.zeros(self._n_params, dtype=float)
        self._Sigma = np.diag(self._prior_variances())
        self._eta_rows = np.zeros(self._n_rows, dtype=float)
        self._dirty: bool = True
        self._n_newton: int = 0
        self._converged: bool = False

    # ------------------------------------------------------------------ setup

    def _build_design_indices(self) -> np.ndarray:
        """Index form of the one-hot design: ``(n_rows, 6)`` parameter positions.

        Row ``r = (t * S + s) * M + m`` carries the six active coefficients
        ``[mu, alpha_t, beta_m, gamma_s, zeta_{t,m}, delta_{s,m}]``. Never a dense matrix
        unless the grid is small enough that dense linear algebra is simply faster.

        Claim: infrastructure
        """
        T, S, M = self._T, self._S, self._M
        t = np.repeat(np.arange(T), S * M)
        s = np.tile(np.repeat(np.arange(S), M), T)
        m = np.tile(np.arange(M), T * S)
        idx = np.empty((self._n_rows, _N_DESIGN_TERMS), dtype=np.intp)
        idx[:, 0] = self._o_mu
        idx[:, 1] = self._o_alpha + t
        idx[:, 2] = self._o_beta + m
        idx[:, 3] = self._o_gamma + s
        idx[:, 4] = self._o_zeta + t * M + m
        idx[:, 5] = self._o_delta + s * M + m
        return idx

    def _init_tau(self, tau_init: float | Mapping[str, float]) -> dict[str, float]:
        if isinstance(tau_init, Mapping):
            tau = {k: float(tau_init.get(k, 1.0)) for k in _TAU_KEYS}
        else:
            tau = {k: float(tau_init) for k in _TAU_KEYS}
        return {k: float(np.clip(v, TAU_MIN, TAU_MAX)) for k, v in tau.items()}

    def _prior_variances(self) -> np.ndarray:
        """Diagonal of the Gaussian prior covariance, in parameter order.

        Claim: infrastructure
        """
        v = np.empty(self._n_params, dtype=float)
        v[self._slices["mu"]] = MU_PRIOR_SD**2
        v[self._slices["task"]] = self._tau["tau_task"] ** 2
        v[self._slices["model"]] = self._tau["tau_model"] ** 2
        v[self._slices["scaffold"]] = self._tau["tau_scaffold"] ** 2
        v[self._slices["tm"]] = self._tau["tau_tm"] ** 2
        v[self._slices["sm"]] = self._tau["tau_sm"] ** 2
        return v

    # ------------------------------------------------------------- data intake

    def add(self, t: int, k: int, s: int, m: int, success: int) -> None:
        """Fold a single Bernoulli outcome into the cell counts (O(1), marks dirty).

        Costs nothing but two array increments; the expensive work is deferred to
        :meth:`fit`, which the allocator calls on a refit schedule rather than per
        rollout.

        Claim: low-cost
        """
        self._successes[t, k, s, m] += float(success)
        self._trials[t, k, s, m] += 1.0
        self._n_obs += 1
        self._dirty = True

    def update(self, rollouts_or_counts: Any) -> None:
        """Incrementally absorb new evidence. Accepts several shapes, all O(input).

        Understood inputs:

        * a single :class:`~quorum.types.Rollout` or any iterable of them;
        * a ``(successes, trials)`` pair of arrays broadcastable to ``(T, K, S, M)``,
          interpreted as *additional* counts to accumulate;
        * a mapping ``{(t, k, s, m): success}`` or ``{(t, k, s, m): (succ, trials)}``.

        Rollout outcomes are taken from ``Rollout.success`` as observed -- including the
        inner truncation rule's guess for a truncated rollout. That is deliberate: this
        model only steers allocation, so a slightly pessimistic training signal costs
        efficiency, never validity. The de-biasing correction belongs to the estimator
        (:func:`quorum.nested_seq.corrected_outcome`), not here.

        Claim: low-cost
        """
        obj = rollouts_or_counts
        if obj is None:
            return
        if isinstance(obj, Rollout):
            self._add_rollout(obj)
        elif isinstance(obj, Mapping):
            self._add_mapping(obj)
        elif isinstance(obj, tuple) and len(obj) == 2 and not isinstance(obj[0], Rollout):
            succ = np.broadcast_to(
                np.asarray(obj[0], dtype=float), self.spec.grid_shape
            )
            trials = np.broadcast_to(
                np.asarray(obj[1], dtype=float), self.spec.grid_shape
            )
            self._successes += succ
            self._trials += trials
            self._n_obs += int(round(float(trials.sum())))
            self._dirty = True
        elif isinstance(obj, np.ndarray):
            raise TypeError(
                "update() needs a (successes, trials) pair, not a bare array"
            )
        elif isinstance(obj, (Iterable, Sequence)):
            for item in obj:
                self.update(item)
        else:
            raise TypeError(f"cannot update HierarchicalCellModel from {type(obj)!r}")

    def _add_rollout(self, r: Rollout) -> None:
        self.add(r.task, r.seed, r.scaffold, r.model, int(r.success))

    def _add_mapping(self, mapping: Mapping[Any, Any]) -> None:
        for key, value in mapping.items():
            t, k, s, m = (int(v) for v in key)
            if isinstance(value, (tuple, list, np.ndarray)):
                succ, trials = float(value[0]), float(value[1])
            else:
                succ, trials = float(value), 1.0
            self._successes[t, k, s, m] += succ
            self._trials[t, k, s, m] += trials
            self._n_obs += int(round(trials))
        self._dirty = True

    # --------------------------------------------------------------- the fit

    def fit(self) -> None:
        """Fit the MAP mode and its Laplace covariance; a no-op if nothing changed.

        Damped Newton / IRLS on the penalised Bernoulli log-likelihood over aggregated
        counts, with an Armijo backtracking line search, a ridge on the Hessian for
        conditioning, convergence on the gradient norm and a hard iteration cap. When
        ``empirical_bayes`` is on, the variance components are refreshed between Newton
        solves by the fixed point ``tau^2 <- (||coef||^2 + tr Sigma_block) / n_block``,
        clipped to ``[0.05, 5]``.

        Claim: low-cost
        """
        if not self._dirty:
            return

        succ_rows, trial_rows = self._row_counts()
        rounds = self.eb_rounds if self.empirical_bayes else 1
        theta = self._theta.copy()

        for round_i in range(max(1, rounds)):
            prior_var = self._prior_variances()
            theta, H = self._newton(theta, succ_rows, trial_rows, prior_var)
            Sigma = _safe_inverse(H)
            if not self.empirical_bayes:
                break
            changed = self._eb_update(theta, Sigma)
            if not changed:
                break
            if round_i == rounds - 1:
                # One final Newton pass at the last accepted tau, so the mode and the
                # covariance are consistent with the same prior.
                prior_var = self._prior_variances()
                theta, H = self._newton(theta, succ_rows, trial_rows, prior_var)
                Sigma = _safe_inverse(H)

        self._theta = theta
        self._Sigma = Sigma
        self._eta_rows = self._eta(theta)
        self._dirty = False

    def _row_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Counts aggregated over the pooled seed axis, flattened to design rows.

        Claim: infrastructure
        """
        succ = self._successes.sum(axis=1).reshape(self._n_rows)
        trials = self._trials.sum(axis=1).reshape(self._n_rows)
        return succ, trials

    def _eta(self, theta: np.ndarray) -> np.ndarray:
        return theta[self._idx].sum(axis=1)

    def _neg_log_post(
        self,
        theta: np.ndarray,
        succ: np.ndarray,
        trials: np.ndarray,
        prior_var: np.ndarray,
    ) -> float:
        eta = self._eta(theta)
        ll = float(np.sum(trials * np.logaddexp(0.0, eta) - succ * eta))
        pen = 0.5 * float(np.sum(theta * theta / prior_var))
        return ll + pen

    def _newton(
        self,
        theta0: np.ndarray,
        succ: np.ndarray,
        trials: np.ndarray,
        prior_var: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Damped Newton / IRLS. Returns the mode and the penalised Hessian there.

        Claim: low-cost
        """
        theta = np.array(theta0, dtype=float, copy=True)
        inv_prior = 1.0 / prior_var
        f_cur = self._neg_log_post(theta, succ, trials, prior_var)
        self._converged = False

        for _ in range(self.max_iter):
            eta = self._eta(theta)
            p = expit(eta)
            resid = succ - trials * p
            grad = self._xt_dot(resid) - inv_prior * theta
            gnorm = float(np.linalg.norm(grad))
            if gnorm <= self.tol:
                self._converged = True
                break

            w = trials * p * (1.0 - p)
            H = self._hessian(w, inv_prior)
            direction = _safe_solve(H, grad)
            slope = float(grad @ direction)
            if not np.isfinite(slope) or slope <= 0.0:
                # Not a descent direction (can only happen through round-off);
                # fall back on a small gradient step.
                direction = grad
                slope = float(grad @ grad)

            step = 1.0
            accepted = False
            for _ls in range(40):
                cand = theta + step * direction
                f_new = self._neg_log_post(cand, succ, trials, prior_var)
                if np.isfinite(f_new) and f_new <= f_cur - 1e-4 * step * slope:
                    theta = cand
                    f_cur = f_new
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                self._converged = True
                break
            if float(np.max(np.abs(step * direction))) < 1e-13:
                self._converged = True
                break
            self._n_newton += 1
        else:
            logger.debug(
                "cell model Newton hit the iteration cap (%d) without reaching tol",
                self.max_iter,
            )

        eta = self._eta(theta)
        p = expit(eta)
        H = self._hessian(trials * p * (1.0 - p), inv_prior)
        return theta, H

    def _xt_dot(self, resid: np.ndarray) -> np.ndarray:
        """``X^T resid`` using the index form (or dense ``X`` on small grids).

        Claim: infrastructure
        """
        if self._dense_X is not None:
            return self._dense_X.T @ resid
        return np.bincount(
            self._grad_scatter,
            weights=np.repeat(resid, _N_DESIGN_TERMS),
            minlength=self._n_params,
        )

    def _hessian(self, w: np.ndarray, inv_prior: np.ndarray) -> np.ndarray:
        """``X^T W X + Lambda^{-1} + ridge I`` -- the penalised Hessian.

        Claim: infrastructure
        """
        P = self._n_params
        if self._dense_X is not None:
            H = self._dense_X.T @ (self._dense_X * w[:, None])
        else:
            H = np.zeros((P, P), dtype=float)
            np.add.at(
                H.reshape(-1),
                self._pair_flat,
                np.repeat(w, _N_DESIGN_TERMS * _N_DESIGN_TERMS),
            )
        H.flat[:: P + 1] += inv_prior + self.ridge
        return H

    def _eb_update(self, theta: np.ndarray, Sigma: np.ndarray) -> bool:
        """One empirical-Bayes fixed-point sweep over the variance components.

        ``tau^2 <- (||coef||^2 + tr Sigma_block) / n_block``, clipped to ``[0.05, 5]``.
        Returns True if any component moved appreciably.

        Claim: low-cost
        """
        moved = False
        diag = np.diag(Sigma)
        for key, name in zip(_TAU_KEYS, BLOCK_NAMES):
            sl = self._slices[name]
            coef = theta[sl]
            n_block = coef.size
            if n_block == 0:
                continue
            tau2 = (float(coef @ coef) + float(diag[sl].sum())) / n_block
            new = float(np.clip(np.sqrt(max(tau2, 1e-12)), TAU_MIN, TAU_MAX))
            if abs(new - self._tau[key]) > 1e-4:
                moved = True
            self._tau[key] = new
        return moved

    # ------------------------------------------------------------- accessors

    @property
    def n_params(self) -> int:
        """Length of the coefficient vector ``[mu, alpha, beta, gamma, zeta, delta]``.

        Claim: infrastructure
        """
        return self._n_params

    @property
    def n_observations(self) -> int:
        """Number of Bernoulli trials folded into the counts so far.

        Claim: infrastructure
        """
        return self._n_obs

    @property
    def counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell ``(successes, trials)`` arrays, each ``(T, K, S, M)`` (copies).

        Claim: infrastructure
        """
        return self._successes.copy(), self._trials.copy()

    @property
    def tau(self) -> dict[str, float]:
        """Current variance components (sds), after any empirical-Bayes updates.

        Claim: low-cost
        """
        self.fit()
        return dict(self._tau)

    @property
    def theta(self) -> np.ndarray:
        """The full MAP coefficient vector, in parameter order.

        Claim: low-cost
        """
        self.fit()
        return self._theta.copy()

    @property
    def mu(self) -> float:
        """Global intercept at the MAP mode.

        Claim: low-cost
        """
        self.fit()
        return float(self._theta[self._o_mu])

    @property
    def alpha(self) -> np.ndarray:
        """Task-difficulty coefficients, shape ``(T,)``.

        Claim: low-cost
        """
        self.fit()
        return self._theta[self._slices["task"]].copy()

    @property
    def beta(self) -> np.ndarray:
        """Model-ability coefficients, shape ``(M,)`` -- the quantity being ranked.

        Claim: low-cost
        """
        self.fit()
        return self._theta[self._slices["model"]].copy()

    @property
    def gamma(self) -> np.ndarray:
        """Scaffold main effects, shape ``(S,)``.

        Scaffolds get their own main effect so a scaffold shift moves absolute scores
        without dragging the ability estimates with it -- the model-side reflection of
        the shift-robustness the estimand already has by pairing.

        Claim: shift-robust
        """
        self.fit()
        return self._theta[self._slices["scaffold"]].copy()

    @property
    def zeta(self) -> np.ndarray:
        """Task x model interactions, shape ``(T, M)``.

        Claim: low-cost
        """
        self.fit()
        return self._theta[self._slices["tm"]].reshape(self._T, self._M).copy()

    @property
    def delta(self) -> np.ndarray:
        """Scaffold x model interactions, shape ``(S, M)`` -- the rank-distorting term.

        Claim: shift-robust
        """
        self.fit()
        return self._theta[self._slices["sm"]].reshape(self._S, self._M).copy()

    @property
    def Sigma(self) -> np.ndarray:  # noqa: N802 - SPEC names this Sigma
        """Laplace posterior covariance ``H^{-1}`` at the mode, shape ``(P, P)``.

        Claim: low-cost
        """
        self.fit()
        return self._Sigma

    @property
    def covariance(self) -> np.ndarray:
        """Alias for :attr:`Sigma`, for callers that prefer a lowercase name.

        Claim: infrastructure
        """
        return self.Sigma

    @property
    def converged(self) -> bool:
        """Whether the last Newton solve met its gradient tolerance.

        Claim: infrastructure
        """
        self.fit()
        return self._converged

    def block_slice(self, name: str) -> slice:
        """Parameter-vector slice of a named coefficient block.

        ``name`` is one of ``mu``, ``task``, ``model``, ``scaffold``, ``tm``, ``sm``.

        Claim: infrastructure
        """
        return self._slices[name]

    def beta_index(self, m: int) -> int:
        """Position of ``beta[m]`` in the full parameter vector.

        Claim: infrastructure
        """
        return self._o_beta + int(m)

    def contrast_vector(self, a: int, c: int) -> np.ndarray:
        """``v = e_a - e_c`` in the FULL parameter space (zeros outside the beta block).

        Claim: infrastructure
        """
        v = np.zeros(self._n_params, dtype=float)
        v[self.beta_index(a)] = 1.0
        v[self.beta_index(c)] = -1.0
        return v

    def cell_design_indices(self, t: int, k: int, s: int, m: int) -> np.ndarray:
        """The six active parameter positions of a cell's one-hot design row.

        ``k`` is accepted and ignored: the seed dimension is pooled, so all seeds of a
        ``(task, scaffold, model)`` triple share a design row. See the module docstring.

        Claim: infrastructure
        """
        return self._idx[self._row_of(t, s, m)].copy()

    def _row_of(self, t: int, s: int, m: int) -> int:
        return (int(t) * self._S + int(s)) * self._M + int(m)

    # ------------------------------------------------------------ predictions

    def predict_logit(self) -> np.ndarray:
        """MAP linear predictor on the full grid, shape ``(T, K, S, M)``.

        Constant across the seed axis by construction (pooled seeds).

        Claim: low-cost
        """
        self.fit()
        eta = self._eta_rows.reshape(self._T, self._S, self._M)
        return np.repeat(eta[:, None, :, :], self._K, axis=1)

    def predict_p(self) -> np.ndarray:
        """Posterior-mean success probability per cell, shape ``(T, K, S, M)``.

        Uses the standard probit/MacKay moderation ``E[sigmoid(eta)] ~=
        sigmoid(eta_hat / sqrt(1 + pi/8 * Var(eta)))`` rather than the plug-in
        ``sigmoid(eta_hat)``, so cells the model has barely seen are correctly shrunk
        towards 0.5 instead of pretending to a confidence the data do not support. Only
        the allocator consumes this, so the approximation is an efficiency choice.

        Claim: low-cost
        """
        self.fit()
        var = self._row_predictive_variance()
        kappa = 1.0 / np.sqrt(1.0 + (np.pi / 8.0) * var)
        p_rows = expit(self._eta_rows * kappa)
        p = p_rows.reshape(self._T, self._S, self._M)
        return np.repeat(p[:, None, :, :], self._K, axis=1)

    def _row_predictive_variance(self) -> np.ndarray:
        """``x^T Sigma x`` for every design row, via fancy indexing (no Python loop).

        Claim: infrastructure
        """
        Sigma = self._Sigma
        sub = Sigma[self._idx[:, :, None], self._idx[:, None, :]]
        return sub.sum(axis=(1, 2))

    def _fisher_weights_rows(self) -> np.ndarray:
        """``u = p(1-p)`` at the mode for every design row (the Fisher weight of 7.1).

        Claim: low-cost
        """
        p = expit(self._eta_rows)
        return p * (1.0 - p)

    # ---------------------------------------------------------- posteriors

    def ability_posterior(
        self, n_draws: int = 512, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Draw abilities from ``N(beta_hat, Sigma_beta)``; shape ``(n_draws, M)``.

        Only the beta sub-block of the Laplace covariance is used. The correlation
        between ``beta`` and ``mu`` is deliberately **not** included: what the ranking
        depends on is *contrasts* ``beta_a - beta_c``, and every such contrast is
        invariant to a common shift of the beta block, so the ``mu``-``beta`` covariance
        cancels out of the decision entirely. Including it would inflate the marginal
        spread of the draws without changing a single pairwise sign probability. For the
        exact contrast variance in the full parameter space, use
        :meth:`contrast_variance`.

        Claim: low-cost
        """
        self.fit()
        gen = _as_rng(rng)
        sl = self._slices["model"]
        mean = self._theta[sl]
        cov = self._Sigma[sl, sl]
        L = _safe_cholesky(cov)
        z = gen.standard_normal((int(n_draws), mean.size))
        return mean[None, :] + z @ L.T

    def rank_posterior(
        self, n_draws: int = 512, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Posterior sign probabilities: ``out[a, c] = P(beta_a > beta_c)``, ``(M, M)``.

        The diagonal is 0.5 by convention (a model is not better than itself, and 0.5 is
        the value that makes "distance from 0.5" a well-behaved flip score). The
        allocator turns this into a per-pair flip probability so that already-resolved
        pairs stop attracting budget -- steering only, never a claim.

        Claim: low-cost
        """
        draws = self.ability_posterior(n_draws=n_draws, rng=rng)
        gt = (draws[:, :, None] > draws[:, None, :]).mean(axis=0)
        np.fill_diagonal(gt, 0.5)
        return gt

    def contrast_variance(self, a: int, c: int) -> float:
        """Posterior variance of the decision contrast: ``v^T Sigma v``, ``v = e_a - e_c``.

        ``v`` lives in the FULL parameter space (a +1 and a -1 in the beta block, zeros
        elsewhere), so this is the exact Laplace variance of ``beta_a - beta_c``
        including their posterior correlation.

        Claim: low-cost
        """
        self.fit()
        ia, ic = self.beta_index(a), self.beta_index(c)
        S = self._Sigma
        return float(S[ia, ia] + S[ic, ic] - 2.0 * S[ia, ic])

    # ------------------------------------------------- value of information

    def info_gain_for_cell(self, t: int, k: int, s: int, m: int, a: int, c: int) -> float:
        """Exact drop in ``Var(beta_a - beta_c)`` from one more rollout at this cell.

        SPEC section 7.1 verbatim. Adding one Bernoulli observation with design row ``x``
        and Fisher weight ``u = p(1-p)`` gives the rank-1 posterior update
        ``Sigma' = Sigma - u Sigma x x^T Sigma / (1 + u x^T Sigma x)``, so::

            gain = u * (v^T Sigma x)^2 / (1 + u * x^T Sigma x)   >= 0

        This is a *c-optimal*, decision-targeted criterion: it scores a cell by how much
        it shrinks the uncertainty of the contrast that decides the ranking, not by how
        much it shrinks uncertainty overall. O(1) per cell given ``Sigma x``.

        ``k`` is accepted and ignored (pooled seeds); seeds differ only in cost.

        Claim: low-cost
        """
        self.fit()
        idx = self._idx[self._row_of(t, s, m)]
        Sigma = self._Sigma
        ia, ic = self.beta_index(a), self.beta_index(c)
        # v^T Sigma x  with v = e_a - e_c and x the sum of the one-hot columns.
        sx_v = float(Sigma[ia, idx].sum() - Sigma[ic, idx].sum())
        xsx = float(Sigma[np.ix_(idx, idx)].sum())
        u = float(self._fisher_weights_rows()[self._row_of(t, s, m)])
        return u * sx_v * sx_v / (1.0 + u * xsx)

    def info_gain_all_cells(self, a: int, c: int) -> np.ndarray:
        """Vectorised :meth:`info_gain_for_cell` over every cell; shape ``(T, K, S, M)``.

        This is what the allocator actually calls on each refit, so it never loops in
        Python: ``Sigma v`` is formed once, ``v^T Sigma x`` becomes a gather-and-sum over
        the six design indices per row, and ``x^T Sigma x`` a ``6 x 6`` gather. Cost is
        O(n_rows) with a constant of 36, independent of the parameter count.

        Values are constant across the seed axis (pooled seeds); the allocator breaks
        that tie with per-cell cost.

        Claim: low-cost
        """
        self.fit()
        Sigma = self._Sigma
        ia, ic = self.beta_index(a), self.beta_index(c)
        sv = Sigma[ia, :] - Sigma[ic, :]            # (P,) == v^T Sigma
        sx_v = sv[self._idx].sum(axis=1)            # (n_rows,) == v^T Sigma x
        xsx = self._row_predictive_variance()       # (n_rows,) == x^T Sigma x
        u = self._fisher_weights_rows()
        gain_rows = u * sx_v * sx_v / (1.0 + u * xsx)
        gain = gain_rows.reshape(self._T, self._S, self._M)
        return np.repeat(gain[:, None, :, :], self._K, axis=1)


# --------------------------------------------------------------------- helpers


def _as_rng(rng: np.random.Generator | None) -> np.random.Generator:
    """Return ``rng``, or a fixed-seed generator so behaviour stays reproducible.

    Claim: infrastructure
    """
    if rng is None:
        logger.debug("no rng supplied; falling back to default_rng(0) for determinism")
        return np.random.default_rng(0)
    return rng


def _safe_solve(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Solve ``H d = g`` for a symmetric positive-definite ``H``, with jitter fallback.

    Claim: infrastructure
    """
    scale = float(np.mean(np.abs(np.diag(H)))) or 1.0
    jitter = 0.0
    for _ in range(6):
        try:
            A = H if jitter == 0.0 else H + jitter * np.eye(H.shape[0])
            cf = sla.cho_factor(A, lower=True, check_finite=False)
            d = sla.cho_solve(cf, g, check_finite=False)
            if np.all(np.isfinite(d)):
                return d
        except (sla.LinAlgError, np.linalg.LinAlgError, ValueError):
            pass
        jitter = scale * 1e-10 if jitter == 0.0 else jitter * 100.0
    logger.debug("cho_factor failed repeatedly; falling back to pseudo-inverse solve")
    return np.linalg.pinv(H) @ g


def _safe_inverse(H: np.ndarray) -> np.ndarray:
    """Invert the penalised Hessian to get the Laplace covariance, with jitter fallback.

    Claim: infrastructure
    """
    n = H.shape[0]
    eye = np.eye(n)
    scale = float(np.mean(np.abs(np.diag(H)))) or 1.0
    jitter = 0.0
    for _ in range(6):
        try:
            A = H if jitter == 0.0 else H + jitter * eye
            cf = sla.cho_factor(A, lower=True, check_finite=False)
            Sigma = sla.cho_solve(cf, eye, check_finite=False)
            if np.all(np.isfinite(Sigma)):
                return 0.5 * (Sigma + Sigma.T)
        except (sla.LinAlgError, np.linalg.LinAlgError, ValueError):
            pass
        jitter = scale * 1e-10 if jitter == 0.0 else jitter * 100.0
    logger.warning("Laplace Hessian was not invertible; using pseudo-inverse")
    Sigma = np.linalg.pinv(H)
    return 0.5 * (Sigma + Sigma.T)


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of a covariance block, with jitter then eigen fallback.

    Claim: infrastructure
    """
    cov = 0.5 * (cov + cov.T)
    scale = float(np.mean(np.abs(np.diag(cov)))) or 1.0
    jitter = 0.0
    for _ in range(6):
        try:
            A = cov if jitter == 0.0 else cov + jitter * np.eye(cov.shape[0])
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            jitter = scale * 1e-12 if jitter == 0.0 else jitter * 100.0
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    return vecs * np.sqrt(vals)[None, :]
