"""Claim (4): inner truncation saves compute without biasing the outer estimator.

The interesting half of this claim is that unbiasedness does **not** depend on the
truncation rule being any good. A deliberately miscalibrated ``q`` must still produce an
unbiased estimate -- miscalibration is allowed to cost variance and nothing else. If that
failed, the nested design would be unusable, because nobody can calibrate a stopping rule
perfectly in advance.

Claim: unbiased-truncation
"""

from __future__ import annotations

import numpy as np
import pytest

from quorum.nested_seq import InnerTruncator, corrected_outcome, outcome_interval
from quorum.simulator import RolloutSimulator, make_truth
from quorum.types import GridSpec


def _collect(truth, truncator, n: int, seed: int = 0):
    """Run ``n`` rollouts of one cell and return (corrected, naive, ground truth) means.

    Claim: unbiased-truncation
    """
    sim = RolloutSimulator(truth, seed=seed)
    rng = np.random.default_rng(seed + 1)
    corrected, naive, actual = [], [], []
    for _ in range(n):
        r = sim.rollout(0, 0, 0, 0, truncator=truncator, rng=rng)
        q = truncator.q_for_rollout(r) if hasattr(truncator, "q_for_rollout") else truncator.q_hat
        corrected.append(corrected_outcome(r, float(q), truncator.audit_rate))
        naive.append(float(r.success))
        actual.append(float(r.true_success))
    return np.array(corrected), np.array(naive), np.array(actual)


@pytest.mark.parametrize("q_hat", [0.0, 0.2])
def test_corrected_outcome_is_unbiased(truth, q_hat: float) -> None:
    """The corrected mean covers the true success rate; the naive one does not.

    Claim: unbiased-truncation
    """
    trunc = InnerTruncator(threshold=-0.6, min_steps=3, audit_rate=0.4, q_hat=q_hat,
                           q_max=0.25, rng=np.random.default_rng(7))
    corrected, naive, actual = _collect(truth, trunc, 8000, seed=11)
    assert trunc.n_fired > 200, "the rule never fired; the test would be vacuous"

    truth_mean = actual.mean()
    se = corrected.std(ddof=1) / np.sqrt(corrected.size)
    assert abs(corrected.mean() - truth_mean) < 4 * se, (
        f"corrected mean {corrected.mean():.4f} vs truth {truth_mean:.4f} (se {se:.4f})"
    )
    # And the uncorrected estimator really is biased low -- otherwise the correction
    # would be solving a problem that does not exist.
    assert naive.mean() < truth_mean - 2 * se


def test_unbiasedness_survives_a_deliberately_wrong_q(truth) -> None:
    """A badly miscalibrated ``q`` costs variance, never validity.

    The rule here claims a 25% success probability on rollouts it is abandoning, when the
    real conditional rate is far lower. The audit subsample repairs the mean regardless,
    which is the whole point of the doubly-robust form.

    Claim: unbiased-truncation
    """
    trunc = InnerTruncator(threshold=-0.6, min_steps=3, audit_rate=0.4, q_hat=0.25,
                           q_max=0.30, rng=np.random.default_rng(5))
    corrected, _naive, actual = _collect(truth, trunc, 8000, seed=13)
    assert trunc.n_fired > 200
    se = corrected.std(ddof=1) / np.sqrt(corrected.size)
    assert abs(corrected.mean() - actual.mean()) < 4 * se, (
        "the correction failed under a miscalibrated q, which would void the nesting"
    )


def test_truncation_actually_saves_steps(truth) -> None:
    """The rule must save real compute, or there is nothing to trade against.

    Claim: unbiased-truncation
    """
    trunc = InnerTruncator(threshold=-0.6, min_steps=3, audit_rate=0.3, q_max=0.25,
                           rng=np.random.default_rng(3))
    sim = RolloutSimulator(truth, seed=2)
    rng = np.random.default_rng(4)
    executed = full = 0
    for _ in range(3000):
        r = sim.rollout(0, 0, 0, 0, truncator=trunc, rng=rng)
        executed += r.n_steps
        full += r.n_steps_full
    saved = 1.0 - executed / full
    assert saved > 0.02, f"only {saved:.3%} of steps saved; the rule is not doing anything"


def test_audit_coin_is_independent_of_the_outcome(truth) -> None:
    """Audited and non-audited firings must have the same conditional success rate.

    If the coin were drawn after peeking at the outcome, the audit subsample would no
    longer represent the truncated population and the correction would be void.

    Claim: unbiased-truncation
    """
    trunc = InnerTruncator(threshold=-0.6, min_steps=3, audit_rate=0.5, q_max=0.25,
                           rng=np.random.default_rng(9))
    sim = RolloutSimulator(truth, seed=6)
    rng = np.random.default_rng(8)
    audited, not_audited = [], []
    for _ in range(6000):
        r = sim.rollout(0, 0, 0, 0, truncator=trunc, rng=rng)
        if r.completed:
            continue
        (audited if r.audited else not_audited).append(float(r.true_success))
    assert len(audited) > 100 and len(not_audited) > 100
    a, b = np.array(audited), np.array(not_audited)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    assert abs(a.mean() - b.mean()) < 4 * se + 1e-9, (
        "audited firings have a different true success rate from non-audited ones, "
        "which means the coin is not independent of the outcome"
    )


def test_outcome_interval_bounds_every_reachable_value() -> None:
    """The declared range must actually contain every value ``corrected_outcome`` emits.

    Claim: unbiased-truncation
    """
    from quorum.types import Rollout

    for rho in (0.2, 0.5, 0.9):
        for q_max in (0.0, 0.25, 0.5):
            lo, hi = outcome_interval(rho, True, q_max)
            for q in (0.0, q_max / 2, q_max):
                for success in (0, 1):
                    for audited in (False, True):
                        r = Rollout(0, 0, 0, 0, success=success, completed=False,
                                    audited=audited)
                        y = corrected_outcome(r, q, rho)
                        assert lo - 1e-9 <= y <= hi + 1e-9, (
                            f"Ytil={y} escapes [{lo}, {hi}] at rho={rho}, q={q}"
                        )
            # completed rollouts contribute the raw outcome
            for success in (0, 1):
                r = Rollout(0, 0, 0, 0, success=success, completed=True)
                y = corrected_outcome(r, 0.0, rho)
                assert lo - 1e-9 <= y <= hi + 1e-9


def test_confidence_gate_keeps_q_below_the_cap(truth) -> None:
    """The rule must decline to fire when its own ``q`` exceeds ``q_max``.

    The declared range depends on ``q_max``; a rule that fired while believing success
    was still plausible would invalidate the bound it was sized against.

    Claim: unbiased-truncation
    """
    trunc = InnerTruncator(threshold=5.0, min_steps=1, audit_rate=0.5, q_hat=0.9,
                           q_max=0.9, rng=np.random.default_rng(1))
    # q_hat sits at the cap, so a threshold that always fires still fires.
    assert trunc.should_stop(np.array([-9.0, -9.0]), 2) is True

    class _OverconfidentQ:
        """A calibrated model that reports a success probability above the cap."""

        def q(self, traj_prefix: np.ndarray, step: int) -> float:
            return 0.85

        def observe_rollout(self, rollout) -> bool:  # pragma: no cover - unused here
            return False

    strict = InnerTruncator(threshold=5.0, min_steps=1, audit_rate=0.5, q_hat=0.05,
                            q_max=0.10, calibrated=_OverconfidentQ(),
                            rng=np.random.default_rng(1))
    assert strict.should_stop(np.array([-9.0, -9.0]), 2) is False
    assert strict.n_declined >= 1

    # The constructor refuses a constant guess above the cap outright, since that would
    # silently disable the rule on every rollout.
    with pytest.raises(ValueError, match="exceeds q_max"):
        InnerTruncator(q_hat=0.2, q_max=0.1)
