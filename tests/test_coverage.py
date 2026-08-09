"""Claim (2): the procedure's error rate does not exceed the nominal level.

The interesting case is the adaptive one. A confidence sequence on i.i.d. data is
standard; keeping it valid while an allocator chooses each next observation from a
data-dependent distribution is what the Horvitz-Thompson construction is for, and it is
the thing that would break silently if it were wrong.

Claim: coverage
"""

from __future__ import annotations

import numpy as np
import pytest

from quorum.nested_seq import EmpiricalBernsteinCS
from quorum.presets import QuorumConfig, build_test

REPS = 300
ALPHA = 0.10


def _mc_slack(alpha: float, reps: int, sigmas: float = 3.0) -> float:
    """Monte-Carlo slack: `sigmas` binomial standard errors on the error-rate estimate.

    With reps=300 and alpha=0.10 the binomial s.e. is sqrt(.1*.9/300) = 0.0173, so a run
    that truly hits the nominal level lands under 0.152 about 99.9% of the time. Failing
    this test therefore means a real violation, not an unlucky seed.

    Claim: coverage
    """
    return alpha + sigmas * float(np.sqrt(alpha * (1 - alpha) / reps))


def _null_error_rate(null_truth, cfg: QuorumConfig, *, reps: int = REPS,
                     alpha: float = ALPHA, cap: int = 700) -> float:
    """Fraction of null runs in which the confidence sequence EVER excludes zero.

    "Ever" matters: an anytime-valid interval may be inspected after every observation,
    so the error event is the union over the whole run, not the final look.

    Claim: coverage
    """
    errors = 0
    for rep in range(reps):
        test = build_test(null_truth, cfg, alpha=alpha, seed=rep, max_rollouts=cap)
        test.alpha_split = "none"
        for channel in test.channels.values():
            channel.alpha = alpha
        while test.n_rollouts < cap:
            test.step()
            if any(c.excludes_zero for c in test.channels.values()):
                errors += 1
                break
    return errors / reps


def test_cs_covers_a_known_mean_under_adaptive_bounds() -> None:
    """The bare confidence sequence covers the truth when ranges vary per observation.

    Claim: coverage
    """
    rng = np.random.default_rng(0)
    misses = 0
    trials = 200
    for _ in range(trials):
        cs = EmpiricalBernsteinCS(0.10, -12.0, 12.0)
        covered = True
        for i in range(250):
            # Predictable, wildly heterogeneous scale -- exactly the shape a
            # Horvitz-Thompson weight produces under an adaptive allocator.
            scale = 8.0 if (i % 11 == 0) else 0.8
            x = float(np.clip(rng.normal(0.0, 0.5 * scale), -scale, scale))
            lo, hi = cs.update(x, -scale, scale)
            if not (lo <= 0.0 <= hi):
                covered = False
                break
        misses += int(not covered)
    assert misses / trials <= _mc_slack(0.10, trials), (
        f"CS missed the true mean in {misses}/{trials} runs"
    )


def test_null_coverage_non_adaptive(null_truth) -> None:
    """Uniform design: the baseline case must hold before the adaptive one means anything.

    Claim: coverage
    """
    rate = _null_error_rate(null_truth, QuorumConfig(allocator=False, cuped=False))
    assert rate <= _mc_slack(ALPHA, REPS), f"error rate {rate:.4f} exceeds nominal {ALPHA}"


def test_null_coverage_adaptive(null_truth) -> None:
    """Adaptive allocation: the load-bearing case.

    If the Horvitz-Thompson weighting used a different pi from the one the block was
    drawn with, or if the exploration floor were not enforced, this is where it shows.

    Claim: coverage
    """
    rate = _null_error_rate(null_truth, QuorumConfig(allocator=True, cuped=False))
    assert rate <= _mc_slack(ALPHA, REPS), f"error rate {rate:.4f} exceeds nominal {ALPHA}"


def test_null_coverage_adaptive_with_cuped(null_truth) -> None:
    """Adaptive allocation plus the control variate.

    A control variate whose mean is even slightly wrong injects a constant into every
    observation, which a confidence sequence cannot see. This test caught exactly that
    bug in the unpaired covariate path during development.

    Claim: coverage
    """
    rate = _null_error_rate(null_truth, QuorumConfig(allocator=True, cuped=True))
    assert rate <= _mc_slack(ALPHA, REPS), f"error rate {rate:.4f} exceeds nominal {ALPHA}"


def test_null_coverage_unpaired_with_cuped(null_truth) -> None:
    """The unpaired regime, where the differenced covariate has known mean zero.

    Claim: coverage
    """
    rate = _null_error_rate(
        null_truth, QuorumConfig(allocator=True, cuped=True, paired=False)
    )
    assert rate <= _mc_slack(ALPHA, REPS), f"error rate {rate:.4f} exceeds nominal {ALPHA}"


@pytest.mark.slow
def test_null_coverage_everything_on(null_truth) -> None:
    """Every contribution at once, including inner truncation.

    Claim: coverage
    """
    cfg = QuorumConfig(allocator=True, cuped=True, truncation=True, audit_rate=0.3)
    rate = _null_error_rate(null_truth, cfg)
    assert rate <= _mc_slack(ALPHA, REPS), f"error rate {rate:.4f} exceeds nominal {ALPHA}"


def test_observations_never_leave_their_declared_range(truth) -> None:
    """No observation may fall outside the bounds its confidence sequence was sized for.

    An out-of-range observation does not merely widen the interval -- it voids the
    supermartingale the guarantee rests on, so the count must be exactly zero.

    Claim: coverage
    """
    for cfg in (
        QuorumConfig(allocator=True, cuped=True),
        QuorumConfig(allocator=True, cuped=True, truncation=True),
        QuorumConfig(allocator=True, cuped=True, paired=False),
    ):
        test = build_test(truth, cfg, alpha=0.05, seed=3, max_rollouts=3000)
        test.run()
        breaches = sum(ch.cs.n_clipped for ch in test.channels.values())
        assert breaches == 0, f"{breaches} observations left their declared range ({cfg})"
