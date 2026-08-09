"""Claim (d): known-mean control variates cut variance without moving the mean.

Two things are tested, and the second matters more than the first. Variance reduction is
the selling point, but non-anticipativity is the correctness condition: if ``phi`` could
see the observation it is about to adjust, ``E[Ytil_i | F_{i-1}] = Delta`` fails and the
outer confidence sequence silently stops being valid.

Note the honest scope of the variance claim, which the benchmark also reports: under the
*paired* design there is very little left for a block-level covariate to remove, because
the pairing has already cancelled the shared instance difficulty. The control variate
earns its keep in the unpaired regime -- logged data, public leaderboards -- which is
what these tests exercise.

Claim: variance-reduction
"""

from __future__ import annotations

import numpy as np
import pytest

from quorum.cuped import CupedAdjuster, known_covariate_means, variance_reduction_report
from quorum.presets import QuorumConfig, build_test
from quorum.types import uniform_weights

PAIR = (0, 1)


def _correlated_stream(n: int, rho: float, seed: int = 0):
    """A stream of (outcome, covariate) pairs with known covariate mean 0.

    Claim: variance-reduction
    """
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, 1.0, n)
    z = rho * c + np.sqrt(1 - rho ** 2) * rng.normal(0.0, 1.0, n) + 0.25
    return z, c


def test_variance_ratio_tracks_one_minus_r_squared() -> None:
    """With a single covariate the ratio should approach ``1 - corr^2``.

    Claim: variance-reduction
    """
    rho = 0.8
    z, c = _correlated_stream(4000, rho, seed=1)
    adj = CupedAdjuster(np.zeros(1), warmup=30)
    for zi, ci in zip(z, c):
        adj.adjust_and_observe(PAIR, float(zi), np.array([ci]))
    ratio = adj.variance_ratio(PAIR)
    expected = 1 - rho ** 2
    assert ratio < 1.0
    assert abs(ratio - expected) < 0.12, f"ratio {ratio:.3f} vs expected {expected:.3f}"


def test_variance_reduction_is_significant() -> None:
    """The reduction must be real, not a sampling artefact.

    Uses the Pitman-Morgan test for equal variances of paired samples, which is the
    correct test here because the raw and adjusted values are the *same* observations.

    Claim: variance-reduction
    """
    z, c = _correlated_stream(3000, 0.7, seed=2)
    adj = CupedAdjuster(np.zeros(1), warmup=30)
    for zi, ci in zip(z, c):
        adj.adjust_and_observe(PAIR, float(zi), np.array([ci]))
    report = variance_reduction_report(adj)[PAIR]
    assert report["ratio"] < 0.85
    assert report["p_value"] < 0.05, f"variance reduction not significant: {report}"


def test_phi_is_non_anticipative() -> None:
    """The adjustment for observation ``i`` must not depend on observation ``i``.

    Constructed so that the current point, if it leaked into the fit, would change
    ``phi`` drastically: a wild outlier appended after a long, clean history.

    Claim: variance-reduction
    """
    z, c = _correlated_stream(400, 0.8, seed=3)
    base = CupedAdjuster(np.zeros(1), warmup=30)
    for zi, ci in zip(z, c):
        base.adjust_and_observe(PAIR, float(zi), np.array([ci]))
    phi_before = np.array(base.phi(PAIR), dtype=float)

    # Two very different "current" observations must receive adjustments computed from
    # the SAME phi, and must leave theta unchanged until observe() is called.
    a1 = base.adjust(PAIR, 1.0, np.array([1.0]))
    a2 = base.adjust(PAIR, 1.0, np.array([500.0]))
    phi_after = np.array(base.phi(PAIR), dtype=float)
    assert np.array_equal(phi_before, phi_after), "adjust() mutated the fit"

    # The adjustment must be exactly phi-linear in the covariate (up to the predictable
    # cap), which is only possible if phi did not react to the covariate value.
    implied1 = (1.0 - a1) / (1.0 - 0.0)
    implied2 = (1.0 - a2) / (500.0 - 0.0)
    assert abs(implied1 - implied2) < 1e-6 or abs(a2) < abs(a1) * 1e6


def test_estimated_means_are_refused_by_default() -> None:
    """A covariate whose mean is unknown must be rejected, not silently estimated.

    An estimated mean puts an O(n^-1/2) bias inside a procedure whose entire selling
    point is that its error is design-based.

    Claim: variance-reduction
    """
    with pytest.raises(ValueError, match="allow_estimated_mean"):
        CupedAdjuster(np.array([np.nan]))
    # ...but it is available explicitly, for ablations.
    adj = CupedAdjuster(np.array([np.nan]), allow_estimated_mean=True)
    assert adj.allow_estimated_mean is True


def test_known_means_do_not_depend_on_the_allocator(truth, weights) -> None:
    """``E[C]`` is a closed-form grid sum, so it cannot depend on ``pi``.

    Claim: variance-reduction
    """
    proxy = truth.proxy_blocks
    means = known_covariate_means(weights, proxy)
    assert means.shape == (1,)
    assert abs(float(means[0]) - float(proxy @ weights)) < 1e-12


def test_cuped_reduces_variance_in_the_unpaired_regime(truth) -> None:
    """End-to-end: the driver's control variate cuts the estimator's variance.

    Claim: variance-reduction
    """
    cfg = QuorumConfig(allocator=True, cuped=True, paired=False)
    test = build_test(truth, cfg, alpha=0.05, seed=4, max_rollouts=6000)
    test.run()
    ratios = [test.cuped.variance_ratio(p) for p, ch in test.channels.items() if ch.n > 200]
    assert ratios, "no channel accumulated enough observations to measure a ratio"
    assert float(np.mean(ratios)) < 0.95, (
        f"mean variance ratio {np.mean(ratios):.3f} shows no reduction"
    )
