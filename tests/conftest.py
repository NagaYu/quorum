"""Shared fixtures. Grids are deliberately small so the whole suite stays under a few minutes.

Claim: infrastructure
"""

from __future__ import annotations

import numpy as np
import pytest

from quorum.simulator import RolloutSimulator, make_truth
from quorum.types import GridSpec, uniform_weights


@pytest.fixture(scope="session")
def small_spec() -> GridSpec:
    """A 4x2x2x3 grid: big enough to exercise partial pooling, small enough to be fast.

    Claim: infrastructure
    """
    return GridSpec(4, 2, 2, 3)


@pytest.fixture(scope="session")
def truth(small_spec: GridSpec):
    """Ground truth with a comfortable ability gap and no rank distortion.

    Claim: infrastructure
    """
    return make_truth(small_spec, seed=0, ability_gap=0.6)


@pytest.fixture(scope="session")
def null_truth():
    """Two models with *identical* ability, so the true contrast is exactly zero.

    Claim: coverage
    """
    return make_truth(GridSpec(4, 2, 2, 2), seed=0, ability=np.zeros(2))


@pytest.fixture
def sim(truth) -> RolloutSimulator:
    """A fresh simulator bound to the shared truth.

    Claim: infrastructure
    """
    return RolloutSimulator(truth, seed=1)


@pytest.fixture
def weights(small_spec: GridSpec) -> np.ndarray:
    """Uniform target weights over blocks.

    Claim: infrastructure
    """
    return uniform_weights(small_spec)
