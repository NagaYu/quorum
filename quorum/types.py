"""Core data types shared by every Quorum module.

These are deliberately plain dataclasses: the statistical machinery is design-based,
so the objects that flow through it are just records of *what was actually bought*
(which cell, at what cost, with what outcome) plus the bookkeeping needed to undo the
effect of inner truncation.

Claim: infrastructure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "GridSpec",
    "Rollout",
    "RankDecision",
    "block_index",
    "unblock_index",
    "block_index_array",
    "uniform_weights",
]


@dataclass(frozen=True)
class GridSpec:
    """Shape of the (task, seed, scaffold, model) evaluation grid.

    A *block* is a (task, seed, scaffold) triple -- the unit that two models can share,
    which is what makes paired comparison (and hence scaffold-marginalised contrasts)
    possible.

    Claim: infrastructure
    """

    n_tasks: int
    n_seeds: int
    n_scaffolds: int
    n_models: int

    @property
    def n_blocks(self) -> int:
        """Number of (task, seed, scaffold) blocks.

        Claim: infrastructure
        """
        return self.n_tasks * self.n_seeds * self.n_scaffolds

    @property
    def n_cells(self) -> int:
        """Number of (task, seed, scaffold, model) cells.

        Claim: infrastructure
        """
        return self.n_blocks * self.n_models

    @property
    def grid_shape(self) -> tuple[int, int, int, int]:
        """(T, K, S, M) shape tuple for truth arrays.

        Claim: infrastructure
        """
        return (self.n_tasks, self.n_seeds, self.n_scaffolds, self.n_models)

    @property
    def block_shape(self) -> tuple[int, int, int]:
        """(T, K, S) shape tuple for block-level arrays.

        Claim: infrastructure
        """
        return (self.n_tasks, self.n_seeds, self.n_scaffolds)


@dataclass
class Rollout:
    """One executed rollout of one model on one block.

    ``success`` is the *observed* outcome and may be the inner truncation rule's guess
    (0) rather than the true outcome; ``completed`` records which. Estimators must never
    consume ``success`` directly for truncated rollouts -- they go through
    :func:`quorum.nested_seq.corrected_outcome`, which restores unbiasedness using the
    audit subsample.

    Claim: unbiased-truncation
    """

    task: int
    seed: int
    scaffold: int
    model: int
    success: int
    completed: bool = True
    n_steps: int = 0
    n_steps_full: int = 0
    cost_usd: float = 0.0
    cost_sec: float = 0.0
    trajectory: np.ndarray = field(default_factory=lambda: np.zeros(0))
    audited: bool = False
    truncated_at: int = -1
    true_success: int = -1

    @property
    def block(self) -> tuple[int, int, int]:
        """The (task, seed, scaffold) triple this rollout was drawn from.

        Claim: infrastructure
        """
        return (self.task, self.seed, self.scaffold)

    @property
    def steps_saved(self) -> int:
        """Steps not executed because the inner rule fired (0 when it did not).

        Claim: unbiased-truncation
        """
        return max(0, self.n_steps_full - self.n_steps)


@dataclass
class RankDecision:
    """Result of a ranking procedure: the order, and how much it cost to certify.

    ``resolved`` is False when the budget ran out before every adjacent pair separated.
    Callers must render an unresolved pair as "not distinguishable at this budget" and
    never as a superiority claim.

    Claim: coverage
    """

    order: list[int]
    resolved: bool
    n_rollouts: int = 0
    cost_usd: float = 0.0
    cost_sec: float = 0.0
    intervals: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)
    undecided_pairs: list[tuple[int, int]] = field(default_factory=list)
    n_steps_executed: int = 0
    n_steps_full: int = 0
    extra_rollouts_needed: dict[tuple[int, int], int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def compute_saved(self) -> float:
        """Fraction of trajectory steps avoided by inner truncation, in [0, 1).

        Claim: unbiased-truncation
        """
        if self.n_steps_full <= 0:
            return 0.0
        return 1.0 - (self.n_steps_executed / self.n_steps_full)

    def is_correct(self, true_order: "list[int] | np.ndarray") -> bool:
        """Whether the returned order matches ground truth exactly.

        Claim: low-cost
        """
        return list(self.order) == list(true_order)


def block_index(t: int, k: int, s: int, n_seeds: int, n_scaffolds: int) -> int:
    """Flatten (task, seed, scaffold) to a block id, row-major over (t, k, s).

    Every module must use this; the Horvitz-Thompson weights are indexed by block, so a
    disagreement about flattening silently corrupts the estimand.

    Claim: infrastructure
    """
    return (t * n_seeds + k) * n_scaffolds + s


def unblock_index(b: int, n_seeds: int, n_scaffolds: int) -> tuple[int, int, int]:
    """Inverse of :func:`block_index`.

    Claim: infrastructure
    """
    s = b % n_scaffolds
    rest = b // n_scaffolds
    k = rest % n_seeds
    t = rest // n_seeds
    return (t, k, s)


def block_index_array(spec: GridSpec) -> np.ndarray:
    """(T, K, S) array whose entries are the flat block ids -- for vectorised code.

    Claim: infrastructure
    """
    return np.arange(spec.n_blocks).reshape(spec.block_shape)


def uniform_weights(spec: GridSpec) -> np.ndarray:
    """Default target weights ``w`` over blocks: uniform, summing to 1.

    ``w`` *defines the estimand* and must never adapt to the data. The allocator adapts
    the sampling distribution ``pi``; the Horvitz-Thompson ratio ``w/pi`` restores the
    fixed target.

    Claim: coverage
    """
    return np.full(spec.n_blocks, 1.0 / spec.n_blocks)
