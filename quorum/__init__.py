"""Quorum -- rank-targeted nested sequential design for LLM evaluation.

The decision an evaluation actually has to make is usually a *ranking*, not a set of
absolute scores. Quorum makes the ranking the estimand and then buys the cheapest
evidence that settles it:

* an **outer** anytime-valid confidence sequence on scaffold-marginalised pair contrasts,
* an **inner** per-rollout truncation rule whose audit-based correction keeps the outer
  estimator unbiased no matter how miscalibrated the rule is,
* a **decision-targeted allocator** that scores each candidate cell by the reduction in
  the uncertainty of the contrast that decides the ranking, per dollar,
* **known-mean control variates** that cut the estimator's variance without touching its
  mean.

Inference is design-based and model-assisted: validity comes from the sampling design
(Horvitz-Thompson weighting plus a martingale argument), while the hierarchical cell
model only steers where the next rollout goes. A wrong model costs efficiency, never
coverage. That separation is the point of the package.

Claim: infrastructure
"""

from __future__ import annotations

from quorum.allocator import STRATEGIES, DecisionTargetedAllocator, mix_to_floor
from quorum.cell_model import HierarchicalCellModel
from quorum.cuped import (
    CupedAdjuster,
    build_covariate_vector,
    covariate_names,
    known_covariate_means,
    pilot_difficulty_from_predictions,
    variance_reduction_report,
)
from quorum.loaders import (
    FrozenGridSimulator,
    load_hf_dataset,
    load_rollouts_csv,
    load_rollouts_jsonl,
    rollouts_from_records,
    to_frozen_grid,
)
from quorum.nested_seq import (
    CalibratedQ,
    EmpiricalBernsteinCS,
    InnerTruncator,
    NestedSequentialTest,
    PairChannel,
    UniformAllocator,
    corrected_outcome,
    outcome_range,
    rollouts_to_resolve,
)
from quorum.simulator import (
    GridTruth,
    RolloutSimulator,
    apply_scaffold_shift,
    default_ability,
    make_truth,
    true_delta,
)
from quorum.types import (
    GridSpec,
    RankDecision,
    Rollout,
    block_index,
    block_index_array,
    unblock_index,
    uniform_weights,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # types
    "GridSpec",
    "Rollout",
    "RankDecision",
    "block_index",
    "unblock_index",
    "block_index_array",
    "uniform_weights",
    # simulator + data
    "GridTruth",
    "make_truth",
    "apply_scaffold_shift",
    "default_ability",
    "true_delta",
    "RolloutSimulator",
    "FrozenGridSimulator",
    "load_rollouts_jsonl",
    "load_rollouts_csv",
    "load_hf_dataset",
    "rollouts_from_records",
    "to_frozen_grid",
    # (a) cell model
    "HierarchicalCellModel",
    # (b) allocation
    "DecisionTargetedAllocator",
    "mix_to_floor",
    "STRATEGIES",
    # (c) nested sequential test
    "EmpiricalBernsteinCS",
    "PairChannel",
    "InnerTruncator",
    "CalibratedQ",
    "NestedSequentialTest",
    "UniformAllocator",
    "corrected_outcome",
    "outcome_range",
    "rollouts_to_resolve",
    # (d) variance reduction
    "CupedAdjuster",
    "build_covariate_vector",
    "known_covariate_means",
    "covariate_names",
    "pilot_difficulty_from_predictions",
    "variance_reduction_report",
]
