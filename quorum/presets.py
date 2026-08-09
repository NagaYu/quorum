"""Named benchmark presets, and the standard way to assemble a Quorum run.

Everything the CLI, the benchmarks and the adapters need in order to build a configured
:class:`quorum.nested_seq.NestedSequentialTest` lives here, so the four surfaces cannot
drift apart in their defaults.

A preset is a *synthetic world*, not a claim about any real system. Model labels supplied
on the command line are attached to indices purely so the report is readable; nothing in
this package infers, measures or asserts anything about a real named model.

Claim: infrastructure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from quorum.allocator import DecisionTargetedAllocator
from quorum.cell_model import HierarchicalCellModel
from quorum.cuped import CupedAdjuster
from quorum.nested_seq import InnerTruncator, NestedSequentialTest
from quorum.simulator import GridTruth, RolloutSimulator, apply_scaffold_shift, make_truth
from quorum.types import GridSpec, uniform_weights

__all__ = [
    "PresetSpec",
    "PRESETS",
    "list_presets",
    "build_truth",
    "build_test",
    "QuorumConfig",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PresetSpec:
    """One named synthetic world: grid shape plus the knobs that set its difficulty.

    ``ability_gap`` is the hardness dial -- it is the logit-scale separation the ranking
    procedure must certify, so halving it roughly quadruples the rollouts any honest
    method needs. ``scaffold_shift`` and ``rank_distortion`` set up the robustness
    scenarios of metric (3).

    Claim: infrastructure
    """

    name: str
    n_tasks: int
    n_seeds: int
    n_scaffolds: int
    n_models: int
    ability_gap: float = 0.45
    scaffold_shift: float = 0.0
    rank_distortion: float = 0.0
    sd_task: float = 1.0
    description: str = ""

    @property
    def spec(self) -> GridSpec:
        """The grid shape of this preset.

        Claim: infrastructure
        """
        return GridSpec(self.n_tasks, self.n_seeds, self.n_scaffolds, self.n_models)


PRESETS: dict[str, PresetSpec] = {
    "synthetic-easy": PresetSpec(
        "synthetic-easy", 6, 3, 3, 4, ability_gap=0.55,
        description="Well-separated models; the ranking is decidable quickly.",
    ),
    "synthetic-hard": PresetSpec(
        "synthetic-hard", 8, 3, 3, 4, ability_gap=0.22,
        description="Closely-matched models; most pairs stay indistinguishable for a long time.",
    ),
    "synthetic-shifted": PresetSpec(
        "synthetic-shifted", 6, 3, 3, 4, ability_gap=0.45, scaffold_shift=1.6,
        description="A large scaffold shift moves absolute scores while preserving the ranking.",
    ),
    "synthetic-distorted": PresetSpec(
        "synthetic-distorted", 6, 3, 3, 4, ability_gap=0.45, scaffold_shift=1.6,
        rank_distortion=0.5,
        description="Scaffold x model interaction genuinely reorders models -- the regime "
                    "where no method should claim a single global ranking.",
    ),
    "synthetic-wide": PresetSpec(
        "synthetic-wide", 10, 4, 3, 6, ability_gap=0.35,
        description="Six models, more tasks: the family-wise correction bites harder.",
    ),
}


def list_presets() -> list[str]:
    """Names of the available synthetic benchmark presets.

    Claim: infrastructure
    """
    return sorted(PRESETS)


def build_truth(preset: str | PresetSpec, *, seed: int = 0) -> GridTruth:
    """Instantiate a preset's ground-truth world.

    Claim: infrastructure
    """
    ps = PRESETS[preset] if isinstance(preset, str) else preset
    truth = make_truth(
        ps.spec,
        seed=seed,
        ability_gap=ps.ability_gap,
        sd_task=ps.sd_task,
        rank_distortion=0.0,
    )
    if ps.scaffold_shift or ps.rank_distortion:
        truth = apply_scaffold_shift(
            truth,
            shift=ps.scaffold_shift,
            rank_distortion=ps.rank_distortion or None,
            seed=seed + 1,
        )
    return truth


@dataclass
class QuorumConfig:
    """Which of the four contributions are switched on for a run.

    Turning them all off reduces the driver to a plain anytime-valid sequential test on a
    uniform design -- condition (C) of the benchmark. That is deliberate: it makes the
    ablation exact rather than approximate, since the same code path serves both.

    ``truncation`` defaults to **False**, and that default is a measured result rather
    than an oversight. Inner truncation does save trajectory steps (2.7%-17.8% here,
    depending on the audit rate), but the audit reweighting that keeps the estimator
    unbiased widens the corrected-outcome range from ``[0, 1]`` to ``[-q_max(1-rho)/rho,
    1/rho]``, and an empirical-Bernstein confidence sequence pays for range directly. On
    this simulator every audit rate we swept cost *more* total dollars and seconds than
    running rollouts to completion. Turn it on when a rollout's steps are the binding
    cost and the rule fires early and confidently; leave it off when the number of
    rollouts is what you are paying for. ``benchmarks/figures.py`` plots the whole
    trade-off curve so the choice can be made on evidence.

    Claim: unbiased-truncation
    """

    allocator: bool = True
    truncation: bool = False
    cuped: bool = True
    paired: bool = True
    strategy: str = "info_cost"
    pi_min_frac: float = 0.20
    audit_rate: float = 0.3
    truncation_threshold: float = -0.8
    q_max: float = 0.25
    cuped_warmup: int = 30
    instance_corr: float = 0.6
    extra: dict[str, Any] = field(default_factory=dict)


def build_test(
    truth: GridTruth,
    config: QuorumConfig | None = None,
    *,
    alpha: float = 0.05,
    seed: int = 0,
    budget_usd: float | None = None,
    budget_sec: float | None = None,
    max_rollouts: int = 200_000,
    candidate_pairs: Any = None,
    simulator: Any = None,
    on_rollout: Callable[[Any], None] | None = None,
) -> NestedSequentialTest:
    """Assemble a fully configured nested sequential test for ``truth``.

    This is the single place the four contributions get wired together, so the CLI, the
    benchmark conditions, the tests and the lighteval adapter cannot disagree about what
    "Quorum" means.

    Claim: low-cost
    """
    cfg = config or QuorumConfig()
    spec = truth.spec
    w = uniform_weights(spec)
    sim = simulator if simulator is not None else RolloutSimulator(
        truth, seed=seed, instance_corr=cfg.instance_corr
    )

    allocator = None
    if cfg.allocator:
        cell_model = HierarchicalCellModel(spec)
        allocator = DecisionTargetedAllocator(
            spec,
            cell_model,
            truth.cost_usd,
            truth.cost_sec,
            w=w,
            strategy=cfg.strategy,
            pi_min_frac=cfg.pi_min_frac,
            rng=np.random.default_rng(seed + 11),
        )

    truncator = None
    if cfg.truncation:
        truncator = InnerTruncator(
            threshold=cfg.truncation_threshold,
            min_steps=3,
            audit_rate=cfg.audit_rate,
            q_max=cfg.q_max,
            rng=np.random.default_rng(seed + 23),
        )

    cuped = None
    proxy_cov = False
    if cfg.cuped:
        proxy = truth.proxy_blocks
        cuped = CupedAdjuster(
            np.asarray([float(proxy @ w)]),
            warmup=cfg.cuped_warmup,
            cov_table=proxy.reshape(-1, 1),
            names=["proxy"],
        )

    test = NestedSequentialTest(
        spec,
        sim,
        alpha=alpha,
        w=w,
        allocator=allocator,
        truncator=truncator,
        cuped=cuped,
        pi_min_frac=cfg.pi_min_frac,
        max_rollouts=max_rollouts,
        budget_usd=budget_usd,
        budget_sec=budget_sec,
        paired=cfg.paired,
        proxy_covariate=proxy_cov,
        candidate_pairs=candidate_pairs,
        rng=np.random.default_rng(seed + 97),
    )
    if on_rollout is not None:
        test.meta_on_rollout = on_rollout  # type: ignore[attr-defined]
    return test
