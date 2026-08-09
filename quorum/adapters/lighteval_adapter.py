"""Drop Quorum's sequential stopping into a lighteval-style evaluation loop.

The integration surface is deliberately a *callable*, not a lighteval type::

    from quorum.adapters import QuorumSequentialSampler

    sampler = QuorumSequentialSampler(tasks=my_tasks, models=["a", "b", "c"])
    decision = sampler.run(lambda task, seed, scaffold, model: my_eval_fn(...))

That is three lines, it works whether or not ``lighteval`` is installed, and it means a
harness does not have to adopt Quorum's data model to get rank-targeted early stopping.
When ``lighteval`` *is* present, :func:`from_lighteval_tasks` and
:func:`to_lighteval_results` handle the conversions.

What the caller gets: instead of running every model on every item, the loop stops as
soon as the *ranking* is certified at level ``alpha`` -- and if it never is, it says so
rather than reporting an order the data does not support.

Claim: low-cost
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from quorum.nested_seq import NestedSequentialTest
from quorum.simulator import make_truth
from quorum.types import GridSpec, RankDecision, Rollout, unblock_index, uniform_weights

__all__ = [
    "QuorumSequentialSampler",
    "rollout_from_lighteval_sample",
    "from_lighteval_tasks",
    "to_lighteval_results",
    "lighteval_available",
]

logger = logging.getLogger(__name__)

RunFn = Callable[[Any, int, Any, Any], Any]


def lighteval_available() -> bool:
    """Whether ``lighteval`` can be imported in this environment.

    Nothing in this module requires it; the check exists so callers can branch and so the
    test suite can assert the no-lighteval path is fully functional.

    Claim: infrastructure
    """
    try:  # pragma: no cover - depends on the environment
        import lighteval  # noqa: F401
    except Exception:
        return False
    return True


def rollout_from_lighteval_sample(
    sample: Mapping[str, Any] | Any,
    *,
    task: int,
    seed: int,
    scaffold: int,
    model: int,
) -> Rollout:
    """Coerce a lighteval-style per-sample metric record into a :class:`Rollout`.

    Accepts a mapping or any object with attributes. Recognised keys, in order of
    preference for the outcome: ``success``, ``acc``, ``accuracy``, ``exact_match``,
    ``score``, ``em``. A float score is thresholded at 0.5, which is what a binary
    ranking estimand requires; a task whose metric is genuinely continuous should be
    binarised deliberately by the caller rather than silently here, and a warning says so.

    Claim: infrastructure
    """
    def get(key: str, default: Any = None) -> Any:
        if isinstance(sample, Mapping):
            return sample.get(key, default)
        return getattr(sample, key, default)

    raw = None
    for key in ("success", "acc", "accuracy", "exact_match", "em", "score"):
        value = get(key)
        if value is not None:
            raw = value
            break
    if raw is None:
        raise ValueError(
            "could not find an outcome field on the sample; expected one of "
            "success/acc/accuracy/exact_match/em/score"
        )
    if isinstance(raw, bool):
        success = int(raw)
    else:
        value = float(raw)
        if value not in (0.0, 1.0):
            logger.warning(
                "binarising a continuous metric at 0.5 (%.4g); the ranking estimand is "
                "a success *rate*, so binarise deliberately if that is not what you want",
                value,
            )
        success = int(value >= 0.5)

    n_steps = int(get("n_steps", 0) or 0)
    return Rollout(
        task=int(task), seed=int(seed), scaffold=int(scaffold), model=int(model),
        success=success,
        completed=bool(get("completed", True)),
        n_steps=n_steps,
        n_steps_full=int(get("n_steps_full", n_steps) or n_steps),
        cost_usd=float(get("cost_usd", 0.0) or 0.0),
        cost_sec=float(get("cost_sec", get("duration", 0.0)) or 0.0),
        trajectory=np.asarray(get("trajectory", []) or [], dtype=float),
        audited=bool(get("audited", False)),
        truncated_at=int(get("truncated_at", -1) or -1),
    )


@dataclass
class _CallableSimulator:
    """Adapts a user ``run_fn`` to the simulator protocol the driver consumes.

    The driver only ever needs ``rollout(t, k, s, m)``; everything else about a
    simulator (costs, trajectories, shocks) is optional and degrades gracefully. That is
    why an arbitrary evaluation harness can be dropped in here unchanged.

    Claim: infrastructure
    """

    run_fn: RunFn
    tasks: Sequence[Any]
    scaffolds: Sequence[Any]
    models: Sequence[Any]
    n_rollouts: int = 0
    total_usd: float = 0.0
    total_sec: float = 0.0

    def rollout(
        self,
        t: int,
        k: int,
        s: int,
        m: int,
        *,
        truncator: Any | None = None,
        rng: Any | None = None,
        shock: float | None = None,
    ) -> Rollout:
        """Invoke the user's evaluation function and normalise the result.

        Claim: infrastructure
        """
        raw = self.run_fn(self.tasks[t], int(k), self.scaffolds[s], self.models[m])
        rollout = (
            raw if isinstance(raw, Rollout)
            else rollout_from_lighteval_sample(raw, task=t, seed=k, scaffold=s, model=m)
        )
        self.n_rollouts += 1
        self.total_usd += float(rollout.cost_usd)
        self.total_sec += float(rollout.cost_sec)
        return rollout


class QuorumSequentialSampler:
    """Rank-targeted sequential sampling over a task list and a set of models.

    The sampler owns the statistics; the caller owns the execution. It decides *which*
    (task, seed, scaffold, model) to evaluate next and when there is enough evidence to
    stop, and it hands back a :class:`~quorum.types.RankDecision` whose ``resolved`` flag
    must be respected: an unresolved decision means the ranking is not supported, not
    that the top of ``order`` won.

    Claim: low-cost
    """

    def __init__(
        self,
        tasks: Sequence[Any],
        models: Sequence[Any],
        *,
        scaffolds: Sequence[Any] | None = None,
        n_seeds: int = 1,
        alpha: float = 0.05,
        max_rollouts: int = 20_000,
        budget_usd: float | None = None,
        budget_sec: float | None = None,
        seed: int = 0,
        use_allocator: bool = True,
    ) -> None:
        if len(tasks) == 0:
            raise ValueError("tasks must be non-empty")
        if len(models) < 2:
            raise ValueError("ranking needs at least two models")
        self.tasks = list(tasks)
        self.models = list(models)
        self.scaffolds = list(scaffolds) if scaffolds else ["default"]
        self.n_seeds = int(n_seeds)
        self.alpha = float(alpha)
        self.max_rollouts = int(max_rollouts)
        self.budget_usd = budget_usd
        self.budget_sec = budget_sec
        self.seed = int(seed)
        self.use_allocator = bool(use_allocator)
        self.spec = GridSpec(len(self.tasks), self.n_seeds, len(self.scaffolds), len(self.models))
        self.decision: RankDecision | None = None

    def run(self, run_fn: RunFn) -> RankDecision:
        """Drive the nested sequential test with the caller's evaluation function.

        Claim: low-cost
        """
        simulator = _CallableSimulator(run_fn, self.tasks, self.scaffolds, self.models)
        allocator = None
        if self.use_allocator:
            try:
                from quorum.allocator import DecisionTargetedAllocator
                from quorum.cell_model import HierarchicalCellModel

                # Costs are unknown before anything has run; a flat prior on cost makes
                # the criterion pure information gain, which is the right default.
                ones = np.ones(self.spec.grid_shape)
                allocator = DecisionTargetedAllocator(
                    self.spec, HierarchicalCellModel(self.spec), ones, ones,
                    w=uniform_weights(self.spec), rng=np.random.default_rng(self.seed + 3),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("falling back to uniform allocation: %s", exc)

        test = NestedSequentialTest(
            self.spec, simulator, alpha=self.alpha, w=uniform_weights(self.spec),
            allocator=allocator, truncator=None, cuped=None,
            max_rollouts=self.max_rollouts, budget_usd=self.budget_usd,
            budget_sec=self.budget_sec, rng=np.random.default_rng(self.seed + 17),
        )
        self.decision = test.run()
        return self.decision

    def summary(self) -> str:
        """Human-readable summary that never overstates an unresolved ranking.

        Claim: coverage
        """
        if self.decision is None:
            return "no run yet"
        d = self.decision
        if not d.resolved:
            return (
                f"{d.n_rollouts} rollouts: ranking NOT established at alpha={self.alpha}; "
                f"{len(d.undecided_pairs)} pair(s) are statistically indistinguishable."
            )
        order = " > ".join(str(self.models[i]) for i in d.order)
        return f"{d.n_rollouts} rollouts: {order}"


def from_lighteval_tasks(
    tasks: Iterable[Any],
    models: Sequence[Any],
    **kwargs: Any,
) -> QuorumSequentialSampler:
    """Build a sampler from lighteval task objects (or plain names).

    Accepts anything iterable; task identity is whatever the objects are, since the
    sampler only ever passes them back to the caller's ``run_fn``.

    Claim: infrastructure
    """
    task_list = list(tasks)
    names = [getattr(t, "name", t) for t in task_list]
    logger.debug("building sampler over %d tasks and %d models", len(names), len(models))
    return QuorumSequentialSampler(task_list, list(models), **kwargs)


def to_lighteval_results(
    decision: RankDecision,
    models: Sequence[Any],
) -> dict[str, Any]:
    """Export a decision as a lighteval-shaped results dict.

    The exported payload carries ``resolved`` and ``indistinguishable_pairs`` alongside
    the order, so a downstream leaderboard can render "we cannot tell these apart"
    instead of silently ranking on noise.

    Claim: coverage
    """
    def label(i: int) -> str:
        return str(models[i]) if 0 <= i < len(models) else f"model-{i}"

    return {
        "quorum": {
            "resolved": bool(decision.resolved),
            "alpha_family_wise": True,
            "order": [label(i) for i in decision.order],
            "n_rollouts": int(decision.n_rollouts),
            "cost_usd": float(decision.cost_usd),
            "cost_sec": float(decision.cost_sec),
            "indistinguishable_pairs": [
                [label(a), label(c)] for a, c in decision.undecided_pairs
            ],
            "intervals": {
                f"{label(a)}|{label(c)}": [float(lo), float(hi)]
                for (a, c), (lo, hi) in decision.intervals.items()
                if a < c
            },
            "extra_rollouts_projection": {
                f"{label(a)}|{label(c)}": int(v)
                for (a, c), v in decision.extra_rollouts_needed.items()
            },
            "note": (
                "Pairs listed as indistinguishable are not ordered by this run. "
                "The projection is an extrapolation, not a guarantee."
            ),
        }
    }
