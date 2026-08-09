"""Load real logged rollouts and replay them through the same Quorum machinery.

Everything in Quorum consumes rollouts through one narrow interface: a ``.rollout()``
method that returns a :class:`~quorum.types.Rollout` for a requested cell. The simulator
implements it by generating a world; :class:`FrozenGridSimulator` implements it by
sampling **with replacement from rollouts that were actually observed** in that cell of a
logged evaluation run. That is the whole trick of this module: the sequential design, the
allocator, the truncation correction and the control variates all run *unchanged* on real
data, so the benchmark numbers and the real-data numbers are produced by the same code
path rather than by two implementations that agree by hope.

Resampling a fixed pool makes the loader a design-based bootstrap: the "truth" is the
empirical cell distribution of the log, so ``Delta`` is a well-defined finite-population
contrast and coverage statements remain meaningful.

Only ``csv``/``json`` from the standard library are used -- the core must stay light --
and :func:`load_hf_dataset` imports ``datasets`` lazily inside the call, so importing
this module never touches the network or a heavy optional dependency.

Claim: infrastructure
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from quorum.types import GridSpec, Rollout, block_index

__all__ = [
    "FrozenGridSimulator",
    "load_hf_dataset",
    "load_rollouts_csv",
    "load_rollouts_jsonl",
    "rollouts_from_records",
    "to_frozen_grid",
]

logger = logging.getLogger(__name__)

_ALIASES: dict[str, tuple[str, ...]] = {
    "task": ("task", "task_id", "task_name", "item", "item_id", "problem", "problem_id",
             "benchmark_item", "doc_id"),
    "seed": ("seed", "seed_id", "instance", "instance_id", "variant", "repeat", "rep",
             "sample", "sample_id"),
    "scaffold": ("scaffold", "scaffold_id", "scaffold_name", "harness", "agent",
                 "prompt", "prompt_id", "template", "config"),
    "model": ("model", "model_id", "model_name", "system", "candidate", "run"),
    "success": ("success", "correct", "is_correct", "passed", "pass", "label", "score",
                "reward", "acc", "accuracy", "exact_match"),
}
"""Accepted column names per logical field, in priority order."""

_OPTIONAL_FIELDS: tuple[str, ...] = (
    "n_steps",
    "n_steps_full",
    "cost_usd",
    "cost_sec",
    "trajectory",
    "completed",
    "audited",
    "truncated_at",
    "true_success",
)

_OPTIONAL_ALIASES: dict[str, tuple[str, ...]] = {
    "n_steps": ("n_steps", "steps", "num_steps", "n_tokens_steps"),
    "n_steps_full": ("n_steps_full", "steps_full", "full_steps"),
    "cost_usd": ("cost_usd", "usd", "cost", "dollars", "price_usd"),
    "cost_sec": ("cost_sec", "sec", "seconds", "latency_sec", "runtime_sec", "duration"),
    "trajectory": ("trajectory", "traj", "steps_trace", "health", "signal"),
    "completed": ("completed", "complete", "finished"),
    "audited": ("audited", "audit"),
    "truncated_at": ("truncated_at", "truncation_step", "stopped_at"),
    "true_success": ("true_success", "gold_success", "oracle_success"),
}

_TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "pass", "passed", "correct", "success"})
_FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "fail", "failed", "incorrect", ""})


def _resolve_columns(columns: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Map logical field names onto the raw column names present in the file.

    Real evaluation logs never use one schema, so a small alias table is the difference
    between "runs on your data" and "runs on data we made up".

    Claim: infrastructure
    """
    lower = {str(c).strip().lower(): str(c) for c in columns}
    required: dict[str, str] = {}
    for field, names in _ALIASES.items():
        for candidate in names:
            if candidate in lower:
                required[field] = lower[candidate]
                break
    optional: dict[str, str] = {}
    for field, names in _OPTIONAL_ALIASES.items():
        for candidate in names:
            if candidate in lower:
                optional[field] = lower[candidate]
                break
    return required, optional


def _as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion that never raises on messy log values.

    Claim: infrastructure
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return default if not np.isfinite(result) else result
    text = str(value).strip()
    if text == "":
        return default
    try:
        result = float(text)
    except ValueError:
        low = text.lower()
        if low in _TRUE_STRINGS:
            return 1.0
        if low in _FALSE_STRINGS:
            return 0.0
        return default
    return default if not np.isfinite(result) else result


def _as_int(value: Any, default: int = 0) -> int:
    """Best-effort int conversion for step counts and indices.

    Claim: infrastructure
    """
    return int(round(_as_float(value, float(default))))


def _as_bool(value: Any, default: bool) -> bool:
    """Best-effort bool conversion for ``completed`` / ``audited`` columns.

    Claim: infrastructure
    """
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    return default


def _as_success(value: Any) -> int:
    """Coerce a success/score column to ``{0, 1}``, thresholding continuous scores at 0.5.

    Claim: infrastructure
    """
    return 1 if _as_float(value, 0.0) >= 0.5 else 0


def _parse_trajectory(value: Any) -> np.ndarray:
    """Parse a per-step scalar trace from JSON, a delimited string, or a sequence.

    An absent or unparseable trace yields an empty array; the inner truncation rule then
    simply never fires on that rollout, which is the correct graceful degradation --
    logs without per-step traces still support every other part of the procedure.

    Claim: unbiased-truncation
    """
    if value is None:
        return np.zeros(0, dtype=float)
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray([_as_float(v, 0.0) for v in value], dtype=float)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.asarray([float(value)], dtype=float)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return np.zeros(0, dtype=float)
    if text[0] in "[(":
        try:
            parsed = json.loads(text.replace("(", "[").replace(")", "]"))
            if isinstance(parsed, (list, tuple)):
                return np.asarray([_as_float(v, 0.0) for v in parsed], dtype=float)
        except (ValueError, TypeError):
            text = text.strip("[]()")
    for sep in (",", ";", "|", " "):
        if sep in text:
            parts = [p for p in text.split(sep) if p.strip() != ""]
            return np.asarray([_as_float(p, 0.0) for p in parts], dtype=float)
    return np.asarray([_as_float(text, 0.0)], dtype=float)


def _sort_levels(values: Iterable[Any]) -> list[Any]:
    """Deterministic level ordering: numeric when possible, else lexicographic.

    Determinism matters: the integer model index is what every downstream array is keyed
    by, so two loads of the same file must produce the same mapping or results silently
    stop being comparable.

    Claim: infrastructure
    """
    unique = list({_hashable(v): v for v in values}.values())
    try:
        numeric = sorted(unique, key=lambda v: float(v))
        return numeric
    except (TypeError, ValueError):
        return sorted(unique, key=lambda v: str(v))


def _hashable(value: Any) -> Any:
    """Normalise a raw level value to something hashable and JSON-friendly.

    Claim: infrastructure
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def rollouts_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source: str = "records",
) -> tuple[list[Rollout], GridSpec, dict[str, Any]]:
    """Turn raw dict-like rows into :class:`~quorum.types.Rollout` objects and a grid spec.

    ``task``, ``model`` and ``success`` are required (under any of their aliases);
    ``seed`` and ``scaffold`` default to a single level when absent, so a plain
    per-item leaderboard dump loads as a ``K=1, S=1`` grid instead of failing. Every
    other column is optional and defaults conservatively: no trace, unit step count,
    zero cost, ``completed=True``.

    Level values (which may be model *names*) are mapped to contiguous integer indices
    by a deterministic sort; the mapping is returned in the metadata so a caller can
    label a table without ever needing the loader to emit model names into a claim.

    Claim: infrastructure
    """
    rows = list(records)
    if not rows:
        raise ValueError(f"no rollout records found in {source}")

    seen: list[str] = []
    seen_set: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen_set:
                seen_set.add(key)
                seen.append(str(key))
    required, optional = _resolve_columns(seen)

    defaulted: list[str] = []
    for field in ("task", "model", "success"):
        if field not in required:
            raise KeyError(
                f"{source}: required column '{field}' not found; "
                f"looked for {list(_ALIASES[field])}, saw {seen}"
            )
    for field in ("seed", "scaffold"):
        if field not in required:
            defaulted.append(field)
            logger.warning(
                "%s: no '%s' column; treating the whole log as a single %s level",
                source, field, field,
            )

    raw: dict[str, list[Any]] = {f: [] for f in ("task", "seed", "scaffold", "model")}
    n_skipped = 0
    kept: list[Mapping[str, Any]] = []
    for row in rows:
        try:
            values = {}
            for field in ("task", "seed", "scaffold", "model"):
                if field in required:
                    values[field] = _hashable(row.get(required[field]))
                else:
                    values[field] = 0
            if values["task"] is None or values["model"] is None:
                raise ValueError("null key")
        except (ValueError, TypeError):
            n_skipped += 1
            continue
        for field in ("task", "seed", "scaffold", "model"):
            raw[field].append(values[field])
        kept.append(row)
    if not kept:
        raise ValueError(f"{source}: every row was unusable ({n_skipped} skipped)")

    levels = {f: _sort_levels(raw[f]) for f in ("task", "seed", "scaffold", "model")}
    index_of = {
        f: {_hashable(v): i for i, v in enumerate(levels[f])}
        for f in ("task", "seed", "scaffold", "model")
    }
    spec = GridSpec(
        n_tasks=len(levels["task"]),
        n_seeds=len(levels["seed"]),
        n_scaffolds=len(levels["scaffold"]),
        n_models=len(levels["model"]),
    )

    rollouts: list[Rollout] = []
    for i, row in enumerate(kept):
        t = index_of["task"][raw["task"][i]]
        k = index_of["seed"][raw["seed"][i]]
        s = index_of["scaffold"][raw["scaffold"][i]]
        m = index_of["model"][raw["model"][i]]
        success = _as_success(row.get(required["success"]))
        traj = _parse_trajectory(row.get(optional["trajectory"])) if "trajectory" in optional else np.zeros(0)
        n_steps = _as_int(row.get(optional["n_steps"]), 0) if "n_steps" in optional else 0
        if n_steps <= 0:
            n_steps = int(traj.size) if traj.size else 1
        n_full = _as_int(row.get(optional["n_steps_full"]), 0) if "n_steps_full" in optional else 0
        n_full = max(n_full, n_steps, int(traj.size), 1)
        completed = _as_bool(row.get(optional["completed"]), True) if "completed" in optional else True
        audited = _as_bool(row.get(optional["audited"]), False) if "audited" in optional else False
        truncated_at = _as_int(row.get(optional["truncated_at"]), -1) if "truncated_at" in optional else -1
        if completed:
            truncated_at = -1
        true_success = (
            _as_success(row.get(optional["true_success"]))
            if "true_success" in optional
            else success
        )
        rollouts.append(
            Rollout(
                task=t,
                seed=k,
                scaffold=s,
                model=m,
                success=success,
                completed=completed,
                n_steps=n_steps,
                n_steps_full=n_full,
                cost_usd=_as_float(row.get(optional["cost_usd"]), 0.0) if "cost_usd" in optional else 0.0,
                cost_sec=_as_float(row.get(optional["cost_sec"]), 0.0) if "cost_sec" in optional else 0.0,
                trajectory=np.asarray(traj, dtype=float),
                audited=audited,
                truncated_at=truncated_at,
                true_success=true_success,
            )
        )

    counts = np.zeros(spec.grid_shape, dtype=int)
    for r in rollouts:
        counts[r.task, r.seed, r.scaffold, r.model] += 1
    empty = int((counts == 0).sum())

    meta: dict[str, Any] = {
        "source": source,
        "n_rows": len(rows),
        "n_rollouts": len(rollouts),
        "n_skipped": n_skipped,
        "columns": seen,
        "resolved_columns": dict(required),
        "optional_columns": dict(optional),
        "defaulted_columns": defaulted,
        "missing_optional": [f for f in _OPTIONAL_FIELDS if f not in optional],
        "levels": {f: [_jsonable(v) for v in levels[f]] for f in levels},
        "index_of": {
            f: {str(_jsonable(v)): i for v, i in index_of[f].items()} for f in index_of
        },
        "grid_shape": list(spec.grid_shape),
        "n_cells": spec.n_cells,
        "n_empty_cells": empty,
        "min_per_cell": int(counts.min()),
        "max_per_cell": int(counts.max()),
        "mean_per_cell": float(counts.mean()),
        "has_trajectories": bool(any(r.trajectory.size > 1 for r in rollouts)),
    }
    if empty:
        logger.warning(
            "%s: %d of %d cells have no observed rollouts; FrozenGridSimulator will "
            "pool a fallback for those cells",
            source, empty, spec.n_cells,
        )
    logger.info(
        "%s: loaded %d rollouts on grid T=%d K=%d S=%d M=%d",
        source, len(rollouts), *spec.grid_shape,
    )
    return rollouts, spec, meta


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars to plain Python so metadata can be serialised.

    Claim: infrastructure
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_rollouts_jsonl(
    path: str | os.PathLike[str],
) -> tuple[list[Rollout], GridSpec, dict[str, Any]]:
    """Load rollout-level records from JSON Lines (or a single JSON array file).

    Blank lines and unparseable lines are skipped and counted in the metadata rather
    than aborting the load: a real log with three corrupt lines out of a million should
    still produce an evaluation, and the count makes the omission auditable.

    Claim: infrastructure
    """
    path = os.fspath(path)
    records: list[Mapping[str, Any]] = []
    n_bad = 0
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        while first.isspace():
            first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"{path}: expected a JSON array of objects")
            for item in payload:
                if isinstance(item, Mapping):
                    records.append(item)
                else:
                    n_bad += 1
        else:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                if isinstance(item, Mapping):
                    records.append(item)
                else:
                    n_bad += 1
    rollouts, spec, meta = rollouts_from_records(records, source=path)
    meta["format"] = "jsonl"
    meta["path"] = path
    meta["n_unparseable_lines"] = n_bad
    return rollouts, spec, meta


def load_rollouts_csv(
    path: str | os.PathLike[str],
) -> tuple[list[Rollout], GridSpec, dict[str, Any]]:
    """Load rollout-level records from a CSV with a header row.

    Uses the stdlib :mod:`csv` module on purpose -- the core of Quorum must import
    nothing heavier than numpy/scipy, and a leaderboard dump is not a reason to pull in
    a dataframe library.

    Claim: infrastructure
    """
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header row")
        records = [dict(row) for row in reader]
    rollouts, spec, meta = rollouts_from_records(records, source=path)
    meta["format"] = "csv"
    meta["path"] = path
    return rollouts, spec, meta


def load_hf_dataset(
    repo_id: str,
    split: str = "train",
    token: str | None = None,
) -> tuple[list[Rollout], GridSpec, dict[str, Any]]:
    """Load rollout-level records from a Hugging Face dataset repo.

    ``datasets`` is imported **inside** this function, never at module import time: the
    core of Quorum must remain importable with numpy and scipy alone, and no other entry
    point may accidentally acquire a network dependency by importing this module.

    Claim: infrastructure
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415  (deliberate lazy import)
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "load_hf_dataset requires the optional 'datasets' dependency: "
            "pip install 'quorum-eval[hub]'"
        ) from exc
    dataset = load_dataset(repo_id, split=split, token=token)
    records = [dict(row) for row in dataset]
    rollouts, spec, meta = rollouts_from_records(records, source=f"hf://{repo_id}#{split}")
    meta["format"] = "hf"
    meta["repo_id"] = repo_id
    meta["split"] = split
    return rollouts, spec, meta


class FrozenGridSimulator:
    """Replays logged rollouts through the ``RolloutSimulator`` interface.

    ``rollout(t, k, s, m)`` samples uniformly **with replacement** from the rollouts
    observed in that cell, so the induced "truth" is the empirical cell distribution of
    the log. Every Quorum component -- the Horvitz-Thompson pair channel, the
    decision-targeted allocator, the inner truncator, CUPED -- therefore runs unchanged
    on real data, and the finite-population contrast ``Delta`` stays well defined.

    A truncator is honoured by *replaying* it against the logged trajectory: if the rule
    fires at step ``j`` the audit coin is flipped once, and a non-audited stop yields the
    rule's guess with cost pro-rated to the steps that would have been executed. Logs
    without per-step traces simply never trigger the rule, which is the right graceful
    degradation rather than a silent fabrication of trajectories.

    Cells with no observations are served, by default, from the nearest non-empty pool
    (same model within the scaffold, then the same model anywhere, then everything) and
    logged once. The alternative -- refusing to serve them -- would break the mandatory
    exploration floor that keeps the confidence sequence finite-width.

    Claim: infrastructure
    """

    def __init__(
        self,
        rollouts: Sequence[Rollout],
        spec: GridSpec,
        *,
        seed: int = 0,
        on_missing: str = "pool",
    ) -> None:
        """Index a rollout log by cell and bind a reproducible RNG stream.

        Claim: infrastructure
        """
        if on_missing not in {"pool", "error", "zero"}:
            raise ValueError("on_missing must be one of {'pool', 'error', 'zero'}")
        self.spec = spec
        self.on_missing = on_missing
        self.seed = int(seed)
        self._rng: np.random.Generator = np.random.default_rng(seed)

        n_models = spec.n_models
        self._by_cell: list[list[Rollout]] = [[] for _ in range(spec.n_cells)]
        self._by_scaffold_model: list[list[Rollout]] = [
            [] for _ in range(spec.n_scaffolds * n_models)
        ]
        self._by_model: list[list[Rollout]] = [[] for _ in range(n_models)]
        self._all: list[Rollout] = []
        self.counts = np.zeros(spec.grid_shape, dtype=int)

        for r in rollouts:
            if not (
                0 <= r.task < spec.n_tasks
                and 0 <= r.seed < spec.n_seeds
                and 0 <= r.scaffold < spec.n_scaffolds
                and 0 <= r.model < n_models
            ):
                raise IndexError(
                    f"rollout {(r.task, r.seed, r.scaffold, r.model)} is outside the grid "
                    f"{spec.grid_shape}"
                )
            cell = self._cell_id(r.task, r.seed, r.scaffold, r.model)
            self._by_cell[cell].append(r)
            self._by_scaffold_model[r.scaffold * n_models + r.model].append(r)
            self._by_model[r.model].append(r)
            self._all.append(r)
            self.counts[r.task, r.seed, r.scaffold, r.model] += 1
        if not self._all:
            raise ValueError("FrozenGridSimulator needs at least one rollout")

        self._warned_cells: set[int] = set()
        self.n_rollouts: int = 0
        self.total_usd: float = 0.0
        self.total_sec: float = 0.0
        self.total_steps: int = 0
        self.total_steps_full: int = 0
        self.n_truncated: int = 0
        self.n_audited: int = 0

    def _cell_id(self, t: int, k: int, s: int, m: int) -> int:
        """Flat cell id, consistent with the row-major block flattening of SPEC 1.

        Claim: infrastructure
        """
        b = block_index(int(t), int(k), int(s), self.spec.n_seeds, self.spec.n_scaffolds)
        return b * self.spec.n_models + int(m)

    def reset_counters(self) -> None:
        """Zero the cost/step accounting (the RNG stream is untouched).

        Claim: infrastructure
        """
        self.n_rollouts = 0
        self.total_usd = 0.0
        self.total_sec = 0.0
        self.total_steps = 0
        self.total_steps_full = 0
        self.n_truncated = 0
        self.n_audited = 0

    def reseed(self, seed: int) -> None:
        """Restart the internal RNG stream, for replication-level reproducibility.

        Claim: infrastructure
        """
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)

    @property
    def counters(self) -> dict[str, float]:
        """Snapshot of the accounting counters, matching ``RolloutSimulator.counters``.

        Claim: low-cost
        """
        return {
            "n_rollouts": float(self.n_rollouts),
            "total_usd": self.total_usd,
            "total_sec": self.total_sec,
            "total_steps": float(self.total_steps),
            "total_steps_full": float(self.total_steps_full),
            "n_truncated": float(self.n_truncated),
            "n_audited": float(self.n_audited),
        }

    @property
    def compute_saved(self) -> float:
        """Fraction of logged trajectory steps avoided by replayed truncation.

        Claim: unbiased-truncation
        """
        if self.total_steps_full <= 0:
            return 0.0
        return 1.0 - (self.total_steps / self.total_steps_full)

    def n_observed(self, t: int, k: int, s: int, m: int) -> int:
        """Number of logged rollouts available in a cell (0 means it will be pooled).

        Claim: infrastructure
        """
        return int(self.counts[int(t), int(k), int(s), int(m)])

    @property
    def empty_cells(self) -> list[tuple[int, int, int, int]]:
        """Cells with no logged rollouts, as ``(t, k, s, m)`` tuples.

        Claim: infrastructure
        """
        return [tuple(int(i) for i in idx) for idx in np.argwhere(self.counts == 0)]

    def mean_costs(self) -> tuple[np.ndarray, np.ndarray]:
        """Observed mean ``$`` and second cost per cell, for the cost-aware allocator.

        Cells with no observations inherit the grand mean, so the allocator always sees a
        finite, positive price and never divides by zero.

        Claim: low-cost
        """
        shape = self.spec.grid_shape
        usd = np.zeros(shape, dtype=float)
        sec = np.zeros(shape, dtype=float)
        for cell, pool in enumerate(self._by_cell):
            if not pool:
                continue
            idx = np.unravel_index(cell, shape)
            usd[idx] = float(np.mean([r.cost_usd for r in pool]))
            sec[idx] = float(np.mean([r.cost_sec for r in pool]))
        grand_usd = float(np.mean([r.cost_usd for r in self._all]))
        grand_sec = float(np.mean([r.cost_sec for r in self._all]))
        empty = self.counts == 0
        usd[empty] = grand_usd
        sec[empty] = grand_sec
        return usd, sec

    def observed_success_rate(self) -> np.ndarray:
        """Per-cell empirical success rate, ``nan`` where nothing was observed.

        This is the finite-population truth the frozen replay is unbiased for; tests and
        diagnostics compare a confidence sequence against contrasts of this array.

        Claim: infrastructure
        """
        rate = np.full(self.spec.grid_shape, np.nan, dtype=float)
        for cell, pool in enumerate(self._by_cell):
            if not pool:
                continue
            idx = np.unravel_index(cell, self.spec.grid_shape)
            rate[idx] = float(np.mean([r.success for r in pool]))
        return rate

    def _pool_for(self, t: int, k: int, s: int, m: int) -> list[Rollout]:
        """Resolve the sampling pool for a cell, applying the missing-cell policy.

        Claim: infrastructure
        """
        cell = self._cell_id(t, k, s, m)
        pool = self._by_cell[cell]
        if pool:
            return pool
        if self.on_missing == "error":
            raise KeyError(f"no logged rollouts for cell (t={t}, k={k}, s={s}, m={m})")
        if cell not in self._warned_cells:
            self._warned_cells.add(cell)
            logger.warning(
                "cell (t=%d, k=%d, s=%d, m=%d) has no logged rollouts; falling back to a "
                "pooled sample (policy=%s)",
                t, k, s, m, self.on_missing,
            )
        if self.on_missing == "zero":
            return []
        fallback = self._by_scaffold_model[int(s) * self.spec.n_models + int(m)]
        if fallback:
            return fallback
        fallback = self._by_model[int(m)]
        return fallback if fallback else self._all

    def rollout(
        self,
        t: int,
        k: int,
        s: int,
        m: int,
        *,
        truncator: Any | None = None,
        rng: np.random.Generator | None = None,
    ) -> Rollout:
        """Draw one logged rollout for a cell, replaying inner truncation if asked.

        The returned object is always a fresh :class:`~quorum.types.Rollout` labelled
        with the *requested* cell, so the log is never mutated and a pooled fallback is
        still attributed to the cell the design paid for. As in the simulator,
        ``true_success`` carries the logged ground truth and must be read only by tests.

        Claim: unbiased-truncation
        """
        shape = self.spec.grid_shape
        for name, value, size in zip(("task", "seed", "scaffold", "model"), (t, k, s, m), shape):
            if not 0 <= int(value) < int(size):
                raise IndexError(f"{name} index {value} out of range [0, {size})")
        t, k, s, m = int(t), int(k), int(s), int(m)
        gen = self._rng if rng is None else rng

        pool = self._pool_for(t, k, s, m)
        if not pool:
            base = Rollout(
                task=t, seed=k, scaffold=s, model=m, success=0, completed=True,
                n_steps=1, n_steps_full=1, true_success=0,
            )
        else:
            base = pool[int(gen.integers(len(pool)))]

        traj = np.asarray(base.trajectory, dtype=float).reshape(-1)
        base_steps = max(int(base.n_steps), int(traj.size), 1)
        n_full = max(int(base.n_steps_full), base_steps)
        true_success = base.true_success if base.true_success in (0, 1) else int(base.success)

        truncated_at = -1
        audited = False
        replayable = truncator is not None and base.completed and traj.size >= 2
        if replayable:
            for j in range(1, int(traj.size)):
                prefix = traj[: j + 1]
                prefix.flags.writeable = False  # the rule observes, it does not edit
                if bool(truncator.should_stop(prefix, j)):
                    truncated_at = j
                    audited = bool(truncator.audit(gen))
                    break

        if truncated_at >= 0:
            completed = False
            if audited:
                n_steps = base_steps
                observed = true_success
            else:
                n_steps = truncated_at + 1
                observed = 0
        else:
            completed = bool(base.completed)
            n_steps = base_steps
            observed = int(base.success)
            audited = bool(base.audited)
            truncated_at = int(base.truncated_at) if not completed else -1

        frac = n_steps / base_steps if base_steps > 0 else 1.0
        cost_usd = float(base.cost_usd) * frac
        cost_sec = float(base.cost_sec) * frac

        self.n_rollouts += 1
        self.total_usd += cost_usd
        self.total_sec += cost_sec
        self.total_steps += n_steps
        self.total_steps_full += n_full
        if truncated_at >= 0:
            self.n_truncated += 1
            if audited:
                self.n_audited += 1

        return Rollout(
            task=t,
            seed=k,
            scaffold=s,
            model=m,
            success=int(observed),
            completed=completed,
            n_steps=int(n_steps),
            n_steps_full=int(n_full),
            cost_usd=cost_usd,
            cost_sec=cost_sec,
            trajectory=traj[:n_steps].copy(),
            audited=audited,
            truncated_at=int(truncated_at),
            true_success=int(true_success),
        )


def to_frozen_grid(
    rollouts: Sequence[Rollout],
    spec: GridSpec,
    *,
    seed: int = 0,
    on_missing: str = "pool",
) -> FrozenGridSimulator:
    """Wrap a logged rollout list as a resampling simulator over the same grid.

    This is the seam between "Quorum on a simulator" and "Quorum on your evaluation
    logs": nothing downstream changes, because the only thing either object promises is
    ``.rollout()`` plus the cost/step counters.

    Claim: infrastructure
    """
    return FrozenGridSimulator(rollouts, spec, seed=seed, on_missing=on_missing)
