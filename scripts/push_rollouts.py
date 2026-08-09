"""Publish rollout-level evaluation data to the Hugging Face Hub.

Rollout-level, not aggregate: the whole point of the Quorum estimators is that they need
the individual outcomes (and their costs and truncation flags) to compute a confidence
sequence. A leaderboard that publishes only means cannot be re-analysed by anyone.

``--dry-run`` builds the dataset, prints the schema and a preview, and exits -- no token,
no network.

Claim: infrastructure
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

SCHEMA_DOC = {
    "task": "int -- index of the task/item within the evaluation grid",
    "seed": "int -- index of the problem-instance variant",
    "scaffold": "int -- index of the agent scaffold / harness configuration",
    "model": "string -- model label (a placeholder in synthetic data)",
    "success": "int 0/1 -- observed outcome; may be a truncation rule's guess",
    "completed": "bool -- False when the inner truncation rule stopped the rollout",
    "n_steps": "int -- trajectory steps actually executed",
    "cost_usd": "float -- dollars spent on this rollout",
    "cost_sec": "float -- wall-clock seconds spent on this rollout",
    "truncated_at": "int -- step at which truncation fired, -1 if it never did",
    "audited": "bool -- True when the audit coin spared this rollout from truncation",
}

CARD = """\
# Rollout-level evaluation records

One row per rollout, produced by or for [Quorum](https://github.com/quorum-eval/quorum).

Rollout-level data is what makes a leaderboard re-analysable: anytime-valid confidence
sequences, paired scaffold-marginalised contrasts and control-variate adjustments all
need the individual outcomes, not the means.

`completed=False` rows were stopped early by an inner truncation rule. **Do not average
`success` over those rows naively** -- the resulting mean is biased low. Use the audit
subsample (`audited=True`) with the doubly-robust correction in
`quorum.nested_seq.corrected_outcome`, which is unbiased regardless of how well
calibrated the truncation rule was.

Model labels in synthetic splits are placeholders and are not claims about real systems.
"""

REQUIRED = ("task", "seed", "scaffold", "model", "success")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream records from a JSON-lines file.

    Claim: infrastructure
    """
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def _read_csv(path: Path) -> Iterator[dict[str, Any]]:
    """Stream records from a CSV file.

    Claim: infrastructure
    """
    with path.open(newline="") as fh:
        yield from csv.DictReader(fh)


def _coerce(record: dict[str, Any]) -> dict[str, Any]:
    """Normalise one record to the published schema.

    Claim: infrastructure
    """
    def as_bool(value: Any, default: bool) -> bool:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "t")

    def as_num(value: Any, default: float, cast: Any = float) -> Any:
        if value is None or value == "":
            return cast(default)
        return cast(float(value))

    out = {
        "task": as_num(record.get("task"), 0, int),
        "seed": as_num(record.get("seed"), 0, int),
        "scaffold": as_num(record.get("scaffold"), 0, int),
        "model": str(record.get("model", "model-0")),
        "success": as_num(record.get("success"), 0, int),
        "completed": as_bool(record.get("completed"), True),
        "n_steps": as_num(record.get("n_steps"), 0, int),
        "cost_usd": as_num(record.get("cost_usd"), 0.0),
        "cost_sec": as_num(record.get("cost_sec"), 0.0),
        "truncated_at": as_num(record.get("truncated_at"), -1, int),
        "audited": as_bool(record.get("audited"), False),
    }
    return out


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read and normalise every record from a JSONL or CSV file.

    Claim: infrastructure
    """
    if not path.exists():
        raise SystemExit(f"push_rollouts: no such file: {path}")
    reader = _read_jsonl if path.suffix in (".jsonl", ".ndjson") else _read_csv
    records = [_coerce(r) for r in reader(path)]
    if not records:
        raise SystemExit(f"push_rollouts: {path} contained no records")
    missing = [c for c in REQUIRED if c not in records[0]]
    if missing:
        raise SystemExit(f"push_rollouts: missing required column(s): {missing}")
    return records


def build_dataset(records: Sequence[dict[str, Any]]) -> Any:
    """Build a ``datasets.Dataset``. Imported lazily so ``--dry-run`` needs no install.

    Claim: infrastructure
    """
    from datasets import Dataset  # noqa: PLC0415 - deliberately lazy

    columns = {key: [r[key] for r in records] for key in records[0]}
    return Dataset.from_dict(columns)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the dataset and optionally push it to the Hub.

    Claim: infrastructure
    """
    p = argparse.ArgumentParser(description="Push rollout-level records to the HF Hub")
    p.add_argument("--input", required=True, help="rollout-level .jsonl or .csv")
    p.add_argument("--repo-id", default=None, help="e.g. your-org/quorum-rollouts")
    p.add_argument("--split", default="train")
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="build and preview only; no token or network needed")
    args = p.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records = load_records(Path(args.input).expanduser())
    n_truncated = sum(1 for r in records if not r["completed"])
    n_audited = sum(1 for r in records if r["audited"])

    print(f"records: {len(records):,}")
    print(f"models:  {sorted({r['model'] for r in records})}")
    print(f"truncated rollouts: {n_truncated:,} (audited: {n_audited:,})")
    print("\nschema:")
    for name, doc in SCHEMA_DOC.items():
        print(f"  {name:<14} {doc}")
    print("\npreview:")
    for record in records[:3]:
        print("  " + json.dumps(record))

    if args.dry_run:
        print("\n--dry-run: nothing was uploaded.")
        return 0

    if not args.repo_id:
        print("\npush_rollouts: --repo-id is required unless --dry-run", file=sys.stderr)
        return 1

    try:
        dataset = build_dataset(records)
    except ImportError:
        print("\npush_rollouts: `pip install datasets` to push (or use --dry-run)",
              file=sys.stderr)
        return 1

    logger.info("pushing %d rows to %s (split=%s, private=%s)",
                len(records), args.repo_id, args.split, args.private)
    dataset.push_to_hub(args.repo_id, split=args.split, private=args.private,
                        token=args.token)
    try:
        from huggingface_hub import DatasetCard

        DatasetCard(CARD).push_to_hub(args.repo_id, repo_type="dataset", token=args.token)
    except Exception as exc:  # pragma: no cover - network
        logger.warning("dataset pushed, but the card could not be written: %s", exc)
    print(f"pushed {len(records):,} rows to {args.repo_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
