"""Adapters that let existing evaluation harnesses drive a Quorum sequential test.

The adapters deliberately depend on nothing but a callable that produces a rollout, so a
harness does not have to adopt Quorum's data model to get rank-targeted early stopping.

Claim: infrastructure
"""

from __future__ import annotations

from quorum.adapters.lighteval_adapter import (
    QuorumSequentialSampler,
    rollout_from_lighteval_sample,
)

__all__ = ["QuorumSequentialSampler", "rollout_from_lighteval_sample"]
