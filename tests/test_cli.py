"""The CLI must never overstate what the data supports.

The output-discipline tests are the important ones here: an unresolved pair has to be
reported as indistinguishable, with no superiority language anywhere in the output.

Claim: coverage
"""

from __future__ import annotations

import json

import pytest

from quorum.cli import build_parser, format_decision, main, parse_budget
from quorum.types import RankDecision

SUPERIORITY_WORDS = ("better than", "beats", "outperforms", "superior", "worse than",
                     "inferior", "loses to", "worst", "best model")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$50", (50.0, None)),
        ("50", (50.0, None)),
        ("50usd", (50.0, None)),
        (" $12.50 ", (12.5, None)),
        ("120s", (None, 120.0)),
        ("30m", (None, 1800.0)),
        ("1h", (None, 3600.0)),
        (None, (None, None)),
    ],
)
def test_parse_budget(text, expected) -> None:
    """Budget strings parse into exactly one of dollars or seconds.

    Claim: infrastructure
    """
    assert parse_budget(text) == expected


def test_parse_budget_rejects_nonsense() -> None:
    """A malformed budget is an error, not a silent default.

    Claim: infrastructure
    """
    with pytest.raises(ValueError):
        parse_budget("about fifty quid")


def test_undecided_output_says_indistinguishable_and_claims_nothing() -> None:
    """An interval containing zero must produce no ordering claim at all.

    Claim: coverage
    """
    decision = RankDecision(
        order=[0, 1], resolved=False, n_rollouts=40, cost_usd=2.0, cost_sec=30.0,
        intervals={(0, 1): (-0.2, 0.3)}, undecided_pairs=[(0, 1)],
    )
    text = format_decision(decision, ["alpha", "beta"], 0.05, focus=(0, 1)).lower()
    assert "indistinguishable" in text
    assert "does not support ordering" in text
    for word in SUPERIORITY_WORDS:
        assert word not in text, f"undecided output contains superiority language: {word!r}"


def test_separated_output_reports_the_direction_the_interval_gives() -> None:
    """When the interval excludes zero the report states the ordering it certifies.

    Claim: coverage
    """
    decision = RankDecision(
        order=[0, 1], resolved=True, n_rollouts=400, cost_usd=9.0, cost_sec=90.0,
        intervals={(0, 1): (0.05, 0.30)}, undecided_pairs=[],
    )
    text = format_decision(decision, ["alpha", "beta"], 0.05, focus=(0, 1))
    assert "separated" in text
    assert "orders alpha above beta" in text
    # Negative orientation must flip the reported direction, not the labels.
    flipped = RankDecision(
        order=[1, 0], resolved=True, n_rollouts=400, cost_usd=9.0, cost_sec=90.0,
        intervals={(0, 1): (-0.30, -0.05)}, undecided_pairs=[],
    )
    text2 = format_decision(flipped, ["alpha", "beta"], 0.05, focus=(0, 1))
    assert "orders beta above alpha" in text2


def test_projection_is_labelled_as_a_projection() -> None:
    """The extra-rollouts number must never read as a guarantee.

    Claim: coverage
    """
    decision = RankDecision(
        order=[0, 1], resolved=False, n_rollouts=200, cost_usd=5.0, cost_sec=50.0,
        intervals={(0, 1): (-0.1, 0.2)}, undecided_pairs=[(0, 1)],
        extra_rollouts_needed={(0, 1): 1234},
    )
    text = format_decision(decision, ["a", "b"], 0.05, focus=(0, 1))
    assert "not a guarantee" in text
    assert "1,234" in text


def test_compare_runs_end_to_end(tmp_path) -> None:
    """The brief's invocation works, and exits 0 when it resolves.

    Claim: infrastructure
    """
    out = tmp_path / "d.json"
    code = main(["compare", "modelA", "modelB", "--benchmark", "synthetic-easy",
                 "--budget", "$50", "--json", str(out)])
    assert code in (0, 2)
    payload = json.loads(out.read_text())
    assert "resolved" in payload and "intervals" in payload
    assert "Synthetic benchmark" in payload["note"]


def test_exhausted_budget_exits_two() -> None:
    """A budget that runs out before the ranking is settled is exit code 2.

    Claim: coverage
    """
    code = main(["compare", "A", "B", "--benchmark", "synthetic-hard", "--budget", "$1"])
    assert code == 2


def test_unknown_benchmark_is_a_clean_error() -> None:
    """A typo'd preset produces a message listing the real ones, not a traceback.

    Claim: infrastructure
    """
    with pytest.raises(SystemExit) as exc:
        main(["compare", "A", "B", "--benchmark", "no-such-preset"])
    assert "synthetic-easy" in str(exc.value)


def test_parser_exposes_the_documented_flags() -> None:
    """Every flag the README documents must actually exist.

    Claim: infrastructure
    """
    parser = build_parser()
    args = parser.parse_args(["compare", "a", "b", "--benchmark", "synthetic-easy",
                              "--budget", "$5", "--alpha", "0.01", "--seed", "3",
                              "--no-allocator", "--no-cuped", "--truncation",
                              "--audit-rate", "0.5", "--unpaired"])
    assert args.alpha == 0.01 and args.no_allocator and args.truncation
