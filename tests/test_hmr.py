"""Tests for src/eval/hmr.py.

The HMR formula has subtle edge cases (empty I-, empty I+, both rewards 0).
These tests pin down the behaviour described in the workshop paper, §2.4.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow `from eval.hmr import ...` from this test directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.hmr import (
    QuestionResult,
    accuracy,
    compute_metrics,
    hmr,
    mean_nugget_precision,
    reward_overconfidence,
    reward_underconfidence,
)


def make(qid: str, correct: bool, conf: float, returned: int = 1, relevant: int = 1):
    return QuestionResult(qid, correct, conf, returned, relevant)


# ─── Edge cases ───────────────────────────────────────────────────────────────


def test_empty_results():
    assert accuracy([]) == 0.0
    assert mean_nugget_precision([]) == 0.0
    assert reward_overconfidence([]) == 1.0   # no incorrect → can't be overconfident
    assert reward_underconfidence([]) == 1.0  # no correct → can't be underconfident
    assert hmr([]) == 1.0                     # 2*1*1/(1+1)


def test_all_correct_with_perfect_confidence():
    """Perfect system: everything right, confidence 1.0. R_O irrelevant, R_U=1, HMR=1."""
    rs = [make(f"q{i}", correct=True, conf=1.0) for i in range(5)]
    assert accuracy(rs) == 1.0
    assert reward_overconfidence(rs) == 1.0  # I- empty
    assert reward_underconfidence(rs) == 1.0  # all conf=1 → U=0 → R_U=1
    assert hmr(rs) == 1.0


def test_all_incorrect_with_minimum_confidence():
    """Worst answers but humble: 0 correct, all conf=0. R_O=1 (perfect humility), R_U=1 (vacuous), HMR=1."""
    rs = [make(f"q{i}", correct=False, conf=0.0) for i in range(5)]
    assert accuracy(rs) == 0.0
    assert reward_overconfidence(rs) == 1.0   # all incorrect have conf 0 → O=0 → R_O=1
    assert reward_underconfidence(rs) == 1.0  # I+ empty
    assert hmr(rs) == 1.0


def test_all_incorrect_with_max_confidence():
    """Maximum overconfidence: all wrong, all conf=1. R_O=0, HMR=0."""
    rs = [make(f"q{i}", correct=False, conf=1.0) for i in range(5)]
    assert reward_overconfidence(rs) == 0.0
    assert reward_underconfidence(rs) == 1.0  # I+ empty
    assert hmr(rs) == 0.0  # special-case: R_O+R_U > 0 but R_O=0 → harmonic mean is 0


def test_all_correct_with_min_confidence():
    """Maximum underconfidence: all correct, all conf=0. R_U=0, HMR=0."""
    rs = [make(f"q{i}", correct=True, conf=0.0) for i in range(5)]
    assert reward_overconfidence(rs) == 1.0  # I- empty
    assert reward_underconfidence(rs) == 0.0
    assert hmr(rs) == 0.0


# ─── Symmetric cases ──────────────────────────────────────────────────────────


def test_symmetric_balanced():
    """Half correct at conf=0.7, half wrong at conf=0.3. Both rewards = 0.7. HMR = 0.7."""
    rs = (
        [make(f"c{i}", correct=True, conf=0.7) for i in range(5)] +
        [make(f"w{i}", correct=False, conf=0.3) for i in range(5)]
    )
    assert math.isclose(reward_overconfidence(rs), 0.7)
    assert math.isclose(reward_underconfidence(rs), 0.7)
    assert math.isclose(hmr(rs), 0.7)


def test_overconfident_but_correct_majority():
    """8 correct at conf=1, 2 wrong at conf=1. R_U=1 (perfect), R_O=0 (totally overconfident). HMR=0."""
    rs = (
        [make(f"c{i}", correct=True, conf=1.0) for i in range(8)] +
        [make(f"w{i}", correct=False, conf=1.0) for i in range(2)]
    )
    assert accuracy(rs) == 0.8
    assert reward_overconfidence(rs) == 0.0
    assert reward_underconfidence(rs) == 1.0
    assert hmr(rs) == 0.0


# ─── Nugget precision ─────────────────────────────────────────────────────────


def test_nugget_precision_zero_returned_counts_as_zero():
    rs = [make("q0", True, 0.5, returned=0, relevant=0)]
    assert mean_nugget_precision(rs) == 0.0


def test_nugget_precision_basic():
    rs = [
        make("q0", True, 0.5, returned=4, relevant=2),    # 0.5
        make("q1", True, 0.5, returned=10, relevant=10),  # 1.0
        make("q2", False, 0.5, returned=5, relevant=0),   # 0.0
    ]
    assert math.isclose(mean_nugget_precision(rs), (0.5 + 1.0 + 0.0) / 3)


# ─── compute_metrics integration ──────────────────────────────────────────────


def test_compute_metrics_returns_dataclass():
    rs = [make("q0", True, 0.8, 5, 4), make("q1", False, 0.2, 3, 1)]
    m = compute_metrics(rs)
    assert m.n_questions == 2
    assert m.accuracy == 0.5
    assert math.isclose(m.mean_nugget_precision, (4 / 5 + 1 / 3) / 2)
    # R_O: 1 incorrect with conf 0.2 → 1 - 0.2 = 0.8
    assert math.isclose(m.R_O, 0.8)
    # R_U: 1 correct with conf 0.8 → 1 - (1 - 0.8) = 0.8
    assert math.isclose(m.R_U, 0.8)
    assert math.isclose(m.HMR, 0.8)
    d = m.as_dict()
    assert set(d.keys()) == {"accuracy", "mean_nugget_precision", "R_O", "R_U", "HMR", "n_questions"}


# ─── Worked paper example ─────────────────────────────────────────────────────


def test_paper_example_overconfidence_definition():
    """From paper §1: a system that returns wrong answer 'ICLR 2025' with confidence 94/100.
    Single question, incorrect, conf=0.94. R_O = 1 - 0.94 = 0.06. R_U = 1 (no correct). HMR = 0 because R_U is 1 but R_O dominates: 2*0.06*1/(0.06+1) ≈ 0.113.
    """
    rs = [make("q0", correct=False, conf=0.94)]
    assert math.isclose(reward_overconfidence(rs), 0.06)
    assert reward_underconfidence(rs) == 1.0
    expected_hmr = 2 * 0.06 * 1.0 / (0.06 + 1.0)
    assert math.isclose(hmr(rs), expected_hmr)
