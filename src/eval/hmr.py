"""
HMR (Harmonic Mean of Rewards) and friends, per Sakai et al., NTCIR-19 R2C2.

Reference: BREV-RAG 2025 workshop paper, Section 2.4.
  R_O = 1 - O / |I^-|     where O = sum of confidences on incorrect answers
  R_U = 1 - U / |I^+|     where U = sum of (1 - confidence) on correct answers
  HMR = 2 * R_O * R_U / (R_O + R_U)

Plus: Accuracy and Mean Nugget Precision, the other two AC metrics.

All inputs are plain Python types — no numpy/torch deps so this module stays
trivially testable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionResult:
    """One row in the AC evaluation table."""

    question_id: str
    correct: bool                       # was the answer judged correct?
    confidence: float                   # in [0, 1] — caller normalizes 0-100 → 0-1
    nuggets_returned: int               # total nuggets the system submitted (incl. bogus)
    nuggets_relevant: int               # nuggets judged relevant by the answer-eval LLM


def accuracy(results: list[QuestionResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.correct) / len(results)


def mean_nugget_precision(results: list[QuestionResult]) -> float:
    """Mean over questions of (relevant / total_returned).

    Per the paper: total_returned includes bogus nuggets — bogus ones are
    counted in the denominator but never the numerator, so a system that
    floods nuggets with hallucinations is penalised.

    Convention: questions where the system returned 0 nuggets contribute 0.
    """
    if not results:
        return 0.0
    per_q = []
    for r in results:
        if r.nuggets_returned <= 0:
            per_q.append(0.0)
        else:
            per_q.append(r.nuggets_relevant / r.nuggets_returned)
    return sum(per_q) / len(per_q)


def reward_overconfidence(results: list[QuestionResult]) -> float:
    """R_O — penalises high confidence on incorrect answers.

    R_O = 1 if there are no incorrect answers (system is perfect, can't be
    overconfident); else 1 - mean_confidence_on_incorrect.
    """
    incorrect = [r for r in results if not r.correct]
    if not incorrect:
        return 1.0
    O = sum(r.confidence for r in incorrect)
    return 1.0 - O / len(incorrect)


def reward_underconfidence(results: list[QuestionResult]) -> float:
    """R_U — penalises low confidence on correct answers.

    R_U = 1 if there are no correct answers (system is fully wrong, can't be
    underconfident); else 1 - mean_(1 - confidence)_on_correct.
    """
    correct = [r for r in results if r.correct]
    if not correct:
        return 1.0
    U = sum(1.0 - r.confidence for r in correct)
    return 1.0 - U / len(correct)


def hmr(results: list[QuestionResult]) -> float:
    """Harmonic Mean of R_O and R_U.

    HMR = 0 if both rewards are 0 (degenerate). Otherwise the harmonic mean.
    """
    R_O = reward_overconfidence(results)
    R_U = reward_underconfidence(results)
    if R_O + R_U == 0:
        return 0.0
    return 2.0 * R_O * R_U / (R_O + R_U)


@dataclass(frozen=True)
class ACMetrics:
    """The three official AC metrics plus the two HMR components."""

    accuracy: float
    mean_nugget_precision: float
    R_O: float
    R_U: float
    HMR: float
    n_questions: int

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "mean_nugget_precision": self.mean_nugget_precision,
            "R_O": self.R_O,
            "R_U": self.R_U,
            "HMR": self.HMR,
            "n_questions": self.n_questions,
        }


def compute_metrics(results: list[QuestionResult]) -> ACMetrics:
    """Compute all five quantities at once."""
    return ACMetrics(
        accuracy=accuracy(results),
        mean_nugget_precision=mean_nugget_precision(results),
        R_O=reward_overconfidence(results),
        R_U=reward_underconfidence(results),
        HMR=hmr(results),
        n_questions=len(results),
    )
