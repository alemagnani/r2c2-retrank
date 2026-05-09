[← Back to results index](README.md)

---

# Nugget-First vs Answer-First — direct head-to-head

## Setup

For each of the first 20 R2C2 topics, on the BITEM-only pool:

- **Nugget-first:** Sonnet reads top-5 passages, emits 3–6 atomic factual nuggets *without* knowing the answer. A second Sonnet call synthesises an answer using only those nuggets.
- **Answer-first (production):** the standard 7-stage pipeline.

Both use the same passages and the same model. We score nugget-first
correctness via token-overlap match against the answer-first answer
(itself self-eval-correct on 19 of 20 topics).

## Result

| Pipeline | Acc | R_O | R_U | HMR |
|---|---:|---:|---:|---:|
| **Answer-first (BITEM-Sonnet)** | 0.950 | 0.950 | 1.000 | **0.963** |
| Nugget-first (this experiment) | 0.900 | 0.525 | 0.921 | 0.669 |
| Δ (NF − AF) | **−0.05** | **−0.425** | −0.08 | **−0.294** |

**Accuracy gap: small.** Both pipelines pick essentially the right
answer on 18–19 of 20 topics.

**HMR gap: huge, almost entirely from R_O.**

## Mechanism: the two disagreement topics

| Topic | NF answer | AF answer | What happened |
|---|---|---|---|
| 0019 ("how many films did George Lazenby play James Bond in?") | "5" | "8" | NF confidently emits "5" from incomplete pool evidence; AF correctly counts to 8 from the answer-first prompt's wider context |
| 0020 (Spencer Bell's Mickey Rooney movies) | "At least 3 (...)" | refused | NF generalises confidently from a partial cast list; AF detects insufficient evidence and refuses |

Both NF errors are **confident wrong answers.** AF avoids them by:
1. Knowing what specifically to look for (the question), and
2. Recognising when the pool does not contain the answer (refusal).

NF, extracting nuggets in isolation, has no notion of "what's missing"
— it just builds an answer from the evidence it has, even when that
evidence is incomplete.

## Why this matters

This is **the [R_O bottleneck thesis](The-RO-Bottleneck.md) in miniature.**
The design choice that suppresses confident-wrong topics wins HMR,
even when accuracy is nearly identical.

## Caveats

- Only 20 topics, not the full 65
- Correctness scoring uses answer-first as ground truth (so the −0.05 accuracy gap is biased toward AF)
- Token-overlap match has its own noise; a Stage-B-like LLM judge would be cleaner
- Single nugget-first prompt template; alternatives exist

The 65-topic full comparison with a cleaner judge is the natural
follow-up; we flag it as future work in the paper.

## Per-topic results table (extract)

| qid | NF answer | AF answer | match |
|---|---|---|---|
| 0001 | Bor Gullet, a Mairan | Bor Gullet (a Mairan creature) | ✓ |
| 0002 | Black Widow | Black Widow | ✓ |
| 0003 | "A robot may not harm humanity..." | (same) | ✓ |
| 0004 | Roger Moore | Roger Moore | ✓ |
| 0005 | 1 | 1 | ✓ |
| ... | ... | ... | ✓ |
| 0010 | "The provided nuggets do not explicitly state..." | Jeff Goldblum | ✓ (token overlap on Goldblum) |
| 0018 | 6 | 6 (Memento, Insomnia, ...) | ✓ |
| 0019 | **5** | **8** | ✗ |
| 0020 | At least 3 (...) | (refused) | ✗ |

## Reproduce

```bash
python scripts/nugget_first_mini.py
```

Output: `data/eval/ac_runs/nugget_first_mini.json`.
