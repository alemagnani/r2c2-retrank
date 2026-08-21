# Official-Results Audit — rebasing paper_draft.tex onto NTCIR-19 R2C2 official scores

**Why this exists.** Every number in `paper_draft.tex` is self-evaluated under our
Sonnet-4.6 judge (Opus-4.7 cross-val), explicitly hedged as *"absent official
organiser scores."* The official scores + draft overview landed **2026-07-28**
(Sijie Tao email). This file is the checklist to swap self-eval → official and to
turn the paper's hedge into a validated claim (or an honest self-eval-vs-official gap,
which is itself a finding).

## Blocked on (download these first)
- [ ] **Official results** → `https://waseda.app.box.com/v/r2c2officialresults`
      (login-gated) → save under `data/eval/official/`.
- [ ] **Overview draft PDF** (`NTCIR_19_R2C2_Task_Overview_draft_0729.pdf`,
      corrected version, Jul 28 9:10pm email) → save under `docs/`.
- [ ] **Gold answers** → `https://waseda.box.com/r2c2topics-and-goldanswers`
      → `data/eval/official/gold/` (for our own post-hoc per-topic analysis).

## Structural facts to extract from the official package (fill in)
| Item | Where | Value |
|---|---|---|
| Our 4 AC runs' official HMR (retrank-AC-1..4) | results | ? |
| Our 4 AC runs' Accuracy | results | ? |
| Our 4 AC runs' Mean Nugget Precision | results | ? |
| Official R_O / R_U split per run (if reported) | results | ? |
| Our best AC run's **rank** among all teams | overview tables | ? |
| Significance-test outcomes involving our runs | overview | ? |
| Our PR runs re-scored via AC labels (nDCG@20) | overview | ? |
| Which config each AC-1..4 maps to | RESOLVED (below) | ✓ |

> **Run mapping (resolved from `docs/results/Final-Submission.md`):**
> - **AC-1** = `ensemble_top1` — saturation recalibration; self HMR 0.974/0.974, Acc 0.938
> - **AC-2** = `bitem_only_refined_sonnet` — **MAIN SYSTEM** (paper's headline); self HMR 0.963/0.963, Acc 0.938
> - **AC-3** = `bitem_only_refined_sonnet_recovered` — Acc hedge; self HMR 0.922/0.951, **Acc 0.954** (highest)
> - **AC-4** = `tier1plus2_refined_sonnet` (6-team pool) — judge-divergent hedge; self HMR 0.874/**0.937**, Acc 0.877
>
> Watch: AC-3/AC-4 have large Sonnet→Opus swings. If the **official** judge
> behaves like Opus, AC-3 (Acc-heavy) or AC-4 (6-team pool) could out-rank AC-2 —
> which would be a headline-changing result for the paper.

## Claim-by-claim audit (self-eval value → official value → verdict)
Verdict = SURVIVES / WEAKENS / CONTRADICTED once official filled in.

### Headline numbers (abstract)
| Claim | Self-eval | Official | Verdict |
|---|---|---|---|
| Main system HMR | 0.963 | ? | ? |
| Main system Accuracy | 94% | ? | ? |
| Main system refusal rate | 6% | ? | ? |
| Main system confident-wrong count | 0 | ? | ? |
| Saturation variant HMR | 0.974 | ? | ? |
| Always-refuse@0.05 baseline HMR | 0.974 | ? | ? |

### Robust findings (CI excludes zero in self-eval — do they hold officially?)
| Finding | Self-eval ΔHMR [CI] | Official | Verdict |
|---|---|---|---|
| Stage-1 LLM tier matters most | −0.127 [−0.296,−0.034] | ? | ? |
| Stage-4 LLM tier matters least | −0.050 [−0.118,−0.004] | ? | ? |
| All-Haiku regression | −0.170 [−0.334,−0.080] | ? | ? |
| Opus-pipeline regression | −0.176 [−0.372,−0.075] | ? | ? |
| Top-1 "ensemble" uplift (= saturation) | +0.012 [+0.008,+0.016] | ? | ? |
| R_O-bottleneck mechanism (Cnf-W ~ −1 corr w/ HMR) | qualitative | ? | ? |

### Suggestive findings (CI crosses zero — official may resolve direction)
| Finding | Self-eval ΔHMR [CI] | Official | Verdict |
|---|---|---|---|
| Pool quality vs Tier-1 | −0.111 [−0.458,+0.044] | ? | ? |
| Sonnet refinement vs broad | +0.085 [−0.045,+0.439] | ? | ? |
| Top-3 ensembling | +0.007 [−0.002,+0.014] | ? | ? |
| retrank+E404 refinement-failure case study | judge-flip | ? | ? |

### Preliminary
| Finding | Self-eval | Official | Verdict |
|---|---|---|---|
| Nugget-first vs answer-first (20 topics) | −0.29 HMR, all R_O | ? | ? |

## The self-eval-vs-official GAP is itself a paper result
Sijie's guidance: *"what's more important is discussing WHY something works."*
The delta between our Sonnet-judge HMR and the official HMR is a first-class finding:
- If official ≈ self-eval → our self-evaluator is well-calibrated to the official
  methodology (validates the whole `ac_eval.py` approach; strong methods contribution).
- If official < self-eval → our judge was optimistic; quantify the gap and where it
  comes from (R_O? nugget precision?). This directly feeds §judge_xval.
- Per-topic: use gold answers to run the failure analysis Sijie explicitly asked for
  (mean scores don't tell the whole story).

## Downstream rewrites once filled
1. Abstract: replace the "absent official organiser scores" hedge (lines ~75–78) with
   the official headline; keep self-eval as the *cross-validation* story, not the
   primary evidence.
2. §Robust-vs-suggestive: add an "official" column; promote/demote findings by whether
   they survived.
3. Add cross-team relative-standing subsection (rank + significance tests) — currently
   absent because we had no competitor scores.
4. Title: organiser convention is `retrank at the NTCIR-19 R2C2 Task`; consider
   `retrank at the NTCIR-19 R2C2 Task: Optimising Under HMR` to keep the thesis +
   be findable in the proceedings.
