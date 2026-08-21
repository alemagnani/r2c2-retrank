# Experiment: How far does calibration carry *without* a large LLM?

## Question
Can a pipeline built only from **small pre-trained encoder models** (cross-encoders,
extractive QA, NLI) — **no generative LLM** — reproduce the calibration advantage that won
R2C2 AC? If HMR stays high while accuracy drops, it proves the **calibration architecture
is the model-agnostic lever** (our central thesis) and quantifies exactly how much the LLM
contributed.

## Design principle
Keep the *architecture* identical to the winning LLM pipeline (answer-first → verify →
calibrate → abstain); swap every generative-LLM component for a small encoder model. The
evaluation still uses the Claude self-evaluator (`ac_eval.py`, validated to official ±0.0004)
so numbers are comparable to the LLM pipeline and the field.

## Models (all encoder-only, no generative LLM)
| Role | Model | Note |
|---|---|---|
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` | already used in PR |
| Extractive answer | `deepset/roberta-base-squad2` | SQuAD2 → span **+ no-answer score** (natural abstention signal) |
| Nugget verification / NLI | `cross-encoder/nli-deberta-v3-base` | passage ⊨ nugget? → citation filter + MNP |
| Calibration | sklearn `IsotonicRegression` / Platt | fit on val250 gold |

"Small" = encoder-only, ≤ ~300M params, no autoregressive generation. (Optional stretch
variant: a small seq2seq like Flan-T5-base as the answer generator — borderline "LLM"; run
only if extractive accuracy is too low, and label it separately.)

## Pipeline stages (mirrors the LLM pipeline)
- **S0. Pool** = BITEM-PG-1 + BITEM-PG-2 (the winning pool). Isolate the *pipeline*, same
  passages as the winning LLM run.
- **S1. Retrieve/rerank**: ms-marco CE reranks pool passages per question → top-k (k≈10).
- **S2. Extractive answer** (answer-first): roberta-squad2 over top-k → best span, span
  score, **no-answer prob**; track source passage (→ citation).
- **S3. Answer-conditional evidence**: re-score passages with CE on `question + answer` to
  find the best supporting passage (mirrors answer-conditional pool refinement).
- **S4. Nugget** = answer-bearing sentence from the supporting passage; cite (PRrun, rank).
- **S5. Verify / citation-filter**: NLI(passage → nugget); keep only entailed nuggets
  (drives MNP).
- **S6. Confidence features**: [span score, 1−no_answer, top CE score, NLI entailment,
  top1−top2 span margin, #entailed nuggets].
- **S7. Calibrate**: fit calibrator (features → P(correct)) on **val250** (label = extracted
  answer matches `answer_hint`); confidence = calibrated P(correct) × 100.
- **S8. Abstain**: if calibrated confidence < τ → "I don't know" / very low confidence. τ
  tuned on val250 to maximise HMR (exactly as the LLM pipeline tuned thresholds).

## Calibration & tuning (all on val250, official 65 held out)
Run S1–S6 on the 248 val250 topics, compute features + correctness vs gold `answer_hint`,
fit the calibrator, and grid-search τ for max proxy-HMR. This mirrors the LLM pipeline's
val250 tuning and keeps the 65 official topics held out.

## Evaluation
Run on the 65 official topics → emit a valid AC run file → score with `ac_eval.py`
(Claude self-evaluator). Report Accuracy, MNP, R_O, R_U, HMR. Compare to:
- LLM pipeline retrank-AC-2 (HMR 0.963), and
- the field (best non-retrank 0.761; median much lower).

## Ablations (isolate what carries)
- **A. Full no-LLM pipeline.**
- **B. − calibration** (raw scores as confidence) → value of the calibration map.
- **C. − abstention** → value of selective refusal.
- **D. − NLI citation filter** → MNP contribution.
- **E. Retrieval-only floor**: answer = top passage's salient sentence, conf = calibrated CE.

## Hypotheses (pre-registered)
- **Accuracy drops a lot** (extractive QA fails multi-hop/implicit/negation): ~0.5–0.7 vs 0.95.
- **MNP holds up moderately** (NLI-gated precision): ~0.5–0.7.
- **HMR is the headline**: if calibration carries, HMR ~0.7–0.85 — which would **still beat
  most of the field and likely place top-3**. The key comparison is *fractional* drop:
  if %ΔAccuracy ≫ %ΔHMR, the calibration thesis is proven.
- **Ablation B collapses HMR** toward the overconfident/sandbagger failure modes → shows the
  calibration map is the load-bearing part.

## Risks
- **Over-abstention → sandbagger mode** (tanks R_U): mitigate by tuning τ for HMR (not
  precision) and using isotonic (sharp) calibration.
- Extractive answers may be marked wrong by the LLM judge on formatting even when span-correct.
- Genuinely reasoning-heavy questions are where the LLM's real value-add will show — that
  gap is itself a finding.

## Deliverable for the paper
A row: "no-LLM pipeline" with (Acc, MNP, HMR) next to the LLM pipeline and the field, plus
the ablation table. Narrative: *the calibration layer transfers to encoder-only models;
the LLM buys answer accuracy, not calibration — and under HMR, calibration is what wins.*

---

## RESULTS (run 2026-08)

Committee ladder, answer-everything operating point (scored by the validated
Sonnet self-evaluator). Calibrator fit on val250; official 65 topics held out.

| Committee | Accuracy | MNP | HMR (answer-all) | margin AUC (official) |
|---|---|---|---|---|
| 1 (roberta) | 0.415 | 0.415 | 0.629 | 0.849 |
| 2 (+electra) | 0.462 | 0.462 | **0.677** | ~0.85 |
| 3 (+minilm) | **0.523** | 0.492 | 0.659 | 0.766 |
| 5 (+bert,+tinyroberta) | 0.523 | 0.492 | 0.644 | 0.736 |
| **LLM pipeline (AC-2)** | 0.953 | 0.835 | 0.963 | (self-eval 0.0004 to official) |

Key findings:
- **Accuracy scales with voting then plateaus at ~0.52** (extractive ceiling; 4th/5th answerer add nothing — correlated errors).
- **Calibration is cheap**: one encoder's answerability margin → correctness AUC 0.85 on official.
- **Accuracy↑ / calibration↓ tradeoff**: voted-answer margin AUC falls 0.85→0.73 as committee grows; HMR peaks at committee-2.
- **Agreement feature** ~redundant with margin at 2 voters (AUC +0.003); becomes competitive at ≥3 voters but combined still < single-model margin.
- **Field placement (answer-all)**: HMR 0.66 → ~11th of 25 official runs, beats 3 LLM teams (Error404, BITEM, hit-u); beats BITEM (0.49 HMR) despite BITEM's 0.92 accuracy — calibration wins.
- **Abstention-gamed**: HMR 0.90 @ accuracy 0.22 (rank 3) — illustrates HMR's modesty-reward, not a real placement.
- **Decomposition**: encoder committee recovers calibration (AUC 0.85) + half the accuracy (0.52); the 0.52→0.95 gap is generative reasoning = the LLM's legitimate, irreplaceable contribution (not judge-gaming).

Artifacts: `data/eval/no_llm_{official,val250}_c{2,3,5}.json` (features),
`data/runs/retrank-AC-noLLM-c{2,3,5}.txt` (runs),
`data/eval/no_llm_c{2,3,5}_metrics.json` (scores).
Scripts: `no_llm_predict.py` (`--qa-models` committee), `no_llm_calibrate.py`.

---

## CONFIDENCE / ACCURACY TECHNIQUE SWEEP (2026-08)

Motivated by SmoothHess (approximate input-Hessian for 2nd-order feature
interactions, Stein's-lemma gradient estimate) and the Laplace/Bayesian-uncertainty
literature. Question: which non-LLM techniques recover the confidence/accuracy an LLM gives?

| # | Technique | Result (official, n=65) | Verdict |
|---|---|---|---|
| 1 | **Interaction calibrator** (2nd-order poly on `[margin,agree,ce]`) | discrimination AUC 0.766 → **0.803** | ✅ helps — *parsimony-critical* (all-feature interactions overfit val250 → AUC collapses to 0.60) |
| 2 | **Ensemble score-variance** (dispersion of members' margins) | AUC **0.506** | ❌ uninformative (members' margins on different scales) |
| 4 | **MC-dropout variance** (K=10, roberta-squad2) | AUC **0.534** | ❌ uninformative |
| 3 | **Last-layer Laplace** (parameter-Hessian → predictive variance) | not run | ⏭ predicted ❌ — same posterior-variance quantity MC-dropout approximates |
| 5 | **Conformal / coverage-controlled abstention** | target-acc knob; guarantee loose (0.90→0.71) under synthetic→real shift | ✅ correct way to set operating point (vs HMR-maxing τ, which games abstention) |
| 6 | **Small generative reader** (flan-t5-base, 250M) | Acc **0.477** < extractive committee 0.523 | ❌ does not break the ceiling — small generation ≈ extraction |

### Synthesis
- **Confidence is cheap and near-solved by the answerability *margin* (aleatoric).**
  Feature *interactions* sharpen it modestly (SmoothHess intuition, +0.04 AUC);
  discrete *agreement* helps (0.76). **Variance-based / Bayesian uncertainty
  (ensemble dispersion, MC-dropout, and by extension Laplace) adds nothing** — the
  useful signal is the point estimate, not its spread.
- **Abstention should be set by conformal coverage control, not HMR-maximisation**
  (the latter collapses to answering ~15% at ~0.14 accuracy — HMR's modesty exploit).
- **Accuracy is the real ceiling (~0.52).** It is *not* broken by adding a small
  generative model — flan-t5-base is *worse* than the extractive committee. The
  0.52→0.95 gap therefore requires **large-scale reasoning**, not the generative
  modality per se. That is the LLM's specific, irreplaceable contribution.

Artifacts: `data/eval/no_llm_official_c3v.json` (dispersion feats),
`data/runs/retrank-AC-noLLM-flant5.txt`, `data/eval/no_llm_flant5_metrics.json`.

---

## STATISTICAL RIGOR (Sakai's methods, per organizer guidance)

Applied the organizers' own evaluation statistics to our internal comparisons.

**Randomised Tukey HSD** (Sakai 2018), paired, B=5000, per-topic accuracy:
| Comparison | ΔAcc | p |
|---|---|---|
| Committee-3 vs single (voting) | +0.108 | **0.141** (n.s.) |
| Committee-5 vs Committee-3 | 0.000 | 1.000 |
| Committee-3 vs flan-t5 | +0.046 | 0.648 |

**Discordance:** committee vs single → fixes 12 wrong, breaks 5 correct (net +7/65).
Bidirectional effect ⇒ gain not significant.

**Power analysis** (Sakai 2018, *Sample Sizes / Effect Sizes / Statistical Power*):
min detectable ΔAcc at n=65, 80% power ≈ **0.15**. Our +0.108 is *below* it ⇒ the
committee ladder is **underpowered, not null**. Only the 0.52→0.95 committee-vs-LLM
gap far exceeds the threshold.

**Paper implication:** report the committee ladder as a *trend*; frame all n=65
comparisons with effect sizes + significance (Sakai's Statistical Reform). Cite
`sakai-sigtest`, `sakai-reform`, `modesty`, `brevrag`.

**Still to try (organizer-guided):** per-topic failure analysis by challenge type
(multi-hop / negation / implicit-entity) — the organizers explicitly requested it and
it turns "0.52 ceiling" into *which* question types need reasoning.

### Per-topic failure analysis (organizer-requested; rule-based typing, no LLM)
| Question type | n | committee-3 acc | LLM acc |
|---|---|---|---|
| factoid | 27 | 0.59 | 1.00 |
| person | 19 | 0.58 | 0.95 |
| count ("how many") | 14 | **0.29** | 0.86 |
| title | 5 | 0.60 | 0.80 |

LLM leads on **every** type; largest gap on **count/aggregation** (+0.57). Localises
the reasoning gap: extraction handles direct factoid/entity lookup but not aggregation.

---

## CROSS-FAMILY VALIDATION (addressing Codex review)

### Held-out self-eval prediction (non-circular)
Our self-evaluator predicts 5 held-out competitor runs' official HMR: Spearman rho=0.60,
MAE 0.12. 3/5 within 0.01-0.13; big miss on slr2c2 (well-calibrated, our simplified judge
under-scored). Genuine but moderate cross-family validity.

### Calibration-transfer (the controlled test)
Hold each competitor's ANSWERS fixed; replace only their confidence with our verifier-capped
Stage-5 layer; re-score under our self-evaluator:
| Team | Acc | their conf HMR | + our layer | dHMR |
|---|---|---|---|---|
| slr2c2-AC-4 | 0.68 | 0.243 | 0.619 | +0.376 |
| Error404-AC-3 | 0.71 | 0.553 | 0.697 | +0.144 |
| BITEM-AC-2 | 0.92 | 0.599 | 0.725 | +0.125 |
| WaterlooClarke-AC-2 | 0.83 | 0.820 | 0.827 | +0.007 |
| hit-u-AC-1 (sandbag) | 0.81 | 0.197 | 0.179 | -0.018 |

**4/5 improved, mean +0.127.** Gains largest on overconfident teams (via R_O); ~0 for the
already-calibrated one; small LOSS for the sandbagger (our layer only suppresses
overconfidence — the R_O-bottleneck asymmetry). Controlled cross-system evidence that the
calibration layer transfers; caveat = scored under our self-evaluator, not official judge.
Scripts: `calib_transfer.py`, `fast_selfeval.py`. Artifacts: `data/eval/transfer_*.json`.

---

## NON-CLAUDE BASE-MODEL SWAP (GPT), run via Codex 2026-08

Full pipeline (answer+nuggets+verifier+Stage-5) run on 3 OpenAI models, 65 topics,
BITEM pool, fixed GPT-5.5 judge. Calibration ON vs OFF (paired, judge cancels).

| Model | Acc | HMR off | HMR on | ΔHMR | R_O off | R_O on | ΔR_O |
|---|---|---|---|---|---|---|---|
| gpt-5.6-sol (flagship) | 0.939 | 0.444 | 0.599 | +0.156 | 0.285 | 0.430 | +0.145 |
| gpt-5.6-luna (efficient) | 0.954 | 0.577 | 0.717 | +0.140 | 0.407 | 0.567 | +0.160 |
| gpt-4.1-mini | 0.923 | 0.399 | 0.899 | +0.500 | 0.250 | 0.830 | +0.580 |

**3/3 positive. Mean ΔHMR +0.265, mean ΔR_O +0.295.** Same signature as Claude (gain via
R_O = verifier caps confidence on wrong answers). CONFIRMS the calibration layer transfers
across model families (Claude + GPT) within our pipeline. Caveat: scored by a GPT judge
(not official Qwen+GPT), so absolute HMR is context; the claim rests on the ON-vs-OFF delta.
Absolute: only gpt-4.1-mini ON (0.899) exceeds field 0.761; none reaches Claude 0.963.
Artifacts: data/eval/nonclaude/{SUMMARY.json, result_*.json, FINAL_REPORT.md}.
