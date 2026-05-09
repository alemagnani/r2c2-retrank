# retrank @ NTCIR-19 R2C2 — Results Wiki

Navigable results from `retrank`'s submission to the NTCIR-19 R2C2
**Answering with Confidence (AC)** subtask.

> **Central thesis.** In confidence-scored RAG at high accuracy, the
> dominant failure mode is *not* missing answers but emitting a small
> number of confident wrong ones. AC pipelines should therefore optimise
> $R_O$ via grounding and abstention, not maximise answer coverage.

[**📄 Read the full paper (PDF)**](../paper_draft.pdf)
&nbsp;·&nbsp;
[**📦 Submitted runs (`retrank-AC.zip`)**](../../data/runs/retrank-AC.zip)
&nbsp;·&nbsp;
[**🧪 Source code**](../../)

---

## 🧭 Pages

### Headline
- [**Headline Results**](Headline-Results.md) — the central diagnostic table (Acc, R_O, HMR under both judges)
- [**The R_O Bottleneck**](The-RO-Bottleneck.md) — why a few confident-wrong topics dominate HMR
- [**Final Submission**](Final-Submission.md) — the four AC runs we packaged and their rationale

### Engineering tradeoffs
- [**Pool Choice**](Pool-Choice.md) — why pool *quality* matters more than pool *count*
- [**Cost vs Quality**](Cost-vs-Quality.md) — per-stage Sonnet/Haiku ablation + measured hybrid
- [**Cross-Pool Aggregation**](Cross-Pool-Aggregation.md) — top-K voter ensembling and what "top-1 ensemble" really is
- [**Calibration & Abstention**](Calibration-and-Abstention.md) — verifier match-score and refusal rules

### Methodology
- [**Pipeline Overview**](Pipeline-Overview.md) — the 7-stage answer-first design
- [**Judge Cross-Validation**](Judge-Cross-Validation.md) — Sonnet vs Opus verdicts
- [**Statistical Robustness**](Statistical-Robustness.md) — bootstrap, McNemar, trivial baselines
- [**Nugget-First vs Answer-First**](Nugget-First-vs-Answer-First.md) — direct head-to-head on 20 topics

### Reference
- [**Per-Candidate Leaderboard**](Per-Candidate-Leaderboard.md) — full table of all 33 candidate runs
- [**Reproducing the Pipeline**](Reproducing-the-Pipeline.md) — how to re-run end-to-end

---

## 📊 The one-paragraph summary

We submitted four AC runs to the NTCIR-19 R2C2 task. Our main system,
**`bitem_only_refined_sonnet`**, reaches HMR **0.963** under both
Sonnet and Opus self-evaluators with **94% accuracy**, **6% refusal**,
and **zero confident-wrong topics**. A post-hoc confidence-saturation
recalibration of the same pipeline (`ensemble_top1`) reaches HMR
**0.974** — but matches the trivial always-refuse-at-conf-0.05
baseline on HMR alone, which is why we report HMR jointly with
accuracy and refusal rate throughout. The strongest engineering
takeaways are: (i) a high-quality single-team passage pool can
outperform naive multi-team pooling, but this is about pool *quality*
not *count*; (ii) Sonnet-class reasoning is most critical at the
candidate-answer stage (Δ HMR −0.13 vs Haiku) and least at the
verifier (−0.05); (iii) the cost-quality recipe `S/H/S/H` does not
hold (measured −0.20, much worse than additive); (iv) on a 20-topic
direct head-to-head, nugget-first matches answer-first on accuracy
(0.90 vs 0.95) but loses 0.29 HMR purely on R_O.

---

## 📅 Timeline

| Date | Milestone |
|---|---|
| 2026-04-24 | PR (Passage Retrieval) submission |
| 2026-05-15 | **AC submission deadline** (4 runs packaged, lint-clean) |
| 2026-08-01 | Official evaluation results expected |
| 2026-09-01 | Paper drafts due |
| 2026-12-08–10 | NTCIR-19 conference (Tokyo) |
