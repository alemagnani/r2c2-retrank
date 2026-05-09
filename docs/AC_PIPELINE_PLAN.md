# AC Pipeline — Detailed Plan

> **Status:** drafted 2026-05-03, revised 2026-05-03 (added Stage 1.5, restructured factorial). No code yet. Approval gate before implementation.
> **Companion docs:** `README.md` Part II (§13–18); `~/.claude/plans/squishy-sleeping-bee.md` (evaluator plan, already implemented).

## 0. One-paragraph summary

We will build a 7-stage pipeline that, for each of the 65 R2C2 questions, produces an answer + confidence + grounded nuggets. The pipeline is **answer-first with verification**: it proposes a candidate answer from a *broad pool* of long, diverse passages (drawn from BITEM + Error404 + WaterlooClarke — never our own short retrank passages), then **refines that pool conditionally on the candidate answer** (Stage 1.5), then extracts nuggets that ground the answer, verifies the nuggets actually justify the answer, and emits a confidence score derived from the verification signal plus retrieval strength. Each stage has its own intermediate evaluation and an iteration loop. We submit **four AC runs in a 2×2 factorial design (Stage 1.5 on/off × calibration strategy)**, plus offline ablations covering all three Stage 1.5 algorithms (CE, Sonnet, lexical) and all six research questions: calibration matters most, diverse pooling beats volume, answer-first beats nugget-first, refusal beats guessing, verification gap is a calibration signal, and **answer-conditional passage selection improves nugget quality**.

---

## 1. Research questions and falsification criteria

Each RQ has (a) a hypothesis we will test, (b) the data point that would confirm it, and (c) the data point that would falsify it.

### RQ1 — Calibration is the dominant lever for HMR

- **Hypothesis:** holding everything else fixed, HMR varies more across calibration strategies than across pool choices.
- **Confirms:** within our 4 submitted runs, the HMR difference between calibration variants > HMR difference between pool variants.
- **Falsifies:** pool choice produces ≥ as much HMR variation as calibration.
- **Why interesting:** if true, the headline R2C2 metric is mostly about how a system *talks about its own uncertainty*, which is the paper's stated thesis (modesty).

### RQ2 — Architectural diversity in pooling beats volume

- **Hypothesis:** pooling 2–3 architecturally diverse teams (low mutual Jaccard) yields higher accuracy and HMR than pooling 5–6 similar teams.
- **Confirms:** diverse-pool variant beats large-similar-pool variant on accuracy, even when total passage count is matched.
- **Falsifies:** more passages always helps regardless of where they come from.
- **Why interesting:** unique to us — we have the only PG/sliding-window submission and have already shown (§14 README) that retrank's pool has 53% docs no other team finds.

### RQ3 — Answer-first + verification beats nugget-first

- **Hypothesis:** an "answer-first then targeted nugget extraction with verification" pipeline produces lower bogus rate, higher Mean Nugget Precision, and higher HMR than a "nugget-first then synthesis" pipeline.
- **Confirms:** offline ablation comparing both pipelines on the same 65 topics + same pool shows answer-first wins on at least 2 of the 3 metrics.
- **Falsifies:** the two pipelines tie or nugget-first wins.
- **Why interesting:** addresses the paper's own §2.1 concern about "post-rationalisation" — this is methodological contribution.

### RQ4 — Refusal beats guessing on coverage gaps

- **Hypothesis:** for questions where retrieval is weak (no high-CE passages found), explicitly outputting "no answer" with very low confidence beats producing a best-guess at moderate confidence.
- **Confirms:** restricted to weak-retrieval topics, refusal-aware variant has higher HMR than guess-anyway variant.
- **Falsifies:** refusal hurts overall HMR or has no effect.
- **Why interesting:** R2C2's nDCG@20 penalty for our PG submission (26% zero-recall on real topics) highlights coverage as a real problem.

### RQ5 — Verification gap correlates with correctness

- **Hypothesis:** the "verification distance" between the candidate answer and the answer the verifier would derive from the entailed nuggets correlates with answer correctness.
- **Confirms:** binning questions by verification gap, accuracy decreases monotonically with gap.
- **Falsifies:** no correlation, or non-monotonic.
- **Why interesting:** if true, gives us a calibration signal that's more grounded than LLM self-report.

### RQ6 — Answer-conditional passage selection improves nugget quality

- **Hypothesis (RQ6a):** after generating a candidate answer, re-ranking the passage pool conditional on that answer (Stage 1.5) yields lower bogus rate, higher MNP, and higher accuracy than using the answer-blind broad pool directly.
- **Confirms:** AC-2 > AC-1 on MNP and accuracy; AC-4 > AC-3 on the same. Effect should be measurable across calibration variants.
- **Falsifies:** Stage 1.5 has no effect or hurts.
- **Why interesting:** simplest possible feedback loop in a RAG pipeline; cheap if it works.
- **Hypothesis (RQ6b):** among Stage 1.5 algorithms, CE on `query | answer` (variant a) ≈ Sonnet pointwise (variant b) > lexical (variant c). I.e., LLM-based and CE-based give similar quality; the cheap lexical heuristic falls behind.
- **Tested by:** OA-5b vs AC-2 (Sonnet vs CE refinement); OA-5c vs AC-2 (lexical vs CE).
- **Why interesting:** if (a) ≈ (b), CE is the practical choice (free, ≈$0); if (b) > (a), the cost of LLM refinement is justified.

---

## 2. Pipeline architecture (answer-first + verification + answer-conditional refinement)

```
                ┌──────────────────────────────┐
                │ 0. Initial broad pool         │   per topic: top-50 by CE
   PR runs ────▶│   Pick long-passage,          │   from BITEM/Error404/WaterlooClarke
                │   diverse-team passages       │   (no answer used yet)
                └──────────────┬───────────────┘
                               │ broad pool (50)
                ┌──────────────▼───────────────┐
                │ 1. Candidate answer          │   Sonnet — holistic read
                │   "From these passages, what │   produces: answer_string
                │    is the answer?"            │   plus self-rated confidence_a
                └──────────────┬───────────────┘
                               │ candidate_answer + confidence_a
                ┌──────────────▼───────────────┐
                │ 1.5 Answer-conditional        │   re-rank broad pool
                │   pool refinement (NEW)       │   using algo (a)/(b)/(c)
                │                                │   take top-30
                │   (a) CE on (query|answer)    │
                │   (b) Sonnet pointwise score  │   * one of submitted runs uses NONE
                │   (c) lexical answer-tokens   │     (control: pool unchanged)
                └──────────────┬───────────────┘
                               │ refined pool (30)
                ┌──────────────▼───────────────┐
                │ 2. Targeted nugget            │   Sonnet — for each nugget,
                │   extraction                  │   pick THE ONE passage that
                │   "Find atomic claims that    │   most directly entails it
                │    support the answer"        │
                └──────────────┬───────────────┘
                               │
                ┌──────────────▼───────────────┐
                │ 3. Self-Stage A check         │   Sonnet — emulates
                │   For each (passage,          │   organizers' bogus check.
                │   nugget): is it entailed?    │   Drop bogus nuggets.
                └──────────────┬───────────────┘
                               │ entailed nuggets
                ┌──────────────▼───────────────┐
                │ 4. Verification               │   Sonnet — given Q + entailed
                │   "Given these nuggets only,  │   nuggets, derive an answer
                │    what is the answer?"       │   independently.
                └──────────────┬───────────────┘
                               │ verifier_answer + match? + verifier_confidence
                ┌──────────────▼───────────────┐
                │ 5. Confidence & refusal       │   pure math + tiny LLM check
                │   Combine signals; if too     │   conf = f(self, verify, gap, retrieval)
                │   weak, override to refusal   │
                └──────────────┬───────────────┘
                               │
                ┌──────────────▼───────────────┐
                │ 6. Format & write             │   ac_format.write_ac_run
                └───────────────────────────────┘
```

### Why answer-first + verification + Stage 1.5

- **Answer-first** focuses nugget extraction on evidence that supports a specific claim (precision ↑).
- **Stage 1.5 (answer-conditional refinement)** uses the candidate answer to re-rank passages so that nugget extraction sees passages most likely to entail the answer. Without it, Stage 2 would still see passages selected purely on query→passage CE relevance, ignoring the answer hypothesis. This is the cheapest possible feedback loop in a RAG pipeline (RQ6).
- **Self-Stage A check** filters bogus nuggets before submission, mirroring exactly what the official judge does.
- **Verification** gives us a *direct* calibration signal: if the verifier (using only the entailed nuggets, not the passages) derives the same answer, we have grounded confidence; if not, we should hedge. This addresses RQ5.

---

## 3. Stage-by-stage detail with hypotheses and intermediate evaluations

For each stage we specify: (a) what the stage does, (b) what we hypothesise, (c) how we evaluate its quality independently of downstream stages, and (d) the iteration mechanism if the eval reveals a problem.

### Stage 0 — Pool selection

**What it does:** for each topic, gather candidate passages from selected PR runs. Default pool is **Tier 1 = BITEM + Error404 + WaterlooClarke** (long passages, low mutual overlap, no retrank). Take top-K from each team; dedup by (doc_id, passage_text_hash); cap final list at 30.

**Hypotheses tested:**
- H0.1 Tier-1 pool (3 diverse PG-style teams) covers the answer-bearing doc on more topics than retrank-only would have. (Already shown via competitor-analysis: §14 README.)
- H0.2 Adding Tier-2 teams (ORG/hit-u/WasedaR2C2 — overlapping cluster) to the pool gives diminishing returns: < 5% accuracy improvement over Tier-1 alone.

**Intermediate evaluation:**
- **E0.1 Coverage check (free):** for each topic, count unique docs in pool. Median should be ≥ 50; min should be ≥ 15.
- **E0.2 CE-quality check (free, already cached):** % of topics where pool contains ≥ 1 passage with CE > 3 (using `data/eval/all_team_ce_scores.pkl`). Target ≥ 80%.
- **E0.3 Hand-spot 5 topics:** for 5 randomly chosen topics, manually inspect 3 passages each. Subjective grade for "could a human answer the question from these?" Target 4/5.

**Iteration mechanism:**
- If E0.1 fails: increase top-K from each team or add Tier-2 teams.
- If E0.2 fails: investigate which topics have weak pools — are they actually unanswerable from the corpus, or do we need different teams? Possibly drop those topics from optimistic estimates.
- If E0.3 fails: rethink pool composition.

**Cost:** zero LLM cost — entirely based on already-computed data.

---

### Stage 1 — Candidate answer generation

**What it does:** Sonnet reads the pool of (up to 30) passages and produces:
- `candidate_answer` (string)
- `confidence_a` (LLM self-report, 0–100)
- `reasoning` (chain of thought, internal use only — not submitted)

**Hypotheses tested:**
- H1.1 Sonnet with all passages in context produces correct candidate answers on ≥ 70% of topics (upper bound on our final accuracy; if only 70% are right at this stage, we can't exceed 70% overall).
- H1.2 The LLM's self-reported `confidence_a` is *not* a reliable calibration signal alone — RQ1's premise.

**Intermediate evaluation:**
- **E1.1 Hand-graded 10-topic spot check:** human (us) checks 10 candidate answers against known facts or quick web search. Target ≥ 7/10 correct.
- **E1.2 Self-evaluator stub:** since we don't yet have nuggets, we can't run the full evaluator. But we can run **a degenerate eval** that takes (question, candidate_answer) and asks Sonnet "is this answer correct given your knowledge?" — and use that as a *cheap proxy* for accuracy on all 65 topics. This is *not* the official metric (it lets the LLM use parametric knowledge), but as a dev signal it's useful.
- **E1.3 Pairwise consistency:** run Stage 1 *twice* with the same pool and temperature 0.0 — they should give identical answers. Then once with temperature 0.7 — % agreement gives a free-ensemble lower bound for confidence.

**Iteration mechanism:**
- If E1.1 < 5/10 correct: prompt is broken, rewrite.
- If E1.2 estimates < 50% accuracy: pool is bad → revisit Stage 0.
- If E1.3 shows < 50% agreement across temperature: questions are ambiguous *or* pool is too weak.

**Cost:** ~$0.50 per pass over 65 topics; ~$2 if we ablate prompts.

**Prompt skeleton:**
```
You are answering a movie-related question using only the passages provided.

Question: {question}

Passages:
[1] (from {team_1}, rank {rank_1}): {text_1}
[2] (from {team_2}, rank {rank_2}): {text_2}
...

Tasks:
1. Determine the answer. Be concise — a name, phrase, sentence, or short
   quotation, matching the granularity the question asks for.
2. If the passages do not contain the answer, output:
     {"answer": "", "confidence": 0, "reason": "passages do not support an answer"}
   Do NOT use external knowledge to fill gaps.
3. Otherwise, rate your confidence 0–100 based ONLY on how strongly the
   passages support the answer (not your prior knowledge).

Reply JSON only:
{"answer": "...", "confidence": <int>, "reason": "<one sentence>"}
```

---

### Stage 1.5 — Answer-conditional pool refinement

**What it does:** re-rank the Stage 0 broad pool of 50 passages using the candidate answer from Stage 1 as additional context. Take the top-30 most likely to entail the answer. If Stage 1 produced a refusal (empty answer), skip this stage and pass the broad pool through unchanged.

Three algorithm variants:

- **(a) CE on `query | answer`** — re-score with cross-encoder on `"<question> | <candidate_answer>"` vs each passage. Free if we cache; ~5 min for all topics on first run.
- **(b) Sonnet pointwise** — for each passage, ask Sonnet "rate 0–3 how strongly this passage supports the candidate answer". ~$3 per pass.
- **(c) Lexical answer-presence** — combine original CE with bonus for passages containing answer's distinctive tokens: `score = ce_orig * (1 + α · token_overlap)`. Free.

**Hypotheses tested (RQ6):**
- H1.5-1 Refined pool produces measurably better downstream metrics (lower bogus rate, higher MNP, higher accuracy) than the broad pool.
- H1.5-2 Among (a)/(b)/(c), CE ≈ Sonnet > lexical.

**Intermediate evaluation:**
- **E1.5-1 Pool-overlap:** for each topic, |refined ∩ broad| / |refined|. Low overlap means refinement is doing real work; high overlap means the answer doesn't change passage ranking much (probably an "easy" topic).
- **E1.5-2 Top-3 lift:** check whether the top-3 refined passages contain the answer-bearing doc more often than top-3 broad. Spot-check 10 topics manually.
- **E1.5-3 End-to-end signal:** measure downstream bogus rate and MNP with refined vs unrefined pool, on the same Stage 2 prompt. Used to confirm/reject H1.5-1.

**Iteration mechanism:**
- If refinement *hurts* metrics: probably the candidate answer is poor and pollutes the refinement signal. Add a "trust gate": only refine when Stage 1 confidence ≥ 50.
- If lexical (c) ≈ CE (a): the topic distribution favours exact-match cases; report and move on.

**Cost:** (a) ~$0; (b) ~$3 per pass; (c) $0.

**Prompt skeleton for variant (b) — Sonnet pointwise:**
```
Question: {question}
Candidate answer: {candidate_answer}

Passage: {passage_text}

Rate 0–3 how strongly the passage SUPPORTS the candidate answer:
  3 = directly states the answer
  2 = strongly implies the answer (multiple supporting facts)
  1 = mentions the answer entity but does not directly support it
  0 = unrelated or contradictory

Reply JSON only: {"score": <0|1|2|3>, "reason": "<one short phrase>"}
```

---

### Stage 2 — Targeted nugget extraction

**What it does:** for each topic, given (question, candidate_answer, pool of passages), Sonnet emits 3–8 atomic nuggets. Each nugget includes the cited PassageKey = (PR_run_name, passage_rank). The prompt explicitly asks for nuggets that **directly support the candidate answer**, not background facts.

**Hypotheses tested:**
- H2.1 Targeted extraction (with answer in context) produces lower bogus rate than untargeted extraction (without answer). Target: < 15% bogus, vs. probably ~30%+ untargeted.
- H2.2 Each nugget can confidently cite exactly one passage that entails it. We do *not* allow multi-passage nuggets.

**Intermediate evaluation:**
- **E2.1 Self-Stage A check (Sonnet):** for each nugget produced, run Stage A bogus check. Bogus rate = % nuggets not entailed. Target ≤ 15%. Already implemented in `scripts/ac_eval.py::stage_a_bogus_check`.
- **E2.2 Hand spot-check 10 nuggets:** subjective grade "is this nugget atomic, supported, and relevant?"
- **E2.3 Nugget-count distribution:** histogram. We expect 3–8 with mode around 4. Many 0-nugget topics → answer was probably wrong / refusal. Many 10+ → prompt is too liberal.

**Iteration mechanism:**
- If E2.1 > 20%: prompt too aggressive. Strengthen "stay close to passage text" instruction; reduce max nuggets.
- If E2.3 has long tail: cap max nuggets in prompt at 8.
- If many nuggets fail Stage A because the cited passage doesn't quite entail: retry with the same nugget but cite a different passage from the pool.

**Cost:** ~$1 per pass over 65 topics for extraction + ~$1.50 for self-Stage A check.

**Prompt skeleton:**
```
You are extracting nuggets from passages to support a candidate answer.

Question: {question}

Candidate answer: {candidate_answer}

Passages (numbered, with citation keys):
[(P1) team={team} rank={rank}]: {text}
[(P2) team={team} rank={rank}]: {text}
...

A nugget is:
- A single atomic factual claim
- Directly relevant to deriving the candidate answer
- ENTAILED by exactly one of the passages above (not your knowledge)
- Stated concisely; you may paraphrase but must not add facts beyond the passage

Extract 3–8 nuggets. For each, cite the SINGLE passage that most directly
supports it. Do NOT include nuggets the passages don't entail — even if you
"know" the answer is correct.

Reply JSON only:
{
  "nuggets": [
    {"text": "...", "cite": "P1"},
    {"text": "...", "cite": "P2"}
  ]
}
```

---

### Stage 3 — Self-Stage A bogus filter

**What it does:** runs `scripts/ac_eval.py::stage_a_bogus_check` on every (passage, nugget) pair we produced. Drops bogus nuggets before submission. **Already implemented and tested.**

**Hypotheses tested:**
- H3.1 Most bogus nuggets are caught here, not by the official judge. Submitted bogus rate ≤ official bogus rate.

**Intermediate evaluation:**
- **E3.1 Bogus rate per topic:** report distribution. Target median ≤ 1 bogus nugget per topic.
- **E3.2 Re-extraction:** for nuggets that failed Stage A, retry Stage 2 with feedback ("this nugget was rejected, try again with a different passage or wording"). Iterate at most twice.

**Cost:** included in Stage 2 (already counted).

---

### Stage 4 — Verification (the calibration source)

**What it does:** given (question, list of *entailed* nuggets — no passages), Sonnet derives an answer using ONLY the nuggets. We then compare:
- `verifier_answer` vs `candidate_answer`
- Match → verification passes; high signal of correctness
- Mismatch → verification fails; low signal of correctness

**Hypotheses tested:**
- H4.1 (answers RQ5) Verification match correlates with actual correctness — high enough to be a useful calibration signal (Spearman ρ > 0.4 between verification-match and Stage-B correctness on a held-out subset).
- H4.2 Verification-match cases correctly identify a high-precision subset: precision on verification-match ≥ 90%; recall on verification-match ≥ 70%.

**Intermediate evaluation:**
- **E4.1 Confusion matrix:** on hand-graded 20 topics, fill in:

  |  | candidate correct | candidate wrong |
  |---|---|---|
  | verifier matches | TP | FP |
  | verifier diverges | FN | TN |

  We want TP/(TP+FP) ≥ 0.9 and TN/(TN+FN) > 0.5.

- **E4.2 Calibration plot:** bin topics by verification outcome (match / partial match / no match). Plot empirical accuracy per bin. We want accuracy(match) >> accuracy(no match).

**Iteration mechanism:**
- If H4.1 fails: nuggets aren't summarising the passages well; re-tune Stage 2 prompt.
- If H4.2 fails on precision: verifier is too lenient; tighten matching criterion (exact-match for short answers, semantic-match for longer ones).

**Cost:** ~$1 per pass; one extra LLM call per topic.

**Prompt skeleton:**
```
You are answering a movie-related question using ONLY the listed nuggets
(treat each as a factually correct claim — do not use external knowledge).

Question: {question}

Nuggets:
[1] {nugget_1_text}
[2] {nugget_2_text}
...

Tasks:
1. Derive the answer using only the nuggets above. If they are insufficient,
   output empty answer.
2. Rate confidence 0–100 based on how directly the nuggets imply the answer.

Reply JSON only:
{"verifier_answer": "...", "verifier_confidence": <int>}
```

After this stage we compute:
- `match`: true iff `candidate_answer ≈ verifier_answer` (string-similarity threshold, normalise case/whitespace, allow synonym tolerance via Sonnet for ambiguous cases)
- `verifier_match_score`: 0 (no match), 1 (partial match), 2 (full match)

---

### Stage 5 — Confidence & refusal logic

**What it does:** combines four signals into a final confidence score, with an explicit refusal path.

Signals available:
- `c_self`: Stage 1 candidate confidence (0–100)
- `c_verify`: Stage 4 verifier confidence (0–100)
- `match_score`: 0 / 1 / 2 from Stage 4
- `n_entailed`: number of entailed nuggets (after Stage 3)
- `max_ce_score`: highest CE score among the cited passages (free, from cached scores)

Calibration formulas (each AC variant uses one):

#### Variant A — Refusal-aware self-report (used in AC-1, AC-3)
```
if n_entailed == 0 OR candidate_answer == "":
    final_answer = ""
    confidence = 5      # not 0 — to avoid divide-by-zero edge cases in HMR
elif match_score == 0:           # verifier disagrees → very low conf
    confidence = min(c_self, 25)
elif match_score == 1:
    confidence = min(c_self, 60)
else:                            # match_score == 2 → full match
    confidence = c_self
```

#### Variant B — Ensemble agreement (used in AC-2, AC-4)
```
Run Stage 1 + Stage 2 + Stage 4 K=5 times with sampling temperature 0.7.
agreement = fraction of runs producing the same final answer.
confidence = round(100 * agreement)
if agreement < 0.4:               # no clear majority → refuse
    final_answer = ""
    confidence = 5
```

**Hypotheses tested:**
- H5.1 (RQ1, RQ4) Variant A and Variant B both produce higher HMR than naive LLM self-report would.
- H5.2 (RQ4) Refusal path improves HMR by ≥ 0.05 over a no-refusal version of each variant.

**Intermediate evaluation:**
- **E5.1 Calibration curve on hand-graded 20-topic subset:** plot predicted confidence vs empirical accuracy in 5 bins. Visually inspect for monotonicity.
- **E5.2 Run the full self-evaluator** (`scripts/ac_eval.py`) on each variant; compare HMR.
- **E5.3 Sweep refusal thresholds**: try `match_score < 0`, `match_score < 1`, `n_entailed < 1`, `n_entailed < 2`. Pick the threshold that maximises HMR on val250 (synthetic) — note: caveat that real topics may differ.

**Iteration mechanism:**
- If E5.1 shows non-monotonic confidence: confidence formula is broken, re-derive.
- If E5.2 shows low HMR even with calibration: investigate whether failures are mostly R_O (overconfident) or R_U (underconfident); rebalance.

**Cost:** Variant A ~$0; Variant B ~$5 per run (5× sampling). Across 4 runs, total ensemble cost ~$10.

---

### Stage 6 — Format and submit

**What it does:** writes XML using `scripts/ac_format.py::write_ac_run`. Validates with `validate(records)`. Bundles 4 files into `retrank-AC.zip`.

**Intermediate evaluation:**
- **E6.1 Round-trip:** parse the file we just wrote and confirm it parses cleanly; nugget counts, confidences, citations match input.
- **E6.2 Spot-check 5 records:** human reads them, confirms they look like proper AC submissions (answer makes sense, nuggets are atomic and on-topic, citations point to real PR run files).
- **E6.3 Self-evaluator one more time:** sanity check final HMR on the formatted file.

---

## 4. Experimental design — 4 submitted runs + offline ablations

Pool source held constant: **Tier-1 = BITEM + Error404 + WaterlooClarke**, top-50 broad pool from Stage 0. The 2×2 varies (a) whether Stage 1.5 is applied and (b) which calibration variant is used.

### The 2×2 factorial (4 submitted AC runs)

| Run | Stage 1.5 (factor 1) | Calibration (factor 2) | Tests |
|---|---|---|---|
| **retrank-AC-1** | None — broad pool used directly | Variant A (refusal-aware) | answer-blind baseline + refusal |
| **retrank-AC-2** | (a) CE on `query | answer` | Variant A | answer-conditioned + refusal |
| **retrank-AC-3** | None — broad pool used directly | Variant B (ensemble) | answer-blind + ensemble |
| **retrank-AC-4** | (a) CE on `query | answer` | Variant B | both factors active |

This isolates:
- **Stage 1.5 effect (RQ6)** = AC-2 − AC-1, AC-4 − AC-3 — does answer-conditional refinement help?
- **Calibration effect (RQ1, partial RQ4)** = AC-3 − AC-1, AC-4 − AC-2
- **Interaction** = (AC-4 − AC-3) − (AC-2 − AC-1)

### Offline ablations (not submitted, evaluated with our self-evaluator)

| Ablation | Variant | What it tests |
|---|---|---|
| **OA-1 — Nugget-first pipeline** | AC-2 config but extract nuggets first, synthesise answer second | RQ3 (answer-first beats nugget-first) |
| **OA-2 — No refusal** | AC-2 config, but always emit a best-guess answer with naive self-report confidence | RQ4 (refusal beats guessing) |
| **OA-3 — Tier-2 supplement** | AC-2 config but pool = Tier-1 + Tier-2 (ORG, hit-u, WasedaR2C2) | H0.2 (diminishing returns of more teams) |
| **OA-4 — Verification-only confidence** | AC-2 config, but conf derived purely from verification match-score | RQ5 (verification gap as signal) |
| **OA-5b — Stage 1.5 = Sonnet pointwise** | AC-2 config, Stage 1.5 = (b) | RQ6b (LLM vs CE refinement) |
| **OA-5c — Stage 1.5 = lexical** | AC-2 config, Stage 1.5 = (c) | RQ6b (cheap heuristic vs semantic) |
| **OA-6 — Smaller pool (BITEM only)** | AC-2 config but pool = BITEM only | pool-source ablation (was originally factor) |
| **OA-7 — retrank-only** | AC-2 config but pool = retrank only | what AC would look like with our PG submission alone (paper completeness) |
| **OA-8 — retrank + Error404** | AC-2 config but pool = retrank + Error404 | does adding retrank to a strong partner widen recall? |

Each ablation costs ~$5–8. Total offline cost ~$45.

### Aggregate analysis

For **RQ1**: `Var(HMR | calibration) > Var(HMR | Stage 1.5)` across the 4 runs.

For **RQ2**: AC-3 vs OA-3 (Tier-1 vs Tier-1+2) and OA-6 vs AC-2 (BITEM vs Tier-1). Already strong evidence from competitor analysis (§14 README).

For **RQ3**: `bogus_rate(AC-2) < bogus_rate(OA-1)` AND `MNP(AC-2) > MNP(OA-1)`.

For **RQ4**: `HMR(AC-2) > HMR(OA-2)`, especially on weak-retrieval subset.

For **RQ5**: scatter plot of `verification_match_score` (0/1/2) vs accuracy across all 4 runs × 65 topics; report Spearman ρ with bootstrap CI.

For **RQ6**: `MNP(AC-2) > MNP(AC-1)` AND `MNP(AC-4) > MNP(AC-3)` AND lower bogus rates. Plus `MNP(AC-2) ≈ MNP(OA-5b) > MNP(OA-5c)` to compare algorithms.

---

## 5. Implementation order, effort, and gates

| Step | Hours | Gate to pass before next step |
|---|---|---|
| 1. Pool selector (Stage 0 broad) | 2 | E0.1 + E0.2 + E0.3 pass |
| 2. Stage 1 prompt + tests on 10 topics | 3 | E1.1 ≥ 7/10 |
| 3. Stage 1.5 algorithm (a) CE refinement | 2 | E1.5-1 + E1.5-2 spot-checks pass |
| 4. Stage 2 + Stage 3 prompts + tests | 4 | E2.1 bogus rate ≤ 20% |
| 5. Stage 4 verifier + match logic | 3 | E4.1 confusion matrix passable |
| 6. Stage 5 confidence Variant A | 1 | E5.1 monotonic curve |
| 7. Stage 5 Variant B (ensemble) | 3 | E5.1 + E5.2 |
| 8. End-to-end one run (AC-2) on real 65 | 3 | E5.2 HMR > 0.30 (rough threshold) |
| 9. Other 3 submitted runs (AC-1/3/4) | 5 | E5.2 reasonable |
| 10. Stage 1.5 algorithms (b) + (c) for OA-5 | 3 | analysis-ready |
| 11. Other ablations (OA-1/2/3/4/6) | 7 | analysis ready |
| 12. Final submission packaging | 1 | E6.1–E6.3 all pass |
| 13. Paper-ready analysis (plots, tables) | 4 | clean numbers |

**Total: ~41 hours.** Schedule: target 6–8 working days, completion by **May 13**, 2 days of slack before May 15 deadline.

---

## 6. Costs

| Component | Cost |
|---|---|
| Stage 1 candidate answer (Sonnet, 65 topics × 4 runs + 7 ablations) | $5 |
| Stage 1.5 (a) CE refinement (free; just CE inference) | $0 |
| Stage 1.5 (b) Sonnet pointwise (one ablation) | $3 |
| Stage 1.5 (c) lexical | $0 |
| Stage 2 nugget extraction | $10 |
| Stage 3 self-Stage A bogus check | $14 |
| Stage 4 verification | $5 |
| Stage 5 Variant B ensembles (5×) | $20 |
| Self-evaluator re-runs on final outputs | $20 |
| Misc dev iteration & prompt tuning | $20 |
| **Total** | **~$100** |

---

## 7. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Pool too sparse → many refusals → low accuracy | Medium | High | E0.2 catches early; expand to Tier 2 if needed |
| Sonnet hallucinates nuggets → high bogus rate | Medium | High | Stage 3 self-check + retry loop + strict prompt |
| Verification too lenient (always matches) | Medium | Medium | Tighten match criterion in Stage 4; manual review on 20 topics |
| Ensemble too expensive | Low | Medium | Cap K=5; use cached candidate answers across stages |
| Confidence calibration plot non-monotonic | Medium | Medium | Iteration in Stage 5; switch to verification-only confidence (OA-4) |
| Submission format error breaks at organizer end | Low | Catastrophic | E6.1 + E6.2 + 2 days of slack before deadline |
| HMR very low for all 4 runs (< 0.2) | Low | High | Indicates fundamental calibration issue; pivot to OA-4 (pure verification) |

---

## 8. What this plan does NOT cover

- **Failure analysis & autoresearch loop**: if HMR is bad, we've talked about a diagnose-and-fix loop similar to PR's autoresearch. Out of scope for now; defer until first results in.
- **Post-Aug 1 retro**: when official results are released, we'll re-run our self-evaluator with the same Sonnet prompts and compare to organizer scores. Publish correlation.
- **Re-submission for PR with longer passages**: we discussed this for hypothetical NTCIR-20; out of scope for May 15.

---

## 9. Open questions to resolve before coding

These are concrete decisions we need to make in the next 24 hours:

1. **Match criterion for Stage 4** (verifier_answer == candidate_answer)
   - Option a: exact string match after lowercasing and stripping
   - Option b: semantic match via Sonnet ("is X the same answer as Y?")
   - Option c: hybrid — exact match first, fallback to Sonnet for near-misses
   - **Recommendation:** option c.

2. **Refusal threshold for Variant A**
   - Refuse when `n_entailed == 0`? Probably yes.
   - Refuse when `match_score == 0`? Probably yes — but verify on subset first.
   - **Recommendation:** start with both, ablate.

3. **Pool size cap per topic**
   - 30 passages × ~2,000 chars each ≈ 60,000 chars context for Stage 1. Sonnet handles 200K, fine.
   - **Recommendation:** 30 hard cap; reduce only if cost is a concern.

4. **Whether to run the full self-evaluator on synthetic val250 first** as a dry-run before real 65
   - Probably yes — cheaper and shakes out bugs. ~$15.
   - **Recommendation:** yes, after Step 6 and before Step 7.

---

## 10. Where progress lives

- **Decisions** — appended to `README.md` §17 (chronological log)
- **Results** — appended to `README.md` §18 (chronological tables)
- **Detailed plan** — this file (`docs/AC_PIPELINE_PLAN.md`)
- **Evaluator code** — `scripts/ac_eval.py`, `src/eval/hmr.py`, `tests/`
- **Pipeline code** (to be built) — `scripts/ac_pool_select.py` (Stage 0), `scripts/ac_pool_refine.py` (Stage 1.5), `scripts/ac_pipeline.py` (Stages 1–6 orchestrator), `scripts/ac_run.py` (CLI driver)
- **Per-run outputs** — `data/runs/retrank-AC-[1..4].txt`
- **Per-run evals** — `data/eval/ac/retrank-AC-[1..4].json`
- **Cache** — `data/eval/ac_cache/` (Stage A bogus, Stage B answer, Stage 4 verification)
