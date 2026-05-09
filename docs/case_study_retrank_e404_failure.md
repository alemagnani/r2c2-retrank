# Case Study: When Sonnet Refinement Misfires
## The retrank+Error404 + Sonnet Refinement Failure (HMR 0.910 → 0.738)

This document analyses the only configuration in our experimental matrix where
Stage 1.5 Sonnet refinement *hurt* end-to-end HMR. It complements
Section §RQ6 / §10 of the paper draft.

## Headline numbers

| Configuration | Acc | MNP | R\_O | R\_U | HMR |
|---|---|---|---|---|---|
| retrank+Error404, broad pool        | 0.923 | 0.622 | 0.870 | 0.954 | 0.910 |
| retrank+Error404, Sonnet refinement | 0.877 | 0.599 | 0.595 | 0.971 | **0.738** |
| Δ                                    | −0.046 | −0.023 | **−0.275** | +0.017 | **−0.172** |

The drop is dominated by R\_O collapsing from 0.870 to 0.595 — i.e.\ the
Sonnet-refined system gives high confidence to wrong answers far more often.

## Mechanism: only 3 topics regressed

Per-topic comparison:
- **3 topics regressed** (broad correct → sonnet incorrect)
- **0 topics improved**
- **57 unchanged correct**, 5 unchanged wrong

So the entire 0.172 HMR drop is concentrated in 3 topics. Let's look at each.

### Topic 0008 — "Name the actor who played Green Lantern in the DCEU and then played Deadpool in the MCU."

Oracle answer: *Ryan Reynolds.*

| | broad | sonnet |
|---|---|---|
| Final answer | Ryan Reynolds | Ryan Reynolds |
| Confidence (final) | 0.99 | 0.99 |
| Stage 4 match\_score | 2 | 2 |
| Stage 3 nuggets kept | 5 | 4 |
| Stage B nuggets relevant | 4 / 5 | **0 / 4** |
| Stage B verdict | correct | **incorrect** |

Same answer, same confidence, same verifier match — but Stage B (the
self-evaluator) marked the sonnet variant incorrect.

**Why?** The broad variant's nuggets included:

1. *"Ryan Reynolds was cast as Hal Jordan / Green Lantern in the Green Lantern film."*
2. *"Ryan Reynolds portrays Wade Wilson / Deadpool in the Marvel Cinematic Universe."*
3. *"Ryan Reynolds took the Green Lantern role after 20th Century Fox had no intention…"*
4. *"Ryan Reynolds portrayed Hal Jordan in the 2011 film Green Lantern."*
5. ***"The MCU film Deadpool & Wolverine integrated the X-Men series' iteration of Wade Wilson…"***

Sonnet's variant has nuggets 1–4 (lightly reworded) but **drops nugget 5**
in favour of:

4'. *"Ryan Reynolds reprised his role as Wade Wilson / Deadpool from Fox's X-Men films in the MCU."*

Both answers ultimately tie Ryan Reynolds to "Deadpool in the MCU". But the
broad variant's nugget 5 makes the bridge *explicit* via the title "Deadpool
& Wolverine" which is a known MCU film. The sonnet variant's substitute
nugget is more abstract ("reprised … from Fox's X-Men films in the MCU"),
which leaves Stage B uncertain whether the question's "Deadpool in the MCU"
is actually established.

**This is a multi-hop argument fragility issue.** The question requires
chaining (Reynolds → Green Lantern in DCEU) ∧ (Reynolds → Deadpool in MCU).
The broad variant has both legs grounded with concrete film titles; the
sonnet variant has one leg expressed via a more inferential nugget. Stage B
prefers explicit grounding.

### Topic 0010 — "Who plays the Wizard of Oz in Wicked the movie?"

Oracle answer: *Jeff Goldblum.*

Both pipelines: final answer "Jeff Goldblum", conf 0.90, match\_score 2.

The 2 nuggets each:

| | broad | sonnet |
|---|---|---|
| Nugget 1 | "Jeff Goldblum appears in a supporting role in the 2024 film Wicked." | (identical) |
| Nugget 2 | "The 2024 Wicked film stars Cynthia Erivo as Elphaba and Ariana Grande as Glinda, with **Jeff Goldblum as the Wizard**…" | "The 2024 film Wicked is set in the Land of Oz and adapts the stage musical which is based…" |

Both nugget-2 candidates were entailed (passed Stage A). But sonnet's
re-ranking elevated a *plot-context* passage over the *cast-listing*
passage. The cast listing nugget *names the role* ("Jeff Goldblum as the
Wizard"); the plot context nugget *doesn't*. Stage B says "the entailed
nuggets establish Goldblum is in the film, but don't confirm he plays the
Wizard role specifically — INCORRECT."

**This is a refinement-prefers-elaboration issue.** Sonnet's pointwise scorer
likes passages that elaborate on the candidate answer's *context* (the film,
its setting), whereas the canonical answer-grounding passage is often a
boring fact-listing (cast tables, infoboxes). The boring passage scored
lower; the elaborate passage scored higher; the boring one was the one we
needed.

### Topic 0016 — "Which movie was released earlier, Avengers: Infinity War or 1917?"

Oracle answer: *Avengers: Infinity War.*

Total pipeline collapse:

| | broad | sonnet |
|---|---|---|
| Stage 1 candidate | "Avengers: Infinity War (April 27, 2018)…" | (same — Stage 1 used broad pool here too) |
| Stage 1.5 refined pool | n/a | re-ranked |
| Stage 2 nuggets returned | 3 | 0 |
| Stage 3 kept / dropped | 3 / 0 | 0 / 0 |
| Stage 4 verifier answer | (uses entailed nuggets) | "" (no nuggets to use) |
| Final answer | "Avengers: Infinity War…" | "" (refused) |

In the sonnet variant, Stage 2 *returned no nuggets*. The Sonnet refinement
pushed out passages that would have been used to extract date-comparison
nuggets, replacing them with passages that mention the films but not their
release dates in the format Stage 2 needed.

**This is a refinement-narrows-too-much issue.** For comparison questions,
the answer-grounding passages are typically dry release-date info. Sonnet
refinement's relevance scoring de-prioritised these in favour of more
"interesting" semantic relationships, leaving Stage 2 unable to extract the
specific date-comparison nuggets it needed.

## Three failure modes, one root cause

All three regressions stem from the same root: **Sonnet refinement on the
retrank+Error404 pool over-scores passages that semantically relate to the
candidate answer at the expense of passages that ground the answer in
specific facts (dates, titles, role names)**.

The retrank+Error404 pool is where this matters most because:
- retrank's 200-character passages are too short to support most direct
  factual claims (we already knew this from §11).
- Error404 has both PG and PO runs with passages of varying lengths and
  styles, including some highly elaborative passages that Sonnet's refiner
  finds more "supportive" than the dry fact-listing passages.

When the candidate answer is correct *and* the question is a multi-hop /
comparison / role-attribution type, the refinement's elaboration bias drops
the very nuggets the answer-evaluator needs.

## Why doesn't this happen on BITEM-only?

BITEM has uniform passage segmentation and length. Sonnet refinement's
re-ranking among BITEM passages doesn't trade dry-but-grounding passages
for elaborative ones — they're all moderately structured. The +0.026 HMR
gain on BITEM-only tracks the modest improvement Sonnet refinement *can*
provide when there's no bias trap.

## Implications for paper §6 (RQ6)

We should:
1. Add this case study (paper appendix or sidebar) to illustrate
   **when refinement helps vs hurts**.
2. Refine RQ6's claim: *"Sonnet refinement helps when pool homogeneity is
   high; it can hurt when pool diversity introduces semantic-vs-grounding
   tension."*
3. Suggest a fix for future work: a hybrid scorer that weights
   "grounding-fact density" alongside semantic support — e.g.\ bonus
   for passages containing the answer's named entities verbatim
   (basically lexical bonus, which our (c) algorithm already uses
   modestly).

## Lesson for the paper's discussion

The R\_O collapse on this configuration is *not* because the answers got
worse — they're the same answers (Ryan Reynolds, Jeff Goldblum) — but
because the supporting nuggets got subtly less explicit, and the strict
Stage B judge holds the system to a higher grounding standard than the
intermediate Stage 4 verifier. The gap between Stage 4 ("does the answer
follow from these nuggets?") and Stage B ("did the nuggets help DERIVE
the answer?") is the mechanism.

This is a cautionary tale about **judge-stage consistency** in calibrated
RAG pipelines: when intermediate verifiers and final evaluators ask
slightly different questions, refinement steps that satisfy one can
violate the other.
