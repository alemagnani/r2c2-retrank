[← Back to results index](README.md)

---

# The R_O Bottleneck

The single most important observation in this work.

## Quantitative claim

Across all 16 base configurations we evaluated:

| Quantity | Range | Notes |
|---|---|---|
| `R_U` (correct-answer reward) | [0.79, 0.98] | narrow band |
| `R_O` (humility reward) | [0.45, 0.95] | **2× wider variance** |
| Accuracy | [0.86, 0.94] | narrow band |
| Cnf-W (confident-wrong topics) | [0, 4] | discrete, dominates HMR |

**Every winning design choice we identify in this paper primarily moves R_O.**
Pool curation, answer-conditional refinement, verifier-based abstention,
and confidence saturation all act on the confident-wrong cell, not on
the correct-answered cell. This is not a coincidence; it falls out of
the harmonic-mean structure of HMR at our high-accuracy operating point.

## A single confident-wrong topic costs ≈ 0.25 R_O

At |I⁻| ≈ 4 (typical for our top systems), each confident-wrong topic
(confidence ≥ 0.9, incorrect) contributes roughly `0.9 / 4 = 0.225`
to the overconfidence sum. Adding *one* such topic to a pipeline
that previously had zero takes R_O from 0.95 to 0.72 → HMR drops by
~0.10. There is no R_U gain that recovers this within accuracy noise.

## Per-topic outcome map

The figure below shows seven candidates × 65 topics. **Red cells (confident-wrong) drive HMR.**
Cnf-W count and final HMR are shown to the right of each row.

![R_O bottleneck](../figures/fig_ro_bottleneck.pdf)

> *Embedded PDF figure — view directly in [`docs/figures/fig_ro_bottleneck.pdf`](../figures/fig_ro_bottleneck.pdf) if not rendering.*

### A surprising secondary finding

The confident-wrong topics are **not the same** across candidates. Each
pipeline has its *own* blind spots. The implication: R_O generalises
poorly across pools — it's not that there are 4 universally hard
topics, it's that each pipeline is overconfident about a different 4.

## Implication for system design

The R_O bottleneck reframes RAG-with-confidence engineering:

1. **Don't optimise answer coverage.** At ≥ 90% accuracy, an extra
   recovered correct answer adds ~ 0.01 to R_U; an extra confident-wrong
   answer subtracts ~ 0.10 from R_O.
2. **Verifier-based abstention is first-class system component**,
   not a post-hoc fix.
3. **Stage-1 refusals on JSON parse failures are *not bugs*.** They are
   correctly-detected unanswerable topics — see [our negative results §](../paper_draft.pdf) for the prompt-hardening attempt that *worsened* HMR by 0.04.

## Three corollaries we test in the paper

- **Pool quality > pool count.** A high-quality single pool has fewer
  Cnf-W; a low-quality single pool has *more* than a multi-team pool.
  See [Pool Choice](Pool-Choice.md).
- **Sonnet at Stage 1 is non-negotiable.** Haiku at Stage 1 produces
  more Cnf-W, not just slightly different answers. See [Cost vs Quality](Cost-vs-Quality.md).
- **Nugget-first vs answer-first.** Direct head-to-head: similar
  accuracy, much worse R_O. See [Nugget-First vs Answer-First](Nugget-First-vs-Answer-First.md).
