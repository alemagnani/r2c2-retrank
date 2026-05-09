[← Back to results index](README.md)

---

# Per-Candidate Leaderboard — full table

All 33 candidate AC configurations, ranked by HMR under the Sonnet self-evaluator.

| Configuration | Pool | Refinement | Cal | Acc | R_O | HMR (S) | HMR (O) |
|---|---|---|:-:|---:|---:|---:|---:|
| ensemble_top1 | multi | per-pipeline | A | 0.938 | 0.950 | **0.974** | 0.974 |
| ensemble_top3 | multi | per-pipeline | A | 0.938 | 0.950 | **0.969** | 0.969 |
| **bitem_only_refined_sonnet (main)** | BITEM | Sonnet | A | 0.938 | 0.950 | **0.963** | 0.963 |
| bitem_only_refined_sonnet_recovered | BITEM | Sonnet (recov.) | A | 0.922 | 0.866 | 0.951 | 0.951 |
| bitem_only_broad | BITEM | none | A | 0.923 | 0.910 | 0.937 | 0.860 |
| ensemble_top12 | multi | per-pipeline | A | 0.923 | 0.844 | 0.918 | — |
| tier1_refined_sonnet | Tier-1 | Sonnet | A | 0.923 | 0.870 | 0.911 | 0.818 |
| retrank+Error404_broad | re+E404 | none | A | 0.923 | 0.870 | 0.910 | — |
| s4-Haiku | BITEM | Sonnet | A | 0.923 | 0.870 | 0.913 | — |
| s2-Haiku | BITEM | Sonnet | A | 0.908 | 0.850 | 0.899 | — |
| tier1+2_refined_sonnet | Tier-1+2 | Sonnet | A | 0.877 | 0.794 | 0.874 | — |
| WaterlooClarke-only | Waterloo | Sonnet | A | 0.862 | 0.822 | 0.870 | — |
| retrank-only_broad | retrank | none | A | 0.508 | 0.929 | 0.858 | — |
| ensemble_all-16 | multi | per-pipeline | A | 0.923 | 0.810 | 0.857 | — |
| s1-Haiku | BITEM | Sonnet | A | 0.892 | 0.750 | 0.836 | — |
| tier1+2_broad | Tier-1+2 | none | A | 0.877 | 0.741 | 0.840 | — |
| tier1_refined_lexical | Tier-1 | lexical | A | 0.908 | 0.733 | 0.833 | — |
| tier1_broad | Tier-1 | none | A | 0.923 | 0.730 | 0.826 | 0.925 |
| tier1_refined_ce | Tier-1 | CE | A | 0.892 | 0.687 | 0.802 | — |
| all-Haiku | BITEM | Haiku throughout | A | 0.877 | 0.708 | 0.792 | — |
| bitem_only_opus | BITEM | Sonnet | A | 0.892 | 0.679 | 0.787 | 0.792 |
| **bitem_only_hybrid_S/H/S/H** | BITEM | mixed S/H | A | 0.923 | 0.634 | 0.765 | — |
| retrank+Error404_refined_sonnet | re+E404 | Sonnet | A | 0.877 | 0.595 | 0.738 | 0.938 |
| Error404-only | Error404 | Sonnet | A | 0.908 | 0.533 | 0.684 | — |
| ... | ... | ... | ... | ... | ... | ... | ... |

(Full table in [docs/paper_draft.pdf Appendix A](../paper_draft.pdf).)

## Quick filters

- **Top by HMR (Sonnet):** `ensemble_top1` > `ensemble_top3` > `bitem_only_refined_sonnet`
- **Top by HMR (Opus):** same three, all at HMR 0.969–0.974
- **Best single-team pool:** BITEM-only refined-Sonnet (0.963)
- **Worst configurations:** retrank+E404_refined_sonnet (Sonnet judge), Error404-only (judge-stable)
- **Most judge-divergent:** retrank+E404_refined_sonnet (0.738 → 0.938 swing); Tier-1 broad (0.826 → 0.925)

## Reproduce

```bash
ls data/eval/ac_runs/*.json | wc -l   # 33+ candidates
python scripts/refusal_decomposition.py
```
