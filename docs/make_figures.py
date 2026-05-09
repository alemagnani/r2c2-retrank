#!/usr/bin/env python3
"""Generate the 4 figures for the paper draft from cached eval JSONs.

Figures:
  fig_pool_size.pdf       — RQ2: HMR vs # teams in pool
  fig_ensemble.pdf        — RQ9: HMR vs top-K ensemble size
  fig_haiku_ablation.pdf  — RQ10: per-stage Sonnet→Haiku HMR cost
  fig_abstention.pdf      — RQ7: HMR vs abstention rate (Tier-1 + Sonnet ref)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_metrics(p: Path) -> dict:
    return json.loads(p.read_text())["metrics"]


# ─── Figure 1: RQ2 — pool size monotonic ─────────────────────────────────────


def fig_pool_size():
    # Pool size (# teams) → HMR for the broad-pool Variant A configurations
    points = [
        ("BITEM-only", 1, "data/eval/ac_runs/bitem_only_broad_A.json"),
        ("retrank+\nError404", 2, "data/eval/ac_runs/retrank_plus_error404_broad_A.json"),
        ("Tier-1\n(BITEM+E404+\nWaterloo)", 3, "data/eval/ac_runs/broad_A.json"),
        ("Tier-1+2\n(+ORG+hit-u+\nWaseda)", 6, "data/eval/ac_runs/tier1plus2_broad_A.json"),
    ]
    teams = [p[1] for p in points]
    hmrs = [load_metrics(BASE / p[2])["HMR"] for p in points]
    accs = [load_metrics(BASE / p[2])["accuracy"] for p in points]
    labels = [p[0] for p in points]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(teams, hmrs, "o-", color="#c0392b", linewidth=2.0, markersize=10,
            label="HMR")
    ax.plot(teams, accs, "s--", color="#34495e", linewidth=1.2, markersize=7,
            alpha=0.7, label="Accuracy")
    for x, y, lab in zip(teams, hmrs, labels):
        ax.annotate(lab, (x, y), xytext=(0, 12), textcoords="offset points",
                    fontsize=8, ha="center")
    ax.set_xlabel("# teams in passage pool")
    ax.set_ylabel("Score")
    ax.set_title("RQ2: Adding teams to the pool monotonically degrades HMR")
    ax.set_xticks(teams)
    ax.set_ylim(0.78, 1.00)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pool_size.pdf", bbox_inches="tight")
    plt.close(fig)
    print("→ fig_pool_size.pdf")


# ─── Figure 2: RQ9 — ensemble size curve ─────────────────────────────────────


def fig_ensemble():
    points = [
        (1, "data/eval/ac_runs/ensemble_top1.json"),
        (2, "data/eval/ac_runs/ensemble_top2.json"),
        (3, "data/eval/ac_runs/ensemble_top3.json"),
        (4, "data/eval/ac_runs/ensemble_top4.json"),
        (5, "data/eval/ac_runs/ensemble_top5.json"),
        (7, "data/eval/ac_runs/ensemble_top7.json"),
        (9, "data/eval/ac_runs/ensemble_top9.json"),
        (12, "data/eval/ac_runs/ensemble_top12.json"),
        (16, "data/eval/ac_runs/ensemble.json"),  # all 16
    ]
    ks = [p[0] for p in points]
    hmrs = [load_metrics(BASE / p[1])["HMR"] for p in points]
    accs = [load_metrics(BASE / p[1])["accuracy"] for p in points]
    baseline = load_metrics(BASE / "data/eval/ac_runs/bitem_only_refined_sonnet_A.json")["HMR"]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(ks, hmrs, "o-", color="#27ae60", linewidth=2.0, markersize=8,
            label="Filtered top-K ensemble HMR")
    ax.plot(ks, accs, "s--", color="#34495e", linewidth=1.2, markersize=7,
            alpha=0.7, label="Accuracy")
    ax.axhline(baseline, color="#7f8c8d", linestyle=":", linewidth=1.5,
               label=f"BITEM-only refined-Sonnet baseline ({baseline:.3f})")
    ax.set_xlabel("Top-K voters (sorted by self-eval HMR)")
    ax.set_ylabel("Score")
    ax.set_title("RQ9: Ensemble size sweet spot — top-1 wins via re-calibration")
    ax.set_xticks([1, 2, 3, 4, 5, 7, 9, 12, 16])
    ax.set_ylim(0.78, 1.00)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ensemble.pdf", bbox_inches="tight")
    plt.close(fig)
    print("→ fig_ensemble.pdf")


# ─── Figure 3: RQ10 — per-stage Haiku ablation ───────────────────────────────


def fig_haiku_ablation():
    variants = [
        ("all-Sonnet", "bitem_only_refined_sonnet_A", "All Sonnet\n(baseline)"),
        ("s4-Haiku",   "bitem_only_s4_haiku_A",       "Stage 4 only"),
        ("s2-Haiku",   "bitem_only_s2_haiku_A",       "Stage 2 only"),
        ("s1-Haiku",   "bitem_only_s1_haiku_A",       "Stage 1 only"),
        ("all-Haiku",  "bitem_only_all_haiku_A",      "All Haiku"),
    ]
    names = [v[2] for v in variants]
    hmrs = [load_metrics(BASE / f"data/eval/ac_runs/{v[1]}.json")["HMR"] for v in variants]
    ros  = [load_metrics(BASE / f"data/eval/ac_runs/{v[1]}.json")["R_O"] for v in variants]
    accs = [load_metrics(BASE / f"data/eval/ac_runs/{v[1]}.json")["accuracy"] for v in variants]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = list(range(len(names)))
    w = 0.25
    bars1 = ax.bar([i-w for i in x], hmrs, w, color="#c0392b", label="HMR")
    bars2 = ax.bar(x, ros, w, color="#e67e22", label=r"$R_O$")
    bars3 = ax.bar([i+w for i in x], accs, w, color="#3498db", label="Accuracy")
    for bar_set in [bars1, bars2, bars3]:
        for b in bar_set:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width()/2, b.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("RQ10: Per-stage Sonnet→Haiku ablation. Stage 1 is most critical.")
    ax.set_ylim(0.65, 1.02)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_haiku_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("→ fig_haiku_ablation.pdf")


# ─── Figure 4: RQ7 — abstention sweep on Tier-1 refined-Sonnet ───────────────


def fig_abstention():
    sweep = json.loads((BASE / "data/eval/ac_runs/abstention_sweep.json").read_text())
    target_pool = "tier1_refined_sonnet"
    if target_pool not in sweep:
        # Try un-prefixed name (older sweep file naming)
        target_pool = "refined_sonnet"
    rules_data = sweep[target_pool]

    # Order rules by abstention rate
    rows = sorted(rules_data.values(), key=lambda r: r["abstention_rate"])
    abst_rates = [r["abstention_rate"] * 100 for r in rows]
    hmrs = [r["metrics"]["HMR"] for r in rows]
    accs = [r["metrics"]["accuracy"] for r in rows]
    rule_names = [r["rule"].replace("_", " ").replace("R", "R") for r in rows]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(abst_rates, hmrs, "o-", color="#8e44ad", linewidth=2.0, markersize=9,
            label="HMR")
    ax.plot(abst_rates, accs, "s--", color="#34495e", linewidth=1.2, markersize=7,
            alpha=0.7, label="Accuracy")
    # Annotate the optimum
    best_idx = max(range(len(hmrs)), key=lambda i: hmrs[i])
    ax.annotate(f"best: {rule_names[best_idx][:18]}\nHMR={hmrs[best_idx]:.3f}",
                (abst_rates[best_idx], hmrs[best_idx]),
                xytext=(15, 10), textcoords="offset points", fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "black", "alpha": 0.6})
    baseline_idx = next(i for i, n in enumerate(rule_names) if "baseline" in n.lower())
    ax.annotate(f"R0 baseline\nHMR={hmrs[baseline_idx]:.3f}",
                (abst_rates[baseline_idx], hmrs[baseline_idx]),
                xytext=(-30, -30), textcoords="offset points", fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "black", "alpha": 0.6})

    ax.set_xlabel("Abstention rate (% topics refused)")
    ax.set_ylabel("Score")
    ax.set_title("RQ7: Abstention threshold sweep on Tier-1 refined-Sonnet")
    ax.set_ylim(0.85, 1.00)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_abstention.pdf", bbox_inches="tight")
    plt.close(fig)
    print("→ fig_abstention.pdf")


# ─── Figure 5: Case study — retrank+Error404 broad vs sonnet ─────────────────


def fig_case_study():
    """Visualise: 3 topics regressed, 0 improved, 57 unchanged correct, 5 unchanged wrong.
    Plus a panel zooming into the 3 regressions."""
    broad = json.loads((BASE / "data/eval/ac_runs/retrank_plus_error404_broad_A.json").read_text())["per_question"]
    sonnet = json.loads((BASE / "data/eval/ac_runs/retrank_plus_error404_refined_sonnet_A.json").read_text())["per_question"]

    # Categorise each of 65 topics
    cats = {"both correct": 0, "broad only": 0, "sonnet only": 0, "both wrong": 0}
    regression_qids = []
    for qid in sorted(broad.keys()):
        b = broad[qid].get("correct", False)
        s = sonnet.get(qid, {}).get("correct", False)
        if b and s:
            cats["both correct"] += 1
        elif b and not s:
            cats["broad only"] += 1
            regression_qids.append(qid)
        elif s and not b:
            cats["sonnet only"] += 1
        else:
            cats["both wrong"] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0),
                                     gridspec_kw={"width_ratios": [1, 1.8]})

    # ── Panel A: per-topic outcome breakdown ─────────────────
    labels = ["Both\ncorrect", "Broad✓\nSonnet✗\n(regression)",
              "Sonnet✓\nBroad✗", "Both\nwrong"]
    counts = [cats["both correct"], cats["broad only"],
              cats["sonnet only"], cats["both wrong"]]
    colors = ["#27ae60", "#c0392b", "#2980b9", "#7f8c8d"]
    bars = ax1.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.6)
    for bar, c in zip(bars, counts):
        ax1.annotate(f"{c}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylabel("# topics (of 65)")
    ax1.set_title("(a) Per-topic regression / improvement breakdown")
    ax1.set_ylim(0, max(counts) * 1.18)
    ax1.grid(True, alpha=0.3, axis="y")

    # ── Panel B: detail of the 3 regressed topics ─────────────
    qid_to_q = {
        "0008": ('"…Green Lantern in DCEU\n  then Deadpool in MCU"', "Ryan Reynolds"),
        "0010": ('"Wizard of Oz\n  in Wicked the movie"',           "Jeff Goldblum"),
        "0016": ('"Released earlier:\n  Infinity War or 1917"',     "Avengers: Infinity War"),
    }
    n_regressed = len(regression_qids)
    y = list(range(n_regressed))

    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.5, n_regressed - 0.5)
    bar_h = 0.30

    legend_handles = []
    for i, qid in enumerate(regression_qids):
        b = broad[qid]
        s = sonnet[qid]
        q_label, ans_oracle = qid_to_q.get(qid, (qid, "?"))

        # Bar 1: broad — relevant nuggets / total
        b_total = b.get("n_returned", 0)
        b_rel = b.get("n_relevant", 0)
        s_total = s.get("n_returned", 0)
        s_rel = s.get("n_relevant", 0)

        # Use grouped bars: broad on top, sonnet below
        ax2.barh(i + bar_h/2 + 0.02, b_total, bar_h, color="#bdc3c7",
                 edgecolor="black", linewidth=0.4)
        h1 = ax2.barh(i + bar_h/2 + 0.02, b_rel, bar_h, color="#27ae60",
                      edgecolor="black", linewidth=0.4, label="broad: relevant")
        ax2.barh(i - bar_h/2 - 0.02, s_total, bar_h, color="#bdc3c7",
                 edgecolor="black", linewidth=0.4)
        h2 = ax2.barh(i - bar_h/2 - 0.02, s_rel, bar_h, color="#c0392b",
                      edgecolor="black", linewidth=0.4, label="sonnet: relevant")

        if i == 0:
            legend_handles = [h1, h2]

        # Annotations
        ax2.text(b_total + 1, i + bar_h/2 + 0.02,
                 f"  {b_rel}/{b_total}", va="center", fontsize=8.5)
        ax2.text(s_total + 1, i - bar_h/2 - 0.02,
                 f"  {s_rel}/{s_total}", va="center", fontsize=8.5)

    # Replace x-tick labels with topic descriptions
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{qid}\n{qid_to_q[qid][0]}\nAnswer: {qid_to_q[qid][1]}"
                          for qid in regression_qids], fontsize=8)
    ax2.set_xlabel("Nuggets (gray = total returned, coloured = relevant per Stage B)")
    ax2.set_title("(b) Detail of the 3 regressed topics (broad ▲ vs sonnet ▼)")
    ax2.set_xlim(0, 8)
    ax2.legend(handles=legend_handles, loc="lower right", framealpha=0.9, fontsize=8)
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.invert_yaxis()

    fig.suptitle(
      "RQ6 case study: retrank+Error404 + Sonnet refinement (HMR 0.910 $\\rightarrow$ 0.738)",
      fontsize=11, y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_case_study.pdf", bbox_inches="tight")
    plt.close(fig)
    print("→ fig_case_study.pdf")


# ─── Figure 6: Sonnet vs Opus judge — scatter ────────────────────────────────


def fig_judge_xval():
    pairs = []
    for opus_path in sorted((BASE / "data/eval/ac_runs").glob("*_opus_judge.json")):
        prefix = opus_path.stem.replace("_opus_judge", "")
        sonnet_path = BASE / f"data/eval/ac_runs/{prefix}.json"
        if not sonnet_path.exists():
            continue
        s = json.loads(sonnet_path.read_text())["metrics"]
        o = json.loads(opus_path.read_text())["metrics"]
        pairs.append({"name": prefix, "s": s["HMR"], "o": o["HMR"]})

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    xs = [p["s"] for p in pairs]
    ys = [p["o"] for p in pairs]
    # 45° identity line
    lo, hi = 0.55, 1.00
    ax.plot([lo, hi], [lo, hi], "--", color="#7f8c8d", linewidth=1, alpha=0.6,
            label="identity (judge agreement)")
    # Bands ±0.05
    ax.fill_between([lo, hi], [lo - 0.05, hi - 0.05], [lo + 0.05, hi + 0.05],
                    color="#bdc3c7", alpha=0.20, label="±0.05 band")
    # Scatter
    ax.scatter(xs, ys, s=70, alpha=0.6, color="#2c3e50",
               edgecolor="white", linewidth=0.8)
    # Annotate top-3 (judge-invariant) and biggest movers
    annotated = set()
    top3 = sorted(pairs, key=lambda p: -p["s"])[:3]
    for p in top3:
        ax.annotate(p["name"][:24], (p["s"], p["o"]),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=7.5, color="#27ae60",
                    arrowprops={"arrowstyle": "-", "color": "#27ae60", "alpha": 0.5})
        annotated.add(p["name"])
    movers = sorted(pairs, key=lambda p: -abs(p["o"] - p["s"]))[:5]
    for p in movers:
        if p["name"] in annotated:
            continue
        ax.annotate(p["name"][:24], (p["s"], p["o"]),
                    xytext=(-95, -10), textcoords="offset points",
                    fontsize=7.5, color="#c0392b",
                    arrowprops={"arrowstyle": "->", "color": "#c0392b", "alpha": 0.6})

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("HMR under Sonnet 4.6 judge")
    ax.set_ylabel("HMR under Opus 4.7 judge")
    ax.set_title("Judge cross-validation: 33 candidates × 2 judges\n"
                 "Top-3 (green) lie exactly on identity; biggest movers (red) labelled")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig_judge_xval.pdf", bbox_inches="tight")
    plt.close(fig)
    print("→ fig_judge_xval.pdf")


if __name__ == "__main__":
    fig_pool_size()
    fig_ensemble()
    fig_haiku_ablation()
    fig_abstention()
    fig_case_study()
    fig_judge_xval()
    print(f"\nAll 6 figures written to {OUT}")
