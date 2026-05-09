#!/usr/bin/env python3
"""Per-topic R_O bottleneck visualisation.

For each top candidate, mark which topics are confident-wrong
(driving R_O) and refused. Visualise that the same 1-3 topics
account for almost all HMR variance — the central thesis figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
EVAL = BASE / "data" / "eval" / "ac_runs"
OUT = BASE / "docs" / "figures" / "fig_ro_bottleneck.pdf"


def per_topic_status(p: Path) -> list[tuple[str, str, float, bool]]:
    """Returns (qid, status, conf, correct)
    status in {'CORRECT', 'REFUSE', 'CONFIDENT_WRONG', 'LOW_CONF_WRONG'}."""
    d = json.loads(p.read_text())
    out = []
    for qid, q in d["per_question"].items():
        ans = (q.get("answer") or "").strip().lower()
        conf = float(q.get("confidence", 0))
        correct = bool(q.get("correct"))
        is_refuse = (not ans) or ans in {"i don't know", "i dont know", "unknown"} or conf <= 0.10
        if correct:
            status = "CORRECT"
        elif is_refuse:
            status = "REFUSE"
        elif conf >= 0.5:
            status = "CONFIDENT_WRONG"
        else:
            status = "LOW_CONF_WRONG"
        out.append((qid, status, conf, correct))
    return out


def main():
    candidates = [
        ("bitem_only_refined_sonnet_A", "BITEM-Sonnet (main)"),
        ("ensemble_top1", "ensemble_top1"),
        ("bitem_only_broad_A", "BITEM-broad"),
        ("refined_sonnet_A", "Tier-1 refined-Sonnet"),
        ("broad_A", "Tier-1 broad"),
        ("bitem_only_opus_A", "BITEM-Opus pipeline"),
        ("error404_only_refined_sonnet_A", "Error404-only Sonnet"),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5))

    rows_data = []
    for stem, label in candidates:
        p = EVAL / f"{stem}.json"
        if not p.exists():
            continue
        rows_data.append((label, per_topic_status(p),
                          json.loads(p.read_text())["metrics"]))

    n_rows = len(rows_data)
    qids = sorted({qid for _, sts, _ in rows_data for qid, *_ in sts})
    n_topics = len(qids)
    qid_to_idx = {q: i for i, q in enumerate(qids)}

    color_map = {
        "CORRECT": "#dcedc8",          # light green
        "REFUSE": "#f0f0f0",           # light grey
        "CONFIDENT_WRONG": "#d32f2f",  # bright red — the R_O killers
        "LOW_CONF_WRONG": "#ffb74d",   # orange — graceful degradation
    }

    for row_idx, (label, sts, metrics) in enumerate(rows_data):
        for qid, status, conf, correct in sts:
            x = qid_to_idx[qid]
            ax.add_patch(plt.Rectangle((x, n_rows - 1 - row_idx), 1, 1,
                                        facecolor=color_map[status],
                                        edgecolor="white", linewidth=0.3))
        # annotate Cnf-W count + HMR on the right
        cnf_w = sum(1 for _, s, _, _ in sts if s == "CONFIDENT_WRONG")
        ax.text(n_topics + 1, n_rows - 1 - row_idx + 0.5,
                f"Cnf-W={cnf_w}  HMR={metrics['HMR']:.3f}",
                va="center", ha="left", fontsize=9, family="monospace")
        ax.text(-1, n_rows - 1 - row_idx + 0.5, label,
                va="center", ha="right", fontsize=9)

    ax.set_xlim(-0.5, n_topics + 18)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        f"Per-topic R$_O$ bottleneck across {n_rows} candidates "
        f"(rows) and {n_topics} topics (columns).\n"
        f"Red cells (confident-wrong) drive HMR — the same handful of topics "
        f"recur across struggling candidates.",
        loc="left", fontsize=10, pad=10,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color_map["CORRECT"],     ec="grey"),
        plt.Rectangle((0, 0), 1, 1, fc=color_map["REFUSE"],      ec="grey"),
        plt.Rectangle((0, 0), 1, 1, fc=color_map["CONFIDENT_WRONG"], ec="grey"),
        plt.Rectangle((0, 0), 1, 1, fc=color_map["LOW_CONF_WRONG"],  ec="grey"),
    ]
    ax.legend(handles, ["correct", "refused", "confident-wrong (R$_O$ killer)",
                        "low-conf wrong"],
              loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4,
              frameon=False, fontsize=9)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print(f"  saved {OUT}")

    # summary stats
    print(f"\n  Topic-level confident-wrong overlap:")
    qid_cnf_w_count = {q: 0 for q in qids}
    for _, sts, _ in rows_data:
        for qid, status, _, _ in sts:
            if status == "CONFIDENT_WRONG":
                qid_cnf_w_count[qid] += 1
    repeats = [(q, c) for q, c in qid_cnf_w_count.items() if c >= 2]
    print(f"    Topics confident-wrong on ≥2 candidates: {len(repeats)}")
    for q, c in sorted(repeats, key=lambda x: -x[1])[:10]:
        print(f"      {q}: confident-wrong on {c} candidates")


if __name__ == "__main__":
    main()
