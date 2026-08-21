#!/usr/bin/env python3
"""Fit a confidence calibrator on val250, apply to the official no-LLM predictions.

HMR objective (from src/eval/hmr.py):
  R_O = 1 - mean(confidence on INCORRECT)   -> want low conf when wrong
  R_U =     mean(confidence on CORRECT)     -> want high conf when right
  HMR = harmonic_mean(R_O, R_U)
So the optimal confidence IS P(correct); we fit that and tune an abstain threshold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

BASE = Path(__file__).resolve().parent.parent
FEATURES = ["margin", "ce", "entail"]


def rows(preds: dict, need_label: bool):
    X, y, keys = [], [], []
    for k, v in preds.items():
        f = v.get("feats")
        if not f:
            continue
        if need_label and "correct" not in v:
            continue
        X.append([f.get(n, 0.0) for n in FEATURES])
        keys.append(k)
        if need_label:
            y.append(1 if v.get("correct") else 0)
    return np.array(X, float), (np.array(y, int) if need_label else None), keys


def hmr_proxy(conf, correct):
    """conf, correct: arrays over answered+abstained items (abstain => correct=0, conf=low)."""
    c = np.asarray(conf, float)
    ok = np.asarray(correct, bool)
    R_O = 1.0 if (~ok).sum() == 0 else 1.0 - c[~ok].mean()
    R_U = 1.0 if ok.sum() == 0 else c[ok].mean()
    return 0.0 if R_O + R_U == 0 else 2 * R_O * R_U / (R_O + R_U), R_O, R_U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=str(BASE / "data/eval/no_llm_val250_preds.json"))
    ap.add_argument("--official", default=str(BASE / "data/eval/no_llm_official_preds.json"))
    ap.add_argument("--out", default=str(BASE / "data/runs/retrank-AC-noLLM-2.txt"))
    ap.add_argument("--features", default="margin,ce", help="subset of margin,ce,entail")
    args = ap.parse_args()

    feats = args.features.split(",")
    idx = [FEATURES.index(f) for f in feats]

    val = json.load(open(args.val))
    Xv, yv, _ = rows(val, need_label=True)
    Xv = Xv[:, idx]
    print(f"val250 calibration instances: {len(yv)}  (positives={yv.sum()}, "
          f"base rate={yv.mean():.3f})")

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xv, yv)
    p_tr = clf.predict_proba(Xv)[:, 1]
    auc = roc_auc_score(yv, p_tr)
    cv = cross_val_score(clf, Xv, yv, cv=5, scoring="roc_auc")
    print(f"features={feats}  train AUC={auc:.3f}  5fold CV AUC={cv.mean():.3f}±{cv.std():.3f}")
    print("  coefs:", {f: round(c, 3) for f, c in zip(feats, clf.coef_[0])},
          "intercept", round(clf.intercept_[0], 3))

    # per-single-feature AUC (which cheap signals actually discriminate)
    for j, f in enumerate(feats):
        try:
            print(f"    single-feature AUC[{f}] = {roc_auc_score(yv, Xv[:, j]):.3f}")
        except Exception:
            pass

    # tune abstain threshold tau on val250 HMR proxy
    ABSTAIN_CONF = 0.05
    best = (-1, 0.0)
    for tau in np.linspace(0.0, 0.6, 31):
        conf = np.where(p_tr < tau, ABSTAIN_CONF, p_tr)
        correct = np.where(p_tr < tau, 0, yv)  # abstain => "I don't know" => incorrect
        h, ro, ru = hmr_proxy(conf, correct)
        if h > best[0]:
            best = (h, tau, ro, ru)
    h, tau, ro, ru = best
    ans_rate = (p_tr >= tau).mean()
    print(f"val250 proxy: best tau={tau:.2f}  HMR={h:.3f} (R_O={ro:.3f} R_U={ru:.3f})  "
          f"answer-rate={ans_rate:.2f}")

    # apply to official
    off = json.load(open(args.official))
    lines = []
    n_abs = 0
    for qid, v in off.items():
        f = v.get("feats")
        if not f:
            lines += [f"<D{qid}>I don't know;5", f"</D{qid}>"]; n_abs += 1; continue
        x = np.array([[f.get(n, 0.0) for n in feats]], float)
        p = float(clf.predict_proba(x[:, list(range(len(feats)))])[:, 1][0])
        if p < tau:
            lines += [f"<D{qid}>I don't know;5", f"</D{qid}>"]; n_abs += 1
        else:
            conf = max(1, min(99, round(p * 100)))
            lines.append(f"<D{qid}>{v['answer']};{conf}")
            lines.append(f"1;{v['run']};{v['prank']};{v['nugget']}")
            lines.append(f"</D{qid}>")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(off)} topics, {n_abs} abstentions")


if __name__ == "__main__":
    main()
