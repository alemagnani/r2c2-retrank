#!/usr/bin/env python3
"""No-LLM AC predictor: dump per-topic answer + citation + calibration features.

Two sources:
  --source official : 65 topics, pool = BITEM-PG-1/2 (the winning pool)
  --source val250   : 248 synthetic topics, pool retrieved from own bm25s index
                      (carries gold answer + correctness for calibration)

Output: JSON {qid: {answer, run, prank, nugget, feats{margin,span,null,ce,entail},
                    gold?, correct?}}. Confidence is assigned later by no_llm_calibrate.py.
"""
import argparse
import json
import math
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PR_DIR = BASE / "data/raw/competitor_runs/PRruns"
TOPICS_XML = BASE / "data/raw/r2c2topics.txt"
VAL250 = BASE / "data/synthetic_val/synthetic_topics_val250.json"
OWN_INDEX = BASE / "data/processed/own_passages/bm25s_index"
OWN_META = BASE / "data/processed/own_passages/passages_meta.pkl"
POOL_RUNS = ["BITEM-PG-1", "BITEM-PG-2"]


def load_topics_xml(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    return [(qid, q.strip()) for qid, q in
            re.findall(r"<qID>(\d+)</qID>\s*<q>(.*?)</q>", text, flags=re.DOTALL)]


def load_pool(run_names):
    pool = {}
    for rn in run_names:
        for line in (PR_DIR / rn).read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split(";", 3)
            if len(parts) < 4:
                continue
            qid, rank, _doc, txt = parts
            pool.setdefault(qid.strip(), []).append((rn, int(rank), txt.strip()))
    return pool


def sent_split(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def best_sentence_for(answer, passage):
    for s in sent_split(passage):
        if answer and answer.lower() in s.lower():
            return s
    ss = sent_split(passage)
    return ss[0] if ss else answer


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()


def normstr(s):
    return " ".join(norm(s))


def is_correct(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return False
    ps, gs = " ".join(p), " ".join(g)
    if ps in gs or gs in ps:
        return True
    inter = len(set(p) & set(g))
    return inter / max(1, len(set(g))) >= 0.6  # token recall of gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["official", "val250"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--qa-models", default="deepset/roberta-base-squad2",
                    help="comma-separated extractive QA models (committee of answerers)")
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering

    print("loading models...", flush=True)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    qa_names = [m.strip() for m in args.qa_models.split(",") if m.strip()]
    qa_models = []
    for name in qa_names:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForQuestionAnswering.from_pretrained(name).eval()
        qa_models.append((name, tok, mdl))
    print(f"answerers: {qa_names}", flush=True)
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")

    @torch.no_grad()
    def extract(question, context, tok, qa_model, max_len=384, max_ans_tok=30):
        enc = tok(question, context, return_tensors="pt", truncation="only_second",
                  max_length=max_len, return_offsets_mapping=True, padding="max_length")
        offsets = enc.pop("offset_mapping")[0]
        seq_ids = enc.sequence_ids(0)
        out = qa_model(**enc)
        s_log, e_log = out.start_logits[0], out.end_logits[0]
        null = (s_log[0] + e_log[0]).item()
        ctx = [i for i, sid in enumerate(seq_ids) if sid == 1]
        if not ctx:
            return "", -1e9, null
        best = (-1e9, 0, 0)
        for s in ctx:
            for e in range(s, min(s + max_ans_tok, ctx[-1] + 1)):
                sc = (s_log[s] + e_log[e]).item()
                if sc > best[0]:
                    best = (sc, s, e)
        sc, s, e = best
        return context[int(offsets[s][0]):int(offsets[e][1])].strip(), sc, null

    # ---- assemble (qid, question, gold, candidate passages) ----
    items = []  # (qid, question, gold_or_None, [(run,rank,text)])
    if args.source == "official":
        topics = load_topics_xml(TOPICS_XML)
        pool = load_pool(POOL_RUNS)
        for qid, q in topics:
            items.append((qid, q, None, pool.get(qid, [])))
    else:
        # Index-free calibration harness: mini-pool of [gold answer_span] + distractor
        # spans from other topics. Also emit a distractor-only ("no-gold") instance per
        # topic so the calibrator sees genuine unanswerable/should-abstain cases.
        import random
        rng = random.Random(13)
        val = [t for t in json.load(open(VAL250)) if t.get("answer_span")]
        spans = [t["answer_span"] for t in val]
        K = 7
        for i, t in enumerate(val):
            q, gold, pos = t["question"], t.get("answer_hint", ""), t["answer_span"]
            dis = [s for s in rng.sample(spans, min(K + 2, len(spans))) if s != pos][:K]
            with_gold = [("gold", 0, pos)] + [("dis", j + 1, d) for j, d in enumerate(dis)]
            no_gold = [("dis", j + 1, d) for j, d in enumerate(dis)]
            items.append((f"{t['topic_id']}#g", q, gold, with_gold))
            items.append((f"{t['topic_id']}#n", q, gold, no_gold))

    if args.limit:
        items = items[: args.limit]

    preds = {}
    for n, (qid, q, gold, cands) in enumerate(items, 1):
        if not cands:
            preds[qid] = {"answer": "", "feats": None, "gold": gold, "correct": False}
            continue
        scores = reranker.predict([(q, txt) for _, _, txt in cands])
        ranked = sorted(zip(scores, cands), key=lambda x: -x[0])[: args.topk]
        # committee of answerers x passages -> candidate spans
        from collections import defaultdict
        cand_list = []           # (ns, ans, margin, span, null, rn, prank, txt, ce, mi)
        per_model_best = {}      # mi -> (margin, ns)
        for ce_score, (rn, prank, txt) in ranked:
            for mi, (name, tok, mdl) in enumerate(qa_models):
                ans, span, null = extract(q, txt[:3000], tok, mdl)
                margin = span - null
                ns = normstr(ans)
                cand_list.append((ns, ans, margin, span, null, rn, prank, txt, float(ce_score), mi))
                if ns and (mi not in per_model_best or margin > per_model_best[mi][0]):
                    per_model_best[mi] = (margin, ns)
        # vote: aggregate weight per normalized answer
        weight = defaultdict(float)
        bestc = {}
        for c in cand_list:
            if not c[0]:
                continue
            weight[c[0]] += max(c[2], 0.0) + 0.1
            if c[0] not in bestc or c[2] > bestc[c[0]][2]:
                bestc[c[0]] = c
        if weight:
            win = max(weight, key=lambda k: (weight[k], bestc[k][2]))
            c = bestc[win]
        else:
            c = cand_list[0]
        ns, answer, margin, span, null, rn, prank, passage, ce, mi = c
        n_models = len(qa_models)
        n_agree = sum(1 for _mi, (mg, pn) in per_model_best.items() if pn == ns and ns)
        agreement = n_agree / n_models if n_models else 0.0
        # Step 2: ensemble score-variance uncertainty (dispersion of members' best margins)
        member_margins = [mg for _mi, (mg, pn) in per_model_best.items()]
        if len(member_margins) > 1:
            mu = sum(member_margins) / len(member_margins)
            margin_std = (sum((x - mu) ** 2 for x in member_margins) / len(member_margins)) ** 0.5
            margin_range = max(member_margins) - min(member_margins)
        else:
            margin_std = margin_range = 0.0
        nugget = best_sentence_for(answer, passage)
        e = nli.predict([(passage[:2000], nugget)])[0]
        ex = [math.exp(v) for v in e]
        entail = ex[1] / sum(ex)
        rec = {"answer": answer, "run": rn, "prank": prank, "nugget": nugget,
               "feats": {"margin": margin, "span": span, "null": null, "ce": ce,
                         "entail": entail, "agreement": agreement, "n_agree": n_agree,
                         "margin_std": margin_std, "margin_range": margin_range}}
        if gold is not None:
            rec["gold"] = gold
            rec["correct"] = is_correct(answer, gold)
        preds[qid] = rec
        if n % 25 == 0:
            print(f"  {n}/{len(items)}", flush=True)

    Path(args.out).write_text(json.dumps(preds, indent=1))
    if args.source == "val250":
        acc = sum(1 for r in preds.values() if r.get("correct")) / len(preds)
        print(f"val250 extractive accuracy (vs gold hint): {acc:.3f}")
    print(f"wrote {args.out} ({len(preds)} topics)")


if __name__ == "__main__":
    main()
