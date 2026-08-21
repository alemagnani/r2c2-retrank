#!/usr/bin/env python3
"""No-LLM AC pipeline (prototype).

Encoder-only: ms-marco cross-encoder rerank -> roberta-squad2 extractive answer
-> NLI cross-encoder nugget verification. Confidence calibration comes later
(fit on val250). This prototype validates S1-S5 end-to-end on a few topics.

Usage: python scripts/no_llm_ac.py --limit 3
"""
import argparse
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PR_DIR = BASE / "data/raw/competitor_runs/PRruns"
TOPICS = BASE / "data/raw/r2c2topics.txt"
POOL_RUNS = ["BITEM-PG-1", "BITEM-PG-2"]  # the winning pool


def load_topics(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    qs = re.findall(r"<qID>(\d+)</qID>\s*<q>(.*?)</q>", text, flags=re.DOTALL)
    return [(qid, q.strip()) for qid, q in qs]


def load_pool(run_names) -> dict[str, list[tuple[str, int, str]]]:
    """qID -> list of (run_name, passage_rank, passage_text)."""
    pool: dict[str, list[tuple[str, int, str]]] = {}
    for rn in run_names:
        for line in (PR_DIR / rn).read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split(";", 3)
            if len(parts) < 4:
                continue
            qid, rank, _doc, txt = parts
            pool.setdefault(qid.strip(), []).append((rn, int(rank), txt.strip()))
    return pool


def sent_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def best_sentence_for(answer: str, passage: str) -> str:
    for s in sent_split(passage):
        if answer and answer.lower() in s.lower():
            return s
    return sent_split(passage)[0] if passage else answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--run-file", default=None, help="write a valid AC run file (all topics)")
    ap.add_argument("--abstain-margin", type=float, default=-1.0,
                    help="abstain if answerability margin below this")
    args = ap.parse_args()
    import math

    print("loading models (first run downloads ~1.3GB)...", flush=True)
    import torch
    from sentence_transformers import CrossEncoder
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    qa_tok = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2").eval()
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")  # [contradiction, entailment, neutral]
    print("models loaded.\n", flush=True)

    @torch.no_grad()
    def extract(question: str, context: str, max_len: int = 384, max_ans_tok: int = 30):
        """Return (answer_text, span_score, null_score). Null = CLS/CLS logits."""
        enc = qa_tok(question, context, return_tensors="pt", truncation="only_second",
                     max_length=max_len, return_offsets_mapping=True, padding="max_length")
        offsets = enc.pop("offset_mapping")[0]
        seq_ids = enc.sequence_ids(0)
        out = qa_model(**enc)
        s_log, e_log = out.start_logits[0], out.end_logits[0]
        null = (s_log[0] + e_log[0]).item()
        # restrict to context tokens
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
        cs, ce = int(offsets[s][0]), int(offsets[e][1])
        return context[cs:ce].strip(), sc, null

    all_topics = load_topics(TOPICS)
    topics = all_topics if args.run_file else all_topics[: args.limit]
    pool = load_pool(POOL_RUNS)
    run_lines: list[str] = []

    def confidence(margin: float, entail: float) -> float:
        p = 1.0 / (1.0 + math.exp(-(margin - 1.0) / 2.0))   # answerability -> [0,1]
        p *= (0.5 + 0.5 * entail)                            # discount weakly-entailed nuggets
        return p

    for qid, q in topics:
        cands = pool.get(qid, [])
        if not cands:
            if args.run_file:
                run_lines.append(f"<D{qid}>I don't know;3")
                run_lines.append(f"</D{qid}>")
            print(f"<D{qid}> [no pool passages]\n")
            continue
        # S1: rerank
        scores = reranker.predict([(q, txt) for _, _, txt in cands])
        ranked = sorted(zip(scores, cands), key=lambda x: -x[0])[: args.topk]
        # S2: extractive answer over top-k passages
        best = None  # (margin, answer, run, rank, passage, ce, span_score, null)
        for ce_score, (rn, prank, txt) in ranked:
            answer, span_score, null = extract(q, txt[:3000])
            margin = span_score - null  # answerability margin
            if best is None or margin > best[0]:
                best = (margin, answer, rn, prank, txt, float(ce_score), span_score, null)
        margin, answer, rn, prank, passage, ce, span_score, null = best
        # S4: nugget = answer-bearing sentence; S5: NLI verify passage -> nugget
        nugget = best_sentence_for(answer, passage)
        e = nli.predict([(passage[:2000], nugget)])[0]
        ex = [math.exp(v) for v in e]
        entail = ex[1] / sum(ex)

        abstain = (margin < args.abstain_margin) or (not answer)
        if abstain:
            conf_int = 5
            out_answer = "I don't know"
        else:
            conf_int = max(1, min(99, round(confidence(margin, entail) * 100)))
            out_answer = answer

        if args.run_file:
            run_lines.append(f"<D{qid}>{out_answer};{conf_int}")
            if not abstain:
                run_lines.append(f"1;{rn};{prank};{nugget}")
            run_lines.append(f"</D{qid}>")
        else:
            print(f"<D{qid}> Q: {q}")
            print(f"   answer      : {answer!r}  (margin={margin:.3f}, span={span_score:.2f}, "
                  f"null={null:.2f}, ce={ce:.2f})")
            print(f"   cite        : {rn} rank {prank}   nli_entail={entail:.3f}   "
                  f"conf={conf_int}{'  [ABSTAIN]' if abstain else ''}")
            print(f"   nugget      : {nugget[:160]!r}")
            print()

    if args.run_file:
        Path(args.run_file).write_text("\n".join(run_lines) + "\n", encoding="utf-8")
        n_abs = sum(1 for l in run_lines if l.startswith("<D") and l.endswith(";5"))
        print(f"wrote {args.run_file}: {len(topics)} topics, ~{n_abs} abstentions")


if __name__ == "__main__":
    main()
