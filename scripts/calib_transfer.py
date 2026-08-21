#!/usr/bin/env python3
"""Calibration-transfer test: keep a competitor's ANSWERS fixed, replace their confidence
with OUR calibration layer (verifier match_score -> Stage-5 confidence cap), and measure the
HMR change. If HMR rises, our confidence layer transfers across systems/model families.

One LLM call per topic returns {correct, match_score}. Stage-5 rule:
  refuse(conf=5) if empty answer / no nuggets; else cap self-confidence at 25/60/inf for
  match_score 0/1/2.
"""
import json, re, sys, time
from pathlib import Path
import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ac_format import parse_ac_run

BASE = Path(__file__).resolve().parent.parent
client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"
qs = dict(re.findall(r'<qID>(\d+)</qID>\s*<q>(.*?)</q>',
                     (BASE / "data/raw/r2c2topics.txt").read_text(), flags=re.DOTALL))
PROMPT = ("For a movie question, judge two things from the answer and its supporting nuggets. "
          "Reply JSON only: {{\"correct\": true/false, \"match_score\": 0|1|2}} where "
          "correct = is the answer factually correct, and match_score = do the nuggets alone "
          "support the answer (2 full, 1 partial, 0 not/contradict).\n"
          "Question: {q}\nAnswer: {a}\nNuggets:\n{n}")


def judge(q, a, nugs):
    if not a.strip() or a.strip().lower() in ("i don't know", "i dont know"):
        return False, 0
    nb = "\n".join(f"- {x.text}" for x in nugs[:8]) or "(none)"
    for _ in range(3):
        try:
            r = client.messages.create(model=MODEL, max_tokens=40,
                messages=[{"role": "user", "content": PROMPT.format(q=q, a=a, n=nb)}])
            t = r.content[0].text
            c = re.search(r'"correct"\s*:\s*(true|false)', t)
            m = re.search(r'"match_score"\s*:\s*([012])', t)
            if c and m:
                return c.group(1) == "true", int(m.group(1))
        except Exception:
            time.sleep(2)
    return False, 0


CAP = {0: 25, 1: 60, 2: 100}


def hmr(conf, ok):
    conf = [c / 100 for c in conf]
    wrong = [c for c, o in zip(conf, ok) if not o]; right = [c for c, o in zip(conf, ok) if o]
    RO = 1.0 if not wrong else 1 - sum(wrong) / len(wrong)
    RU = 1.0 if not right else sum(right) / len(right)
    return round((0 if RO + RU == 0 else 2 * RO * RU / (RO + RU)), 4), round(RO, 3), round(RU, 3)


def main():
    run = sys.argv[1]
    recs = parse_ac_run(BASE / f"data/eval/competitor_runs_converted/{run}.txt")
    ok, orig_conf, trans_conf = [], [], []
    for i, rec in enumerate(recs):
        correct, ms = judge(qs.get(rec.question_id, ""), rec.answer, rec.nuggets)
        ok.append(correct)
        orig_conf.append(rec.confidence_raw)
        if not rec.nuggets or not rec.answer.strip():
            trans_conf.append(5)
        else:
            trans_conf.append(min(rec.confidence_raw, CAP[ms]))
        if (i + 1) % 20 == 0:
            print(f"  {run}: {i+1}/{len(recs)}", flush=True)
    ho, roo, ruo = hmr(orig_conf, ok)
    ht, rot, rut = hmr(trans_conf, ok)
    out = {"run": run, "acc": round(sum(ok) / len(ok), 3),
           "orig_conf_HMR": ho, "orig_R_O": roo,
           "transfer_HMR": ht, "transfer_R_O": rot, "delta_HMR": round(ht - ho, 4)}
    (BASE / f"data/eval/transfer_{run}.json").write_text(json.dumps(out))
    print(out, flush=True)


if __name__ == "__main__":
    main()
