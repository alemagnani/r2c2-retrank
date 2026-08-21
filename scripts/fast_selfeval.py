#!/usr/bin/env python3
"""Fast self-eval of a competitor AC run: per-topic answer-correctness judgment
(Sonnet) + the run's own confidence -> HMR. Used to test whether our self-evaluator
predicts other teams' official HMR (a held-out, non-circular check)."""
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
PROMPT = ("Judge whether the system's answer to a movie question is correct, using the "
          "answer and its supporting nuggets. Reply JSON only: {{\"correct\": true/false}}.\n"
          "Question: {q}\nAnswer: {a}\nSupporting nuggets:\n{n}")


def judge(q, a, nugs):
    if not a.strip() or a.strip().lower() in ("i don't know", "i dont know"):
        return False
    nb = "\n".join(f"- {x.text}" for x in nugs[:8]) or "(none)"
    for _ in range(3):
        try:
            r = client.messages.create(model=MODEL, max_tokens=20,
                messages=[{"role": "user", "content": PROMPT.format(q=q, a=a, n=nb)}])
            m = re.search(r'"correct"\s*:\s*(true|false)', r.content[0].text)
            if m:
                return m.group(1) == "true"
        except Exception:
            time.sleep(2)
    return False


def hmr(conf, ok):
    wrong = [c for c, o in zip(conf, ok) if not o]
    right = [c for c, o in zip(conf, ok) if o]
    RO = 1.0 if not wrong else 1 - sum(wrong) / len(wrong)
    RU = 1.0 if not right else sum(right) / len(right)
    return (0 if RO + RU == 0 else 2 * RO * RU / (RO + RU)), RO, RU, sum(ok) / len(ok)


def main():
    run = sys.argv[1]
    recs = parse_ac_run(BASE / f"data/eval/competitor_runs_converted/{run}.txt")
    conf, ok = [], []
    for i, rec in enumerate(recs):
        ok.append(judge(qs.get(rec.question_id, ""), rec.answer, rec.nuggets))
        conf.append(rec.confidence)
        if (i + 1) % 20 == 0:
            print(f"  {run}: {i+1}/{len(recs)}", flush=True)
    h, ro, ru, acc = hmr(conf, ok)
    out = {"run": run, "selfeval_HMR": round(h, 4), "R_O": round(ro, 3),
           "R_U": round(ru, 3), "acc": round(acc, 3)}
    (BASE / f"data/eval/fastselfeval_{run}.json").write_text(json.dumps(out))
    print(out, flush=True)


if __name__ == "__main__":
    main()
