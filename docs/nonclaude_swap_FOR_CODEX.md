# EXPERIMENT SPEC FOR CODEX — Non-Claude base-model swap (retrank / NTCIR-19 R2C2 AC)

**You are Codex, running on the retrank repo at `/home/alessandro/workspace/r2c2-retrank`.**
Execute the experiment below and emit results in the JSON format in §8. Everything you need
is in this file. Do not ask questions; make the reasonable choices noted and record them.

---

## 1. Objective & hypothesis

The retrank AC pipeline ("answer-first + independent-verifier confidence calibration") won
NTCIR-19 R2C2 using **Claude**. A reviewer's remaining objection: the "model-agnostic
calibration layer" claim is unproven because every configuration was a Claude tier. This
experiment runs the **same pipeline with a NON-Claude model** and tests whether the
calibration layer still helps.

**Hypothesis (H):** With a non-Claude answer engine, applying our Stage-5 calibration layer
(verifier-capped confidence) raises **HMR** over the same answers with raw self-confidence,
driven by an increase in **R_O** (overconfidence reward). This mirrors the Claude result
(main run R_O 0.95, R_U 0.98, HMR 0.963 official).

**Primary success criterion:** `calibration_on.HMR - calibration_off.HMR > 0` with the gain
concentrated in `R_O`. **Secondary:** absolute HMR of the non-Claude pipeline vs the field
(best non-retrank official HMR = 0.761; retrank main = 0.963), scored under a fixed judge.

---

## 2. Models to run — EVERY available OpenAI / ChatGPT model

We have **OpenAI access only** (no Qwen/Llama). Run the pipeline **once per chat-capable
OpenAI model** available on the account, as the answer+verifier model `M`. GPT is a different
family from Claude — and is one of R2C2's two official judges (Qwen-3.5, GPT-5.5) — so this is
a genuine cross-family test.

**Discover the models programmatically, then run all of them:**
```python
from openai import OpenAI
import re
client = OpenAI()  # uses OPENAI_API_KEY
avail = sorted(m.id for m in client.models.list().data)
MODELS = [m for m in avail
          if re.match(r'^(gpt-|o[0-9]|chatgpt)', m)
          and not re.search(r'(embed|audio|image|tts|whisper|moderation|realtime|search|transcribe|dall|preview-tts)', m)]
print("running:", MODELS)
```
If discovery is noisy, use this curated list and drop any that error: `gpt-5.5`, `gpt-5`,
`gpt-5-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`, `o3`, `o4-mini` (plus any
newer). **Record the exact model id for each result and skip models that reject the request.**
(For `o*` reasoning models, omit `temperature` and use `max_completion_tokens` if the API
requires it.)

**Judge `J` (fixed across all models):** the calibration ON-vs-OFF comparison holds the judge
fixed, so it cancels out. Use **one** judge for every model so correctness labels are
consistent — recommended `gpt-5.5` (or the strongest available). Record it. The primary
result (ON vs OFF `delta_HMR`) is judge-agnostic; treat absolute HMR only relative to runs
scored by the *same* judge.

---

## 3. Data inventory (exact paths, all present in the repo)

- **Topics (65):** `data/raw/r2c2topics.txt` — XML, entries `\n<question>\n<qID>0001</qID>\n<q>...</q>\n</question>`.
- **Pool (winning pool):** `data/raw/competitor_runs/PRruns/BITEM-PG-1` and `.../BITEM-PG-2`
  — one passage per line, `qID;rank;docID;passageText` (split on `;` into 4 fields; text may
  contain no extra `;`). Use both files; per topic, take passages for that `qID`.
- **Reference (our Claude numbers), for context only:** `data/eval/official/official.AC.csv`
  (retrank-AC-2 official HMR 0.9626, Acc 0.9531, R_O 0.95, R_U 0.976).
- **Output dir (create):** `data/eval/nonclaude/`.

---

## 4. Pipeline (faithful reproduction). Per topic:

- **Stage 0 — Pool:** gather all passages for the topic from BITEM-PG-1 + BITEM-PG-2, sort by
  their `rank`, keep top **20**, label them `P1..P20`. (CE re-ranking is optional; skipping it
  is an acceptable simplification — note it.)
- **Stage 1 — Candidate answer** (`CANDIDATE_PROMPT`): returns `answer`, `confidence` (0–100,
  = `c_self`), `reason`.
- **Stage 2 — Nuggets** (`NUGGET_PROMPT`): returns 3–8 nuggets, each with `text` + `cite` (P#).
  If Stage 1 returned an empty answer, skip to refusal.
- **Stage 4 — Independent verifier** (`VERIFIER_PROMPT` then `MATCH_PROMPT`): derive an answer
  from the **nuggets only**, then compare it to the Stage-1 candidate → `match_score ∈ {0,1,2}`.
- **Stage 5 — Confidence (EXACT rule):**
  - if no nuggets OR empty candidate answer → **refuse**: `answer="I don't know"`, `conf=5`.
  - else `conf_on  = min(c_self, {0:25, 1:60, 2:100}[match_score])`  (calibration ON)
  - and   `conf_off = c_self`                                        (calibration OFF, baseline)

Produce **two runs** with identical answers/nuggets: one using `conf_on`, one using `conf_off`.

## 5. Scoring (judge `J`). Per topic:

- **Correctness** (`ANSWER_EVAL_PROMPT`): given question + answer + nuggets → `correct` (bool).
  (A refusal / "I don't know" is `correct=false`.)
- Correctness is identical for both runs (same answers); only confidence differs.

## 6. Metrics (compute exactly this):

Let `conf` be in [0,1] (divide by 100). Over the 65 topics:
- `R_O = 1 - mean(conf over INCORRECT topics)`   (1.0 if none incorrect)
- `R_U = mean(conf over CORRECT topics)`          (1.0 if none correct)
- `HMR = 2*R_O*R_U/(R_O+R_U)` (0 if both 0)
- `accuracy = mean(correct)`; `mean_conf = mean(conf)`
Compute for both `calibration_on` and `calibration_off`.

---

## 7. Reference implementation (adapt & run; Python 3, `pip install openai`)

```python
import os, re, json, math
from pathlib import Path
from openai import OpenAI

BASE = Path("/home/alessandro/workspace/r2c2-retrank")
client = OpenAI()                     # uses OPENAI_API_KEY
MODEL = os.environ["MODEL"]           # set per run, e.g. gpt-5.5 / gpt-4o / o3
JUDGE = os.environ.get("JUDGE", "gpt-5.5")   # ONE fixed judge across all models

def chat(model, prompt, max_tokens=400):
    for _ in range(4):
        try:
            r = client.chat.completions.create(model=model, max_tokens=max_tokens,
                temperature=0, messages=[{"role":"user","content":prompt}])
            return r.choices[0].message.content
        except Exception:
            import time; time.sleep(3)
    return ""

def js(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    try: return json.loads(m.group(0)) if m else {}
    except Exception: return {}

# ---- prompts (verbatim from the retrank pipeline) ----
CANDIDATE_PROMPT = '''You are answering a movie question using ONLY the passages provided. Do not use external knowledge to fill gaps.
Question: {question}
Passages (numbered, with citation keys; truncated where long):
{passages_block}
Tasks:
1. Determine the answer. Be concise. For counting/arithmetic, output ONLY the numeric or list answer.
2. If the passages do not contain a clear answer, output {{"answer":"","confidence":0,"reason":"passages do not support an answer"}}.
3. Otherwise rate confidence 0-100 based ONLY on how strongly the passages support the answer.
Reply with JSON ONLY:
{{"answer":"<answer or empty>","confidence":<0-100 int>,"reason":"<one short sentence>"}}'''

NUGGET_PROMPT = '''You are extracting nuggets from passages to ground a candidate answer.
Question: {question}
Candidate answer: {candidate_answer}
Passages (numbered with citation keys; some may be irrelevant):
{passages_block}
A nugget is a SINGLE atomic factual claim, RELEVANT to the candidate answer, ENTAILED by exactly ONE passage, concise, no external knowledge. Extract 3-8 nuggets; cite EXACTLY ONE passage (P<N>) each.
Reply with JSON only:
{{"nuggets":[{{"text":"<claim>","cite":"P<N>","reason":"<phrase>"}}]}}'''

VERIFIER_PROMPT = '''You are answering a movie question using ONLY the listed nuggets. Treat each nugget as correct. No external knowledge.
Question: {question}
Nuggets:
{nuggets_block}
1. Derive the answer using only the nuggets. Be concise. 2. If insufficient, empty answer. 3. Rate confidence 0-100.
Reply JSON only:
{{"verifier_answer":"<answer or empty>","verifier_confidence":<0-100 int>,"reason":"<sentence>"}}'''

MATCH_PROMPT = '''Are these two answers equivalent (same name/value/quotation; paraphrasing/formatting ok)? Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}
Reply JSON only: {{"match_score":0|1|2,"reason":"<sentence>"}}
2=full match, 1=partial, 0=no match/empty'''

ANSWER_EVAL_PROMPT = '''You are evaluating a system answer to a movie question.
Question: {question}
System answer: {answer}
Entailed nuggets (assume factually correct):
{nuggets_block}
1. Assuming the nuggets are correct, is the system answer correct? Be strict about exact entity/value matches.
Reply JSON only: {{"correct":true|false,"reason":"<sentence>"}}'''

# ---- data ----
topics = re.findall(r"<qID>(\d+)</qID>\s*<q>(.*?)</q>",
                    (BASE/"data/raw/r2c2topics.txt").read_text(), re.S)
pool = {}
for rn in ["BITEM-PG-1", "BITEM-PG-2"]:
    for line in (BASE/"data/raw/competitor_runs/PRruns"/rn).read_text(errors="replace").splitlines():
        p = line.split(";", 3)
        if len(p) == 4:
            pool.setdefault(p[0].strip(), []).append((int(p[1]), p[3].strip()))

CAP = {0:25, 1:60, 2:100}
rows = []
for qid, q in topics:
    ps = [t for _, t in sorted(pool.get(qid, []))[:20]]
    block = "\n".join(f"P{i+1}: {t[:1200]}" for i, t in enumerate(ps))
    s1 = js(chat(MODEL, CANDIDATE_PROMPT.format(question=q, passages_block=block)))
    ans = (s1.get("answer") or "").strip(); c_self = int(s1.get("confidence") or 0)
    nuggets = []
    if ans:
        s2 = js(chat(MODEL, NUGGET_PROMPT.format(question=q, candidate_answer=ans, passages_block=block)))
        nuggets = [n.get("text","") for n in (s2.get("nuggets") or []) if n.get("text")]
    if not ans or not nuggets:
        rows.append({"qid":qid,"answer":"I don't know","nuggets":nuggets,"c_self":c_self,
                     "match":0,"conf_on":5,"conf_off":c_self,"refused":True}); continue
    nb = "\n".join(f"- {t}" for t in nuggets)
    v = js(chat(MODEL, VERIFIER_PROMPT.format(question=q, nuggets_block=nb)))
    vans = (v.get("verifier_answer") or "").strip()
    ms = js(chat(MODEL, MATCH_PROMPT.format(question=q, answer_a=ans, answer_b=vans))).get("match_score", 0)
    ms = int(ms) if ms in (0,1,2,"0","1","2") else 0
    rows.append({"qid":qid,"answer":ans,"nuggets":nuggets,"c_self":c_self,"match":ms,
                 "conf_on":min(c_self,CAP[ms]),"conf_off":c_self,"refused":False})

# ---- judge correctness (once per topic) ----
for r in rows:
    nb = "\n".join(f"{i+1}. {t}" for i,t in enumerate(r["nuggets"])) or "(none)"
    j = js(chat(JUDGE, ANSWER_EVAL_PROMPT.format(question=dict(topics)[r["qid"]],
              answer=r["answer"], nuggets_block=nb)))
    r["correct"] = bool(j.get("correct")) and not r["refused"]

def metrics(key):
    conf = [r[key]/100 for r in rows]; ok = [r["correct"] for r in rows]
    wrong=[c for c,o in zip(conf,ok) if not o]; right=[c for c,o in zip(conf,ok) if o]
    RO = 1.0 if not wrong else 1-sum(wrong)/len(wrong)
    RU = 1.0 if not right else sum(right)/len(right)
    HMR = 0 if RO+RU==0 else 2*RO*RU/(RO+RU)
    return {"accuracy":round(sum(ok)/len(ok),4),"R_O":round(RO,4),"R_U":round(RU,4),
            "HMR":round(HMR,4),"mean_conf":round(sum(conf)/len(conf),4)}

out = {"model":MODEL,"judge":JUDGE,"n_topics":len(rows),"ce_rerank":False,
       "calibration_off":metrics("conf_off"),"calibration_on":metrics("conf_on")}
out["delta_HMR"]=round(out["calibration_on"]["HMR"]-out["calibration_off"]["HMR"],4)
out["delta_R_O"]=round(out["calibration_on"]["R_O"]-out["calibration_off"]["R_O"],4)
out["per_topic"]=[{k:r[k] for k in ("qid","answer","correct","c_self","match","conf_on","conf_off","refused")} for r in rows]
Path(BASE/"data/eval/nonclaude").mkdir(parents=True, exist_ok=True)
name = MODEL.replace("/","_")
(BASE/f"data/eval/nonclaude/result_{name}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k:out[k] for k in ("model","judge","calibration_off","calibration_on","delta_HMR","delta_R_O")}, indent=1))
```

Run once per model with a **fixed judge**, writing one result file each:
```bash
export OPENAI_API_KEY=sk-...; export JUDGE=gpt-5.5
for M in gpt-5.5 gpt-5 gpt-5-mini gpt-4.1 gpt-4.1-mini gpt-4o gpt-4o-mini o3 o4-mini; do
  MODEL="$M" python3 run_nonclaude.py || echo "skipped $M"
done
# (or discover the model list via client.models.list() as in §2 and loop over that)
```

---

## 8. Output format (what to hand back)

For **each model** you ran, one JSON file `data/eval/nonclaude/result_<model>.json` with:
```json
{
  "model": "<answer+verifier model>",
  "judge": "<judge model>",
  "n_topics": 65,
  "ce_rerank": false,
  "calibration_off": {"accuracy":0.0,"R_O":0.0,"R_U":0.0,"HMR":0.0,"mean_conf":0.0},
  "calibration_on":  {"accuracy":0.0,"R_O":0.0,"R_U":0.0,"HMR":0.0,"mean_conf":0.0},
  "delta_HMR": 0.0,
  "delta_R_O": 0.0,
  "per_topic": [ {"qid":"0001","answer":"...","correct":true,"c_self":90,"match":2,"conf_on":90,"conf_off":90,"refused":false}, ... ]
}
```
Also print the compact summary block (model, judge, both metric sets, delta_HMR, delta_R_O).

**Additionally**, after running all models, write `data/eval/nonclaude/SUMMARY.json`:
```json
{
  "judge": "gpt-5.5",
  "models": [
    {"model":"gpt-5.5","accuracy":0.0,"HMR_off":0.0,"HMR_on":0.0,"delta_HMR":0.0,"delta_R_O":0.0},
    ...
  ],
  "n_models": 0,
  "n_models_positive_delta_HMR": 0,
  "mean_delta_HMR": 0.0,
  "mean_delta_R_O": 0.0
}
```
This is the single file I most need back.

---

## 9. Interpretation (what the numbers mean)

- **H supported (primary):** `delta_HMR > 0` and `delta_R_O > 0` ⇒ our calibration layer
  transfers to this non-Claude (GPT) model — the confidence wrapper is model-agnostic within
  the pipeline, not a Claude artefact. Report `delta_HMR`, `delta_R_O` **per model**, and an
  **aggregate**: the number of OpenAI models with `delta_HMR>0`, and the mean `delta_HMR` /
  mean `delta_R_O` across all models run. Consistent positive deltas across the whole GPT
  family is the strong result.
- **Family nuance to record (do not omit):** GPT-5.5 is one of the two official judges. That
  does not affect the ON-vs-OFF comparison (judge held fixed, cancels out), but if you use a
  GPT judge, note it; the primary claim rests on the delta, not absolute HMR.
- **Absolute context (secondary, same-judge only):** compare `calibration_on.HMR` to the
  field's best non-retrank official run (0.761) and to retrank's Claude main run (official
  0.963). Note the judge; do **not** cross-compare judges.
- **Expected shape:** accuracy will likely be lower than Claude's 0.95 (weaker/open models),
  but if `delta_HMR>0` via `R_O`, calibration still helps — the key claim. A near-zero or
  negative delta on a model that is already well-calibrated is still informative (report it).

## 10. Gotchas
- Some models ignore "JSON only"; the `js()` regex extracts the first `{...}`. If parse rates
  are low, add "Return ONLY valid JSON, no markdown fences." to prompts.
- Local models: set `max_tokens` generously; set `temperature=0`.
- Record any deviation (CE rerank on/off, top-K, model version, parse failures) in the JSON.
- Do NOT tune anything on the 65 official topics; run once, report.
