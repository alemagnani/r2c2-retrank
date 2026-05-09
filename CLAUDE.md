# R2C2-Retrank: NTCIR-19 R2C2 PR + AC

## Task
NTCIR-19 R2C2 (RAG Responses: Confident and Correct?). Two subtasks:

- **PR (Passage Retrieval)** — for each question, return up to 20 ranked passages
  from the corpus. **No formal length cap on passages.** (Earlier iterations of this
  project assumed a "≤200 chars" rule based on a misread of the corpus
  `text_snippet` field; the actual rule has no length limit. Organizer reference
  runs use ~2,400-char passages.) Submitted April 24, 2026.
- **AC (Answering with Confidence)** — for each question, return one answer string,
  a confidence score 0–100, and a list of grounded nuggets each citing a passage
  from any team's PR run. Due **May 15, 2026**.

Evaluation:
- AC scored on three metrics: **Accuracy**, **Mean Nugget Precision**, **HMR**
  (Sakai's Harmonic Mean of Rewards — penalises overconfidence and underconfidence
  separately).
- PR re-scored using AC labels: passage relevance grade = number of relevant
  nuggets it provided across all teams' AC runs → nDCG@20.

## Status (current)
- ✅ PR submitted: 4 runs (`retrank-PG-1` through `retrank-PG-4`), packaged as
  `data/runs/retrank-PR.zip`.
- ✅ Other teams' PR runs downloaded to `data/raw/competitor_runs/PRruns/`
  (22 runs, 7 teams including ours and the organizer).
- ⏳ AC pipeline + self-evaluator in progress.

## Data layout
```
data/
  raw/
    competitor_runs/PRruns/   — all teams' PR runs (downloaded post-deadline)
    r2c2topics.txt            — official 65 topics (XML)
  processed/
    own_passages/             — our PG sliding-window index (BM25S + FAISS)
    passages_meta.pkl         — organizer-delivered passage metadata
    lora_biencoder/           — our LoRA-fine-tuned bi-encoder (didn't help)
  synthetic_val/              — val250 evaluation set (248 topics)
  eval/
    phase2/                   — Phase 2/3 sweep results
    all_team_ce_scores.pkl    — CE scores for every team's passages
  runs/
    retrank-PG-[1-4].txt      — submitted runs
    retrank-PR.zip            — submitted bundle
    phase2/                   — internal sweep run files
```

## Stack
- Python 3.11+, PyTorch, transformers, sentence-transformers
- BM25 via BM25S (`bm25s`)
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- LLM judge: Claude Sonnet 4.6 (production); Haiku for cheap sweeps
- Arch Linux, PyCharm, Claude Code

## Submission format

PR (already submitted):
```
qID;PassageRank;DocumentID;PassageText
0001;1;665860;A portable negative pressure ventilator was a type of medical device...
```

AC (next):
```
<D001>[AnswerString];[Confidence]
[NuggetNum];[PRrunname];[PassageRank];[NuggetText]
[NuggetNum];[PRrunname];[PassageRank];[NuggetText]
...
</D001>
```
- Up to 4 AC runs per team
- File naming: `[TeamName]-AC-[1..4]`, zipped as `[TeamName]-AC.zip`
- Email to `ntcir19r2c2org@list.waseda.jp`

## Key dates
- Apr 17, 2026 — original PR deadline
- Apr 24, 2026 — extended PR deadline (used)
- **May 15, 2026** — AC deadline (next)
- Aug 1, 2026 — official evaluation results released
- Sep 1, 2026 — paper drafts due
- Nov 1, 2026 — camera-ready
- Dec 8–10, 2026 — NTCIR-19 conference (Tokyo)

## Critical facts
- Corpus: Wikipedia + Wookieepedia movie articles
- Organizers provide pre-segmented passages, but their own `text_snippet` field
  is a 200-char preview, not the full passage. Real passage lengths in
  organizer reference runs (`ORG-PO-3/4`) average ~2,400 chars.
- AC participants may pool passages from any team's PR run. Our analysis shows
  retrank's PR runs have only 16% Jaccard with the average other team — we are
  the most divergent team, and pooling with Error404 + BITEM yields the highest
  quality recall.
