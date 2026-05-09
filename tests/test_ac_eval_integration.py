"""Integration test: parse a tiny AC run, run the evaluator end-to-end with a
mocked LLM (no network), confirm metrics come out as expected.

Hand-crafted scenario:
  - 3 questions
  - Q1 correct + confident (conf 90)
  - Q2 incorrect + overconfident (conf 90) → drives R_O down
  - Q3 correct + underconfident (conf 10) → drives R_U down
  - Each question has 2 nuggets; one is bogus (will be flagged by Stage A)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ac_eval
from ac_format import ACRecord, Nugget


SAMPLE_AC_RUN = """<D0001>Anthony Mackie;90
1;test-run;1;The Manchurian Candidate (2004) starred Anthony Mackie
2;test-run;2;Anthony Mackie also starred in Captain America: Brave New World
</D0001>
<D0002>Steven Spielberg;90
1;test-run;3;Christopher Nolan directed Oppenheimer in 2023
2;test-run;4;Oppenheimer was the most recent film by Nolan as of 2025
</D0002>
<D0003>Pixar;10
1;test-run;5;Toy Story 4 was animated by Pixar
2;test-run;6;Pixar produced Toy Story 4 in 2019
</D0003>
"""


class MockClient:
    """Stand-in Anthropic client. Returns canned responses based on prompt content."""

    def __init__(self):
        self.messages = self
        self.calls = []

    def create(self, *, model, max_tokens, messages):
        prompt = messages[0]["content"]
        self.calls.append(prompt)

        # Stage A — entailment. Mark Q3-nugget-2 as bogus to test bogus filtering.
        if "Does the passage entail the nugget" in prompt:
            entailed = True
            if "Pixar produced Toy Story 4 in 2019" in prompt:
                entailed = False
            return _resp(f'{{"entailed": {str(entailed).lower()}, "reason": "test"}}')

        # Stage B — answer eval. Return correct=True for Q1 and Q3, incorrect for Q2.
        if "evaluating a system-generated answer" in prompt:
            if "Anthony Mackie" in prompt:
                return _resp('{"correct": true, "relevant_nugget_indices": [1, 2], "reason": "test"}')
            if "Spielberg" in prompt:
                return _resp('{"correct": false, "relevant_nugget_indices": [], "reason": "wrong director"}')
            if "Pixar" in prompt:
                # Only nugget 1 entailed (nugget 2 was filtered as bogus)
                return _resp('{"correct": true, "relevant_nugget_indices": [1], "reason": "test"}')

        return _resp('{"entailed": true, "reason": "fallback"}')


class _Resp:
    def __init__(self, text):
        self.content = [type("C", (), {"text": text})()]


def _resp(text):
    return _Resp(text)


def test_full_eval(tmp_path: Path, monkeypatch):
    # Arrange — write fake AC run and a minimal "PR run" file
    ac_path = tmp_path / "test-AC-1.txt"
    ac_path.write_text(SAMPLE_AC_RUN)

    pr_dir = tmp_path / "pr"
    pr_dir.mkdir()
    pr_path = pr_dir / "test-run"
    pr_path.write_text("""0001;1;d1;The Manchurian Candidate stars Anthony Mackie
0001;2;d2;Anthony Mackie portrayed Captain America in 2025
0002;3;d3;Christopher Nolan directed Oppenheimer in 2023
0002;4;d4;Oppenheimer was Nolan's most recent film
0003;5;d5;Toy Story 4 is a Pixar film
0003;6;d6;Toy Story 4 came out in 2019
""")

    topics_path = tmp_path / "topics.txt"
    topics_path.write_text("""<question><qID>0001</qID><q>Who played Captain America?</q></question>
<question><qID>0002</qID><q>Who directed Oppenheimer?</q></question>
<question><qID>0003</qID><q>Which studio made Toy Story 4?</q></question>
""")

    # Don't actually instantiate Anthropic
    monkeypatch.setattr(ac_eval.anthropic, "Anthropic", lambda: MockClient())
    # Make rate-limiting trivial
    monkeypatch.setattr(ac_eval, "_call_llm",
                        lambda client, model, prompt, retries=4, min_interval=1.2:
                        _mock_llm(client, prompt))

    cache_dir = tmp_path / "cache"
    out_path = tmp_path / "out.json"

    # Act
    import json as _json
    sys_argv_save = sys.argv
    try:
        sys.argv = [
            "ac_eval.py",
            "--ac-run", str(ac_path),
            "--pr-runs-dir", str(pr_dir),
            "--topics", str(topics_path),
            "--output", str(out_path),
            "--cache-dir", str(cache_dir),
        ]
        ac_eval.main()
    finally:
        sys.argv = sys_argv_save

    out = _json.loads(out_path.read_text())
    metrics = out["metrics"]
    per_q = out["per_question"]

    # Assert
    assert metrics["n_questions"] == 3
    # Q1 correct, Q2 incorrect, Q3 correct → accuracy 2/3
    assert abs(metrics["accuracy"] - 2 / 3) < 1e-9
    # Q3 had nugget 2 flagged bogus → relevant=1, returned=2 → 0.5; Q1: 2/2=1.0; Q2: 0/2=0.0
    assert abs(metrics["mean_nugget_precision"] - (1.0 + 0.0 + 0.5) / 3) < 1e-9
    # R_O: only incorrect = Q2 with conf 0.9 → 1 - 0.9 = 0.1
    assert abs(metrics["R_O"] - 0.1) < 1e-9
    # R_U: correct = Q1 (conf 0.9) and Q3 (conf 0.1) → mean (1-conf) = (0.1+0.9)/2 = 0.5 → R_U = 0.5
    assert abs(metrics["R_U"] - 0.5) < 1e-9
    # HMR = 2*0.1*0.5/(0.1+0.5) = 1/6
    assert abs(metrics["HMR"] - (2 * 0.1 * 0.5 / 0.6)) < 1e-9

    # Per-question detail
    assert per_q["0003"]["n_entailed"] == 1   # one nugget was bogus
    assert per_q["0003"]["n_relevant"] == 1
    assert per_q["0002"]["correct"] is False


def _mock_llm(_client, prompt):
    """Mimic the real _call_llm by returning parsed JSON dict for our canned prompts."""
    if "Does the passage entail the nugget" in prompt:
        if "Pixar produced Toy Story 4 in 2019" in prompt:
            return {"entailed": False, "reason": "test bogus"}
        return {"entailed": True, "reason": "test entailed"}
    if "evaluating a system-generated answer" in prompt:
        if "Anthony Mackie" in prompt:
            return {"correct": True, "relevant_nugget_indices": [1, 2], "reason": "test"}
        if "Spielberg" in prompt:
            return {"correct": False, "relevant_nugget_indices": [], "reason": "wrong"}
        if "Pixar" in prompt:
            return {"correct": True, "relevant_nugget_indices": [1], "reason": "test"}
    return {"entailed": True, "reason": "fallback"}
