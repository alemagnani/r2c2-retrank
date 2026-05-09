"""
AC run file parser/serializer.

Format (one record per question):

    <D001>[AnswerString];[Confidence]
    [NuggetNum];[PRrunname];[PassageRank];[NuggetText]
    [NuggetNum];[PRrunname];[PassageRank];[NuggetText]
    ...
    </D001>

Confidence is an integer 0-100 in submissions; we normalise to [0, 1] in our
internal records to match the HMR math.

Example:
    <D001>Anthony Mackie;95
    1;retrank-PG-1;3;The Manchurian Candidate (2004) starred Anthony Mackie
    2;Error404-PO-1;7;Anthony Mackie portrayed Captain America in the 2025 film
    </D001>
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Nugget:
    num: int                # 1-indexed nugget number within the record
    pr_run_name: str        # e.g. "retrank-PG-1"
    passage_rank: int       # 1..20
    text: str

    def passage_key(self) -> tuple[str, int]:
        return (self.pr_run_name, self.passage_rank)


@dataclass
class ACRecord:
    question_id: str        # e.g. "0001" (organizer ID; the <D...> tag)
    answer: str             # answer string
    confidence_raw: int     # 0-100 as stored in the file
    nuggets: list[Nugget] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Normalised confidence in [0, 1]."""
        return self.confidence_raw / 100.0


_OPEN_TAG = re.compile(r"^<D(\d+)>(.*)$")
_CLOSE_TAG = re.compile(r"^</D\d+>\s*$")


def parse_ac_run(path: Path) -> list[ACRecord]:
    """Parse a [TeamName]-AC-[N] file. Lenient about whitespace.

    Raises ValueError on malformed input with a line number for context.
    """
    records: list[ACRecord] = []
    current: ACRecord | None = None
    text = Path(path).read_text(encoding="utf-8")

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if not line:
            continue

        open_m = _OPEN_TAG.match(line)
        if open_m:
            if current is not None:
                raise ValueError(
                    f"line {lineno}: opening <D...> while {current.question_id} not closed"
                )
            qid_int, rest = open_m.group(1), open_m.group(2)
            # rest is "[Answer];[Confidence]"
            if ";" not in rest:
                raise ValueError(
                    f"line {lineno}: expected 'Answer;Confidence' after <D{qid_int}>, got {rest!r}"
                )
            answer, conf_str = rest.rsplit(";", 1)
            try:
                conf = int(conf_str.strip())
            except ValueError as e:
                raise ValueError(f"line {lineno}: bad confidence integer {conf_str!r}") from e
            if not (0 <= conf <= 100):
                raise ValueError(f"line {lineno}: confidence {conf} out of [0,100]")
            current = ACRecord(question_id=qid_int.zfill(4), answer=answer, confidence_raw=conf)
            continue

        if _CLOSE_TAG.match(line):
            if current is None:
                raise ValueError(f"line {lineno}: closing tag with no open record")
            records.append(current)
            current = None
            continue

        # Inside a record: nugget line
        if current is None:
            raise ValueError(f"line {lineno}: content outside any <D...> tag: {line!r}")
        parts = line.split(";", 3)
        if len(parts) != 4:
            raise ValueError(
                f"line {lineno}: expected 'NuggetNum;PRrunname;PassageRank;NuggetText', got {line!r}"
            )
        num_s, pr_run, rank_s, nugget_text = parts
        try:
            num = int(num_s.strip())
            rank = int(rank_s.strip())
        except ValueError as e:
            raise ValueError(f"line {lineno}: bad integers in nugget line: {e}") from e
        current.nuggets.append(
            Nugget(num=num, pr_run_name=pr_run.strip(), passage_rank=rank, text=nugget_text)
        )

    if current is not None:
        raise ValueError(f"unclosed record for question {current.question_id}")

    return records


def write_ac_run(path: Path, records: list[ACRecord]) -> None:
    """Serialise records back to the file format. Sorts by question_id ascending."""
    out_lines: list[str] = []
    for r in sorted(records, key=lambda x: x.question_id):
        # tag uses the qid as the integer (drop leading zeros for compactness, but the
        # spec example uses zero-padded so we preserve it)
        out_lines.append(f"<D{r.question_id}>{r.answer};{r.confidence_raw}")
        for n in sorted(r.nuggets, key=lambda nn: nn.num):
            out_lines.append(f"{n.num};{n.pr_run_name};{n.passage_rank};{n.text}")
        out_lines.append(f"</D{r.question_id}>")
    Path(path).write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def validate(records: list[ACRecord]) -> list[str]:
    """Return a list of warnings (not errors) about likely format issues."""
    warnings: list[str] = []
    seen_qids: set[str] = set()
    for r in records:
        if r.question_id in seen_qids:
            warnings.append(f"duplicate question_id {r.question_id}")
        seen_qids.add(r.question_id)
        if not r.answer.strip():
            warnings.append(f"{r.question_id}: empty answer string")
        if not r.nuggets:
            warnings.append(f"{r.question_id}: zero nuggets returned (precision will be 0)")
        nugget_nums = [n.num for n in r.nuggets]
        if nugget_nums != sorted(nugget_nums):
            warnings.append(f"{r.question_id}: nugget numbers not sorted")
        if len(set(nugget_nums)) != len(nugget_nums):
            warnings.append(f"{r.question_id}: duplicate nugget numbers")
    return warnings
