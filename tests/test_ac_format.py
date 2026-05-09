"""Tests for scripts/ac_format.py — round-trip parse/write and edge cases."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ac_format import ACRecord, Nugget, parse_ac_run, validate, write_ac_run


SAMPLE = """<D0001>Anthony Mackie;95
1;retrank-PG-1;3;The Manchurian Candidate (2004) starred Anthony Mackie
2;Error404-PO-1;7;Anthony Mackie portrayed Captain America in the 2025 film
</D0001>
<D0002>;0
</D0002>
"""


def test_parse_sample(tmp_path: Path):
    p = tmp_path / "sample-AC-1"
    p.write_text(SAMPLE)
    records = parse_ac_run(p)
    assert len(records) == 2

    r1 = records[0]
    assert r1.question_id == "0001"
    assert r1.answer == "Anthony Mackie"
    assert r1.confidence_raw == 95
    assert r1.confidence == 0.95
    assert len(r1.nuggets) == 2
    assert r1.nuggets[0].pr_run_name == "retrank-PG-1"
    assert r1.nuggets[0].passage_rank == 3
    assert r1.nuggets[1].pr_run_name == "Error404-PO-1"

    r2 = records[1]
    assert r2.answer == ""
    assert r2.confidence_raw == 0
    assert r2.nuggets == []


def test_round_trip(tmp_path: Path):
    src = tmp_path / "src"
    src.write_text(SAMPLE)
    records = parse_ac_run(src)
    out = tmp_path / "out"
    write_ac_run(out, records)
    records2 = parse_ac_run(out)
    assert len(records) == len(records2)
    for a, b in zip(records, records2):
        assert a.question_id == b.question_id
        assert a.answer == b.answer
        assert a.confidence_raw == b.confidence_raw
        assert len(a.nuggets) == len(b.nuggets)


def test_malformed_no_close(tmp_path: Path):
    p = tmp_path / "bad"
    p.write_text("<D0001>x;50\n1;run;1;nugget\n")
    with pytest.raises(ValueError, match="unclosed"):
        parse_ac_run(p)


def test_malformed_bad_confidence(tmp_path: Path):
    p = tmp_path / "bad"
    p.write_text("<D0001>x;abc\n</D0001>\n")
    with pytest.raises(ValueError, match="bad confidence"):
        parse_ac_run(p)


def test_malformed_confidence_out_of_range(tmp_path: Path):
    p = tmp_path / "bad"
    p.write_text("<D0001>x;150\n</D0001>\n")
    with pytest.raises(ValueError, match="out of"):
        parse_ac_run(p)


def test_validate_warnings():
    rec = ACRecord(
        question_id="0001",
        answer="",
        confidence_raw=50,
        nuggets=[Nugget(2, "r", 1, "t"), Nugget(1, "r", 1, "t"), Nugget(1, "r", 1, "t2")],
    )
    warnings = validate([rec, rec])
    text = "\n".join(warnings)
    assert "empty answer" in text
    assert "not sorted" in text
    assert "duplicate nugget numbers" in text
    assert "duplicate question_id" in text


def test_passage_text_with_semicolon_in_nugget(tmp_path: Path):
    """Nugget text can contain semicolons; we split with maxsplit=3."""
    p = tmp_path / "x"
    p.write_text("<D0001>foo;50\n1;run;1;Text with; embedded semicolon\n</D0001>\n")
    records = parse_ac_run(p)
    assert records[0].nuggets[0].text == "Text with; embedded semicolon"
