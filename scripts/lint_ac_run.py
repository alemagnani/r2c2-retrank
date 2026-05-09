#!/usr/bin/env python3
"""Lint an AC submission file against the R2C2 spec.

Format:
  <Dxxxx>AnswerString;Confidence
  NuggetNum;PRrunname;PassageRank;NuggetText
  ...
  </Dxxxx>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

EXPECTED_TOPICS = 65


def lint(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    open_re = re.compile(r"^<D(\d{4})>(.*?);(\d+(?:\.\d+)?)$")
    close_re = re.compile(r"^</D(\d{4})>$")
    nugget_re = re.compile(r"^(\d+);([^;]+);(\d+);(.+)$")

    current = None
    nugget_idx = 0
    seen_topics: list[str] = []
    seen_refs: set[tuple] = set()

    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        m = open_re.match(line)
        if m:
            if current is not None:
                errors.append(f"L{i}: opened <D{m.group(1)}> while <D{current}> still open")
            current = m.group(1)
            nugget_idx = 0
            seen_refs = set()
            seen_topics.append(current)
            ans, conf = m.group(2), float(m.group(3))
            if not 0 <= conf <= 100:
                errors.append(f"L{i} D{current}: confidence {conf} out of [0,100]")
            if not ans.strip():
                errors.append(f"L{i} D{current}: empty answer string")
            continue
        m = close_re.match(line)
        if m:
            if current is None:
                errors.append(f"L{i}: </D{m.group(1)}> with no open topic")
            elif m.group(1) != current:
                errors.append(f"L{i}: </D{m.group(1)}> closes <D{current}>")
            current = None
            nugget_idx = 0
            continue
        m = nugget_re.match(line)
        if m:
            if current is None:
                errors.append(f"L{i}: nugget line outside <D...>")
                continue
            num = int(m.group(1))
            nugget_idx += 1
            if num != nugget_idx:
                errors.append(f"L{i} D{current}: nugget num {num}, expected {nugget_idx}")
            run, rank, txt = m.group(2), int(m.group(3)), m.group(4)
            if rank < 1 or rank > 20:
                errors.append(f"L{i} D{current}: passage rank {rank} out of [1,20]")
            ref = (run, rank, txt)
            if ref in seen_refs:
                errors.append(f"L{i} D{current}: duplicate nugget {ref[:2]} text")
            seen_refs.add(ref)
            if not txt.strip():
                errors.append(f"L{i} D{current}: empty nugget text")
            continue
        errors.append(f"L{i}: unparseable line: {line[:80]!r}")

    if current is not None:
        errors.append(f"EOF: <D{current}> never closed")

    if len(seen_topics) != EXPECTED_TOPICS:
        errors.append(f"topic count={len(seen_topics)}, expected {EXPECTED_TOPICS}")
    dups = {t for t in seen_topics if seen_topics.count(t) > 1}
    if dups:
        errors.append(f"duplicate topic IDs: {sorted(dups)}")
    expected_ids = {f"{i:04d}" for i in range(1, EXPECTED_TOPICS + 1)}
    missing = expected_ids - set(seen_topics)
    extra = set(seen_topics) - expected_ids
    if missing:
        errors.append(f"missing topic IDs: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected topic IDs: {sorted(extra)}")

    return errors


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: lint_ac_run.py FILE [FILE...]")
        sys.exit(2)
    rc = 0
    for p in paths:
        errs = lint(p)
        tag = "OK " if not errs else "FAIL"
        print(f"[{tag}] {p.name}  ({len(errs)} issue{'s' if len(errs)!=1 else ''})")
        for e in errs[:30]:
            print(f"   {e}")
        if len(errs) > 30:
            print(f"   ... and {len(errs)-30} more")
        if errs:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
