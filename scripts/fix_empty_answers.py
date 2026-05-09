#!/usr/bin/env python3
"""Replace empty answer strings in AC run files with 'I don't know'.

Empty answers are deliberate refusals from variant A; some scorers may
reject empty fields, so we encode the refusal as a literal token while
preserving the low confidence value.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

REFUSAL = "I don't know"
OPEN_RE = re.compile(r"^(<D\d{4}>);(\d+(?:\.\d+)?)$")


def fix(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    fixed = 0
    out = []
    for line in lines:
        stripped = line.rstrip("\n")
        m = OPEN_RE.match(stripped)
        if m:
            tag, conf = m.group(1), m.group(2)
            out.append(f"{tag}{REFUSAL};{conf}\n")
            fixed += 1
        else:
            out.append(line)
    if fixed:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text("".join(out), encoding="utf-8")
    return fixed


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    for p in paths:
        n = fix(p)
        print(f"  {p.name}: {n} empty answer{'s' if n != 1 else ''} replaced")


if __name__ == "__main__":
    main()
