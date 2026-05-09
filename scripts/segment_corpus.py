#!/usr/bin/env python3
"""
Segment corpus documents into passages of <=200 chars.

Input JSONL:  {doc_id, url, text}
Output JSONL: {doc_id, url, passage_rank, passage_text}
"""

import argparse
import json
import re
import sys
from pathlib import Path


MAX_CHARS = 200


def segment_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """
    Split text into passages of at most max_chars characters.
    - Collapses all whitespace/newlines into single spaces first.
    - Breaks only on word boundaries (spaces).
    """
    # Collapse newlines and extra whitespace into single spaces
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    passages = []
    while text:
        if len(text) <= max_chars:
            passages.append(text)
            break
        # Find the last space within the limit
        cut = text.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            # No space found — force-cut at max_chars (shouldn't happen for normal text)
            cut = max_chars
        passages.append(text[:cut])
        text = text[cut:].lstrip()

    return passages


def main():
    parser = argparse.ArgumentParser(description="Segment corpus into <=200-char passages")
    parser.add_argument("input", help="Input JSONL corpus file (doc_id, url, text)")
    parser.add_argument("output", help="Output JSONL passages file")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS,
                        help=f"Max passage length in characters (default: {MAX_CHARS})")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_passages = 0

    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            url = doc.get("url", "")
            text = doc.get("text", "")

            passages = segment_text(text, args.max_chars)
            for rank, passage in enumerate(passages, start=1):
                record = {
                    "doc_id": doc_id,
                    "url": url,
                    "passage_rank": rank,
                    "passage_text": passage,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_docs += 1
            total_passages += len(passages)

    avg = total_passages / total_docs if total_docs else 0
    print(f"Total docs:              {total_docs:,}")
    print(f"Total passages:          {total_passages:,}")
    print(f"Avg passages per doc:    {avg:.2f}")
    print(f"Output written to:       {output_path}")


if __name__ == "__main__":
    main()
