#!/usr/bin/env python3
"""
Build a supplementary index from web-fetched movie data.

Pipeline:
  1. For each topic, identify the movie/entity being asked about.
  2. Fetch the Wikipedia article via the Wikipedia REST API (free, no key needed).
  3. Use Claude Haiku to enrich each article with:
     - A structured Q&A summary (5-10 facts about the movie)
     - Expanded passage text mentioning key details
  4. Segment into ≤200-char passages and add to a second BM25 index.
  5. Also build a second FAISS index from these passages.

This supplements the organizer corpus for topics where the answer is missing
(e.g., Interstellar, Monsters Inc. slogan, Love Actually soundtrack).

Usage:
    python scripts/build_web_index.py \
        --topics data/raw/r2c2topics.txt \
        --output-dir data/processed/web_index \
        --model claude-haiku-4-5-20251001
"""

import argparse
import json
import pickle
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

import anthropic
import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent.parent

IDENTIFY_PROMPT = """\
For the following question about a movie or Star Wars, identify:
1. The main movie title (exact Wikipedia article name if possible)
2. 2-3 key search terms to find the Wikipedia article

Question: {question}

Reply in JSON:
{{"movie_title": "...", "search_terms": ["...", "..."]}}"""

ENRICH_PROMPT = """\
You are enriching a Wikipedia passage about a movie for a question-answering system.

Movie: {title}
Wikipedia excerpt:
{excerpt}

Question we need to answer: {question}

Write 3-5 short factual statements (each ≤180 chars) that directly help answer the \
question, using specific facts from the excerpt or your knowledge. \
Focus on names, dates, numbers, and specific details.

One statement per line, no bullets or numbering."""

WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIPEDIA_SEARCH = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&format=json&srlimit=3"


def load_topics(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8")
    if next(l for l in content.splitlines() if l.strip()).startswith("<"):
        pairs = re.findall(r"<qID>\s*(\S+?)\s*</qID>.*?<q>(.*?)</q>", content, re.DOTALL)
        return {qid.strip(): q.strip() for qid, q in pairs}
    import csv
    topics = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tid = (row.get("topic_id") or row.get("qid") or "").strip()
            q = (row.get("question") or row.get("title") or "").strip()
            if tid and q:
                topics[tid] = q
    return topics


def call_claude(client, model, prompt, max_tokens=400, min_interval=1.4):
    for attempt in range(4):
        start = time.monotonic()
        try:
            r = client.messages.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            elapsed = time.monotonic() - start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            return r.content[0].text.strip()
        except Exception as e:
            wait = min(60, 5 * 2 ** attempt)
            print(f"  API error ({e}), retrying in {wait}s")
            time.sleep(wait)
    return ""


def wikipedia_fetch(title: str) -> str:
    """Fetch Wikipedia article summary via REST API."""
    url = WIKIPEDIA_API.format(title=urllib.parse.quote(title))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "R2C2-Research/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return data.get("extract", "")
    except Exception:
        return ""


def wikipedia_search(query: str) -> list[str]:
    """Search Wikipedia for article titles matching query."""
    url = WIKIPEDIA_SEARCH.format(q=urllib.parse.quote(query))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "R2C2-Research/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return [hit["title"] for hit in data["query"]["search"]]
    except Exception:
        return []


def segment(text: str, max_len: int = 180) -> list[str]:
    """Split text into ≤max_len char chunks on sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    buf = ""
    for sent in sentences:
        if not sent.strip():
            continue
        if len(buf) + len(sent) + 1 <= max_len:
            buf = (buf + " " + sent).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = sent[:max_len]
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) > 20]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default=str(BASE / "data/raw/r2c2topics.txt"))
    parser.add_argument("--output-dir", default=str(BASE / "data/processed/web_index"))
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--dense-model",
                        default="sentence-transformers/msmarco-MiniLM-L6-cos-v5")
    parser.add_argument("--cache", default=str(BASE / "data/processed/web_cache.json"),
                        help="Cache file for Wikipedia fetches and Haiku enrichments")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache)

    # Load cache
    cache = {}
    if cache_path.exists():
        with cache_path.open() as f:
            cache = json.load(f)
        print(f"Loaded cache with {len(cache)} entries")

    topics = load_topics(Path(args.topics))
    client = anthropic.Anthropic()

    all_passages = []  # list of {doc_id, text_snippet}

    print(f"\nProcessing {len(topics)} topics ...")
    for tid, question in sorted(topics.items()):
        print(f"\n[{tid}] {question[:70]}")

        cache_key = f"identify_{tid}"
        cached_meta = cache.get(cache_key)
        if cached_meta and isinstance(cached_meta, dict) and "movie_title" in cached_meta:
            meta = cached_meta
        else:
            raw = call_claude(client, args.model,
                              IDENTIFY_PROMPT.format(question=question))
            try:
                text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
                parsed = json.loads(text)
                # Handle case where Claude returns a list instead of a dict
                if isinstance(parsed, list):
                    meta = parsed[0] if parsed and isinstance(parsed[0], dict) else {}
                else:
                    meta = parsed if isinstance(parsed, dict) else {}
            except Exception:
                meta = {}
            if not isinstance(meta, dict) or "movie_title" not in meta:
                title_m = re.search(r'"movie_title"\s*:\s*"([^"]+)"', raw)
                meta = {"movie_title": title_m.group(1) if title_m else question,
                        "search_terms": [question]}
            cache[cache_key] = meta
            with cache_path.open("w") as f:
                json.dump(cache, f)

        movie_title = meta.get("movie_title", question)
        search_terms = meta.get("search_terms", [question])
        print(f"  Movie: {movie_title}")

        # Fetch Wikipedia content
        wiki_key = f"wiki_{movie_title}"
        if wiki_key in cache:
            wiki_text = cache[wiki_key]
        else:
            wiki_text = wikipedia_fetch(movie_title)
            if not wiki_text:
                # Try search
                candidates = wikipedia_search(movie_title)
                for cand in candidates[:2]:
                    wiki_text = wikipedia_fetch(cand)
                    if wiki_text:
                        print(f"  Found via search: {cand}")
                        break
            cache[wiki_key] = wiki_text
            with cache_path.open("w") as f:
                json.dump(cache, f)

        if not wiki_text:
            print(f"  No Wikipedia content found")
            continue

        print(f"  Wikipedia: {len(wiki_text)} chars")

        # Segment Wikipedia article into passages
        wiki_passages = segment(wiki_text)
        doc_id = f"web_{movie_title.replace(' ', '_')}"
        for p in wiki_passages:
            all_passages.append({"doc_id": doc_id, "text_snippet": p})

        # Enrich with Haiku: generate targeted factual statements
        enrich_key = f"enrich_{tid}"
        if enrich_key in cache:
            enriched = cache[enrich_key]
        else:
            prompt = ENRICH_PROMPT.format(
                title=movie_title,
                excerpt=wiki_text[:800],
                question=question,
            )
            raw_enrich = call_claude(client, args.model, prompt, max_tokens=512)
            enriched = [l.strip() for l in raw_enrich.splitlines()
                        if l.strip() and len(l.strip()) > 10]
            cache[enrich_key] = enriched
            with cache_path.open("w") as f:
                json.dump(cache, f)

        enrich_doc_id = f"enrich_{tid}_{movie_title.replace(' ', '_')}"
        for stmt in enriched:
            all_passages.append({"doc_id": enrich_doc_id, "text_snippet": stmt[:200]})
        print(f"  Added {len(wiki_passages)} wiki + {len(enriched)} enriched passages")

    print(f"\nTotal web passages: {len(all_passages)}")

    # Save metadata
    meta_path = out_dir / "web_meta.pkl"
    with meta_path.open("wb") as f:
        pickle.dump(all_passages, f)
    print(f"Metadata saved to {meta_path}")

    # Build BM25 index
    print("\nBuilding BM25S index over web passages ...")
    corpus_tokens = bm25s.tokenize(
        [p["text_snippet"] for p in all_passages],
        stopwords="en", stemmer=None, show_progress=True,
    )
    retriever = bm25s.BM25(method="bm25+", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=True)
    bm25_dir = out_dir / "bm25s_web_index"
    retriever.save(str(bm25_dir))
    print(f"BM25 index saved to {bm25_dir}")

    # Build FAISS index
    print(f"\nBuilding FAISS dense index with {args.dense_model} ...")
    model = SentenceTransformer(args.dense_model)
    texts = [p["text_snippet"] for p in all_passages]
    embeddings = model.encode(
        texts, batch_size=256, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss_path = out_dir / "faiss_web.bin"
    faiss.write_index(index, str(faiss_path))
    print(f"FAISS index saved to {faiss_path} ({index.ntotal} vectors)")

    print(f"\nWeb index complete. Use with autoresearch_loop.py --web-index {out_dir}")


if __name__ == "__main__":
    main()
