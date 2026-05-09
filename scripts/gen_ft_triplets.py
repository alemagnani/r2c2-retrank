#!/usr/bin/env python3
"""
Generate fine-tuning triplets for bi-encoder LoRA training.

For each sampled movie article:
  1. Extract sliding-window passages (same 200-char/stride-100 as index)
  2. Single Haiku call: generate question + identify positive passage + suggest confusable movies
  3. Look up confusable movies in corpus title index → pull their passages as hard negatives
  4. Fallback: intra-doc hard negatives if confusable movies not found

Output: data/processed/ft_triplets.jsonl
  {"query": "...", "positive": "...", "hard_negatives": ["...", ...],
   "challenge_type": "...", "source_doc_id": "...", "confusable_movies": [...]}

Target: ~3500 triplets from ~700 docs (5 questions per doc)

Usage:
    python scripts/gen_ft_triplets.py
    python scripts/gen_ft_triplets.py --n-docs 700 --n-per-doc 5 --resume
    python scripts/gen_ft_triplets.py --n-docs 100 --n-per-doc 2  # quick test
"""

import argparse
import json
import pickle
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent

MAX_LEN = 200
STRIDE = 100
MIN_LEN = 50
MIN_DOC_LEN = 400   # skip very short articles

# Patterns that indicate a movie/film article (checked against first passage)
MOVIE_PATTERNS = re.compile(
    r"\b(?:film|movie|animated film|documentary|short film|feature film|"
    r"directed by|screenplay|cinematograph|box office|released in \d{4}|"
    r"\d{4} .{0,30} film)\b",
    re.IGNORECASE,
)
MAX_ARTICLE_CHARS = 4000  # truncate for prompt

CHALLENGE_TYPES = [
    "plot_detail",
    "character_name",
    "actor_identity",
    "production_detail",
    "director_writer",
    "multi_hop",
    "vocabulary_gap",
    "implicit_entity",
    "real_vs_fictional",
    "release_order",
    "count_number",
    "award_detail",
    "music_detail",
    "slogan_quote",
    "voice_actor",
    "cameo_appearance",
    "box_office_rank",
    "alternate_title",
    "casting_change",
]

# Weight toward types that are hardest for dense retrieval
CHALLENGE_WEIGHTS = {
    "vocabulary_gap": 3,
    "implicit_entity": 3,
    "multi_hop": 3,
    "real_vs_fictional": 2,
    "alternate_title": 2,
    "casting_change": 2,
    "plot_detail": 2,
    "character_name": 2,
    "actor_identity": 2,
    "production_detail": 2,
    "director_writer": 1,
    "release_order": 1,
    "count_number": 1,
    "award_detail": 1,
    "music_detail": 1,
    "slogan_quote": 1,
    "voice_actor": 1,
    "cameo_appearance": 1,
    "box_office_rank": 1,
}

GENERATION_PROMPT = """\
You are creating training data for a movie information retrieval system.

Article title (movie): {title}
Article text:
{article_text}

Candidate passages (already segmented to ≤200 characters each, numbered):
{passages_list}

Your tasks:
1. Pick a CHALLENGE TYPE from this list: {challenge_types}
2. Write a natural English question of that challenge type about this movie.
   The question should be answerable from the article and should NOT mention the movie title directly if possible.
3. Identify which passage number BEST answers the question. The passage text must appear VERBATIM in the list above.
4. Suggest 2-3 REAL movie titles that would be commonly confused with this movie
   (same genre, era, director, similar cast or plot — movies that a retrieval system might confuse).

Reply with JSON only:
{{
  "challenge_type": "<type from list>",
  "question": "<natural question>",
  "positive_passage_num": <integer, 1-based>,
  "positive_passage_text": "<exact text from the numbered list>",
  "confusable_movies": ["Movie Title A", "Movie Title B", "Movie Title C"],
  "reasoning": "<one sentence why this passage answers the question>"
}}"""


def sliding_window_passages(text: str, max_len: int = MAX_LEN,
                             stride: int = STRIDE, min_len: int = MIN_LEN) -> list[str]:
    if not text:
        return []
    words = text.split()
    passages = []
    stride_words = max(1, stride // 6)
    i = 0
    while i < len(words):
        chunk, length, j = [], 0, i
        while j < len(words) and length + len(words[j]) + 1 <= max_len:
            chunk.append(words[j])
            length += len(words[j]) + 1
            j += 1
        if j == i:
            j = i + 1
        p = " ".join(chunk).strip()
        if len(p) >= min_len:
            passages.append(p)
        i += stride_words
    return list(dict.fromkeys(passages))


def extract_title(first_passage: str) -> str:
    """Best-effort title from first passage text."""
    text = first_passage.strip()
    # Remove parenthetical disambiguation
    text = re.sub(r"\s*\([^)]*\)", "", text)
    # Take up to first "is a", "was a", " – ", " - ", or 60 chars
    for pat in [r"^(.+?)\s+(?:is|was)\s+(?:a|an|the)\b",
                r"^(.+?)\s+[–\-]\s",
                r"^(.{5,50}?)[,.]"]:
        m = re.match(pat, text)
        if m:
            return m.group(1).strip()
    return text[:50].strip()


def build_title_index(meta: list[dict]) -> tuple[dict, dict]:
    """
    Returns:
      doc_texts: doc_id -> full reconstructed text
      title_index: normalized_title -> doc_id
    """
    by_doc = defaultdict(list)
    for m in meta:
        by_doc[m["doc_id"]].append(m)

    doc_texts = {}
    title_index = {}

    for doc_id, chunks in by_doc.items():
        sorted_chunks = sorted(chunks, key=lambda x: x["chunk_id"])
        full_text = " ".join(c["text_snippet"] for c in sorted_chunks)
        doc_texts[doc_id] = full_text

        title = extract_title(sorted_chunks[0]["text_snippet"])
        key = re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()
        if key:
            title_index[key] = doc_id

    return doc_texts, title_index


def lookup_movie(title: str, title_index: dict) -> str | None:
    """Fuzzy title lookup: exact → strip articles → word subset match."""
    key = re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()
    if key in title_index:
        return title_index[key]
    # Strip leading articles
    stripped = re.sub(r"^(?:the|a|an) ", "", key).strip()
    if stripped in title_index:
        return title_index[stripped]
    # Allow "part ii" → "part 2" normalization
    normalized = key.replace(" ii", " 2").replace(" iii", " 3").replace(" iv", " 4")
    if normalized in title_index:
        return title_index[normalized]
    # Word subset: all words of lookup key present in some index key
    words = set(key.split())
    if len(words) >= 2:
        for idx_key, doc_id in title_index.items():
            if words.issubset(set(idx_key.split())):
                return doc_id
    return None


def get_hard_negatives_from_doc(doc_id: str, doc_texts: dict,
                                positive_text: str, n: int = 3) -> list[str]:
    """Pull passages from same doc that are NOT the positive — intra-doc hard negatives."""
    passages = sliding_window_passages(doc_texts.get(doc_id, ""))
    candidates = [p for p in passages if p != positive_text and len(p) >= MIN_LEN]
    random.shuffle(candidates)
    return candidates[:n]


def call_haiku(client: anthropic.Anthropic, model: str, prompt: str,
               retries: int = 4) -> dict | None:
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            wait = min(60, 5 * 2 ** attempt)
            print(f"  API error: {e}, retry in {wait}s")
            time.sleep(wait)
    return None


def weighted_challenge_types(n: int) -> list[str]:
    """Sample n challenge types according to CHALLENGE_WEIGHTS."""
    pool = []
    for ct in CHALLENGE_TYPES:
        pool.extend([ct] * CHALLENGE_WEIGHTS.get(ct, 1))
    return random.choices(pool, k=n)


def process_doc(doc_id: str, doc_text: str, title: str,
                client: anthropic.Anthropic, model: str,
                title_index: dict, doc_texts: dict,
                n_questions: int) -> list[dict]:
    """Generate n_questions triplets for one doc. Returns list of valid triplets."""
    passages = sliding_window_passages(doc_text)
    if len(passages) < 5:
        return []

    # Cap article for prompt
    article_for_prompt = doc_text[:MAX_ARTICLE_CHARS]

    # Build numbered passage list for prompt (cap at 40 passages)
    display_passages = passages[:40]
    passages_list = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(display_passages))
    passage_set = set(passages)  # for validation

    challenge_type_list = ", ".join(CHALLENGE_TYPES)
    triplets = []
    used_challenge_types = set()

    for _ in range(n_questions):
        prompt = GENERATION_PROMPT.format(
            title=title,
            article_text=article_for_prompt,
            passages_list=passages_list,
            challenge_types=challenge_type_list,
        )

        data = call_haiku(client, model, prompt)
        if not data:
            continue

        question = data.get("question", "").strip()
        challenge_type = data.get("challenge_type", "").strip()
        positive_text = data.get("positive_passage_text", "").strip()
        confusable_movies = data.get("confusable_movies", [])

        # Validate
        if not question or not positive_text:
            continue
        if positive_text not in passage_set:
            # Try to match by position
            pnum = data.get("positive_passage_num", 0)
            if isinstance(pnum, int) and 1 <= pnum <= len(display_passages):
                positive_text = display_passages[pnum - 1]
            else:
                continue  # can't recover

        # Avoid duplicate challenge types per doc
        if challenge_type in used_challenge_types:
            time.sleep(1.0)
            continue
        used_challenge_types.add(challenge_type)

        # Find hard negatives from confusable movies
        hard_negatives = []
        found_confusable = []
        for movie_title in confusable_movies[:3]:
            conf_doc_id = lookup_movie(movie_title, title_index)
            if conf_doc_id and conf_doc_id != doc_id:
                conf_passages = sliding_window_passages(doc_texts.get(conf_doc_id, ""))
                if conf_passages:
                    # Pick a passage that's somewhat long (more confusable)
                    candidates = [p for p in conf_passages if len(p) >= 80]
                    if candidates:
                        hard_negatives.append(random.choice(candidates))
                        found_confusable.append(movie_title)

        # Fallback: intra-doc hard negatives
        if len(hard_negatives) < 2:
            intra = get_hard_negatives_from_doc(doc_id, doc_texts, positive_text,
                                                n=3 - len(hard_negatives))
            hard_negatives.extend(intra)

        if not hard_negatives:
            continue

        triplets.append({
            "query": question,
            "positive": positive_text,
            "hard_negatives": hard_negatives[:3],
            "challenge_type": challenge_type,
            "source_doc_id": doc_id,
            "source_title": title,
            "confusable_movies_suggested": confusable_movies,
            "confusable_movies_found": found_confusable,
            "reasoning": data.get("reasoning", ""),
        })

        time.sleep(1.2)  # rate limiting

    return triplets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=str(BASE / "data/processed/passages_meta.pkl"))
    parser.add_argument("--output", default=str(BASE / "data/processed/ft_triplets.jsonl"))
    parser.add_argument("--cache", default=str(BASE / "data/processed/ft_triplets_cache.json"))
    parser.add_argument("--n-docs", type=int, default=700, help="Number of docs to sample")
    parser.add_argument("--n-per-doc", type=int, default=5, help="Questions per doc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed docs")
    parser.add_argument("--min-doc-len", type=int, default=MIN_DOC_LEN)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading corpus metadata from {args.meta} ...")
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    print(f"  {len(meta):,} passages")

    print("Building doc texts and title index ...")
    doc_texts, title_index = build_title_index(meta)
    print(f"  {len(doc_texts):,} docs, {len(title_index):,} title entries")

    # Filter to docs with sufficient text AND likely movie articles
    # (first passage must mention film/movie/directed-by/etc.)
    by_doc_first = {}
    for m in meta:
        if m["doc_id"] not in by_doc_first:
            by_doc_first[m["doc_id"]] = m["text_snippet"]

    good_docs = [
        (doc_id, text) for doc_id, text in doc_texts.items()
        if len(text) >= args.min_doc_len
        and MOVIE_PATTERNS.search(by_doc_first.get(doc_id, "")[:300])
    ]
    print(f"  {len(good_docs):,} likely movie docs with >= {args.min_doc_len} chars")

    # Sample docs
    random.shuffle(good_docs)
    sampled = good_docs[:args.n_docs]
    print(f"  Sampling {len(sampled)} docs (seed={args.seed})")

    # Load resume cache
    cache_path = Path(args.cache)
    cache: dict[str, list] = {}
    if args.resume and cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"  Resuming: {len(cache)} docs already processed")

    client = anthropic.Anthropic()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_triplets = []
    # Load existing output if resuming
    if args.resume and output_path.exists():
        with output_path.open() as f:
            for line in f:
                all_triplets.append(json.loads(line))
        print(f"  Loaded {len(all_triplets)} existing triplets")

    total = len(sampled)
    new_triplets = 0

    with output_path.open("a" if args.resume else "w") as fout:
        for i, (doc_id, doc_text) in enumerate(sampled):
            if doc_id in cache:
                continue

            # Extract title from first passage (already cached in by_doc_first)
            title = extract_title(by_doc_first.get(doc_id, doc_id))

            triplets = process_doc(
                doc_id, doc_text, title,
                client, args.model,
                title_index, doc_texts,
                args.n_per_doc,
            )

            cache[doc_id] = len(triplets)
            for t in triplets:
                fout.write(json.dumps(t) + "\n")
                new_triplets += 1

            # Save cache periodically
            if (i + 1) % 20 == 0:
                cache_path.write_text(json.dumps(cache, indent=2))

            found_pct = sum(
                1 for t in triplets if t["confusable_movies_found"]
            ) / max(len(triplets), 1) * 100
            print(f"  [{i+1}/{total}] {title[:40]:<40} "
                  f"{len(triplets)} triplets "
                  f"(conf_found={found_pct:.0f}%)")

    cache_path.write_text(json.dumps(cache, indent=2))

    total_triplets = len(all_triplets) + new_triplets if args.resume else new_triplets
    print(f"\nDone. {total_triplets} total triplets written to {output_path}")
    print(f"  New this run: {new_triplets}")
    print(f"  Docs processed: {len(cache)}")

    # Quick stats
    if output_path.exists():
        triplets_all = []
        with output_path.open() as f:
            for line in f:
                triplets_all.append(json.loads(line))
        from collections import Counter
        ct_counts = Counter(t["challenge_type"] for t in triplets_all)
        conf_found = sum(1 for t in triplets_all if t["confusable_movies_found"])
        print(f"\nChallenge type distribution ({len(triplets_all)} triplets):")
        for ct, cnt in ct_counts.most_common():
            print(f"  {ct:<25} {cnt:>4}")
        print(f"Confusable movie found (corpus): {conf_found}/{len(triplets_all)} "
              f"({conf_found/max(len(triplets_all),1)*100:.0f}%)")


if __name__ == "__main__":
    main()
