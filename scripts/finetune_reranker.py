#!/usr/bin/env python3
"""
Fine-tune the cross-encoder reranker with LoRA + pairwise margin loss.

Trains only LoRA adapters on query/value projections of the BERT backbone,
preserving pretrained MS MARCO knowledge while adapting to movie/entertainment
domain relevance patterns.

Data (held-out synthetic only — real test topics NEVER used):
  train : 70% of held-out synthetic topics
  dev   : 15% of held-out synthetic topics (early stopping / epoch selection)
  test  : 15% of held-out synthetic topics (final held-out eval, reported once)

Forgetting check: evaluates MRR@10 on a 200-query MS MARCO dev sample both
before and after fine-tuning to verify we haven't lost general retrieval quality.

Usage:
    python scripts/finetune_reranker.py \\
        --qrels data/synthetic_autoresearch_val65/qrels_loop_claude_haiku_4_5_20251001.json \\
        --val65  data/synthetic_val/synthetic_topics_val65.json \\
        --output data/processed/lora_reranker \\
        --base-model cross-encoder/ms-marco-MiniLM-L-12-v2 \\
        --epochs 5 --batch-size 32 --lora-rank 16
"""

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

BASE = Path(__file__).resolve().parent.parent


# ── Data loading ───────────────────────────────────────────────────────────────

def load_qrels(path: Path) -> dict:
    with path.open() as f:
        raw = json.load(f)
    return {
        tid: {tuple(k.split("\x00", 1)): int(v) for k, v in passages.items()}
        for tid, passages in raw.items()
    }


def load_topics_json(path: Path) -> dict[str, str]:
    with path.open() as f:
        records = json.load(f)
    return {
        (rec.get("loop_topic_id") or rec.get("topic_id", "")): rec.get("question", "")
        for rec in records
        if (rec.get("loop_topic_id") or rec.get("topic_id")) and rec.get("question")
    }


def build_pairwise_examples(qrels: dict, topics: dict, max_neg_per_pos: int = 5,
                             seed: int = 42) -> list[tuple[str, str, str]]:
    """
    Returns list of (query, positive_passage, negative_passage).
    For each topic, pairs every positive (score>=1) with up to max_neg_per_pos
    hard negatives (score==0, retrieved by BM25/dense so genuinely confusable).
    Passages with score=2 are weighted 2x by including them twice.
    """
    rng = random.Random(seed)
    examples = []
    for tid, passages in qrels.items():
        question = topics.get(tid, "")
        if not question:
            continue
        pos = [(text, score) for (_, text), score in passages.items() if score >= 1]
        neg = [text for (_, text), score in passages.items() if score == 0]
        if not pos or not neg:
            continue
        rng.shuffle(neg)
        hard_negs = neg[:max_neg_per_pos * len(pos)]
        neg_pool = hard_negs or neg
        for text, score in pos:
            # Score-2 passages appear twice (higher gradient weight)
            repeats = 2 if score == 2 else 1
            sampled_negs = rng.sample(neg_pool, min(max_neg_per_pos, len(neg_pool)))
            for neg_text in sampled_negs:
                for _ in range(repeats):
                    examples.append((question, text, neg_text))
    rng.shuffle(examples)
    return examples


# ── Dataset ────────────────────────────────────────────────────────────────────

class PairwiseDataset(Dataset):
    def __init__(self, examples: list, tokenizer, max_length: int = 256):
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        query, pos, neg = self.examples[idx]
        enc_pos = self.tokenizer(
            query, pos,
            max_length=self.max_length, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        enc_neg = self.tokenizer(
            query, neg,
            max_length=self.max_length, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        return {
            "pos_input_ids":      enc_pos["input_ids"].squeeze(0),
            "pos_attention_mask": enc_pos["attention_mask"].squeeze(0),
            "neg_input_ids":      enc_neg["input_ids"].squeeze(0),
            "neg_attention_mask": enc_neg["attention_mask"].squeeze(0),
        }


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def score_pairs(model, tokenizer, pairs: list[tuple[str, str]],
                batch_size: int = 64, device: str = "cpu") -> list[float]:
    """Score (query, passage) pairs and return raw logits."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i: i + batch_size]
            queries  = [q for q, _ in batch]
            passages = [p for _, p in batch]
            enc = tokenizer(
                queries, passages,
                max_length=256, truncation=True, padding=True,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            scores.extend(out.logits.squeeze(-1).cpu().tolist())
    return scores


def eval_ndcg(model, tokenizer, qrels: dict, topics: dict,
              device: str = "cpu", k: int = 20) -> float:
    """Compute mean nDCG@k on a qrels dict."""
    def dcg(rels, k):
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))

    topic_ndcgs = []
    for tid, passages in qrels.items():
        question = topics.get(tid, "")
        if not question:
            continue
        items = list(passages.items())
        if not items:
            continue
        pairs  = [(question, text) for (_, text), _ in items]
        labels = [score for (_, _), score in items]
        raw_scores = score_pairs(model, tokenizer, pairs, device=device)
        ranked = [lbl for _, lbl in sorted(zip(raw_scores, labels), reverse=True)]
        ideal  = sorted(labels, reverse=True)
        ideal_dcg = dcg(ideal, k)
        topic_ndcg = dcg(ranked, k) / ideal_dcg if ideal_dcg > 0 else 0.0
        topic_ndcgs.append(topic_ndcg)
    return float(np.mean(topic_ndcgs)) if topic_ndcgs else 0.0


def eval_msmarco_mrr(model, tokenizer, device: str = "cpu",
                     n_queries: int = 200, seed: int = 42) -> float:
    """
    MRR@10 on a small MS MARCO dev sample — forgetting check.
    Uses the cached HuggingFace dataset (no download if already cached).
    Returns -1.0 if dataset unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("ms_marco", "v1.1", split="validation",
                          trust_remote_code=True)
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), min(n_queries, len(ds)))
        sample = [ds[i] for i in indices]

        reciprocal_ranks = []
        for item in sample:
            query    = item["query"]
            passages = item["passages"]["passage_text"]
            labels   = item["passages"]["is_selected"]
            if sum(labels) == 0:
                continue
            pairs  = [(query, p) for p in passages]
            scores = score_pairs(model, tokenizer, pairs, device=device)
            ranked = [lbl for _, lbl in sorted(zip(scores, labels), reverse=True)]
            for rank, rel in enumerate(ranked[:10], 1):
                if rel:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    except Exception as e:
        print(f"  MS MARCO eval skipped: {e}")
        return -1.0


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, margin: float = 1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        pos_ids  = batch["pos_input_ids"].to(device)
        pos_mask = batch["pos_attention_mask"].to(device)
        neg_ids  = batch["neg_input_ids"].to(device)
        neg_mask = batch["neg_attention_mask"].to(device)

        pos_scores = model(input_ids=pos_ids, attention_mask=pos_mask).logits.squeeze(-1)
        neg_scores = model(input_ids=neg_ids, attention_mask=neg_mask).logits.squeeze(-1)

        # Margin ranking loss: max(0, margin - (pos - neg))
        loss = F.margin_ranking_loss(
            pos_scores, neg_scores,
            target=torch.ones_like(pos_scores),
            margin=margin,
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels",       default=str(BASE / "data/synthetic_autoresearch_val65/qrels_loop_claude_haiku_4_5_20251001.json"))
    parser.add_argument("--val65",       default=str(BASE / "data/synthetic_val/synthetic_topics_val65.json"),
                        help="Val65 topic IDs to exclude from training (used in autoresearch loop)")
    parser.add_argument("--topics-file", default=str(BASE / "data/synthetic_val/synthetic_topics_loop.json"),
                        help="Synthetic topics file for question text lookup (needs loop_topic_id → question)")
    parser.add_argument("--base-model",  default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--output",      default=str(BASE / "data/processed/lora_reranker"))
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lora-rank",   type=int,   default=16)
    parser.add_argument("--lora-alpha",  type=int,   default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--margin",      type=float, default=1.0)
    parser.add_argument("--max-neg-per-pos", type=int, default=5)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--max-length",  type=int,   default=256)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--msmarco-queries", type=int, default=200,
                        help="Number of MS MARCO dev queries for forgetting check (0=skip)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load topics ────────────────────────────────────────────────────────────
    print("Loading topics ...")
    topics = load_topics_json(Path(args.topics_file))
    print(f"  {len(topics)} topics loaded")

    # ── Load qrels — exclude val65 and real test topics ────────────────────────
    print("Loading qrels ...")
    all_qrels = load_qrels(Path(args.qrels))

    val65_ids = set(load_topics_json(Path(args.val65)).keys())
    held_out = {
        tid: v for tid, v in all_qrels.items()
        if tid.startswith("L") and tid not in val65_ids
    }
    print(f"  Held-out synthetic topics: {len(held_out)}")

    # ── Train / dev / test split by topic ──────────────────────────────────────
    topic_ids = sorted(held_out.keys())
    random.shuffle(topic_ids)
    n = len(topic_ids)
    n_train = int(0.70 * n)
    n_dev   = int(0.15 * n)
    train_ids = set(topic_ids[:n_train])
    dev_ids   = set(topic_ids[n_train: n_train + n_dev])
    test_ids  = set(topic_ids[n_train + n_dev:])

    train_qrels = {tid: held_out[tid] for tid in train_ids}
    dev_qrels   = {tid: held_out[tid] for tid in dev_ids}
    test_qrels  = {tid: held_out[tid] for tid in test_ids}
    print(f"  Train: {len(train_qrels)} topics | Dev: {len(dev_qrels)} | Test: {len(test_qrels)}")

    # ── Build pairwise training examples ──────────────────────────────────────
    print("Building pairwise training examples ...")
    train_examples = build_pairwise_examples(
        train_qrels, topics, max_neg_per_pos=args.max_neg_per_pos, seed=args.seed
    )
    print(f"  {len(train_examples)} pairwise examples")

    # ── Load tokenizer and model ───────────────────────────────────────────────
    print(f"Loading base model: {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1
    )

    # ── Apply LoRA ─────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],   # BERT attention projections
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # ── Baseline evaluation (before any training) ──────────────────────────────
    print("\n── Baseline (before fine-tuning) ─────────────────────────────────")
    base_dev_ndcg  = eval_ndcg(model, tokenizer, dev_qrels,  topics, device)
    base_test_ndcg = eval_ndcg(model, tokenizer, test_qrels, topics, device)
    print(f"  Dev  nDCG@20: {base_dev_ndcg:.4f}")
    print(f"  Test nDCG@20: {base_test_ndcg:.4f}")
    if args.msmarco_queries > 0:
        print(f"  Evaluating MS MARCO ({args.msmarco_queries} queries) ...")
        base_msmarco_mrr = eval_msmarco_mrr(model, tokenizer, device,
                                             n_queries=args.msmarco_queries,
                                             seed=args.seed)
        if base_msmarco_mrr >= 0:
            print(f"  MS MARCO MRR@10: {base_msmarco_mrr:.4f}")
    else:
        base_msmarco_mrr = -1.0

    # ── DataLoader ─────────────────────────────────────────────────────────────
    dataset    = PairwiseDataset(train_examples, tokenizer, args.max_length)
    loader     = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, pin_memory=(device == "cuda"))

    optimizer  = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps   = len(loader) * args.epochs
    warmup_steps  = max(1, total_steps // 10)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\n── Training ({args.epochs} epochs, batch={args.batch_size}, "
          f"lr={args.lr}, rank={args.lora_rank}) ─────")
    best_dev_ndcg   = base_dev_ndcg
    best_epoch      = 0
    history         = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, scheduler,
                                 device, margin=args.margin)
        dev_ndcg   = eval_ndcg(model, tokenizer, dev_qrels, topics, device)
        print(f"  Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
              f"dev nDCG@20={dev_ndcg:.4f}", end="")

        if dev_ndcg > best_dev_ndcg:
            best_dev_ndcg = dev_ndcg
            best_epoch    = epoch
            model.save_pretrained(str(out_dir / "best_adapter"))
            tokenizer.save_pretrained(str(out_dir / "best_adapter"))
            print("  ✓ saved", end="")
        print()
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "dev_ndcg": dev_ndcg})

    print(f"\nBest epoch: {best_epoch}  dev nDCG@20={best_dev_ndcg:.4f}")

    # ── Final evaluation on held-out test set ─────────────────────────────────
    print("\n── Final evaluation (held-out test set) ──────────────────────────")
    if best_epoch > 0:
        # Reload best checkpoint for test eval
        best_model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=1
        )
        best_model = PeftModel.from_pretrained(
            best_model, str(out_dir / "best_adapter")
        )
        best_model = best_model.to(device)
    else:
        best_model = model  # no improvement over baseline

    final_test_ndcg = eval_ndcg(best_model, tokenizer, test_qrels, topics, device)
    print(f"  Test nDCG@20:  baseline={base_test_ndcg:.4f}  "
          f"fine-tuned={final_test_ndcg:.4f}  "
          f"delta={final_test_ndcg - base_test_ndcg:+.4f}")

    if args.msmarco_queries > 0:
        print(f"  Checking MS MARCO forgetting ({args.msmarco_queries} queries) ...")
        ft_msmarco_mrr = eval_msmarco_mrr(best_model, tokenizer, device,
                                           n_queries=args.msmarco_queries,
                                           seed=args.seed)
        if ft_msmarco_mrr >= 0 and base_msmarco_mrr >= 0:
            print(f"  MS MARCO MRR@10: baseline={base_msmarco_mrr:.4f}  "
                  f"fine-tuned={ft_msmarco_mrr:.4f}  "
                  f"delta={ft_msmarco_mrr - base_msmarco_mrr:+.4f}")
            if ft_msmarco_mrr < base_msmarco_mrr - 0.02:
                print("  ⚠ WARNING: MRR dropped >2pp — possible catastrophic forgetting")
            else:
                print("  ✓ No significant forgetting detected")
    else:
        ft_msmarco_mrr = -1.0

    # ── Save merged model ──────────────────────────────────────────────────────
    merged_dir = out_dir / "merged"
    print(f"\nSaving merged model to {merged_dir} ...")
    merged = best_model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print("  Done.")

    # ── Save training report ───────────────────────────────────────────────────
    report = {
        "base_model":         args.base_model,
        "lora_rank":          args.lora_rank,
        "lora_alpha":         args.lora_alpha,
        "epochs":             args.epochs,
        "best_epoch":         best_epoch,
        "train_examples":     len(train_examples),
        "train_topics":       len(train_qrels),
        "dev_topics":         len(dev_qrels),
        "test_topics":        len(test_qrels),
        "baseline": {
            "dev_ndcg20":     base_dev_ndcg,
            "test_ndcg20":    base_test_ndcg,
            "msmarco_mrr10":  base_msmarco_mrr,
        },
        "fine_tuned": {
            "dev_ndcg20":     best_dev_ndcg,
            "test_ndcg20":    final_test_ndcg,
            "msmarco_mrr10":  ft_msmarco_mrr,
        },
        "epoch_history":      history,
    }
    with (out_dir / "training_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Training report saved to {out_dir / 'training_report.json'}")
    print(f"\nTo use in autoresearch_loop.py:")
    print(f"  --cross-encoder {merged_dir}")


if __name__ == "__main__":
    main()
