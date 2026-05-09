#!/usr/bin/env python3
"""
LoRA fine-tune the bi-encoder on domain-specific movie retrieval triplets.

Architecture:
  Base model: sentence-transformers/msmarco-MiniLM-L6-cos-v5 (22M params)
  LoRA: r=16, alpha=32 on q+v projections → ~180K trainable params
  Loss: MultipleNegativesRankingLoss (in-batch negatives + explicit hard negatives)
  Epochs: 3, batch_size: 32

Input: data/processed/ft_triplets.jsonl
  {"query": "...", "positive": "...", "hard_negatives": ["...", ...], ...}

Output: data/processed/lora_biencoder/  (merged SentenceTransformer, ready to use)

Usage:
    python scripts/finetune_biencoder_lora.py
    python scripts/finetune_biencoder_lora.py --epochs 5 --batch-size 64
    python scripts/finetune_biencoder_lora.py --dry-run   # 50 examples, 1 epoch
"""

import argparse
import json
import random
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent


def load_triplets(path: Path, max_hard_negatives: int = 2) -> list[dict]:
    triplets = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("query") and rec.get("positive"):
                triplets.append({
                    "query": rec["query"],
                    "positive": rec["positive"],
                    "hard_negatives": rec.get("hard_negatives", [])[:max_hard_negatives],
                    "challenge_type": rec.get("challenge_type", ""),
                })
    return triplets


def build_examples(triplets: list[dict], include_hard_negatives: bool = True):
    """
    Build InputExample list.
    sentence-transformers MNR loss format:
      texts=[query, positive]         — in-batch negatives only
      texts=[query, positive, neg1, neg2, ...]  — + explicit hard negatives
    """
    from sentence_transformers import InputExample
    examples = []
    for t in triplets:
        if include_hard_negatives and t["hard_negatives"]:
            texts = [t["query"], t["positive"]] + t["hard_negatives"]
        else:
            texts = [t["query"], t["positive"]]
        examples.append(InputExample(texts=texts))
    return examples


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.1):
    """Apply LoRA adapters to q+v projections in all attention layers."""
    from peft import get_peft_model, LoraConfig, TaskType

    # Detect target modules: MiniLM uses "query" and "value" naming
    # Verify by inspecting named modules
    transformer = model._first_module().auto_model
    named = [n for n, _ in transformer.named_modules()]
    # MiniLM: encoder.layer.N.attention.self.{query,value}
    if any("attention.self.query" in n for n in named):
        target_modules = ["query", "value"]
    elif any(".q_proj" in n for n in named):
        target_modules = ["q_proj", "v_proj"]
    else:
        target_modules = ["query", "value"]  # fallback

    print(f"  LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    peft_model = get_peft_model(transformer, lora_config)
    peft_model.print_trainable_parameters()
    model._first_module().auto_model = peft_model
    return model, peft_model


def merge_lora(model, peft_model):
    """Merge LoRA adapters into base weights for clean SentenceTransformer save."""
    merged = peft_model.merge_and_unload()
    model._first_module().auto_model = merged
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets",
                        default=str(BASE / "data/processed/ft_triplets.jsonl"))
    parser.add_argument("--base-model",
                        default="sentence-transformers/msmarco-MiniLM-L6-cos-v5")
    parser.add_argument("--output",
                        default=str(BASE / "data/processed/lora_biencoder"))
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-hard-negatives", type=int, default=2,
                        help="Hard negatives per example to include")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use 50 examples, 1 epoch — quick sanity check")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(16)

    # Imports here to avoid slow startup when just checking args
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    print(f"Loading triplets from {args.triplets} ...")
    triplets = load_triplets(Path(args.triplets), args.max_hard_negatives)
    print(f"  {len(triplets)} triplets loaded")

    if args.dry_run:
        print("  [DRY RUN] Using 50 examples, 1 epoch")
        triplets = triplets[:50]
        args.epochs = 1

    random.shuffle(triplets)

    # Challenge type distribution
    from collections import Counter
    ct_dist = Counter(t["challenge_type"] for t in triplets)
    print(f"  Challenge types: {dict(ct_dist.most_common(5))} ...")
    hard_neg_pct = sum(1 for t in triplets if t["hard_negatives"]) / len(triplets) * 100
    print(f"  Triplets with hard negatives: {hard_neg_pct:.0f}%")

    print(f"\nLoading base model: {args.base_model} ...")
    model = SentenceTransformer(args.base_model)

    print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha}) ...")
    model, peft_model = apply_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    examples = build_examples(triplets, include_hard_negatives=True)
    print(f"\nBuilt {len(examples)} training examples")

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size,
                                  drop_last=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)
    total_steps = len(train_dataloader) * args.epochs

    print(f"Training: {args.epochs} epochs × {len(train_dataloader)} steps "
          f"= {total_steps} total steps")
    print(f"  batch_size={args.batch_size}, warmup={warmup_steps} steps")
    print(f"  device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Train (sentence-transformers v3 API)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        checkpoint_path=str(output_path / "checkpoints"),
        checkpoint_save_steps=max(100, len(train_dataloader)),
    )

    print("\nMerging LoRA adapters into base weights ...")
    model = merge_lora(model, peft_model)

    print(f"Saving merged SentenceTransformer to {output_path} ...")
    model.save(str(output_path))

    # Quick sanity check
    print("\nSanity check — encoding two passages ...")
    test_q = "Who directed this acclaimed science fiction film set in space?"
    test_pos = "The film was directed by Stanley Kubrick and released in 1968."
    test_neg = "The box office gross for the opening weekend was $2.1 million."
    embs = model.encode([test_q, test_pos, test_neg], normalize_embeddings=True)
    sim_pos = float(embs[0] @ embs[1])
    sim_neg = float(embs[0] @ embs[2])
    print(f"  sim(q, positive) = {sim_pos:.3f}")
    print(f"  sim(q, negative) = {sim_neg:.3f}")
    print(f"  margin = {sim_pos - sim_neg:.3f} {'✓' if sim_pos > sim_neg else '✗'}")

    print(f"\nDone. Fine-tuned model at: {output_path}")
    print("Next step — rebuild FAISS index:")
    print(f"  python scripts/build_own_index.py --dense-only --model {output_path} \\")
    print(f"    --out-dir data/processed/own_passages_lora")


if __name__ == "__main__":
    main()
