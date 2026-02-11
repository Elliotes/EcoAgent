#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train RATED encoder without contrastive learning.
Uses supervised CosineSimilarityLoss with explicit positive/negative pairs (no InfoNCE).
For ablation to measure contrastive learning contribution.
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_encoder_supervised(
    pairs_df: pd.DataFrame,
    tools_df: pd.DataFrame,
    base_model: str,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 1000,
    use_fp16: bool = False,
    max_seq_len: int = 256,
    use_gradient_checkpointing: bool = False,
    neg_per_pos: int = 1
):
    print("\n" + "=" * 60)
    print("Train Encoder (supervised, no InfoNCE contrastive)")
    print("=" * 60)
    print(f"[Encoder] Loading base model: {base_model}")
    encoder = SentenceTransformer(base_model)
    encoder.max_seq_length = max_seq_len
    d = encoder.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {d}, max_seq_length: {max_seq_len}")
    if use_gradient_checkpointing:
        try:
            encoder._first_module().auto_model.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled")
        except Exception as exc:
            print(f"  Warning: could not enable gradient checkpointing: {exc}")
    print("\n[Data] Preparing training data (positive and negative pairs)...")
    uid_col = "uid" if "uid" in tools_df.columns else tools_df.columns[0]
    text_col = "tool_text" if "tool_text" in tools_df.columns else "description"
    
    uid2text = dict(zip(
        tools_df[uid_col].astype(str),
        tools_df[text_col].astype(str)
    ))
    pairs_with_text = pairs_df.copy()
    pairs_with_text["tool_text"] = pairs_with_text["tool_uid"].astype(str).map(uid2text)
    pairs_with_text = pairs_with_text.dropna(subset=["tool_text", "query"]).reset_index(drop=True)
    positive_pairs = pairs_with_text[pairs_with_text["label"] == 1]
    negative_pairs = pairs_with_text[pairs_with_text["label"] == 0]
    print(f"  Positive pairs: {len(positive_pairs)}, negative: {len(negative_pairs)}")
    if len(negative_pairs) > len(positive_pairs) * neg_per_pos:
        n_neg = len(positive_pairs) * neg_per_pos
        print(f"  Sampling {n_neg} negative pairs")
        negative_pairs = negative_pairs.sample(n=n_neg, random_state=42)
    train_examples = []
    for _, row in tqdm(positive_pairs.iterrows(), total=len(positive_pairs), desc="Positive samples"):
        train_examples.append(InputExample(
            texts=[f"query: {row['query']}", f"passage: {row['tool_text']}"],
            label=1.0
        ))
    for _, row in tqdm(negative_pairs.iterrows(), total=len(negative_pairs), desc="Negative samples"):
        train_examples.append(InputExample(
            texts=[f"query: {row['query']}", f"passage: {row['tool_text']}"],
            label=0.0
        ))
    print(f"  Total samples: {len(train_examples)}")
    print("\n[Loss] CosineSimilarityLoss (supervised, no InfoNCE)...")
    train_loss = losses.CosineSimilarityLoss(encoder)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    print(f"\n[Train] Training encoder ({epochs} epochs)...")
    print(f"  Batch size: {batch_size}, fp16: {use_fp16}, gradient checkpointing: {use_gradient_checkpointing}")
    fit_kwargs = {
        "train_objectives": [(train_dataloader, train_loss)],
        "epochs": epochs,
        "warmup_steps": warmup_steps,
        "output_path": str(output_dir / "encoder"),
        "show_progress_bar": True,
        "optimizer_params": {"lr": learning_rate},
    }
    if use_fp16:
        try:
            fit_kwargs["use_amp"] = True
        except Exception:
            print("  Warning: fp16 may not be supported, using fp32")
    encoder.fit(**fit_kwargs)
    print(f"\n[Save] Encoder saved to: {output_dir / 'encoder'}")
    return encoder


def main():
    parser = argparse.ArgumentParser(
        description="Train RATED encoder without contrastive learning (ablation)"
    )
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Training pairs CSV (query, tool_uid, label)"
    )
    parser.add_argument(
        "--tools",
        type=str,
        required=True,
        help="Tools CSV (glob supported)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="thenlper/gte-large",
        help="Base encoder model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (e.g. 4 for GTE-large)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Warmup steps"
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use mixed precision (fp16)"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Max sequence length"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--neg-per-pos",
        type=int,
        default=0,
        help="Deprecated; kept for compatibility"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("RATED Encoder training (w/o contrastive learning)")
    print("=" * 60)
    print("\n[Load] Loading data...")
    pairs_df = pd.read_csv(args.pairs)
    print(f"  Pairs: {len(pairs_df)}")
    tool_files = glob.glob(args.tools)
    if not tool_files:
        raise FileNotFoundError(f"No tool files found matching pattern: {args.tools}")
    tools_list = []
    for f in tool_files:
        df = pd.read_csv(f)
        tools_list.append(df)
    tools_df = pd.concat(tools_list, ignore_index=True)
    tools_df = tools_df.drop_duplicates(subset=['uid'], keep='first')
    print(f"  Tools: {len(tools_df)}")
    encoder = train_encoder_supervised(
        pairs_df=pairs_df,
        tools_df=tools_df,
        base_model=args.base_model,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        use_fp16=args.use_fp16,
        max_seq_len=args.max_seq_len,
        use_gradient_checkpointing=args.gradient_checkpointing,
        neg_per_pos=args.neg_per_pos
    )
    config = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_seq_len": args.max_seq_len,
        "loss": "CosineSimilarityLoss",
        "training_method": "supervised_learning_with_explicit_negatives",
        "contrastive_learning": False,
        "uses_negative_samples": True,
        "neg_per_pos": args.neg_per_pos,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat()
    }
    
    config_file = output_dir / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n[Save] Config saved to: {config_file}")
    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)
    print(f"\nModel path: {output_dir / 'encoder'}")
    print("\nNext: evaluate with compare_embedding_models.py")


if __name__ == '__main__':
    main()
