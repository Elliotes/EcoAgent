#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RATED joint training: train encoder and initialize bandit together.
Encoder is trained with contrastive learning; bandit is warm-up initialized on training data.
"""

import argparse
import glob
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def compute_static_reliability_scores(
    tools_df: pd.DataFrame,
    sbert_model: SentenceTransformer = None
) -> Dict[str, float]:
    tools = tools_df.copy()
    tools["uid"] = tools["uid"].astype(str)
    tools["name"] = tools["name"].fillna("").astype(str)
    tools["description"] = tools["description"].fillna("").astype(str)
    tools["tags"] = tools.get("tags", "").fillna("").astype(str)
    tools["source"] = tools.get("source", "").fillna("").astype(str)
    if sbert_model is not None:
        names = tools["name"].tolist()
        descs = tools["description"].tolist()
        name_embs = sbert_model.encode(names, batch_size=64, show_progress_bar=True)
        desc_embs = sbert_model.encode(descs, batch_size=64, show_progress_bar=True)
        name_embs = name_embs / (np.linalg.norm(name_embs, axis=1, keepdims=True) + 1e-9)
        desc_embs = desc_embs / (np.linalg.norm(desc_embs, axis=1, keepdims=True) + 1e-9)
        cons = np.sum(name_embs * desc_embs, axis=1)
        cons = np.nan_to_num(cons, nan=0.0)
    else:
        corpus = (tools["name"] + " " + tools["description"]).tolist()
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
        X = vec.fit_transform(corpus)
        name_X = vec.transform(tools["name"].tolist())
        desc_X = vec.transform(tools["description"].tolist())
        cons = cosine_similarity(name_X, desc_X).diagonal()
        cons = np.nan_to_num(cons, nan=0.0)
    
    # 2. Source Credibility
    def src_weight(s: str) -> float:
        s = (s or "").lower()
        if "official" in s:
            return 1.0
        if "langchain" in s or "crewai" in s or "llamaindex" in s:
            return 0.9
        if "n8n" in s:
            return 0.8
        return 0.6
    
    src = tools["source"].map(src_weight).to_numpy(dtype=float)
    desc_len = tools["description"].str.len().to_numpy(dtype=float)
    tag_len = tools["tags"].str.len().to_numpy(dtype=float)
    dl = (desc_len - np.median(desc_len)) / (np.std(desc_len) + 1e-9)
    tl = (tag_len - np.median(tag_len)) / (np.std(tag_len) + 1e-9)
    qual = _sigmoid(0.6 * dl + 0.4 * tl)
    
    # 4. Weighted combination
    alpha, beta, gamma = 0.5, 0.3, 0.2
    r_static = alpha * cons + beta * src + gamma * qual
    r_static = np.clip(r_static, 0.0, 1.0)
    
    return dict(zip(tools["uid"].tolist(), r_static.tolist()))


@dataclass
class LinUCBArm:
    A: np.ndarray
    b: np.ndarray


class DiscountedLinUCB:
    def __init__(
        self,
        d: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        ridge: float = 1.0,
        use_diagonal: bool = True,
        dtype: np.dtype = np.float32
    ):
        self.d = d
        self.alpha = alpha
        self.gamma = gamma
        self.ridge = ridge
        self.use_diagonal = use_diagonal
        self.dtype = dtype
        self.arms: Dict[str, LinUCBArm] = {}
    
    def _get_arm(self, arm_id: str) -> LinUCBArm:
        if arm_id not in self.arms:
            if self.use_diagonal:
                A = np.ones((self.d,), dtype=self.dtype) * self.ridge
            else:
                A = (np.eye(self.d, dtype=self.dtype) * self.ridge)
            b = np.zeros((self.d,), dtype=self.dtype)
            self.arms[arm_id] = LinUCBArm(A=A, b=b)
        return self.arms[arm_id]
    
    def initialize_with_static_reliability(self, tool_uid: str, static_reliability: float, default_context: np.ndarray):
        arm = self._get_arm(tool_uid)
        arm.b = static_reliability * default_context.copy()
    
    def score(self, arm_id: str, x: np.ndarray) -> float:
        arm = self._get_arm(arm_id)
        x = x.astype(self.dtype, copy=False)
        if self.use_diagonal:
            A_inv = 1.0 / (arm.A + 1e-9)
            theta = A_inv * arm.b
            mu = float(np.dot(theta, x))
            sigma = float(np.sqrt(np.sum((x * x) * A_inv)))
        else:
            A_inv = np.linalg.inv(arm.A)
            theta = A_inv @ arm.b
            mu = float(theta @ x)
            sigma = float(np.sqrt(x @ A_inv @ x))
        return mu + self.alpha * sigma
    
    def update(self, arm_id: str, x: np.ndarray, reward: float):
        arm = self._get_arm(arm_id)
        x = x.astype(self.dtype, copy=False)
        if self.use_diagonal:
            arm.A = self.gamma * arm.A + (x * x)
        else:
            arm.A = self.gamma * arm.A + np.outer(x, x)
        arm.b = self.gamma * arm.b + (reward * x)
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        arms_dict = {}
        for arm_id, arm in self.arms.items():
            arms_dict[arm_id] = {
                'A': arm.A,
                'b': arm.b
            }
        dtype_str = str(self.dtype)
        if 'float32' in dtype_str:
            dtype_str = 'float32'
        elif 'float64' in dtype_str:
            dtype_str = 'float64'
        
        with open(path, 'wb') as f:
            pickle.dump({
                'd': self.d,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'ridge': self.ridge,
                'use_diagonal': self.use_diagonal,
                'dtype': dtype_str,
                'arms': arms_dict
            }, f)
    
    @classmethod
    def load(cls, path: Path):
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'LinUCBArm':
                    return LinUCBArm
                if name == 'DiscountedLinUCB':
                    return cls
                try:
                    return super().find_class(module, name)
                except (AttributeError, ModuleNotFoundError) as e:
                    if name == 'LinUCBArm':
                        return LinUCBArm
                    if name == 'DiscountedLinUCB':
                        return cls
                    raise e
        
        with open(path, 'rb') as f:
            try:
                unpickler = CustomUnpickler(f)
                data = unpickler.load()
            except Exception:
                f.seek(0)
                try:
                    data = pickle.load(f)
                except Exception:
                    f.seek(0)
                    unpickler = CustomUnpickler(f)
                    data = unpickler.load()
        
        dtype_val = data.get('dtype', 'float32')
        if isinstance(dtype_val, str):
            if 'numpy.float32' in dtype_val or 'float32' in dtype_val:
                dtype_obj = np.float32
            elif 'numpy.float64' in dtype_val or 'float64' in dtype_val:
                dtype_obj = np.float64
            else:
                # Try to parse as dtype string
                try:
                    dtype_obj = np.dtype(dtype_val)
                except:
                    dtype_obj = np.float32
        elif isinstance(dtype_val, type) and issubclass(dtype_val, np.generic):
            # It's already a numpy dtype type (e.g., np.float32)
            dtype_obj = dtype_val
        elif isinstance(dtype_val, np.dtype):
            # It's already a numpy dtype object
            dtype_obj = dtype_val
        else:
            # Default fallback
            dtype_obj = np.float32
        
        bandit = cls(
            d=data['d'],
            alpha=data['alpha'],
            gamma=data['gamma'],
            ridge=data.get('ridge', 1.0),
            use_diagonal=data.get('use_diagonal', True),
            dtype=dtype_obj
        )
        arms_dict = data.get('arms', {})
        bandit.arms = {}
        for arm_id, arm_data in arms_dict.items():
            if isinstance(arm_data, dict):
                bandit.arms[arm_id] = LinUCBArm(A=arm_data['A'], b=arm_data['b'])
            else:
                bandit.arms[arm_id] = arm_data
        return bandit


def train_encoder_contrastive(
    pairs_df: pd.DataFrame,
    tools_df: pd.DataFrame,
    base_model: str,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    warmup_steps: int = 1000,
    temperature: float = 0.05,
    use_fp16: bool = False,
    max_seq_len: int = 256,
    use_gradient_checkpointing: bool = False
):
    print("\n" + "=" * 60)
    print("Stage 1: Train Encoder (contrastive)")
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
    print("\n[Data] Preparing contrastive training data...")
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
    print(f"  Positive pairs: {len(positive_pairs)}")
    train_examples = []
    for _, row in tqdm(positive_pairs.iterrows(), total=len(positive_pairs), desc="Preparing samples"):
        train_examples.append(InputExample(
            texts=[f"query: {row['query']}", f"passage: {row['tool_text']}"],
            label=1.0
        ))
    
    print("\n[Loss] InfoNCE loss...")
    train_loss = losses.MultipleNegativesRankingLoss(encoder, scale=1.0/temperature)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    print(f"\n[Train] Training encoder ({epochs} epochs)...")
    print(f"  Batch size: {batch_size}, fp16: {use_fp16}")
    if batch_size < 2:
        print("  Warning: batch_size < 2 gives no in-batch negatives, loss may be 0")
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


def initialize_bandit_with_training_data(
    encoder: SentenceTransformer,
    pairs_df: pd.DataFrame,
    tools_df: pd.DataFrame,
    static_reliability: Dict[str, float],
    output_dir: Path,
    warmup_steps: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    top_k: int = 10,
    seed: int = 42,
    use_diagonal: bool = True,
    init_tool_scope: str = "seen"
):
    print("\n" + "=" * 60)
    print("Stage 2: Initialize Bandit (warm-up)")
    print("=" * 60)
    np.random.seed(seed)
    d = encoder.get_sentence_embedding_dimension()
    print(f"\n[Bandit] Initializing Discounted LinUCB...")
    bandit = DiscountedLinUCB(
        d=d,
        alpha=alpha,
        gamma=gamma,
        use_diagonal=use_diagonal,
        dtype=np.float32
    )
    print(f"  Use diagonal: {use_diagonal}")
    print("[Bandit] Initializing bandit with static reliability...")
    default_context = (np.ones(d, dtype=np.float32) / np.sqrt(d)).astype(np.float32)
    if init_tool_scope == "all":
        init_tool_uids = list(static_reliability.keys())
    else:
        init_tool_uids = pairs_df["tool_uid"].astype(str).unique().tolist()
    print(f"  Initializing {len(init_tool_uids)} tools (scope={init_tool_scope})")
    for tool_uid in tqdm(init_tool_uids, desc="Initializing tools"):
        r_static = float(static_reliability.get(tool_uid, 0.0))
        bandit.initialize_with_static_reliability(tool_uid, r_static, default_context)
    print("\n[Embed] Preparing tool embeddings...")
    uid_col = "uid" if "uid" in tools_df.columns else tools_df.columns[0]
    text_col = "tool_text" if "tool_text" in tools_df.columns else "description"
    tool_uids = tools_df[uid_col].astype(str).tolist()
    tool_texts = tools_df[text_col].astype(str).tolist()
    tool_embeddings = encoder.encode(tool_texts, batch_size=64, show_progress_bar=True)
    tool_embeddings = tool_embeddings / (np.linalg.norm(tool_embeddings, axis=1, keepdims=True) + 1e-9)
    uid_to_emb = dict(zip(tool_uids, tool_embeddings))
    print(f"\n[Warm-up] Warming up bandit on training data ({warmup_steps} steps)...")
    positive_pairs = pairs_df[pairs_df["label"] == 1].copy()
    if len(positive_pairs) > warmup_steps:
        positive_pairs = positive_pairs.sample(n=warmup_steps, random_state=seed)
    
    queries = positive_pairs["query"].astype(str).tolist()
    query_embeddings = encoder.encode(queries, batch_size=64, show_progress_bar=True)
    query_embeddings = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-9)
    
    successes = []
    for i, (_, row) in enumerate(tqdm(positive_pairs.iterrows(), total=len(positive_pairs), desc="Warm-up")):
        query = str(row["query"])
        tool_uid = str(row["tool_uid"])
        x_t = query_embeddings[i]
        if tool_uid in uid_to_emb:
            tool_emb = uid_to_emb[tool_uid]
            similarities = np.dot(tool_embeddings, tool_emb)
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            candidates = [tool_uids[idx] for idx in top_k_indices]
        else:
            candidates = np.random.choice(tool_uids, size=min(top_k, len(tool_uids)), replace=False).tolist()
        best_tool = None
        best_score = -np.inf
        for cand_uid in candidates:
            if cand_uid in uid_to_emb:
                score = bandit.score(cand_uid, x_t)
                if score > best_score:
                    best_score = score
                    best_tool = cand_uid
        
        if best_tool is None:
            best_tool = candidates[0] if candidates else tool_uid
        semantic_correct = 1.0 if best_tool == tool_uid else 0.0
        tool_status = 1.0
        reward = semantic_correct * tool_status
        successes.append(int(reward))
        bandit.update(best_tool, x_t, reward)
    success_rate = np.mean(successes) * 100
    print(f"\n[Warm-up] Done.")
    print(f"  Success rate: {success_rate:.2f}%, trained tools: {len(bandit.arms)}")
    bandit_path = output_dir / "bandit_warmup.pkl"
    bandit.save(bandit_path)
    print(f"[Save] Bandit saved to: {bandit_path}")
    
    return bandit


def main():
    parser = argparse.ArgumentParser(
        description="RATED joint training: train encoder and initialize bandit together"
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
        help="Tools CSV (optionally with reliability_score column)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="BAAI/bge-m3",
        help="Base encoder model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ecoagent_syn_20260115/models/rated_joint",
        help="Output directory"
    )
    parser.add_argument(
        "--encoder-epochs",
        type=int,
        default=3,
        help="Encoder training epochs"
    )
    parser.add_argument(
        "--encoder-batch-size",
        type=int,
        default=2,
        help="Encoder batch size (e.g. 2 for BGE-M3, 4-8 for smaller models)"
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use mixed precision (fp16) to reduce memory"
    )
    parser.add_argument(
        "--encoder-max-seq-len",
        type=int,
        default=256,
        help="Encoder max sequence length (e.g. 128/256)"
    )
    parser.add_argument(
        "--encoder-gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--encoder-path",
        type=str,
        default=None,
        help="Path to trained encoder (skip Stage 1 if set)"
    )
    parser.add_argument(
        "--skip-encoder",
        action="store_true",
        help="Skip Stage 1 (load encoder from output-dir/encoder)"
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=2e-5,
        help="Encoder learning rate"
    )
    parser.add_argument(
        "--encoder-warmup-steps",
        type=int,
        default=1000,
        help="Encoder warmup steps"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="InfoNCE temperature"
    )
    parser.add_argument(
        "--bandit-warmup-steps",
        type=int,
        default=2000,
        help="Bandit warm-up steps"
    )
    parser.add_argument(
        "--bandit-alpha",
        type=float,
        default=0.1,
        help="Bandit exploration alpha"
    )
    parser.add_argument(
        "--bandit-gamma",
        type=float,
        default=0.95,
        help="Bandit discount factor gamma"
    )
    parser.add_argument(
        "--bandit-top-k",
        type=int,
        default=10,
        help="Bandit candidate tool count"
    )
    parser.add_argument(
        "--bandit-full-matrix",
        action="store_true",
        help="Use full-matrix LinUCB (higher memory)"
    )
    parser.add_argument(
        "--bandit-init-scope",
        type=str,
        default="seen",
        choices=["seen", "all"],
        help="Bandit init scope: seen=tools in pairs, all=all tools"
    )
    parser.add_argument(
        "--reliability-file",
        type=str,
        default=None,
        help="Static reliability CSV (optional if tools CSV has reliability_score)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("RATED joint training")
    print("=" * 60)
    print("\n[Load] Loading data...")
    pairs_df = pd.read_csv(args.pairs)
    
    # Support glob pattern for tools file
    tool_files = glob.glob(args.tools)
    if not tool_files:
        raise FileNotFoundError(f"No tool files found matching pattern: {args.tools}")
    
    # Load and merge all tool files
    tools_list = []
    for f in tool_files:
        df = pd.read_csv(f)
        tools_list.append(df)
    tools_df = pd.concat(tools_list, ignore_index=True)
    
    # Deduplicate by uid (use first column if 'uid' not found)
    uid_col = "uid" if "uid" in tools_df.columns else tools_df.columns[0]
    tools_df = tools_df.drop_duplicates(subset=[uid_col], keep='first')
    print(f"  Pairs: {len(pairs_df)}, tools: {len(tools_df)} (from {len(tool_files)} file(s))")
    print("\n[Reliability] Static reliability...")
    if "reliability_score" in tools_df.columns:
        uid_col = "uid" if "uid" in tools_df.columns else tools_df.columns[0]
        static_reliability = dict(zip(
            tools_df[uid_col].astype(str),
            tools_df["reliability_score"].astype(float)
        ))
        print(f"  Loaded reliability for {len(static_reliability)} tools from tools file")
    elif args.reliability_file and Path(args.reliability_file).exists():
        print(f"  Loading from: {args.reliability_file}")
        reliability_df = pd.read_csv(args.reliability_file)
        uid_col = "uid" if "uid" in reliability_df.columns else reliability_df.columns[0]
        static_reliability = dict(zip(
            reliability_df[uid_col].astype(str),
            reliability_df["reliability_score"].astype(float)
        ))
    else:
        print("  Computing reliability (TF-IDF)...")
        static_reliability = compute_static_reliability_scores(tools_df, sbert_model=None)
    print(f"  Mean reliability: {np.mean(list(static_reliability.values())):.4f}")
    if args.encoder_path:
        encoder_path = Path(args.encoder_path)
        print(f"\n[Encoder] Using trained model: {encoder_path}")
        encoder = SentenceTransformer(str(encoder_path))
    elif args.skip_encoder:
        encoder_path = output_dir / "encoder"
        print(f"\n[Encoder] Skipping training, loading: {encoder_path}")
        encoder = SentenceTransformer(str(encoder_path))
    else:
        encoder = train_encoder_contrastive(
            pairs_df=pairs_df,
            tools_df=tools_df,
            base_model=args.base_model,
            output_dir=output_dir,
            epochs=args.encoder_epochs,
            batch_size=args.encoder_batch_size,
            learning_rate=args.encoder_lr,
            warmup_steps=args.encoder_warmup_steps,
            temperature=args.temperature,
            use_fp16=args.use_fp16,
            max_seq_len=args.encoder_max_seq_len,
            use_gradient_checkpointing=args.encoder_gradient_checkpointing
        )
    
    bandit = initialize_bandit_with_training_data(
        encoder=encoder,
        pairs_df=pairs_df,
        tools_df=tools_df,
        static_reliability=static_reliability,
        output_dir=output_dir,
        warmup_steps=args.bandit_warmup_steps,
        alpha=args.bandit_alpha,
        gamma=args.bandit_gamma,
        top_k=args.bandit_top_k,
        seed=args.seed,
        use_diagonal=(not args.bandit_full_matrix),
        init_tool_scope=args.bandit_init_scope
    )
    config = {
        "base_model": args.base_model,
        "encoder_epochs": args.encoder_epochs,
        "encoder_batch_size": args.encoder_batch_size,
        "encoder_lr": args.encoder_lr,
        "temperature": args.temperature,
        "bandit_warmup_steps": args.bandit_warmup_steps,
        "bandit_alpha": args.bandit_alpha,
        "bandit_gamma": args.bandit_gamma,
        "bandit_top_k": args.bandit_top_k,
        "bandit_full_matrix": args.bandit_full_matrix,
        "bandit_init_scope": args.bandit_init_scope,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    config_path = output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("\n" + "=" * 60)
    print("Joint training complete.")
    print("=" * 60)
    print(f"\nOutput: {output_dir}")
    print(f"  Encoder: {output_dir / 'encoder'}")
    print(f"  Bandit: {output_dir / 'bandit_warmup.pkl'}")
    print(f"  Config: {config_path}")
    print("\nNext: run online evaluation with the trained model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
