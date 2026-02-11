#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit warm-up training: initialize with static reliability scores and pre-train on EcoAgent-Syn.
Steps: (1) compute static reliability from metadata; (2) initialize bandit with it;
(3) warm-up train on EcoAgent-Syn (e.g. steps 1-2000); (4) save bandit for online evaluation.
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


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
        print("[Static Reliability] Computing consistency scores with SBERT...")
        names = tools["name"].tolist()
        descs = tools["description"].tolist()
        name_embs = sbert_model.encode(names, batch_size=64, show_progress_bar=True)
        desc_embs = sbert_model.encode(descs, batch_size=64, show_progress_bar=True)
        name_embs = name_embs / (np.linalg.norm(name_embs, axis=1, keepdims=True) + 1e-9)
        desc_embs = desc_embs / (np.linalg.norm(desc_embs, axis=1, keepdims=True) + 1e-9)
        cons = np.sum(name_embs * desc_embs, axis=1)
        cons = np.nan_to_num(cons, nan=0.0)
    else:
        print("[Static Reliability] Computing consistency scores with TF-IDF...")
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
    alpha, beta, gamma = 0.5, 0.3, 0.2
    r_static = alpha * cons + beta * src + gamma * qual
    r_static = np.clip(r_static, 0.0, 1.0)
    
    return dict(zip(tools["uid"].tolist(), r_static.tolist()))


@dataclass
class LinUCBArm:
    A: np.ndarray
    b: np.ndarray


class DiscountedLinUCB:
    def __init__(self, d: int, alpha: float = 0.1, gamma: float = 0.95, ridge: float = 1.0):
        self.d = d
        self.alpha = alpha
        self.gamma = gamma
        self.ridge = ridge
        self.arms: Dict[str, LinUCBArm] = {}
    
    def _get_arm(self, arm_id: str) -> LinUCBArm:
        if arm_id not in self.arms:
            A = np.eye(self.d) * self.ridge
            b = np.zeros((self.d,), dtype=float)
            self.arms[arm_id] = LinUCBArm(A=A, b=b)
        return self.arms[arm_id]
    
    def initialize_with_static_reliability(
        self,
        tool_uid: str,
        static_reliability: float,
        default_context: np.ndarray
    ):
        arm = self._get_arm(tool_uid)
        arm.b = static_reliability * default_context.copy()

    def score(self, arm_id: str, x: np.ndarray) -> float:
        arm = self._get_arm(arm_id)
        A_inv = np.linalg.inv(arm.A)
        theta = A_inv @ arm.b
        mu = float(theta @ x)
        sigma = float(np.sqrt(x @ A_inv @ x))
        return mu + self.alpha * sigma
    
    def update(self, arm_id: str, x: np.ndarray, reward: float):
        arm = self._get_arm(arm_id)
        arm.A = self.gamma * arm.A + np.outer(x, x)
        arm.b = self.gamma * arm.b + reward * x
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'd': self.d,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'ridge': self.ridge,
                'arms': self.arms
            }, f)
    
    @classmethod
    def load(cls, path: Path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        bandit = cls(d=data['d'], alpha=data['alpha'], gamma=data['gamma'], ridge=data['ridge'])
        bandit.arms = data['arms']
        return bandit


def main():
    parser = argparse.ArgumentParser(
        description="Bandit warm-up: initialize with static reliability and pre-train on EcoAgent-Syn"
    )
    parser.add_argument("--pairs", type=str, required=True, help="Training pairs CSV (query, tool_uid, label)")
    parser.add_argument("--tools", type=str, required=True, help="Tools CSV (tools_clean_*.csv)")
    parser.add_argument("--encoder-model", type=str, default="BAAI/bge-m3",
                        help="Encoder model path (for context vectors)")
    parser.add_argument("--sbert-model", type=str, default=None,
                        help="SBERT model for static reliability (default: TF-IDF)")
    parser.add_argument("--reliability-file", type=str, default=None,
                        help="Existing static reliability CSV (must have reliability_score). If set, skip recompute.")
    parser.add_argument("--output-dir", type=str, default="data/ecoagent_syn_20260115/models/bandit",
                        help="Output directory")
    parser.add_argument("--warmup-steps", type=int, default=2000,
                        help="Warm-up steps (default 2000)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Exploration alpha (default 0.1)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor gamma (default 0.95)")
    parser.add_argument("--top-k", type=int, default=10, help="Candidate tools per query (default 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"[Load] Loading pairs from {args.pairs}")
    pairs_df = pd.read_csv(args.pairs)
    print(f"  Total pairs: {len(pairs_df)}")
    
    print(f"[Load] Loading tools from {args.tools}")
    tools_df = pd.read_csv(args.tools)
    print(f"  Total tools: {len(tools_df)}")
    print(f"[Encoder] Loading encoder model: {args.encoder_model}")
    encoder = SentenceTransformer(args.encoder_model)
    d = encoder.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {d}")
    
    print("\n[Static Reliability] Loading/computing static reliability scores...")
    if args.reliability_file and Path(args.reliability_file).exists():
        print(f"  Loading from existing file: {args.reliability_file}")
        reliability_df = pd.read_csv(args.reliability_file)
        uid_col = "uid" if "uid" in reliability_df.columns else reliability_df.columns[0]
        score_col = "reliability_score"
        if score_col not in reliability_df.columns:
            raise ValueError(f"Reliability file must contain column '{score_col}'")
        
        static_reliability = dict(zip(
            reliability_df[uid_col].astype(str),
            reliability_df[score_col].astype(float)
        ))
        print(f"  Loaded reliability scores for {len(static_reliability)} tools")
        print(f"  Mean reliability: {np.mean(list(static_reliability.values())):.4f}")
        print(f"  Score range: [{np.min(list(static_reliability.values())):.4f}, {np.max(list(static_reliability.values())):.4f}]")
    else:
        if args.reliability_file:
            print(f"  Warning: reliability file not found: {args.reliability_file}")
            print("  Recomputing reliability scores...")
        sbert_model = None
        if args.sbert_model:
            sbert_model = SentenceTransformer(args.sbert_model)
        static_reliability = compute_static_reliability_scores(tools_df, sbert_model)
        print(f"  Computed reliability scores for {len(static_reliability)} tools")
        print(f"  Mean reliability: {np.mean(list(static_reliability.values())):.4f}")
    print("\n[Bandit] Initializing Discounted LinUCB...")
    bandit = DiscountedLinUCB(d=d, alpha=args.alpha, gamma=args.gamma)
    print("[Bandit] Initializing bandit parameters with static reliability scores...")
    default_context = np.ones(d) / np.sqrt(d)
    for tool_uid, r_static in tqdm(static_reliability.items(), desc="Initializing tools"):
        bandit.initialize_with_static_reliability(tool_uid, r_static, default_context)
    print("\n[Prepare] Preparing tool embeddings...")
    uid_col = "uid" if "uid" in tools_df.columns else tools_df.columns[0]
    text_col = "tool_text" if "tool_text" in tools_df.columns else "description"
    tool_uids = tools_df[uid_col].astype(str).tolist()
    tool_texts = tools_df[text_col].astype(str).tolist()
    tool_embeddings = encoder.encode(tool_texts, batch_size=64, show_progress_bar=True)
    tool_embeddings = tool_embeddings / (np.linalg.norm(tool_embeddings, axis=1, keepdims=True) + 1e-9)
    uid_to_emb = dict(zip(tool_uids, tool_embeddings))
    print(f"\n[Warm-up] Training bandit on {args.warmup_steps} steps...")
    positive_pairs = pairs_df[pairs_df["label"] == 1].copy()
    if len(positive_pairs) > args.warmup_steps:
        positive_pairs = positive_pairs.sample(n=args.warmup_steps, random_state=args.seed)
    
    queries = positive_pairs["query"].astype(str).tolist()
    query_embeddings = encoder.encode(queries, batch_size=64, show_progress_bar=True)
    query_embeddings = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-9)
    
    successes = []
    for i, (_, row) in enumerate(tqdm(positive_pairs.iterrows(), total=len(positive_pairs), desc="Warm-up training")):
        query = str(row["query"])
        tool_uid = str(row["tool_uid"])
        x_t = query_embeddings[i]
        if tool_uid in uid_to_emb:
            tool_emb = uid_to_emb[tool_uid]
            similarities = np.dot(tool_embeddings, tool_emb)
            top_k_indices = np.argsort(similarities)[::-1][:args.top_k]
            candidates = [tool_uids[idx] for idx in top_k_indices]
        else:
            candidates = np.random.choice(tool_uids, size=min(args.top_k, len(tool_uids)), replace=False).tolist()
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
    print(f"\n[Warm-up] Training completed!")
    print(f"  Success rate: {success_rate:.2f}%")
    print(f"  Trained tools: {len(bandit.arms)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bandit_path = output_dir / f"bandit_warmup_{timestamp}.pkl"
    bandit.save(bandit_path)
    print(f"\n[Save] Bandit model saved to {bandit_path}")
    reliability_path = output_dir / f"static_reliability_{timestamp}.json"
    with open(reliability_path, 'w') as f:
        json.dump(static_reliability, f, indent=2)
    print(f"[Save] Static reliability scores saved to {reliability_path}")
    stats = {
        "warmup_steps": args.warmup_steps,
        "success_rate": float(success_rate),
        "trained_tools": len(bandit.arms),
        "alpha": args.alpha,
        "gamma": args.gamma,
        "encoder_model": args.encoder_model,
        "timestamp": timestamp
    }
    stats_path = output_dir / f"warmup_stats_{timestamp}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[Save] Training statistics saved to {stats_path}")
    
    print("\n" + "="*50)
    print("Bandit Warm-up Training Completed!")
    print("="*50)
    print(f"Bandit model: {bandit_path}")
    print(f"Static reliability: {reliability_path}")
    print(f"Statistics: {stats_path}")
    print("="*50)


if __name__ == "__main__":
    main()
