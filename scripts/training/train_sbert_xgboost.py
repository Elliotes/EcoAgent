#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a (query, tool) matcher with SBERT embeddings + XGBoost.
X = [ q_emb || t_emb || cosine(q_emb, t_emb) ], y = 0/1

Inputs:
  --pairs   CSV with columns: query, tool_uid, label
  --tools   CSV with columns: uid, tool_text (or auto-detect via heuristics)
  --model   SentenceTransformer name or local path (default: all-MiniLM-L6-v2)

Outputs (with timestamp):
  model_xgb_YYYYmmdd_HHMMSS.joblib
  metrics_xgb_YYYYmmdd_HHMMSS.json
"""

import argparse, json, re, os, hashlib
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from xgboost import XGBClassifier


# ---------- utils ----------
def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def detect_tool_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cand = [("uid","tool_text"),("tool_uid","tool_text"),("uid","text"),
            ("tool_uid","text"),("name","description"),("id","text")]
    low = {c.lower():c for c in df.columns}
    for a,b in cand:
        if a in low and b in low:
            return low[a], low[b]
    # fallback: first two columns
    cols = list(df.columns)
    return cols[0], cols[1]

def eval_ranking(df_te: pd.DataFrame, proba: np.ndarray, ks=(1,3,5)) -> Dict:
    out = {}
    df = df_te.copy()
    df["score"] = proba
    groups = df.groupby("query", sort=False)

    # P@K
    for k in ks:
        hits = []
        for _, g in groups:
            gk = g.sort_values("score", ascending=False).head(k)
            hits.append(1.0 if (gk["label"].sum() > 0) else 0.0)
        out[f"P@{k}"] = float(np.mean(hits)) if hits else 0.0

    # MRR
    mrrs = []
    for _, g in groups:
        g = g.sort_values("score", ascending=False)
        rr = 0.0
        for i, lab in enumerate(g["label"].to_numpy(), start=1):
            if lab == 1:
                rr = 1.0 / i
                break
        mrrs.append(rr)
    out["MRR"] = float(np.mean(mrrs)) if mrrs else 0.0
    return out

def score_from_clf(clf, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        return p[:,1] if p.ndim == 2 and p.shape[1] >= 2 else np.asarray(p).ravel()
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        return s[:,1] if (hasattr(s, "ndim") and s.ndim==2 and s.shape[1]>=2) else np.asarray(s).ravel()
    return np.asarray(clf.predict(X), dtype="float32")

def build_features(q_emb: np.ndarray, t_emb: np.ndarray) -> np.ndarray:
    # Compute cosine similarity for each pair (q_emb[i], t_emb[i])
    # Since embeddings are normalized, dot product = cosine similarity
    cos = np.sum(q_emb * t_emb, axis=1, keepdims=True).astype("float32")
    return np.hstack([q_emb, t_emb, cos]).astype("float32")

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ---------- encoding (unique texts + smart cache with per-text index) ----------
def encode_unique(model: SentenceTransformer, texts: pd.Series,
                  batch_size: int, cache_dir: Path, tag: str, model_id: str) -> Tuple[np.ndarray, Dict[str,int]]:
    """
    Encode unique texts with smart caching:
    1. First try exact set match (fastest)
    2. Then try per-text cache (allows reuse across different sets)
    3. Finally encode new texts and update both caches
    """
    texts = texts.astype(str)
    uniq = texts.unique()
    idx_map = {t:i for i,t in enumerate(uniq)}
    uniq_list = uniq.tolist()

    cache_dir.mkdir(parents=True, exist_ok=True)
    model_hash = sha1_str(model_id)[:16]
    
    # Strategy 1: Try exact set match first (fastest for same set)
    uniq_sorted = sorted(uniq_list)
    cache_key = sha1_str(model_id + f"|{len(uniq_sorted)}|" + sha1_str("||".join(uniq_sorted)))
    exact_cache = cache_dir / f"emb_{tag}_{cache_key}.npy"
    
    if exact_cache.exists():
        emb_u = np.load(exact_cache)
        print(f"[Cache] Loaded {tag} embeddings from exact cache ({len(uniq)} unique texts)")
        emb = emb_u[[idx_map[t] for t in texts.tolist()]]
        return emb, idx_map
    
    # Strategy 2: Try per-text cache (allows reuse across different sets)
    index_file = cache_dir / f"index_{tag}_{model_hash}.json"
    text_to_emb_file = {}
    
    if index_file.exists():
        try:
            with open(index_file, 'r') as f:
                text_to_emb_file = json.load(f)
        except:
            text_to_emb_file = {}
    
    # Load cached embeddings
    cached_embeddings = {}
    texts_to_encode = []
    
    for text in uniq_list:
        if text in text_to_emb_file:
            emb_file = Path(text_to_emb_file[text])
            if emb_file.exists():
                try:
                    cached_embeddings[text] = np.load(emb_file)[0]  # [0] to get single embedding
                    continue
                except:
                    pass
        
        # Try direct per-text cache file
        text_hash = sha1_str(text)[:16]
        emb_file = cache_dir / f"emb_{tag}_single_{text_hash}.npy"
        if emb_file.exists():
            try:
                cached_embeddings[text] = np.load(emb_file)[0]
                text_to_emb_file[text] = str(emb_file)
                continue
            except:
                pass
        
        # Need to encode
        texts_to_encode.append(text)
    
    # Encode new texts
    if texts_to_encode:
        cached_count = len(cached_embeddings)
        print(f"[Encode] {tag}: {cached_count}/{len(uniq)} in cache, encoding {len(texts_to_encode)} new...")
        new_emb = model.encode(texts_to_encode,
                              batch_size=batch_size,
                              show_progress_bar=len(texts_to_encode) > 100,
                              convert_to_numpy=True,
                              normalize_embeddings=True).astype("float32")
        
        # Save per-text cache files and update index
        for i, text in enumerate(texts_to_encode):
            text_hash = sha1_str(text)[:16]
            emb_file = cache_dir / f"emb_{tag}_single_{text_hash}.npy"
            np.save(emb_file, new_emb[i:i+1])
            cached_embeddings[text] = new_emb[i]
            text_to_emb_file[text] = str(emb_file)
        
        # Save updated index
        with open(index_file, 'w') as f:
            json.dump(text_to_emb_file, f)
    else:
        print(f"[Cache] All {len(uniq)} {tag} texts loaded from per-text cache")
    
    # Build embedding array
    emb_u = np.array([cached_embeddings[t] for t in uniq_list])
    
    # Also save exact set cache for faster loading next time
    np.save(exact_cache, emb_u)

    # map back
    emb = emb_u[[idx_map[t] for t in texts.tolist()]]
    return emb, idx_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/query_tool_pairs.csv", help="CSV: query, tool_uid, label")
    ap.add_argument("--tools", default="data/tools_clean_20251011_114108.csv", help="CSV: uid, tool_text (or auto-detect)")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SBERT model name or path")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--outdir", default="data/matcher_sbert_xgb_out")
    ap.add_argument("--cache-emb-dir", default="data/sbert_cache")
    # XGB hyperparams
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.8)
    ap.add_argument("--scale-pos-weight", type=float, default=1.0, help="class imbalance handling")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_emb_dir)

    # 1) load data
    pairs = pd.read_csv(args.pairs)
    tools = pd.read_csv(args.tools)

    # sanity check
    for c in ["query","tool_uid","label"]:
        if c not in pairs.columns:
            raise ValueError(f"[pairs] missing column: {c}")
    uid_col, text_col = detect_tool_columns(tools)
    if uid_col not in tools.columns or text_col not in tools.columns:
        raise ValueError(f"[tools] missing expected columns (got {list(tools.columns)})")
    uid2text = dict(zip(tools[uid_col].astype(str), tools[text_col].astype(str)))
    pairs["tool_text"] = pairs["tool_uid"].astype(str).map(uid2text)
    pairs = pairs.dropna(subset=["tool_text","query"]).reset_index(drop=True)

    # 2) load SBERT
    print(f"[SBERT] loading model: {args.model}")
    sbert = SentenceTransformer(args.model)
    model_id = getattr(sbert, "model_card", None) or args.model

    # 3) encode tools from FULL tool library (not just pairs) for better cache reuse
    print("[Encode] tools from full library (unique, cached)…")
    all_tool_texts = pd.Series(tools[text_col].astype(str).unique())
    t_emb_full, tool_idx_map_full = encode_unique(sbert, all_tool_texts, args.batch_size, cache_dir, "t_full", str(model_id))
    # Map pairs' tool_text to full tool embeddings
    tool_text_to_idx = {text: i for i, text in enumerate(all_tool_texts)}
    t_emb = t_emb_full[[tool_text_to_idx.get(text, 0) for text in pairs["tool_text"].astype(str)]]
    
    # 4) encode queries (unique & cached)
    print("[Encode] queries (unique, cached)…")
    q_emb, _ = encode_unique(sbert, pairs["query"], args.batch_size, cache_dir, "q", str(model_id))

    # 4) features & labels
    X = build_features(q_emb, t_emb)
    y = pairs["label"].astype(int).to_numpy()

    # 5) Split by query (avoid data leakage: same query in train and test)
    unique_queries = pairs["query"].unique()
    
    # Stratified split by query: ensure queries with positive labels are distributed
    query_labels = pairs.groupby("query")["label"].max()  # 1 if query has any positive, else 0
    query_df = pd.DataFrame({
        "query": unique_queries,
        "has_positive": query_labels[unique_queries].values
    })
    
    # Split queries (not pairs) into train/test
    qtr, qte, _, _ = train_test_split(
        query_df["query"].values,
        query_df["has_positive"].values,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=query_df["has_positive"].values
    )
    
    # Map pairs to train/test based on query
    train_mask = pairs["query"].isin(qtr)
    test_mask = pairs["query"].isin(qte)
    
    Xtr, Xte = X[train_mask], X[test_mask]
    ytr, yte = y[train_mask], y[test_mask]
    df_tr = pairs[train_mask][["query","tool_uid","label"]].reset_index(drop=True)
    df_te = pairs[test_mask][["query","tool_uid","label"]].reset_index(drop=True)
    
    print(f"[Split] Train: {len(Xtr)} pairs ({len(qtr)} unique queries)")
    print(f"[Split] Test:  {len(Xte)} pairs ({len(qte)} unique queries)")

    # 6) train XGBoost
    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=args.scale_pos_weight,  # >1.0 if positives rarer
        tree_method="hist",
        random_state=args.random_state,
        n_jobs=-1
    )
    clf.fit(Xtr, ytr)

    # 7) evaluate
    prob_te = score_from_clf(clf, Xte)
    y_pred = (prob_te >= 0.5).astype(int)

    report = classification_report(yte, y_pred, digits=4, output_dict=True)
    try:
        auc = float(roc_auc_score(yte, prob_te))
    except Exception:
        auc = None
    rank = eval_ranking(df_te.assign(label=yte), prob_te, ks=(1,3,5))

    # 8) save
    ts = now_ts()
    model_path = outdir / f"model_xgb_{ts}.joblib"
    metrics_path = outdir / f"metrics_xgb_{ts}.json"
    joblib.dump(clf, model_path)

    metrics = {
        "clf": "xgboost",
        "embed": "sbert",
        "sbert_model": args.model,
        "pairs_file": args.pairs,  # Store pairs file path to identify TF-IDF vs SBERT pairs
        "feature_dim": int(X.shape[1]),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "split_method": "by_query",  # Important: split by query to avoid leakage
        "train_queries": int(len(qtr)),
        "test_queries": int(len(qte)),
        "train_pairs": int(len(Xtr)),
        "test_pairs": int(len(Xte)),
        "xgb": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "lr": args.lr,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample,
            "scale_pos_weight": args.scale_pos_weight
        },
        "classification": report,
        "roc_auc": auc,
        "ranking": rank
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Model   : {model_path}")
    print(f"Metrics : {metrics_path}")
    print(f"P@K/MRR : {rank}")
    print(f"ROC-AUC : {auc}")
    print(f"Feature dim: {X.shape[1]}  (q:{q_emb.shape[1]} + t:{t_emb.shape[1]} + cos:1)")
    

if __name__ == "__main__":
    main()
