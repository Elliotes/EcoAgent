#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate EcoAgent-Syn dataset: use LLM to generate synthetic queries and build positive/negative pairs.
Pipeline: (1) Gemini generates 10-20 diverse queries per tool; (2) quality filter (5-30 words, low Jaccard vs description);
(3) negatives: hard semantic (SBERT), hard lexical (shared keywords), random; mix 70% hard + 30% random;
(4) output: query, tool_slug, label, neg_type, split.
Usage: python generate_ecoagent_syn.py
Note: default tools path data/tools_20260116.jsonl or data/raw/; output ecoagent_syn_YYYYMMDD_HHMMSS.jsonl; API key from config/gemini.env if not set.
"""

import argparse
import json
import re
import random
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google.generativeai not available. Install with: pip install google-generativeai")


def load_gemini_api_key_from_config():
    config_path = Path(__file__).parent.parent.parent / "config" / "gemini.env"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        if key.strip() == "GEMINI_API_KEY" and value.strip():
                            os.environ["GEMINI_API_KEY"] = value.strip()
                            return value.strip()
        except Exception as e:
            if os.environ.get("GEMINI_API_KEY"):
                return os.environ.get("GEMINI_API_KEY")
    return None


FUNCTIONAL_KEYWORDS = {
    "send", "email", "scrape", "webhook", "crm", "api", "request", "response",
    "fetch", "download", "upload", "parse", "transform", "convert", "format",
    "search", "query", "filter", "sort", "aggregate", "calculate", "compute",
    "create", "update", "delete", "read", "write", "save", "load", "store",
    "authenticate", "authorize", "validate", "verify", "check", "test",
    "notify", "alert", "message", "chat", "post", "publish", "share"
}


def normalize_text(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s\-\._]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def jaccard_similarity(s1: str, s2: str) -> float:
    words1 = set(normalize_text(s1).split())
    words2 = set(normalize_text(s2).split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def extract_functional_keywords(text: str) -> Set[str]:
    words = set(normalize_text(text).split())
    return words & FUNCTIONAL_KEYWORDS


def build_tool_description(tool: Dict) -> str:
    parts = []
    if tool.get("name"):
        parts.append(tool["name"])
    if tool.get("description"):
        parts.append(tool["description"])
    if tool.get("tags") and isinstance(tool["tags"], list):
        parts.append(f"Tags: {', '.join(tool['tags'])}")
    return ". ".join(parts)


def generate_queries_with_gemini(
    tool_name: str,
    tool_description: str,
    num_queries: int = 15,
    api_key: str = None
) -> List[str]:
    if not GEMINI_AVAILABLE:
        raise ImportError("google.generativeai not available")
    
    if api_key:
        genai.configure(api_key=api_key)
    elif os.environ.get("GEMINI_API_KEY"):
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    else:
        raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY env var or use --gemini-api-key")
    model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    prompt = f"""You are generating diverse user queries for an AI agent tool recommendation system.

Tool Name: {tool_name}
Tool Description: {tool_description}

Generate {num_queries} diverse, natural user queries that would lead to using this tool. Requirements:
1. Each query should be 5-30 words
2. Vary the intent types: direct action requests, troubleshooting questions, how-to questions
3. Use natural, conversational language
4. DO NOT copy phrases verbatim from the tool description
5. Make queries task-oriented and practical

Output format: One query per line, no numbering or bullets."""

    last_error = None
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            queries = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
            queries = [re.sub(r'^[\d\-\.\)\s]+', '', q).strip() for q in queries]
            queries = [q for q in queries if q and len(q) > 0]
            return queries[:num_queries]
        except Exception as e:
            last_error = e
            continue
    try:
        available_models = list(genai.list_models())
        for m in available_models:
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name.split("/")[-1]
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    queries = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
                    queries = [re.sub(r'^[\d\-\.\)\s]+', '', q).strip() for q in queries]
                    queries = [q for q in queries if q and len(q) > 0]
                    return queries[:num_queries]
                except Exception:
                    continue
    except Exception:
        pass
    print(f"Error generating queries for {tool_name}: {last_error}")
    return []


def filter_queries(queries: List[str], tool_description: str, min_words: int = 5, max_words: int = 30, max_jaccard: float = 0.5) -> List[str]:
    filtered = []
    desc_normalized = normalize_text(tool_description)
    for q in queries:
        q_normalized = normalize_text(q)
        words = q_normalized.split()
        if len(words) < min_words or len(words) > max_words:
            continue
        jaccard = jaccard_similarity(q, tool_description)
        if jaccard > max_jaccard:
            continue
        
        filtered.append(q)
    
    return filtered


def find_hard_semantic_negatives(
    query: str,
    tool_embeddings: np.ndarray,
    tool_indices: List[int],
    target_tool_idx: int,
    top_k: int = 50
) -> List[int]:
    query_embedding = sbert_model.encode([query], show_progress_bar=False)
    similarities = cosine_similarity(query_embedding, tool_embeddings[tool_indices])[0]
    candidate_indices = [i for i in range(len(tool_indices)) if tool_indices[i] != target_tool_idx]
    candidate_similarities = [(similarities[i], tool_indices[i]) for i in candidate_indices]
    candidate_similarities.sort(reverse=True)
    return [idx for _, idx in candidate_similarities[:top_k]]


def find_hard_lexical_negatives(
    query: str,
    tools: List[Dict],
    target_tool_idx: int,
    top_k: int = 50
) -> List[int]:
    query_keywords = extract_functional_keywords(query)
    if not query_keywords:
        return []
    scores = []
    for i, tool in enumerate(tools):
        if i == target_tool_idx:
            continue
        tool_text = build_tool_description(tool)
        tool_keywords = extract_functional_keywords(tool_text)
        overlap = len(query_keywords & tool_keywords)
        if overlap > 0:
            scores.append((overlap, i))
    scores.sort(reverse=True)
    return [idx for _, idx in scores[:top_k]]


def sample_negatives(
    query: str,
    tools: List[Dict],
    tool_embeddings: np.ndarray,
    target_tool_idx: int,
    num_negatives: int,
    hard_ratio: float = 0.7
) -> List[Tuple[int, str]]:
    num_hard = int(num_negatives * hard_ratio)
    num_random = num_negatives - num_hard
    negatives = []
    semantic_candidates = find_hard_semantic_negatives(
        query, tool_embeddings, list(range(len(tools))), target_tool_idx, top_k=100
    )
    if semantic_candidates:
        sampled = random.sample(semantic_candidates, min(num_hard, len(semantic_candidates)))
        negatives.extend([(idx, "hard_semantic") for idx in sampled])
        num_hard -= len(sampled)
    if num_hard > 0:
        lexical_candidates = find_hard_lexical_negatives(query, tools, target_tool_idx, top_k=100)
        lexical_candidates = [idx for idx in lexical_candidates if idx not in [n[0] for n in negatives]]
        if lexical_candidates:
            sampled = random.sample(lexical_candidates, min(num_hard, len(lexical_candidates)))
            negatives.extend([(idx, "hard_lexical") for idx in sampled])
            num_hard -= len(sampled)
    all_indices = list(range(len(tools)))
    all_indices.remove(target_tool_idx)
    available = [idx for idx in all_indices if idx not in [n[0] for n in negatives]]
    if available:
        num_random = min(num_random + num_hard, len(available))
        sampled = random.sample(available, num_random)
        negatives.extend([(idx, "random") for idx in sampled])
    
    return negatives[:num_negatives]


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ap = argparse.ArgumentParser(description="Generate EcoAgent-Syn dataset")
    ap.add_argument("--queries-per-tool", type=int, default=15,
                    help="Number of queries to generate per tool (default: 15)")
    ap.add_argument("--neg-per-pos", type=int, default=2,
                    help="Number of negative samples per positive pair (default: 2)")
    ap.add_argument("--hard-ratio", type=float, default=0.7,
                    help="Ratio of hard negatives (default: 0.7)")
    ap.add_argument("--gemini-api-key", type=str, default=None,
                    help="Gemini API key (or set GEMINI_API_KEY env var)")
    ap.add_argument("--sbert-model", type=str, default="all-MiniLM-L6-v2",
                    help="SBERT model for semantic similarity (default: all-MiniLM-L6-v2)")
    ap.add_argument("--train-ratio", type=float, default=0.9,
                    help="Ratio of training data (default: 0.9)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing output file")
    ap.add_argument("--verbose", action="store_true",
                    help="Verbose output")
    ap.add_argument("--output-suffix", type=str, default=None,
                    help="Suffix before .jsonl, e.g. gte -> ecoagent_syn_YYYYMMDD_HHMMSS_gte.jsonl")
    ap.add_argument("--output-dir", type=str, default="data",
                    help="Output directory (default: data)")
    ap.add_argument("--encode-batch-size", type=int, default=128,
                    help="Batch size for tool encoding (default 128; use 32 for gte-large if OOM)")
    ap.add_argument("--tools", type=str, default=None,
                    help="Tools JSONL path; default data/tools_20260116.jsonl or data/raw/tools_20260116.jsonl")
    ap.add_argument("--progress-every", type=int, default=50,
                    help="Print progress every N tools (default 50); 0=disable. Main cost: 1 Gemini call per tool.")
    args = ap.parse_args()

    if args.tools:
        tools_file = Path(args.tools) if Path(args.tools).is_absolute() else (project_root / args.tools)
    else:
        tools_file = project_root / "data" / "tools_20260116.jsonl"
        if not tools_file.exists():
            tools_file = project_root / "data" / "raw" / "tools_20260116.jsonl"
    fname = f"ecoagent_syn_{timestamp}"
    if args.output_suffix and str(args.output_suffix).strip():
        fname += f"_{args.output_suffix.strip()}"
    output_file = (project_root / args.output_dir) / f"{fname}.jsonl"
    if not args.gemini_api_key:
        loaded_key = load_gemini_api_key_from_config()
        if loaded_key:
            args.gemini_api_key = loaded_key
    print(f"[1/5] Loading tools from {tools_file}...")
    tools = []
    with open(tools_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tools.append(json.loads(line))
    print(f"  Loaded {len(tools)} tools")
    processed_tools = set()
    if args.resume and output_file.exists():
        print(f"[Resume] Checking existing output...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("label") == 1:
                        tool_slug = item.get("tool_slug")
                        if tool_slug:
                            for i, tool in enumerate(tools):
                                if tool.get("slug") == tool_slug:
                                    processed_tools.add(i)
                                    break
        print(f"  Found {len(processed_tools)} already processed tools")
    print(f"[2/5] Loading SBERT model: {args.sbert_model}...")
    global sbert_model
    sbert_model = SentenceTransformer(args.sbert_model)
    print(f"[3/5] Encoding tool descriptions...")
    tool_texts = [build_tool_description(tool) for tool in tools]
    tool_embeddings = sbert_model.encode(tool_texts, show_progress_bar=True, batch_size=args.encode_batch_size)
    print(f"  Encoded {len(tool_embeddings)} tool descriptions")
    print(f"[4/5] Generating queries and pairs...")
    if args.progress_every:
        print(f"  Progress every {args.progress_every} tools (also when redirected/nohup).")
    else:
        print(f"  Progress printing disabled (--progress-every 0).")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume else "w"
    start_time = time.time()
    total_queries = 0
    pe = max(1, int(args.progress_every)) if args.progress_every else 0

    with open(output_file, mode, encoding="utf-8") as f:
        pbar = tqdm(
            total=len(tools),
            desc="Processing tools",
            disable=not sys.stdout.isatty(),
            mininterval=30,
        )

        for i, tool in enumerate(tools):
            if i in processed_tools:
                pbar.update(1)
                if pe and (i + 1) % pe == 0:
                    el = time.time() - start_time
                    r = (i + 1) / el if el > 0 else 0
                    eta = (len(tools) - i - 1) / r if r > 0 else 0
                    print(f"[Progress] tool {i+1}/{len(tools)}, ~{total_queries} queries, elapsed {el/60:.1f} min, ETA {eta/60:.1f} min", flush=True)
                continue
            tool_slug = tool.get("slug") or tool.get("name", "").lower().replace(" ", "-")
            tool_name = tool.get("name", "")
            tool_desc = build_tool_description(tool)
            queries = generate_queries_with_gemini(
                tool_name, tool_desc, num_queries=args.queries_per_tool,
                api_key=args.gemini_api_key
            )
            queries = filter_queries(queries, tool_desc)
            if not queries:
                if args.verbose:
                    print(f"  [WARN] No valid queries for {tool_name}")
                pbar.update(1)
                if pe and (i + 1) % pe == 0:
                    el = time.time() - start_time
                    r = (i + 1) / el if el > 0 else 0
                    eta = (len(tools) - i - 1) / r if r > 0 else 0
                    print(f"[Progress] tool {i+1}/{len(tools)}, ~{total_queries} queries, elapsed {el/60:.1f} min, ETA {eta/60:.1f} min", flush=True)
                continue
            for query in queries:
                positive_item = {
                    "query": query,
                    "tool_slug": tool_slug,
                    "label": 1,
                    "neg_type": None,
                    "split": "train" if random.random() < args.train_ratio else "val"
                }
                f.write(json.dumps(positive_item, ensure_ascii=False) + "\n")
                negatives = sample_negatives(
                    query, tools, tool_embeddings, i,
                    num_negatives=args.neg_per_pos,
                    hard_ratio=args.hard_ratio
                )

                for neg_idx, neg_type in negatives:
                    neg_tool = tools[neg_idx]
                    neg_slug = neg_tool.get("slug") or neg_tool.get("name", "").lower().replace(" ", "-")

                    negative_item = {
                        "query": query,
                        "tool_slug": neg_slug,
                        "label": 0,
                        "neg_type": neg_type,
                        "split": "train" if random.random() < args.train_ratio else "val"
                    }
                    f.write(json.dumps(negative_item, ensure_ascii=False) + "\n")

            total_queries += len(queries)
            pbar.update(1)
            if args.verbose:
                print(f"  Generated {len(queries)} queries for {tool_name}")
            if pe and (i + 1) % pe == 0:
                el = time.time() - start_time
                r = (i + 1) / el if el > 0 else 0
                eta = (len(tools) - i - 1) / r if r > 0 else 0
                print(f"[Progress] tool {i+1}/{len(tools)}, ~{total_queries} queries, elapsed {el/60:.1f} min, ETA {eta/60:.1f} min", flush=True)
        pbar.close()
    print(f"[5/5] Generating statistics...")
    stats = {"total": 0, "positive": 0, "negative": 0, "train": 0, "val": 0}
    neg_type_counts = Counter()
    
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                stats["total"] += 1
                if item["label"] == 1:
                    stats["positive"] += 1
                else:
                    stats["negative"] += 1
                    if item.get("neg_type"):
                        neg_type_counts[item["neg_type"]] += 1
                
                if item.get("split") == "train":
                    stats["train"] += 1
                else:
                    stats["val"] += 1
    print(f"\n[Done] EcoAgent-Syn dataset generation complete.")
    print(f"  Output: {output_file}")
    print(f"  Total samples: {stats['total']}")
    print(f"  Positive: {stats['positive']}")
    print(f"  Negative: {stats['negative']}")
    print(f"  Train: {stats['train']}")
    print(f"  Val: {stats['val']}")
    print(f"  Negative type distribution:")
    for neg_type, count in neg_type_counts.most_common():
        print(f"    {neg_type}: {count}")


if __name__ == "__main__":
    main()

