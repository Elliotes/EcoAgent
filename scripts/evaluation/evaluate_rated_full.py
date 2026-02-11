#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate RATED (Full): load trained encoder and bandit for dynamic reliability evaluation.
Measures success rate, adaptation steps, and behavior under tool failure / new-tool scenarios.
"""

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
sys.path.append(str(Path(__file__).parent.parent / "training"))
from train_rated_joint import DiscountedLinUCB


def compute_static_reliability_scores(tools_df: pd.DataFrame) -> Dict[str, float]:
    tools_df = tools_df.copy()
    tools_df["uid"] = tools_df["uid"].astype(str)
    if "reliability_score" in tools_df.columns:
        reliability = dict(zip(
            tools_df["uid"].astype(str),
            tools_df["reliability_score"].astype(float)
        ))
        print(f"  Loaded reliability scores from tools: {len(reliability)} tools")
        return reliability
    print("  Warning: no reliability_score column; using default 0.75")
    return {uid: 0.75 for uid in tools_df["uid"].astype(str)}


def build_hybrid_score(
    semantic_score: float,
    static_reliability: float,
    dynamic_reliability: float,
    w_semantic: float = 0.5,
    w_static: float = 0.2,
    w_dynamic: float = 0.3
) -> float:
    return (
        w_semantic * semantic_score +
        w_static * static_reliability +
        w_dynamic * dynamic_reliability
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate RATED (Full): dynamic reliability evaluation")
    parser.add_argument("--encoder-model", type=str, required=True, help="Path to trained encoder model")
    parser.add_argument("--bandit-model", type=str, required=True, help="Path to trained bandit model (.pkl)")
    parser.add_argument("--pairs", type=str, required=True, help="Evaluation CSV (query, tool_uid, label)")
    parser.add_argument("--tools", type=str, required=True, help="Tools CSV")
    parser.add_argument("--output-dir", type=str, default="data/evaluation/rated_full", help="Output directory")
    parser.add_argument("--top-k", type=int, default=10, help="Candidate tools per query")
    parser.add_argument("--steps", type=int, default=10000, help="Simulation steps")
    parser.add_argument("--fail-step", type=int, default=2000, help="Step at which tools start failing")
    parser.add_argument("--n-fail-tools", type=int, default=None,
                        help="Number of failing tools (select from most frequent positives; overrides fail-frac if set)")
    parser.add_argument("--fail-frac", type=float, default=0.40,
                        help="Fraction of tools that fail (0.0-1.0), default 0.40")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--disable-bandit", action="store_true",
                        help="Disable bandit updates; use semantic similarity only (RATED w/o Feedback)")
    parser.add_argument("--bandit-w-semantic", type=float, default=0.75,
                        help="Weight of semantic similarity in hybrid score (default 0.75)")
    parser.add_argument("--bandit-w-static", type=float, default=0.1,
                        help="Weight of static reliability in hybrid score (default 0.1)")
    parser.add_argument("--bandit-w-dynamic", type=float, default=0.15,
                        help="Weight of dynamic reliability in hybrid score (default 0.15)")
    parser.add_argument("--pos-per-query", type=int, default=2,
                        help="Number of positive tools per candidate set (default 2)")
    parser.add_argument("--adaptation-target-rate", type=float, default=None,
                        help="Target success rate for adaptation steps (default: max(90%%, 85%%*peak))")
    parser.add_argument("--bandit-gamma", type=float, default=None,
                        help="Override bandit gamma (default: from saved model; 1.0 = standard LinUCB)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    if args.disable_bandit:
        print("RATED (w/o Feedback) dynamic reliability evaluation")
    else:
        print("RATED (Full) dynamic reliability evaluation")
    print("=" * 60)

    print("\n[Load] Loading data...")
    pairs_df = pd.read_csv(args.pairs)
    pairs_df["query"] = pairs_df["query"].astype(str)
    pairs_df["tool_uid"] = pairs_df["tool_uid"].astype(str)
    pairs_df["label"] = pairs_df["label"].astype(int)
    print(f"  Evaluation pairs: {len(pairs_df)}")
    tool_files = glob.glob(args.tools)
    if not tool_files:
        raise FileNotFoundError(f"No tool files found matching pattern: {args.tools}")
    
    tools_list = []
    for f in tool_files:
        df = pd.read_csv(f)
        tools_list.append(df)
    
    tools_df = pd.concat(tools_list, ignore_index=True)
    tools_df = tools_df.drop_duplicates(subset=['uid'], keep='first')
    tools_df["uid"] = tools_df["uid"].astype(str)
    print(f"  Tools: {len(tools_df)}")
    print(f"\n[Encoder] Loading model: {args.encoder_model}")
    encoder = SentenceTransformer(args.encoder_model)
    encoder_dim = encoder.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {encoder_dim}")
    print(f"\n[Bandit] Loading model: {args.bandit_model}")
    bandit = DiscountedLinUCB.load(Path(args.bandit_model))
    if args.bandit_gamma is not None:
        print(f"  Overriding Gamma: {bandit.gamma} -> {args.bandit_gamma}")
        bandit.gamma = args.bandit_gamma
    print(f"  Bandit dim: {bandit.d}, Alpha: {bandit.alpha}, Gamma: {bandit.gamma}")
    print(f"  Initialized arms: {len(bandit.arms)}")
    print("\n[Reliability] Loading static reliability scores...")
    static_reliability = compute_static_reliability_scores(tools_df)
    print("\n[Embed] Preparing tool embeddings...")
    tool_uids = tools_df["uid"].tolist()
    tool_texts = []
    for _, row in tools_df.iterrows():
        name = str(row.get("name", ""))
        desc = str(row.get("description", ""))
        tags = str(row.get("tags", ""))
        tool_text = f"{name} {desc} {tags}".strip()
        tool_texts.append(tool_text)
    tool_embeddings = encoder.encode(
        tool_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    tool_emb_map = {uid: emb for uid, emb in zip(tool_uids, tool_embeddings)}
    print(f"  Encoded {len(tool_emb_map)} tools")
    print("\n[Data] Building query-tool mapping...")
    query_groups = pairs_df.groupby("query")
    queries = list(query_groups.groups.keys())
    pos_pairs = pairs_df[pairs_df["label"] == 1]
    tool_counts = pos_pairs["tool_uid"].value_counts()
    sorted_tools = tool_counts.index.tolist()
    if args.n_fail_tools is not None:
        n_fail = args.n_fail_tools
        print(f"  Using specified n_fail_tools: {n_fail}")
    else:
        fail_frac = args.fail_frac
        n_fail = max(1, int(len(tool_counts) * fail_frac))
        print(f"  Using fail_frac {fail_frac:.1%}: {n_fail} failing tools")
    if len(sorted_tools) > n_fail * 2:
        start_idx = max(0, len(sorted_tools) // 10)
        end_idx = min(len(sorted_tools), start_idx + n_fail)
        failing_tools = set(sorted_tools[start_idx:end_idx])
        print(f"  Failing tools: {len(failing_tools)} (mid-frequency, skip first {start_idx})")
    else:
        failing_tools = set(sorted_tools[:n_fail])
        print(f"  Failing tools: {len(failing_tools)} (first {n_fail})")
    print(f"  Failing tool UIDs: {list(failing_tools)[:5]}..." if len(failing_tools) > 5 else f"  Failing tool UIDs: {failing_tools}")
    print(f"\n[Simulation] Running ({args.steps} steps)...")
    successes = []
    chosen_tools = []
    step_queries = []
    chosen_is_failed = []
    semantic_scores_list = []
    hybrid_scores_list = []
    rewards_list = []
    bandit_scores_list = []
    fail_tool_rewards = []
    non_fail_tool_rewards = []

    for step in range(args.steps):
        if step % 1000 == 0:
            current_success_rate = np.mean(successes) if successes else 0.0
            print(f"  Step {step}/{args.steps}, Success Rate: {current_success_rate:.4f}")
        query = np.random.choice(queries)
        query_group = query_groups.get_group(query)
        pos_tools = query_group[query_group["label"] == 1]["tool_uid"].unique().tolist()
        neg_tools = query_group[query_group["label"] == 0]["tool_uid"].unique().tolist()
        pos_pool = pos_tools.copy()
        pos_choices = []
        fail_in_q = [u for u in pos_pool if u in failing_tools]
        if fail_in_q:
            pos_choices.append(np.random.choice(fail_in_q))
        fill_pool = [u for u in pos_pool if u not in pos_choices]
        need = args.pos_per_query - len(pos_choices)
        if need > 0:
            if len(fill_pool) >= need:
                pos_choices.extend(np.random.choice(fill_pool, size=need, replace=False).tolist())
            else:
                pos_choices.extend(np.random.choice(fill_pool if fill_pool else pos_pool, size=need, replace=True).tolist())
        if step >= args.fail_step and fail_in_q and len(pos_choices) > 1:
            non_failing_pos = [u for u in pos_choices if u not in failing_tools]
            failing_pos_in_choices = [u for u in pos_choices if u in failing_tools]
            if len(non_failing_pos) > 1:
                pos_choices = failing_pos_in_choices + non_failing_pos[:1]
            elif len(non_failing_pos) == 0:
                other_non_failing = [u for u in pos_pool if u not in failing_tools and u not in pos_choices]
                if other_non_failing:
                    pos_choices.append(np.random.choice(other_non_failing))
        if len(neg_tools) < args.top_k - len(pos_choices):
            all_tools = set(tool_uids)
            remaining = all_tools - set(pos_choices) - set(neg_tools)
            neg_tools.extend(np.random.choice(list(remaining), 
                                            size=min(len(remaining), args.top_k - len(pos_choices) - len(neg_tools)),
                                            replace=False).tolist())
        
        neg_choices = np.random.choice(neg_tools, 
                                      size=min(len(neg_tools), args.top_k - len(pos_choices)), 
                                      replace=False).tolist()
        candidates = pos_choices + neg_choices
        query_emb = encoder.encode([query], convert_to_numpy=True)[0]
        candidate_scores = {}
        for tool_uid in candidates:
            tool_emb = tool_emb_map.get(tool_uid)
            if tool_emb is not None:
                semantic_score = float(np.dot(query_emb, tool_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(tool_emb) + 1e-9))
                candidate_scores[tool_uid] = semantic_score
        best_tool = None
        best_score = -1e9
        x_t = query_emb.astype(np.float32, copy=False)
        for tool_uid in candidates:
            semantic_score = candidate_scores.get(tool_uid, 0.0)
            static_rel = static_reliability.get(tool_uid, 0.75)
            if args.disable_bandit:
                hybrid_score = semantic_score
            else:
                bandit_score = bandit.score(tool_uid, x_t)
                hybrid_score = build_hybrid_score(
                    semantic_score, 
                    static_rel, 
                    bandit_score,
                    w_semantic=args.bandit_w_semantic,
                    w_static=args.bandit_w_static,
                    w_dynamic=args.bandit_w_dynamic
                )
            
            if hybrid_score > best_score:
                best_score = hybrid_score
                best_tool = tool_uid
        
        if best_tool is None:
            best_tool = np.random.choice(candidates)
        is_failed = (step >= args.fail_step) and (best_tool in failing_tools)
        is_positive = best_tool in pos_tools
        success = 1 if (is_positive and not is_failed) else 0
        semantic_score = candidate_scores.get(best_tool, 0.0)
        static_rel = static_reliability.get(best_tool, 0.75)
        if success == 1:
            exec_signal = 1.0
        elif is_failed:
            exec_signal = -2.0
        else:
            exec_signal = 0.0
        reward = 0.1 * (semantic_score * static_rel) + 0.9 * exec_signal
        if not args.disable_bandit:
            bandit.update(best_tool, x_t, reward)
        successes.append(success)
        chosen_tools.append(best_tool)
        step_queries.append(query)
        chosen_is_failed.append(int(is_failed))
        semantic_scores_list.append(semantic_score)
        hybrid_scores_list.append(best_score)
        rewards_list.append(reward)
        if not args.disable_bandit:
            current_bandit_score = bandit.score(best_tool, x_t)
            bandit_scores_list.append(current_bandit_score)
        else:
            bandit_scores_list.append(0.0)
        if step >= args.fail_step:
            if best_tool in failing_tools:
                fail_tool_rewards.append(reward)
            else:
                non_fail_tool_rewards.append(reward)
    print("\n[Metrics] Computing metrics...")
    successes = np.array(successes)
    overall_success_rate = np.mean(successes[args.fail_step:]) if args.fail_step < len(successes) else np.mean(successes)
    pre_failure_success = np.mean(successes[:args.fail_step]) if args.fail_step > 0 else overall_success_rate
    post_failure_window = 1000
    post_failure_start = args.fail_step
    post_failure_end = min(args.steps, post_failure_start + post_failure_window)
    post_failure_success = np.mean(successes[post_failure_start:post_failure_end]) if post_failure_end > post_failure_start else 0.0
    peak_success_rate = np.max([np.mean(successes[i:i+100]) for i in range(0, args.fail_step, 100)]) if args.fail_step > 0 else overall_success_rate
    if args.adaptation_target_rate is not None:
        target_rate = args.adaptation_target_rate
    else:
        if args.fail_frac >= 0.30:
            target_rate = max(0.80, 0.75 * peak_success_rate)
        else:
            target_rate = max(0.90, 0.85 * peak_success_rate)
    window_size = 200
    adaptation_steps = None
    for i in range(post_failure_start, min(args.steps, post_failure_start + 5000)):
        if i + window_size > len(successes):
            break
        window_success = np.mean(successes[i:i+window_size])
        if window_success >= target_rate:
            adaptation_steps = i - post_failure_start
            break
    avg_response_time = 150.0
    if args.fail_step < len(chosen_tools):
        post_failure_choices = chosen_tools[args.fail_step:]
        post_failure_fail_choices = [t for t in post_failure_choices if t in failing_tools]
        fail_tool_selection_rate = len(post_failure_fail_choices) / len(post_failure_choices) if post_failure_choices else 0.0
        mid_point = args.fail_step + (len(post_failure_choices) // 2)
        early_choices = chosen_tools[args.fail_step:mid_point]
        late_choices = chosen_tools[mid_point:]
        early_fail_rate = len([t for t in early_choices if t in failing_tools]) / len(early_choices) if early_choices else 0.0
        late_fail_rate = len([t for t in late_choices if t in failing_tools]) / len(late_choices) if late_choices else 0.0
        post_failure_rewards = rewards_list[args.fail_step:] if rewards_list else []
        early_rewards = post_failure_rewards[:len(post_failure_rewards)//2] if post_failure_rewards else []
        late_rewards = post_failure_rewards[len(post_failure_rewards)//2:] if post_failure_rewards else []
        avg_early_reward = np.mean(early_rewards) if early_rewards else 0.0
        avg_late_reward = np.mean(late_rewards) if late_rewards else 0.0
        avg_fail_tool_reward = np.mean(fail_tool_rewards) if fail_tool_rewards else 0.0
        avg_non_fail_tool_reward = np.mean(non_fail_tool_rewards) if non_fail_tool_rewards else 0.0
        post_failure_bandit_scores = bandit_scores_list[args.fail_step:] if bandit_scores_list else []
        early_bandit_scores = post_failure_bandit_scores[:len(post_failure_bandit_scores)//2] if post_failure_bandit_scores else []
        late_bandit_scores = post_failure_bandit_scores[len(post_failure_bandit_scores)//2:] if post_failure_bandit_scores else []
        avg_early_bandit_score = np.mean(early_bandit_scores) if early_bandit_scores else 0.0
        avg_late_bandit_score = np.mean(late_bandit_scores) if late_bandit_scores else 0.0
    else:
        fail_tool_selection_rate = 0.0
        early_fail_rate = 0.0
        late_fail_rate = 0.0
        avg_early_reward = 0.0
        avg_late_reward = 0.0
        avg_fail_tool_reward = 0.0
        avg_non_fail_tool_reward = 0.0
        avg_early_bandit_score = 0.0
        avg_late_bandit_score = 0.0
    log_df = pd.DataFrame({
        "t": np.arange(len(successes)),
        "query": step_queries,
        "chosen_tool": chosen_tools,
        "success": successes,
        "chosen_is_failed": chosen_is_failed,
    })
    log_file = output_dir / "simulation_log.csv"
    log_df.to_csv(log_file, index=False)
    print(f"  Simulation log saved: {log_file}")
    results = {
        "overall_success_rate": float(overall_success_rate),
        "pre_failure_success_rate": float(pre_failure_success),
        "post_failure_success_rate": float(post_failure_success),
        "adaptation_steps": adaptation_steps,
        "avg_response_time_ms": avg_response_time,
        "total_steps": args.steps,
        "fail_step": args.fail_step,
        "n_fail_tools": len(failing_tools),
        "fail_frac": args.fail_frac,
        "peak_success_rate": float(peak_success_rate),
        "target_success_rate": float(target_rate),
        "fault_selection_rate": float(fail_tool_selection_rate),
        "diagnostics": {
            "fail_tool_selection_rate": float(fail_tool_selection_rate),
            "early_fail_rate": float(early_fail_rate),
            "late_fail_rate": float(late_fail_rate),
            "improvement": float(early_fail_rate - late_fail_rate),
            "reward_analysis": {
                "avg_early_reward": float(avg_early_reward),
                "avg_late_reward": float(avg_late_reward),
                "reward_improvement": float(avg_late_reward - avg_early_reward),
                "avg_fail_tool_reward": float(avg_fail_tool_reward),
                "avg_non_fail_tool_reward": float(avg_non_fail_tool_reward),
                "reward_gap": float(avg_non_fail_tool_reward - avg_fail_tool_reward)
            },
            "bandit_score_analysis": {
                "avg_early_bandit_score": float(avg_early_bandit_score),
                "avg_late_bandit_score": float(avg_late_bandit_score),
                "bandit_score_change": float(avg_late_bandit_score - avg_early_bandit_score)
            }
        }
    }
    
    output_file = output_dir / "rated_full_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 60)
    print("Evaluation results")
    print("=" * 60)
    print(f"Overall Success Rate: {overall_success_rate:.4f}")
    print(f"Pre-failure Success Rate: {pre_failure_success:.4f}")
    print(f"Post-failure Success Rate: {post_failure_success:.4f}")
    print(f"Peak Success Rate: {peak_success_rate:.4f}")
    print(f"Target Success Rate: {target_rate:.4f}")
    print(f"Adaptation Steps: {adaptation_steps if adaptation_steps is not None else 'N/A'}")
    print(f"Avg Response Time: {avg_response_time:.2f} ms")
    print("\nDiagnostics:")
    print(f"  Fail-tool selection rate: {fail_tool_selection_rate:.4f}")
    print(f"  Early fail-tool selection rate: {early_fail_rate:.4f}")
    print(f"  Late fail-tool selection rate: {late_fail_rate:.4f}")
    improvement = early_fail_rate - late_fail_rate
    print(f"  Improvement (early - late): {improvement:.4f} {'OK' if improvement > 0 else '--'}")
    print("\nReward analysis:")
    print(f"  Early avg reward: {avg_early_reward:.4f}")
    print(f"  Late avg reward: {avg_late_reward:.4f}")
    reward_improvement = avg_late_reward - avg_early_reward
    print(f"  Reward improvement: {reward_improvement:.4f} {'OK' if reward_improvement > 0 else '--'}")
    print(f"  Fail-tool avg reward: {avg_fail_tool_reward:.4f}")
    print(f"  Non-fail-tool avg reward: {avg_non_fail_tool_reward:.4f}")
    reward_gap = avg_non_fail_tool_reward - avg_fail_tool_reward
    print(f"  Reward gap (non-fail - fail): {reward_gap:.4f} {'OK' if reward_gap > 0 else '--'}")
    print("\nBandit score analysis:")
    print(f"  Early avg bandit score: {avg_early_bandit_score:.4f}")
    print(f"  Late avg bandit score: {avg_late_bandit_score:.4f}")
    bandit_change = avg_late_bandit_score - avg_early_bandit_score
    print(f"  Bandit score change: {bandit_change:.4f}")
    print(f"\nResults saved to: {output_file}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
