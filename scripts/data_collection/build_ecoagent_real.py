#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build EcoAgent-Real dataset from ToolBench match results (stage2_verified).
Outputs JSONL compatible with EcoAgent-Syn format.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def load_toolbench_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = ['query_id', 'query', 'tool_uid']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"ToolBench file missing required column: {col}")
    if 'llm_label' in df.columns:
        df = df[df['llm_label'] == 1].copy()
        print(f"  - Kept {len(df)} records after LLM verification filter")
    df['intent_type'] = 'Action'
    df['source'] = 'ToolBench'
    return df


def create_ground_truth_list(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('query_id').agg({
        'tool_uid': lambda x: x.tolist(),
        'query': 'first',
        'intent_type': 'first',
        'source': 'first'
    }).reset_index()
    grouped.rename(columns={'tool_uid': 'ground_truth'}, inplace=True)
    return grouped


def convert_to_jsonl(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            item = {
                'query_id': int(row['query_id']),
                'query': str(row['query']),
                'ground_truth': row['ground_truth'] if isinstance(row['ground_truth'], list) else [row['ground_truth']],
                'intent_type': str(row['intent_type']),
                'source': str(row['source'])
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def generate_statistics(df: pd.DataFrame) -> Dict:
    n = len(df)
    action = len(df[df['intent_type'] == 'Action']) if 'intent_type' in df.columns else n
    info = len(df[df['intent_type'] == 'Info']) if 'intent_type' in df.columns else 0
    stats = {
        'total_queries': n,
        'action_oriented': action,
        'informational': info,
        'from_toolbench': len(df[df['source'] == 'ToolBench']) if 'source' in df.columns else n,
        'from_ms_marco': 0,
        'action_percentage': round(action / n * 100, 2) if n > 0 else 0,
        'info_percentage': round(info / n * 100, 2) if n > 0 else 0,
        'avg_tools_per_query': round(df['ground_truth'].apply(len).mean(), 2) if n > 0 else 0,
        'unique_tools': len(set([t for tools in df['ground_truth'] for t in tools])) if n > 0 else 0
    }
    return stats


def main():
    ap = argparse.ArgumentParser(description="Build EcoAgent-Real dataset (ToolBench only)")
    ap.add_argument(
        "--toolbench",
        type=str,
        default="data/ecoagent_real_toolbench/ecoagent_real_stage2_verified.csv",
        help="ToolBench match result CSV (e.g. ecoagent_real_stage2_verified.csv)"
    )
    ap.add_argument(
        "--output",
        type=str,
        default="data/ecoagent_real_toolbench/ecoagent_real_eval.jsonl",
        help="Output JSONL path"
    )
    ap.add_argument(
        "--stats-output",
        type=str,
        default="data/ecoagent_real_toolbench/ecoagent_real_eval_stats.json",
        help="Output stats JSON path"
    )
    args = ap.parse_args()

    print("[1/3] Loading ToolBench results...")
    toolbench_df = load_toolbench_results(args.toolbench)
    print(f"  - Loaded {len(toolbench_df)} ToolBench match records")

    print("[2/3] Building ground_truth per query...")
    final_df = create_ground_truth_list(toolbench_df)
    print(f"  - {len(final_df)} unique queries")

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"data/ecoagent_real_{timestamp}.jsonl")

    print(f"[3/3] Writing to {output_path}...")
    convert_to_jsonl(final_df, output_path)
    print(f"  - Saved {len(final_df)} records")

    stats = generate_statistics(final_df)
    if args.stats_output:
        stats_path = Path(args.stats_output)
    else:
        stats_path = output_path.parent / "ecoagent_real_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("EcoAgent-Real dataset build complete.")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Stats:  {stats_path}")
    print("\nStats:")
    print(f"  total_queries:       {stats['total_queries']}")
    print(f"  action_oriented:     {stats['action_oriented']} ({stats['action_percentage']}%)")
    print(f"  informational:       {stats['informational']} ({stats['info_percentage']}%)")
    print(f"  from_toolbench:      {stats['from_toolbench']}")
    print(f"  from_ms_marco:       {stats['from_ms_marco']}")
    print(f"  avg_tools_per_query: {stats['avg_tools_per_query']}")
    print(f"  unique_tools:        {stats['unique_tools']}")


if __name__ == "__main__":
    main()
