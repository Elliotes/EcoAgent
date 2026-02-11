#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward functions for bandit: r_t = I(a_t = a*) x Status(a_t, t).
I(a_t = a*): 1 if selected tool matches ground truth, 0 otherwise.
Status(a_t, t): 1 if tool is available and working, 0 if failed/timeout/unavailable.
"""

from typing import Dict, Optional, Callable
import numpy as np


def compute_reward_binary(
    selected_tool: str,
    ground_truth_tool: str,
    tool_status: int,
    failure_schedule: Optional[Dict] = None,
    current_step: Optional[int] = None
) -> float:
    semantic_correct = 1.0 if selected_tool == ground_truth_tool else 0.0
    if failure_schedule is not None and current_step is not None:
        if selected_tool in failure_schedule.get("failing_tools", []):
            failure_interval = failure_schedule.get("failure_interval", {})
            if selected_tool in failure_interval:
                start, end = failure_interval[selected_tool]
                if start <= current_step <= end:
                    operational_status = 0.0
                else:
                    operational_status = float(tool_status)
            else:
                operational_status = float(tool_status)
        else:
            operational_status = float(tool_status)
    else:
        operational_status = float(tool_status)
    reward = semantic_correct * operational_status
    return float(reward)


def compute_reward_with_latency_penalty(
    selected_tool: str,
    ground_truth_tool: str,
    tool_status: int,
    latency_ms: Optional[float] = None,
    timeout_threshold_ms: float = 5000.0,
    latency_penalty_weight: float = 0.1
) -> float:
    base_reward = compute_reward_binary(selected_tool, ground_truth_tool, tool_status)
    if base_reward == 0.0:
        return 0.0
    if latency_ms is not None:
        if latency_ms > timeout_threshold_ms:
            return 0.0
        latency_penalty = min(latency_ms / timeout_threshold_ms, 1.0)
        reward = base_reward - latency_penalty_weight * latency_penalty
        return max(0.0, reward)
    return base_reward


def compute_reward_hybrid(
    selected_tool: str,
    ground_truth_tool: str,
    tool_status: int,
    semantic_relevance: float,
    static_reliability: float,
    exec_signal_weight: float = 0.8
) -> float:
    if selected_tool == ground_truth_tool and tool_status == 1:
        exec_signal = 1.0
    elif tool_status == 0:
        exec_signal = -1.0
    else:
        exec_signal = 0.0
    relevance_reliability = semantic_relevance * static_reliability
    reward = (1.0 - exec_signal_weight) * relevance_reliability + exec_signal_weight * exec_signal
    return float(reward)


default_reward_function: Callable = compute_reward_binary
