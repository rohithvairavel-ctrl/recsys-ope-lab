"""Simple online / offline metrics helpers."""

from __future__ import annotations

import numpy as np


def mean_reward(rewards: np.ndarray) -> float:
    return float(np.mean(rewards))


def ndcg_at_k(relevance: np.ndarray, k: int = 10) -> float:
    """relevance: ideal relevance scores in ranked order (first = top)."""
    rel = np.asarray(relevance[:k], dtype=np.float64)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.dot(rel, discounts))
    ideal = np.sort(relevance)[::-1][:k]
    idcg = float(np.dot(ideal, 1.0 / np.log2(np.arange(2, ideal.size + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def policy_regret(true_value: float, oracle_value: float) -> float:
    return float(oracle_value - true_value)
