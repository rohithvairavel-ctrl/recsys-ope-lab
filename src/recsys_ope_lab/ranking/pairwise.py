"""Pairwise RankNet-style scorer on logged pairs.

For each user with at least one positive and one negative in the log (or
sampled negatives), train a linear model so that score(pos) > score(neg).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from recsys_ope_lab.data.world import LoggedBanditDataset, PreferenceWorld
from recsys_ope_lab.policies import ScorePolicy


class PairwiseRanker:
    def __init__(self, C: float = 1.0, max_pairs: int = 20000, seed: int = 0):
        self.C = C
        self.max_pairs = max_pairs
        self.seed = seed
        self.model: Optional[LogisticRegression] = None

    def fit(
        self,
        data: LoggedBanditDataset,
        world: PreferenceWorld,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "PairwiseRanker":
        if data.features is None:
            raise ValueError("features required")
        rng = np.random.default_rng(self.seed)
        pos = np.where(data.rewards > 0.5)[0]
        neg = np.where(data.rewards <= 0.5)[0]
        if len(pos) == 0 or len(neg) == 0:
            # Degenerate: fall back to pointwise labels on feature diffs vs mean
            X = data.features
            y = (data.rewards > data.rewards.mean()).astype(np.int64)
            self.model = LogisticRegression(C=self.C, max_iter=500, random_state=self.seed)
            self.model.fit(X, y, sample_weight=sample_weight)
            return self

        n_pairs = min(self.max_pairs, len(pos) * 2)
        pos_idx = rng.choice(pos, size=n_pairs, replace=True)
        neg_idx = rng.choice(neg, size=n_pairs, replace=True)
        X = data.features[pos_idx] - data.features[neg_idx]
        y = np.ones(n_pairs, dtype=np.int64)
        # Add reverse pairs for balance
        X = np.vstack([X, -X])
        y = np.concatenate([y, np.zeros(n_pairs, dtype=np.int64)])
        w = None
        if sample_weight is not None:
            w = np.concatenate([sample_weight[pos_idx], sample_weight[pos_idx]])
        self.model = LogisticRegression(C=self.C, max_iter=500, random_state=self.seed)
        self.model.fit(X, y, sample_weight=w)
        return self

    def score_features(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("not fitted")
        # Decision function on absolute features ≈ relevance score
        return self.model.decision_function(features)

    def build_full_score_matrix(self, world: PreferenceWorld) -> np.ndarray:
        """Score all user–item pairs (dense; fine for synthetic catalog sizes)."""
        if self.model is None:
            raise RuntimeError("not fitted")
        scores = np.empty((world.n_users, world.n_items), dtype=np.float64)
        # Batch by user to limit memory
        for u in range(world.n_users):
            items = np.arange(world.n_items)
            users = np.full(world.n_items, u)
            feats = world.pair_features(users, items)
            scores[u] = self.score_features(feats)
        return scores

    def as_policy(self, world: PreferenceWorld) -> ScorePolicy:
        return ScorePolicy(self.build_full_score_matrix(world))
