"""Pointwise ranking on a retrieved candidate set.

Fits a logistic regression on pair features → P(click). At inference, scores
candidates and returns a sorted list (or a softmax policy over the candidate
set, with residual mass on non-candidates for propensity computation).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from recsys_ope_lab.data.world import LoggedBanditDataset, PreferenceWorld
from recsys_ope_lab.policies import Policy, ScorePolicy


class PointwiseRanker:
    def __init__(self, C: float = 1.0, seed: int = 0):
        self.C = C
        self.seed = seed
        self.model: Optional[LogisticRegression] = None

    def fit(
        self,
        data: LoggedBanditDataset,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "PointwiseRanker":
        if data.features is None:
            raise ValueError("LoggedBanditDataset.features required for pointwise fit")
        y = (data.rewards > 0.5).astype(np.int64)
        # Need both classes
        if y.min() == y.max():
            y = (data.rewards >= data.rewards.mean()).astype(np.int64)
        self.model = LogisticRegression(
            C=self.C,
            max_iter=500,
            random_state=self.seed,
            solver="lbfgs",
        )
        self.model.fit(data.features, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("not fitted")
        return self.model.predict_proba(features)[:, 1]

    def score_candidates(
        self,
        world: PreferenceWorld,
        user_id: int,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        u = np.full(len(candidate_items), user_id, dtype=np.int64)
        feats = world.pair_features(u, candidate_items)
        return self.predict_proba(feats)

    def build_full_score_matrix(
        self,
        world: PreferenceWorld,
        candidate_fn,
        top_k: int = 50,
    ) -> np.ndarray:
        """Score top-k candidates per user; non-candidates get a low floor score.

        ``candidate_fn(user_id) -> array of item ids``.
        """
        scores = np.full((world.n_users, world.n_items), -1e6, dtype=np.float64)
        for u in range(world.n_users):
            cands = np.asarray(candidate_fn(u), dtype=np.int64)[:top_k]
            if len(cands) == 0:
                continue
            s = self.score_candidates(world, u, cands)
            scores[u, cands] = s
        return scores

    def as_policy_from_matrix(self, score_matrix: np.ndarray) -> ScorePolicy:
        return ScorePolicy(score_matrix)
