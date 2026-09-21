"""Popularity-based candidate retrieval baseline."""

from __future__ import annotations

import numpy as np

from recsys_ope_lab.data.world import LoggedBanditDataset, PreferenceWorld
from recsys_ope_lab.policies import ScorePolicy


class PopularityRetriever:
    """Rank items by empirical (or true) popularity mass."""

    def __init__(self):
        self.item_scores = None  # type: Optional[np.ndarray]

    def fit_from_logs(self, data: LoggedBanditDataset, n_items: int) -> "PopularityRetriever":
        counts = np.bincount(data.item_ids, minlength=n_items).astype(np.float64)
        # Laplace smooth so cold items are not zeroed out entirely
        counts += 1.0
        self.item_scores = counts / counts.sum()
        return self

    def fit_from_world(self, world: PreferenceWorld) -> "PopularityRetriever":
        self.item_scores = world.item_popularity.copy()
        return self

    def score_matrix(self, n_users: int) -> np.ndarray:
        if self.item_scores is None:
            raise RuntimeError("call fit_* first")
        return np.broadcast_to(self.item_scores, (n_users, len(self.item_scores))).copy()

    def top_k(self, k: int) -> np.ndarray:
        if self.item_scores is None:
            raise RuntimeError("call fit_* first")
        return np.argsort(-self.item_scores)[:k]

    def as_policy(self, n_users: int) -> ScorePolicy:
        return ScorePolicy(self.score_matrix(n_users))
