"""Reward model q̂(x, a) for Doubly Robust OPE."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from recsys_ope_lab.data.world import LoggedBanditDataset, PreferenceWorld
from recsys_ope_lab.policies import Policy


class RewardModel:
    """Predict E[r | user, item] from pair features.

    Uses logistic regression for binary clicks, or ridge for continuous rewards.
    """

    def __init__(self, kind: str = "logistic", C: float = 1.0, seed: int = 0):
        if kind not in {"logistic", "ridge"}:
            raise ValueError("kind must be 'logistic' or 'ridge'")
        self.kind = kind
        self.C = C
        self.seed = seed
        self.model = None

    def fit(
        self,
        data: LoggedBanditDataset,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "RewardModel":
        if data.features is None:
            raise ValueError("features required")
        X = data.features
        y = data.rewards
        if self.kind == "logistic":
            labels = (y > 0.5).astype(np.int64)
            if labels.min() == labels.max():
                labels = (y >= y.mean()).astype(np.int64)
            self.model = LogisticRegression(
                C=self.C, max_iter=500, random_state=self.seed, solver="lbfgs"
            )
            self.model.fit(X, labels, sample_weight=sample_weight)
        else:
            self.model = Ridge(alpha=1.0 / max(self.C, 1e-6), random_state=self.seed)
            self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("not fitted")
        if self.kind == "logistic":
            return self.model.predict_proba(features)[:, 1]
        return self.model.predict(features)

    def predict_pairs(
        self, world: PreferenceWorld, user_ids: np.ndarray, item_ids: np.ndarray
    ) -> np.ndarray:
        feats = world.pair_features(user_ids, item_ids)
        return self.predict(feats)

    def predict_policy_value_per_context(
        self,
        world: PreferenceWorld,
        user_ids: np.ndarray,
        target: Policy,
        temperature: float = 1.0,
        max_items_mc: Optional[int] = None,
    ) -> np.ndarray:
        """Estimate q̂(x, π_e) = Σ_a π_e(a|x) q̂(x, a) for each logged user.

        For small catalogs we sum exactly; for large, Monte-Carlo over items.
        """
        out = np.empty(len(user_ids), dtype=np.float64)
        cache: dict[int, float] = {}
        n_items = world.n_items
        for t, u in enumerate(user_ids):
            u = int(u)
            if u in cache:
                out[t] = cache[u]
                continue
            probs = target.probs(u, temperature=temperature)
            if max_items_mc is not None and max_items_mc < n_items:
                # MC: sample items from target, average q̂
                rng = np.random.default_rng(self.seed + u)
                sample_i = rng.choice(n_items, size=max_items_mc, p=probs)
                users = np.full(max_items_mc, u)
                q = self.predict_pairs(world, users, sample_i)
                val = float(q.mean())
            else:
                users = np.full(n_items, u)
                items = np.arange(n_items)
                q = self.predict_pairs(world, users, items)
                val = float(np.dot(probs, q))
            cache[u] = val
            out[t] = val
        return out

    @staticmethod
    def misspecified_constant(mean_reward: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return constant q̂ for misspecification case studies."""
        q = np.full(n, mean_reward, dtype=np.float64)
        return q, q.copy()
