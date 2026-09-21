"""Matrix-factorization / two-tower style scorer (numpy SGD).

User and item towers are linear embeddings. Training minimizes a weighted
logistic loss on logged (user, item, reward) triples. Optional IPS weights
debias the training distribution toward the target policy support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import expit

from recsys_ope_lab.data.world import LoggedBanditDataset
from recsys_ope_lab.policies import ScorePolicy


@dataclass
class MFConfig:
    n_factors: int = 16
    lr: float = 0.05
    reg: float = 1e-3
    n_epochs: int = 12
    neg_ratio: int = 2
    seed: int = 0


class MatrixFactorizationScorer:
    """Implicit/explicit MF with negative sampling + optional sample weights."""

    def __init__(self, n_users: int, n_items: int, config: Optional[MFConfig] = None):
        self.n_users = n_users
        self.n_items = n_items
        self.config = config or MFConfig()
        rng = np.random.default_rng(self.config.seed)
        scale = 0.1
        k = self.config.n_factors
        self.user_factors = rng.normal(0, scale, size=(n_users, k))
        self.item_factors = rng.normal(0, scale, size=(n_items, k))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = 0.0

    def _predict_logit(self, u: int, i: int) -> float:
        return float(
            self.global_bias
            + self.user_bias[u]
            + self.item_bias[i]
            + self.user_factors[u] @ self.item_factors[i]
        )

    def fit(
        self,
        data: LoggedBanditDataset,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "MatrixFactorizationScorer":
        cfg = self.config
        rng = np.random.default_rng(cfg.seed + 1)
        n = len(data)
        weights = np.ones(n) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
        weights = weights / (weights.mean() + 1e-12)

        # Build training pairs: positives from clicks, random negatives
        pos_mask = data.rewards > 0.5
        pos_u = data.user_ids[pos_mask]
        pos_i = data.item_ids[pos_mask]
        pos_w = weights[pos_mask]
        if len(pos_u) == 0:
            # Fall back to treating reward as soft label on all rows
            pos_u, pos_i, pos_w = data.user_ids, data.item_ids, weights
            soft = True
            soft_y = data.rewards
        else:
            soft = False
            soft_y = None

        lr, reg = cfg.lr, cfg.reg
        for _epoch in range(cfg.n_epochs):
            order = rng.permutation(len(pos_u))
            for idx in order:
                u = int(pos_u[idx])
                i = int(pos_i[idx])
                w = float(pos_w[idx])
                y = float(soft_y[idx]) if soft else 1.0
                self._sgd_step(u, i, y, w, lr, reg)
                # Negatives
                for _ in range(cfg.neg_ratio):
                    j = int(rng.integers(0, self.n_items))
                    self._sgd_step(u, j, 0.0, w, lr, reg)
        return self

    def _sgd_step(self, u: int, i: int, y: float, w: float, lr: float, reg: float) -> None:
        logit = self._predict_logit(u, i)
        pred = expit(logit)
        err = w * (pred - y)  # gradient of BCE w.r.t. logit
        # Update biases
        self.global_bias -= lr * err
        self.user_bias[u] -= lr * (err + reg * self.user_bias[u])
        self.item_bias[i] -= lr * (err + reg * self.item_bias[i])
        uf = self.user_factors[u]
        it = self.item_factors[i]
        self.user_factors[u] = uf - lr * (err * it + reg * uf)
        self.item_factors[i] = it - lr * (err * uf + reg * it)

    def score_matrix(self) -> np.ndarray:
        scores = (
            self.user_factors @ self.item_factors.T
            + self.user_bias[:, None]
            + self.item_bias[None, :]
            + self.global_bias
        )
        return scores

    def as_policy(self) -> ScorePolicy:
        return ScorePolicy(self.score_matrix())

    def top_k_for_user(self, user_id: int, k: int) -> np.ndarray:
        s = self.score_matrix()[user_id]
        return np.argsort(-s)[:k]


class TwoTowerRetriever(MatrixFactorizationScorer):
    """Alias emphasizing the two-tower (user tower × item tower) view."""

    def retrieve(self, user_id: int, k: int) -> np.ndarray:
        return self.top_k_for_user(user_id, k)
