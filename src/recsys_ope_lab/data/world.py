"""Synthetic preference world with popularity skew.

Users and items live in a shared latent space. True click/reward probability
is a logistic of the inner product plus item bias. Item *exposure popularity*
follows a Zipf-like distribution independent of (but correlated with) quality,
so logging policies that chase popularity create selection bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import expit


@dataclass
class LoggedBanditDataset:
    """Contextual bandit log: one recommended item per impression.

    Attributes
    ----------
    user_ids, item_ids : arrays of length n
        Chosen action under the logging policy.
    rewards : binary or real rewards (clicks / engagements).
    propensities : π_b(a|u) for the logged action (must be > 0).
    features : optional (n, d) side features for reward models.
    """

    user_ids: np.ndarray
    item_ids: np.ndarray
    rewards: np.ndarray
    propensities: np.ndarray
    features: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(len(self.user_ids))

    def subset(self, idx: np.ndarray) -> "LoggedBanditDataset":
        feats = None if self.features is None else self.features[idx]
        return LoggedBanditDataset(
            user_ids=self.user_ids[idx],
            item_ids=self.item_ids[idx],
            rewards=self.rewards[idx],
            propensities=self.propensities[idx],
            features=feats,
        )


class PreferenceWorld:
    """Ground-truth preference simulator.

    Parameters
    ----------
    n_users, n_items : catalog sizes
    n_factors : latent dimensionality
    popularity_alpha : Zipf exponent for item popularity mass (>1 → heavier head)
    quality_popularity_corr : how strongly item bias tracks popularity rank
    noise_scale : std of Gaussian noise on logit before sigmoid
    seed : RNG seed
    """

    def __init__(
        self,
        n_users: int = 500,
        n_items: int = 200,
        n_factors: int = 16,
        popularity_alpha: float = 1.2,
        quality_popularity_corr: float = 0.4,
        noise_scale: float = 0.15,
        seed: int = 0,
    ):
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_factors = int(n_factors)
        self.popularity_alpha = float(popularity_alpha)
        self.quality_popularity_corr = float(quality_popularity_corr)
        self.noise_scale = float(noise_scale)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        self.user_factors = self.rng.normal(0, 1.0, size=(n_users, n_factors))
        self.item_factors = self.rng.normal(0, 1.0, size=(n_items, n_factors))

        # Zipf popularity over items (rank 1 = most popular)
        ranks = np.arange(1, n_items + 1, dtype=np.float64)
        pop = ranks ** (-self.popularity_alpha)
        pop /= pop.sum()
        # Shuffle so popularity is not perfectly aligned with item index
        perm = self.rng.permutation(n_items)
        self.item_popularity = np.zeros(n_items, dtype=np.float64)
        self.item_popularity[perm] = pop
        self._pop_rank = np.argsort(-self.item_popularity)  # descending

        # Item bias: mix of quality noise + popularity signal
        quality = self.rng.normal(0, 1.0, size=n_items)
        pop_z = (self.item_popularity - self.item_popularity.mean()) / (
            self.item_popularity.std() + 1e-12
        )
        c = self.quality_popularity_corr
        self.item_bias = c * pop_z + np.sqrt(max(1.0 - c * c, 0.0)) * quality
        self.user_bias = self.rng.normal(0, 0.3, size=n_users)

        # Cache true relevance matrix P(click | u, i) without impression noise
        logits = (
            self.user_factors @ self.item_factors.T
            + self.user_bias[:, None]
            + self.item_bias[None, :]
        )
        # Scale logits into a reasonable click-rate band
        logits = logits / np.sqrt(n_factors)
        self.true_relevance = expit(logits)

    def relevance(self, user_id: int, item_id: int) -> float:
        return float(self.true_relevance[user_id, item_id])

    def expected_reward(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        return self.true_relevance[user_id, np.asarray(item_ids)]

    def sample_reward(
        self,
        user_id: int,
        item_id: int,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Bernoulli click with optional logit noise (unobserved confounder proxy)."""
        rng = rng or self.rng
        p = self.true_relevance[user_id, item_id]
        if self.noise_scale > 0:
            # Small logit noise → slightly different realized propensity of click
            logit = np.log(p / (1 - p + 1e-12) + 1e-12)
            logit = logit + rng.normal(0, self.noise_scale)
            p = float(expit(logit))
        return float(rng.random() < p)

    def oracle_scores(self) -> np.ndarray:
        """Return the true relevance matrix as scores (for oracle policy)."""
        return self.true_relevance.copy()

    def popularity_scores(self) -> np.ndarray:
        """Broadcast item popularity as user-independent scores."""
        return np.broadcast_to(self.item_popularity, (self.n_users, self.n_items)).copy()

    def pair_features(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Build simple user–item features for reward / ranking models.

        Features: user factors, item factors, elementwise product, biases,
        popularity, and interaction count proxies.
        """
        u = self.user_factors[user_ids]
        i = self.item_factors[item_ids]
        prod = u * i
        ub = self.user_bias[user_ids][:, None]
        ib = self.item_bias[item_ids][:, None]
        pop = self.item_popularity[item_ids][:, None]
        return np.hstack([u, i, prod, ub, ib, pop])

    def policy_value(self, policy, n_eval_users: Optional[int] = None) -> float:
        """True expected reward of a (stochastic) policy — online oracle.

        Averages E_{a~π(·|u)}[r(u,a)] over users (or a subset).
        """
        from recsys_ope_lab.policies import Policy

        if not isinstance(policy, Policy):
            raise TypeError("policy must be a Policy")
        users = (
            np.arange(self.n_users)
            if n_eval_users is None
            else np.arange(min(n_eval_users, self.n_users))
        )
        total = 0.0
        for u in users:
            p = policy.probs(int(u))
            total += float(np.dot(p, self.true_relevance[u]))
        return total / len(users)

    def greedy_policy_value(self, score_matrix: np.ndarray) -> float:
        """Expected reward of the deterministic argmax policy on ``score_matrix``."""
        picks = np.argmax(score_matrix, axis=1)
        return float(self.true_relevance[np.arange(self.n_users), picks].mean())
