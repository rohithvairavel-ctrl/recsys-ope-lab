"""Policy abstractions for logging and target recommendation policies.

Policies map a user id to a categorical distribution over items (or a
deterministic argmax). Logged data stores the *propensity* of the chosen
action under the logging policy so OPE estimators can reweight.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ActionDraw:
    """One logged recommendation action for a user."""

    user_id: int
    item_id: int
    propensity: float
    scores: Optional[np.ndarray] = None  # full score vector if available


class Policy(ABC):
    """Base class: score items for a user and optionally sample."""

    @abstractmethod
    def scores(self, user_id: int) -> np.ndarray:
        """Return unnormalized scores over all items for ``user_id``."""

    def probs(self, user_id: int, temperature: float = 1.0) -> np.ndarray:
        """Softmax of scores with temperature (higher = flatter)."""
        s = self.scores(user_id).astype(np.float64)
        t = max(float(temperature), 1e-8)
        z = s / t
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def sample(
        self,
        user_id: int,
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> ActionDraw:
        p = self.probs(user_id, temperature=temperature)
        item = int(rng.choice(len(p), p=p))
        return ActionDraw(user_id=user_id, item_id=item, propensity=float(p[item]), scores=p)

    def greedy(self, user_id: int) -> int:
        return int(np.argmax(self.scores(user_id)))

    def propensity(self, user_id: int, item_id: int, temperature: float = 1.0) -> float:
        return float(self.probs(user_id, temperature=temperature)[item_id])


class ScorePolicy(Policy):
    """Policy backed by an explicit (n_users, n_items) score matrix."""

    def __init__(self, score_matrix: np.ndarray):
        if score_matrix.ndim != 2:
            raise ValueError("score_matrix must be 2-D")
        self.score_matrix = np.asarray(score_matrix, dtype=np.float64)

    def scores(self, user_id: int) -> np.ndarray:
        return self.score_matrix[user_id]


class MixturePolicy(Policy):
    """Mixture of two policies: with prob ``epsilon`` use ``explore``, else ``exploit``.

    Propensity under the mixture is closed-form:
    ``π(a|u) = (1-ε)·π_exploit(a|u) + ε·π_explore(a|u)``.
    """

    def __init__(self, exploit: Policy, explore: Policy, epsilon: float):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.exploit = exploit
        self.explore = explore
        self.epsilon = float(epsilon)

    def scores(self, user_id: int) -> np.ndarray:
        # Scores are not well-defined for a mixture; expose exploit scores for greedy.
        return self.exploit.scores(user_id)

    def probs(self, user_id: int, temperature: float = 1.0) -> np.ndarray:
        pe = self.exploit.probs(user_id, temperature=temperature)
        px = self.explore.probs(user_id, temperature=temperature)
        return (1.0 - self.epsilon) * pe + self.epsilon * px

    def sample(
        self,
        user_id: int,
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> ActionDraw:
        p = self.probs(user_id, temperature=temperature)
        item = int(rng.choice(len(p), p=p))
        return ActionDraw(user_id=user_id, item_id=item, propensity=float(p[item]), scores=p)
