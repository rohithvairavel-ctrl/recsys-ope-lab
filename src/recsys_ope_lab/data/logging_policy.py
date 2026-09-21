"""Logging policies that generate biased interaction logs with known propensities."""

from __future__ import annotations

from typing import Optional

import numpy as np

from recsys_ope_lab.data.world import PreferenceWorld, LoggedBanditDataset
from recsys_ope_lab.policies import Policy, ScorePolicy, MixturePolicy, ActionDraw


class PopularityPolicy(ScorePolicy):
    """Recommend proportional to (or greedy on) global item popularity."""

    def __init__(self, world: PreferenceWorld):
        super().__init__(world.popularity_scores())


class SoftmaxLoggingPolicy(ScorePolicy):
    """Softmax over a base score matrix (e.g. popularity or a production scorer).

    Temperature controls exploration: low T → peaked (high bias, low variance
    for OPE inverse), high T → near-uniform (better overlap, flatter traffic).

    The temperature is part of the *policy definition*. Callers should not
    override it via ``probs(..., temperature=)``; that kwarg is ignored so
    OPE code paths that pass a default temperature stay consistent with
    ``policy_value`` / logging.
    """

    def __init__(self, score_matrix: np.ndarray, temperature: float = 1.0):
        super().__init__(score_matrix)
        self.temperature = float(temperature)

    def probs(self, user_id: int, temperature: float = 1.0) -> np.ndarray:
        return ScorePolicy.probs(self, user_id, temperature=self.temperature)

    def sample(
        self,
        user_id: int,
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> ActionDraw:
        return ScorePolicy.sample(self, user_id, rng, temperature=self.temperature)

    def propensity(self, user_id: int, item_id: int, temperature: float = 1.0) -> float:
        return ScorePolicy.propensity(self, user_id, item_id, temperature=self.temperature)


class EpsilonGreedyLoggingPolicy(MixturePolicy):
    """ε-greedy: exploit a base policy, explore uniform (or popularity)."""

    def __init__(
        self,
        exploit: Policy,
        n_items: int,
        epsilon: float = 0.1,
        explore: Optional[Policy] = None,
    ):
        if explore is None:
            # Uniform explore policy
            explore = ScorePolicy(np.ones((1, n_items)))  # scores() will broadcast poorly

            class _Uniform(Policy):
                def __init__(self, n: int):
                    self.n = n

                def scores(self, user_id: int) -> np.ndarray:
                    return np.ones(self.n)

                def probs(self, user_id: int, temperature: float = 1.0) -> np.ndarray:
                    return np.full(self.n, 1.0 / self.n)

            explore = _Uniform(n_items)
        super().__init__(exploit=exploit, explore=explore, epsilon=epsilon)


def log_bandit_data(
    world: PreferenceWorld,
    logging_policy: Policy,
    n_impressions: int,
    rng: Optional[np.random.Generator] = None,
    temperature: float = 1.0,
) -> LoggedBanditDataset:
    """Roll out ``logging_policy`` for ``n_impressions`` and record propensities.

    Users are sampled uniformly; the logging policy chooses an item; reward is
    drawn from the world. Features are attached for downstream reward models.
    """
    rng = rng or np.random.default_rng(world.seed + 17)
    users = rng.integers(0, world.n_users, size=n_impressions)
    items = np.empty(n_impressions, dtype=np.int64)
    props = np.empty(n_impressions, dtype=np.float64)
    rewards = np.empty(n_impressions, dtype=np.float64)

    # SoftmaxLoggingPolicy has its own temperature; others accept the kwarg
    for t, u in enumerate(users):
        u = int(u)
        if isinstance(logging_policy, SoftmaxLoggingPolicy):
            draw = logging_policy.sample(u, rng)
        else:
            draw = logging_policy.sample(u, rng, temperature=temperature)
        items[t] = draw.item_id
        props[t] = max(draw.propensity, 1e-12)
        rewards[t] = world.sample_reward(u, draw.item_id, rng=rng)

    feats = world.pair_features(users, items)
    return LoggedBanditDataset(
        user_ids=users.astype(np.int64),
        item_ids=items,
        rewards=rewards,
        propensities=props,
        features=feats,
    )
