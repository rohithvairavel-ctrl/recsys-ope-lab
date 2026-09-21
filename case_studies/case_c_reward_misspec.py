"""Case C — Reward-model misspecification hurts Doubly Robust.

DR is consistent if *either* propensities or q̂ is correct. When overlap is
mediocre *and* q̂ is badly misspecified (here: a constant mean predictor),
DR can be worse than SNIPS. A well-specified q̂ restores DR's advantage.
"""

from __future__ import annotations

import numpy as np

from recsys_ope_lab.data.world import PreferenceWorld
from recsys_ope_lab.data.logging_policy import (
    SoftmaxLoggingPolicy,
    EpsilonGreedyLoggingPolicy,
    PopularityPolicy,
    log_bandit_data,
)
from recsys_ope_lab.ope.estimators import (
    ips,
    snips,
    doubly_robust,
    target_propensities_for_logged_actions,
)
from recsys_ope_lab.ope.reward_model import RewardModel


def run(seed: int = 42, n_impressions: int = 7000) -> dict:
    world = PreferenceWorld(
        n_users=280,
        n_items=110,
        n_factors=12,
        popularity_alpha=1.3,
        quality_popularity_corr=0.3,
        seed=seed,
    )
    rng = np.random.default_rng(seed)

    pop = PopularityPolicy(world)
    behavior = EpsilonGreedyLoggingPolicy(exploit=pop, n_items=world.n_items, epsilon=0.08)
    data = log_bandit_data(world, behavior, n_impressions=n_impressions, rng=rng)

    target = SoftmaxLoggingPolicy(world.oracle_scores(), temperature=0.55)
    v_target = world.policy_value(target)
    pi_e = target_propensities_for_logged_actions(target, data.user_ids, data.item_ids)

    # Good reward model
    good = RewardModel(kind="logistic", seed=seed)
    good.fit(data)
    q_log_good = good.predict_pairs(world, data.user_ids, data.item_ids)
    q_tgt_good = good.predict_policy_value_per_context(world, data.user_ids, target)

    # Misspecified: constant = mean logged reward (ignores context/action)
    mean_r = float(data.rewards.mean())
    q_log_bad = np.full(len(data), mean_r)
    q_tgt_bad = np.full(len(data), mean_r)

    clip = 25.0
    res = {
        "V_target_oracle": v_target,
        "ips": ips(data.rewards, data.propensities, pi_e, clip=clip).as_dict(),
        "snips": snips(data.rewards, data.propensities, pi_e, clip=clip).as_dict(),
        "dr_good_q": doubly_robust(
            data.rewards, data.propensities, pi_e, q_log_good, q_tgt_good, clip=clip
        ).as_dict(),
        "dr_bad_q": doubly_robust(
            data.rewards, data.propensities, pi_e, q_log_bad, q_tgt_bad, clip=clip
        ).as_dict(),
    }
    res["errors"] = {
        "ips": abs(res["ips"]["estimate"] - v_target),
        "snips": abs(res["snips"]["estimate"] - v_target),
        "dr_good_q": abs(res["dr_good_q"]["estimate"] - v_target),
        "dr_bad_q": abs(res["dr_bad_q"]["estimate"] - v_target),
    }
    res["takeaway"] = (
        "Double robustness is not magic: under poor overlap, a misspecified "
        "reward model can make DR worse than SNIPS. Validate q̂; report both."
    )
    return res


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
