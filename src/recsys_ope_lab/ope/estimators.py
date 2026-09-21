"""Off-policy evaluation estimators: IPS, SNIPS, Doubly Robust.

Setup (contextual bandit): logged data under behavior policy π_b, evaluate
target policy π_e. For each impression t with context x_t, action a_t, reward
r_t, and known propensity π_b(a_t|x_t):

  importance weight  w_t = π_e(a_t|x_t) / π_b(a_t|x_t)

  IPS:   V̂ = (1/n) Σ w_t r_t
  SNIPS: V̂ = Σ w_t r_t / Σ w_t
  DR:    V̂ = (1/n) Σ [ q̂(x_t, π_e) + w_t (r_t - q̂(x_t, a_t)) ]

where q̂ is a reward model. DR remains consistent if *either* the propensities
or the reward model is correct (double robustness).

Variance / CIs use the empirical variance of the per-sample contributions
( asymptotic N(0, σ²/n) ), which is standard but fails under extreme weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from recsys_ope_lab.policies import Policy


@dataclass
class OPEResult:
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    n: int
    method: str
    ess: float
    extra: Optional[dict] = None

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "estimate": self.estimate,
            "std_error": self.std_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "ess": self.ess,
            **(self.extra or {}),
        }


def _importance_weights(
    target_propensities: np.ndarray,
    behavior_propensities: np.ndarray,
    clip: Optional[float] = None,
) -> np.ndarray:
    b = np.maximum(np.asarray(behavior_propensities, dtype=np.float64), 1e-12)
    w = np.asarray(target_propensities, dtype=np.float64) / b
    if clip is not None:
        w = np.minimum(w, float(clip))
    return w


def _ess(w: np.ndarray) -> float:
    """Kish effective sample size: (Σ w)² / Σ w²."""
    s1 = w.sum()
    s2 = np.square(w).sum()
    if s2 <= 0:
        return 0.0
    return float((s1 * s1) / s2)


def _result_from_samples(samples: np.ndarray, method: str, ess: float, extra: Optional[dict] = None) -> OPEResult:
    n = len(samples)
    est = float(samples.mean()) if n else float("nan")
    # Sample std of contributions; SE = s / sqrt(n)
    if n > 1:
        se = float(samples.std(ddof=1) / np.sqrt(n))
    else:
        se = float("inf")
    z = stats.norm.ppf(0.975)
    return OPEResult(
        estimate=est,
        std_error=se,
        ci_low=est - z * se,
        ci_high=est + z * se,
        n=n,
        method=method,
        ess=ess,
        extra=extra,
    )


def target_propensities_for_logged_actions(
    target: Policy,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """π_e(a_t | u_t) for each logged row."""
    out = np.empty(len(user_ids), dtype=np.float64)
    # Cache per-user probs
    cache: dict[int, np.ndarray] = {}
    for t, (u, i) in enumerate(zip(user_ids, item_ids)):
        u = int(u)
        if u not in cache:
            cache[u] = target.probs(u, temperature=temperature)
        out[t] = cache[u][int(i)]
    return out


def target_value_from_reward_model(
    reward_hat_on_target: np.ndarray,
) -> np.ndarray:
    """Per-row q̂(x, π_e). Caller supplies already-evaluated expectations."""
    return np.asarray(reward_hat_on_target, dtype=np.float64)


def ips(
    rewards: np.ndarray,
    behavior_propensities: np.ndarray,
    target_propensities: np.ndarray,
    clip: Optional[float] = None,
) -> OPEResult:
    """Inverse Propensity Scoring."""
    w = _importance_weights(target_propensities, behavior_propensities, clip=clip)
    samples = w * np.asarray(rewards, dtype=np.float64)
    return _result_from_samples(samples, method="IPS", ess=_ess(w), extra={"clip": clip})


def snips(
    rewards: np.ndarray,
    behavior_propensities: np.ndarray,
    target_propensities: np.ndarray,
    clip: Optional[float] = None,
) -> OPEResult:
    """Self-Normalized IPS (reduces variance; small bias).

    Variance uses the delta-method / influence-function approximation for the
    ratio estimator: φ = (w r - V̂ w) / w̄.
    """
    w = _importance_weights(target_propensities, behavior_propensities, clip=clip)
    r = np.asarray(rewards, dtype=np.float64)
    w_sum = w.sum()
    if w_sum <= 0:
        return OPEResult(
            estimate=float("nan"),
            std_error=float("inf"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=len(r),
            method="SNIPS",
            ess=0.0,
            extra={"clip": clip},
        )
    est = float((w * r).sum() / w_sum)
    # Influence function for the ratio estimator (mean-zero); use for SE only.
    phi = (w * r - est * w) / (w.mean() + 1e-12)
    n = len(r)
    if n > 1:
        se = float(phi.std(ddof=1) / np.sqrt(n))
    else:
        se = float("inf")
    z = stats.norm.ppf(0.975)
    return OPEResult(
        estimate=est,
        std_error=se,
        ci_low=est - z * se,
        ci_high=est + z * se,
        n=n,
        method="SNIPS",
        ess=_ess(w),
        extra={"clip": clip},
    )


def doubly_robust(
    rewards: np.ndarray,
    behavior_propensities: np.ndarray,
    target_propensities: np.ndarray,
    q_hat_logged: np.ndarray,
    q_hat_target: np.ndarray,
    clip: Optional[float] = None,
) -> OPEResult:
    """Doubly Robust: q̂(x, π_e) + w (r - q̂(x, a)).

    ``q_hat_logged`` : q̂(x_t, a_t) for the logged action
    ``q_hat_target`` : E_{a~π_e}[q̂(x_t, a)]  (or q̂(x_t, π_e(x_t)) for deterministic)
    """
    w = _importance_weights(target_propensities, behavior_propensities, clip=clip)
    r = np.asarray(rewards, dtype=np.float64)
    q_a = np.asarray(q_hat_logged, dtype=np.float64)
    q_e = np.asarray(q_hat_target, dtype=np.float64)
    samples = q_e + w * (r - q_a)
    return _result_from_samples(samples, method="DR", ess=_ess(w), extra={"clip": clip})


def naive_on_policy(rewards: np.ndarray) -> OPEResult:
    """Naive mean reward on the *logged* policy — NOT a valid estimate of π_e.

    Included explicitly so case studies can show how badly it lies when the
    logging policy differs from the candidate policy.
    """
    r = np.asarray(rewards, dtype=np.float64)
    return _result_from_samples(r, method="naive_logged_mean", ess=float(len(r)))


def evaluate_policy_ope(
    rewards: np.ndarray,
    behavior_propensities: np.ndarray,
    target_propensities: np.ndarray,
    q_hat_logged: Optional[np.ndarray] = None,
    q_hat_target: Optional[np.ndarray] = None,
    clip: Optional[float] = None,
) -> dict[str, OPEResult]:
    """Run IPS, SNIPS, and (if reward model provided) DR; plus naive baseline."""
    out = {
        "naive": naive_on_policy(rewards),
        "ips": ips(rewards, behavior_propensities, target_propensities, clip=clip),
        "snips": snips(rewards, behavior_propensities, target_propensities, clip=clip),
    }
    if q_hat_logged is not None and q_hat_target is not None:
        out["dr"] = doubly_robust(
            rewards,
            behavior_propensities,
            target_propensities,
            q_hat_logged,
            q_hat_target,
            clip=clip,
        )
    return out
