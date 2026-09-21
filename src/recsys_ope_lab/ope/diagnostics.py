"""OPE diagnostics: propensity overlap, ESS, clipping sensitivity."""

from __future__ import annotations

from typing import Optional

import numpy as np

from recsys_ope_lab.ope.estimators import ips, snips, doubly_robust, OPEResult


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    s1 = w.sum()
    s2 = np.square(w).sum()
    if s2 <= 0:
        return 0.0
    return float((s1 * s1) / s2)


def propensity_summary(
    behavior_propensities: np.ndarray,
    target_propensities: np.ndarray,
    clip: Optional[float] = None,
) -> dict:
    b = np.maximum(np.asarray(behavior_propensities, dtype=np.float64), 1e-12)
    e = np.asarray(target_propensities, dtype=np.float64)
    w = e / b
    if clip is not None:
        w_clipped = np.minimum(w, float(clip))
    else:
        w_clipped = w
    return {
        "n": int(len(w)),
        "behavior_prop_min": float(b.min()),
        "behavior_prop_p01": float(np.quantile(b, 0.01)),
        "behavior_prop_median": float(np.median(b)),
        "weight_max": float(w.max()),
        "weight_p99": float(np.quantile(w, 0.99)),
        "weight_mean": float(w.mean()),
        "ess": effective_sample_size(w),
        "ess_clipped": effective_sample_size(w_clipped),
        "ess_ratio": effective_sample_size(w) / max(len(w), 1),
        "frac_weight_gt_10": float(np.mean(w > 10)),
        "frac_weight_gt_100": float(np.mean(w > 100)),
        "clip": clip,
    }


def overlap_report(
    behavior_propensities: np.ndarray,
    target_propensities: np.ndarray,
    ess_warn_ratio: float = 0.05,
) -> dict:
    """Human-readable overlap diagnosis for staff-level review."""
    summary = propensity_summary(behavior_propensities, target_propensities)
    warnings = []
    if summary["ess_ratio"] < ess_warn_ratio:
        warnings.append(
            f"ESS/n = {summary['ess_ratio']:.3f} < {ess_warn_ratio}: "
            "target policy is poorly supported by the logging policy."
        )
    if summary["weight_max"] > 1e3:
        warnings.append(
            f"Max importance weight = {summary['weight_max']:.1f}: "
            "IPS variance likely dominated by a few rows; prefer SNIPS/DR + clipping."
        )
    if summary["behavior_prop_min"] < 1e-6:
        warnings.append(
            "Near-zero logging propensities present; ensure actions in the target "
            "support were explorable under π_b (common support assumption)."
        )
    summary["warnings"] = warnings
    summary["ok"] = len(warnings) == 0
    return summary


def clipping_sensitivity(
    rewards: np.ndarray,
    behavior_propensities: np.ndarray,
    target_propensities: np.ndarray,
    clips: Optional[list[float]] = None,
    q_hat_logged: Optional[np.ndarray] = None,
    q_hat_target: Optional[np.ndarray] = None,
) -> list[dict]:
    """Sweep IPS/SNIPS/(DR) across clip thresholds to show bias–variance tradeoff."""
    if clips is None:
        clips = [None, 5.0, 10.0, 20.0, 50.0, 100.0]
    rows = []
    for c in clips:
        row = {"clip": c}
        for name, fn in [("ips", ips), ("snips", snips)]:
            res: OPEResult = fn(rewards, behavior_propensities, target_propensities, clip=c)
            row[f"{name}_estimate"] = res.estimate
            row[f"{name}_se"] = res.std_error
            row[f"{name}_ess"] = res.ess
        if q_hat_logged is not None and q_hat_target is not None:
            res = doubly_robust(
                rewards,
                behavior_propensities,
                target_propensities,
                q_hat_logged,
                q_hat_target,
                clip=c,
            )
            row["dr_estimate"] = res.estimate
            row["dr_se"] = res.std_error
            row["dr_ess"] = res.ess
        rows.append(row)
    return rows
