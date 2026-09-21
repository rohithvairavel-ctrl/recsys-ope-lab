"""recsys-ope-lab: synthetic recommender logging + off-policy evaluation.

Signals the Google recsys / DS hiring bar around logging policies, selection
bias, OPE estimators (IPS / SNIPS / DR), and offline→online honesty.
"""

__version__ = "0.1.0"

from recsys_ope_lab.data.world import PreferenceWorld
from recsys_ope_lab.data.logging_policy import SoftmaxLoggingPolicy, EpsilonGreedyLoggingPolicy
from recsys_ope_lab.ope.estimators import ips, snips, doubly_robust, OPEResult

__all__ = [
    "__version__",
    "PreferenceWorld",
    "SoftmaxLoggingPolicy",
    "EpsilonGreedyLoggingPolicy",
    "ips",
    "snips",
    "doubly_robust",
    "OPEResult",
]
