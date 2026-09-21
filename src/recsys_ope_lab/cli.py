"""CLI entrypoint: run the three case studies end-to-end."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recsys-ope",
        description="Recommender + off-policy evaluation lab case studies",
    )
    parser.add_argument(
        "case",
        nargs="?",
        default="all",
        choices=["all", "a", "b", "c"],
        help="Which case study to run (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from case_studies import case_a_naive_vs_ope, case_b_overlap_clipping, case_c_reward_misspec

    runners = {
        "a": case_a_naive_vs_ope.run,
        "b": case_b_overlap_clipping.run,
        "c": case_c_reward_misspec.run,
    }
    selected = list(runners.keys()) if args.case == "all" else [args.case]
    results = {}
    for key in selected:
        print(f"\n===== Case {key.upper()} =====")
        results[key] = runners[key](seed=args.seed)
        print(json.dumps(results[key], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
