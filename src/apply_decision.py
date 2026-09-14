"""
Reads results/diode/diode_comparison.json and results/da2k/da2k_comparison.json
(produced by run_track_a_diode.py and run_track_b_da2k.py) and applies the
pre-registered decision rule from src/decision_rule.py.

Usage:
    python -m src.apply_decision --results-dir results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.decision_rule import DecisionInputs, apply_decision_rule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    with open(results_dir / "diode" / "diode_comparison.json") as f:
        diode = json.load(f)
    with open(results_dir / "da2k" / "da2k_comparison.json") as f:
        da2k = json.load(f)

    inputs = DecisionInputs(
        delta_da2k_pp=da2k["delta_da2k_pp"],
        delta_abs_rel=diode["delta_abs_rel"],
        delta_delta1=diode["delta_delta1"],
    )
    verdict, reason = apply_decision_rule(inputs)

    out = {
        "verdict": verdict.value,
        "reason": reason,
        "inputs": inputs.__dict__,
    }
    print(json.dumps(out, indent=2))

    with open(results_dir / "final_verdict.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
