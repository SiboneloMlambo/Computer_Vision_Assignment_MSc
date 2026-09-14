"""
Predefined decision rule (pre-registered before results are seen).

Thresholds, verbatim from the proposal:
    Strong reproduction : DeltaDA2K >= +5.0 pp
                           AND |DeltaAbsRel| <= 0.01
                           AND |Deltadelta1| <= 0.02
    Partial reproduction: predicted direction (DeltaDA2K > 0, DIODE roughly flat)
                           but exactly one magnitude threshold is missed.
    Not reproduced      : no meaningful DA-2K-vs-DIODE asymmetry.

Do not edit these thresholds after seeing results -- that is the
whole point of pre-registering them. If they turn out to be wrong for
the data you collect, say so in run_log.md instead of moving the goalposts.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(Enum):
    STRONG = "strong reproduction"
    PARTIAL = "partial reproduction"
    NOT_REPRODUCED = "not reproduced"


@dataclass
class DecisionInputs:
    delta_da2k_pp: float          # Accuracy(V2-S) - Accuracy(V1-S), in percentage points
    delta_abs_rel: float          # AbsRel(V2-S) - AbsRel(V1-S)
    delta_delta1: float           # delta1(V2-S) - delta1(V1-S)

    # Thresholds (override only before running, never after seeing results)
    da2k_threshold_pp: float = 5.0
    abs_rel_threshold: float = 0.01
    delta1_threshold: float = 0.02


def apply_decision_rule(inputs: DecisionInputs) -> tuple[Verdict, str]:
    da2k_pass = inputs.delta_da2k_pp >= inputs.da2k_threshold_pp
    abs_rel_pass = abs(inputs.delta_abs_rel) <= inputs.abs_rel_threshold
    delta1_pass = abs(inputs.delta_delta1) <= inputs.delta1_threshold

    direction_ok = inputs.delta_da2k_pp > 0

    checks_passed = sum([da2k_pass, abs_rel_pass, delta1_pass])

    if da2k_pass and abs_rel_pass and delta1_pass:
        verdict = Verdict.STRONG
        reason = (
            f"DA-2K gain {inputs.delta_da2k_pp:+.1f}pp >= {inputs.da2k_threshold_pp}pp, "
            f"|dAbsRel|={abs(inputs.delta_abs_rel):.4f} <= {inputs.abs_rel_threshold}, "
            f"|d-delta1|={abs(inputs.delta_delta1):.4f} <= {inputs.delta1_threshold}."
        )
    elif direction_ok and checks_passed == 2:
        verdict = Verdict.PARTIAL
        reason = (
            f"Predicted direction present (DA-2K gain {inputs.delta_da2k_pp:+.1f}pp) "
            f"but exactly one magnitude threshold missed: "
            f"da2k_pass={da2k_pass}, abs_rel_pass={abs_rel_pass}, delta1_pass={delta1_pass}."
        )
    else:
        verdict = Verdict.NOT_REPRODUCED
        reason = (
            f"No meaningful DA-2K-vs-DIODE asymmetry: "
            f"DA-2K gain {inputs.delta_da2k_pp:+.1f}pp, "
            f"dAbsRel={inputs.delta_abs_rel:+.4f}, d-delta1={inputs.delta_delta1:+.4f}."
        )

    return verdict, reason


if __name__ == "__main__":
    # Self-check with a toy example matching the paper's own reported numbers,
    # to confirm the rule fires "strong" on the values it was designed around.
    # This is NOT a substitute for real results -- see the proposal's note
    # that published values are reference targets only.
    example = DecisionInputs(
        delta_da2k_pp=95.3 - 88.5,   # +6.8pp, paper Table 3
        delta_abs_rel=0.073 - 0.076,  # -0.003, paper Table 2
        delta_delta1=0.942 - 0.939,   # +0.003, paper Table 2
    )
    verdict, reason = apply_decision_rule(example)
    print(f"[Reference-values self-check, NOT a reproduced result] {verdict.value}")
    print(reason)
