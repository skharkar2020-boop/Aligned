"""Independent scientific verifier for the Cpd-7 exposure-response task.

Recomputes subject-level AUC and per-study exposure-response correlations
from the verifier's own copy of the data (tests/data/, identical to
environment/data/) and checks that the agent's nominated species is the one
whose association genuinely survives across both dosing regimens, not one
that only looked good in a single-cohort view.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))
TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
RESULT_PATH = OUTPUT_DIR / "result.json"
DATA_DIR = TESTS_DIR / "data"

ANALYTES = {"parent", "M1", "M2"}
EXPECTED_KEYS = {"nominated_species", "single_dose_association", "multi_dose_association"}

_trapz = getattr(np, "trapezoid", None) or np.trapz

# A candidate's steady-state association must exceed every other candidate's
# by at least this much (in the verifier's own recomputation) to count as
# genuinely dominant, not a marginal artifact of one AUC-estimation method.
# Derived empirically, not guessed: across a 6-seed sweep at the shipped
# sample size (n=10/dose), the true driver's margin over the runner-up
# ranged 0.054-0.087 (floor 0.054, mean 0.075). This threshold sits clearly
# below that observed floor (~45% headroom) so a reasonable alternative AUC
# or correlation methodology still passes, while a wrong nomination (whose
# recomputed margin is negative in every tested seed) still fails. See
# task/README.md for the full sweep record.
MIN_DOMINANCE_MARGIN = 0.03
# How far the agent's reported association may drift from the verifier's
# independent recomputation for the same species before it looks fabricated
# or derived from a materially different (and therefore unverifiable) method.
MAX_REPORT_DEVIATION = 0.20


def _subject_auc(pk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject_id, analyte), group in pk.groupby(["subject_id", "analyte"]):
        group = group.sort_values("time_hr")
        auc = _trapz(group["conc_ng_ml"].to_numpy(), group["time_hr"].to_numpy())
        rows.append({"subject_id": subject_id, "analyte": analyte, "auc": auc})
    return pd.DataFrame(rows).pivot(index="subject_id", columns="analyte", values="auc").reset_index()


def _association(auc: np.ndarray, response: np.ndarray) -> float:
    """Rank correlation (Spearman), matching solution/solve.py -- see that
    file for why rank correlation rather than raw Pearson."""
    if np.std(auc) == 0 or np.std(response) == 0:
        return 0.0
    auc_rank = pd.Series(auc).rank().to_numpy()
    response_rank = pd.Series(response).rank().to_numpy()
    return float(np.corrcoef(auc_rank, response_rank)[0, 1])


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    assert RESULT_PATH.exists(), f"missing output: {RESULT_PATH}"
    return json.loads(RESULT_PATH.read_text())


@pytest.fixture(scope="module")
def recomputed() -> dict[str, dict[str, float]]:
    pk = pd.read_csv(DATA_DIR / "pk_concentrations.csv")
    pd_response = pd.read_csv(DATA_DIR / "pd_response.csv")
    subjects = pd.read_csv(DATA_DIR / "subjects.csv")

    auc_wide = _subject_auc(pk)
    merged = auc_wide.merge(pd_response[["subject_id", "response"]], on="subject_id").merge(
        subjects[["subject_id", "n_doses"]], on="subject_id"
    )
    single = merged[merged["n_doses"] == 1]
    multi = merged[merged["n_doses"] > 1]

    return {
        "single": {a: _association(single[a].to_numpy(), single["response"].to_numpy()) for a in ANALYTES},
        "multi": {a: _association(multi[a].to_numpy(), multi["response"].to_numpy()) for a in ANALYTES},
    }


def test_result_schema_and_finite_values(result: dict[str, object]) -> None:
    assert set(result) == EXPECTED_KEYS
    assert result["nominated_species"] in ANALYTES
    for key in ("single_dose_association", "multi_dose_association"):
        value = result[key]
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(float(value))
        assert -1.0 - 1e-6 <= float(value) <= 1.0 + 1e-6


def test_nominated_species_dominates_the_steady_state_regimen(
    result: dict[str, object], recomputed: dict[str, dict[str, float]]
) -> None:
    nominated = result["nominated_species"]
    multi = recomputed["multi"]
    nominated_strength = abs(multi[nominated])
    runner_up_strength = max(abs(multi[a]) for a in ANALYTES if a != nominated)
    assert nominated_strength - runner_up_strength >= MIN_DOMINANCE_MARGIN, (
        f"nominated species {nominated!r} does not clearly dominate the multi-dose "
        f"steady-state association (recomputed: {multi}); a correct nomination must "
        f"be the one whose exposure-response relationship survives repeated dosing, "
        f"not one that merely ties with an alternative"
    )


def test_reported_associations_match_independent_recomputation(
    result: dict[str, object], recomputed: dict[str, dict[str, float]]
) -> None:
    nominated = result["nominated_species"]
    for key, cohort in (("single_dose_association", "single"), ("multi_dose_association", "multi")):
        reported = float(result[key])
        independent = recomputed[cohort][nominated]
        assert abs(reported - independent) <= MAX_REPORT_DEVIATION, (
            f"reported {key}={reported} for {nominated!r} is inconsistent with the "
            f"independently recomputed value {independent:.3f}; evidence must be "
            f"derivable from the supplied data, not asserted"
        )
