"""Independent verifier for lipophilic-efficiency-trap.

Recomputes compound_metrics.csv and result.json from the verifier's own
copy of the data (tests/data/, never the agent's workspace data) and checks
the agent's outputs against that independent recomputation.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKSPACE_DIR / "output")))
TESTS_DIR = Path(os.environ.get("TESTS_DIR", str(Path(__file__).parent)))
DATA_DIR = TESTS_DIR / "data"

SELECTIVITY_THRESHOLD = 0.6
SOLUBILITY_MIN = 20.0
CLINT_MAX = 45.0
PAPP_MIN = 8.0

REL_TOL = 0.01
ABS_FLOOR = 0.02
AGG_ABS_TOL = 0.03

EXPECTED_COMPOUND_METRICS_COLUMNS = [
    "compound_id",
    "pic50_target",
    "pic50_offtarget",
    "selectivity_index",
    "lle",
    "le",
    "developability_pass",
]
EXPECTED_RESULT_KEYS = {
    "naive_top_potency_id",
    "n_developability_pass",
    "clogp_target_potency_r",
    "clogp_offtarget_potency_r",
    "nominated_lead_id",
}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _independent_recomputation() -> tuple[pd.DataFrame, dict]:
    assay = pd.read_csv(DATA_DIR / "assay_results.csv")
    adme = pd.read_csv(DATA_DIR / "adme_properties.csv")

    passed = assay[assay["qc_flag"] == "pass"].copy()
    passed["pic50"] = 9.0 - np.log10(passed["ic50_nm"])
    avg = passed.groupby(["compound_id", "assay"])["pic50"].mean().unstack()
    avg = avg.rename(columns={"target_ic50_nm": "pic50_target", "offtarget_ic50_nm": "pic50_offtarget"})
    avg = avg.reset_index()[["compound_id", "pic50_target", "pic50_offtarget"]]

    merged = avg.merge(adme, on="compound_id", how="inner")
    merged["selectivity_index"] = merged["pic50_target"] - merged["pic50_offtarget"]
    merged["lle"] = merged["pic50_target"] - merged["clogp"]
    merged["le"] = 1.4 * merged["pic50_target"] / merged["heavy_atom_count"]
    merged["developability_pass"] = (
        (merged["solubility_um"] >= SOLUBILITY_MIN)
        & (merged["clint_ul_min_mg"] <= CLINT_MAX)
        & (merged["papp_1e6cm_s"] >= PAPP_MIN)
    )

    compound_metrics = merged[EXPECTED_COMPOUND_METRICS_COLUMNS].sort_values("compound_id").reset_index(drop=True)

    naive_top_potency_id = merged.sort_values(
        ["pic50_target", "compound_id"], ascending=[False, True]
    ).iloc[0]["compound_id"]
    gate = merged[merged["developability_pass"] & (merged["selectivity_index"] >= SELECTIVITY_THRESHOLD)]
    gate_sorted = gate.sort_values(["lle", "compound_id"], ascending=[False, True])
    nominated_lead_id = gate_sorted.iloc[0]["compound_id"]

    result = {
        "naive_top_potency_id": naive_top_potency_id,
        "n_developability_pass": int(merged["developability_pass"].sum()),
        "clogp_target_potency_r": _pearson(merged["clogp"], merged["pic50_target"]),
        "clogp_offtarget_potency_r": _pearson(merged["clogp"], merged["pic50_offtarget"]),
        "nominated_lead_id": nominated_lead_id,
    }
    return compound_metrics, result


@pytest.fixture(scope="module")
def expected():
    return _independent_recomputation()


@pytest.fixture(scope="module")
def submitted_compound_metrics():
    path = OUTPUT_DIR / "compound_metrics.csv"
    assert path.exists(), f"missing {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def submitted_result():
    path = OUTPUT_DIR / "result.json"
    assert path.exists(), f"missing {path}"
    return json.loads(path.read_text())


def test_compound_metrics_schema_and_completeness(submitted_compound_metrics, expected):
    df = submitted_compound_metrics
    exp_df, _ = expected
    assert set(EXPECTED_COMPOUND_METRICS_COLUMNS).issubset(set(df.columns)), (
        f"missing columns: {set(EXPECTED_COMPOUND_METRICS_COLUMNS) - set(df.columns)}"
    )
    assert len(df) == len(exp_df) == 28, f"expected 28 rows, got {len(df)}"
    assert set(df["compound_id"]) == set(exp_df["compound_id"]), "compound_id set mismatch"
    assert df["compound_id"].is_unique, "duplicate compound_id rows"


def test_compound_metrics_values_match_recomputation(submitted_compound_metrics, expected):
    df = submitted_compound_metrics.set_index("compound_id").sort_index()
    exp_df, _ = expected
    exp = exp_df.set_index("compound_id").sort_index()

    for col in ["pic50_target", "pic50_offtarget", "selectivity_index", "lle", "le"]:
        for cid in exp.index:
            got = float(df.loc[cid, col])
            want = float(exp.loc[cid, col])
            tol = max(ABS_FLOOR, REL_TOL * abs(want))
            assert math.isclose(got, want, abs_tol=tol), (
                f"{cid}.{col}: got {got}, expected {want} (tol {tol})"
            )

    for cid in exp.index:
        got_pass = bool(df.loc[cid, "developability_pass"])
        want_pass = bool(exp.loc[cid, "developability_pass"])
        assert got_pass == want_pass, f"{cid}.developability_pass: got {got_pass}, expected {want_pass}"


def test_result_schema_and_finite_values(submitted_result):
    result = submitted_result
    assert EXPECTED_RESULT_KEYS.issubset(set(result.keys())), (
        f"missing keys: {EXPECTED_RESULT_KEYS - set(result.keys())}"
    )
    for key in ["clogp_target_potency_r", "clogp_offtarget_potency_r"]:
        val = result[key]
        assert isinstance(val, (int, float)), f"{key} must be numeric, got {type(val)}"
        assert math.isfinite(val), f"{key} is not finite: {val}"
    assert isinstance(result["n_developability_pass"], int), (
        f"n_developability_pass must be an int, got {type(result['n_developability_pass'])}"
    )
    assert 0 <= result["n_developability_pass"] <= 28, (
        f"n_developability_pass out of range: {result['n_developability_pass']}"
    )
    for key in ["naive_top_potency_id", "nominated_lead_id"]:
        assert isinstance(result[key], str) and result[key].startswith("C"), (
            f"{key} must be a compound_id string, got {result[key]!r}"
        )


def test_reported_aggregates_match_independent_recomputation(submitted_result, expected):
    _, exp = expected
    result = submitted_result

    assert result["naive_top_potency_id"] == exp["naive_top_potency_id"], (
        f"naive_top_potency_id: got {result['naive_top_potency_id']}, expected {exp['naive_top_potency_id']}"
    )
    assert result["n_developability_pass"] == exp["n_developability_pass"], (
        f"n_developability_pass: got {result['n_developability_pass']}, expected {exp['n_developability_pass']}"
    )
    for key in ["clogp_target_potency_r", "clogp_offtarget_potency_r"]:
        got, want = float(result[key]), float(exp[key])
        assert math.isclose(got, want, abs_tol=AGG_ABS_TOL), f"{key}: got {got}, expected {want}"


def test_nominated_lead_matches_independent_recomputation(submitted_result, expected):
    _, exp = expected
    assert submitted_result["nominated_lead_id"] == exp["nominated_lead_id"], (
        f"nominated_lead_id: got {submitted_result['nominated_lead_id']!r}, "
        f"expected {exp['nominated_lead_id']!r} (independently recomputed from tests/data)"
    )
