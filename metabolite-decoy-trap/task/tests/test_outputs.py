"""Independent scientific verifier for the Cpd-7 exposure-response task.

Recomputes subject_metrics.csv and result.json from the verifier's own copy
of the data (tests/data/, byte-identical to environment/data/) and checks:

  - subject_metrics.csv has exactly the 96 expected (subject_id, analyte)
    rows, each present exactly once, with every subject-level field within
    a tight numerical tolerance of an independent recomputation (the task
    requires the trapezoidal rule exclusively, so a loose tolerance would
    mask a materially different -- and therefore wrong -- integration
    method);
  - result.json has the correct schema, finite values in valid ranges, and
    every reported aggregate is consistent with an independent
    recomputation from subject_metrics-level data. Two of the required
    fields (spearman_naive_pd_r, bootstrap_pd_r_ci_low) are real,
    independently-recomputed statistics but are NOT part of the nomination
    decision below -- they exist to prevent an agent from using "which
    fields sound most statistically sophisticated" as a proxy for "which
    fields determine the answer";
  - the nominated species is the one that simultaneously dominates the
    other two candidates on all three disclosed decision criteria
    (cohort-adjusted, repeated-dose, and lagged/sustained-exposure
    association) -- not merely the one with the strongest naive
    correlation.
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
SUBJECT_METRICS_PATH = OUTPUT_DIR / "subject_metrics.csv"
DATA_DIR = TESTS_DIR / "data"

ANALYTES = ["parent", "M1", "M2"]
SUBJECT_METRICS_COLUMNS = {"subject_id", "analyte", "cmax", "tmax_hr", "auc0_6", "auc0_12", "trough", "relevant_auc"}
RESULT_KEYS = {"nominated_species"} | {
    f"{a}_{suffix}"
    for a in ANALYTES
    for suffix in (
        "cmax", "tmax_hr", "naive_pd_r", "cohort_adjusted_pd_r",
        "repeated_dose_pd_r", "lagged_pd_r", "naive_min_loo_r", "lagged_min_loo_r",
        "spearman_naive_pd_r", "bootstrap_pd_r_ci_low",
    )
} | {f"{a}_accumulation_ratio" for a in ("M1", "M2")}

# Trapezoidal-rule-only tolerance: the task requires one specific
# integration method, so a loose tolerance would mask a materially
# different (and therefore wrong) method.
SUBJECT_METRIC_REL_TOL = 0.01
SUBJECT_METRIC_ABS_FLOOR = 0.05
# Aggregate/correlation fields in result.json compound several
# subject-level tolerances; still tight, just not as tight as a single
# trapezoidal integral.
AGG_CORR_TOL = 0.02
AGG_RATIO_REL_TOL = 0.05
# bootstrap_pd_r_ci_low is a Monte Carlo estimate (2.5th percentile over
# resamples), not a deterministic quantity like the other correlation
# fields -- seed-to-seed variance at >=1000 draws was measured at
# std ~0.01-0.02 across 20 independent seeds on the shipped dataset, so
# this tolerance comfortably covers legitimate resampling noise while
# still catching a materially wrong computation.
BOOTSTRAP_TOL = 0.08
# Nominated species must beat the best of the other two candidates by at
# least this much on each of the three disclosed decision criteria,
# independently, in the verifier's own recomputation -- chosen well below
# the >=0.11 margin observed on every criterion in the shipped dataset, so
# it tolerates a reasonable alternative computation while still rejecting
# a wrong nomination (a wrong nomination fails by 0.4-0.7 here).
MIN_DOMINANCE_MARGIN = 0.05

_trapz = getattr(np, "trapezoid", None) or np.trapz


def trapz_with_origin(g: pd.DataFrame, t0: float, t1: float) -> float:
    g = g.sort_values("time_hr")
    win = g[(g["time_hr"] >= t0) & (g["time_hr"] <= t1)]
    if len(win) == 0:
        return float("nan")
    times = win["time_hr"].to_numpy()
    concs = win["conc_ng_ml"].to_numpy()
    if times[0] > t0:
        times = np.concatenate([[t0], times])
        concs = np.concatenate([[0.0], concs])
    if len(times) < 2:
        return float("nan")
    return float(_trapz(concs, times))


def build_subject_metrics(pk: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    subjects = subjects.copy()
    subjects["last_dose_time"] = np.where(
        subjects["n_doses"] > 1, (subjects["n_doses"] - 1) * subjects["tau_hr"], 0
    )
    rows = []
    for _, srow in subjects.iterrows():
        sid, n_doses, tau, ldt = srow["subject_id"], srow["n_doses"], srow["tau_hr"], srow["last_dose_time"]
        is_multi = n_doses > 1
        g_all = pk[pk["subject_id"] == sid]
        for analyte in ANALYTES:
            ga = g_all[g_all["analyte"] == analyte]
            cmax = float(ga["conc_ng_ml"].max())
            tmax_abs = float(ga.loc[ga["conc_ng_ml"].idxmax(), "time_hr"])
            row = {
                "subject_id": sid,
                "analyte": analyte,
                "cmax": round(cmax, 3),
                "tmax_hr": round(tmax_abs - ldt, 2),
                "auc0_6": round(trapz_with_origin(ga, ldt, ldt + 6), 3),
                "auc0_12": round(trapz_with_origin(ga, ldt, ldt + 12), 3),
            }
            if is_multi:
                trough_rows = ga[ga["time_hr"] == ldt]["conc_ng_ml"]
                row["trough"] = round(float(trough_rows.iloc[0]), 3) if len(trough_rows) else float("nan")
                row["relevant_auc"] = row["auc0_12"]
            else:
                row["trough"] = float("nan")
                row["relevant_auc"] = round(trapz_with_origin(ga, 8, 24), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def pearson(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def cohort_centered_r(x, y, cohort) -> float:
    x, y = np.asarray(x, dtype=float).copy(), np.asarray(y, dtype=float).copy()
    cohort = np.asarray(cohort)
    for c in np.unique(cohort):
        m = cohort == c
        x[m] = x[m] - x[m].mean()
        y[m] = y[m] - y[m].mean()
    return pearson(x, y)


def leave_one_out_min(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    vals = []
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        vals.append(pearson(x[mask], y[mask]))
    return min(vals) if vals else 0.0


def spearman(x, y) -> float:
    x, y = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return pearson(x, y)


def bootstrap_ci_low(x, y, n_draws: int = 2000, seed: int = 1234) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        idx = rng.integers(0, n, n)
        draws.append(pearson(x[idx], y[idx]))
    return float(np.percentile(draws, 2.5))


def within_tol(reported: float, recomputed: float, rel_tol: float, abs_floor: float) -> bool:
    if not (math.isfinite(reported) and math.isfinite(recomputed)):
        return False
    return abs(reported - recomputed) <= max(abs_floor, rel_tol * abs(recomputed))


@pytest.fixture(scope="module")
def raw_data():
    pk = pd.read_csv(DATA_DIR / "pk_concentrations.csv")
    pd_response = pd.read_csv(DATA_DIR / "pd_response.csv")[["subject_id", "response"]]
    subjects = pd.read_csv(DATA_DIR / "subjects.csv")
    return pk, pd_response, subjects


@pytest.fixture(scope="module")
def recomputed_subject_metrics(raw_data) -> pd.DataFrame:
    pk, _, subjects = raw_data
    return build_subject_metrics(pk, subjects)


@pytest.fixture(scope="module")
def recomputed_population(raw_data, recomputed_subject_metrics) -> dict[str, dict[str, float]]:
    pk, pd_response, subjects = raw_data
    subjects = subjects.copy()
    subjects["last_dose_time"] = np.where(
        subjects["n_doses"] > 1, (subjects["n_doses"] - 1) * subjects["tau_hr"], 0
    )
    merged = (
        recomputed_subject_metrics.merge(pd_response, on="subject_id")
        .merge(subjects[["subject_id", "cohort", "dose_mg", "n_doses"]], on="subject_id")
    )
    single = merged[merged["n_doses"] == 1]
    multi = merged[merged["n_doses"] > 1]

    out: dict[str, dict[str, float]] = {}
    for a in ANALYTES:
        sa = single[single["analyte"] == a]
        ma = merged[merged["analyte"] == a]
        mua = multi[multi["analyte"] == a]
        out[a] = {
            "cmax": float(sa["cmax"].mean()),
            "tmax_hr": float(sa["tmax_hr"].mean()),
            "naive_pd_r": pearson(sa["auc0_6"], sa["response"]),
            "cohort_adjusted_pd_r": cohort_centered_r(ma["auc0_12"], ma["response"], ma["cohort"]),
            "repeated_dose_pd_r": pearson(mua["relevant_auc"], mua["response"]),
            "lagged_pd_r": pearson(ma["relevant_auc"], ma["response"]),
            "naive_min_loo_r": leave_one_out_min(sa["auc0_6"], sa["response"]),
            "lagged_min_loo_r": leave_one_out_min(ma["relevant_auc"], ma["response"]),
            "spearman_naive_pd_r": spearman(sa["auc0_6"], sa["response"]),
            "bootstrap_pd_r_ci_low": bootstrap_ci_low(sa["auc0_6"].to_numpy(), sa["response"].to_numpy()),
        }
        if a in ("M1", "M2"):
            ss_auctau = mua["relevant_auc"].mean()
            single_auc0tau = sa["auc0_12"].mean()
            out[a]["accumulation_ratio"] = float(ss_auctau / single_auc0tau)
    return out


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    assert RESULT_PATH.exists(), f"missing output: {RESULT_PATH}"
    return json.loads(RESULT_PATH.read_text())


@pytest.fixture(scope="module")
def agent_subject_metrics() -> pd.DataFrame:
    assert SUBJECT_METRICS_PATH.exists(), f"missing output: {SUBJECT_METRICS_PATH}"
    return pd.read_csv(SUBJECT_METRICS_PATH)


def test_subject_metrics_schema_and_completeness(agent_subject_metrics, raw_data):
    _, _, subjects = raw_data
    assert SUBJECT_METRICS_COLUMNS.issubset(set(agent_subject_metrics.columns))

    expected_pairs = {(sid, a) for sid in subjects["subject_id"] for a in ANALYTES}
    actual_pairs = set(zip(agent_subject_metrics["subject_id"], agent_subject_metrics["analyte"]))
    assert actual_pairs == expected_pairs, (
        f"subject_metrics.csv must contain exactly one row per (subject_id, analyte); "
        f"missing={expected_pairs - actual_pairs}, unexpected={actual_pairs - expected_pairs}"
    )
    assert len(agent_subject_metrics) == len(expected_pairs), "duplicate (subject_id, analyte) rows found"


def test_subject_metrics_values_match_recomputation(agent_subject_metrics, recomputed_subject_metrics):
    agent = agent_subject_metrics.set_index(["subject_id", "analyte"])
    ref = recomputed_subject_metrics.set_index(["subject_id", "analyte"])

    numeric_fields = ["cmax", "auc0_6", "auc0_12", "relevant_auc"]
    for key, ref_row in ref.iterrows():
        assert key in agent.index, f"missing row {key}"
        agent_row = agent.loc[key]

        # tmax_hr must match exactly (discrete argmax over a fixed sample grid).
        assert agent_row["tmax_hr"] == pytest.approx(ref_row["tmax_hr"], abs=1e-6), (
            f"{key}: tmax_hr must exactly match the observed peak sample's time-since-last-dose"
        )

        for field in numeric_fields:
            assert within_tol(float(agent_row[field]), float(ref_row[field]), SUBJECT_METRIC_REL_TOL, SUBJECT_METRIC_ABS_FLOOR), (
                f"{key}: {field}={agent_row[field]} inconsistent with independent recomputation "
                f"{ref_row[field]} (tolerance: {SUBJECT_METRIC_REL_TOL:.0%} relative or {SUBJECT_METRIC_ABS_FLOOR} absolute)"
            )

        if pd.notna(ref_row["trough"]):
            assert within_tol(float(agent_row["trough"]), float(ref_row["trough"]), SUBJECT_METRIC_REL_TOL, SUBJECT_METRIC_ABS_FLOOR), (
                f"{key}: trough inconsistent with independent recomputation"
            )
        else:
            assert pd.isna(agent_row["trough"]) or agent_row["trough"] == "", (
                f"{key}: single-dose subjects have no trough sample; trough must be blank"
            )


def test_result_schema_and_finite_values(result):
    assert set(result) == RESULT_KEYS, f"missing={RESULT_KEYS - set(result)}, unexpected={set(result) - RESULT_KEYS}"
    assert result["nominated_species"] in ANALYTES

    for a in ANALYTES:
        for suffix in ("naive_pd_r", "cohort_adjusted_pd_r", "repeated_dose_pd_r", "lagged_pd_r", "naive_min_loo_r", "lagged_min_loo_r", "spearman_naive_pd_r", "bootstrap_pd_r_ci_low"):
            value = result[f"{a}_{suffix}"]
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert math.isfinite(float(value))
            assert -1.0 - 1e-6 <= float(value) <= 1.0 + 1e-6
        for suffix in ("cmax", "tmax_hr"):
            value = result[f"{a}_{suffix}"]
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert math.isfinite(float(value)) and float(value) >= 0

    for a in ("M1", "M2"):
        value = result[f"{a}_accumulation_ratio"]
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(float(value)) and float(value) > 0


def test_reported_aggregates_match_independent_recomputation(result, recomputed_population):
    for a in ANALYTES:
        ref = recomputed_population[a]
        for suffix in ("naive_pd_r", "cohort_adjusted_pd_r", "repeated_dose_pd_r", "lagged_pd_r", "naive_min_loo_r", "lagged_min_loo_r", "spearman_naive_pd_r"):
            reported = float(result[f"{a}_{suffix}"])
            assert abs(reported - ref[suffix]) <= AGG_CORR_TOL, (
                f"{a}_{suffix}={reported} inconsistent with independent recomputation {ref[suffix]:.3f}"
            )
        reported_boot = float(result[f"{a}_bootstrap_pd_r_ci_low"])
        assert abs(reported_boot - ref["bootstrap_pd_r_ci_low"]) <= BOOTSTRAP_TOL, (
            f"{a}_bootstrap_pd_r_ci_low={reported_boot} inconsistent with independent recomputation "
            f"{ref['bootstrap_pd_r_ci_low']:.3f} (tolerance {BOOTSTRAP_TOL} accounts for Monte Carlo resampling noise)"
        )
        for suffix in ("cmax", "tmax_hr"):
            reported = float(result[f"{a}_{suffix}"])
            assert within_tol(reported, ref[suffix], SUBJECT_METRIC_REL_TOL, SUBJECT_METRIC_ABS_FLOOR), (
                f"{a}_{suffix}={reported} inconsistent with independent recomputation {ref[suffix]:.3f}"
            )
    for a in ("M1", "M2"):
        reported = float(result[f"{a}_accumulation_ratio"])
        ref = recomputed_population[a]["accumulation_ratio"]
        assert within_tol(reported, ref, AGG_RATIO_REL_TOL, 0.1), (
            f"{a}_accumulation_ratio={reported} inconsistent with independent recomputation {ref:.3f}"
        )


def test_nominated_species_dominates_all_three_criteria(result, recomputed_population):
    nominated = result["nominated_species"]
    criteria = ("cohort_adjusted_pd_r", "repeated_dose_pd_r", "lagged_pd_r")
    for criterion in criteria:
        nominated_value = recomputed_population[nominated][criterion]
        runner_up = max(recomputed_population[a][criterion] for a in ANALYTES if a != nominated)
        assert nominated_value - runner_up >= MIN_DOMINANCE_MARGIN, (
            f"nominated species {nominated!r} does not dominate on {criterion} "
            f"(recomputed: {[(a, round(recomputed_population[a][criterion], 3)) for a in ANALYTES]}); "
            f"a correct nomination must simultaneously lead on all three disclosed decision criteria "
            f"(dose x regimen-adjusted, repeated-dose, and lagged/sustained-exposure association), "
            f"not merely have the strongest naive correlation"
        )
