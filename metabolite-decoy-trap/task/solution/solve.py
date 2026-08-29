"""Reference solution: exposure-response species nomination for Cpd-7.

Derives the answer from the visible PK/PD tables only. Reads DATA_DIR,
writes OUTPUT_DIR/subject_metrics.csv and OUTPUT_DIR/result.json.

Coordinate/integration conventions (must match instruction.md and the
verifier exactly):
  - pk_concentrations.csv's time_hr is absolute time since the first dose.
    last_dose_time = (n_doses - 1) * tau_hr (0 for single-dose subjects).
  - tmax_hr and the auc0_6/auc0_12 window boundaries are relative to each
    subject's most recent dose (time_hr - last_dose_time).
  - All AUCs use the trapezoidal rule exclusively. A window that starts at
    time-since-last-dose = 0 and has no observed sample exactly there
    (true for every single-dose subject, whose first sample is at 0.25h)
    is anchored with a synthetic (0, 0.0) point before integrating --
    concentration is 0 at the moment of a dose. Multi-dose subjects have a
    genuine observed 0h (trough) sample, so no synthetic point is used
    there.
  - relevant_auc: single-dose subjects = AUC[8h, 24h] post-dose (the
    sustained/lagged window); repeated-dose subjects = AUC over
    [last_dose_time, last_dose_time + tau] -- identical to auc0_12 for
    those rows (tau_hr = 12h in this dataset), by definition.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
DATA_DIR = Path(os.environ.get("DATA_DIR", str(WORKSPACE_DIR / "data")))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKSPACE_DIR / "output")))

ANALYTES = ["parent", "M1", "M2"]

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
    return pd.DataFrame(
        rows, columns=["subject_id", "analyte", "cmax", "tmax_hr", "auc0_6", "auc0_12", "trough", "relevant_auc"]
    )


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def cohort_centered_r(x: np.ndarray, y: np.ndarray, cohort: np.ndarray) -> float:
    """Correlation after centering both variables within each of the 4
    dose x regimen cohorts -- equivalent to controlling for dose, regimen,
    and their interaction (a saturated 2x2 design)."""
    x, y = np.asarray(x, dtype=float).copy(), np.asarray(y, dtype=float).copy()
    cohort = np.asarray(cohort)
    for c in np.unique(cohort):
        m = cohort == c
        x[m] = x[m] - x[m].mean()
        y[m] = y[m] - y[m].mean()
    return pearson(x, y)


def leave_one_out_min(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    vals = []
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        vals.append(pearson(x[mask], y[mask]))
    return min(vals) if vals else 0.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return pearson(x, y)


def bootstrap_ci_low(x: np.ndarray, y: np.ndarray, n_draws: int = 2000, seed: int = 42) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        idx = rng.integers(0, n, n)
        draws.append(pearson(x[idx], y[idx]))
    return float(np.percentile(draws, 2.5))


def main() -> None:
    pk = pd.read_csv(DATA_DIR / "pk_concentrations.csv")
    pd_response = pd.read_csv(DATA_DIR / "pd_response.csv")[["subject_id", "response"]]
    subjects = pd.read_csv(DATA_DIR / "subjects.csv")

    subject_metrics = build_subject_metrics(pk, subjects)

    subjects = subjects.copy()
    subjects["last_dose_time"] = np.where(
        subjects["n_doses"] > 1, (subjects["n_doses"] - 1) * subjects["tau_hr"], 0
    )
    merged = (
        subject_metrics.merge(pd_response, on="subject_id")
        .merge(subjects[["subject_id", "cohort", "dose_mg", "n_doses"]], on="subject_id")
    )
    single = merged[merged["n_doses"] == 1]
    multi = merged[merged["n_doses"] > 1]

    result: dict[str, object] = {}
    naive_r = {}
    adj_r = {}
    rep_r = {}
    lag_r = {}
    for a in ANALYTES:
        sa = single[single["analyte"] == a]
        ma = merged[merged["analyte"] == a]
        mua = multi[multi["analyte"] == a]

        result[f"{a}_cmax"] = round(float(sa["cmax"].mean()), 3)
        result[f"{a}_tmax_hr"] = round(float(sa["tmax_hr"].mean()), 3)

        naive_r[a] = pearson(sa["auc0_6"], sa["response"])
        adj_r[a] = cohort_centered_r(ma["auc0_12"], ma["response"], ma["cohort"])
        rep_r[a] = pearson(mua["relevant_auc"], mua["response"])
        lag_r[a] = pearson(ma["relevant_auc"], ma["response"])

        result[f"{a}_naive_pd_r"] = round(naive_r[a], 4)
        result[f"{a}_cohort_adjusted_pd_r"] = round(adj_r[a], 4)
        result[f"{a}_repeated_dose_pd_r"] = round(rep_r[a], 4)
        result[f"{a}_lagged_pd_r"] = round(lag_r[a], 4)
        result[f"{a}_naive_min_loo_r"] = round(leave_one_out_min(sa["auc0_6"], sa["response"]), 4)
        result[f"{a}_lagged_min_loo_r"] = round(leave_one_out_min(ma["relevant_auc"], ma["response"]), 4)
        result[f"{a}_spearman_naive_pd_r"] = round(spearman(sa["auc0_6"], sa["response"]), 4)
        result[f"{a}_bootstrap_pd_r_ci_low"] = round(bootstrap_ci_low(sa["auc0_6"].to_numpy(), sa["response"].to_numpy()), 4)

    for a in ["M1", "M2"]:
        ss_auctau = multi[multi["analyte"] == a]["relevant_auc"].mean()
        single_auc0tau = single[single["analyte"] == a]["auc0_12"].mean()
        result[f"{a}_accumulation_ratio"] = round(float(ss_auctau / single_auc0tau), 4)

    # Nomination: the species whose relationship is dose x regimen-adjusted,
    # holds under repeated dosing, and reflects the sustained/lagged
    # exposure window -- not the one with the strongest naive correlation.
    nominated = max(ANALYTES, key=lambda a: adj_r[a] + rep_r[a] + lag_r[a])
    result = {"nominated_species": nominated, **result}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subject_metrics.to_csv(OUTPUT_DIR / "subject_metrics.csv", index=False)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
