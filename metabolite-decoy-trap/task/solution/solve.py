"""Reference solution: exposure-response species nomination for Cpd-7.

Derives the answer from the visible PK/PD tables only. Reads DATA_DIR,
writes OUTPUT_DIR/result.json.
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


def compute_subject_auc(pk: pd.DataFrame) -> pd.DataFrame:
    """Trapezoidal AUC(0-tau or 0-48h) per subject per analyte from the sparse,
    noisy concentration samples visible in pk_concentrations.csv."""
    rows = []
    for (subject_id, analyte), group in pk.groupby(["subject_id", "analyte"]):
        group = group.sort_values("time_hr")
        auc = _trapz(group["conc_ng_ml"].to_numpy(), group["time_hr"].to_numpy())
        rows.append({"subject_id": subject_id, "analyte": analyte, "auc": auc})
    long_auc = pd.DataFrame(rows)
    return long_auc.pivot(index="subject_id", columns="analyte", values="auc").reset_index()


def association(auc: np.ndarray, response: np.ndarray) -> float:
    """Pearson correlation between an analyte's exposure and PD response."""
    if np.std(auc) == 0 or np.std(response) == 0:
        return 0.0
    return float(np.corrcoef(auc, response)[0, 1])


def main() -> None:
    pk = pd.read_csv(DATA_DIR / "pk_concentrations.csv")
    pd_response = pd.read_csv(DATA_DIR / "pd_response.csv")
    subjects = pd.read_csv(DATA_DIR / "subjects.csv")

    auc_wide = compute_subject_auc(pk)
    merged = (
        auc_wide.merge(pd_response[["subject_id", "response"]], on="subject_id")
        .merge(subjects[["subject_id", "n_doses"]], on="subject_id")
    )

    # Two dosing regimens are present in the data (n_doses == 1 vs > 1); treat
    # them as independent studies rather than pooling across regimens.
    single = merged[merged["n_doses"] == 1]
    multi = merged[merged["n_doses"] > 1]

    single_assoc = {a: association(single[a].to_numpy(), single["response"].to_numpy()) for a in ANALYTES}
    multi_assoc = {a: association(multi[a].to_numpy(), multi["response"].to_numpy()) for a in ANALYTES}

    # A candidate whose exposure-response relationship is an artifact of one
    # study's dosing design should not hold up once the dosing regimen
    # changes. Nominate the species whose association is strongest and most
    # consistent under the repeated-dosing, steady-state regimen, since that
    # condition gives each subject the most stable, well-characterized
    # exposure estimate.
    nominated = max(ANALYTES, key=lambda a: abs(multi_assoc[a]))

    result = {
        "nominated_species": nominated,
        "single_dose_association": round(single_assoc[nominated], 4),
        "multi_dose_association": round(multi_assoc[nominated], 4),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
