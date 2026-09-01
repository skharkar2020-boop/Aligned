"""Reference solution: hit-to-lead compound nomination for Kinase X.

Derives the answer from the visible assay/ADME tables only. Reads DATA_DIR,
writes OUTPUT_DIR/compound_metrics.csv and OUTPUT_DIR/result.json.

Conventions (must match instruction.md and the verifier exactly):
  - pIC50 = 9 - log10(IC50_nM).
  - Only `pass`-flagged replicates are averaged; `fail`-flagged replicates
    (assay QC artifacts) are excluded entirely.
  - developability_pass requires solubility_um >= 20 AND clint_ul_min_mg <=
    45 AND papp_1e6cm_s >= 8 (all three).
  - nominated_lead_id is not a disclosed formula: the reference solution
    additionally requires selectivity_index >= 0.6 among developability_pass
    compounds, then nominates the highest-lle compound in that set. Neither
    this threshold nor the gate-then-argmax procedure appears in
    instruction.md or in any result.json field -- see solution/process.md.
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

SELECTIVITY_THRESHOLD = 0.6
SOLUBILITY_MIN = 20.0
CLINT_MAX = 45.0
PAPP_MIN = 8.0

ASSAY_TO_FIELD = {"target_ic50_nm": "pic50_target", "offtarget_ic50_nm": "pic50_offtarget"}


def pic50_from_ic50(ic50_nm: pd.Series) -> pd.Series:
    return 9.0 - np.log10(ic50_nm)


def build_compound_metrics(assay: pd.DataFrame, adme: pd.DataFrame) -> pd.DataFrame:
    passed = assay[assay["qc_flag"] == "pass"].copy()
    passed["pic50"] = pic50_from_ic50(passed["ic50_nm"])
    avg = passed.groupby(["compound_id", "assay"])["pic50"].mean().unstack()
    avg = avg.rename(columns=ASSAY_TO_FIELD)[["pic50_target", "pic50_offtarget"]]
    avg = avg.reset_index()

    merged = avg.merge(adme, on="compound_id", how="inner")
    merged["selectivity_index"] = merged["pic50_target"] - merged["pic50_offtarget"]
    merged["lle"] = merged["pic50_target"] - merged["clogp"]
    merged["le"] = 1.4 * merged["pic50_target"] / merged["heavy_atom_count"]
    merged["developability_pass"] = (
        (merged["solubility_um"] >= SOLUBILITY_MIN)
        & (merged["clint_ul_min_mg"] <= CLINT_MAX)
        & (merged["papp_1e6cm_s"] >= PAPP_MIN)
    )

    out = merged[
        ["compound_id", "pic50_target", "pic50_offtarget", "selectivity_index", "lle", "le", "developability_pass"]
    ].copy()
    for col in ["pic50_target", "pic50_offtarget", "selectivity_index", "lle", "le"]:
        out[col] = out[col].round(4)
    return out.sort_values("compound_id").reset_index(drop=True)


def main() -> None:
    assay = pd.read_csv(DATA_DIR / "assay_results.csv")
    adme = pd.read_csv(DATA_DIR / "adme_properties.csv")

    compound_metrics = build_compound_metrics(assay, adme)

    naive_top_potency_id = compound_metrics.sort_values(
        ["pic50_target", "compound_id"], ascending=[False, True]
    ).iloc[0]["compound_id"]

    gate = compound_metrics[
        compound_metrics["developability_pass"] & (compound_metrics["selectivity_index"] >= SELECTIVITY_THRESHOLD)
    ]
    gate_sorted = gate.sort_values(["lle", "compound_id"], ascending=[False, True])
    nominated_lead_id = gate_sorted.iloc[0]["compound_id"]

    result = {
        "naive_top_potency_id": naive_top_potency_id,
        "n_developability_pass": int(compound_metrics["developability_pass"].sum()),
        "nominated_lead_id": nominated_lead_id,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    compound_metrics.to_csv(OUTPUT_DIR / "compound_metrics.csv", index=False)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
