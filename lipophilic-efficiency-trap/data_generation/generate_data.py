"""Synthetic data generator for the lipophilic-efficiency-trap task.

Generates a Hit-to-Lead SAR dataset for a fictional kinase target ("Kinase X")
and a single chemical series of 30 analogs. Produces two public tables
(assay_results.csv, adme_properties.csv) and one private ground-truth table
(compounds_ground_truth.csv, not shipped to the agent).

Deterministic seeding: every per-compound and per-replicate draw is seeded
from a SHA-256 hash of a stable string key, never Python's randomized
built-in hash(), so output is reproducible across interpreter runs.

Generative model (the "trap"):
  - clogp (lipophilicity) is drawn per compound -- the confound.
  - intrinsic_fit is an independent per-compound latent variable representing
    genuine, logP-independent structure-based potency (pharmacophore fit).
  - True on-target potency depends on BOTH intrinsic_fit and clogp (filling a
    hydrophobic pocket is a real, if partial, driver of potency).
  - True off-target (anti-target paralog) potency depends on clogp PLUS an
    independent "off-target idiosyncrasy" latent (real paralog-binding
    events that are not pure bulk-lipophilicity effects), which keeps the
    clogp/off-target relationship real but far from deterministic (r ~ 0.7,
    not ~1.0 -- a scatter, not a single sortable column).
  - Metabolic clearance depends on clogp PLUS an independent "metabolic
    idiosyncrasy" latent (a labile-group liability unrelated to bulk
    lipophilicity), so a low-clogp compound is not automatically metabolism-
    clean and a moderate-clogp compound is not automatically disqualified.
  - 24 background compounds are drawn from this generative process to
    provide a realistic, noisy population. 6 compounds (C101-C106) have
    their latent variables pinned to realize six specific, competing
    medicinal-chemistry archetypes documented below -- this is what makes
    the dataset a genuine multi-criterion judgment problem rather than a
    single-column giveaway.

Archetypes (see solution/process.md for the full numeric validation):
  - C101 (A): highest raw on-target potency; fails developability (clint)
    because that potency is bought almost entirely with high clogp.
  - C102 (B): lowest clogp, cleanest ADME; passes every check but is
    dominated on potency, LLE, and selectivity by the intended lead --
    a defensible but suboptimal "play it safe" choice.
  - C103 (C): the single highest selectivity_index in the dataset (a lucky
    off-target-idiosyncrasy draw); mediocre LLE, so it wins one axis and
    loses the integrated comparison.
  - C104 (D): the single highest LLE in the ENTIRE population (ungated);
    fails developability via an independent metabolic liability unrelated
    to its (low-moderate) clogp -- attractive until you check ADME.
  - C105 (E, the intended lead): passes developability and has clearly
    positive selectivity, and among compounds satisfying both has the
    highest LLE. Does not dominate every competitor (A beats it on raw
    potency, C beats it on selectivity, D and F beat it on raw LLE) but is
    the only compound that is simultaneously developable, genuinely
    selective, and efficient.
  - C106 (F): passes every disclosed developability threshold and has the
    highest LLE among developability-passing compounds -- so a mechanical
    "apply the disclosed gate, then sort by LLE" workflow selects F, not
    the intended lead. F's selectivity_index is negative (more potent
    against the off-target paralog than the primary target): plainly,
    inspectably non-selective, not a borderline call against some
    undisclosed numeric cutoff.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PUBLIC_DIR = HERE / "public"
PRIVATE_DIR = HERE / "private"

N_BACKGROUND = 24
N_REPLICATES = 3

CLOGP_LOW, CLOGP_HIGH = 2.4, 5.5
TPSA_LOW, TPSA_HIGH = 40.0, 110.0
BACKGROUND_FIT_SD = 0.85

TARGET_BASE = 6.5
TARGET_INTRINSIC_WEIGHT = 0.6
TARGET_LOGP_WEIGHT = 0.25

OFFTARGET_BASE = 5.3
OFFTARGET_LOGP_WEIGHT = 0.55
OFFTARGET_IDIO_SD = 0.55  # background-only independent off-target-affinity noise

ASSAY_REPLICATE_NOISE_SD = 0.06  # log units, per-replicate pIC50 noise
QC_FAIL_RATE = 0.15  # fraction of (compound, assay) pairs with one corrupted replicate

MW_BASE, MW_LOGP_WEIGHT, MW_NOISE_SD = 380.0, 28.0, 12.0
CLINT_BASE, CLINT_LOGP_WEIGHT, CLINT_NOISE_SD, CLINT_FLOOR = 8.0, 9.5, 4.0, 2.0
CLINT_IDIO_SD = 7.0  # background-only independent metabolic-liability noise
SOL_BASE, SOL_LOGP_WEIGHT, SOL_NOISE_SD, SOL_FLOOR = 260.0, 42.0, 15.0, 3.0
PAPP_BASE, PAPP_TPSA_WEIGHT, PAPP_NOISE_SD, PAPP_FLOOR = 32.0, 0.28, 3.0, 1.0

# Archetype compound_ids and their pinned latent variables:
# (clogp, intrinsic_fit, offtarget_idio, metab_idio, tpsa)
ARCHETYPES: dict[str, tuple[float, float, float, float, float]] = {
    "C101": (5.00, 1.55, 0.10, -1.0, 60.0),   # A: naive potency winner -> fails developability (clint)
    "C102": (2.00, 0.70, 0.05, 0.0, 55.0),    # B: cleanest properties, dominated on potency/LLE/selectivity
    "C103": (3.30, 0.35, -1.55, -0.5, 41.0),  # C: best selectivity, mediocre LLE
    "C104": (2.55, 2.10, 0.35, 20.0, 55.0),   # D: best raw LLE in the population, fails developability (clint)
    "C105": (2.75, 2.05, -0.20, 0.0, 65.0),   # E: intended lead -- integrated balance, no single dominant axis
    "C106": (2.20, 1.70, 2.00, 0.0, 55.0),    # F: passes developability, best LLE among passers, non-selective
}


def stable_seed(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def rng_for(key: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed(key))


def pic50_to_ic50_nm(pic50: np.ndarray) -> np.ndarray:
    return np.power(10.0, 9.0 - pic50)


def build_compounds() -> pd.DataFrame:
    rows = []
    ids = [f"C{i:03d}" for i in range(1, N_BACKGROUND + 1)] + list(ARCHETYPES.keys())
    for cid in ids:
        r = rng_for(f"lipophilic-efficiency-trap::compound::{cid}")
        if cid in ARCHETYPES:
            clogp, intrinsic_fit, offtarget_idio, metab_idio, tpsa = ARCHETYPES[cid]
            # Burn the same draws a background compound would use, so the archetype
            # override is a pure substitution and the rest of the stream is unaffected.
            _ = r.uniform(CLOGP_LOW, CLOGP_HIGH)
            _ = r.normal(0.0, BACKGROUND_FIT_SD)
            _ = r.uniform(TPSA_LOW, TPSA_HIGH)
        else:
            clogp = float(r.uniform(CLOGP_LOW, CLOGP_HIGH))
            intrinsic_fit = float(r.normal(0.0, BACKGROUND_FIT_SD))
            tpsa = float(r.uniform(TPSA_LOW, TPSA_HIGH))
            offtarget_idio = float(r.normal(0.0, OFFTARGET_IDIO_SD))
            metab_idio = float(r.normal(0.0, CLINT_IDIO_SD))

        true_pic50_target = TARGET_BASE + TARGET_INTRINSIC_WEIGHT * intrinsic_fit + TARGET_LOGP_WEIGHT * clogp
        true_pic50_offtarget = OFFTARGET_BASE + OFFTARGET_LOGP_WEIGHT * clogp + offtarget_idio

        mw = MW_BASE + MW_LOGP_WEIGHT * clogp + float(r.normal(0.0, MW_NOISE_SD))
        clint = max(
            CLINT_FLOOR,
            CLINT_BASE + CLINT_LOGP_WEIGHT * clogp + metab_idio + float(r.normal(0.0, CLINT_NOISE_SD)),
        )
        solubility = max(SOL_FLOOR, SOL_BASE - SOL_LOGP_WEIGHT * clogp + float(r.normal(0.0, SOL_NOISE_SD)))
        papp = max(PAPP_FLOOR, PAPP_BASE - PAPP_TPSA_WEIGHT * tpsa + float(r.normal(0.0, PAPP_NOISE_SD)))
        heavy_atom_count = int(round(mw / 13.6))

        rows.append(
            {
                "compound_id": cid,
                "clogp": round(clogp, 3),
                "intrinsic_fit": round(intrinsic_fit, 4),
                "tpsa": round(tpsa, 2),
                "offtarget_idio": round(offtarget_idio, 4),
                "metab_idio": round(metab_idio, 4),
                "true_pic50_target": round(true_pic50_target, 4),
                "true_pic50_offtarget": round(true_pic50_offtarget, 4),
                "mw": round(mw, 2),
                "heavy_atom_count": heavy_atom_count,
                "clint_ul_min_mg": round(clint, 2),
                "solubility_um": round(solubility, 2),
                "papp_1e6cm_s": round(papp, 2),
            }
        )
    return pd.DataFrame(rows)


def build_assay_results(compounds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, c in compounds.iterrows():
        cid = c["compound_id"]
        for assay, true_pic50 in [
            ("target_ic50_nm", c["true_pic50_target"]),
            ("offtarget_ic50_nm", c["true_pic50_offtarget"]),
        ]:
            r = rng_for(f"lipophilic-efficiency-trap::assay::{cid}::{assay}")
            qc_fail_roll = r.uniform(0.0, 1.0)
            fail_replicate = int(r.integers(0, N_REPLICATES)) if qc_fail_roll < QC_FAIL_RATE else -1
            for rep in range(N_REPLICATES):
                noisy_pic50 = true_pic50 + float(r.normal(0.0, ASSAY_REPLICATE_NOISE_SD))
                if rep == fail_replicate:
                    corrupt_factor = float(r.choice([0.1, 10.0]))
                    ic50_nm = float(pic50_to_ic50_nm(np.array([noisy_pic50]))[0]) * corrupt_factor
                    qc_flag = "fail"
                else:
                    ic50_nm = float(pic50_to_ic50_nm(np.array([noisy_pic50]))[0])
                    qc_flag = "pass"
                rows.append(
                    {
                        "compound_id": cid,
                        "assay": assay,
                        "replicate": rep + 1,
                        "ic50_nm": round(ic50_nm, 3),
                        "qc_flag": qc_flag,
                    }
                )
    return pd.DataFrame(rows)


def build_adme_properties(compounds: pd.DataFrame) -> pd.DataFrame:
    return compounds[
        ["compound_id", "clogp", "mw", "heavy_atom_count", "tpsa", "papp_1e6cm_s", "clint_ul_min_mg", "solubility_um"]
    ].copy()


def main() -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

    compounds = build_compounds()
    assay_results = build_assay_results(compounds)
    adme_properties = build_adme_properties(compounds)

    assay_results.to_csv(PUBLIC_DIR / "assay_results.csv", index=False)
    adme_properties.to_csv(PUBLIC_DIR / "adme_properties.csv", index=False)
    compounds.to_csv(PRIVATE_DIR / "compounds_ground_truth.csv", index=False)

    print(f"Wrote {len(assay_results)} assay rows and {len(adme_properties)} ADME rows for {len(compounds)} compounds.")


if __name__ == "__main__":
    main()
