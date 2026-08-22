"""
Synthetic PK/PD dataset generator for the Cpd-7 active-metabolite trap.

Ground truth (never exposed in output data/column names):
  - Parent (Cpd-7) converts to two metabolites, M1 and M2.
  - M1 is the TRUE pharmacodynamic driver (Emax model on M1 AUC), but its
    EC50 is set above the exposure a single dose reaches (only steady-state
    repeat dosing crosses it), so M1's real signal is buried in noise in the
    single-dose cohort and only shows up once dosing continues to steady
    state.
  - M2 is a DECOY: a hidden per-subject covariate Z independently (a) lowers
    M2 clearance (raising M2 AUC) in the uninduced state and (b) shifts PD
    severity directly, so in the single-dose cohort M2 exposure carries a
    modest but real spurious correlation with PD while M1 looks like noise.
    Repeated dosing triggers autoinduction of M2's clearance pathway,
    overriding the Z-linked clearance and weakening M2's edge, while M1's
    now-crossed EC50 threshold makes its true signal dominant.

This creates a hard negative: naive single-cohort correlation picks M2 (the
only one that looks like anything in that cohort); correct
generalization-aware analysis across both cohorts picks M1. Verified
empirically across 6 independent seeds (see analysis notes in task repo).
"""

import hashlib

import numpy as np
import pandas as pd
from scipy.integrate import odeint

rng = np.random.default_rng(42)


def stable_seed(*parts) -> int:
    """Deterministic replacement for hash((...)) % 2**31.

    Python's built-in hash() randomizes string hashing per-process
    (PYTHONHASHSEED) unless explicitly disabled, so seeding subject-level RNGs
    with hash() makes the "same" script produce different data on every run.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(key).hexdigest(), 16) % (2**31)

# ---- True PK parameters (population means, log-normal IIV) ----
CL_PARENT = 8.0      # L/hr
V_PARENT = 40.0       # L
KA = 1.2              # 1/hr absorption

K_FORM_M1 = 0.35      # 1/hr, parent -> M1, dose-proportional (no saturation)
CL_M1 = 6.0           # L/hr
V_M1 = 25.0           # L

K_FORM_M2_VMAX = 4.0  # mg/hr, saturable formation of M2 (Michaelis-Menten in parent conc)
K_FORM_M2_KM = 1.5    # mg/L, saturation constant
CL_M2_BASE = 5.0      # L/hr, baseline (uninduced) clearance scale
CL_M2_INDUCED = 11.0  # L/hr, population clearance after repeated-dose autoinduction (Z-independent)
V_M2 = 20.0           # L

# Hidden per-subject covariate (e.g. hepatic/renal function phenotype).
# Confound: Z lowers M2 clearance (raises M2 AUC) in the UNINDUCED state,
# and independently shifts PD severity -- but has no PK link to M1.
Z_CV = 0.40
GAMMA_Z = 14.0  # PD severity contribution scale

# PD: true mechanistic driver is M1 exposure only (Emax model), PLUS the
# independent Z-linked severity term (present in both cohorts equally).
# EC50/HILL are set so the M1 pharmacodynamic effect only engages once
# exposure crosses a threshold that a single dose does not reach but
# steady-state repeat dosing does -- a common PK/PD pattern (e.g. an effect
# that requires accumulation) and the reason M1's true signal is invisible
# in the single-dose cohort and dominant in the multi-dose cohort.
EMAX = 80.0
EC50_AUC_M1 = 300.0   # ng*hr/mL equivalent scale
HILL = 3.0
PD_NOISE_SD = 3.0

PK_NOISE_CV = 0.12  # 12% proportional assay noise


def simulate_subject(dose_mg, n_doses, tau_hr, iiv_seed, obs_end_hr=None, z_value=1.0):
    """Simulate one subject's PK for parent/M1/M2 over a dosing regimen."""
    rs = np.random.default_rng(iiv_seed)
    # log-normal inter-individual variability, ~25% CV on key params
    cl_p = CL_PARENT * np.exp(rs.normal(0, 0.25))
    v_p = V_PARENT * np.exp(rs.normal(0, 0.15))
    kfm1 = K_FORM_M1 * np.exp(rs.normal(0, 0.20))
    cl_m1 = CL_M1 * np.exp(rs.normal(0, 0.20))
    v_m1 = V_M1 * np.exp(rs.normal(0, 0.15))
    vmax_m2 = K_FORM_M2_VMAX * np.exp(rs.normal(0, 0.25))
    km_m2 = K_FORM_M2_KM * np.exp(rs.normal(0, 0.20))
    v_m2 = V_M2 * np.exp(rs.normal(0, 0.15))

    if n_doses == 1:
        # Uninduced: Z lowers clearance (raises M2 AUC) -- confound active
        cl_m2 = CL_M2_BASE * z_value * np.exp(rs.normal(0, 0.15))
    else:
        # Repeated dosing -> autoinduction overrides baseline Z-linked clearance
        cl_m2 = CL_M2_INDUCED * np.exp(rs.normal(0, 0.15))

    def odes(y, t):
        a_gut, c_p, c_m1, c_m2 = y
        # dosing handled externally via pulse resets; here just continuous ODEs
        dgut = -KA * a_gut
        absorbed = KA * a_gut / v_p
        elim_p = (cl_p / v_p) * c_p
        form_m1 = (kfm1 * c_p)          # dose-proportional formation
        form_m2 = (vmax_m2 * c_p) / (km_m2 + c_p)  # saturable formation
        dcp = absorbed - elim_p  # parent loss via elimination only (metabolite formation is a minor mass fraction)
        dcm1 = form_m1 - (cl_m1 / v_m1) * c_m1
        dcm2 = form_m2 - (cl_m2 / v_m2) * c_m2
        return [dgut, dcp, dcm1, dcm2]

    dt_fine = 0.05
    if n_doses == 1:
        # single dose: observe for obs_end_hr regardless of tau_hr
        t_end = obs_end_hr if obs_end_hr is not None else 48.0
        dose_times = [0.0]
    else:
        t_end = (n_doses - 1) * tau_hr + (obs_end_hr if obs_end_hr is not None else tau_hr)
        dose_times = [i * tau_hr for i in range(n_doses)]
    t_grid = np.arange(0, t_end + dt_fine, dt_fine)

    y = np.array([0.0, 0.0, 0.0, 0.0])
    all_t = []
    all_y = []
    seg_starts = dose_times + [t_end]

    for i in range(n_doses):
        y[0] += dose_mg  # oral dose added to gut depot
        seg_t = t_grid[(t_grid >= seg_starts[i]) & (t_grid < seg_starts[i + 1])]
        if len(seg_t) == 0:
            continue
        seg_t_rel = seg_t - seg_starts[i]
        sol = odeint(odes, y, np.concatenate([[0], seg_t_rel]))
        y = sol[-1].copy()
        all_t.append(seg_t)
        all_y.append(sol[1:])

    t_full = np.concatenate(all_t)
    y_full = np.vstack(all_y)
    c_parent = np.clip(y_full[:, 1], 0, None)
    c_m1 = np.clip(y_full[:, 2], 0, None)
    c_m2 = np.clip(y_full[:, 3], 0, None)

    return t_full, c_parent, c_m1, c_m2


def auc_trapz(t, c):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(c, t)


def build_cohort(cohort_name, dose_levels, n_subjects_per_dose, n_doses, tau_hr, subject_offset):
    records_pk = []
    records_pd = []
    records_subj = []
    subj_id = subject_offset
    sample_times_single = np.array([0.5, 1, 2, 4, 8, 12, 24, 36, 48])
    for dose in dose_levels:
        for rep in range(n_subjects_per_dose):
            subj_id += 1
            seed = stable_seed(cohort_name, dose, rep)
            z_seed = stable_seed(cohort_name, dose, rep, "z")
            z_value = float(np.exp(np.random.default_rng(z_seed).normal(0, Z_CV)))
            obs_end = 48.0 if n_doses == 1 else tau_hr
            t_full, c_p, c_m1, c_m2 = simulate_subject(
                dose, n_doses, tau_hr, seed, obs_end_hr=obs_end, z_value=z_value
            )

            # sample sparse PK timepoints within final dosing interval for realism
            t_end = (n_doses - 1) * tau_hr + obs_end
            last_interval_start = (n_doses - 1) * tau_hr
            if n_doses == 1:
                sample_t = sample_times_single[sample_times_single <= t_end]
            else:
                sample_t = last_interval_start + np.array([0.5, 1, 2, 4, 8, 12, 24])
                sample_t = sample_t[sample_t <= t_end]

            for st in sample_t:
                idx = np.argmin(np.abs(t_full - st))
                for analyte, conc in [("parent", c_p[idx]), ("M1", c_m1[idx]), ("M2", c_m2[idx])]:
                    noisy = max(0.0, conc * (1 + rng.normal(0, PK_NOISE_CV)))
                    records_pk.append({
                        "subject_id": f"S{subj_id:03d}",
                        "cohort": cohort_name,
                        "time_hr": round(float(st), 2),
                        "analyte": analyte,
                        "conc_ng_ml": round(noisy, 3),
                    })

            # exposure metrics (full profile, not just sparse samples -- "true" AUC)
            auc_p = auc_trapz(t_full, c_p)
            auc_m1 = auc_trapz(t_full, c_m1)
            auc_m2 = auc_trapz(t_full, c_m2)

            # PD driven by M1 AUC (true mechanistic term) PLUS an independent
            # Z-linked severity term present identically in both cohorts.
            pd_mechanistic = EMAX * (auc_m1 ** HILL) / (EC50_AUC_M1 ** HILL + auc_m1 ** HILL)
            pd_severity = GAMMA_Z * (-np.log(z_value))
            pd_true = pd_mechanistic + pd_severity
            pd_obs = np.clip(pd_true + rng.normal(0, PD_NOISE_SD), 0, 100)

            records_pd.append({
                "subject_id": f"S{subj_id:03d}",
                "cohort": cohort_name,
                "response": round(float(pd_obs), 2),
            })
            records_subj.append({
                "subject_id": f"S{subj_id:03d}",
                "cohort": cohort_name,
                "dose_mg": dose,
                "n_doses": n_doses,
                "tau_hr": tau_hr,
                "true_auc_parent": round(float(auc_p), 3),
                "true_auc_m1": round(float(auc_m1), 3),
                "true_auc_m2": round(float(auc_m2), 3),
                "_z_debug_only": round(z_value, 4),
            })

    return records_pk, records_pd, records_subj, subj_id


def main():
    pk_all, pd_all, subj_all = [], [], []
    next_id = 0

    # Single-dose cohort: wide dose range -> M2 (saturable) still looks dose-proportional
    # at these lower doses, so M2 exposure tracks dose (and thus PD) almost as well as M1.
    doses_single = [10, 25, 50, 100, 200]
    pk, pd_, subj, next_id = build_cohort(
        "single_dose", doses_single, n_subjects_per_dose=10, n_doses=1, tau_hr=0, subject_offset=next_id
    )
    pk_all += pk; pd_all += pd_; subj_all += subj

    # Multi-dose steady-state cohort: repeated dosing triggers autoinduction of
    # the M2 clearance pathway, overriding the baseline Z-linked clearance that
    # created the spurious M2-PD correlation in the single-dose cohort.
    doses_multi = [25, 50, 100, 200]
    pk2, pd2, subj2, next_id = build_cohort(
        "multi_dose_steady_state", doses_multi, n_subjects_per_dose=10, n_doses=7, tau_hr=24, subject_offset=next_id
    )
    pk_all += pk2; pd_all += pd2; subj_all += subj2

    df_pk = pd.DataFrame(pk_all)
    df_pd = pd.DataFrame(pd_all)
    df_subj_full = pd.DataFrame(subj_all)

    # Public bundle: strip every ground-truth column. Only design variables
    # (dose/regimen) and the subject/cohort keys are visible to the task.
    public_cols = ["subject_id", "cohort", "dose_mg", "n_doses", "tau_hr"]
    df_subj_public = df_subj_full[public_cols]

    import os

    public_dir = "public"
    private_dir = "private"
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(private_dir, exist_ok=True)

    df_pk.to_csv(os.path.join(public_dir, "pk_concentrations.csv"), index=False)
    df_pd.to_csv(os.path.join(public_dir, "pd_response.csv"), index=False)
    df_subj_public.to_csv(os.path.join(public_dir, "subjects.csv"), index=False)

    # Full table (with true_auc_* / _z_debug_only) is kept only for our own
    # authoring validation. It must never be copied into environment/data or
    # tests/data.
    df_subj_full.to_csv(os.path.join(private_dir, "subjects_ground_truth.csv"), index=False)

    return df_pk, df_pd, df_subj_public, df_subj_full


if __name__ == "__main__":
    df_pk, df_pd, df_subj_public, df_subj_full = main()
    print("Generated:")
    print(f"  public/pk_concentrations.csv: {len(df_pk)} rows")
    print(f"  public/pd_response.csv: {len(df_pd)} rows")
    print(f"  public/subjects.csv: {len(df_subj_public)} rows")
    print(f"  private/subjects_ground_truth.csv: {len(df_subj_full)} rows")
