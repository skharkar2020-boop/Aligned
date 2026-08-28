"""
Synthetic PK/PD dataset generator v2 for the metabolite-decoy-trap task.

Ground truth (never exposed in output column names):
  - M2 is the TRUE pharmacodynamic driver. PD is generated as a function of
    M2's *sustained/lagged* exposure only (AUC[8,24]h for single-dose
    subjects, steady-state AUCtau for multi-dose subjects) plus noise. M1 has
    no causal role in PD at all.
  - M1 is a highly plausible DECOY: fast formation/elimination gives it the
    highest Cmax and earliest Tmax, and a strong *pooled, unstratified*
    correlation with PD -- but that correlation is a Simpson's-paradox
    artifact. M1's exposure scales with administered dose (deterministic
    PK), and PD (via M2, which also scales with dose) rises with dose too,
    so M1 and PD move together *between* dose cohorts even though M1 has no
    real relationship with PD *within* a dose cohort (M1's within-subject
    variability is independent noise, uncorrelated with the independent
    noise on M2 that actually drives PD).
  - Two salted "influential" subjects get inflated M1 exposure *and* an
    independent PD bump, further inflating M1's naive pooled correlation.
    Removing either subject (leave-one-out) should collapse M1's apparent
    correlation while M2's (a real, distributed signal) stays stable.
  - M1 clears essentially fully within one dosing interval (negligible
    accumulation); M2 persists across the interval and genuinely
    accumulates at steady state.

None of this mechanism is stated in instruction.md. The agent must discover
it by computing dose-adjusted, regimen-specific, lagged, and leave-one-out
robustness metrics and comparing candidates across all of them, not by
trusting a single naive pooled correlation.
"""

import hashlib
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Deterministic seeding (avoid Python's per-process-randomized hash()).
# ---------------------------------------------------------------------------


def stable_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(key).hexdigest(), 16) % (2**31)


def rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))


# ---------------------------------------------------------------------------
# PK model: closed-form Bateman function per analyte, dose-proportional
# amplitude, linear superposition across repeated doses.
#   C(t) = scale * kf/(kf - ke) * (exp(-ke*t) - exp(-kf*t))   for t > 0
# ---------------------------------------------------------------------------

M1_KF, M1_KE = 1.2, 0.40          # fast formation, fast elimination
M2_KF, M2_KE = 0.25, 0.06         # slow formation, slow elimination
PARENT_KF, PARENT_KE = 2.5, 0.60  # fast absorption-like rise, fast decline

M1_SCALE_PER_DOSE = 0.85          # (ng/mL) per mg dose, population value
M2_SCALE_PER_DOSE = 0.32
PARENT_SCALE_PER_DOSE = 1.10

# M1 is kept low-noise / tightly dose-predictable on purpose: a clean,
# highly plausible decoy signal (Cmax/AUC both track dose cleanly, which is
# exactly what makes its naive pooled correlation with PD -- via the shared
# dose confound -- look convincing). M2's *shape* parameters (kf/ke, which
# drive Cmax and Tmax) are deliberately noisier than its *scale* parameter,
# so a subject's M2 Cmax does not closely predict that subject's M2
# sustained/AUC exposure -- the intended Cmax-vs-AUC decoupling trap.
M1_IIV_KF, M1_IIV_KE, M1_IIV_SCALE = 0.06, 0.06, 0.07
M2_IIV_KF, M2_IIV_KE, M2_IIV_SCALE = 1.40, 0.20, 0.12
PARENT_IIV_KF, PARENT_IIV_KE, PARENT_IIV_SCALE = 0.10, 0.08, 0.12
PK_ASSAY_CV = 0.08      # proportional assay noise on sampled concentrations

TAU = 12.0
N_DOSES_MULTI = 7
DOSE_LOW = 50.0
DOSE_HIGH = 250.0
N_SUBJECTS_PER_CELL = 8  # 4 cells (dose x regimen) x 8 = 32 subjects

TIME_OFFSETS_SINGLE = [0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48]
# Multi-dose offsets are relative to the LAST dose; 0.0 is a genuine
# observed pre-dose trough sample (not an assumption). 0.25-12h cover the
# steady-state dosing interval (tau=12h); 16/24/36/48h are a terminal tail
# extending past the interval.
TIME_OFFSETS_MULTI = [0.0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48]

# PD: pure function of M2's sustained/lagged exposure. sqrt-compressed so
# the single-dose (16h window) and steady-state (12h AUCtau, accumulated)
# scales don't blow out the noise-to-signal ratio; tuned by iterating
# data_generation/generate_data.py against validate_v2.py (see README).
PD_BASELINE = 15.0
PD_SLOPE = 21.0
PD_NOISE_SD = 38.0

# Salted influential subjects: 3 subjects (within the single-dose cohort)
# get a moderate M1 amplitude bump plus an independent, noisy PD bump, M2
# left untouched. Deliberately subtle (not a cartoonish single extreme
# outlier): a real subject with mildly elevated M1 clearance/formation
# variability, coincidentally also on the higher end of PD for unrelated
# reasons, of exactly the kind that can inflate a naive correlation without
# being an obvious inspection-visible outlier.
INFLUENTIAL_M1_MULT = 3.0
INFLUENTIAL_PD_BUMP_MEAN = 45.0
INFLUENTIAL_PD_BUMP_SD = 8.0


def bateman(t, kf, ke, scale):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    mask = t > 0
    out[mask] = scale * (kf / (kf - ke)) * (np.exp(-ke * t[mask]) - np.exp(-kf * t[mask]))
    return np.clip(out, 0.0, None)


def superpose(t_eval, dose_times, kf, ke, scale):
    t_eval = np.asarray(t_eval, dtype=float)
    c = np.zeros_like(t_eval)
    for dt in dose_times:
        c = c + bateman(t_eval - dt, kf, ke, scale)
    return c


def fine_auc(fn, t0, t1, n=4000):
    """High-resolution trapezoidal AUC of a continuous curve fn over [t0, t1]."""
    if t1 <= t0:
        return 0.0
    tt = np.linspace(t0, t1, n)
    cc = fn(tt)
    return float(np.trapezoid(cc, tt))


class SubjectPK:
    """Subject-specific closed-form PK curves for parent/M1/M2."""

    def __init__(self, subject_id, dose, n_doses, tau):
        self.dose = dose
        self.n_doses = n_doses
        self.tau = tau
        self.dose_times = [0.0] if n_doses == 1 else [i * tau for i in range(n_doses)]
        self.last_dose_time = self.dose_times[-1]

        rs_m1 = rng_for(subject_id, "m1")
        rs_m2 = rng_for(subject_id, "m2")
        rs_p = rng_for(subject_id, "parent")

        self.kf1 = M1_KF * np.exp(rs_m1.normal(0, M1_IIV_KF))
        self.ke1 = M1_KE * np.exp(rs_m1.normal(0, M1_IIV_KE))
        self.scale1 = M1_SCALE_PER_DOSE * dose * np.exp(rs_m1.normal(0, M1_IIV_SCALE))

        self.kf2 = M2_KF * np.exp(rs_m2.normal(0, M2_IIV_KF))
        self.ke2 = M2_KE * np.exp(rs_m2.normal(0, M2_IIV_KE))
        self.scale2 = M2_SCALE_PER_DOSE * dose * np.exp(rs_m2.normal(0, M2_IIV_SCALE))

        self.kfp = PARENT_KF * np.exp(rs_p.normal(0, PARENT_IIV_KF))
        self.kep = PARENT_KE * np.exp(rs_p.normal(0, PARENT_IIV_KE))
        self.scalep = PARENT_SCALE_PER_DOSE * dose * np.exp(rs_p.normal(0, PARENT_IIV_SCALE))

    def c_m1(self, t):
        return superpose(t, self.dose_times, self.kf1, self.ke1, self.scale1)

    def c_m2(self, t):
        return superpose(t, self.dose_times, self.kf2, self.ke2, self.scale2)

    def c_parent(self, t):
        return superpose(t, self.dose_times, self.kfp, self.kep, self.scalep)

    def apply_influential_boost(self, mult):
        self.scale1 *= mult


def sample_points(pk: SubjectPK):
    if pk.n_doses == 1:
        return list(TIME_OFFSETS_SINGLE)
    return [pk.last_dose_time + off for off in TIME_OFFSETS_MULTI]


def build_subject(subj_id, cohort, dose, n_doses, tau, influential=False):
    pk = SubjectPK(subj_id, dose, n_doses, tau)
    if influential:
        pk.apply_influential_boost(INFLUENTIAL_M1_MULT)

    times = sample_points(pk)
    assay_rng = rng_for(subj_id, "assay")

    pk_rows = []
    for analyte, fn in (("parent", pk.c_parent), ("M1", pk.c_m1), ("M2", pk.c_m2)):
        true_c = fn(times)
        noisy = np.clip(true_c * (1 + assay_rng.normal(0, PK_ASSAY_CV, size=len(times))), 0.0, None)
        for t, c in zip(times, noisy):
            pk_rows.append(
                {
                    "subject_id": subj_id,
                    "cohort": cohort,
                    "time_hr": round(float(t), 2),
                    "analyte": analyte,
                    "conc_ng_ml": round(float(c), 4),
                }
            )

    # "True" (fine-grid, noise-free) relevant-exposure metric that drives PD.
    if n_doses == 1:
        relevant_m2_auc = fine_auc(pk.c_m2, 8.0, 24.0)
    else:
        relevant_m2_auc = fine_auc(pk.c_m2, pk.last_dose_time, pk.last_dose_time + tau)

    pd_rng = rng_for(subj_id, "pd")
    pd_true = PD_BASELINE + PD_SLOPE * np.sqrt(relevant_m2_auc)
    if influential:
        pd_bump = max(0.0, pd_rng.normal(INFLUENTIAL_PD_BUMP_MEAN, INFLUENTIAL_PD_BUMP_SD))
    else:
        pd_bump = 0.0
    pd_obs = max(0.0, pd_true + pd_bump + pd_rng.normal(0, PD_NOISE_SD))

    pd_row = {"subject_id": subj_id, "cohort": cohort, "response": round(float(pd_obs), 3)}
    subj_row = {
        "subject_id": subj_id,
        "cohort": cohort,
        "dose_mg": dose,
        "n_doses": n_doses,
        "tau_hr": tau if n_doses > 1 else 0,
    }
    ground_truth_row = {
        "subject_id": subj_id,
        "cohort": cohort,
        "influential": influential,
        "true_relevant_m2_auc": round(relevant_m2_auc, 4),
        "true_m1_cmax": round(float(np.max(pk.c_m1(np.linspace(0.01, pk.last_dose_time + 48, 4000)))), 4),
        "true_m2_cmax": round(float(np.max(pk.c_m2(np.linspace(0.01, pk.last_dose_time + 48, 4000)))), 4),
    }
    return pk_rows, pd_row, subj_row, ground_truth_row


def main():
    cells = [
        ("single_low", DOSE_LOW, 1, 0),
        ("single_high", DOSE_HIGH, 1, 0),
        ("multi_low", DOSE_LOW, N_DOSES_MULTI, TAU),
        ("multi_high", DOSE_HIGH, N_DOSES_MULTI, TAU),
    ]

    # 2 moderately-influential subjects (one per dose level), enough
    # individual leverage to show up in single-point leave-one-out, but a
    # 2.3x M1 bump (not a cartoonish 4x) plus a noisy (not fixed) PD bump --
    # subtle enough not to be obvious by inspection.
    influential_slots = {("single_low", 2), ("single_high", 5)}

    pk_all, pd_all, subj_all, gt_all = [], [], [], []
    subj_counter = 0
    for cohort, dose, n_doses, tau in cells:
        for rep in range(N_SUBJECTS_PER_CELL):
            subj_counter += 1
            subj_id = f"S{subj_counter:03d}"
            influential = (cohort, rep) in influential_slots
            pk_rows, pd_row, subj_row, gt_row = build_subject(
                subj_id, cohort, dose, n_doses, tau, influential=influential
            )
            pk_all += pk_rows
            pd_all.append(pd_row)
            subj_all.append(subj_row)
            gt_all.append(gt_row)

    df_pk = pd.DataFrame(pk_all)
    df_pd = pd.DataFrame(pd_all)
    df_subj = pd.DataFrame(subj_all)
    df_gt = pd.DataFrame(gt_all)

    public_dir = os.path.join(os.path.dirname(__file__), "public")
    private_dir = os.path.join(os.path.dirname(__file__), "private")
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(private_dir, exist_ok=True)

    df_pk.to_csv(os.path.join(public_dir, "pk_concentrations.csv"), index=False)
    df_pd.to_csv(os.path.join(public_dir, "pd_response.csv"), index=False)
    df_subj.to_csv(os.path.join(public_dir, "subjects.csv"), index=False)
    df_gt.to_csv(os.path.join(private_dir, "subjects_ground_truth.csv"), index=False)

    return df_pk, df_pd, df_subj, df_gt


if __name__ == "__main__":
    df_pk, df_pd, df_subj, df_gt = main()
    print("Generated:")
    print(f"  public/pk_concentrations.csv: {len(df_pk)} rows")
    print(f"  public/pd_response.csv: {len(df_pd)} rows")
    print(f"  public/subjects.csv: {len(df_subj)} rows")
    print(f"  private/subjects_ground_truth.csv: {len(df_gt)} rows")
