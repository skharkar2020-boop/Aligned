# Cpd-7 exposure-response species nomination

Cpd-7 is a small-molecule drug candidate metabolized in humans to two
circulating species, M1 and M2, alongside the unchanged parent compound. The
clinical pharmacology team needs to select a single exposure-response
biomarker for dose selection in later trials: the one species among parent,
M1, and M2 whose systemic exposure genuinely explains the observed
pharmacodynamic (PD) response.

Nominate the species (`parent`, `M1`, or `M2`) whose exposure-response
relationship is best supported by the available evidence.

## Data

The task inputs are available under `/workspace/data`:

- `pk_concentrations.csv`: sparse, noisy pharmacokinetic concentration-time
  samples. Columns: `subject_id`, `cohort`, `time_hr`, `analyte` (`parent`,
  `M1`, or `M2`), `conc_ng_ml` (plasma concentration, ng/mL). `time_hr` is
  **absolute time since each subject's first dose** for every row, including
  subjects who received repeated doses.
- `pd_response.csv`: one pharmacodynamic response measurement per subject.
  Columns: `subject_id`, `cohort`, `response`.
- `subjects.csv`: per-subject dosing record. Columns: `subject_id`,
  `cohort`, `dose_mg`, `n_doses` (number of doses administered), `tau_hr`
  (dosing interval in hours, for subjects who received more than one dose).
  A subject's most recent dose occurred at
  `last_dose_time = (n_doses - 1) * tau_hr` (0 for a subject who received a
  single dose).

All three files share `subject_id` as the join key.

## Deliverable 1: `/workspace/output/subject_metrics.csv`

One row per `(subject_id, analyte)` pair — 96 rows total (32 subjects × 3
analytes), each present exactly once. Columns:

- `subject_id`, `analyte`
- `cmax`: the observed maximum concentration for that subject/analyte.
- `tmax_hr`: the time of that maximum, **relative to the subject's most
  recent dose** (i.e. `time_hr - last_dose_time`), not absolute time since
  the first dose.
- `auc0_6`, `auc0_12`: the area under the concentration-time curve over the
  first 6 and 12 hours **after the subject's most recent dose**, computed
  with the trapezoidal rule. A window's lower bound (time-since-last-dose =
  0) has no directly observed sample for a subject who received only one
  dose (the first visible sample is at 0.25h); anchor the integration with
  a synthetic `(0, 0)` point there, since concentration is 0 at the moment
  of a dose. A subject who received repeated doses has a genuine observed
  sample at time-since-last-dose = 0 (the pre-dose trough); use that
  observed value directly rather than assuming 0.
- `trough`: for a subject who received repeated doses, the observed
  concentration at time-since-last-dose = 0. Leave blank for a subject who
  received a single dose (no trough exists).
- `relevant_auc`: for a subject who received a single dose, the AUC from
  8h to 24h after that dose (the sustained/late exposure window). For a
  subject who received repeated doses, the steady-state AUC over one full
  dosing interval after the most recent dose (`auc0_12` computed over
  `[last_dose_time, last_dose_time + tau_hr]`) — the same value as
  `auc0_12` for that subject, by definition, not a separate computation.

## Deliverable 2: `/workspace/output/result.json`

A JSON object reporting, for **each of the three candidates** (`parent`,
`M1`, `M2`), the following population-level quantities derived from
`subject_metrics.csv` and `pd_response.csv`:

- `{species}_cmax`, `{species}_tmax_hr`: mean across subjects who received a
  single dose.
- `{species}_naive_pd_r`: Pearson correlation between `auc0_6` and PD
  response, across subjects who received a single dose, with no adjustment
  for dose.
- `{species}_cohort_adjusted_pd_r`: Pearson correlation between `auc0_12`
  and PD response after centering both variables within each of the four
  dose-level × dosing-regimen groups present in the data (i.e. subtracting
  each group's own mean before correlating) — equivalent to controlling for
  dose, regimen, and their interaction.
- `{species}_repeated_dose_pd_r`: Pearson correlation between
  `relevant_auc` and PD response, across subjects who received repeated
  doses only.
- `{species}_lagged_pd_r`: Pearson correlation between `relevant_auc` and
  PD response, across all subjects.
- `{species}_naive_min_loo_r`: the minimum Pearson correlation (same basis
  as `naive_pd_r`) obtained after removing any single subject (leave-one-out
  over the single-dose subjects), one at a time.
- `{species}_lagged_min_loo_r`: the minimum Pearson correlation (same basis
  as `lagged_pd_r`) obtained after removing any single subject, one at a
  time.
- `{species}_spearman_naive_pd_r`: the Spearman rank correlation (same basis
  as `naive_pd_r`: `auc0_6` vs. PD response, single-dose subjects).
- `{species}_bootstrap_pd_r_ci_low`: the 2.5th percentile of the Pearson
  correlation (same basis as `naive_pd_r`) across at least 1,000 bootstrap
  resamples (sampling subjects with replacement, same sample size each
  draw).

Additionally, for `M1` and `M2` only:

- `{species}_accumulation_ratio`: mean steady-state `relevant_auc` (across
  subjects who received repeated doses) divided by mean `auc0_12` (across
  subjects who received a single dose), matched across the full population
  (not per dose level).

Finally:

- `nominated_species`: one of `"parent"`, `"M1"`, `"M2"` — your final
  nomination.

Do not hardcode expected output values. All quantities must be genuinely
derived from the visible data using the trapezoidal rule for every AUC
calculation. Read `DATA_DIR`/`OUTPUT_DIR` from the environment rather than
assuming the paths above.
