# Cpd-7 exposure-response species nomination

Cpd-7 is a small-molecule drug candidate metabolized in humans to two
circulating species, M1 and M2, alongside the unchanged parent compound. The
clinical pharmacology team needs to select a single exposure-response
biomarker for dose selection in later trials: the one species among parent,
M1, and M2 whose systemic exposure genuinely explains the observed
pharmacodynamic (PD) response.

Nominate the species (`parent`, `M1`, or `M2`) whose exposure-response
relationship is best supported by the available evidence, and report how
strong that relationship is.

## Data

The task inputs are available under `/workspace/data`:

- `pk_concentrations.csv`: sparse, noisy pharmacokinetic concentration-time
  samples. Columns: `subject_id`, `cohort`, `time_hr` (hours post first
  dose), `analyte` (`parent`, `M1`, or `M2`), `conc_ng_ml` (plasma
  concentration, ng/mL).
- `pd_response.csv`: one pharmacodynamic response measurement per subject.
  Columns: `subject_id`, `cohort`, `response`.
- `subjects.csv`: per-subject dosing record. Columns: `subject_id`,
  `cohort`, `dose_mg`, `n_doses` (number of doses administered before PK
  sampling), `tau_hr` (dosing interval in hours, where repeated dosing
  applies).

All three files share `subject_id` as the join key.

## Deliverable

Produce `/workspace/output/result.json`, a JSON object with exactly these
keys:

- `nominated_species`: one of `"parent"`, `"M1"`, `"M2"` — the species you
  nominate as the true PD driver.
- `single_dose_association`: a finite number in `[-1, 1]` quantifying the
  strength of the nominated species' exposure-response relationship among
  subjects who received a single dose.
- `multi_dose_association`: a finite number in `[-1, 1]` quantifying that
  same relationship among subjects who received repeated dosing to steady
  state.

Both association values must be genuinely derived from the visible data for
your nominated species, not hardcoded and not copied from a different
candidate. Read `DATA_DIR`/`OUTPUT_DIR` from the environment rather than
assuming the paths above. Make the computation reproducible from the visible
inputs.
