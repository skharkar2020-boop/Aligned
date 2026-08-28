# Intended solution process

## Background for reviewers

Cpd-7 is metabolized to two circulating species, M1 and M2, alongside the
parent compound. **M2 is the true pharmacodynamic driver**: PD response is
generated (in the private data-generation process, never exposed to the
agent) as a function of M2's sustained/lagged exposure only — AUC[8h,24h]
for single-dose subjects, steady-state AUCτ for repeated-dose subjects.

**M1 is a highly plausible decoy.** Its formation and elimination are both
fast (Tmax ~1-2h, terminal half-life ~2h), giving it the highest Cmax, the
largest early-exposure AUC (AUC0-6), and a strong *naive, unadjusted*
correlation with PD — but that correlation is a Simpson's-paradox artifact.
M1's exposure scales with administered dose (deterministic PK), and PD (via
M2, which also scales with dose) rises with dose too, so M1 and PD move
together *between* the two dose cohorts even though M1 has no real
relationship with PD *within* a dose cohort: M1's within-subject variability
is independent per-subject noise, uncorrelated with the independent noise
on M2 that actually drives PD. M1 also shows essentially no accumulation
(it clears within one dosing interval), so its steady-state exposure looks
almost identical to its single-dose exposure.

**Parent** is a second, weaker decoy: it dominates the very earliest
concentrations (fastest absorption/decline of the three species) and its
naive correlation is comparable to M1's (again a dose-confound artifact,
not a real relationship — parent clears too fast to plausibly still be
active at the PD-assessment time), but it collapses just as hard once
dose/regimen is controlled for.

One subject in the single-low-dose cohort has a moderately (not
extremely) inflated M1 exposure together with an independent, noisy PD
elevation — a plausible, non-obvious co-occurrence, not a visually obvious
outlier — which inflates M1's naive correlation somewhat further and makes
that correlation *fragile*: it drops noticeably under leave-one-out, while
M2's real, distributed relationship does not.

None of this mechanism is stated in `instruction.md`. The agent must
discover it empirically by computing the disclosed set of per-candidate
metrics and comparing which candidate's relationship survives
dose/regimen-adjustment, repeated dosing, and single-subject removal, rather
than trusting the strongest-looking naive correlation.

## Steps

1. Inspect the three input tables. Notice `subjects.csv` gives `n_doses` and
   `tau_hr`, meaning `time_hr` in `pk_concentrations.csv` needs to be
   converted to time-since-most-recent-dose for `tmax_hr`/`auc0_6`/`auc0_12`
   to be well-defined and comparable across subjects on different regimens.
2. For each subject × analyte, compute `cmax`, `tmax_hr` (relative to the
   most recent dose), `auc0_6`, `auc0_12` (trapezoidal rule, anchored with a
   synthetic `(0,0)` point when the window's lower bound has no observed
   sample), `trough` (repeated-dose subjects only), and `relevant_auc`
   (AUC[8,24] for single-dose subjects; steady-state AUCτ, i.e. the same
   value as `auc0_12`, for repeated-dose subjects) — this is
   `subject_metrics.csv`.
3. Compute, per candidate species: a naive correlation (AUC0-6 vs PD, no
   adjustment), a dose×regimen-adjusted correlation (cohort-centered), a
   repeated-dose-only correlation, a pooled lagged/sustained-exposure
   correlation, and leave-one-out minima for the naive and lagged
   correlations. Compute Cmax/Tmax means and, for M1/M2, the accumulation
   ratio.
4. Compare the three candidates. In the naive, unadjusted view, M1 (and to
   a similar extent parent) can look as good as or better than M2 — that
   view alone is not sufficient evidence. Check which candidate's
   relationship survives dose/regimen adjustment, holds up under repeated
   dosing, reflects the pharmacologically relevant sustained-exposure
   window, and is not driven by any single subject. The species whose
   evidence generalizes across all of these is the one to nominate.
5. Validate: confirm the winning candidate's adjusted/repeated-dose/lagged
   correlations clearly exceed the other two candidates' corresponding
   values, not merely edge them out within noise, and confirm its
   leave-one-out minimum on the lagged metric stays high (robust), unlike
   the naive metric for the decoy candidates.
6. Write `subject_metrics.csv` and `result.json` from the environment
   variables `DATA_DIR`/`OUTPUT_DIR`.

## Validation performed

Re-running the pipeline against the same input files reproduces identical
subject-level and population-level values (no unseeded randomness in the
analysis itself). On the shipped dataset:

| metric | parent | M1 | M2 |
|---|---:|---:|---:|
| naive_pd_r | 0.684 | 0.667 | 0.408 |
| cohort_adjusted_pd_r | -0.117 | 0.119 | **0.609** |
| repeated_dose_pd_r | 0.807 | 0.682 | **0.968** |
| lagged_pd_r | 0.760 | 0.727 | **0.927** |

M2 dominates all three disclosed decision criteria (cohort-adjusted,
repeated-dose, lagged) by a wide margin (0.16-0.49) in every case, while
parent narrowly beats M1 on the naive metric alone (0.684 vs 0.667) —
confirming the naive view genuinely disagrees with the adjusted/robust
view, which is the intended trap. M1's naive correlation (0.667) is
noticeably fragile under leave-one-out (drops to 0.621 when the single
salted subject is removed), while M2's lagged correlation (0.927) stays
essentially unchanged under leave-one-out (0.905) — confirming M2's
evidence is a real, distributed signal and not an artifact of any one
subject. Manually verified that nominating `"M1"` or `"parent"` fails the
verifier's three-criterion dominance check (both fail on
`cohort_adjusted_pd_r` by a margin of roughly 0.5, far beyond the 0.05
threshold), while `"M2"` passes all checks.
