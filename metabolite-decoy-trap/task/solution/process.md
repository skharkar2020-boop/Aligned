# Intended solution process

## Background for reviewers

Cpd-7 is metabolized to two circulating species, M1 and M2. M1 is the true
pharmacodynamic driver (an Emax relationship between M1 exposure and
response). M2's apparent association with PD is a confound: a hidden
per-subject covariate independently (a) raises M2 exposure by lowering its
clearance and (b) shifts PD severity directly. In the single-dose data this
covariate makes M2 exposure track PD almost as well as M1. Under repeated
dosing to steady state, the pathway that clears M2 undergoes autoinduction,
which overrides the covariate-linked clearance and collapses M2's apparent
edge; M1's relationship holds up under both dosing regimens because its
formation and elimination are dose-proportional throughout. None of this
mechanism is stated in `instruction.md` — the agent has to discover it
empirically by checking whether each candidate's association survives across
both dosing regimens present in the data, rather than trusting a single
cohort or a naive pooled correlation.

## Steps

1. Inspect `pk_concentrations.csv`, `pd_response.csv`, and `subjects.csv`.
   Notice the data spans two dosing regimens, distinguished by `n_doses`
   (single administration vs. seven repeated doses at steady state) — this is
   the key structural fact the instruction does not call out by name.
2. For each subject and analyte (`parent`, `M1`, `M2`), integrate the sparse,
   noisy concentration-time samples with the trapezoidal rule to obtain a
   subject-level exposure (AUC) estimate. This is the standard non-compartmental
   estimate available from sparse sampling; no compartmental model needs to be
   fit.
3. Compute each candidate's exposure-response association (rank correlation
   between AUC and PD response — more robust to the nonlinear exposure-response
   shape and to outlier subjects than a raw linear correlation) separately
   within each dosing regimen. Do not pool subjects across regimens before
   correlating — the two
   regimens differ enough in AUC scale and subject count that pooling can
   distort or mask the relationship.
4. Compare the three candidates. In the single-dose data, the decoy can look
   as good as or better than the true driver — that view alone is not
   sufficient evidence. Check which candidate's association is strongest and
   holds up once the dosing regimen changes to repeated dosing at steady
   state, since that is the more demanding, better-characterized condition
   (each subject's exposure estimate is more stable at steady state, and
   confounds tied to a single-dose administration are more likely to wash
   out). The species whose relationship is dominant there is the one whose
   evidence generalizes.
5. Validate: confirm the winning candidate is not merely the best of three by
   a negligible margin — its association in the steady-state regimen should
   clearly exceed the other two candidates, not edge them out within noise.
6. Write `result.json` with the nominated species and its association value
   in each regimen, from the environment variables `DATA_DIR`/`OUTPUT_DIR`.

## Validation performed

Re-running the pipeline against the same input files reproduces identical
AUC and correlation values (no unseeded randomness in the analysis itself).
The three candidates' steady-state associations were inspected side by side
to confirm the winning margin is not a coin flip: the reference computation
finds the nominated species' steady-state association clearly ahead of the
runner-up, while the runner-up in the single-dose-only view is a different
species entirely — i.e., a generalization-aware analysis and a single-cohort
analysis disagree, which is the intended trap.
