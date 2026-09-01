# RNA-seq Treatment-Associated Gene Identification

A scientist is analyzing RNA-seq data to identify a gene associated with treatment across two independent cohorts.

An existing differential-expression pipeline has previously been used for this analysis, but the results are no longer reproducing the findings that were expected. The goal is to investigate the existing analysis, make any necessary corrections, and identify the treatment-associated gene that is best supported by the data.

The data are available in `/workspace/data/`.

`/workspace/data/expression_matrix.csv` contains raw RNA-seq counts for 300 genes across 24 samples.

`/workspace/data/sample_metadata.csv` contains the corresponding sample information, including treatment group, cohort, and available QC information.

Additional historical or pilot material may be present in the repository and can be used for context, but previous results should not be assumed to be correct.

Analyze the data and nominate one scientifically defensible treatment-associated gene for further investigation.

Save the final result to `/workspace/output/result.json` using this JSON schema:

```json
{
  "top_gene": "<gene>",
  "log2_fold_change": 0.0,
  "adjusted_p_value": 0.0,
  "analysis_strategy": "<strategy>",
  "cohort1_log2_fold_change": 0.0,
  "cohort2_log2_fold_change": 0.0,
  "heterogeneity_assessment": "<assessment>",
  "verified_matching_sample_ids": true,
  "rejected_competing_gene": "<gene>",
  "rationale": "<brief scientific justification>"
}
```

Use one of the following values for `analysis_strategy`, choosing whichever one actually describes the approach used to reach the final answer:

* `strategy_a` — pool all samples from both cohorts into a single combined analysis, without separating by cohort.
* `strategy_b` — use only cohort1's samples.
* `strategy_c` — use only cohort2's samples.
* `strategy_d` — analyze each cohort separately, then combine the two cohorts' effect-size estimates into one using a formal meta-analysis procedure.
* `strategy_e` — analyze each cohort separately, and treat a candidate gene as trustworthy only if it independently shows a significant effect in each cohort on its own, with the effect pointing in the same direction in both cohorts, rather than mathematically combining the two cohorts' results into a single number.

More than one of these strategies can look like a reasonable choice on its own, and the resulting nomination and supporting numbers will differ depending on which one was actually used. The value reported here should be the one that genuinely describes how the final, defensible answer was reached, not simply a plausible-sounding label.

When checking whether a candidate flagged in one cohort also holds up in the other, use that other cohort's own nominal (uncorrected) p-value for the confirmation test, not a p-value freshly corrected for multiple testing across the whole gene panel again. A candidate already competed against the full panel once, at the cohort where it was first identified; re-applying full-panel correction a second time in the confirming cohort tests it against that same multiple-testing burden twice and penalizes a real, replicating effect for having already cleared it once.

`log2_fold_change` and `adjusted_p_value` should be the nominated gene's own statistics from whichever single cohort gives it the stronger statistical support — not a pooled or averaged figure. `cohort1_log2_fold_change` and `cohort2_log2_fold_change` are where each cohort's own individual estimate belongs.

Use one of the following values for `heterogeneity_assessment`:

* `consistent_both_cohorts`
* `stronger_in_cohort1_weaker_in_cohort2`
* `stronger_in_cohort2_weaker_in_cohort1`
* `opposite_direction_between_cohorts`

All reported values should be derived from the supplied data and the final analysis. The `rejected_competing_gene` should represent the strongest alternative considered, and the `rationale` should briefly explain why the nominated gene was preferred. Specifically, if the rejected candidate's own treatment effect points in a different direction between the two cohorts, the rationale should say so explicitly rather than only citing significance — direction and significance are separate properties, and a rejection has to address whichever one actually applies.
