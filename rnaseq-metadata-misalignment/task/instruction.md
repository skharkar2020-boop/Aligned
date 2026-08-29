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

Use one of the following values for `analysis_strategy`:

* `strategy_a`
* `strategy_b`
* `strategy_c`
* `strategy_d`
* `strategy_e`

Use one of the following values for `heterogeneity_assessment`:

* `consistent_both_cohorts`
* `stronger_in_cohort1_weaker_in_cohort2`
* `stronger_in_cohort2_weaker_in_cohort1`
* `opposite_direction_between_cohorts`

All reported values should be derived from the supplied data and the final analysis. The `rejected_competing_gene` should represent the strongest alternative considered, and the `rationale` should briefly explain why the nominated gene was preferred.
