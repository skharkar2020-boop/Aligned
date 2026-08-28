We have an RNA-seq differential-expression pipeline that was previously trusted, but it is no longer reproducing the findings we expected. Your job is to work out what has gone wrong, make the necessary fixes to the existing pipeline, and identify the treatment-associated gene that is best supported by the data.

The dataset contains 24 samples from two independent cohorts and 300 genes. Raw RNA-seq counts are provided in `expression_matrix.csv`, while `sample_metadata.csv` contains the sample IDs, treatment groups, cohort information, and available QC information. There may also be older or pilot material in the repository. You can use this to understand the history of the analysis, but do not assume that the old result is correct.

Start by running the existing pipeline as it is. Work out whether there are problems in the way the data are being handled or assumptions that could make the differential-expression results unreliable. Fix what is necessary, but try to repair the existing workflow rather than replacing it with a completely different analysis.

One thing that should be checked carefully is sample matching. Make sure that the columns in the expression matrix actually correspond to the correct samples in the metadata by matching on `sample_id`. Do not rely on the files already being in the same order. Report whether you were able to verify the matching successfully.

Because the samples come from two independent cohorts, look at the treatment effect both across the full dataset and within each cohort. Consider whether conclusions from the pooled analysis are consistent with the cohort-specific results. Compare the strongest candidates across cohorts and consider their effect sizes, directions, statistical support, and how much the effects differ between cohorts.

Also pay attention to signs of technical structure in the data. Not every technical effect will necessarily appear as an obvious batch column in the metadata. If a particularly strong signal appears in only one cohort, investigate whether it is part of a wider expression pattern that could have a technical explanation. At the same time, do not assume that every difference between cohorts is technical or that an entire cohort should simply be removed. Some cohort-to-cohort variation may reflect genuine biology.

There is no required statistical formula or meta-analysis method for solving the task. Use an approach that makes sense for two independent cohorts and explain your reasoning. Different reasonable analyses may point toward different genes, so the final choice should be based on the overall strength and reliability of the evidence rather than simply picking the gene with the largest fold change, smallest p-value, or most similar effect between cohorts.

Save the final result to the required `result.json` location in exactly this format:

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

The `top_gene` should be the single treatment-associated gene you think is most defensible after repairing and validating the analysis. The reported statistics should come from your analysis of the supplied data rather than from historical results. The cohort-specific fold changes should also be calculated from the correctly matched data.

For `rejected_competing_gene`, report the strongest alternative that you seriously considered but ultimately rejected. In the `rationale`, briefly explain why the final gene is more convincing than that alternative.

Most importantly, treat this as an investigation rather than an attempt to reproduce a predetermined answer. Historical agreement by itself is not evidence that the pipeline is correct. The final result should be independently supported by the supplied data after you have checked the pipeline, sample identities, cohort behaviour, and possible technical structure.
