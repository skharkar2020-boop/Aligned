REPLACE THIS FILE. The task structure changed since your last draft (the
`confounded_cohort` field is gone; neither cohort can be discarded
wholesale anymore -- both carry real evidence, and the reconciliation now
requires weighing evidence strength, not just picking a side) -- your
previous instruction.md described an output schema that no longer exists.
Per this repository's authoring rules (README.md, "Write the agent
contract"), instruction.md must be hand-written by the task's human
author, in their own voice, and an LLM asked to write it must refuse --
which is why this is a skeleton and not a finished prompt. Everything
below the line is a fact sheet of what the verifier and reference solution
actually touch now, so the file above is quick to write and nothing you
need gets left out by accident. Delete this notice and everything below
the line once you've written the real prompt.

---

Facts the verifier (task/tests/test_outputs.py) and reference solution
(task/solution/solve.py) depend on -- every one of these needs to appear in
your prompt or be obvious from the visible data, or an agent failure is
this task's bug, not a scientific one:

- Input files, at absolute paths under `/workspace/data/`:
  - `expression_matrix.csv` -- genes as rows, sample IDs as columns, raw
    integer read counts.
  - `sample_metadata.csv` -- columns `sample_id, condition, cohort, qc_pass`.
    `condition` is `control` or `treated`. `cohort` is `cohort1` or
    `cohort2` -- two independent runs (cohort2 later, a confirmatory
    replicate).
  - `prior_pilot_report.md` -- a short prior report naming a candidate top
    gene from an earlier, much smaller, single-cohort pilot. State plainly
    that it exists and that its numbers are from an underpowered,
    non-confirmatory analysis and must not be reported as-is; do not say
    which gene it names or by how much its numbers are off.
  - `pipeline/` -- the previously-working analysis code, runnable as
    `python -m pipeline.run_pipeline` from `/workspace/data/`. State that
    this ran successfully before and is now producing a result inconsistent
    with what was previously reported -- do not say which function or which
    dependency is implicated, and do not say the pipeline is cohort-blind.
  - 24 samples total: 12 per cohort, 6 control / 6 treated within each.
- Output: exactly one file, `/workspace/output/result.json`, with keys:
  - `top_gene` (string)
  - `log2_fold_change` (number, treated vs. control, from whichever
    cohort's own analysis gives this gene its stronger evidence)
  - `adjusted_p_value` (number, same cohort as log2_fold_change)
  - `analysis_strategy` (string) -- must be one of exactly:
    `"pooled"`, `"cohort1_only"`, `"cohort2_only"`,
    `"fixed_effect_meta_analysis"`, `"per_cohort_independent_replication"`.
    State this closed list verbatim; do not say which one is correct or
    how to decide.
  - `cohort1_log2_fold_change` (number) / `cohort2_log2_fold_change`
    (number) -- the reported top gene's own fold-change computed
    independently within each cohort.
  - `heterogeneity_assessment` (string) -- must be one of exactly:
    `"consistent_both_cohorts"`, `"stronger_in_cohort1_weaker_in_cohort2"`,
    `"stronger_in_cohort2_weaker_in_cohort1"`,
    `"opposite_direction_between_cohorts"`. State this closed list
    verbatim; do not say how to classify a given pair of numbers into it.
  - `verified_matching_sample_ids` (boolean) -- whether the agent
    explicitly confirmed, for every sample, that its expression-matrix
    identity and metadata `sample_id` match before the analysis was run.
    State this requirement plainly; do not describe how to compute it.
  - `rejected_competing_gene` (string) -- the single strongest competing
    hit that the agent considered and ruled out, and why. State that at
    least one other gene will look like a strong candidate and turn out
    not to hold up under scrutiny, and that the agent must name it.
  - `rationale` (string) -- a short free-text explanation of the
    reasoning; not graded on content, only checked for presence.
- Reasonable framing for the practitioner decision: a computational biology
  team is deciding whether to advance a compound based on its top
  transcriptional response gene; a confirmatory second cohort was run, the
  pipeline that used to produce this analysis is behaving inconsistently,
  and more than one gene will look like a plausible answer depending on
  how the two cohorts are combined -- the team needs an
  independently-verified answer, not just a rerun of the old code.
- Do not name: `np.float`, `sort_values`, `align.py`, `qc.py`, pandas,
  `TRUE_GENE`/`CONFOUND_GENE`/`CONSISTENCY_GENE`/`SENTINEL`, or any
  specific library/function/gene. Do not say how many code bugs there are
  or where they are. Do not say how the two cohorts should be combined,
  what a "latent factor" is, or that any gene's technical signature is
  inferable from correlations with other genes -- the agent has to
  discover all of this from the data itself.

See `task/solution/process.md` for the intended diagnostic path and
`task/tests/test_outputs.py` for the exact tolerances and why they were set
that way.
