REPLACE THIS FILE. The task structure changed since your last draft (two
cohorts that disagree, not a single cohort with a batch quirk) -- your
previous instruction.md described data that no longer exists. Per this
repository's authoring rules (README.md, "Write the agent contract"),
instruction.md must be hand-written by the task's human author, in their
own voice, and an LLM asked to write it must refuse -- which is why this is
a skeleton and not a finished prompt. Everything below the line is a fact
sheet of what the verifier and reference solution actually touch now, so
the file above is quick to write and nothing you need gets left out by
accident. Delete this notice and everything below the line once you've
written the real prompt.

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
  - `pipeline/` -- the previously-working analysis code, runnable as
    `python -m pipeline.run_pipeline` from `/workspace/data/`. State that
    this ran successfully before and is now producing a result inconsistent
    with what was previously reported -- do not say which function or which
    dependency is implicated, and do not say the pipeline is cohort-blind.
  - 24 samples total: 12 per cohort, 6 control / 6 treated within each.
- Output: exactly one file, `/workspace/output/result.json`, with keys:
  - `top_gene` (string)
  - `log2_fold_change` (number, treated vs. control)
  - `adjusted_p_value` (number)
  - `verified_matching_sample_ids` (integer) -- the number of samples whose
    expression-matrix identity and metadata `sample_id` were explicitly
    confirmed to match before the analysis was run. State this requirement
    plainly; do not describe how to compute it.
  - `confounded_cohort` (string, `"cohort1"` or `"cohort2"`) -- which
    cohort's own result the agent determined was not trustworthy, and why
    it isn't. State that the two cohorts, analyzed independently, disagree
    on the top gene, and that the agent must determine which cohort's
    result to trust and report the other as confounded. Do not say why they
    disagree, what causes the confound, or how to detect it.
- Do not name: `np.float`, `sort_values`, `align.py`, `qc.py`, pandas, or
  any specific library/function. Do not say how many code bugs there are or
  where they are. Do not mention processing dates, reagent lots, or noise
  levels -- the agent has to discover the confound's nature from the data
  itself.
- Reasonable framing for the practitioner decision: a computational biology
  team is deciding whether to advance a compound based on its top
  transcriptional response gene; a confirmatory second cohort was run, and
  its result doesn't match the original, so the team needs an
  independently-verified answer and an explanation of which run to trust
  before reporting further.

See `task/solution/process.md` for the intended diagnostic path and
`task/tests/test_outputs.py` for the exact tolerances and why they were set
that way.
