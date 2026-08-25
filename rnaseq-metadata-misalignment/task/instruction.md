REPLACE THIS FILE. Per this repository's authoring rules (README.md, "Write
the agent contract"), instruction.md must be hand-written by the task's
human author, in their own voice, and an LLM asked to write it must refuse
-- which is why this is a skeleton and not a finished prompt. Everything
below the line is a fact sheet of what the verifier and reference solution
actually touch, so the file above is quick to write and nothing you need
gets left out by accident. Delete this notice and everything below the line
once you've written the real prompt.

---

Facts the verifier (task/tests/test_outputs.py) and reference solution
(task/solution/solve.py) depend on -- every one of these needs to appear in
your prompt or be obvious from the visible data, or an agent failure is
this task's bug, not a scientific one:

- Input files, at absolute paths under `/workspace/data/`:
  - `expression_matrix.csv` -- genes as rows, sample IDs as columns, raw
    integer read counts.
  - `sample_metadata.csv` -- columns `sample_id, condition, batch, qc_pass`.
    `condition` is `control` or `treated`.
  - `pipeline/` -- the previously-working analysis code (`io.py`, `qc.py`,
    `align.py`, `stats.py`, `run_pipeline.py`), runnable as
    `python -m pipeline.run_pipeline` from `/workspace/data/`. State that
    this ran successfully before and is now producing a result inconsistent
    with what was previously reported -- do not say which function or which
    dependency is implicated.
  - 13 samples total. Two of the sample IDs look like near-duplicates
    (`sample_2` / `sample_02`); both are real, independent samples.
- Output: exactly one file, `/workspace/output/result.json`, with keys:
  - `top_gene` (string)
  - `log2_fold_change` (number, treated vs. control)
  - `adjusted_p_value` (number)
  - `verified_matching_sample_ids` (integer) -- the number of samples whose
    expression-matrix identity and metadata `sample_id` were explicitly
    confirmed to match before the analysis was run. State this requirement
    plainly; do not describe how to compute it.
- Do not name: `np.float`, `sort_values`, `align.py`, `qc.py`, pandas, or any
  specific library/function. Do not say how many bugs there are or where
  they are. Do not mention `sample_02` behaving any differently from any
  other sample ID.
- Reasonable framing for the practitioner decision: a computational biology
  team is deciding whether to advance a compound based on its top
  transcriptional response gene; before, this pipeline reported one, but a
  recent change means its output can no longer be trusted without
  independent verification.

See `task/solution/process.md` for the intended diagnostic path and
`task/tests/test_outputs.py` for the exact tolerances and why they were set
that way.
