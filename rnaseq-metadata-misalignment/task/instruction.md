Task: Verify the top transcriptional hit for a candidate compound

1. Background

A computational biology team which profiles compounds for their transcriptional effects, comparing treated samples against controls, is trying to decide whether a particular compound is worth advancing. Normally this decision leans on a pipeline they've had running for a while, and up to now, whatever it spat out got trusted and passed along. That's no longer true. Something about the current output means it can't just be taken at face value anymore, and the result needs to be re-derived independently and actually checked before anyone reports it further.

So this isn't a "run the script, copy the number out" job. What's needed is a top-hit result someone could stand behind: correct, reproducible, and verified against the raw inputs rather than just trusted because the pipeline said so. How you get there, whether that's fixing the existing code, working around it, or rebuilding pieces of it, is up to your judgment.

2. Inputs

- `/workspace/data/expression_matrix.csv`: a genes × samples matrix of raw integer counts.
- `/workspace/data/sample_metadata.csv`: one row per sample, with columns `sample_id`, `condition`, `batch`, `qc_pass`. `condition` takes values `control` or `treated`.
- `/workspace/data/pipeline/`: the existing analysis code, run as a module via `python -m pipeline.run_pipeline`. This is the same code that used to produce trusted results, so it's a fine place to start, not something to throw out on principle.

There are 13 samples in total.

3. Constraints and context

- Heads up: the pipeline doesn't run cleanly out of the box right now, and you'll likely hit an error before you get any output at all. That's expected, not a sign that your environment is set up wrong.
- You won't be told how many things are wrong, where they are, or what kind of issues to expect. Go in ready to actually inspect and validate the code, rather than just poking at it until it stops complaining.
- Whatever you land on should be deterministic: someone else rerunning your analysis should get the same numbers you did.

4. Required output

Write a single file to `/workspace/output/result.json` with exactly these fields:

```json
{
  "top_gene": "string",
  "log2_fold_change": 0.0,
  "adjusted_p_value": 0.0,
  "verified_matching_sample_ids": 0
}
```

- `top_gene`: the top differentially expressed gene, treated vs. control.
- `log2_fold_change`: its log2 fold change, treated relative to control.
- `adjusted_p_value`: its multiple-testing-adjusted p-value.
- `verified_matching_sample_ids`: how many samples you explicitly confirmed, before running any analysis, actually match up between the expression matrix and the metadata's `sample_id` column. This has to be a check you do and count yourself, since it's not something the pipeline will tell you.
