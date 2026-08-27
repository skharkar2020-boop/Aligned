# Intended solution process

## Background for reviewers

The environment ships a real, previously-working differential-expression
pipeline (`/workspace/data/pipeline/`) that the agent is told is now
producing results that don't match what was reported before. The task is
not to write a DE pipeline from scratch; it is to run the one provided,
diagnose why it no longer produces a trustworthy answer, and fix it. Three
independent defects are layered into that pipeline, and fixing them in the
wrong order (or only partially) each produces a different, non-crashing,
individually plausible wrong answer rather than an error.

1. `pipeline/io.py` casts the expression matrix with `np.float`, an alias
   NumPy removed in 1.24. The pipeline crashes on the very first run. This
   is a one-line fix and it is tempting to treat it as "the bug" -- the
   pipeline does go on to run and print a complete-looking ranked gene
   table once it's fixed.
2. `pipeline/qc.py`'s per-sample QC report sorts the metadata table by
   `sample_id` as a string for its printout, and that sorted table is what
   the rest of the run then uses as the canonical sample list. String
   sorting of `sample_1 .. sample_12` is not numeric order
   (`sample_10 < sample_2` lexicographically), so this silently reorders
   the metadata relative to the expression matrix's own sample order.
3. `pipeline/align.py` combines expression and metadata by resetting both
   to a plain positional index and concatenating them side by side, rather
   than joining on `sample_id`. The expression matrix's columns are in
   sequencer acquisition order (samples are randomized across lanes so lane
   is not confounded with condition) and were never guaranteed to match the
   metadata's row order to begin with, so this was unsafe even before (2)
   made the metadata order diverge further. `align.py` also branches on the
   installed pandas major version (a leftover from when the pipeline
   supported pandas 1.x and 2.x side by side); neither branch is
   index-based, so upgrading the pandas pin changes which wrong
   permutation you get rather than fixing anything.

The `assert combined.shape[0] == metadata.shape[0]` immediately after the
merge is a real check that was already in the code, and it always passes --
it confirms sample *count* survived the merge, not that each row's
expression profile still belongs to the sample_id printed next to it.

Two of the thirteen samples, `sample_2` and `sample_02`, are a genuine
near-duplicate-looking pair: distinct biological replicates from two
collection batches with different ID conventions, in opposite conditions.
A fix that "cleans up" IDs by normalizing away things like leading zeros
before joining, rather than joining on the literal `sample_id` string, will
merge these two into one and silently corrupt the analysis in the same
undetectable way as the original bug.

Fixing the alignment is necessary but not sufficient. `sample_02` is also
the sole member of `batch` `batch2`; every other sample is `batch1`. A
batch with exactly one sample is perfectly confounded with itself: nothing
in the data can separate "this is a real biological effect in this sample"
from "this is a technical artifact of processing it differently," because
there is no second batch2 sample to compare it against. The scientifically
defensible move is to run the differential-expression comparison on the
samples from batches large enough to support that distinction -- here, the
12 batch1 samples -- while still verifying `sample_02`'s identity like any
other sample. Running the comparison on all 13 without accounting for this
produces a different, still non-crashing, still nominally significant top
gene: not an error, a wrong answer that happens to look complete.

## Steps

1. Run `python -m pipeline.run_pipeline` (or equivalent) against
   `/workspace/data/`. It crashes on `np.float`. Fix the cast (`float` or
   `np.float64`); this is a real but unrelated packaging issue, not the
   substance of the task.
2. Run it again. It now completes and prints a full ranked
   differential-expression table -- but the top hit is not statistically
   convincing (its adjusted p-value is not small) and, critically, nothing
   in the run's output says so explicitly. Re-running after only swapping
   the pandas version pinned in the environment changes which gene comes
   out on top, which is a strong sign the alignment between expression
   columns and metadata rows -- not the statistics -- is the thing that's
   broken.
3. Trace the pipeline's own merge step (`align.py`) and the metadata
   handling that feeds it (`qc.py`) rather than trusting the shape
   assertion. Confirm, sample by sample, that the `sample_id` a row's
   expression profile came from is the same `sample_id` its condition label
   came from -- shape equality does not establish this.
4. Reimplement the merge so it joins the expression matrix's columns to the
   metadata's `sample_id` values explicitly (e.g. `expr[metadata["sample_id"]]`
   or an explicit ID-keyed merge), never by resetting both to a positional
   index first. Keep `sample_2` and `sample_02` as the two distinct IDs they
   are; do not normalize, fuzzy-match, or deduplicate sample identifiers.
5. Before reporting, explicitly count how many of the 13 samples have their
   expression-matrix identity and metadata `sample_id` confirmed equal at
   the row used in the analysis (not just that the row counts match) and
   report that count alongside the result. A pipeline that is really
   joining by ID gets all 13; a pipeline still joining by position, however
   it was patched, will not. This count is independent of step 6 below --
   every sample gets ID-verified regardless of whether it ends up usable
   for the statistical comparison.
6. Check each sample's `batch`. `sample_02` is the only member of its
   batch; a batch of one cannot be separated from a real condition effect.
   Exclude it from the differential-expression comparison for that reason
   (not because its ID is ambiguous -- it already passed step 5) and use
   the 12 batch1 samples for the comparison itself.
7. Recompute log2(CPM + 1) per sample and run the pipeline's own
   differential-expression step (Welch's t-test per gene, treated vs.
   control, Benjamini-Hochberg FDR correction) on the correctly joined,
   batch-filtered table. Take the gene with the smallest adjusted p-value
   as the top hit.
8. Write `result.json` with the top gene's symbol, its log2 fold-change
   (treated vs. control), its BH-adjusted p-value, and the verified sample
   count (13, from step 5 -- not the 12 used in step 7), using
   `DATA_DIR`/`OUTPUT_DIR`.

## Validation performed

Re-running the fixed pipeline against the same input files under two
different installed pandas major versions (1.x and 2.x) reproduces
identical output -- confirming the fix is genuinely ID-based rather than
incidentally correct for one pandas version's default behavior. The
reference result was cross-checked against an independent, from-scratch
recomputation (same CPM/t-test/BH procedure, implemented separately from
the pipeline module) that joins on `sample_id` directly from the two raw
input files and applies the same batch filter; the two agree exactly.

Several wrong-but-plausible scenarios were run against the same data as an
internal check that the verifier would actually catch them, none of which
crash or produce a NaN:
- Both misalignment branches (pandas<2 and pandas>=2, after only fixing the
  crash) land on two different null genes, neither statistically
  significant.
- Dropping `sample_02` as a perceived duplicate of `sample_2` reports only
  12 verified samples instead of 13.
- Mislabeling `sample_02`'s condition (the failure mode of matching it to
  `sample_2`'s metadata row) knocks the true top gene out of first place
  entirely, with nothing reaching significance.
- Running the comparison on all 13 correctly-ID-verified samples --
  correct alignment, but without the batch-2 exclusion -- reports a
  *different* gene as top, at an adjusted p-value (~1e-3) that clears an
  unwary "is it significant" check on its own. This is the closest wrong
  answer to correct: everything about it is internally consistent except
  the batch judgment, and it is deliberately calibrated to be a confident,
  plausible-looking wrong answer rather than an obviously broken one.

All are visibly different from the locked reference values on at least one
checked field.
