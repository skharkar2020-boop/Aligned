# Intended solution process

## Background for reviewers

The environment ships a real, previously-working differential-expression
pipeline (`/workspace/data/pipeline/`) that the agent is told is now
producing results that don't match what was reported before. The task is
not to write a DE pipeline from scratch; it is to run the one provided,
diagnose why it no longer produces a trustworthy answer, and fix it.

Three independent code defects are layered into that pipeline, and fixing
them in the wrong order (or only partially) each produces a different,
non-crashing, individually plausible wrong answer rather than an error:

1. `pipeline/io.py` casts the expression matrix with `np.float`, an alias
   NumPy removed in 1.24. The pipeline crashes on the very first run. This
   is a one-line fix and it is tempting to treat it as "the bug" -- the
   pipeline does go on to run and print a complete-looking ranked gene
   table once it's fixed.
2. `pipeline/qc.py`'s per-sample QC report sorts the metadata table by
   `sample_id` as a string for its printout, and that sorted table is what
   the rest of the pipeline then uses as the canonical sample list. String
   sorting of `sample_1 .. sample_24` is not numeric order (`sample_10`
   sorts before `sample_2`), so this silently reorders the metadata
   relative to the expression matrix's own sample order.
3. `pipeline/align.py` combines expression and metadata by resetting both
   to a plain positional index and concatenating them side by side, rather
   than joining on `sample_id`. The expression matrix's columns are in
   sequencer acquisition order (samples are randomized across lanes so
   lane is not confounded with condition) and were never guaranteed to
   match the metadata's row order to begin with, so this was unsafe even
   before (2) made it worse. `align.py` also branches on the installed
   pandas major version (a leftover from when the pipeline supported
   pandas 1.x and 2.x side by side); neither branch is index-based, so
   upgrading the pandas pin changes which wrong permutation you get rather
   than fixing anything.

The `assert combined.shape[0] == metadata.shape[0]` immediately after the
merge is a real check that was already in the code, and it always passes
-- it confirms sample *count* survived the merge, not that each row's
expression profile still belongs to the sample_id printed next to it.

Fixing the alignment is necessary but not sufficient. Even with every
sample correctly ID-matched, the pipeline pools all 24 samples into one
comparison, with no awareness that they come from two different cohorts.
`sample_metadata.csv`'s `cohort` column separates `cohort1`
(`sample_1`-`sample_12`) from `cohort2` (`sample_13`-`sample_24`), a later,
independent confirmatory run. Analyzed separately, the two cohorts
disagree: each produces a complete, internally consistent,
non-crashing differential-expression result, and they name two different
top genes. Pooling all 24 samples (the pipeline's default once alignment
is fixed) does not resolve this -- it happens to still name the right gene
in this dataset, but with a fold-change and p-value contaminated by mixing
in the confounded cohort, which is why the verifier checks the actual
numbers and not just the gene name.

The correct resolution requires recognizing which cohort's result is not
trustworthy and why. `cohort2`'s control and treated samples were
processed at different times (a real, staggered-processing/reagent-lot
confound, confounded with condition only within that cohort), and its own
top gene under this analysis does not hold up in `cohort1` at all -- not
even a same-signed nominal signal. `cohort1`'s top gene, by contrast, does
show up in `cohort2` too: weaker and short of formal significance in the
noisier `cohort2`, but a real, same-signed, nominally significant effect
-- unlike `cohort2`'s own top gene in `cohort1`. That asymmetry is the
evidence: the gene whose effect only appears alongside a specific,
identifiable confound is the artifact; the gene whose effect persists to
some degree even without that confound is the real one.

## Steps

1. Run `python -m pipeline.run_pipeline` (or equivalent) against
   `/workspace/data/`. It crashes on `np.float`. Fix the cast (`float` or
   `np.float64`); this is a real but unrelated packaging issue, not the
   substance of the task.
2. Run it again. It now completes and prints a full ranked
   differential-expression table -- but re-running after only swapping the
   pandas version pinned in the environment changes which gene comes out
   on top, a strong sign the alignment between expression columns and
   metadata rows, not the statistics, is broken.
3. Trace the pipeline's own merge step (`align.py`) and the metadata
   handling that feeds it (`qc.py`) rather than trusting the shape
   assertion. Confirm, sample by sample, that the `sample_id` a row's
   expression profile came from is the same `sample_id` its condition
   label came from.
4. Reimplement the merge so it joins the expression matrix's columns to
   the metadata's `sample_id` values explicitly, never by resetting both
   to a positional index first.
5. Before reporting, explicitly count how many of the 24 samples have
   their expression-matrix identity and metadata `sample_id` confirmed
   equal at the row used in the analysis, and report that count alongside
   the result.
6. Notice the `cohort` column and split the analysis by cohort instead of
   pooling. Run the same per-gene differential-expression procedure
   independently within `cohort1` and within `cohort2`.
7. Compare the two cohorts' top genes. For each cohort's own top gene,
   check whether it shows any real, same-signed signal in the *other*
   cohort (a much lower bar than formal significance there, since one
   cohort is noisier). The cohort whose top gene does not clear even that
   bar in the other cohort is the confounded one; exclude its result.
8. Write `result.json` with the top gene's symbol, its log2 fold-change
   and BH-adjusted p-value from the *trustworthy* cohort's own analysis,
   the total verified sample count (24, from step 5), and which cohort was
   identified as confounded, using `DATA_DIR`/`OUTPUT_DIR`.

## Validation performed

Re-running the fixed pipeline against the same input files under two
different installed pandas major versions (1.x and 2.x) reproduces
identical output. The reference result was cross-checked against an
independent, from-scratch recomputation (same CPM/t-test/BH procedure and
replication check, implemented separately from both the pipeline module
and solve.py) that joins on `sample_id` directly from the two raw input
files; the two agree exactly.

Several wrong-but-plausible scenarios were run against the same data as an
internal check that the verifier would actually catch them, none of which
crash or produce a NaN:
- Both misalignment branches (pandas<2 and pandas>=2, after only fixing
  the crash) land on unrelated null genes, neither statistically
  significant.
- Pooling all 24 correctly-ID-verified samples without splitting by
  cohort reports the right gene but a fold-change contaminated by the
  confounded cohort (2.31 vs. the correct 2.70 -- outside tolerance).
- Trusting `cohort2`'s own result (correct alignment, correct per-sample
  verification, but the wrong cohort) reports a different, still
  confidently significant top gene, and reports the wrong
  `confounded_cohort` as well -- the closest wrong answer to correct on
  every field except the one that actually required cross-cohort
  reasoning.

All are visibly different from the locked reference values on at least one
checked field.
