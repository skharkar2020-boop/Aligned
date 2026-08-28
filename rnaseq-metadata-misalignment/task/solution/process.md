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

## Why fixing the alignment is necessary but not sufficient

Even with every sample correctly ID-matched, `sample_metadata.csv`'s
`cohort` column separates `cohort1` (`sample_1`-`sample_12`, the original
run) from `cohort2` (`sample_13`-`sample_24`, a later, independent
confirmatory run). The pipeline pools all 24 samples into one comparison
by default, with no cohort awareness at all -- and several plausible ways
of handling the two cohorts each name a different top gene:

- **Naive pooled DE** (all 24 samples, cohort-blind): still names the
  correct gene here, but its fold-change and p-value are contaminated by
  mixing a strong, low-noise cohort1 effect with a much weaker cohort2
  effect for the same gene. The number does not match either cohort's own
  honest estimate and falls outside the verifier's tolerance.
- **Trusting cohort2 alone** (defensible on its face -- it's the later,
  "confirmatory" run, and its own top hit is dramatically significant):
  names a *different* gene, one with a huge effect confined entirely to
  cohort2 and no support at all in cohort1.
- **A sign-blind combined-p meta-analysis** (a real, common mistake:
  plugging each candidate gene's two independent cohort-level p-values
  into Fisher's method without first checking that the two cohorts agree
  on effect *direction*): lands on the same wrong gene as trusting cohort2
  alone. That gene's extremely small cohort2 p-value dominates the
  combination even though cohort1 shows essentially no effect (and a
  slightly opposite sign) -- the combination never checked whether the two
  cohorts were actually telling the same story.
- **Preferring whichever gene replicates most *consistently*** across
  cohorts, without also weighing how *strong* that evidence is: promotes a
  real, moderate, highly-consistent effect over a gene with a stronger
  overall case (a much stronger cohort1 effect that is real, but weaker
  and closer to noise, in cohort2). Consistency alone is not sufficient
  when it comes at the cost of materially weaker evidence.

None of these four are coding errors -- each is a real, internally
consistent, non-crashing statistical analysis of correctly-ID-aligned
data. That is what makes reconciling them a judgment call rather than a
lookup.

## The correct resolution

1. Run the same per-gene DE procedure (log2-CPM, Welch's t-test, BH-FDR)
   independently within `cohort1` and within `cohort2`. Never pool.
2. For any gene near the top of either cohort's own ranking, require that
   it also show a *nominally* significant (raw p < 0.05 -- not
   BH-adjusted; cohort2 is noisier, so a real effect is not expected to
   survive multiple-testing correction there), *same-signed* effect
   independently in the **other** cohort too. A gene whose apparent effect
   is confined to one cohort and shows nothing (not even a weak,
   same-direction nominal signal) in the other has failed to replicate --
   that asymmetry, not a bigger number, is the signature of a
   cohort-specific technical artifact rather than biology. The rejected
   gene here loads heavily on a latent factor that only carries a real
   signal within `cohort2`; nothing in the public data names that factor
   directly, but genes that cluster together on it (moving together
   within `cohort2`, flat in `cohort1`) are visible in the expression data
   itself for anyone who looks for that structure (e.g. correlating
   candidate genes against each other within a cohort, or a quick PCA).
3. More than one gene can pass that replication bar -- real biological
   effects are not required to be identical in magnitude across cohorts,
   and moderate heterogeneity (stronger in one cohort, weaker but real in
   the other, same direction) is expected, not disqualifying. When more
   than one candidate clears the bar, prefer the one with the stronger
   *combined* evidence (e.g. Fisher's method on the two nominal p-values,
   now legitimately applicable because direction agreement has already
   been confirmed) over the one that is merely the most uniform across
   cohorts. A smaller, very consistent effect is not automatically the
   safer answer if a materially stronger, still-replicating effect is
   available.
4. Report the surviving gene's own result from whichever cohort gives it
   the stronger (smaller p-value) evidence, alongside both cohorts'
   individual fold-changes and a categorical description of how they
   relate to each other (consistent, or stronger in one cohort than the
   other), which cohort's own top hit was rejected and why, and the
   verified per-sample ID count.
5. A prior report exists on disk naming the same gene from an earlier,
   much smaller, single-cohort pilot. Its numbers do not match this
   dataset's independently recomputed result (different n, no
   confirmatory cohort) and are not sufficient evidence on their own --
   the analysis has to be reproduced from the current data, not copied
   from that file.

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
   equal at the row used in the analysis.
6. Notice the `cohort` column and split the analysis by cohort instead of
   pooling. Run the same DE procedure independently within `cohort1` and
   within `cohort2`.
7. For each cohort's own top candidates, check whether they hold up
   (nominal significance, same sign) in the *other* cohort. Reject
   whichever prominent candidate fails that check, and identify why its
   evidence is concentrated in one cohort (look for other genes that move
   together with it, confined to that same cohort -- a latent technical
   axis, not biology).
8. Among genes that do replicate in both cohorts, prefer the one with the
   stronger combined evidence over the one that is merely the most
   uniform, unless the "stronger" candidate is actually the rejected
   artifact from step 7.
9. Write `result.json` with the surviving gene's symbol, its log2
   fold-change and BH-adjusted p-value from its stronger-evidence cohort,
   which analysis strategy was used, both cohorts' own fold-changes for
   that gene, a categorical heterogeneity assessment, the verified sample
   count, which competing gene was rejected, and a short rationale --
   using `DATA_DIR`/`OUTPUT_DIR`.

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
crash or produce a NaN -- see task/README.md's calibration table for the
actual numbers:

- Naive pooled DE names the correct gene but a contaminated fold-change
  and adjusted p-value, both outside tolerance.
- Trusting `cohort2` alone names a different gene entirely, with
  everything about it (fold-changes, heterogeneity label, rejected gene)
  wrong.
- A sign-blind combined-p meta-analysis lands on the same wrong gene as
  trusting `cohort2` alone, for a different (but equally plausible)
  reason.
- Preferring the most cross-cohort-consistent gene over the one with
  materially stronger evidence names a real, replicating, but
  weaker-evidence gene -- wrong on every numeric field.

All are visibly different from the locked reference values on at least one
checked field, and each fails for a different underlying reason.
