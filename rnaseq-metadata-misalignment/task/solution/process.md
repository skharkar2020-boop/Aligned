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

- **Naive pooled DE** (all 24 samples, cohort-blind): names a *different*
  gene outright here, not merely a contaminated estimate of the right one
  -- pooling a strong-cohort1/weak-cohort2 effect against a gene with a
  smaller but uniform effect in both cohorts favors the uniform one, once
  the two are close enough in overall strength.
- **Trusting cohort2 alone** (defensible on its face -- it's the later,
  "confirmatory" run, and its own top hit is dramatically significant):
  names a *different* gene, one with a huge effect confined almost
  entirely to cohort2. Its own cohort1 result now clears nominal
  significance on its own (p<0.05) -- but with the opposite sign, so
  checking significance alone, without checking direction, would wrongly
  call it replicated.
- **A sign-blind combined-p meta-analysis** (a real, common mistake:
  plugging each candidate gene's two independent cohort-level p-values
  into Fisher's method without first checking that the two cohorts agree
  on effect *direction*): on this dataset it happens to still land on the
  correct gene, but only because that gene's two p-values both happen to
  be individually smaller -- the method itself never checked whether the
  two cohorts were telling the same story, so citing it as the
  `analysis_strategy` describes an unprincipled method that got lucky
  here, not a defensible one.
- **Preferring whichever gene replicates most *consistently*** across
  cohorts, without also weighing how *strong* that evidence is: promotes a
  real, moderate, highly-consistent effect over a gene with a stronger
  overall case (a much stronger cohort1 effect that is real, but weaker
  and closer to noise, in cohort2). The consistent gene's combined
  evidence is now only about 6x weaker than the correct answer's (not
  1000x) -- a real, close call, not an easy miss. Consistency alone is not
  sufficient when it comes at the cost of materially weaker evidence.
- **Checking nominal significance in both cohorts without checking sign
  agreement**: admits the confounded gene as "replicated" on the strength
  of its now-nominally-significant cohort1 p-value alone, missing that the
  sign is reversed from its overwhelming cohort2 effect.
- **Flagging any gene with a dramatic, cohort-specific difference as a
  technical artifact by default**: a second gene exists that is
  independently significant in *each* cohort alone but with opposite
  signs, and carries no loading on the latent technical factor at all --
  it is real, sign-reversing biology, not a processing artifact. Rejecting
  it from `top_gene` is correct (it fails the same-sign replication
  requirement, same as the technical artifact does), but citing it as
  `rejected_competing_gene` is not: that field is reserved for the
  strongest candidate that fails replication, and the technical artifact's
  own single-cohort evidence is stronger.

None of these are coding errors -- each is a real, internally consistent,
non-crashing statistical analysis of correctly-ID-aligned data. That is
what makes reconciling them a judgment call rather than a lookup.

## The correct resolution

1. Run the same per-gene DE procedure (log2-CPM, Welch's t-test, BH-FDR)
   independently within `cohort1` and within `cohort2`. Never pool.
2. For any gene near the top of either cohort's own ranking, require BOTH
   that it show a *nominally* significant (raw p < 0.05, not
   BH-re-adjusted within cohort2) effect independently in the **other**
   cohort too, AND that the two cohorts' effects have the same sign.
   Nominal, not re-adjusted, is the deliberate choice, not an arbitrary
   one: this is a two-stage discovery-then-confirm design (cohort1 flags
   candidates, cohort2 checks whether they hold up), and standard practice
   in that design is to test the confirmatory cohort at a nominal
   threshold for the specific candidate already flagged, not to re-run a
   fresh whole-panel multiple-testing correction there. The candidate
   already paid its multiple-testing "cost" once, at the discovery stage,
   by having to be the best hit among 300 genes in cohort1; re-correcting
   across all 300 genes again in cohort2 penalizes it a second time for
   the same thing and makes a real, replicating effect look like it
   failed. (This was not a hypothetical concern: a fresh trial run against
   this exact dataset independently re-derived the whole analysis
   correctly -- crash and misalignment fixed, per-cohort split, same-sign
   replication check -- but applied BH-FDR within cohort2 as its
   replication bar instead of the nominal threshold, which flips the
   answer to CONSISTENCY_GENE: TRUE_GENE's cohort2 raw p=0.016 clears
   0.05 easily, but its cohort2-only BH-adjusted p=0.33 does not. Verified
   on the locked dataset, not asserted.) Checking significance without
   checking sign is also not sufficient, for a separate reason:
   the technical-artifact gene's own cohort1 result now clears p<0.05 on
   its own, and only fails because it points the opposite direction from
   its cohort2 effect. A gene whose apparent effect is confined to one
   cohort and shows nothing (not even a weak, same-direction nominal
   signal) in the other has failed to replicate for a different reason --
   that asymmetry, not a bigger number, is the signature of a
   cohort-specific technical artifact rather than biology. The technical
   artifact here loads heavily on a latent factor that only carries a real
   signal within `cohort2`; nothing in the public data names that factor
   directly, but genes that cluster together on it (moving together within
   `cohort2`, flat in `cohort1`) are visible in the expression data itself
   for anyone who looks for that structure (e.g. correlating candidate
   genes against each other within a cohort, or a quick PCA) -- this is
   useful corroborating evidence for the rationale, but the deterministic
   verifier does not require it: a small correlation-based "same technical
   module" check was tried during authoring and dropped, because at n=6
   per cohort-condition group it is too noisy to reliably separate a
   genuinely-entangled gene from an unrelated one, even averaged across
   every gene that loads on the factor.
   NOT every gene with a large, cohort-specific, sign-reversing effect is a
   technical artifact, either -- one candidate in this dataset is
   independently significant in each cohort alone, with opposite signs,
   but carries no loading on the latent factor at all: real biological
   heterogeneity that happens to flip sign between an original and a
   confirmatory cohort. It still fails the same-sign replication
   requirement (same as the artifact does), but it is not itself evidence
   of a processing problem, and should not be cited as the rejected
   competing gene when a stronger, genuinely-technical candidate exists.
3. More than one gene can pass the replication bar -- real biological
   effects are not required to be identical in magnitude across cohorts,
   and moderate heterogeneity (stronger in one cohort, weaker but real in
   the other, same direction) is expected, not disqualifying. Up to three
   candidates can clear it at once: the true gene, a smaller-but-consistent
   gene, and a gene whose apparent significance is partly inflated by
   the same latent factor as the technical artifact (its own cohort1
   effect is real and modest; its cohort2 effect looks more impressive
   than that alone would justify). Prefer the one with the strongest
   *combined* evidence (e.g. Fisher's method on the two nominal p-values,
   now legitimately applicable because direction agreement has already
   been confirmed) over the one that is merely the most uniform across
   cohorts, or the one with the single most dramatic cohort2 number. A
   smaller, very consistent effect is not automatically the safer answer
   if a materially stronger, still-replicating effect is available; nor is
   a gene automatically trustworthy just because it technically cleared
   the same-cohort bar.
4. Report the surviving gene's own result from whichever cohort gives it
   the stronger (smaller p-value) evidence, alongside both cohorts'
   individual fold-changes and a categorical description of how they
   relate to each other (consistent, or stronger in one cohort than the
   other), which cohort's own top hit was rejected and why, and the
   verified per-sample ID count.
5. A prior report exists on disk describing an earlier, much smaller,
   single-cohort pilot -- deliberately without naming a gene or giving
   numbers specific enough to identify one, so it cannot function as an
   answer shortcut. It is not sufficient evidence on its own; the
   analysis has to be established entirely from the current data.

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

- Naive pooled DE names a different gene outright (not merely a
  contaminated estimate of the right one).
- Trusting `cohort2` alone names a different gene entirely, with
  everything about it (fold-changes, heterogeneity label, rejected gene)
  wrong; its own cohort1 result is now nominally significant but
  wrong-signed.
- A sign-blind combined-p meta-analysis happens to still land on the
  correct gene here, but for reasons unrelated to sound method -- citing
  it as the strategy used is itself a defensible-sounding but incorrect
  description of how the answer was actually reached.
- Preferring the most cross-cohort-consistent gene over the one with
  materially stronger evidence names a real, replicating, but
  weaker-evidence gene -- wrong on every numeric field, though the margin
  is now close enough (~6x, not ~1000x) to be a genuine judgment call.
- Admitting a candidate as "replicated" on significance alone, without
  checking that both cohorts agree on effect direction, wrongly accepts
  the technical artifact.
- Citing the real-but-sign-reversing heterogeneous gene as the rejected
  competitor (rather than the genuine technical artifact) is wrong even
  though excluding it from `top_gene` is correct -- its own single-cohort
  evidence is real but weaker than the actual technical artifact's.

All are visibly different from the locked reference values on at least one
checked field, and each fails for a different underlying reason.
