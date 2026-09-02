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
  on effect *direction*): the method itself never checked whether the two
  cohorts were telling the same story, and on this dataset that omission
  is costly -- combined with never checking variance stability either, it
  lands on the round-7 variance-fragile decoy (see below), not the
  strongest generalizable candidate.
- **Trusting the full, otherwise-correct per-cohort replication workflow**
  (ID-based alignment, per-cohort DE, same-sign nominal significance in
  both cohorts, leave-one-out robustness, preferring the strongest RAW
  combined evidence among survivors) **without also checking variance
  stability**: a gene exists in this dataset that is mechanically an
  entirely ordinary hit -- the same generative process as the strongest
  candidate (same baseline expression level, same true underlying
  variability), differing only in which natural random draw it happened
  to land on -- and clears every one of those checks more convincingly:
  BH-significant in both cohorts (the correct answer is BH-significant
  only in cohort1), a more consistent effect size across cohorts, and
  stronger raw combined evidence. The reason is that its own within-group
  sample variance in one cohort happened, purely by chance, to come out
  anomalously small: with only 6 samples per group (5 residual degrees of
  freedom), a single gene's own variance estimate is itself a noisy
  quantity, and an unusually small realized variance inflates the
  apparent significance of an otherwise-ordinary effect. This is a
  genuinely different failure from the others in this list: it is not a
  missing check for sign, significance, or leave-one-out robustness -- it
  is a completely correct execution of the full replication procedure
  that still reaches the wrong answer, because that procedure alone never
  asks whether a candidate's own apparent strength is an artifact of a
  favorable small-sample variance estimate.
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
3. A candidate that passes the same-sign, nominally-significant check in
   both cohorts still is not automatically trustworthy: check whether that
   significance is *robust*, not just present. Within the cohort where the
   candidate's confirming evidence is weakest, drop each treated sample
   once (leave-one-out) and recompute its p-value against the full control
   group; the worst (largest) p-value seen across those single-sample
   drops must still clear the same nominal significance bar used
   everywhere else. A candidate whose apparent replication only holds
   because of one or two individual samples is not a real replicated
   effect, however clean and consistent its headline fold-changes look.
   One designed gene in this dataset clears ordinary same-sign nominal
   significance in both cohorts, with a strikingly *consistent* effect
   size across them, and still fails this check -- its cohort2 result
   depends on two specific samples and evaporates without them.
4. More than one gene can pass the replication-plus-robustness bar --
   real biological effects are not required to be identical in magnitude
   across cohorts, and moderate heterogeneity (stronger in one cohort,
   weaker but real in the other, same direction) is expected, not
   disqualifying. Prefer the one with the strongest *combined* evidence
   (e.g. Fisher's method on the two nominal p-values, now legitimately
   applicable because direction agreement has already been confirmed)
   over the one that is merely the most uniform across cohorts, or the
   one with the single most dramatic cohort2 number. A smaller, very
   consistent effect is not automatically the safer answer if a
   materially stronger, still-replicating effect is available; nor is a
   gene automatically trustworthy just because it technically cleared the
   same-cohort bar.
5. That combined-evidence comparison has to use MODERATED per-cohort
   p-values, not raw ones (round 7). With only 6 samples per group (5
   residual degrees of freedom), a single gene's own within-group
   variance estimate is itself a noisy quantity, and a candidate whose
   apparent strength depends on an unusually (and unreliably) small
   realized variance in one cohort is not the same scientific claim as
   one whose strength holds up once that variance estimate is stabilized.
   Fit an empirical-Bayes prior (degrees of freedom and typical variance)
   from every candidate gene's own pooled within-group variance in that
   cohort -- the same moment-based estimator behind limma's eBayes, with
   no hand-picked prior weight, since limma-voom, edgeR's quasi-likelihood
   pipeline, and DESeq2 (each estimating their own moderation strength
   from the data) independently agree with the conclusion this produces
   on the locked dataset -- shrink each candidate's variance toward that
   prior, and recompute the combined-evidence ranking with the resulting
   moderated p-values. One designed gene in this dataset clears same-sign
   nominal significance, BH-FDR in both cohorts, and leave-one-out
   robustness more convincingly than the strongest generalizable
   candidate does on raw evidence, and still loses once variance
   moderation is applied.
6. Report the surviving gene's own result from whichever cohort gives it
   the stronger (smaller p-value) evidence, alongside both cohorts'
   individual fold-changes and a categorical description of how they
   relate to each other (consistent, or stronger in one cohort than the
   other), which cohort's own top hit was rejected and why, and the
   verified per-sample ID count.
7. A prior report exists on disk describing an earlier, much smaller,
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
8. For any candidate that clears step 7, check robustness: within the
   cohort where its evidence is weakest, drop each treated sample once and
   recompute the p-value against the full control group. If the worst
   (largest) p-value seen across those single-sample drops no longer
   clears the same significance bar the full-sample result did, the
   apparent replication is not real -- it is being carried by one or two
   samples, not the treated group as a whole.
9. For any candidate that clears step 8, re-rank using MODERATED combined
   evidence rather than raw: fit an empirical-Bayes prior (degrees of
   freedom and typical variance) from every candidate's own pooled
   within-group variance in each cohort, shrink each candidate's variance
   toward that prior, and recompute each cohort's p-value with the
   shrunk variance before combining across cohorts. A candidate whose raw
   strength depends on an unusually small realized variance in one cohort
   will lose ground to one whose strength is not built on that.
10. Among genes that replicate, are robust, and hold up under moderated
   combined evidence, prefer the strongest, unless the "stronger"
   candidate is actually the rejected artifact from step 7, the fragile
   candidate from step 8, or the variance-fragile candidate from step 9.
11. Write `result.json` with the surviving gene's symbol, its log2
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

- Naive pooled DE and a sign-blind combined-p meta-analysis both name the
  round-7 variance-fragile decoy outright -- not merely a contaminated
  estimate of the strongest generalizable candidate, a completely
  different gene.
- Trusting `cohort2` alone names a different gene entirely, with
  everything about it (fold-changes, heterogeneity label, rejected gene)
  wrong; its own cohort1 result is now nominally significant but
  wrong-signed.
- Running the complete, otherwise-correct replication procedure
  (alignment, per-cohort DE, same-sign nominal significance, leave-one-out
  robustness, prefer strongest RAW combined evidence) without also
  checking variance stability also lands on the variance-fragile decoy:
  it clears every one of those checks more convincingly, because its
  apparent strength happens to rest on an anomalously small realized
  variance in one cohort.
- Preferring the most cross-cohort-consistent gene over the one with
  materially stronger evidence (and never checking robustness or variance
  stability) names a gene that looks genuinely replicating on the
  surface -- its cohort1 and cohort2 fold-changes are far more consistent
  than the correct answer's -- but whose cohort2 significance turns out to
  depend on one or two individual samples rather than the treated group as
  a whole; wrong on every numeric field.
- Admitting a candidate as "replicated" on significance alone, without
  checking that both cohorts agree on effect direction, wrongly folds the
  technical artifact (and the real-but-sign-reversing heterogeneous gene)
  into the replicated pool; on this locked dataset the reported top gene
  is wrong too (it still lands on the variance-fragile decoy), and
  `rejected_competing_gene` is separately wrong because neither
  reversed-sign gene remains available to report there.
- Citing the real-but-sign-reversing heterogeneous gene as the rejected
  competitor (rather than the genuine technical artifact) is wrong even
  though excluding it from `top_gene` is correct -- its own single-cohort
  evidence is real but weaker than the actual technical artifact's.

All are visibly different from the locked reference values on at least one
checked field, and each fails for a different underlying reason. The
moderated-variance tiebreak was independently validated against real
limma-voom, edgeR's quasi-likelihood pipeline, and DESeq2 -- each
estimating its own moderation strength from the data, none given any
generator-known parameter -- and all three agree with the conclusion this
task's own (self-calibrating, no hand-picked prior weight) reimplementation
reaches. See task/README.md's "Round 7" section for the full comparison and
numbers, and "Round 6" for the leave-one-out robustness mechanism this
builds on.
