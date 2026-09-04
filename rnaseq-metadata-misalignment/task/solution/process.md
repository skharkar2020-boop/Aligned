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
  on effect *direction*) and **checking nominal significance in both
  cohorts without checking sign agreement** are both real methodological
  gaps -- a sound analysis still has to check direction explicitly, not
  just significance -- but as of round 9 (module layer + composition-aware
  normalization), neither one actually diverges from the correct answer on
  the *locked* dataset: the confounded gene's own cohort1 result (p=0.0505)
  no longer quite clears nominal significance even without a sign check,
  so it is not available for either shortcut to mistakenly admit. This is
  a genuine, checked fact about the current calibration, not an oversight
  -- see task/README.md's "Round 9" section. Both gaps are still worth
  understanding and are still exercised by the module-layer competitor and
  the CPM-normalization scenarios below, just not by this specific
  mechanism any more.
- **Trusting the full, otherwise-correct per-cohort replication workflow**
  (ID-based alignment, per-cohort DE, same-sign nominal significance in
  both cohorts, leave-one-out robustness, preferring the strongest RAW
  combined evidence among survivors) **without also checking variance
  stability**: a gene exists in this dataset that is mechanically an
  entirely ordinary hit -- the same generative process as the strongest
  candidate (same baseline expression level, same true underlying
  variability), differing only in which natural random draw it happened
  to land on -- and, on the pre-round-9 dataset, cleared every one of
  those checks more convincingly on raw evidence than the correct answer
  did, because its own within-group sample variance in one cohort
  happened, purely by chance, to come out anomalously small (round 7/7b).
  As of round 9, this specific candidate no longer beats the correct
  answer on *raw* combined evidence either -- the module layer's
  composition effects moved the surrounding numbers enough that ranking by
  raw (non-moderated) combined evidence among gate-survivors also now
  recovers the correct answer directly (checked on the locked dataset; see
  task/README.md's "Round 9" section). Moderated variance is still the
  scientifically correct choice, and still measurably reduces this
  candidate's apparent advantage when it is applied (its raw combined
  evidence gets *weaker*, not stronger, once shrunk toward the panel's
  typical variance) -- it is no longer the single mechanism that
  determines the winner between this candidate and the correct answer,
  now that other checks (module-scale composition awareness, the
  module-layer competitor's own comparison) also constrain the answer.
  This was an explicit, considered decision, not an unnoticed regression:
  forcing this specific candidate to keep beating the correct answer on
  raw evidence would mean re-tuning it against the round-9 backdrop
  purely to preserve an older failure mode, which is backwards -- the
  benchmark's difficulty is now expected to come from the combined chain
  of checks (alignment, cohort separation, composition-aware
  normalization, moderated inference, replication/robustness, optional
  module-specificity reasoning) rather than from any one candidate's
  pairwise ranking being individually load-bearing.
- **Running the complete, otherwise-correct replication-plus-robustness-
  plus-moderation procedure but leaving the supplied pipeline's own
  total-count CPM normalization in place** (rather than switching to a
  composition-aware method once the module-scale response is apparent):
  promotes a different gene, one of the module-layer's own real,
  replicated, robust background responders (see the module-layer
  competitor discussion below). Naming that competitor as suspicious in a
  written rationale is not sufficient by itself -- CPM's own reference
  point moves along with the module layer's broad response, which
  specifically dampens TRUE_GENE's own apparent evidence relative to
  everything else on the panel, so the underlying per-gene statistics
  still have to come from a normalization that is not itself biased by
  the very pattern being reasoned about.
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

One further gene deserves separate mention here rather than being folded
into the "wrong strategies" list above, because it is not a wrong-strategy
artifact at all: a real, module-scale background responder (see
`gene_modules.csv`) that is same-signed, nominally significant in both
cohorts, and robust to the same leave-one-out check as everything else --
sitting in a module where roughly a quarter of the other members
independently show the same kind of broad, non-specific activity, unlike
TRUE_GENE's own module, where it is essentially the only responder. It is
not eliminated by the same-sign, significance, or robustness checks; it
loses because TRUE_GENE's own combined evidence (moderated, cross-cohort)
is stronger, by a comfortable margin under median-of-ratios normalization
and confirmed independently under limma-voom/TMM and edgeR-QL/TMM (see
task/README.md's "Round 9" section) -- not by a package-specific numerical
coincidence. Its own module's high hit-rate is useful corroborating
context for why it is the less specific signal, not a required or scored
part of the resolution.

## The correct resolution

1. Run the same per-gene DE procedure (Welch's t-test, BH-FDR, on
   properly-normalized expression -- see step 1b) independently within
   `cohort1` and within `cohort2`. Never pool.

1b. A substantial, broadly-distributed share of the 300-gene panel carries
   a real, module-scale treatment response (see `gene_modules.csv`), not
   one isolated hit. Once that much of the panel is genuinely responding,
   the supplied pipeline's own normalization (`pipeline/stats.py`'s
   `compute_log2_cpm`, total-count CPM) is no longer a neutral choice: its
   own reference point (each sample's total read count) shifts along with
   the treatment effect, which biases every other gene's apparent
   fold-change in the same direction -- including TRUE_GENE's own. The
   reference solution uses median-of-ratios normalization (Anders & Huber
   2010, the same size-factor method DESeq2 uses internally) instead:
   each sample's size factor is the median, across genes with a positive
   count in every sample, of that gene's count divided by its geometric
   mean across samples. This was validated against real DESeq2's own
   `estimateSizeFactors` on the locked dataset (size factors agree to
   <0.3%) and the resulting candidate ranking cross-checked against real
   limma-voom and edgeR's quasi-likelihood pipeline, both TMM-normalized
   (see task/README.md's "Round 9" section) -- none of the three need to
   compute numerically identical size factors to agree on the winner.
   Median-of-ratios is not the only acceptable choice; some
   composition-aware normalization is necessary, plain total-count CPM
   is not sufficient, once the data itself shows this pattern.
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
   answer away from the strongest generalizable candidate: TRUE_GENE's
   cohort2 raw p=4.3e-3 clears 0.05 easily, but its cohort2-only
   BH-adjusted p=0.098 does not. Verified on the locked dataset, not
   asserted.) Checking significance without
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
   significance is *robust*, not just present. Within each cohort, drop
   each individual sample once -- control or treated, not only treated --
   refit the moderated-variance prior on that reduced sample set (see
   step 9's shrinkage procedure), and recompute the candidate's p-value;
   the worst (largest) p-value seen across those single-sample drops must
   still clear the same nominal significance bar used everywhere else.
   Both the symmetry (any sample, not only treated) and the refit (the
   shrinkage prior is re-estimated on the data actually being analyzed
   each time, not reused from the full-sample fit) matter: a robustness
   check that only ever drops treated samples, or that silently reuses a
   stale full-sample moderation estimate, is not actually testing what it
   claims to. A candidate whose apparent replication only holds because
   of one or two individual samples is not a real replicated effect,
   however clean and consistent its headline fold-changes look.
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
6b. Notice that a large, broadly-distributed share of the panel (see
   `gene_modules.csv`) shows real, nominally significant, same-signed
   activity in both cohorts, not just the one or two candidates being
   scrutinized -- a pattern total-count CPM does not handle neutrally,
   since its own reference point (each sample's total read count) shifts
   along with that broad response. Switch to a composition-aware
   normalization (e.g. median-of-ratios, DESeq2's own method) before
   trusting any candidate's fold-change or significance from here on.
7. For each cohort's own top candidates, check whether they hold up
   (nominal significance, same sign) in the *other* cohort. Reject
   whichever prominent candidate fails that check, and identify why its
   evidence is concentrated in one cohort (look for other genes that move
   together with it, confined to that same cohort -- a latent technical
   axis, not biology).
8. For any candidate that clears step 7, check robustness: within each
   cohort, drop each individual sample once -- control or treated -- refit
   the moderated-variance prior on the reduced sample set, and recompute
   the p-value. If the worst (largest) p-value seen across those
   single-sample drops no longer clears the same significance bar the
   full-sample result did, the apparent replication is not real -- it is
   being carried by one or two samples, not the treated group as a whole.
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

- Naive pooled DE names the round-7 variance-fragile decoy outright -- not
  merely a contaminated estimate of the strongest generalizable candidate,
  a completely different gene.
- Trusting `cohort2` alone names a different gene entirely (the technical
  artifact), with everything about it (fold-changes, heterogeneity label,
  rejected gene) wrong.
- Leaving the supplied pipeline's own total-count CPM normalization in
  place, rather than switching to a composition-aware method, while
  otherwise running the complete, correct procedure, names the round-9
  module-layer competitor -- a real, replicated, robust background
  responder whose own module shows unusually broad, non-specific activity;
  CPM's own reference point moves along with that broad module-scale
  response, which specifically dampens the correct candidate's own
  apparent evidence enough to flip the ranking.
- Preferring the most cross-cohort-consistent gene among gate-survivors
  over the one with materially stronger combined evidence (never checking
  moderated combined evidence at all) still names the round-6
  fragile-replication decoy: its cohort1 and cohort2 fold-changes are far
  more consistent with each other than the correct answer's own
  (deliberately asymmetric) pair, and it does clear symmetric leave-one-out
  robustness, but its combined evidence is measurably weaker than the
  correct answer's once properly moderated; consistency and robustness
  alone are not the same claim as "strongest supported."
- Citing the real-but-sign-reversing heterogeneous gene as the rejected
  competitor (rather than the genuine technical artifact) is wrong even
  though excluding it from `top_gene` is correct -- its own single-cohort
  evidence is real but weaker than the actual technical artifact's.

As of round 9 (module layer + composition-aware normalization), two
scenarios that used to independently diverge from the correct answer no
longer do on the locked dataset, and this is a deliberate, checked
acceptance rather than an oversight: a sign-blind combined-p meta-analysis,
and running the complete replication-plus-robustness procedure but ranking
survivors by RAW rather than moderated combined evidence. Both now recover
the correct top gene directly on this dataset -- the technical artifact's
own cohort1 result (p=0.0505) no longer quite clears nominal significance
even without a sign check, and the round-9 composition effects moved the
raw-combined-evidence ordering among gate-survivors enough that moderation
no longer needs to flip the winner between the correct answer and the
round-7 variance-fragile decoy, only to (still correctly) widen the margin
between them. Moderated, composition-aware, cross-cohort-replicated
reasoning remains the scientifically correct and fully-used procedure; it
is simply no longer the sole mechanism standing between the agent and the
right answer, now that the module layer and normalization choice
contribute their own, independent constraints. See task/README.md's
"Round 9" section for the full numeric comparison and the explicit
decision not to re-tune any candidate purely to keep an older
single-mechanism trap load-bearing.

All are visibly different from the locked reference values on at least one
checked field, and each fails for a different underlying reason. The
moderated-variance tiebreak, and the symmetric (any-sample, moderation-
refit) robustness check described in step 3/8 above, were both
independently validated against real limma-voom, edgeR's quasi-likelihood
pipeline, and DESeq2 -- each estimating its own moderation strength from
the data, none given any generator-known parameter -- and all three agree
with the conclusion this task's own (self-calibrating, no hand-picked
prior weight) reimplementation reaches. See task/README.md's "Round 8"
section for the full comparison and numbers behind the symmetric-LOO
robustness definition and the 0.90->0.95 calibration it required, "Round
7"/"Round 7b" for the moderated-variance tiebreak this builds on, and
"Round 6" for the original leave-one-out robustness concept.
