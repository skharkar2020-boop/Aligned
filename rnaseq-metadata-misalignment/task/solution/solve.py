"""Reference solution: fix the compound-response RNA-seq pipeline, reconcile
two cohorts that disagree, and report the top differentially expressed gene.

Root causes fixed here (found by running /workspace/data/pipeline/run_pipeline.py
and tracing the failure, not by inspecting this file -- an agent never sees it):

1. pipeline/io.py calls `np.float`, an alias NumPy removed in 1.24. This is
   what crashes the pipeline on the very first run. Trivial fix, but fixing
   only this and stopping here is a trap: the pipeline then runs to
   completion and prints a complete, plausible-looking differential
   expression table -- with the wrong top gene.

2. pipeline/qc.py's `filter_and_report_samples` sorts the metadata table by
   `sample_id` (a string column) for its printout, and that sorted table is
   what the rest of the pipeline then treats as canonical. String sorting
   of `sample_1 .. sample_24` is not numeric order (`sample_10` sorts
   before `sample_2`), so this silently reorders the metadata relative to
   the expression matrix's own sample order.

3. pipeline/align.py combines the (now-reordered) metadata with the
   expression matrix by resetting both to a plain positional index and
   concatenating side by side -- i.e., by row position, not by sample_id.
   The expression matrix's columns are in sequencer acquisition order
   (samples are randomized across lanes so lane is not confounded with
   condition) and were never guaranteed to match the metadata's row order
   to begin with, so this was unsafe even before (2) made it worse.
   `align.py` also branches on the installed pandas major version, a
   leftover from when the pipeline supported pandas 1.x and 2.x side by
   side; neither branch is index-based, so upgrading the pandas pin
   changes which wrong permutation you get rather than fixing anything.

The only safe fix is to join expression columns to metadata rows by
sample_id explicitly, never by position.

Fixing the alignment is necessary but not sufficient. Even once every
sample is correctly ID-matched, the pipeline pools all 24 samples into one
comparison, ignoring which cohort each sample came from. The metadata's
`cohort` column separates two independent runs, `cohort1` and `cohort2`.
Analyzed separately, they disagree on the top gene: `cohort1` and
`cohort2` alone each produce a complete, internally consistent,
non-crashing differential-expression result -- and they point at two
different genes. Pooling all 24 samples together (the pipeline's default
behavior once alignment is fixed) does not resolve this; it happens to
still name the right gene here, but with a fold-change and p-value
contaminated by mixing in the confounded cohort, which is why comparing
against the locked reference matters even when the gene name alone looks
right.

The correct resolution requires recognizing which cohort's result is not
trustworthy and why, not just picking whichever number looks bigger.
`cohort2` is a later, independent confirmatory run; its control and
treated samples were processed at different times (a real,
identifiable processing-date/reagent-lot confound, confounded with
condition only within that cohort), and its own top gene under this
analysis does not hold up at all in `cohort1` (it is not even nominally
significant there). `cohort1`'s top gene, by contrast, shows up in both
cohorts -- weaker and short of significance in the noisier `cohort2`, but
present, unlike `cohort2`'s own top gene in `cohort1`. That asymmetry is
the evidence, not a coin flip: the gene whose effect only appears when a
specific confound is present is the artifact; the gene whose effect
persists to some degree even without that confound is the real one. The
correct final answer is `cohort1`'s own result.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace/data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))

sys.path.insert(0, str(DATA_DIR))
from pipeline import stats as pipeline_stats  # noqa: E402


def differential_expression_for_cohort(
    expr: pd.DataFrame, metadata: pd.DataFrame, cohort: str
) -> pd.DataFrame:
    """ID-based, single-cohort differential expression. Never mixes samples
    across cohorts and never joins by position.
    """
    ids = metadata.loc[metadata["cohort"] == cohort, "sample_id"].tolist()
    missing = [sid for sid in ids if sid not in expr.columns]
    if missing:
        raise ValueError(f"metadata sample_id(s) with no matching expression column: {missing}")
    counts = expr[ids]
    condition = pd.Series(
        metadata.set_index("sample_id").loc[ids, "condition"].to_numpy(), index=ids
    )
    log2cpm = pipeline_stats.compute_log2_cpm(counts)
    de_table = pipeline_stats.differential_expression(log2cpm, condition)
    return de_table.sort_values("adjusted_p_value")


def main() -> None:
    expr = pd.read_csv(DATA_DIR / "expression_matrix.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "sample_metadata.csv")
    metadata = metadata[metadata["qc_pass"]].reset_index(drop=True)

    all_ids = metadata["sample_id"].tolist()

    # Explicit, per-sample identity check -- the diagnostic the shipped
    # pipeline's shape-only assertion never actually performed. Every
    # sample is ID-verified regardless of which cohort it belongs to.
    verified_matching_sample_ids = sum(1 for sid in all_ids if sid in expr.columns)

    cohorts = sorted(metadata["cohort"].unique())
    de_by_cohort = {c: differential_expression_for_cohort(expr, metadata, c) for c in cohorts}

    # Reconciliation: for each cohort's own top gene, check whether it
    # shows any real, consistent signal in the other cohort too. A gene
    # whose apparent effect is confined to one cohort and absent (not even
    # a weak, consistent-direction signal) in the other is the one to
    # distrust; the cohort that produced it is the confounded one.
    def other_cohort_supports(candidate_gene: str, home_cohort: str) -> bool:
        for other in cohorts:
            if other == home_cohort:
                continue
            other_de = de_by_cohort[other]
            if candidate_gene not in other_de.index:
                continue
            rank = list(other_de.index).index(candidate_gene) + 1
            # "Some real signal" is a much lower bar than significance --
            # cohort2 is noisier, so cohort1's true effect is not expected
            # to reach formal significance there. A top-quartile rank with
            # a same-signed fold change is enough to count as support; the
            # confounded cohort's own top gene, by contrast, does not even
            # clear that bar in the other cohort (see task/README.md).
            same_direction = np.sign(other_de.loc[candidate_gene, "log2_fold_change"]) == np.sign(
                de_by_cohort[home_cohort].loc[candidate_gene, "log2_fold_change"]
            )
            if rank <= len(other_de) // 4 and same_direction:
                return True
        return False

    candidates = {c: de_by_cohort[c].index[0] for c in cohorts}
    supported = {c: other_cohort_supports(candidates[c], c) for c in cohorts}

    trustworthy = [c for c in cohorts if supported[c]]
    confounded = [c for c in cohorts if not supported[c]]
    if len(trustworthy) != 1 or len(confounded) != len(cohorts) - 1:
        raise RuntimeError(
            f"expected exactly one cohort's top gene to replicate; got supported={supported}"
        )
    trusted_cohort = trustworthy[0]
    confounded_cohort = confounded[0]

    top_de = de_by_cohort[trusted_cohort]
    top_gene = top_de.index[0]
    top_row = top_de.iloc[0]

    result = {
        "top_gene": str(top_gene),
        "log2_fold_change": round(float(top_row["log2_fold_change"]), 4),
        "adjusted_p_value": float(top_row["adjusted_p_value"]),
        "verified_matching_sample_ids": int(verified_matching_sample_ids),
        "confounded_cohort": str(confounded_cohort),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
