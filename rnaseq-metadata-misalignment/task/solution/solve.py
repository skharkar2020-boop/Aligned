"""Reference solution: fix the compound-response RNA-seq pipeline, reconcile
two cohorts whose naive analyses disagree, and report a defensible top
differentially expressed gene.

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
`cohort` column separates two independent runs. Several naive strategies
each look reasonable and each land on a different, wrong-in-some-way
answer:

- Naive pooled DE (cohort-blind, the pipeline's default once alignment is
  fixed) mixes a strong, homogeneous cohort1 effect with a much weaker
  cohort2 effect for the true gene; it happens to still name the right
  gene, but its fold-change and p-value are contaminated by the pooling
  and do not match either cohort's own honest estimate.
- Trusting cohort2 alone (reasonable on its face: it is the later,
  "confirmatory" run) names a *different* gene entirely -- one with a huge
  effect confined to cohort2 and nothing in cohort1.
- A naive meta-analysis that combines each gene's two cohort-level
  p-values without first checking that the two cohorts agree on effect
  *direction* (a real, common mistake -- e.g. plugging both p-values into
  Fisher's method and taking whichever gene comes out smallest) lands on
  the same wrong gene as trusting cohort2 alone, because that gene's
  extremely small cohort2 p-value dominates the combination even though
  cohort1 shows no real effect (and, if anything, the opposite sign).

The only strategy that survives scrutiny is per-cohort independent
replication: run the same DE procedure separately within each cohort, and
require a candidate to be *nominally* significant (raw p < 0.05, not
BH-adjusted -- cohort2 is noisier, so a real effect need not survive
multiple-testing correction there) with the *same-signed* effect
independently in each cohort. A gene whose apparent effect is confined to
one cohort and absent (not even a weak, same-direction nominal signal) in
the other fails this check outright -- that is the signature of a
cohort-specific technical artifact, not biology.

More than one gene can pass that bar. When that happens, prefer the one
with the stronger combined evidence (Fisher's method on the two
independent nominal p-values, now legitimately combined because direction
has already been confirmed to agree) -- this is what separates a gene with
a strong, real effect in one cohort and a real-but-weaker echo in the
other from a gene with a smaller but very consistent effect in both. Real,
moderate heterogeneity in effect size between cohorts is expected and is
not by itself a reason to distrust a gene; only the wrong-cohort dominance
and the sign-blind failure mode above are.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace/data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))

sys.path.insert(0, str(DATA_DIR))
from pipeline import stats as pipeline_stats  # noqa: E402

NOMINAL_P_THRESHOLD = 0.05


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


def fisher_combined_p(p1: float, p2: float) -> float:
    statistic = -2.0 * (np.log(max(p1, 1e-300)) + np.log(max(p2, 1e-300)))
    return float(scipy_stats.chi2.sf(statistic, df=4))


def classify_heterogeneity(log2fc_c1: float, log2fc_c2: float) -> str:
    if np.sign(log2fc_c1) != np.sign(log2fc_c2) and log2fc_c1 != 0.0 and log2fc_c2 != 0.0:
        return "opposite_direction_between_cohorts"
    a, b = abs(log2fc_c1), abs(log2fc_c2)
    larger, smaller = max(a, b), min(a, b)
    if larger == 0.0:
        return "consistent_both_cohorts"
    ratio = smaller / larger
    if ratio >= 0.4:
        return "consistent_both_cohorts"
    return "stronger_in_cohort1_weaker_in_cohort2" if a >= b else "stronger_in_cohort2_weaker_in_cohort1"


def main() -> None:
    expr = pd.read_csv(DATA_DIR / "expression_matrix.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "sample_metadata.csv")
    metadata = metadata[metadata["qc_pass"]].reset_index(drop=True)

    all_ids = metadata["sample_id"].tolist()

    # Explicit, per-sample identity check -- the diagnostic the shipped
    # pipeline's shape-only assertion never actually performed.
    verified_matching_sample_ids = all(sid in expr.columns for sid in all_ids)

    cohorts = sorted(metadata["cohort"].unique())
    if len(cohorts) != 2:
        raise RuntimeError(f"expected exactly two cohorts, got {cohorts}")
    cohort1, cohort2 = cohorts
    de_by_cohort = {c: differential_expression_for_cohort(expr, metadata, c) for c in cohorts}
    de1, de2 = de_by_cohort[cohort1], de_by_cohort[cohort2]

    # Candidate pool: anything that ranks near the top of either cohort's
    # own analysis (never trust one cohort's ranking alone).
    candidates = sorted(set(de1.index[:10]) | set(de2.index[:10]))

    replicated = []
    for gene in candidates:
        r1, r2 = de1.loc[gene], de2.loc[gene]
        same_sign = np.sign(r1["log2_fold_change"]) == np.sign(r2["log2_fold_change"])
        both_nominal = r1["p_value"] < NOMINAL_P_THRESHOLD and r2["p_value"] < NOMINAL_P_THRESHOLD
        if same_sign and both_nominal:
            combined_p = fisher_combined_p(float(r1["p_value"]), float(r2["p_value"]))
            replicated.append((gene, combined_p, r1, r2))

    if not replicated:
        raise RuntimeError("no gene passed the independent-replication check in both cohorts")
    replicated.sort(key=lambda item: item[1])
    top_gene, _, r1, r2 = replicated[0]

    # The rejected competitor: among genes that fail replication, the one
    # with the single strongest one-cohort result -- the "extremely strong
    # signal driven mainly by one cohort" story that a naive cohort-alone
    # or sign-blind combined analysis would have reported instead.
    rejected_competing_gene = None
    best_single_cohort_p = None
    for gene in candidates:
        if gene == top_gene:
            continue
        r1g, r2g = de1.loc[gene], de2.loc[gene]
        same_sign = np.sign(r1g["log2_fold_change"]) == np.sign(r2g["log2_fold_change"])
        both_nominal = r1g["p_value"] < NOMINAL_P_THRESHOLD and r2g["p_value"] < NOMINAL_P_THRESHOLD
        if same_sign and both_nominal:
            continue
        candidate_p = min(float(r1g["p_value"]), float(r2g["p_value"]))
        if best_single_cohort_p is None or candidate_p < best_single_cohort_p:
            best_single_cohort_p = candidate_p
            rejected_competing_gene = str(gene)

    c1_fc = float(r1["log2_fold_change"])
    c2_fc = float(r2["log2_fold_change"])
    home = r1 if float(r1["p_value"]) <= float(r2["p_value"]) else r2
    heterogeneity_assessment = classify_heterogeneity(c1_fc, c2_fc)

    rationale = (
        f"{top_gene} is the only strong candidate that shows a nominally "
        f"significant, same-signed effect independently in both {cohort1} "
        f"(log2FC={c1_fc:.3f}) and {cohort2} (log2FC={c2_fc:.3f}); the "
        f"heterogeneity between cohorts is real but moderate "
        f"({heterogeneity_assessment}), not a sign flip or a null cohort. "
        f"{rejected_competing_gene} was rejected: its strongest evidence is "
        f"confined to a single cohort and does not replicate, same-signed, "
        f"in the other -- naive pooled DE and a sign-blind combined-p "
        f"meta-analysis are both misled by that gene's outsized single-"
        f"cohort effect, and cohort2-trusting-alone reports it outright as "
        f"the top hit. A prior pilot report on file names {top_gene} as "
        f"well, but with different (smaller-n, single-cohort) numbers that "
        f"do not match this independently recomputed result and are not "
        f"sufficient evidence on their own."
    )

    result = {
        "top_gene": str(top_gene),
        "log2_fold_change": round(float(home["log2_fold_change"]), 4),
        "adjusted_p_value": float(home["adjusted_p_value"]),
        "analysis_strategy": "per_cohort_independent_replication",
        "cohort1_log2_fold_change": round(c1_fc, 4),
        "cohort2_log2_fold_change": round(c2_fc, 4),
        "heterogeneity_assessment": heterogeneity_assessment,
        "verified_matching_sample_ids": bool(verified_matching_sample_ids),
        "rejected_competing_gene": rejected_competing_gene,
        "rationale": rationale,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
