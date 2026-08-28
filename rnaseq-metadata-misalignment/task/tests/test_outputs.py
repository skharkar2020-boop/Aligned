"""Independent verifier for the compound-response RNA-seq reconciliation task.

Recomputes the differential-expression result from the verifier's own copy
of the raw data (tests/data/, byte-identical to environment/data/) using an
implementation written separately from anything under environment/data/
pipeline/ or solution/solve.py, and checks the submitted result against
that recomputation and against internal consistency requirements that only
a genuinely ID-based, per-cohort-reconciled analysis satisfies.

Plain script, no test framework: every check the agent's output must pass
is stated here directly, with no dependency beyond the numeric/data
libraries the analysis itself legitimately needs (numpy, pandas, scipy).
Run as `python3 test_outputs.py`; prints PASS/FAIL per check and exits 0
only if every check passed.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))
TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
RESULT_PATH = OUTPUT_DIR / "result.json"
DATA_DIR = TESTS_DIR / "data"

EXPECTED_KEYS = {
    "top_gene",
    "log2_fold_change",
    "adjusted_p_value",
    "analysis_strategy",
    "cohort1_log2_fold_change",
    "cohort2_log2_fold_change",
    "heterogeneity_assessment",
    "verified_matching_sample_ids",
    "rejected_competing_gene",
    "rationale",
}
TOTAL_SAMPLES = 24

# analysis_strategy names the agent may legitimately report having used;
# only one of them is actually correct here (see
# test_analysis_strategy_matches_recomputation), but any other value is a
# schema violation, not just a wrong answer.
VALID_ANALYSIS_STRATEGIES = {
    "pooled",
    "cohort1_only",
    "cohort2_only",
    "fixed_effect_meta_analysis",
    "per_cohort_independent_replication",
}
VALID_HETEROGENEITY_LABELS = {
    "consistent_both_cohorts",
    "stronger_in_cohort1_weaker_in_cohort2",
    "stronger_in_cohort2_weaker_in_cohort1",
    "opposite_direction_between_cohorts",
}

# log2 fold-change is a plain difference of group means in log2(CPM+1)
# space within one cohort, so it is essentially method-invariant (Welch vs.
# Student t, or a different multiple-testing correction, do not move it).
# Naive pooling of both cohorts moves the true top gene's own fold-change
# by >0.8 in the locked dataset (2.13 home-cohort vs. 1.32 pooled); every
# other wrong scenario checked during authoring names a different gene
# entirely. This tolerance stays generous to method choice while still
# separating a real fix from a contaminated pooled estimate.
LOG2FC_ABS_TOL = 0.2

# adjusted_p_value is sensitive to the exact test/correction choice, so we
# only require it land clearly in significant territory and be within a
# wide log-scale band of the reference value (~5.7e-05 in the locked
# dataset). Naive pooling lands at adjusted p~1.1e-02 for the same gene,
# comfortably outside ADJ_P_MAX_FOR_SIGNIFICANT.
ADJ_P_MAX_FOR_SIGNIFICANT = 5e-4
ADJ_P_LOG10_TOL = 3.0

# A candidate gene "replicates" in the other cohort if it clears plain
# nominal significance there (not BH-adjusted -- cohort2 is deliberately
# noisier) with the same-signed effect. See task/README.md for the actual
# locked-dataset values: the true top gene is nominally significant,
# same-signed, in both cohorts; the rejected competing gene is not even
# nominally significant (and is wrong-signed) in cohort1.
NOMINAL_P_THRESHOLD = 0.05

RATIONALE_MIN_LEN = 40


def _compute_log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    library_sizes = counts.sum(axis=0)
    cpm = counts.div(library_sizes, axis=1) * 1e6
    return np.log2(cpm + 1.0)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty(n)
    out[order] = adjusted
    return out


def _differential_expression(counts: pd.DataFrame, condition: pd.Series) -> pd.DataFrame:
    log2cpm = _compute_log2_cpm(counts)
    a = log2cpm[condition[condition == "control"].index].to_numpy(dtype=float)
    b = log2cpm[condition[condition == "treated"].index].to_numpy(dtype=float)
    t_stat, p_value = scipy_stats.ttest_ind(b, a, axis=1, equal_var=False)
    log2_fold_change = b.mean(axis=1) - a.mean(axis=1)
    adjusted_p_value = _benjamini_hochberg(p_value)
    return pd.DataFrame(
        {
            "gene": log2cpm.index,
            "log2_fold_change": log2_fold_change,
            "p_value": p_value,
            "adjusted_p_value": adjusted_p_value,
        }
    ).set_index("gene")


def _differential_expression_for_cohort(
    expr: pd.DataFrame, metadata: pd.DataFrame, cohort: str
) -> pd.DataFrame:
    ids = metadata.loc[metadata["cohort"] == cohort, "sample_id"].tolist()
    counts = expr[ids]  # ID-based column selection, never positional
    condition = pd.Series(
        metadata.set_index("sample_id").loc[ids, "condition"].to_numpy(), index=ids
    )
    return _differential_expression(counts, condition).sort_values("adjusted_p_value")


def _fisher_combined_p(p1: float, p2: float) -> float:
    statistic = -2.0 * (np.log(max(p1, 1e-300)) + np.log(max(p2, 1e-300)))
    return float(scipy_stats.chi2.sf(statistic, df=4))


def _classify_heterogeneity(log2fc_c1: float, log2fc_c2: float) -> str:
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


def _load_result() -> dict[str, object]:
    assert RESULT_PATH.exists(), f"missing output: {RESULT_PATH}"
    return json.loads(RESULT_PATH.read_text())


def _compute_reference() -> dict[str, object]:
    expr = pd.read_csv(DATA_DIR / "expression_matrix.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "sample_metadata.csv")
    metadata = metadata[metadata["qc_pass"]].reset_index(drop=True)

    cohorts = sorted(metadata["cohort"].unique())
    assert cohorts == ["cohort1", "cohort2"], f"unexpected cohorts in fixture data: {cohorts}"

    de1 = _differential_expression_for_cohort(expr, metadata, "cohort1")
    de2 = _differential_expression_for_cohort(expr, metadata, "cohort2")

    candidates = sorted(set(de1.index[:10]) | set(de2.index[:10]))

    replicated = []
    for gene in candidates:
        r1, r2 = de1.loc[gene], de2.loc[gene]
        same_sign = np.sign(r1["log2_fold_change"]) == np.sign(r2["log2_fold_change"])
        both_nominal = r1["p_value"] < NOMINAL_P_THRESHOLD and r2["p_value"] < NOMINAL_P_THRESHOLD
        if same_sign and both_nominal:
            combined_p = _fisher_combined_p(float(r1["p_value"]), float(r2["p_value"]))
            replicated.append((gene, combined_p, r1, r2))

    assert replicated, "expected at least one gene to pass the independent-replication gate in the fixture data"
    replicated.sort(key=lambda item: item[1])
    top_gene, _, r1, r2 = replicated[0]

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

    return {
        "top_gene": str(top_gene),
        "log2_fold_change": float(home["log2_fold_change"]),
        "adjusted_p_value": float(home["adjusted_p_value"]),
        "analysis_strategy": "per_cohort_independent_replication",
        "cohort1_log2_fold_change": c1_fc,
        "cohort2_log2_fold_change": c2_fc,
        "heterogeneity_assessment": _classify_heterogeneity(c1_fc, c2_fc),
        "rejected_competing_gene": rejected_competing_gene,
    }


def test_result_schema_and_finite_values(result: dict[str, object], reference: dict[str, object]) -> None:
    assert set(result) == EXPECTED_KEYS
    assert isinstance(result["top_gene"], str) and result["top_gene"]
    assert isinstance(result["rejected_competing_gene"], str) and result["rejected_competing_gene"]

    for key in ("log2_fold_change", "adjusted_p_value", "cohort1_log2_fold_change", "cohort2_log2_fold_change"):
        value = result[key]
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(float(value))

    assert 0.0 <= float(result["adjusted_p_value"]) <= 1.0 + 1e-9

    verified = result["verified_matching_sample_ids"]
    assert isinstance(verified, bool), f"verified_matching_sample_ids must be a boolean, got {type(verified).__name__}"

    assert result["analysis_strategy"] in VALID_ANALYSIS_STRATEGIES, (
        f"analysis_strategy must be one of {sorted(VALID_ANALYSIS_STRATEGIES)}, got {result['analysis_strategy']!r}"
    )
    assert result["heterogeneity_assessment"] in VALID_HETEROGENEITY_LABELS, (
        f"heterogeneity_assessment must be one of {sorted(VALID_HETEROGENEITY_LABELS)}, "
        f"got {result['heterogeneity_assessment']!r}"
    )

    assert isinstance(result["rationale"], str) and len(result["rationale"].strip()) >= RATIONALE_MIN_LEN, (
        f"rationale must be a non-trivial explanation (>= {RATIONALE_MIN_LEN} chars); this is a presence check "
        f"only, never graded on content"
    )


def test_all_samples_were_id_verified(result: dict[str, object], reference: dict[str, object]) -> None:
    assert result["verified_matching_sample_ids"] is True, (
        f"expected all {TOTAL_SAMPLES} samples explicitly ID-verified before analysis, "
        f"got verified_matching_sample_ids={result['verified_matching_sample_ids']!r}"
    )


def test_top_gene_matches_independent_recomputation(result: dict[str, object], reference: dict[str, object]) -> None:
    assert result["top_gene"] == reference["top_gene"], (
        f"reported top gene {result['top_gene']!r} does not match the independent "
        f"ID-based, per-cohort-reconciled recomputation ({reference['top_gene']!r}); this is the "
        f"expected symptom of naive pooling, trusting a single cohort, or a sign-blind "
        f"combined-p meta-analysis"
    )


def test_analysis_strategy_matches_recomputation(result: dict[str, object], reference: dict[str, object]) -> None:
    assert result["analysis_strategy"] == reference["analysis_strategy"], (
        f"reported analysis_strategy {result['analysis_strategy']!r} does not match the strategy "
        f"that actually survives scrutiny on this data ({reference['analysis_strategy']!r})"
    )


def test_cohort_log2_fold_changes_match_recomputation(
    result: dict[str, object], reference: dict[str, object]
) -> None:
    for key in ("cohort1_log2_fold_change", "cohort2_log2_fold_change"):
        reported = float(result[key])
        expected = float(reference[key])
        assert abs(reported - expected) <= LOG2FC_ABS_TOL, (
            f"reported {key}={reported} for {result['top_gene']!r} is inconsistent with the "
            f"independently recomputed value {expected:.4f} (tolerance {LOG2FC_ABS_TOL})"
        )


def test_heterogeneity_assessment_matches_recomputation(
    result: dict[str, object], reference: dict[str, object]
) -> None:
    assert result["heterogeneity_assessment"] == reference["heterogeneity_assessment"], (
        f"reported heterogeneity_assessment {result['heterogeneity_assessment']!r} does not match "
        f"the independently recomputed classification ({reference['heterogeneity_assessment']!r}) "
        f"of this gene's own pair of cohort effect sizes"
    )


def test_rejected_competing_gene_matches_recomputation(
    result: dict[str, object], reference: dict[str, object]
) -> None:
    assert result["rejected_competing_gene"] == reference["rejected_competing_gene"], (
        f"reported rejected_competing_gene {result['rejected_competing_gene']!r} does not match the "
        f"independently recomputed answer ({reference['rejected_competing_gene']!r}); this is the "
        f"expected symptom of not identifying which competing hit fails the cross-cohort "
        f"replication check"
    )


def test_log2_fold_change_matches_independent_recomputation(
    result: dict[str, object], reference: dict[str, object]
) -> None:
    reported = float(result["log2_fold_change"])
    expected = reference["log2_fold_change"]
    assert abs(reported - expected) <= LOG2FC_ABS_TOL, (
        f"reported log2_fold_change={reported} for {result['top_gene']!r} is inconsistent "
        f"with the independently recomputed home-cohort value {expected:.4f} (tolerance {LOG2FC_ABS_TOL})"
    )


def test_adjusted_p_value_is_significant_and_matches_recomputation(
    result: dict[str, object], reference: dict[str, object]
) -> None:
    reported = float(result["adjusted_p_value"])
    expected = reference["adjusted_p_value"]

    assert reported <= ADJ_P_MAX_FOR_SIGNIFICANT, (
        f"reported adjusted_p_value={reported} for {result['top_gene']!r} is not clearly "
        f"significant (must be <= {ADJ_P_MAX_FOR_SIGNIFICANT}); naive pooling of both cohorts "
        f"lands far above this bound for the same gene"
    )

    reported_floor = max(reported, 1e-300)
    expected_floor = max(expected, 1e-300)
    log_gap = abs(math.log10(reported_floor) - math.log10(expected_floor))
    assert log_gap <= ADJ_P_LOG10_TOL, (
        f"reported adjusted_p_value={reported} for {result['top_gene']!r} is too far "
        f"(log10 gap={log_gap:.2f}) from the independently recomputed value {expected:.3e} "
        f"(tolerance {ADJ_P_LOG10_TOL} log10 units)"
    )


CHECKS = [
    test_result_schema_and_finite_values,
    test_all_samples_were_id_verified,
    test_top_gene_matches_independent_recomputation,
    test_analysis_strategy_matches_recomputation,
    test_cohort_log2_fold_changes_match_recomputation,
    test_heterogeneity_assessment_matches_recomputation,
    test_rejected_competing_gene_matches_recomputation,
    test_log2_fold_change_matches_independent_recomputation,
    test_adjusted_p_value_is_significant_and_matches_recomputation,
]


def main() -> int:
    try:
        result = _load_result()
    except Exception as exc:  # noqa: BLE001 - report and fail, don't crash uninformatively
        print(f"FAIL setup: could not load result: {exc}")
        return 1

    try:
        reference = _compute_reference()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL setup: could not compute independent reference: {exc}")
        traceback.print_exc()
        return 1

    failures = 0
    for check in CHECKS:
        name = check.__name__
        try:
            check(result, reference)
        except AssertionError as exc:
            print(f"FAIL {name}: {exc}")
            failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: unexpected error: {exc}")
            traceback.print_exc()
            failures += 1
        else:
            print(f"PASS {name}")

    total = len(CHECKS)
    print(f"\n{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
