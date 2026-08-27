"""Independent verifier for the compound-response RNA-seq reconciliation task.

Recomputes the differential-expression result from the verifier's own copy
of the raw data (tests/data/, byte-identical to environment/data/) using an
implementation written separately from anything under environment/data/
pipeline/ or solution/solve.py, and checks the submitted result against
that recomputation and against internal consistency requirements that only
a genuinely ID-based, cross-cohort-reconciled analysis satisfies.

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
    "verified_matching_sample_ids",
    "confounded_cohort",
}
TOTAL_SAMPLES = 24
VALID_COHORTS = {"cohort1", "cohort2"}

# log2 fold-change is a plain difference of group means in log2(CPM+1)
# space within one cohort, so it is essentially method-invariant (Welch vs.
# Student t, or a different multiple-testing correction, do not move it).
# Every wrong scenario checked during authoring (misalignment under either
# pandas branch, naively pooling both cohorts, reporting the confounded
# cohort's own result) moved it by >=0.3, so this stays generous to method
# choice while still separating a real fix from a wrong one.
LOG2FC_ABS_TOL = 0.2

# adjusted_p_value is sensitive to the exact test/correction choice, so we
# only require it land clearly in significant territory and be within a
# wide log-scale band of the reference value (~3.6e-07). Every wrong
# scenario checked during authoring landed at adjusted p-value >= 1e-3 for
# whichever gene it reported as top, far outside both bounds below.
ADJ_P_MAX_FOR_SIGNIFICANT = 5e-4
ADJ_P_LOG10_TOL = 3.0

# A candidate gene "replicates" in the other cohort if it clears plain
# nominal significance there (not BH-adjusted -- cohort2 is deliberately
# noisier, so the true effect is not expected to survive multiple-testing
# correction there) with the same-signed effect. This is intentionally a
# much lower bar than significance in the home cohort; see task/README.md
# for the actual values from the locked dataset (true top gene: raw
# p~6.5e-04 in the other cohort, same sign; the confounded cohort's own top
# gene: raw p~0.13 in the other cohort, opposite sign).
NOMINAL_P_THRESHOLD = 0.05


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


def _load_result() -> dict[str, object]:
    assert RESULT_PATH.exists(), f"missing output: {RESULT_PATH}"
    return json.loads(RESULT_PATH.read_text())


def _compute_reference() -> dict[str, object]:
    expr = pd.read_csv(DATA_DIR / "expression_matrix.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "sample_metadata.csv")
    metadata = metadata[metadata["qc_pass"]].reset_index(drop=True)

    de_by_cohort: dict[str, pd.DataFrame] = {}
    for cohort in sorted(metadata["cohort"].unique()):
        ids = metadata.loc[metadata["cohort"] == cohort, "sample_id"].tolist()
        counts = expr[ids]  # ID-based column selection, never positional
        condition = pd.Series(
            metadata.set_index("sample_id").loc[ids, "condition"].to_numpy(), index=ids
        )
        de_by_cohort[cohort] = _differential_expression(counts, condition).sort_values("adjusted_p_value")

    # Which cohort's own top gene fails to show even a nominal, same-signed
    # effect in the other cohort? That cohort is the confounded one.
    replicates = {}
    for cohort, de in de_by_cohort.items():
        other_cohorts = [c for c in de_by_cohort if c != cohort]
        top_gene = de.index[0]
        home_sign = np.sign(de.loc[top_gene, "log2_fold_change"])
        ok = False
        for other in other_cohorts:
            other_de = de_by_cohort[other]
            if top_gene not in other_de.index:
                continue
            other_row = other_de.loc[top_gene]
            if other_row["p_value"] < NOMINAL_P_THRESHOLD and np.sign(other_row["log2_fold_change"]) == home_sign:
                ok = True
                break
        replicates[cohort] = ok

    trustworthy = [c for c, ok in replicates.items() if ok]
    confounded = [c for c, ok in replicates.items() if not ok]
    assert len(trustworthy) == 1 and len(confounded) == len(de_by_cohort) - 1, (
        f"expected exactly one cohort to replicate in the fixture data; got {replicates}"
    )
    trusted_cohort = trustworthy[0]
    confounded_cohort = confounded[0]

    top_de = de_by_cohort[trusted_cohort]
    top = top_de.iloc[0]

    return {
        "top_gene": str(top_de.index[0]),
        "log2_fold_change": float(top["log2_fold_change"]),
        "adjusted_p_value": float(top["adjusted_p_value"]),
        "confounded_cohort": confounded_cohort,
    }


def test_result_schema_and_finite_values(result: dict[str, object], reference: dict[str, object]) -> None:
    assert set(result) == EXPECTED_KEYS
    assert isinstance(result["top_gene"], str) and result["top_gene"]

    for key in ("log2_fold_change", "adjusted_p_value"):
        value = result[key]
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(float(value))

    assert 0.0 <= float(result["adjusted_p_value"]) <= 1.0 + 1e-9

    verified = result["verified_matching_sample_ids"]
    assert isinstance(verified, int) and not isinstance(verified, bool)
    assert 0 <= verified <= TOTAL_SAMPLES

    assert result["confounded_cohort"] in VALID_COHORTS, (
        f"confounded_cohort must be one of {sorted(VALID_COHORTS)}, got {result['confounded_cohort']!r}"
    )


def test_all_samples_were_id_verified(result: dict[str, object], reference: dict[str, object]) -> None:
    """Every sample_id in the metadata has a same-named column in the
    expression matrix, so a genuinely ID-based join verifies all 24.
    Fewer means an ID was dropped, deduplicated, or otherwise not actually
    checked before the analysis ran.
    """
    assert result["verified_matching_sample_ids"] == TOTAL_SAMPLES, (
        f"expected all {TOTAL_SAMPLES} samples explicitly ID-verified before analysis, "
        f"got {result['verified_matching_sample_ids']}"
    )


def test_confounded_cohort_matches_independent_recomputation(
    result: dict[str, object], reference: dict[str, object]
) -> None:
    assert result["confounded_cohort"] == reference["confounded_cohort"], (
        f"reported confounded_cohort {result['confounded_cohort']!r} does not match the "
        f"independent recomputation ({reference['confounded_cohort']!r}); this is the "
        f"expected symptom of not actually checking whether each cohort's top gene "
        f"replicates in the other"
    )


def test_top_gene_matches_independent_recomputation(result: dict[str, object], reference: dict[str, object]) -> None:
    assert result["top_gene"] == reference["top_gene"], (
        f"reported top gene {result['top_gene']!r} does not match the independent "
        f"ID-based, cross-cohort-reconciled recomputation ({reference['top_gene']!r}); this "
        f"is the expected symptom of a sample/label pairing that is internally consistent "
        f"but wrong, of pooling both cohorts instead of analyzing them separately, or of "
        f"trusting the confounded cohort's own result"
    )


def test_log2_fold_change_matches_independent_recomputation(
    result: dict[str, object], reference: dict[str, object]
) -> None:
    reported = float(result["log2_fold_change"])
    expected = reference["log2_fold_change"]
    assert abs(reported - expected) <= LOG2FC_ABS_TOL, (
        f"reported log2_fold_change={reported} for {result['top_gene']!r} is inconsistent "
        f"with the independently recomputed value {expected:.4f} (tolerance {LOG2FC_ABS_TOL})"
    )


def test_adjusted_p_value_is_significant_and_matches_recomputation(
    result: dict[str, object], reference: dict[str, object]
) -> None:
    reported = float(result["adjusted_p_value"])
    expected = reference["adjusted_p_value"]

    assert reported <= ADJ_P_MAX_FOR_SIGNIFICANT, (
        f"reported adjusted_p_value={reported} for {result['top_gene']!r} is not clearly "
        f"significant (must be <= {ADJ_P_MAX_FOR_SIGNIFICANT}); every wrong scenario "
        f"checked during authoring lands far above this bound"
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
    test_confounded_cohort_matches_independent_recomputation,
    test_top_gene_matches_independent_recomputation,
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
