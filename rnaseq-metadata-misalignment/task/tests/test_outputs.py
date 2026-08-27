"""Independent verifier for the compound-response RNA-seq realignment task.

Recomputes the differential-expression result from the verifier's own copy
of the raw data (tests/data/, byte-identical to environment/data/) using an
implementation written separately from anything under environment/data/
pipeline/, and checks the submitted result against that recomputation and
against internal consistency requirements that only a genuinely ID-based,
batch-aware analysis satisfies.

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

EXPECTED_KEYS = {"top_gene", "log2_fold_change", "adjusted_p_value", "verified_matching_sample_ids"}
TOTAL_SAMPLES = 13

# log2 fold-change is a plain difference of group means in log2(CPM+1) space,
# so it is essentially method-invariant (Welch vs. Student t, or a different
# multiple-testing correction, do not move it); every wrong scenario checked
# during authoring (three distinct misalignments plus the batch-blind
# inclusion of sample_02) moved it by >=0.15, so this stays generous to
# method choice while still separating a real fix from a wrong one.
LOG2FC_ABS_TOL = 0.15

# adjusted_p_value is sensitive to the exact test/correction choice, so we
# only require it land clearly in significant territory and be within a
# wide log-scale band of the reference value. The reference value itself is
# 5.7e-06; every wrong scenario checked during authoring (two
# pandas-version-dependent misalignments, two sample_02/sample_2
# ID-confusion mistakes, and naively including the batch-confounded
# sample_02) landed at adjusted p-value >= 1e-3, far outside both bounds
# below.
ADJ_P_MAX_FOR_SIGNIFICANT = 5e-4
ADJ_P_LOG10_TOL = 2.5


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


def _load_result() -> dict[str, object]:
    assert RESULT_PATH.exists(), f"missing output: {RESULT_PATH}"
    return json.loads(RESULT_PATH.read_text())


def _compute_reference() -> dict[str, object]:
    expr = pd.read_csv(DATA_DIR / "expression_matrix.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "sample_metadata.csv")
    metadata = metadata[metadata["qc_pass"]].reset_index(drop=True)

    # Batch-confound rule: a batch with fewer than 2 samples cannot be
    # distinguished from a real biological effect, so it is excluded from
    # the comparison (though every sample, including it, is still
    # ID-verifiable -- that count is checked separately, not derived here).
    batch_sizes = metadata.groupby("batch")["sample_id"].transform("count")
    comparison_ids = metadata.loc[batch_sizes >= 2, "sample_id"].tolist()

    counts = expr[comparison_ids]  # ID-based column selection, never positional
    condition = pd.Series(
        metadata.set_index("sample_id").loc[comparison_ids, "condition"].to_numpy(),
        index=comparison_ids,
    )

    log2cpm = _compute_log2_cpm(counts)
    a_ids = condition[condition == "control"].index
    b_ids = condition[condition == "treated"].index
    a = log2cpm[a_ids].to_numpy(dtype=float)
    b = log2cpm[b_ids].to_numpy(dtype=float)

    _, p_value = scipy_stats.ttest_ind(b, a, axis=1, equal_var=False)
    log2fc = b.mean(axis=1) - a.mean(axis=1)
    padj = _benjamini_hochberg(p_value)

    de = pd.DataFrame(
        {"gene": log2cpm.index, "log2_fold_change": log2fc, "adjusted_p_value": padj}
    ).set_index("gene")
    de = de.sort_values("adjusted_p_value")

    return {
        "top_gene": str(de.index[0]),
        "log2_fold_change": float(de.iloc[0]["log2_fold_change"]),
        "adjusted_p_value": float(de.iloc[0]["adjusted_p_value"]),
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


def test_all_thirteen_samples_were_id_verified(result: dict[str, object], reference: dict[str, object]) -> None:
    """Every sample_id in the metadata has a same-named column in the
    expression matrix, so a genuinely ID-based join verifies all 13,
    including sample_02 (which is still excluded from the statistical
    comparison below for an unrelated, batch-confound reason). Fewer than
    13 means an ID was dropped, deduplicated, or otherwise not actually
    checked before the analysis ran.
    """
    assert result["verified_matching_sample_ids"] == TOTAL_SAMPLES, (
        f"expected all {TOTAL_SAMPLES} samples explicitly ID-verified before analysis, "
        f"got {result['verified_matching_sample_ids']}"
    )


def test_top_gene_matches_independent_recomputation(result: dict[str, object], reference: dict[str, object]) -> None:
    assert result["top_gene"] == reference["top_gene"], (
        f"reported top gene {result['top_gene']!r} does not match the independent "
        f"ID-based, batch-aware recomputation ({reference['top_gene']!r}); this is the "
        f"expected symptom of either a sample/label pairing that is internally "
        f"consistent but wrong, or of running the comparison on all 13 samples "
        f"without accounting for the single-sample batch2"
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
    test_all_thirteen_samples_were_id_verified,
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
