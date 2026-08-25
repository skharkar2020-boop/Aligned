"""Differential expression statistics: CPM normalization, Welch's t-test, BH-FDR."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compute_log2_cpm(counts_by_sample: pd.DataFrame) -> pd.DataFrame:
    """counts_by_sample: genes x samples raw counts -> genes x samples log2(CPM + 1)."""
    library_sizes = counts_by_sample.sum(axis=0)
    cpm = counts_by_sample.div(library_sizes, axis=1) * 1e6
    return np.log2(cpm + 1.0)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty(n)
    out[order] = adjusted
    return out


def differential_expression(
    log2cpm: pd.DataFrame,
    condition: pd.Series,
    group_a: str = "control",
    group_b: str = "treated",
) -> pd.DataFrame:
    """log2cpm: genes x samples. condition: sample -> group label, indexed like log2cpm's columns."""
    a_samples = condition[condition == group_a].index
    b_samples = condition[condition == group_b].index

    a = log2cpm[a_samples].to_numpy(dtype=float)
    b = log2cpm[b_samples].to_numpy(dtype=float)

    t_stat, p_value = scipy_stats.ttest_ind(b, a, axis=1, equal_var=False)
    log2_fold_change = b.mean(axis=1) - a.mean(axis=1)
    adjusted_p_value = benjamini_hochberg(p_value)

    return pd.DataFrame(
        {
            "gene": log2cpm.index,
            "log2_fold_change": log2_fold_change,
            "p_value": p_value,
            "adjusted_p_value": adjusted_p_value,
        }
    ).set_index("gene")
