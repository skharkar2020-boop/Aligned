"""Sample-level QC gating and reporting for the compound-response pipeline."""

from __future__ import annotations

import pandas as pd


def filter_and_report_samples(metadata: pd.DataFrame) -> pd.DataFrame:
    """Drop any sample that failed upstream QC and print a per-sample report.

    Returns the cleaned metadata table that the rest of the run treats as
    the canonical sample list.
    """
    passed = metadata[metadata["qc_pass"]].copy()
    dropped = len(metadata) - len(passed)
    if dropped:
        print(f"[QC] dropped {dropped} sample(s) that failed QC")

    # Sort so the printed report reads in a stable, human-friendly order.
    report_order = passed.sort_values("sample_id")
    print(f"[QC] {len(report_order)} samples passed QC:")
    for _, row in report_order.iterrows():
        print(f"  {row['sample_id']}: {row['condition']} ({row['batch']})")

    return report_order
