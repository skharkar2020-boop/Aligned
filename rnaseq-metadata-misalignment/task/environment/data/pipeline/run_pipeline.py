"""Compound-response differential expression pipeline.

Historically run after every RNA-seq batch to rank genes by their response
to compound treatment vs. vehicle control. Reads DATA_DIR, writes a ranked
differential-expression table to OUTPUT_DIR.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from pipeline import align, io, qc, stats


def run(data_dir: Path) -> pd.DataFrame:
    expr = io.load_expression_matrix(data_dir / "expression_matrix.csv")
    metadata_raw = io.load_metadata(data_dir / "sample_metadata.csv")
    metadata = qc.filter_and_report_samples(metadata_raw)

    combined = align.combine_expression_and_metadata(expr, metadata)

    # Sanity check before running the analysis.
    assert combined.shape[0] == metadata.shape[0], (
        f"row count mismatch: combined={combined.shape[0]} metadata={metadata.shape[0]}"
    )
    print(f"[QC] {combined.shape[0]} samples present in the combined analysis table")

    gene_cols = [c for c in combined.columns if c not in metadata_raw.columns]
    log2cpm = stats.compute_log2_cpm(combined[gene_cols].T)
    condition = combined["condition"]

    de_table = stats.differential_expression(log2cpm, condition)
    return de_table.sort_values("adjusted_p_value")


if __name__ == "__main__":
    data_dir = Path(os.environ.get("DATA_DIR", "/workspace/data"))
    out_dir = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run(data_dir)
    result.to_csv(out_dir / "differential_expression.csv")
    print(result.head(10))
