"""Reference solution: fix the compound-response RNA-seq pipeline and report
the top differentially expressed gene.

Root causes fixed here (found by running /workspace/data/pipeline/run_pipeline.py
and tracing the failure, not by inspecting this file -- an agent never sees it):

1. pipeline/io.py calls `np.float`, an alias NumPy removed in 1.24. This is
   what crashes the pipeline on the very first run. Trivial fix, but fixing
   only this and stopping here is a trap: the pipeline then runs to
   completion and prints a complete, plausible-looking differential
   expression table -- with the wrong top gene.

2. pipeline/qc.py's `filter_and_report_samples` sorts the metadata table by
   `sample_id` (a string column) for its printed report, and that
   lexicographically-sorted table is what the rest of the pipeline then
   treats as canonical. "sample_10" < "sample_2" as strings, so this is not
   the same order as the expression matrix's sample_id-keyed columns.

3. pipeline/align.py combines the (now-reordered) metadata with the
   expression matrix by resetting both to a plain positional index and
   concatenating side by side -- i.e., by row position, not by sample_id.
   Because the expression matrix's columns are in sequencer acquisition
   order (samples are randomized across lanes precisely so lane/batch is
   not confounded with condition) and never matched the metadata's original
   row order to begin with, this was never safe, even before (2) made it
   worse. The `assert combined.shape[0] == metadata.shape[0]` right after
   this only checks sample *count*, not sample *identity* -- it passes
   whether or not the rows actually correspond, so it gives no warning.

   align.py also branches on the installed pandas major version, a leftover
   from when the team supported pandas 1.x and 2.x side by side; neither
   branch is index-based, so "upgrading pandas" changes the positional
   scramble rather than fixing it.

The only safe fix is to join expression columns to metadata rows by
sample_id explicitly, never by position, and to keep sample_02 and sample_2
(a genuine second, later-arriving replicate vs. the original -- not a typo
or a duplicate) as the two distinct samples they are: a normalize-and-match
scheme that treats them as the same ID drops one sample or silently
mislabels the other's condition, either of which changes the answer.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace/data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))

sys.path.insert(0, str(DATA_DIR))
from pipeline import stats as pipeline_stats  # noqa: E402


def main() -> None:
    expr = pd.read_csv(DATA_DIR / "expression_matrix.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "sample_metadata.csv")

    metadata = metadata[metadata["qc_pass"]].reset_index(drop=True)

    # Join by sample_id explicitly -- never by row/column position. This is
    # the only step that differs from the shipped pipeline's align.py.
    ordered_ids = metadata["sample_id"].tolist()
    missing = [sid for sid in ordered_ids if sid not in expr.columns]
    if missing:
        raise ValueError(f"metadata sample_id(s) with no matching expression column: {missing}")
    counts_ordered = expr[ordered_ids]

    # Explicit, per-sample identity check -- the diagnostic the shipped
    # pipeline's shape-only assertion never actually performed.
    verified_matching_sample_ids = sum(
        1
        for sid, col in zip(ordered_ids, counts_ordered.columns)
        if sid == col and sid in set(metadata["sample_id"])
    )

    condition = pd.Series(metadata["condition"].to_numpy(), index=ordered_ids)

    log2cpm = pipeline_stats.compute_log2_cpm(counts_ordered)
    de_table = pipeline_stats.differential_expression(log2cpm, condition)
    de_table = de_table.sort_values("adjusted_p_value")

    top_gene = de_table.index[0]
    top_row = de_table.iloc[0]

    result = {
        "top_gene": str(top_gene),
        "log2_fold_change": round(float(top_row["log2_fold_change"]), 4),
        "adjusted_p_value": float(top_row["adjusted_p_value"]),
        "verified_matching_sample_ids": int(verified_matching_sample_ids),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
