"""Loading helpers for the compound-response RNA-seq pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_expression_matrix(path: Path) -> pd.DataFrame:
    """Load the raw counts matrix (genes as rows, sample IDs as columns)."""
    counts = pd.read_csv(path, index_col=0)
    # Make sure every column is a numeric dtype before CPM normalization.
    counts = counts.astype(np.float)
    return counts


def load_metadata(path: Path) -> pd.DataFrame:
    """Load the per-sample metadata table (sample_id, condition, batch, qc_pass)."""
    return pd.read_csv(path)
