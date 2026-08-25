"""Synthetic bulk RNA-seq dataset generator for the compound-response DE task.

Ground truth (never exposed via file/column names in the public data):
  - 13 samples: 7 vehicle-control, 6 compound-treated.
  - Two of the 13 are a genuine near-duplicate ID pair, `sample_2` and
    `sample_02` -- distinct biological replicates from two different
    collection batches (batch1 used unpadded numeric IDs; batch2, added
    later, used zero-padded IDs). Both are real, independent samples with
    their own counts; neither is a copy of the other.
  - The expression matrix's sample (column) order is the sequencer
    acquisition order, which is intentionally NOT the metadata table's row
    order: real cores randomize sample-to-lane assignment specifically to
    avoid confounding batch/lane with condition, so a pipeline that assumes
    matching row order between the two tables is wrong from the start, even
    before any downstream re-sort makes it worse.
  - Three genes carry a real condition effect (log2 fold-change, treated
    over control): TRUE_TOP is the strongest and lowest-noise; DECOY_A and
    DECOY_B are close enough in effect size that a wrong (but internally
    consistent, non-crashing) sample/label pairing can plausibly promote
    either of them to "top hit" instead. All other genes carry no true
    condition effect.

Counts are simulated gene-by-gene as Poisson-lognormal: a per-gene, per
condition mean count scaled by a per-sample library-size factor, with
multiplicative log-normal biological noise on top (the standard
overdispersion picture for bulk RNA-seq, without requiring a dedicated
negative-binomial DE package).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "task" / "environment" / "data"))

from pipeline import stats as pipeline_stats  # noqa: E402  (path set above)


def stable_seed(*parts) -> int:
    """Deterministic replacement for hash((...)) % 2**31 (PYTHONHASHSEED-proof)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(key).hexdigest(), 16) % (2**31)


N_GENES = 300
GENE_PREFIXES = [
    "ZNF", "TMEM", "ABCB", "SLC", "CYP", "MAP", "RPL", "RPS", "GPR", "TNFR",
    "CDK", "MMP", "COL", "ACTB", "MYO", "HSP", "CAT", "SOD", "NOS", "PTP",
    "PDE", "ADRB", "HTR", "DRD", "GRIN", "CACNA", "SCN", "KCNH", "ATP",
    "NDUF", "COX", "UQCR", "TUBB", "RAB", "ARF", "SEC", "VPS", "USP",
    "RNF", "DDX", "EIF", "PSMB", "PSMD", "HLA", "IFIT", "OAS", "IRF",
    "STAT", "SMAD", "FOX", "SOX", "KLF", "GATA", "NFAT", "BCL", "CASP",
]


def make_gene_symbols(n: int, rng: np.random.Generator) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    prefix_i = 0
    while len(symbols) < n:
        prefix = GENE_PREFIXES[prefix_i % len(GENE_PREFIXES)]
        suffix = int(rng.integers(1, 999))
        symbol = f"{prefix}{suffix}"
        prefix_i += 1
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def build_sample_roster() -> pd.DataFrame:
    """Metadata row order == LIMS registration order (public/sample_metadata.csv)."""
    rows = []
    for i in range(1, 13):
        condition = "control" if i <= 6 else "treated"
        rows.append({"sample_id": f"sample_{i}", "condition": condition, "batch": "batch1", "qc_pass": True})
    # Extra batch-2 replicate, registered later (to shore up the treated arm),
    # zero-padded ID convention. Genuinely distinct sample -- NOT a duplicate
    # of sample_2 -- and deliberately in the OPPOSITE condition from sample_2,
    # so confusing the two during alignment actually corrupts the analysis
    # rather than being a harmless mix-up.
    rows.append({"sample_id": "sample_02", "condition": "treated", "batch": "batch2", "qc_pass": True})
    return pd.DataFrame(rows)


def acquisition_order(sample_ids: list[str], rng: np.random.Generator) -> list[str]:
    order = list(sample_ids)
    rng.shuffle(order)
    return order


def simulate_counts(metadata: pd.DataFrame, gene_symbols: list[str], rng: np.random.Generator):
    sample_ids = metadata["sample_id"].tolist()
    condition_by_id = dict(zip(metadata["sample_id"], metadata["condition"]))

    n_samples = len(sample_ids)
    # Small per-sample library-size variation (deterministic per sample_id).
    size_factor = {
        sid: float(np.exp(np.random.default_rng(stable_seed("size_factor", sid)).normal(0, 0.08)))
        for sid in sample_ids
    }

    designed = {
        "TRUE_TOP": {"log2fc": 2.6, "sigma": 0.15, "baseline": 520.0},
        "DECOY_A": {"log2fc": 2.3, "sigma": 0.30, "baseline": 300.0},
        "DECOY_B": {"log2fc": 2.2, "sigma": 0.30, "baseline": 300.0},
    }
    # Fixed, non-obvious positions in the gene panel (not gene[0], gene[1], gene[2]).
    designed_positions = {"TRUE_TOP": 137, "DECOY_A": 42, "DECOY_B": 216}
    gene_symbols = list(gene_symbols)
    for name, pos in designed_positions.items():
        gene_symbols[pos] = name  # overwrite panel entries with the designed gene names

    counts = np.zeros((len(gene_symbols), n_samples), dtype=np.int64)

    for gi, gene in enumerate(gene_symbols):
        gseed = stable_seed("gene", gene)
        grng = np.random.default_rng(gseed)
        if gene in designed:
            baseline = designed[gene]["baseline"]
            sigma = designed[gene]["sigma"]
            log2fc = designed[gene]["log2fc"]
        else:
            baseline = float(np.exp(grng.uniform(np.log(20.0), np.log(3000.0))))
            sigma = float(grng.uniform(0.25, 0.45))
            log2fc = 0.0

        for si, sid in enumerate(sample_ids):
            is_treated = condition_by_id[sid] == "treated"
            effect = (2.0**log2fc) if (is_treated and gene in designed) else 1.0
            mu = baseline * size_factor[sid] * effect
            noisy_mu = mu * float(np.exp(grng.normal(0, sigma)))
            counts[gi, si] = grng.poisson(noisy_mu)

    counts_df = pd.DataFrame(counts, index=gene_symbols, columns=sample_ids)
    return counts_df, designed_positions


def lock_ground_truth(counts_df: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """Correct, ID-based analysis: the answer key.

    Aligns strictly by sample_id (never by row/column position), which is
    the only thing that is scientifically defensible given that the
    expression matrix and metadata are independently ordered.
    """
    ordered_ids = metadata["sample_id"].tolist()
    counts_ordered = counts_df[ordered_ids]  # explicit ID-based column selection
    condition = pd.Series(metadata["condition"].to_numpy(), index=ordered_ids)

    log2cpm = pipeline_stats.compute_log2_cpm(counts_ordered)
    de_table = pipeline_stats.differential_expression(log2cpm, condition)
    de_table = de_table.sort_values("adjusted_p_value")

    top = de_table.iloc[0]
    verified_matching_sample_ids = sum(
        1 for sid in ordered_ids if sid in counts_df.columns and sid in metadata["sample_id"].to_numpy()
    )

    return {
        "top_gene": str(de_table.index[0]),
        "log2_fold_change": round(float(top["log2_fold_change"]), 4),
        "adjusted_p_value": float(top["adjusted_p_value"]),
        "verified_matching_sample_ids": int(verified_matching_sample_ids),
        "full_ranked_table": de_table.reset_index().to_dict(orient="records"),
    }


def main() -> None:
    rng = np.random.default_rng(stable_seed("rnaseq-metadata-misalignment", "gene-panel"))
    gene_symbols = make_gene_symbols(N_GENES, rng)

    metadata = build_sample_roster()
    counts_df, designed_positions = simulate_counts(metadata, gene_symbols, rng)

    acq_rng = np.random.default_rng(stable_seed("acquisition-order"))
    acq_order = acquisition_order(metadata["sample_id"].tolist(), acq_rng)
    expr_public = counts_df[acq_order]

    public_dir = REPO_ROOT / "data_generation" / "public"
    private_dir = REPO_ROOT / "data_generation" / "private"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(public_dir / "sample_metadata.csv", index=False)
    expr_public.to_csv(public_dir / "expression_matrix.csv")

    ground_truth = lock_ground_truth(counts_df, metadata)
    ground_truth["designed_gene_positions"] = designed_positions
    (private_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))

    print("Generated:")
    print(f"  public/sample_metadata.csv: {len(metadata)} rows")
    print(f"  public/expression_matrix.csv: {expr_public.shape[0]} genes x {expr_public.shape[1]} samples")
    print(f"  private/ground_truth.json: top_gene={ground_truth['top_gene']!r}")
    print(json.dumps({k: v for k, v in ground_truth.items() if k != "full_ranked_table"}, indent=2))


if __name__ == "__main__":
    main()
