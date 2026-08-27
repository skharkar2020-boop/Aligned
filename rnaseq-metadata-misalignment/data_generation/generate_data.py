"""Synthetic bulk RNA-seq dataset generator for the compound-response DE task.

Ground truth (never exposed via file/column names in the public data):
  - 24 samples across two independent cohorts, cohort1 (sample_1..sample_12)
    and cohort2 (sample_13..sample_24), each 6 vehicle-control / 6
    compound-treated. cohort2 is a later, independent confirmatory run.
  - Three genes carry a real condition effect present identically in BOTH
    cohorts (log2 fold-change, treated over control): TRUE_TOP is the
    strongest and lowest-noise; DECOY_A and DECOY_B are close enough in
    effect size that a wrong (but internally consistent, non-crashing)
    sample/label pairing can plausibly promote either of them to "top hit"
    instead within a single cohort. All other genes, including
    CONFOUND_GENE, carry no true condition effect.
  - cohort2 was processed with materially less rigor than cohort1 (see
    COHORT2_EXTRA_SIGMA): every gene's biological noise is higher there,
    which is why cohort2 alone is a weaker replication of a real effect
    even though the true effect size is identical across cohorts.
  - CONFOUND_GENE carries a real, deliberate technical artifact specific to
    cohort2: cohort2's control and treated samples were processed on two
    different dates/reagent lots (a realistic staggered-processing
    scenario), and CONFOUND_GENE happens to be sensitive to that
    processing difference. Because processing date is confounded with
    condition *only within cohort2*, this shows up as a strong, spurious
    condition effect there and nowhere else -- verified empirically (see
    task/README.md) that a naive cohort2-only analysis ranks CONFOUND_GENE
    above TRUE_TOP, while TRUE_TOP is the one that actually replicates
    (is independently significant in both cohorts) and CONFOUND_GENE does
    not (null in cohort1). The correct answer comes from cohort1 alone,
    the cohort without the processing-date confound; cohort2 is
    informative context, not evidence to trust on its own.
  - The expression matrix's sample (column) order is the sequencer
    acquisition order, which is intentionally NOT the metadata table's row
    order: real cores randomize sample-to-lane assignment specifically to
    avoid confounding batch/lane with condition, so a pipeline that assumes
    matching row order between the two tables is wrong from the start, even
    before any downstream re-sort makes it worse.

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

# cohort2 was a rushed, less-controlled confirmatory run: every gene's
# biological noise is higher there than in cohort1. This alone does not
# invalidate cohort2 (a real effect can still replicate through more
# noise), but it weakens cohort2's own evidence relative to cohort1's.
COHORT2_EXTRA_SIGMA = 0.40

# CONFOUND_GENE's cohort2 samples: control processed on date A, treated on
# date B (staggered processing, confounded with condition only in cohort2).
# The shift is a real, deliberate technical artifact, not biology -- see
# module docstring. Calibrated (see task/README.md for the sweep) so that
# a naive cohort2-only analysis ranks CONFOUND_GENE above TRUE_TOP with a
# comfortable margin (~10x in adjusted p-value at the values below), while
# TRUE_TOP still replicates as independently significant in both cohorts
# and CONFOUND_GENE stays null in cohort1 (no processing-date confound
# there).
CONFOUND_GENE_LOG2_SHIFT = 3.5


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
    """Metadata row order == LIMS registration order (public/sample_metadata.csv).

    cohort1: sample_1..sample_12 (6 control, 6 treated), the original run.
    cohort2: sample_13..sample_24 (6 control, 6 treated), a later,
    independent confirmatory run -- registered as a block, after cohort1.
    """
    rows = []
    for i in range(1, 13):
        condition = "control" if i <= 6 else "treated"
        rows.append({"sample_id": f"sample_{i}", "condition": condition, "cohort": "cohort1", "qc_pass": True})
    for i in range(13, 25):
        j = i - 12
        condition = "control" if j <= 6 else "treated"
        rows.append({"sample_id": f"sample_{i}", "condition": condition, "cohort": "cohort2", "qc_pass": True})
    return pd.DataFrame(rows)


def acquisition_order(sample_ids: list[str], rng: np.random.Generator) -> list[str]:
    order = list(sample_ids)
    rng.shuffle(order)
    return order


def simulate_counts(metadata: pd.DataFrame, gene_symbols: list[str], rng: np.random.Generator):
    """Note on identifiers: gene_symbols (from make_gene_symbols) are used
    as-is, including at the four "designed" positions below -- the panel
    never contains a literal string like "TRUE_TOP" anywhere in the public
    data. "TRUE_TOP"/"DECOY_A"/"DECOY_B"/"CONFOUND_GENE" are internal-only
    labels for this authoring code and for ground_truth.json (private,
    never shipped); designed_positions maps each label to a *position*,
    and the actual gene symbol at that position is whatever
    make_gene_symbols already generated there.
    """
    sample_ids = metadata["sample_id"].tolist()
    condition_by_id = dict(zip(metadata["sample_id"], metadata["condition"]))
    cohort_by_id = dict(zip(metadata["sample_id"], metadata["cohort"]))

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
        "CONFOUND_GENE": {"log2fc": 0.0, "sigma": 0.30, "baseline": 400.0},
    }
    # Fixed, non-obvious positions in the gene panel (not gene[0], gene[1], gene[2]).
    designed_positions = {"TRUE_TOP": 137, "DECOY_A": 42, "DECOY_B": 216, "CONFOUND_GENE": 88}
    gene_symbols = list(gene_symbols)
    position_to_label = {pos: label for label, pos in designed_positions.items()}
    designed_gene_symbols = {label: gene_symbols[pos] for label, pos in designed_positions.items()}
    confound_gene_symbol = designed_gene_symbols["CONFOUND_GENE"]

    counts = np.zeros((len(gene_symbols), n_samples), dtype=np.int64)

    for gi, gene in enumerate(gene_symbols):
        gseed = stable_seed("gene", gene)
        grng = np.random.default_rng(gseed)
        label = position_to_label.get(gi)
        if label is not None:
            baseline = designed[label]["baseline"]
            base_sigma = designed[label]["sigma"]
            log2fc = designed[label]["log2fc"]
        else:
            baseline = float(np.exp(grng.uniform(np.log(20.0), np.log(3000.0))))
            base_sigma = float(grng.uniform(0.25, 0.45))
            log2fc = 0.0

        for si, sid in enumerate(sample_ids):
            is_treated = condition_by_id[sid] == "treated"
            in_cohort2 = cohort_by_id[sid] == "cohort2"

            effect = (2.0**log2fc) if (is_treated and log2fc != 0.0) else 1.0

            # CONFOUND_GENE only: a real technical shift specific to
            # cohort2's treated samples (processing-date/reagent-lot
            # confound), on top of (not instead of) its null biological
            # effect above.
            confound = 1.0
            if gene == confound_gene_symbol and in_cohort2 and is_treated:
                confound = 2.0**CONFOUND_GENE_LOG2_SHIFT

            sigma = base_sigma + (COHORT2_EXTRA_SIGMA if in_cohort2 else 0.0)

            mu = baseline * size_factor[sid] * effect * confound
            noisy_mu = mu * float(np.exp(grng.normal(0, sigma)))
            counts[gi, si] = grng.poisson(noisy_mu)

    counts_df = pd.DataFrame(counts, index=gene_symbols, columns=sample_ids)
    return counts_df, designed_positions, designed_gene_symbols


def cohort_sample_ids(metadata: pd.DataFrame, cohort: str) -> list[str]:
    return metadata.loc[metadata["cohort"] == cohort, "sample_id"].tolist()


def differential_expression_for_cohort(
    counts_df: pd.DataFrame, metadata: pd.DataFrame, cohort: str
) -> pd.DataFrame:
    """ID-based, single-cohort differential expression (plain per-gene
    Welch's t-test + BH-FDR). Never mixes samples across cohorts.
    """
    ids = cohort_sample_ids(metadata, cohort)
    counts = counts_df[ids]  # explicit ID-based column selection
    condition = pd.Series(
        metadata.set_index("sample_id").loc[ids, "condition"].to_numpy(), index=ids
    )
    log2cpm = pipeline_stats.compute_log2_cpm(counts)
    de_table = pipeline_stats.differential_expression(log2cpm, condition)
    return de_table.sort_values("adjusted_p_value")


def lock_ground_truth(counts_df: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """Correct analysis: the answer key.

    Two independent requirements, both necessary:
    1. ID-based alignment -- every sample's expression profile is matched to
       its metadata by sample_id, never by row/column position, since the
       expression matrix and metadata are independently ordered.
    2. Cross-cohort reconciliation -- cohort1 and cohort2 are each
       internally consistent, individually plausible, non-crashing
       analyses that disagree on the top gene. The defensible answer is
       cohort1's, the cohort without the processing-date confound;
       cohort2's top hit does not replicate in cohort1, and vice versa,
       TRUE_TOP's real effect is significant in both.
    """
    all_ids = metadata["sample_id"].tolist()
    verified_matching_sample_ids = sum(
        1 for sid in all_ids if sid in counts_df.columns and sid in metadata["sample_id"].to_numpy()
    )

    cohort1_de = differential_expression_for_cohort(counts_df, metadata, "cohort1")
    cohort2_de = differential_expression_for_cohort(counts_df, metadata, "cohort2")

    top = cohort1_de.iloc[0]

    return {
        "top_gene": str(cohort1_de.index[0]),
        "log2_fold_change": round(float(top["log2_fold_change"]), 4),
        "adjusted_p_value": float(top["adjusted_p_value"]),
        "verified_matching_sample_ids": int(verified_matching_sample_ids),
        "confounded_cohort": "cohort2",
        "cohort1_top_gene": str(cohort1_de.index[0]),
        "cohort2_top_gene": str(cohort2_de.index[0]),
        "cohort1_ranked_table": cohort1_de.reset_index().to_dict(orient="records"),
        "cohort2_ranked_table": cohort2_de.reset_index().to_dict(orient="records"),
    }


def main() -> None:
    rng = np.random.default_rng(stable_seed("rnaseq-metadata-misalignment", "gene-panel"))
    gene_symbols = make_gene_symbols(N_GENES, rng)

    metadata = build_sample_roster()
    counts_df, designed_positions, designed_gene_symbols = simulate_counts(metadata, gene_symbols, rng)

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
    ground_truth["designed_gene_symbols"] = designed_gene_symbols
    (private_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))

    print("Generated:")
    print(f"  public/sample_metadata.csv: {len(metadata)} rows")
    print(f"  public/expression_matrix.csv: {expr_public.shape[0]} genes x {expr_public.shape[1]} samples")
    print(f"  private/ground_truth.json: top_gene={ground_truth['top_gene']!r}")
    summary = {k: v for k, v in ground_truth.items() if "ranked_table" not in k}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
