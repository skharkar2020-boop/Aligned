"""Synthetic bulk RNA-seq dataset generator for the compound-response DE task.

Ground truth (never exposed via file/column names in the public data):
  - 13 samples: 6 vehicle-control, 7 compound-treated.
  - Two of the 13 are a genuine near-duplicate ID pair, `sample_2` and
    `sample_02` -- distinct biological replicates from two different
    collection batches (batch1 used unpadded numeric IDs; batch2, added
    later, used zero-padded IDs). Both are real, independent samples with
    their own counts; neither is a copy of the other.
  - `sample_02` is the sole member of `batch2`. A single-sample batch is
    perfectly confounded with itself: no statistical procedure can tell a
    real biological signal in that one sample apart from a batch-specific
    technical artifact, because there is no second batch2 sample to compare
    it against. The scientifically defensible choice is to run the primary
    differential-expression comparison on the balanced, single-batch cohort
    (the 12 batch1 samples) and treat `sample_02` as informative context,
    not as a usable data point for the comparison -- while still verifying
    its identity like any other sample. To make that choice consequential
    rather than a coin flip, `sample_02`'s TRUE_TOP count carries a real,
    deliberate batch-specific technical artifact (see BATCH2_ARTIFACT_*
    below): naively including it without recognizing the confound changes
    the answer.
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
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "task" / "environment" / "data"))

from pipeline import stats as pipeline_stats  # noqa: E402  (path set above)


def stable_seed(*parts) -> int:
    """Deterministic replacement for hash((...)) % 2**31 (PYTHONHASHSEED-proof)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(key).hexdigest(), 16) % (2**31)


# sample_02 (batch2, n=1) carries a real technical artifact on TRUE_TOP: its
# count is scaled to BATCH2_ARTIFACT_FACTOR of what the same biological
# effect + noise draw would otherwise give it. Calibrated (see
# task/README.md for the sweep) to land within the control group's own
# TRUE_TOP range -- i.e. it looks like an unremarkable low value, not an
# outlier -- while being strong enough that naively including sample_02 in
# the treated group flips the top hit away from TRUE_TOP. The flip boundary
# was found empirically between factor 0.32 and 0.35; 0.15 keeps a
# comfortable margin below it.
BATCH2_ARTIFACT_GENE = "TRUE_TOP"
BATCH2_ARTIFACT_FACTOR = 0.15

# VARIANCE_TRAP is a null gene (no true condition effect) constructed so its
# 12 batch1 samples land in two tight, deterministic clusters with a small
# apparent gap between them -- by chance/construction, not biology. Its own
# per-gene sample variance is far below the panel's typical spread, which is
# exactly the situation a plain per-gene t-test cannot distinguish from a
# real, low-noise effect: with only 6 samples per group, a per-gene variance
# estimate is itself extremely noisy, and a gene that happens to land with
# low realized variance produces an artificially inflated t-statistic. A
# gene-panel-wide variance-shrinkage step (moderating each gene's variance
# estimate toward the panel's typical value, weighted by how much prior
# confidence to place in that typical value vs. this gene's own noisy
# estimate -- the same idea behind limma's empirical-Bayes moderated t-test
# and DESeq2's dispersion shrinkage) correctly recognizes this gene's
# variance estimate as unreliable and pulls it back up, suppressing the
# false positive; a plain per-gene t-test cannot. Verified empirically (see
# task/README.md): under a plain Welch's t-test VARIANCE_TRAP outranks
# TRUE_TOP; under moderated variance (any prior weight tested from 4 to 8)
# TRUE_TOP is restored and VARIANCE_TRAP drops to the noise floor.
VARIANCE_TRAP_GENE = "VARIANCE_TRAP"
VARIANCE_TRAP_POSITION = 88
VARIANCE_TRAP_BASELINE_LOG2CPM = 12.0
VARIANCE_TRAP_APPARENT_DELTA = 0.25
VARIANCE_TRAP_TARGET_VARIANCE = 0.0005  # per-group sample variance (ddof=1) in log2(CPM+1) space
MODERATION_PRIOR_WEIGHT = 6.0  # d0: simplified, fixed prior weight (not fit via REML/method-of-moments)

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
    designed_positions = {
        "TRUE_TOP": 137,
        "DECOY_A": 42,
        "DECOY_B": 216,
        "VARIANCE_TRAP": VARIANCE_TRAP_POSITION,
    }
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

    # Apply the batch2 technical artifact: this only touches sample_02's
    # stored value for BATCH2_ARTIFACT_GENE, as a deterministic post-hoc
    # scale -- it does not consume any RNG draws, so it cannot perturb any
    # other sample's or gene's simulated counts.
    batch2_ids = metadata.loc[metadata["batch"] != "batch1", "sample_id"].tolist()
    for sid in batch2_ids:
        if sid in counts_df.columns:
            original = counts_df.loc[BATCH2_ARTIFACT_GENE, sid]
            counts_df.loc[BATCH2_ARTIFACT_GENE, sid] = int(round(original * BATCH2_ARTIFACT_FACTOR))

    inject_variance_trap(counts_df, metadata)

    return counts_df, designed_positions


def inject_variance_trap(counts_df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """Overwrite VARIANCE_TRAP_GENE's batch1 counts with a deterministic,
    tightly-clustered pattern (see VARIANCE_TRAP_* constants above).

    Uses each sample's own already-simulated library size (this gene's own
    contribution to it is negligible) to convert a target log2(CPM+1) value
    into a raw count, so that the pipeline's own CPM normalization recovers
    the intended log2cpm pattern almost exactly (up to integer rounding,
    which is kept small by using a high baseline CPM). Deterministic --
    consumes no RNG draws, so it cannot perturb any other gene or sample.
    """
    batch1 = metadata[metadata["batch"] == "batch1"]
    control_ids = batch1.loc[batch1["condition"] == "control", "sample_id"].tolist()
    treated_ids = batch1.loc[batch1["condition"] == "treated", "sample_id"].tolist()
    assert len(control_ids) == len(treated_ids) == 6

    lib_sizes = counts_df.sum(axis=0)

    raw_offsets = np.array([-1.5, -0.9, -0.3, 0.3, 0.9, 1.5])
    raw_offsets = raw_offsets - raw_offsets.mean()
    unit_offsets = raw_offsets / np.sqrt(raw_offsets.var(ddof=1))
    scaled_offsets = unit_offsets * np.sqrt(VARIANCE_TRAP_TARGET_VARIANCE)

    def set_counts(sample_ids: list[str], center_log2cpm: float) -> None:
        for offset, sid in zip(scaled_offsets, sample_ids):
            target_log2cpm = center_log2cpm + offset
            cpm = 2.0**target_log2cpm - 1.0
            count = int(round(cpm * lib_sizes[sid] / 1e6))
            counts_df.loc[VARIANCE_TRAP_GENE, sid] = count

    set_counts(control_ids, VARIANCE_TRAP_BASELINE_LOG2CPM)
    set_counts(treated_ids, VARIANCE_TRAP_BASELINE_LOG2CPM + VARIANCE_TRAP_APPARENT_DELTA)


def batch_confound_free_sample_ids(metadata: pd.DataFrame) -> list[str]:
    """Sample IDs from batches with >= 2 members.

    A batch effect cannot be distinguished from a real biological effect in
    a batch with only one sample -- there is nothing within that batch to
    compare it to. The defensible primary comparison uses only samples from
    batches large enough to support that distinction.
    """
    batch_sizes = metadata.groupby("batch")["sample_id"].transform("count")
    return metadata.loc[batch_sizes >= 2, "sample_id"].tolist()


def moderated_differential_expression(
    log2cpm: pd.DataFrame,
    condition: pd.Series,
    group_a: str = "control",
    group_b: str = "treated",
    prior_weight: float = MODERATION_PRIOR_WEIGHT,
) -> pd.DataFrame:
    """Per-gene pooled-variance t-test with empirical-Bayes-style variance
    moderation: each gene's own (noisy, small-n) variance estimate is
    shrunk toward the panel-wide typical variance, weighted by
    `prior_weight` vs. this gene's residual degrees of freedom. Simplified
    relative to limma's eBayes (prior_weight is fixed, not fit via
    method-of-moments/REML), but the mechanism -- and why it matters with
    only 6 samples per group -- is the same.
    """
    a_ids = condition[condition == group_a].index
    b_ids = condition[condition == group_b].index
    a = log2cpm[a_ids].to_numpy(dtype=float)
    b = log2cpm[b_ids].to_numpy(dtype=float)
    n1, n2 = a.shape[1], b.shape[1]
    residual_df = n1 + n2 - 2

    pooled_var = ((n1 - 1) * a.var(axis=1, ddof=1) + (n2 - 1) * b.var(axis=1, ddof=1)) / residual_df
    prior_var = float(np.median(pooled_var))
    shrunk_var = (prior_weight * prior_var + residual_df * pooled_var) / (prior_weight + residual_df)

    log2_fold_change = b.mean(axis=1) - a.mean(axis=1)
    standard_error = np.sqrt(shrunk_var * (1.0 / n1 + 1.0 / n2))
    t_stat = log2_fold_change / standard_error
    moderated_df = prior_weight + residual_df
    p_value = 2.0 * scipy_stats.t.sf(np.abs(t_stat), df=moderated_df)
    adjusted_p_value = pipeline_stats.benjamini_hochberg(p_value)

    return pd.DataFrame(
        {
            "gene": log2cpm.index,
            "log2_fold_change": log2_fold_change,
            "p_value": p_value,
            "adjusted_p_value": adjusted_p_value,
        }
    ).set_index("gene")


def lock_ground_truth(counts_df: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """Correct analysis: the answer key.

    Three independent requirements, all necessary:
    1. ID-based alignment -- every sample's expression profile is matched to
       its metadata by sample_id, never by row/column position, since the
       expression matrix and metadata are independently ordered.
    2. Batch-confound awareness -- the differential-expression comparison
       itself only uses samples from batches with >= 2 members, since a
       single-sample batch cannot be distinguished from a real biological
       effect. sample_02 (batch2, n=1) is still ID-verified like every other
       sample; it is excluded from the statistical comparison, not from
       verification.
    3. Variance moderation -- with only 6 samples per group, a plain
       per-gene t-test's variance estimate is itself unstable enough that a
       null gene with a by-chance tight variance can outrank a real,
       low-noise effect. Moderating each gene's variance toward the panel's
       typical value is necessary to get the right answer, not just a nicer
       one.
    """
    all_ids = metadata["sample_id"].tolist()
    verified_matching_sample_ids = sum(
        1 for sid in all_ids if sid in counts_df.columns and sid in metadata["sample_id"].to_numpy()
    )

    comparison_ids = batch_confound_free_sample_ids(metadata)
    counts_ordered = counts_df[comparison_ids]  # explicit ID-based column selection
    condition = pd.Series(
        metadata.set_index("sample_id").loc[comparison_ids, "condition"].to_numpy(), index=comparison_ids
    )

    log2cpm = pipeline_stats.compute_log2_cpm(counts_ordered)
    de_table = moderated_differential_expression(log2cpm, condition)
    de_table = de_table.sort_values("adjusted_p_value")

    top = de_table.iloc[0]

    return {
        "top_gene": str(de_table.index[0]),
        "log2_fold_change": round(float(top["log2_fold_change"]), 4),
        "adjusted_p_value": float(top["adjusted_p_value"]),
        "verified_matching_sample_ids": int(verified_matching_sample_ids),
        "comparison_sample_ids": comparison_ids,
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
