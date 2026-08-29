"""Synthetic bulk RNA-seq dataset generator for the compound-response DE task.

Ground truth (never exposed via file/column names in the public data):
  - 24 samples across two independent cohorts, cohort1 (sample_1..sample_12)
    and cohort2 (sample_13..sample_24), each 6 vehicle-control / 6
    compound-treated. cohort2 is a later, independent confirmatory run.
  - TRUE_GENE has a real condition effect present in BOTH cohorts, same
    direction, but with real biological heterogeneity: strong in cohort1,
    weaker (but still real, still same-signed) in cohort2. Neither cohort
    can be discarded wholesale -- this is legitimate effect-size
    heterogeneity, not "null in one cohort."
  - CONFOUND_GENE has no true biological condition effect in either cohort.
    It carries a real, deliberate *latent* technical artifact confined to
    cohort2: a continuous per-sample latent factor Z (never written to any
    public column) that is partially, not perfectly, correlated with
    condition within cohort2 only (a realistic staggered-processing/
    reagent-lot scenario). CONFOUND_GENE's expression loads heavily on Z,
    so within cohort2 it shows an extremely strong, spurious,
    condition-aligned shift; in cohort1 (where Z carries no signal) it
    shows nothing. This makes CONFOUND_GENE the single strongest hit in a
    naive pooled or cohort2-only analysis, well above TRUE_GENE.
  - CONSISTENCY_GENE has a real biological effect, same direction and
    similar magnitude in both cohorts -- but weaker overall combined
    evidence than TRUE_GENE (roughly 5-10x weaker by Fisher's method, not
    orders of magnitude), calibrated as a genuine close call rather than a
    landslide: a naive "prefer whatever replicates most consistently" rule
    promotes CONSISTENCY_GENE over TRUE_GENE, when TRUE_GENE's evidence is
    actually stronger once real, moderate heterogeneity is properly
    tolerated, but the margin is real, not trivial.
  - GHOST_REPLICATOR is a third gene that clears the same nominal-
    significance-in-both-cohorts bar as TRUE_GENE and CONSISTENCY_GENE
    (small real biological effect, same direction both cohorts) but whose
    cohort2 evidence also loads partially on the latent factor Z, so it
    looks more impressive (particularly in cohort2 alone, or in a naive
    pooled/fold-change-only comparison) than its actual combined evidence
    supports. It never wins the Fisher-combined-p comparison against
    TRUE_GENE or CONSISTENCY_GENE, but expands the replicated-candidate
    set from two genes to three, and its large cohort2 fold-change is a
    plausible-looking distractor for a pooled-DE or cohort2-only strategy.
  - CONFOUND_GENE's own cohort1 result is calibrated to be a near-miss, not
    a clean null: it clears nominal p<0.05 in cohort1 on its own, but with
    the OPPOSITE sign from its overwhelming cohort2 effect -- a naive
    replication check that tests significance without also checking effect
    direction would incorrectly treat it as replicated.
  - REAL_HETEROGENEITY_GENE has a real, independently significant effect in
    each cohort individually, but the two cohorts' effects have opposite
    sign (not a technical artifact -- it carries no loading on the latent
    factor Z at all, unlike CONFOUND_GENE and the sentinels). It fails the
    same-sign replication requirement for the same procedural reason
    CONFOUND_GENE does, but for a genuinely different, non-technical
    reason: real cross-cohort biological heterogeneity can also produce a
    sign flip, not just a technical artifact, so failing replication is not
    itself proof that a candidate's cohort-level significance is a
    processing artifact.
  - SENTINEL_1..SENTINEL_6 carry no true biological effect and no
    intentional labeling anywhere in public data. Each loads (with varying
    sign and magnitude) on the same latent factor Z as CONFOUND_GENE, so
    within cohort2 they covary with CONFOUND_GENE and with each other --
    the technical axis is recoverable from expression structure itself
    (e.g. correlation among genes within cohort2, or a PCA/factor axis),
    never from a labeled column.
  - The latent factor Z is defined only for cohort2 samples (cohort1 has no
    technical artifact at all: Z contributes nothing there). Z's condition
    means are separated by construction but its within-group spread means
    any single sample's Z, and hence any single gene's shift, is noisy --
    this is a real confound, not a deterministic label.
  - The expression matrix's sample (column) order is the sequencer
    acquisition order, which is intentionally NOT the metadata table's row
    order: real cores randomize sample-to-lane assignment specifically to
    avoid confounding lane with condition, so a pipeline that assumes
    matching row order between the two tables is wrong from the start, even
    before any downstream re-sort makes it worse.

Counts are simulated gene-by-gene as Poisson-lognormal: a per-gene, per
condition mean count scaled by a per-sample library-size factor and (for
the handful of genes that load on it) the shared latent factor, with
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

# Latent technical factor Z: defined only for cohort2 samples, partially
# (not perfectly) correlated with condition there. DELTA_Z is half the gap
# between the condition-group means; Z_SD is the within-group spread, kept
# tight relative to the gap so the confound is real but noisy (a sample's
# own Z is not a perfect condition label). Calibrated (task/README.md) so
# CONFOUND_GENE's z_loading turns this into a strong, spurious,
# condition-aligned shift within cohort2 only.
DELTA_Z = 1.2
Z_SD = 0.3


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


def latent_z_by_sample(metadata: pd.DataFrame) -> dict[str, float]:
    """Per-sample draw of the shared latent technical factor.

    Zero for every cohort1 sample (no technical artifact there at all).
    For cohort2, drawn from a condition-separated Normal so it is
    correlated with condition but not a deterministic function of it. The
    same draw is reused for every gene that loads on this factor, which is
    what makes those genes covary with each other -- a single shared axis,
    not independent per-gene noise.
    """
    z: dict[str, float] = {}
    for _, row in metadata.iterrows():
        sid = row["sample_id"]
        if row["cohort"] != "cohort2":
            z[sid] = 0.0
            continue
        mean = DELTA_Z if row["condition"] == "treated" else -DELTA_Z
        draw_rng = np.random.default_rng(stable_seed("latent_z", sid))
        z[sid] = float(mean + draw_rng.normal(0.0, Z_SD))
    return z


# Internal-only authoring labels -> simulation parameters. Never written as
# literal strings into public data; designed_positions maps each label to a
# fixed panel position, and the actual gene symbol shipped there is
# whatever make_gene_symbols already generated at that position.
DESIGNED_GENES = {
    "TRUE_GENE": {"baseline": 500.0, "sigma": 0.25, "log2fc_c1": 1.8, "log2fc_c2": 0.8, "z_loading": 0.0},
    "CONFOUND_GENE": {"baseline": 400.0, "sigma": 0.25, "log2fc_c1": 0.0, "log2fc_c2": 0.0, "z_loading": 2.5},
    "CONSISTENCY_GENE": {"baseline": 350.0, "sigma": 0.18, "log2fc_c1": 1.2, "log2fc_c2": 1.2, "z_loading": 0.0},
    "GHOST_REPLICATOR": {"baseline": 300.0, "sigma": 0.28, "log2fc_c1": 0.42, "log2fc_c2": 0.42, "z_loading": 0.8},
    "REAL_HETEROGENEITY_GENE": {"baseline": 300.0, "sigma": 0.22, "log2fc_c1": 0.8, "log2fc_c2": -0.8, "z_loading": 0.0},
    "SENTINEL_1": {"baseline": 280.0, "sigma": 0.30, "log2fc_c1": 0.0, "log2fc_c2": 0.0, "z_loading": 0.8},
    "SENTINEL_2": {"baseline": 260.0, "sigma": 0.30, "log2fc_c1": 0.0, "log2fc_c2": 0.0, "z_loading": -0.6},
    "SENTINEL_3": {"baseline": 300.0, "sigma": 0.30, "log2fc_c1": 0.0, "log2fc_c2": 0.0, "z_loading": 1.0},
    "SENTINEL_4": {"baseline": 240.0, "sigma": 0.30, "log2fc_c1": 0.0, "log2fc_c2": 0.0, "z_loading": -0.9},
    "SENTINEL_5": {"baseline": 320.0, "sigma": 0.30, "log2fc_c1": 0.0, "log2fc_c2": 0.0, "z_loading": 0.7},
    "SENTINEL_6": {"baseline": 270.0, "sigma": 0.30, "log2fc_c1": 0.0, "log2fc_c2": 0.0, "z_loading": -0.5},
}
# Fixed, non-obvious positions in the 300-gene panel.
DESIGNED_POSITIONS = {
    "TRUE_GENE": 137,
    "CONFOUND_GENE": 153,
    "CONSISTENCY_GENE": 203,
    "GHOST_REPLICATOR": 33,
    "REAL_HETEROGENEITY_GENE": 260,
    "SENTINEL_1": 15,
    "SENTINEL_2": 249,
    "SENTINEL_3": 61,
    "SENTINEL_4": 178,
    "SENTINEL_5": 294,
    "SENTINEL_6": 112,
}


def simulate_counts(metadata: pd.DataFrame, gene_symbols: list[str], rng: np.random.Generator):
    sample_ids = metadata["sample_id"].tolist()
    condition_by_id = dict(zip(metadata["sample_id"], metadata["condition"]))
    cohort_by_id = dict(zip(metadata["sample_id"], metadata["cohort"]))
    z_by_id = latent_z_by_sample(metadata)

    n_samples = len(sample_ids)
    size_factor = {
        sid: float(np.exp(np.random.default_rng(stable_seed("size_factor", sid)).normal(0, 0.08)))
        for sid in sample_ids
    }

    gene_symbols = list(gene_symbols)
    position_to_label = {pos: label for label, pos in DESIGNED_POSITIONS.items()}
    designed_gene_symbols = {label: gene_symbols[pos] for label, pos in DESIGNED_POSITIONS.items()}

    counts = np.zeros((len(gene_symbols), n_samples), dtype=np.int64)

    for gi, gene in enumerate(gene_symbols):
        gseed = stable_seed("gene", gene)
        grng = np.random.default_rng(gseed)
        label = position_to_label.get(gi)
        if label is not None:
            spec = DESIGNED_GENES[label]
            baseline = spec["baseline"]
            base_sigma = spec["sigma"]
            log2fc_c1 = spec["log2fc_c1"]
            log2fc_c2 = spec["log2fc_c2"]
            z_loading = spec["z_loading"]
        else:
            baseline = float(np.exp(grng.uniform(np.log(20.0), np.log(3000.0))))
            base_sigma = float(grng.uniform(0.25, 0.45))
            log2fc_c1 = 0.0
            log2fc_c2 = 0.0
            z_loading = 0.0

        for si, sid in enumerate(sample_ids):
            is_treated = condition_by_id[sid] == "treated"
            in_cohort2 = cohort_by_id[sid] == "cohort2"

            log2fc = log2fc_c2 if in_cohort2 else log2fc_c1
            effect = (2.0**log2fc) if (is_treated and log2fc != 0.0) else 1.0

            z_shift = (2.0 ** (z_loading * z_by_id[sid])) if z_loading != 0.0 else 1.0

            mu = baseline * size_factor[sid] * effect * z_shift
            noisy_mu = mu * float(np.exp(grng.normal(0, base_sigma)))
            counts[gi, si] = grng.poisson(noisy_mu)

    counts_df = pd.DataFrame(counts, index=gene_symbols, columns=sample_ids)
    return counts_df, DESIGNED_POSITIONS, designed_gene_symbols


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


def pooled_differential_expression(counts_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Naive pooled DE: all 24 samples in one comparison, cohort-blind."""
    ids = metadata["sample_id"].tolist()
    counts = counts_df[ids]
    condition = pd.Series(
        metadata.set_index("sample_id").loc[ids, "condition"].to_numpy(), index=ids
    )
    log2cpm = pipeline_stats.compute_log2_cpm(counts)
    de_table = pipeline_stats.differential_expression(log2cpm, condition)
    return de_table.sort_values("adjusted_p_value")


def fisher_combined_p(p1: float, p2: float) -> float:
    """Fisher's method for combining two independent nominal p-values."""
    from scipy import stats as scipy_stats

    statistic = -2.0 * (np.log(max(p1, 1e-300)) + np.log(max(p2, 1e-300)))
    return float(scipy_stats.chi2.sf(statistic, df=4))


NOMINAL_P_THRESHOLD = 0.05


def classify_heterogeneity(log2fc_c1: float, log2fc_c2: float) -> str:
    """Categorical, independently-recomputable heterogeneity label for a
    gene's own pair of per-cohort effect sizes. Same-signed pair: the
    weaker cohort must retain at least 40% of the stronger cohort's
    magnitude to count as "consistent"; otherwise it's "stronger_in_*".
    Opposite-signed pairs are their own category.
    """
    if np.sign(log2fc_c1) != np.sign(log2fc_c2) and log2fc_c1 != 0.0 and log2fc_c2 != 0.0:
        return "opposite_direction_between_cohorts"
    a, b = abs(log2fc_c1), abs(log2fc_c2)
    larger, smaller = max(a, b), min(a, b)
    if larger == 0.0:
        return "consistent_both_cohorts"
    ratio = smaller / larger
    if ratio >= 0.4:
        return "consistent_both_cohorts"
    return "stronger_in_cohort1_weaker_in_cohort2" if a >= b else "stronger_in_cohort2_weaker_in_cohort1"


def lock_ground_truth(counts_df: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """Correct analysis: the answer key.

    Required, in order:
    1. ID-based alignment -- every sample's expression profile is matched to
       its metadata by sample_id, never by row/column position.
    2. Per-cohort (never pooled) differential expression.
    3. Independent replication: a candidate is only trustworthy if it shows
       nominally significant (raw p < 0.05), same-signed evidence
       independently in EACH cohort -- a real, standard-genomics
       reconciliation strategy that (unlike a formulaic random-effects
       meta-analysis at k=2) does not mechanically punish a gene's own
       legitimate cross-cohort heterogeneity.
    4. Among genes that clear that bar, prefer the one with the stronger
       combined evidence (Fisher's method on the two independent nominal
       p-values) -- this is what separates TRUE_GENE (strong in cohort1,
       real but weaker in cohort2) from CONSISTENCY_GENE (moderate and
       consistent in both, but with less overall evidence).
    """
    all_ids = metadata["sample_id"].tolist()
    verified_matching_sample_ids = all(sid in counts_df.columns for sid in all_ids)

    cohort1_de = differential_expression_for_cohort(counts_df, metadata, "cohort1")
    cohort2_de = differential_expression_for_cohort(counts_df, metadata, "cohort2")
    pooled_de = pooled_differential_expression(counts_df, metadata)

    candidates = sorted(set(cohort1_de.index[:10]) | set(cohort2_de.index[:10]) | {pooled_de.index[0]})

    replicated = []
    for gene in candidates:
        r1 = cohort1_de.loc[gene]
        r2 = cohort2_de.loc[gene]
        same_sign = np.sign(r1["log2_fold_change"]) == np.sign(r2["log2_fold_change"])
        both_nominal = r1["p_value"] < NOMINAL_P_THRESHOLD and r2["p_value"] < NOMINAL_P_THRESHOLD
        if same_sign and both_nominal:
            combined_p = fisher_combined_p(float(r1["p_value"]), float(r2["p_value"]))
            replicated.append((gene, combined_p, r1, r2))

    assert replicated, "expected at least one gene to pass the independent-replication gate"
    replicated.sort(key=lambda item: item[1])
    top_gene, _, r1, r2 = replicated[0]

    # The rejected competing gene: among genes that fail the replication
    # gate, the one with the single strongest one-cohort result (smallest
    # single-cohort p-value) -- i.e. the "extremely strong signal driven
    # mainly by one cohort" story, not whatever happens to rank highest
    # once pooled variance mixes both cohorts together.
    rejected_competing_gene = None
    best_single_cohort_p = None
    search_pool = sorted(set(cohort1_de.index[:15]) | set(cohort2_de.index[:15]))
    for gene in search_pool:
        if gene == top_gene:
            continue
        r1g, r2g = cohort1_de.loc[gene], cohort2_de.loc[gene]
        same_sign = np.sign(r1g["log2_fold_change"]) == np.sign(r2g["log2_fold_change"])
        both_nominal = r1g["p_value"] < NOMINAL_P_THRESHOLD and r2g["p_value"] < NOMINAL_P_THRESHOLD
        if same_sign and both_nominal:
            continue
        candidate_p = min(float(r1g["p_value"]), float(r2g["p_value"]))
        if best_single_cohort_p is None or candidate_p < best_single_cohort_p:
            best_single_cohort_p = candidate_p
            rejected_competing_gene = str(gene)

    c1_fc = float(r1["log2_fold_change"])
    c2_fc = float(r2["log2_fold_change"])

    # Headline log2_fold_change/adjusted_p_value: the top gene's own result
    # from its *home* cohort -- whichever of the two independent analyses
    # gives it the stronger (smaller p-value) evidence -- never the naive
    # 24-sample pooled number, which mixes two differently-behaved cohorts
    # into one estimate and is deliberately a wrong strategy here (see
    # pooled_ranked_table: naive pooling promotes a different gene
    # entirely, one that fails the replication check below).
    home = r1 if float(r1["p_value"]) <= float(r2["p_value"]) else r2

    return {
        "top_gene": str(top_gene),
        "log2_fold_change": round(float(home["log2_fold_change"]), 4),
        "adjusted_p_value": float(home["adjusted_p_value"]),
        "analysis_strategy": "strategy_e",  # neutral code for per-cohort independent replication
        "cohort1_log2_fold_change": round(c1_fc, 4),
        "cohort2_log2_fold_change": round(c2_fc, 4),
        "heterogeneity_assessment": classify_heterogeneity(c1_fc, c2_fc),
        "verified_matching_sample_ids": bool(verified_matching_sample_ids),
        "rejected_competing_gene": rejected_competing_gene,
        "cohort1_ranked_table": cohort1_de.reset_index().to_dict(orient="records"),
        "cohort2_ranked_table": cohort2_de.reset_index().to_dict(orient="records"),
        "pooled_ranked_table": pooled_de.reset_index().to_dict(orient="records"),
    }


def write_prior_report(public_dir: Path, designed_gene_symbols: dict, ground_truth: dict) -> None:
    """A stale prior report the agent will find alongside the pipeline: it
    names the right gene, but from a much smaller, single-cohort pilot with
    different numbers -- present so blindly copying it fails the numeric
    tolerances, and independent recomputation is still required.
    """
    true_symbol = designed_gene_symbols["TRUE_GENE"]
    pilot_log2fc = round(ground_truth["cohort1_log2_fold_change"] - 0.9, 2)
    pilot_padj = 8.1e-3
    text = (
        "# Prior pilot analysis (archival)\n\n"
        f"Compound-response pilot, n=3 control / n=3 treated (single site, no "
        "confirmatory cohort). Top transcriptional responder:\n\n"
        f"- gene: {true_symbol}\n"
        f"- log2 fold-change (treated vs. control): {pilot_log2fc}\n"
        f"- adjusted p-value: {pilot_padj:.1e}\n\n"
        "This pilot was underpowered and predates the confirmatory cohort2 "
        "run. Do not report these numbers directly -- they are provided as "
        "historical context only and must be independently reproduced "
        "against the current data before being cited in any go/no-go "
        "decision.\n"
    )
    (public_dir / "prior_pilot_report.md").write_text(text)


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
    write_prior_report(public_dir, designed_gene_symbols, ground_truth)

    ground_truth["designed_gene_positions"] = designed_positions
    ground_truth["designed_gene_symbols"] = designed_gene_symbols
    (private_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))

    print("Generated:")
    print(f"  public/sample_metadata.csv: {len(metadata)} rows")
    print(f"  public/expression_matrix.csv: {expr_public.shape[0]} genes x {expr_public.shape[1]} samples")
    print(f"  public/prior_pilot_report.md")
    print(f"  private/ground_truth.json: top_gene={ground_truth['top_gene']!r}")
    summary = {k: v for k, v in ground_truth.items() if "ranked_table" not in k}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
