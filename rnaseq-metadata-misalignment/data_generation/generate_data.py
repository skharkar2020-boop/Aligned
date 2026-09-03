"""Synthetic bulk RNA-seq dataset generator for the compound-response DE task.

Ground truth (never exposed via file/column names in the public data):
  - 24 samples across two independent cohorts, cohort1 (sample_1..sample_12)
    and cohort2 (sample_13..sample_24), each 6 vehicle-control / 6
    compound-treated. cohort2 is a later, independent confirmatory run.
  - TRUE_GENE is the strongest generalizable treatment-associated
    candidate in this panel -- not "the only gene with a real effect" (RNA-
    seq treatment responses are rarely that clean), but the one whose
    cross-cohort evidence remains credible once small-sample variance
    uncertainty is properly accounted for (see VARIANCE_DECOY_GENE below).
    It has a real condition effect present in BOTH cohorts, same
    direction, but with real biological heterogeneity: strong in cohort1,
    weaker (but still real, still same-signed) in cohort2. Neither cohort
    can be discarded wholesale -- this is legitimate effect-size
    heterogeneity, not "null in one cohort." As of round 7 its own headline
    effect is deliberately modest (log2fc_c1=1.3, not a landslide winner
    against VARIANCE_DECOY_GENE on raw/unmoderated evidence). Round 8
    raised log2fc_c2 from 0.9 to 0.95 -- a real oracle-validity defect, not
    a difficulty tweak: under a real, symmetric (any single sample, not
    only treated) leave-one-out robustness check with the moderation prior
    refit on each reduced sample set, TRUE_GENE's original cohort2
    evidence (raw p~0.0067) was too thin to survive losing any one of its
    two lowest-expression control samples, confirmed independently by
    limma-voom, edgeR-QL, and DESeq2, not just this task's own moment
    estimator. See cohort_symmetric_worst_case_loo_p below and
    task/README.md's "Round 8" section for the full numeric case,
    including the numerical search that confirmed 0.95 is the *lowest*
    value that is both genuinely robust and still requires the moderated-
    variance distinction -- at 0.975 and above, a "replicate + symmetric
    LOO robust, rank by raw evidence" workflow that never touches
    moderation already recovers TRUE_GENE on its own.
  - VARIANCE_DECOY_GENE (round 7, recalibrated round 7b) is mechanically
    an entirely ordinary gene: the same generative model as TRUE_GENE
    (baseline=500, sigma=0.25, a real, uniform condition effect in both
    cohorts, no influential-sample trick, no latent-factor loading). It
    differs from TRUE_GENE only in which natural noise draw it happens to
    land on (panel position 119, chosen by searching many candidate
    positions for one where this was true, not by manufacturing any
    sample's value): by chance, its cohort2 within-group sample variance
    came out smaller than its own cohort1 variance (~0.06 vs. ~0.12, a
    real but moderate ~2x swing -- round 7b deliberately softened this
    from round 7's original ~3x/0.03-vs-0.06 swing, see below), which
    inflates its apparent significance there under a plain per-gene Welch
    test -- with only 6 samples per group, a single gene's own variance
    estimate is itself a noisy quantity (5 residual degrees of freedom),
    and this is a real instance of that noise working in the decoy's
    favor. Under ordinary Welch/BH/leave-one-out/Mann-Whitney analysis,
    VARIANCE_DECOY_GENE beats TRUE_GENE on nearly every axis: stronger raw
    combined evidence, BH-significant in both cohorts (TRUE_GENE is
    BH-significant only in cohort1), and a more consistent effect size
    across cohorts. Recovering TRUE_GENE requires recognizing that this
    apparent strength is not robust to standard small-sample variance
    moderation: empirical-Bayes-shrunk variance (see
    fit_ebayes_prior/cohort_moderated_p_value below -- a faithful,
    self-calibrating reimplementation of the same moment-based estimator
    behind limma's eBayes, with no hand-picked prior weight) pulls the
    decoy's smaller cohort2 variance back toward the panel's typical
    value, which is enough to flip the cross-cohort combined-evidence
    ranking in TRUE_GENE's favor. Verified independently against real
    limma-voom, edgeR's quasi-likelihood pipeline, and DESeq2 on this
    exact locked dataset (see task/README.md): all three, each estimating
    their own moderation strength from the data rather than from any
    generator-known parameter, agree with this conclusion.
  - NEAR_MISS_GENE (round 7b) closes a specific loophole found in a real
    Harbor trajectory campaign: agents were reaching TRUE_GENE not by
    computing genuine moderated variance, but by a much shallower proxy --
    eyeballing whether a candidate's OWN pooled variance looks internally
    consistent between its two cohorts (TRUE_GENE's happened to be ~0.085
    in both; VARIANCE_DECOY_GENE's ~3x mismatch stood out by contrast).
    That proxy is not the intended statistical reasoning (real moderation
    borrows strength from the whole panel's variance distribution, not
    from a single gene's own two numbers), so round 7b both softened
    VARIANCE_DECOY_GENE's mismatch (above) and added NEAR_MISS_GENE: a
    second, entirely ordinary gene (same generative model again, smaller
    effect: log2fc_c1=0.8, log2fc_c2=0.7, panel position 34) whose own
    cross-cohort variance ratio (~1.02) is just as clean as TRUE_GENE's,
    and which independently clears full-panel BH significance in BOTH
    cohorts with a more consistent effect size than TRUE_GENE -- i.e. it
    passes every shallow checklist item (clean own-variance, BH-significant
    both cohorts, consistent effect) that a naive-but-plausible-looking
    analysis might use to shortlist a winner. It is a real, unmanipulated,
    weaker treatment-associated signal (RNA-seq responses are rarely
    single-gene-clean; a real analysis can and should surface more than
    one plausible candidate), not a manufactured flaw -- it loses to
    TRUE_GENE under genuine full-panel moderated combined-p by a clear
    margin, confirmed against real limma-voom, edgeR-QL, and DESeq2 (see
    task/README.md), but "clean own-variance + BH-significant both
    cohorts + consistent effect" alone no longer uniquely identifies
    TRUE_GENE now that two genes satisfy it. Distinguishing TRUE_GENE from
    both decoys requires integrating properly-moderated combined evidence
    across the whole candidate set, not any single diagnostic in
    isolation.
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
  - CONSISTENCY_GENE (round 6) is a fragile-replication decoy, not a
    latent-factor confound: every cohort2-treated sample gets a real,
    modest effect (log2fc_c2=0.65), but two of the six also get a large
    additional per-sample boost (+1.8), chosen deterministically and not
    tied to any public column. This makes its headline cohort2 statistics
    (log2FC, nominal p, BH-adjusted p, a Mann-Whitney nonparametric check)
    look as attractive as TRUE_GENE's, or more so, and its cohort1 and
    cohort2 effects look similarly sized (unlike TRUE_GENE's real
    heterogeneity) -- an ordinary Welch/BH/effect-size/nonparametric
    workup plausibly prefers it. But its apparent cohort2 replication is
    disproportionately carried by those two samples: dropping either one
    (a standard leave-one-out / influence check) pushes its cohort2
    p-value from a real hit (~0.04) to clearly non-significant (~0.10),
    while TRUE_GENE's own (weaker-looking, more heterogeneous) cohort2
    evidence stays significant under removal of any single treated
    sample. The scientific claim "this effect independently replicates in
    cohort2" is true for TRUE_GENE and false for CONSISTENCY_GENE, and
    that distinction -- not an arbitrary significance-correction
    convention -- is what a same-sign-and-independently-significant
    replication check should actually be testing.
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

# Fragile-replication decoy mechanism (round 6, CONSISTENCY_GENE): two of
# the six cohort2-treated samples get an extra additive log2fc on top of
# the gene's normal per-sample effect, calibrated (see
# consistency_gene_influential_samples and cohort_robust_to_loo below) so
# headline cohort2 statistics look as strong as or stronger than TRUE_GENE's,
# but a leave-one-out check on the treated group reveals the apparent
# replication depends disproportionately on those two samples. Not tied to
# any public column or to the latent factor Z used elsewhere.
N_INFLUENTIAL_SAMPLES = 2
INFLUENTIAL_EXTRA_LOG2FC = 1.8


def consistency_gene_influential_samples(metadata: pd.DataFrame) -> set[str]:
    """Deterministically pick which cohort2-treated samples get the extra
    per-sample boost for CONSISTENCY_GENE. Independent of Z and of any
    public column -- this is a per-sample expression-level effect, not a
    labeled subgroup.
    """
    treated_ids = metadata.loc[
        (metadata["cohort"] == "cohort2") & (metadata["condition"] == "treated"), "sample_id"
    ].tolist()
    pick_rng = np.random.default_rng(stable_seed("consistency_influential_samples"))
    chosen = pick_rng.choice(treated_ids, size=N_INFLUENTIAL_SAMPLES, replace=False)
    return set(chosen)


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
    "TRUE_GENE": {"baseline": 500.0, "sigma": 0.25, "log2fc_c1": 1.3, "log2fc_c2": 0.95, "z_loading": 0.0},
    "CONFOUND_GENE": {"baseline": 400.0, "sigma": 0.25, "log2fc_c1": 0.0, "log2fc_c2": 0.0, "z_loading": 2.5},
    "CONSISTENCY_GENE": {"baseline": 350.0, "sigma": 0.18, "log2fc_c1": 1.2, "log2fc_c2": 0.65, "z_loading": 0.0},
    "GHOST_REPLICATOR": {"baseline": 300.0, "sigma": 0.28, "log2fc_c1": 0.42, "log2fc_c2": 0.42, "z_loading": 0.8},
    "REAL_HETEROGENEITY_GENE": {"baseline": 300.0, "sigma": 0.22, "log2fc_c1": 0.8, "log2fc_c2": -0.8, "z_loading": 0.0},
    "VARIANCE_DECOY_GENE": {"baseline": 500.0, "sigma": 0.25, "log2fc_c1": 1.0, "log2fc_c2": 0.9, "z_loading": 0.0},
    "NEAR_MISS_GENE": {"baseline": 500.0, "sigma": 0.25, "log2fc_c1": 0.8, "log2fc_c2": 0.7, "z_loading": 0.0},
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
    "VARIANCE_DECOY_GENE": 119,
    "NEAR_MISS_GENE": 34,
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
    influential_ids = consistency_gene_influential_samples(metadata)

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
            if label == "CONSISTENCY_GENE" and in_cohort2 and is_treated and sid in influential_ids:
                log2fc = log2fc + INFLUENTIAL_EXTRA_LOG2FC
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


def cohort_symmetric_worst_case_loo_p(
    counts_df: pd.DataFrame, metadata: pd.DataFrame, cohort: str, gene: str
) -> float:
    """Robustness check: within one cohort, drop each individual sample once
    -- control or treated, not only treated -- refit the moderated-variance
    prior (fit_ebayes_prior) on that reduced sample set, recompute the
    gene's moderated p-value against the remaining samples, and return the
    WORST (largest) p-value seen across all single-sample drops.

    Both the symmetry (any sample, not only treated) and the refit (the
    shrinkage prior is re-estimated on the actual reduced dataset being
    analyzed, not reused from the full-sample fit) are required for this to
    be a genuine robustness claim: a candidate's apparent evidence should
    not depend on any one sample being present, and a robustness check that
    only ever removes treated samples, or that reuses a stale full-sample
    moderation estimate while claiming to test a smaller one, is not
    actually testing what it claims to. This reuses the same
    NOMINAL_P_THRESHOLD already used for the ordinary significance check --
    a candidate's cohort-level significance has to survive removing its
    single most favorable sample, not clear a second, independently
    invented cutoff.
    """
    ids = metadata.loc[metadata["cohort"] == cohort, "sample_id"].tolist()
    condition = metadata.set_index("sample_id").loc[ids, "condition"]

    worst_p = 0.0
    for drop in ids:
        remaining = [sid for sid in ids if sid != drop]
        control_ids = [sid for sid in remaining if condition[sid] == "control"]
        treated_ids = [sid for sid in remaining if condition[sid] == "treated"]
        n1, n2 = len(control_ids), len(treated_ids)
        df_g = n1 + n2 - 2

        log2cpm = pipeline_stats.compute_log2_cpm(counts_df[remaining])
        var_c = log2cpm[control_ids].var(axis=1, ddof=1)
        var_t = log2cpm[treated_ids].var(axis=1, ddof=1)
        pooled_var = ((n1 - 1) * var_c + (n2 - 1) * var_t) / df_g

        d0, s0_sq = fit_ebayes_prior(pooled_var.to_numpy(), df_g)
        if np.isinf(d0):
            mod_var = s0_sq
            mod_df = 1e6
        else:
            mod_var = (d0 * s0_sq + df_g * float(pooled_var[gene])) / (d0 + df_g)
            mod_df = d0 + df_g

        mean_diff = float(log2cpm.loc[gene, treated_ids].mean() - log2cpm.loc[gene, control_ids].mean())
        se = float(np.sqrt(mod_var * (1.0 / n1 + 1.0 / n2)))

        from scipy import stats as scipy_stats

        t_mod = mean_diff / se
        p = float(2.0 * scipy_stats.t.sf(abs(t_mod), df=mod_df))
        worst_p = max(worst_p, p)
    return worst_p


def fit_ebayes_prior(pooled_vars: np.ndarray, df_g: float) -> tuple[float, float]:
    """Method-of-moments fit of the empirical-Bayes prior degrees of
    freedom (d0) and prior variance (s0^2) for the moderated-variance model
    behind limma's eBayes and (in substance) DESeq2's/edgeR's own
    dispersion-shrinkage machinery (Smyth 2004): each gene's own variance
    estimate is treated as a noisy draw around a shared, panel-wide typical
    variance, and the strength of that shrinkage is itself estimated from
    how much the per-gene variances actually vary across the panel -- not a
    fixed, hand-chosen constant. This is the same closed-form moment
    estimator limma::fitFDist implements; verified independently against
    real limma-voom, edgeR's quasi-likelihood pipeline, and DESeq2 on this
    exact dataset (see task/README.md) -- all three, each estimating their
    own moderation strength from the data, agree with this estimator's
    conclusion.
    """
    from scipy import special, optimize

    pooled_vars = np.asarray(pooled_vars, dtype=float)
    pooled_vars = pooled_vars[pooled_vars > 0]
    z = np.log(pooled_vars)
    e = z - (special.digamma(df_g / 2.0) - np.log(df_g / 2.0))
    mean_e = float(e.mean())
    var_e = float(e.var(ddof=1))
    target = var_e - float(special.polygamma(1, df_g / 2.0))

    if target <= 0:
        return float("inf"), float(np.exp(mean_e))

    def f(x: float) -> float:
        return float(special.polygamma(1, x)) - target

    x = optimize.brentq(f, 1e-6, 1e6, xtol=1e-10)
    d0 = 2.0 * x
    s0_sq = float(np.exp(mean_e + special.digamma(d0 / 2.0) - np.log(d0 / 2.0)))
    return d0, s0_sq


def cohort_moderated_p_value(
    counts_df: pd.DataFrame, metadata: pd.DataFrame, cohort: str, gene: str
) -> float:
    """Moderated (empirical-Bayes shrunk-variance) Welch-style p-value for
    one gene within one cohort. The prior (d0, s0^2) is fit once per cohort
    from every gene's own pooled within-group variance in that cohort (see
    fit_ebayes_prior) -- self-calibrating to this dataset, not to any
    hand-picked constant or to knowledge of which gene is being evaluated.
    """
    from scipy import stats as scipy_stats

    ids = metadata.loc[metadata["cohort"] == cohort, "sample_id"].tolist()
    condition = metadata.set_index("sample_id").loc[ids, "condition"]
    control_ids = [sid for sid in ids if condition[sid] == "control"]
    treated_ids = [sid for sid in ids if condition[sid] == "treated"]
    n1, n2 = len(control_ids), len(treated_ids)
    df_g = n1 + n2 - 2

    log2cpm = pipeline_stats.compute_log2_cpm(counts_df[ids])
    var_c = log2cpm[control_ids].var(axis=1, ddof=1)
    var_t = log2cpm[treated_ids].var(axis=1, ddof=1)
    pooled_var = ((n1 - 1) * var_c + (n2 - 1) * var_t) / df_g

    d0, s0_sq = fit_ebayes_prior(pooled_var.to_numpy(), df_g)
    if np.isinf(d0):
        mod_var = s0_sq
        mod_df = 1e6
    else:
        mod_var = (d0 * s0_sq + df_g * float(pooled_var[gene])) / (d0 + df_g)
        mod_df = d0 + df_g

    mean_diff = float(log2cpm.loc[gene, treated_ids].mean() - log2cpm.loc[gene, control_ids].mean())
    se = float(np.sqrt(mod_var * (1.0 / n1 + 1.0 / n2)))
    t_mod = mean_diff / se
    return float(2.0 * scipy_stats.t.sf(abs(t_mod), df=mod_df))


def moderated_combined_p(counts_df: pd.DataFrame, metadata: pd.DataFrame, gene: str) -> float:
    """Fisher-combine the two cohorts' moderated p-values for one gene."""
    p1 = cohort_moderated_p_value(counts_df, metadata, "cohort1", gene)
    p2 = cohort_moderated_p_value(counts_df, metadata, "cohort2", gene)
    return fisher_combined_p(p1, p2)


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
    4. That significance must also be ROBUST: it has to survive removing
       any single sample from that cohort -- control or treated, not only
       treated -- with the moderation prior refit on the reduced sample
       set each time (see cohort_symmetric_worst_case_loo_p), not just
       hold with all samples included or with a stale full-sample
       moderation estimate reused. Round 6 introduced this as a raw,
       treated-only check to catch CONSISTENCY_GENE's fragile replication
       (a cohort2 result carried disproportionately by two treated
       samples); round 8 found that check too narrow to be a genuine
       robustness claim -- both CONSISTENCY_GENE and (originally)
       TRUE_GENE itself can look "robust" under a treated-only, non-refit
       version while TRUE_GENE's real cohort2 evidence was still fragile
       to losing a single control sample. The symmetric, refit version is
       what actually separates a genuinely broadly-supported result from
       one that only looks that way under a narrower check.
    4b. This means more than one candidate can now clear the robustness
       bar (CONSISTENCY_GENE among them) without being the final answer --
       that is expected and correct; step 5's moderated combined-evidence
       comparison is what does the final discriminating.
    5. Among genes that clear that bar, prefer the one with the stronger
       combined evidence -- but "stronger" is evaluated on MODERATED
       per-cohort p-values (empirical-Bayes shrunk variance, prior
       strength self-estimated from this dataset's own per-gene variances,
       see cohort_moderated_p_value/fit_ebayes_prior), not raw/unmoderated
       ones. Round 7's distinction: a candidate can look stronger on raw
       evidence purely because its own small-sample variance estimate
       happened, by chance, to come out unusually (and unreliably) small
       in one cohort; moderated variance corrects for that before the
       final combined-evidence comparison is made.
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
        if not (same_sign and both_nominal):
            continue
        c1_robust = cohort_symmetric_worst_case_loo_p(counts_df, metadata, "cohort1", gene) < NOMINAL_P_THRESHOLD
        c2_robust = cohort_symmetric_worst_case_loo_p(counts_df, metadata, "cohort2", gene) < NOMINAL_P_THRESHOLD
        if c1_robust and c2_robust:
            combined_p = moderated_combined_p(counts_df, metadata, gene)
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
        both_robust = both_nominal and (
            cohort_symmetric_worst_case_loo_p(counts_df, metadata, "cohort1", gene) < NOMINAL_P_THRESHOLD
            and cohort_symmetric_worst_case_loo_p(counts_df, metadata, "cohort2", gene) < NOMINAL_P_THRESHOLD
        )
        if same_sign and both_robust:
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


def write_prior_report(public_dir: Path) -> None:
    """A prior-phase report the agent will find alongside the pipeline.

    Deliberately does NOT name a gene or give gene-identifying numbers --
    an earlier design named the correct gene outright (with only the
    numbers made stale), which review caught as a genuine anti-cheat leak:
    naming the gene at all hands over the task's central judgment call
    regardless of whether the accompanying numbers are wrong. This version
    establishes the same "don't trust historical claims blindly, reproduce
    independently" narrative purely through methodology caveats, with
    nothing in the file that narrows down which of the 300 genes is the
    intended answer.
    """
    text = (
        "# Prior pilot analysis (archival)\n\n"
        "An earlier, single-site pilot (n=3 control / n=3 treated, no "
        "confirmatory cohort) reported a candidate transcriptional "
        "response to the compound. The pilot's sample size was too small "
        "to support a reliable conclusion on its own, and it predates the "
        "confirmatory cohort2 run entirely.\n\n"
        "The pilot's underlying data and summary tables were not retained "
        "alongside this repository. Any claim about which gene responded, "
        "or by how much, needs to be established independently from the "
        "current two-cohort dataset -- the existence of this pilot is not "
        "itself evidence for or against any specific candidate, and its "
        "prior conclusion should not be assumed to be correct.\n"
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
    write_prior_report(public_dir)

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
