# Kinase X hit-to-lead: nominate the compound to progress

A hit-to-lead chemistry campaign against "Kinase X" has produced a series of
28 analogs (`C001`-`C028`) from a single scaffold. Each analog has been
profiled in vitro against the primary target and a structurally related
off-target kinase paralog ("Kinase X-related", a selectivity liability if
bound), plus a standard panel of physicochemical and ADME properties. The
project team needs to nominate a single analog to progress into lead
optimization.

Nominate the compound (`compound_id`) that a hit-to-lead chemistry team
should progress as the lead, based on the evidence in the data below.

## Data

The task inputs are available under `/workspace/data`:

- `assay_results.csv`: raw potency assay measurements. Columns:
  `compound_id`, `assay` (`target_ic50_nm` or `offtarget_ic50_nm`),
  `replicate` (1-3), `ic50_nm` (IC50 in nanomolar for that replicate),
  `qc_flag` (`pass` or `fail`). Each `(compound_id, assay)` pair has 3
  replicate measurements; some individual replicates are flagged `fail` by
  quality control (assay artifact) and must be excluded before averaging.
  Never include a `fail`-flagged replicate in any calculation.
- `adme_properties.csv`: one row per compound. Columns: `compound_id`,
  `clogp` (calculated lipophilicity), `mw` (molecular weight, Da),
  `heavy_atom_count`, `tpsa` (topological polar surface area, Å²),
  `papp_1e6cm_s` (Caco-2 apparent permeability, ×10⁻⁶ cm/s; higher is
  better), `clint_ul_min_mg` (human liver microsomal intrinsic clearance,
  µL/min/mg; higher means less metabolically stable, i.e. worse),
  `solubility_um` (kinetic aqueous solubility, µM; higher is better).

Both files share `compound_id` as the join key.

## Deliverable 1: `/workspace/output/compound_metrics.csv`

One row per compound — 28 rows total, each `compound_id` present exactly
once. Columns:

- `compound_id`
- `pic50_target`: pIC50 against the primary target
  (`pIC50 = 9 - log10(IC50_nM)`), averaged across only the `pass`-flagged
  replicates of the `target_ic50_nm` assay for that compound.
- `pic50_offtarget`: pIC50 against the off-target paralog, computed the same
  way from the `offtarget_ic50_nm` assay's `pass`-flagged replicates.
- `selectivity_index`: `pic50_target - pic50_offtarget` (selectivity margin
  in log units; e.g. 1.0 means 10-fold selective for the primary target).
- `lle`: lipophilic ligand efficiency, `pic50_target - clogp`.
- `le`: ligand efficiency, `1.4 * pic50_target / heavy_atom_count`.
- `developability_pass`: `true` if and only if `solubility_um >= 20` AND
  `clint_ul_min_mg <= 45` AND `papp_1e6cm_s >= 8`; `false` otherwise.

## Deliverable 2: `/workspace/output/result.json`

A JSON object with the following fields:

- `naive_top_potency_id`: the `compound_id` with the single highest
  `pic50_target` across all 28 compounds.
- `n_developability_pass`: the count of compounds with
  `developability_pass == true`.
- `nominated_lead_id`: the `compound_id` a hit-to-lead chemistry team should
  actually progress into lead optimization, using `compound_metrics.csv`
  and the other `result.json` fields above as evidence. The most potent
  compound by raw `pic50_target` is not necessarily the right nomination.

Do not hardcode expected output values. All quantities must be genuinely
derived from the visible data. Read `DATA_DIR`/`OUTPUT_DIR` from the
environment rather than assuming the paths above.
