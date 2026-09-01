# Solution process

## Mechanism

The dataset models a single-scaffold hit-to-lead series against a fictional
kinase target ("Kinase X"). Two forces drive each compound's measured
potency:

1. **Genuine, logP-independent SAR** (`intrinsic_fit`, latent, never shown
   to the agent): how well a given substituent's geometry fits the primary
   target's binding pocket. This is real medicinal chemistry signal.
2. **Nonspecific lipophilic binding** (`clogp`, shown): bulkier, more
   lipophilic substituents fill hydrophobic pockets and boost potency
   somewhat -- but this effect is *not* specific to the primary target. The
   off-target paralog's binding pocket has different geometry, so it does
   not reward `intrinsic_fit` at all -- only `clogp` drives off-target
   affinity there.

Consequently:

- Raw on-target potency (`pic50_target`) is driven by both `intrinsic_fit`
  and `clogp`, so a naive "most potent wins" analysis is dominated by
  whichever compounds happen to carry the most lipophilic substituents --
  not necessarily the ones with the best genuine pharmacophore fit.
- Off-target potency (`pic50_offtarget`) is driven almost entirely by
  `clogp` (r ~ 0.99 in the shipped dataset). Selectivity index
  (`pic50_target - pic50_offtarget`) therefore isolates the genuine SAR
  contribution, rewarding compounds that achieve potency through real fit
  rather than brute-force lipophilicity.
- Metabolic clearance rises and solubility falls with `clogp` (standard,
  well-documented lipophilicity/ADME trade-offs), so the most lipophilic,
  most raw-potent compounds also tend to fail the developability gates.

The net effect: the single most potent compound by raw `pic50_target`
achieves that potency mostly through lipophilicity, and fails the
developability gate outright. The correct lead is a compound with more
modest raw potency but a much better lipophilic ligand efficiency (LLE)
and selectivity margin, once it clears the same developability bar.

## Solve steps (mirrors `solution/solve.py`)

1. Load `assay_results.csv` and `adme_properties.csv`.
2. Filter `assay_results.csv` to `qc_flag == "pass"` rows only.
3. Convert each surviving replicate's `ic50_nm` to pIC50
   (`9 - log10(ic50_nm)`) and average within `(compound_id, assay)`.
4. Pivot to `pic50_target` / `pic50_offtarget` per compound; merge with
   `adme_properties.csv`.
5. Compute `selectivity_index = pic50_target - pic50_offtarget`,
   `lle = pic50_target - clogp`, `le = 1.4 * pic50_target / heavy_atom_count`.
6. Compute `developability_pass` from the three disclosed ADME thresholds.
7. Write `compound_metrics.csv`.
8. In `result.json`: report `naive_top_potency_id` (argmax `pic50_target`,
   unconditional), the two `clogp` correlations, and `n_developability_pass`.
9. For `nominated_lead_id`: this is the judgment call the instruction leaves
   to the agent. The reference solution restricts to compounds with
   `developability_pass == true` AND `selectivity_index >= 0.6` (a ~4-fold
   selectivity bar), then nominates the highest-`lle` compound within that
   set. Neither the 0.6 threshold nor the gate-then-argmax procedure is
   stated in `instruction.md` -- an agent has to recognize, from
   `compound_metrics.csv` and the two disclosed correlations, that raw
   potency is confounded by lipophilicity and that a developability- and
   selectivity-aware efficiency read is the right basis for the nomination.

## Validated numbers (shipped dataset)

| | value |
|---|---:|
| `naive_top_potency_id` | C022 (pic50_target = 8.699, clogp = 5.10, fails developability gate) |
| `nominated_lead_id` | C027 (pic50_target = 8.131, clogp = 2.36, LLE = 5.770) |
| `top_gate_passing_lle` margin (C027 vs. 2nd-place C028) | 0.160 |
| `n_developability_pass` | 15 / 28 |
| `n_gate_passing` (developability + selectivity >= 0.6) | 5 / 28 |
| `clogp_target_potency_r` | 0.494 |
| `clogp_offtarget_potency_r` | 0.999 |
