# Solution process

## Mechanism

The dataset models a single-scaffold hit-to-lead series against a fictional
kinase target ("Kinase X"): 24 generic background compounds plus 6 compounds
(`C101`-`C106`) whose latent variables are deliberately placed to realize six
competing, individually-plausible medicinal-chemistry narratives. Three
forces drive each compound's measured properties, all latent and never shown
to the agent directly:

1. **Genuine, logP-independent SAR** (`intrinsic_fit`): how well a
   substituent's geometry fits the primary target's binding pocket. Real
   signal, drives on-target potency only.
2. **Off-target idiosyncrasy** (`offtarget_idio`): paralog-binding events
   that are not pure bulk-lipophilicity effects. This keeps the
   clogP/off-target relationship real (compounds do get more promiscuous as
   they get more lipophilic, on average) but far short of deterministic
   (`r(clogp, pic50_offtarget) ~= 0.57-0.72` across runs) -- a scatter, not a
   sortable column.
3. **Metabolic idiosyncrasy** (`metab_idio`): a labile-group liability
   unrelated to bulk lipophilicity. A low-clogP compound is not automatically
   metabolism-clean, and a moderate-clogP compound is not automatically
   disqualified -- clint failure has to be checked per compound, not inferred
   from clogP.

`clogp` itself (shown) contributes to on-target potency (filling a
hydrophobic pocket is a real, if partial, driver) and, on average, to
off-target potency and metabolic clearance -- the classic lipophilicity/ADME
trade-off -- but the two idiosyncrasy latents mean no single visible column
determines any outcome by itself.

## The six archetypes (plus one emergent background near-competitor)

| id | role | pIC50 | SI | LLE | dev_pass | Why it's initially attractive | Why it's not the answer |
|---|---|---:|---:|---:|:---:|---|---|
| C101 (A) | naive potency winner | 8.66 | 0.49 | 3.66 | **fail** | Highest raw potency in the dataset | Bought almost entirely with clogP=5.0; fails clint |
| C104 (D) | raw LLE winner (ungated) | 8.38 | 1.35 | **5.83** | **fail** | Best LLE of any compound, near-top potency | Independent metabolic liability fails clint despite low-moderate clogP=2.55 |
| C106 (F) | best LLE *among developability-passers* | 8.05 | **-0.49** | 5.85 | pass | Passes every disclosed threshold; best LLE of any passer | `selectivity_index` is negative -- more potent against the off-target paralog than the primary target. Plainly non-selective, not a borderline call |
| C103 (C) | selectivity winner | 7.51 | **1.94** | 4.21 | pass | Best selectivity in the whole series | LLE is mediocre -- the selectivity came from a favorable off-target-idiosyncrasy draw, not from efficient, genuine SAR |
| C102 (B) | cleanest properties | 7.40 | 0.99 | 5.40 | pass | Lowest clogP, cleanest ADME margins | Simply weaker than the intended lead on potency, LLE, and selectivity -- a defensible but suboptimal conservative pick |
| **C105 (E)** | **intended lead** | 8.48 | 1.87 | **5.73** | pass | Best LLE among compounds that are BOTH developable AND genuinely selective | (this is the answer) |

`C105` does **not** dominate the field: `C101` beats it on raw potency,
`C103` beats it on selectivity, and `C104`/`C106` both beat it on raw LLE.
It wins only once developability, selectivity, and efficiency are all
required simultaneously -- exactly the integration a real hit-to-lead
decision requires.

## Why the mechanical shortcut fails (validated, not assumed)

`instruction.md` discloses exact formulas for every `compound_metrics.csv`
column, including the three-part `developability_pass` threshold. It
discloses **no** selectivity threshold and no combination rule for
`nominated_lead_id`. This was verified against the actual mechanical
shortcut a lazy or non-domain-expert agent might run:

> filter to `developability_pass == true` (the one disclosed hard rule),
> discard failures, sort survivors by `lle`, take the top row.

That workflow returns **C106**, not C105 -- because C106 passes
developability and has higher LLE than C105, and nothing in the disclosed
contract tells the agent to also check selectivity. Recovering C105 requires
recognizing, from the visible `selectivity_index` column itself, that a
negative value (more potent against the paralog than the target) is
disqualifying -- a judgment that does not depend on knowing any exact
numeric cutoff, since C106's value (-0.49) is unambiguously bad by
inspection, not a borderline case near a hidden threshold.

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
   unconditional) and `n_developability_pass`.
9. For `nominated_lead_id` (not a disclosed formula): restrict to compounds
   with `developability_pass == true` AND `selectivity_index > 0` -- a
   natural zero-point ("genuinely more potent against the target than the
   paralog"), not a tuned magic number -- then nominate the highest-`lle`
   compound among what remains.

## Validated numbers (shipped dataset, computed internally, not disclosed)

| | value |
|---|---:|
| `naive_top_potency_id` | C101 (pic50_target = 8.663, clogp = 5.00, fails developability) |
| Mechanical-shortcut (dev_pass-only + argmax lle) result | C106 -- wrong, selectivity_index = -0.494 |
| `nominated_lead_id` | C105 (pic50_target = 8.476, clogp = 2.75, LLE = 5.726) |
| Margin over 2nd-place gate-passer (C102) | 0.322 LLE units |
| Margin stability under an alternate averaging method (mean-IC50-then-log vs. mean-pIC50) | 0.3221 vs. 0.3228 -- not method-sensitive |
| `n_developability_pass` | 14 / 30 |
| clogp vs. pic50_offtarget correlation | ~0.57-0.72 (real trend, not deterministic) |

Non-domination, confirmed by direct comparison: C101 beats C105 on
`pic50_target`; C103 beats C105 on `selectivity_index`; C104 and C106 both
beat C105 on `lle`.

## Design history

An earlier version of this dataset used a single dominant confound (clogP
almost perfectly determined off-target potency, `r ~= 0.999`) with one
clear winner. A 3-agent trajectory run against that version passed 12/12 --
every agent found the answer by sorting the raw table. Two rounds of
revision followed: (1) reporting `clogp`/potency correlations as computed
`result.json` fields was removed after a trial showed an agent quoting the
reported number verbatim as its reasoning, rather than deriving the insight
itself; (2) the single-confound landscape was replaced with the six-archetype
design here after a mechanical-shortcut test showed the disclosed
developability threshold alone (no selectivity reasoning) was already
sufficient to recover the old answer. C106 was added specifically to close
that gap.
