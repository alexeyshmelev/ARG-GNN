# Four-parent biological paths execution pilot v1

**Status: executable preparation implemented, simulation unexecuted, root review/freeze pending.** The proposal files remain unchanged. This stage prepares two TRAIN and two VAL2 Mb parents, with TEST0. It launches no Relate/model/GPU job and establishes no ancestry result.

## Exact model and panel

configs/biological_paths_execution_pilot_v1.json materializes the msprime demography, all280 ordered haplotype records and four new parent/seed records. Model: author-source11 populations with sizes50,000/20,000/50,000/3,000/2,000/15,000/20,000/3,000/5,000/15,000/15,000 in present_bronze,Yam,Neo,EHG,WHG,NE,BAA,CHG,Ana,WA,trunk order. Fixed backward admixtures are Bronze166→Neo/Yam0.5/0.5; Neo259→WHG/Ana0.25/0.75; Yam177→CHG/EHG0.5/0.5; BAA169→CHG/Ana0.25/0.75. Explicit merges are WHG/EHG600→NE, CHG/Ana800→WA, NE/WA1500→trunk, plus census350.

The140 diploids comprise48 pure-source reference diploids/96 haplotypes,32 mixed reference-only diploids/64 haplotypes and60 query diploids/120 haplotypes. This differs from the author's476 diagnostic diploids. Query cohorts are present32 diploids at0, Bronze8 at135, BAA4 at155, Neo8 at200 and Yam8 at170. Reference dates are the exact sorted-age quantiles already specified in proposal v1, materialized with original author line numbers from the pinned example file, SHA2563cadb16abbc0a1913eadb183d324eadc26c035d61e5116cd58675f52a942b7de. Sample order follows the author simulation cohort order: present,Bronze,BAA,Neo,Yam,WHG,EHG,Ana,CHG; reference roles precede queries within a cohort. Every diploid supplies two consecutive haplotypes.

Reduced dates do not redefine pulse times. Source-derived sizes/events are explicit in the JSON; population activity is validated against every exact sample date. Older-than-census pure references are allowed. Only queries require a census ancestor at350.

## Seeds and metadata-only collision check

Seed domain is arggnn|biological_paths_execution_pilot_v1, master20261010, split/train-or-val, parent000-or001, and separate ancestry/mutations purposes. SHA256 first8 bytes are mapped into1..2^32−1. Eight distinct new seeds:

| Parent | Ancestry | Mutations |
|---|---:|---:|
| train000 | 2226234658 | 1355992149 |
| train001 | 223852707 | 1061155424 |
| val000 | 3389613403 | 2419496571 |
| val001 | 3811974427 | 3318312349 |

The config records the exact reviewed metadata list/hashes. The initial read inspected1869 existing config/manifest/metadata JSON files, including the frozen earlier parent registry/prior inventory, yielding678 distinct explicit seed values; zero proposed collisions. No tree, genotype or target array was opened. This is a listed-history check, not a claim about future/unlisted seeds. A changed reviewed registry requires root review rather than silently replacing a parent or reseeding.

## Observation/truth separation

Simulation uses existing main-runtime msprime1.3.4/tskit1.0.3/NumPy2.4.6/Python3.11: Hudson ancestry,2,000,000 bases, uniform recombination1e-8, mutation1.25e-8, discrete genomes, explicit JC69 mutation model and no recorded migrations. These explicit defaults are a source-derived execution condition, not the published200 Mb/empirical-map numerical reproduction.

Raw census-bearing ancestry and six-path targets are saved only under each parent's truth namespace. Paths1/2/3/4 are Ana_nonBAA/CHG_nonBAA/WHG/EHG; BAA Ana/CHG are paths5/6. C4 orderANA,CHG,WHG,EHG is derived by mapping paths1+5,2+6,3,4. Both full raw truth intervals and fixed100 bp midpoint targets are retained, without purity deletion.

Before mutation observations, simplify all sampled leaves with keep_unary=False, merging census-only boundaries. Clear internal populations, individuals, migrations, provenance, metadata and extra flags. Reject surviving census nodes. Draw mutations on that cleaned genealogy; artificial census nodes cannot introduce observation segmentation or mutation-placement differences. The resulting cleaned_oracle.trees is explicitly an oracle observation, **not the primary inferred graph**. Actual primary trees await root's Relate execution using genotype assets only. No packed F47/F48 inferred input or ancestry performance is fabricated.

Future output layout under data/biological_paths_execution_pilot_v1:

- Per-parent truth/census_ancestry.trees, full_truth_intervals.npz and targets_at_fixed_midpoints.npz.
- Per-parent observed/cleaned_oracle.trees, genotypes.vcf, ordered sample_manifest.json, sample_order.txt, haplotype sample_ages.generations, population_labels.txt, site_coordinate_manifest.json, uniform_recombination_map.txt and fixed scoring_coordinates.npz.
- Root manifest.json with parent identities/seeds, file hashes, completed/failed status and explicitly pending Relate/model work.

VCF sites are integer zero-based mutation positions+1, phased diploid sample names are parent-qualified, and the uniform map is shifted consistently. Contig name22 does not claim real GRCh37 coordinates or an empirical map. Keep all raw observed variants; record later Relate SNP attrition. No truth-derived coordinate or interval selection enters observations.

An existing output directory is rejected. Partial failures are preserved with their parent/error in the manifest; no automatic replacement, retry, window resampling or score filtering occurs. Root remains responsible for project storage/resource admission before launch.

## Limited validation performed

The default CLI validates the fixed model,280-row template,96 pure refs/120 queries, runtime and metadata seed collision check without simulation. One narrow handmade280-leaf counterfactual passed: changing a hidden census population changes six-path/C4 targets[2,120], while the cleaned observation is identical, its genotype matrix is preserved and two census-only truth intervals become one observed tree. This invokes neither sim_ancestry nor sim_mutations. It is not a new synthetic suite or biological evaluation.

Read-only checks:

```sh
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python scripts/prepare_biological_paths_pilot.py
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python scripts/prepare_biological_paths_pilot.py --check-counterfactual
```

Root-only execution after reviewing/freezing all four implementation/config files:

```sh
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python scripts/prepare_biological_paths_pilot.py --execute --expected-config-sha256 REVIEWED_SHA256
```

The explicit config hash is required. Root should then run the genuine genotype→Relate chain and actual native baseline fits. Final biological holdout sizes remain unset and belong to later development-informed power/freeze.
