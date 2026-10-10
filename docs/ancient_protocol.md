# Four-source ancient-European oracle protocol

## Scope and exact provenance

This is a separate benchmark family named `ancient-europe-stdpopsim-0.3.0-oracle`. It uses the installed stdpopsim **0.3.0** `HomSap` demographic model **AncientEurope_4A21**, with its demographic parameters unchanged. The additional census event supplies external truth and is removed from model inputs. It is not a reproduction of ARGMix or Nature numerical results, not an inferred-ARG benchmark, and not evidence of accuracy on real ancient genomes.

The source is https://github.com/popsim-consortium/stdpopsim/blob/0.3.0/stdpopsim/catalog/HomSap/demographic_models.py . The installed model cites https://doi.org/10.1101/2022.05.04.490594 . stdpopsim package metadata identifies its licence as GNU GPLv3+. The runtime dependency and model citation are recorded. No AncestralPaths source code is copied. Public AncestralPaths, Nature supplement and stdpopsim configurations differ, as detailed in `benchmark_plan.md`.

The canonical JSON representation of the installed unmodified demographic model has SHA256 `37e53cbed70dcbbf4e67cbe33226b4ed58d140d2f812e48333bf260362be4d60`. Every replicate stores the complete parameter representation, its hash, stdpopsim version, model identifier, generation time, source URL and citations. The generator refuses another stdpopsim version. Seed streams include a benchmark-family namespace separate from the three-source toy generator.

| Population or event | Pinned value |
|---|---|
| OOA, NE, WA | Effective size 5,000 |
| CHG, WHG, EHG | Effective size 10,000 |
| ANA, NEO | Effective size 50,000 |
| YAM | Effective size 5,000 |
| Bronze/present population | Growth rate 0.067, size 50,000 at generation 140 |
| Bronze admixture | Generation 140, YAM/NEO 50/50 |
| YAM admixture | Generation 180, EHG/CHG 50/50 |
| NEO admixture | Generation 200, WHG/ANA 25/75 |
| WHG/EHG divergence | Generation 600 |
| CHG/ANA divergence | Generation 800 |
| NE/WA divergence | Generation 1,500 |
| Generation time | 29 years |
| Census | Generation 350, external label augmentation |

The source class order is **ANA, CHG, WHG, EHG**, with indices 0, 1, 2, 3. The demographic admixture proportions imply population-average path proportions 0.375, 0.25, 0.125, 0.25 before finite-sample variation. Do not enforce those proportions by selecting replicates or balancing away difficult ancestry tracts.

## Observed cohorts and valid times

The first experiment has pure-source references only. ANA references are sampled at generation 260, CHG at 300, WHG and EHG at 250. These dates are drawn directly from catalog defaults and are within each population's active era. Query diploids are sampled from Bronze **explicitly at generation 0**. The catalog Bronze default is generation 135 and must not be used for present-day queries.

Census 350 lies strictly within the common source-separation interval (200,600). All three admixture events have occurred backwards in time before the census, while none of the four source populations has merged yet. Chosen references are younger than the census and within their source population lifetime. Configuration validation rejects a census at an admixture/divergence boundary, an invalid source sampling date, and a reference older than the declared census under this protocol. The first generator does not add Neolithic or Yamnaya admixed reference cohorts.

Reference cohort identities and radiocarbon-like sampling ages are observed covariates. Internal population identities and census ancestry are hidden. A query's observed present-day cohort and sampling age do not reveal its local source ancestry.

## Truth extraction and removal of the census artifact

Simulate with `msprime.sim_ancestry(model="hudson")` on a copy of the pinned demography with an added census event, then `msprime.sim_mutations` with an independent seed. Uniform recombination is an explicit pilot simplification, not a claim of chromosome-22 map reproduction. Default mutation rate is 1.25e-8.

At each fixed genomic midpoint, traverse a query's true tree to the ancestor flagged `msprime.NODE_IS_CEN_EVENT`. Its population at generation 350 supplies the four-source class. Missing census ancestry, a wrong census time or a population outside the four sources is an error. Grid positions are chosen independently of marginal-tree segmentation. Full-resolution switch counts are also measured over the raw census-bearing genealogy.

Before observation construction, simplify the tree sequence over every sampled leaf with `keep_unary=False` and obtain the raw-to-observed node mapping. This collapses all census-only nodes and merges boundaries that existed only because census node identity changed. Retaining these boundaries would reveal ancestry segmentation even if population features were blanked. Any branching census node surviving simplification is rejected. All node populations become NULL, individual associations are removed, all node/edge/site/mutation metadata is dropped, all extra flags are cleared, and population, individual, migration and provenance tables are removed. Simulated allele states, sample ages, coalescent times, native topology and sample identities remain available. Genotype matrices and sampled-pair TMRCAs are verified to be preserved.

The model sees only the cleaned `GraphSequence`: four reference source one-hots, query/leaf indicators, fixed-scale log age and parent branch, native edges, stable sampled leaf routing IDs, and descendant reference landmarks. Queries and internal nodes have zero source one-hots. No census nodes, census markers, census-derived interval segmentation, internal population IDs, metadata or ground truth enter the sequence. K=4 and F=8 require no change to the existing graph contract.

A unit fixture has two raw trees distinguished only by census identity. The external labels change from ANA to CHG inside the chromosome, while the cleaned observation has one tree. This ensures that labeling neither assumes tree homogeneity nor retains census segmentation as a shortcut. Additional tests alter hidden census populations without changing observed topology, preserve mutations and pairwise TMRCAs, and check dates and deterministic seed streams.

## API and storage

`AncientSimulationConfig` contains sequence length, uniform recombination and mutation rates, reference/query diploid counts, four reference dates, census time, grid count, landmarks per source, a graph-expansion limit, and the pinned version.

`simulate_ancient_replicate(config, master_seed=31, split="train", replicate=0)` returns an `AncientDataset` compatible with `SimulatedDataset`. Its fields are `sequence`, external `labels[P,Q]`, external nearest-TMRCA `baseline_probabilities[P,Q,4]`, raw census-bearing `tree_sequence`, `metadata`, and cleaned `observed_tree_sequence`.

`generate_ancient_dataset(directory,config,master_seed=31,train_replicates=3,val_replicates=1,test_replicates=2)` has the same manifest structure as the toy generator. `save_ancient_replicate` writes `tensors.pt.gz`, `truth.trees.gz`, `observed.trees.gz` and `metadata.json`. Existing `load_replicate` loads compact tensors without loading the audit genealogies. Existing fixed-window helpers work unchanged, retaining all cleaned tree boundaries, global chromosome coordinates and parent split identity.

Metadata includes haplotype rows keyed by `(sample_id,haplotype)`, raw and observed node IDs, diploid sample identity, sampling age, query/reference role and observed reference class. Both query haplotypes are excluded from references. A mapping keeps external label columns aligned after simplification. Model demography and truth labels stay in audit metadata or target files, not input tensors.

Tiny diagnostic defaults are 20 kb, three reference diploids per source, four query diploids, 128 midpoint grid positions and three reference landmarks per source. Those defaults test correctness and the materially different ancient reference domain. They do not provide enough independent ancestry tracts for an ordering claim. The primary agent must freeze a larger protocol after resource measurement, without selecting replicates by learned-model performance. All train/validation/test assignments are at independent parent-replicate level, before any tiling.

## Paired Relate inference track required next

The inferred-tree track must use genotypes from the **same** simulated parent replicate and the **same** fixed-coordinate external labels. The true-tree and inferred-tree experiments are separate reported conditions. Never construct labels from inferred-tree majority ancestry or discard intervals according to true purity.

1. Freeze the physical coordinate convention. tskit coordinates are zero-based and VCF positions are one-based integers. Export a site mapping `(original_site_id, original_position, VCF_position)` and chromosome length rather than assuming default rounding can be reversed. Coordinate conversion must be explicit when querying Relate intervals.
2. Export a phased diploid VCF with stable sample IDs, the observed reference/cohort table and diploid/haplotype order manifest. Preserve source/query roles and sample ages. Generate Relate `.haps`/`.sample`, record any preprocessing reorder, and derive the per-haplotype `--sample_ages` file in exactly the `.sample` order. Match both haplotypes by `(sample_id,haplotype)` after `ConvertToTreeSequence`. Do not assume raw simulated node IDs survive inference.
3. Run a pinned Relate version with recorded mutation rate, recombination map, seed, effective population-size or coalescent-rate model, masks, ancestral polarization and preprocessing choices. Map/build consistency is mandatory. Begin with a clearly disclosed synthetic map and polarization scheme, then add a realistic chromosome map. Record SNP attrition, inference failures and actual genomic coverage.
4. Pack inferred topology and estimated ages with the exact same observed reference labels and fixed scoring coordinates. Supply model features from inferred trees only. Raw census-bearing truth stays external. If inference drops samples or does not cover chromosome ends, report coverage and use a prespecified mask rather than silently changing queries or labels. The common target grid is unchanged within reported coverage.
5. Report sampled-grid accuracy for continuity with the pilot and add physical-span accuracy, class metrics, dosage errors, calibration, ancestry tract/boundary metrics and paired uncertainty over independent parent replicates. An inferred tree can cover multiple true ancestry classes. Keep mixed intervals in evaluation and report its constant-per-tree resolution limit.

## Error and phasing perturbations

The generator currently applies no genotype errors, imputation model or phase errors. Do not interpret oracle results as robustness to those errors.

For the next track, draw independent prespecified error seeds and corrupt **observed genotypes** after truth construction. A genotype allele error changes the observation but leaves the ancestral truth of the physical chromosome unchanged. Use fixed rates including the ARGMix paper's .001 only as one documented condition, alongside zero-error control. Reference and query error policies must be explicit. Run tree inference separately for each corrupted observation and retain the original raw truth.

Phase-switch errors are permutations of the two haplotypes of a diploid individual along genomic intervals. Save the true-to-observed haplotype permutation at every position. For haplotype ancestry metrics, align labels to the observed haplotype using that known permutation, or define and disclose an original-haplotype recovery task. Diploid ancestry-dose metrics are invariant to such swaps and should be primary when phase identity is ambiguous. Do not score phase-corrupted leaf IDs against unpermuted haplotype labels by accident. Never supply the phase-truth permutation as a model feature.

Missingness and imputation must similarly operate on genotype observations and record their own masks/seeds. Add map and demographic misspecification as held-out regimes, not validation feedback. Require ancient source-proxy drift, reference imbalance, longer chromosomes, alternative maps and admixed-reference experiments before broader applicability claims.

## Execution constraints

All files remain under `/workspace-SR003.nfs2/ARG-GNN`. This module installs nothing and launches no GPU work. Main agent owns bulk generation, inference, training and commits. Tiny diagnostics and correctness tests run with CUDA disabled. Observe the 450 GB warning threshold and remain below 500 GB including environments and temporary files. At most two project GPUs may run and only on eligible physical GPUs 2–5 through the project launcher.

## Tiny diagnostic recorded on 2026-10-10

One in-memory CPU-only default instance, master seed 31, train role, replicate 0, used ancestry seed 2527869820 and mutation seed 2501666375. It generated 20 raw and 20 cleaned trees, 31 census nodes, zero retained census nodes, 1,260 packed observed nodes and 21 mutation sites. Fixed-grid class counts in ANA/CHG/WHG/EHG order were `[384,256,256,128]`. Nearest-reference TMRCA grid accuracy was 31.25%. Packed tensor storage was 187,648 bytes. In-memory generation took about 0.123 seconds, not a throughput guarantee. All eight query haplotypes had zero true ancestry switches, so this tiny diagnostic is not evidence about ordering. It was not used for a learned-model comparison, no inference was run, and no bulk data were generated. Reserve fresh final evaluation seeds. Seven focused ancient-generator CPU tests passed.
