# Benchmark contract and provenance

## What the implemented pilot can establish

The initial benchmark is a neutral three-source pulse simulation with present-day labeled source references and unlabeled admixed query haplotypes. All model inputs use the **true simulated genealogy**. This is an oracle-input engineering pilot, not an inferred-ARG experiment and not an ancient-European local-ancestry application. Success on it supports implementation and mechanism checks only. The ancient-European demographic benchmark and Relate inference track remain required before main application claims.

`SimulationConfig` defaults to 100 kb, source/ancestral/admixed effective size 5,000, source split 3,000 generations ago, admixture pulse 50 generations ago, proportions 0.34/0.33/0.33, 8 diploid references per source, 8 diploid queries, uniform recombination 1e-8, mutation 1.25e-8, and 128 midpoint grid coordinates. These small defaults have only 0.05 expected recombination opportunities per haplotype since the pulse. They must not be cited as ordering evidence.

The prespecified ordering-pilot alternative uses 1 Mb, effective size 500, pulse 300 generations ago and split 2,000 generations ago. Start with 16 train, 4 validation and 8 test independent parent replicates, keeping the reference/query counts and 128 grid points fixed. This has 3 expected recombination opportunities per haplotype since the pulse, but actual ancestry switches must be checked from recorded truth. This is still a toy oracle experiment. Expand only after measured storage, memory, reference-baseline performance and truth-switch diagnostics are reviewed.

No sequence is selected for looking easy or showing favorable model results. Recombination-opportunity estimates are not observed ancestry-switch counts. Metadata records full-resolution true ancestry-switch counts separately from switches visible on the fixed grid. If the prespecified pilot has too few switches, report the limitation and define a new configuration before evaluating the models.

## Simulator and tensor APIs

`simulate_replicate(config, master_seed=13, split="train", replicate=0)` returns `SimulatedDataset(sequence, labels, baseline_probabilities, tree_sequence, metadata)`. `generate_dataset(directory, config, master_seed=13, train_replicates=3, val_replicates=1, test_replicates=2)` saves parents and `manifest.json`. `save_replicate(dataset, directory)` and `load_replicate(directory)` operate on compressed `tensors.pt.gz` and `metadata.json`. The true genealogy and migrations are retained as `truth.trees.gz` for auditing. Training load avoids loading that audit artifact.

`GraphSequence` contains observed tensors only:

| Field | Shape | Meaning |
|---|---|---|
| x | N,F | Reference one-hots, query/leaf indicators, fixed-scale log node age and parent branch |
| edge_index | 2,E | Bidirectional native marginal-tree edges |
| node_graph | N | Marginal-tree index |
| node_ids | N | Stable extant leaf ID, or -1 for internal nodes |
| clades | N,C | Descendant reference-landmark indicators |
| intervals | G,2 | All marginal intervals, in normalized parent-chromosome coordinates |
| query_node_ids | Q | Unique target leaf IDs and output order |
| positions | P | Fixed mid-bin genomic scoring coordinates |
| target_graph | P | Marginal tree containing each position |
| window_bounds | 2 or None | Declared global-coordinate tile extent, full [0,1] when absent |

For K=3, feature columns are source-reference one-hots 0,1,2, followed by `is_query_leaf`, `is_leaf`, `log_time`, and `log_parent_branch`. Ages and branches use `log1p(value)/log1p(10000)` with a fixed feature scale independent of simulated event parameters. Internal and query nodes have zero source one-hots. Default landmarks are four observed reference leaves per source. Stable IDs route correspondence and outputs, never learned ID embeddings.

Ground truth `labels[P,Q]` and the deterministic comparator `baseline_probabilities[P,Q,K]` remain external. `nearest_tmrca_probabilities` votes among all references tied for the smallest TMRCA, sharing ties equally. The comparator uses source labels and genealogy only, never migration truth. Metadata stores source-reference IDs/classes, landmark IDs, feature names, configuration, versions and split-specific ancestry/mutation seeds.

`slice_genomic_window(sequence,left,right)` returns the clipped sequence and retained position indexes. `split_fixed_windows(dataset,windows)` returns in-memory fixed-coordinate tiles with identically sliced labels and baseline probabilities. It preserves all tree boundaries inside each tile, clips boundary trees, and retains global parent-chromosome coordinates. Every tile inherits its parent replicate and split. Do not split tiles independently or interpret them as independent replicate evidence. Full parent truth is stored once. No overlapping-window expansion is needed.

## True labels and scientific separation

msprime `Demography.add_admixture` defines a backwards-time transition from ADM to one of SRC0/SRC1/SRC2 at the pulse. `sim_ancestry(...,model="hudson",record_migrations=True)` records the moved lineage node, interval, source, destination and event time. At a fixed coordinate, a query descends from exactly one matching ADM-to-source migration lineage. The destination yields its class. Missing or ambiguous matches are errors. Migration interval coordinates are half-open.

Labels therefore can change inside a constant-topology marginal tree. They are never obtained by first-SNP or tree-majority labeling, and mixed intervals are not removed. The tree packer never reads migration tables or internal-node population fields. A unit case has one tree and two ancestry labels across its interior boundary, ensuring the labeling procedure does not silently assume tree homogeneity. Another changes internal populations and deletes migrations while requiring identical input tensors.

The true genealogy is intentionally informative about ancestry. That oracle topology is disclosed, rather than conflated with an inferred genealogy. Mutations are generated and retained for a later inference track. This pilot does not model genotype error, missingness, phasing uncertainty, source-proxy drift, ancient sampling times, selection or continuous migration.

## Split and evaluation rules

Split independent simulated genome/ARG replicates before extracting coordinates or tiles. Ancestry and mutation seed streams are derived independently from the master seed, split role and replicate index. All models use the same genotype/tree realization, coordinates, observed reference roles and labels. Validation governs selection, calibration and stopping. Test data remain untouched until choices freeze.

Fixed-coordinate grid ancestry accuracy is a pilot metric and must be called grid accuracy. In the main benchmark, add full physical-span ancestry accuracy over the union of true and predicted boundaries, genetic-span accuracy, class precision/recall/F1, calibration, local-ancestry dosage errors, tract-length and boundary metrics, coverage, and errors near true boundaries. Use paired uncertainty over independent parent replicates. SNPs, trees, queries and tiles from one parent are correlated observations.

Do not expose internal populations, migration ancestry, census ancestry, target truth, founder donor identities or truth-derived breakpoints to any model. Leave both haplotypes of query individuals out of references. Query IDs never carry class information. Reference-only and pure-plus-admixed reference conditions must be reported separately. Negative controls include node reindexing, temporal-order disruption, correspondence removal and context-matched GNN+GRU and transformer baselines.

## Historical baselines and exact source provenance

### ARGMix

Public source: https://github.com/AI-sandbox/argmix/tree/c2481248eb0de5fda659d68f59c51e0e27becd20 . Commit date 2026-08-11. MIT licence. There are no tags/releases, and public commit history starts with an August 2026 release. The current code does not provide the April 2026 paper's exact simulation/configuration package.

Paper: https://www.biorxiv.org/content/10.64898/2026.04.05.714076v1.full.pdf . Metadata: https://api.biorxiv.org/details/biorxiv/10.64898/2026.04.05.714076 . April 7, 2026 v1, CC-BY 4.0. Methods report an ancient-European msprime demography excluding Bronze Age Anatolians, chromosome-22 recombination map, genotype error .001, Relate N30,000 and mutation1.25e-8. Train: 500,000 examples from five replicates. Validation: separate replicate/100,000. IID test: three additional replicates/100,000 each. OOD: five misspecified replicates/100,000 each. Sizes: Ana22, CHG13, EHG42, Neo159, WHG48, Yam13, post-Neolithic654 and present660 = 1,611 diploids. Paper subgraphs use 25 references, three ancestor nodes and one target, with asymmetric TMRCA-relative attention clamped at 1,000 generations.

Current forward simulator: https://github.com/AI-sandbox/argmix/blob/c2481248eb0de5fda659d68f59c51e0e27becd20/sim/scripts/forward_sim_multievent.py . It copies tracts from real phased founders with Poisson(time_gens x Morgan length) breakpoints, samples source/donor each tract, and propagates founder ancestry. It has no pedigree, drift, new mutation or true ARG. Multi-event code uses each absolute time_gens, not time gaps. This is a separate mosaic generator, not an exact reproduction of demographic coalescent simulation.

NPZ: variant_chrom/variant_pos/variant_cm[S], ancestry_names[K], population__sample_ids[D], population__hap_labels[S,2D]. Only the present ADMIX cohort enters the simulated VCF. Reference panel columns: sample_id, ancestry, age, cohort_role and optional founder_split. Roles: reference_source/reference_only/source_only/other. Founder splits train/val/test must precede simulation. Only train founders enter ARG references, but default template has no test replicate. Explicitly add a test founder pool.

Truth resolver defaults to first_snp and supports majority/min_label_purity: https://github.com/AI-sandbox/argmix/blob/c2481248eb0de5fda659d68f59c51e0e27becd20/src/truth_labels.py . Do not make these interval labels or purity exclusions the main benchmark truth. Current metrics report SNP ancestry accuracy, per-class recall-like accuracy, coverage and soft diploid-dose Pearson correlation squared: https://github.com/AI-sandbox/argmix/blob/c2481248eb0de5fda659d68f59c51e0e27becd20/src/final_test_metrics.py . Current defaults have 15 references, log2 TMRCA buckets to14 and transformer+GRU, so distinguish it from the paper per-tree architecture.

### AncestralPaths and European demographic models

Source: https://github.com/AliPearson/AncestralPaths/tree/77db3cfae100641c35218b87e1451e14f3074666 . Accepted archive: https://zenodo.org/records/8319452 . No licence file or standard GitHub licence. Zenodo says other-open. Independently implement the mathematical method with citation rather than assuming unrestricted source reuse.

Paper: https://www.nature.com/articles/s41586-023-06705-1 . Supplementary Note 3: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-023-06705-1/MediaObjects/41586_2023_6705_MOESM1_ESM.pdf . Public code, supplement and stdpopsim European models differ and must not be treated as interchangeable exact replicas.

Public simulate.py uses ten 2 Mb chromosomes, recombination1e-8 and mutation1.25e-8 with no explicit seeds. Sizes: present/Bronze50,000, Yam20,000, Neo50,000, WHG2,000, EHG3,000, Ana5,000, CHG3,000, BAA20,000, NE/WA/trunk15,000. Pulses: Bronze <- Neo/Yam50/50, Neo <- WHG/Ana25/75, Yam <- CHG/EHG50/50, BAA <- CHG/Ana25/75. Pulse times are max(cohort sample age)+1. Source splits at600/800 and basal split1500. Census350 yields ancestry paths. Example ages yield Bronze166, Neo259, Yam177, BAA169 generations. Always adds91 present diploids. Supplement reports200 Mb experiments, different train counts and BAA170.

The GNN method traverses five informative ancestors, records incremental reference-cohort descendant proportions and node ages, pads absent rows with-15, then classifies via MLP. Public training uses hardcoded sample ranges, requiring schema-based independent implementation. Relate pipeline includes ConvertFromVcf, GenerateSNPAnnotations, RelateParallel with N30,000/mu1.25e-8/sample ages, population-size re-estimation for five iterations, and ConvertToTreeSequence. Current ARGMix does not re-estimate population size by default.

stdpopsim AncientEurope_4A21: https://github.com/popsim-consortium/stdpopsim/blob/main/stdpopsim/catalog/HomSap/demographic_models.py . Its current parameters differ: YAM5,000, WHG/EHG/CHG10,000, ANA/NEO/Bronze50,000, NE/WA/OOA5,000, Bronze/YAM/NEO pulses140/180/200, exponential growth .067, generation time29. Pin whichever specification is used and give it a distinct benchmark-family name.

### ARGformer

Source: https://github.com/AI-sandbox/ARGformer/tree/f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596 . Release https://github.com/AI-sandbox/ARGformer/releases/tag/v1.0.0 , 2026-09-07, Apache2.0, no dataset/checkpoint release assets. Paper DOI https://doi.org/10.1093/bioinformatics/btag438 .

The preparation script extracts unique full leaf-to-root paths from a single tree sequence, builds a node-ID vocabulary over the whole ARG, then randomly splits paths90/10. Special tokens PAD0/MASK1/CLS2/SEP3, raw node_ID tokens follow. Dataset dictionaries contain raw_sequences or tokenized_sequences plus vocab. Unknown node IDs raise KeyError. This is graph-specific/transductive and cannot be an unmodified inductive independent-ARG local-ancestry baseline. A feature-tokenized path-transformer comparison must be called an adaptation.

## Mandatory next benchmark stages

1. Validate engineering pilot and matched TMRCA/prior/AncestralPaths-feature baselines.
2. Independently simulate a pinned ancient-European model with heterogeneous ancient sample ages, remove BAA only as an explicit documented choice, then infer Relate trees from corresponding genotypes. Keep true-tree and inferred-tree results separate.
3. Compare per-tree GNN, GNN+GRU, ARGMix and ARGMix+GRU on the same coordinates, references, data resolution and tuning budget, with the ordered-tree model and ablations.
4. Hold out complete demographic parameter regimes: admixture times/fractions, divergence/Ne, reference counts/imbalance, source-proxy drift, ghost ancestry, migration, recombination-map mismatch, and genotype/missingness/phasing/inference errors.
5. Add different chromosomes/maps and independent reference cohorts. Real-data inferred labels never constitute ground truth. Founder-mosaic robustness is a separate family.

All files, environments, caches, logs and artifacts remain inside /workspace-SR003.nfs2/ARG-GNN. Warn at450GB and stop bulk writes before500GB. Main agent owns installation, commits, pushes and GPU scheduling. At most two project GPUs, drawn only from physical2–5 after checking availability. This data module launches no GPU work.

## Recorded engineering diagnostic

On 2026-10-10, a single CPU-only instance of the prespecified ordering-pilot config (master seed13, test role, replicate0) generated225 marginal trees,28,575 packed nodes,394 mutations and41 true ancestry switches over16 query haplotypes. Full-resolution per-query switch counts were `[3,4,3,3,0,5,2,3,3,1,4,2,3,1,3,1]`, all41 visible on the128-point grid. Packed tensor storage was3,582,836 bytes. One in-memory generation took0.45seconds, which is not a throughput guarantee. Nearest-reference TMRCA grid accuracy was98.828125%. The limited improvement headroom means this diagnostic cannot establish the need for ordering. No genotype-based inference or learned-model comparison was run. The inspected seed is a development diagnostic, not an untouched final evaluation case. Nine focused CPU tests passed, including exact migration boundaries, ambiguous/missing truth rejection, hidden-annotation separation, deterministic independent seed streams, compressed storage, and recombining fixed-window preservation.
