# Biological benchmark contract v1

**Status: proposal, 2026-10-10. Not frozen, executable, completed or validated.** This is a new biological-study plan. It changes no existing engineering lock or source. Final holdout counts remain undecided pending a genuine biological execution pilot and prespecified power analysis. Only this document and configs/biological_benchmark_contract_v1.json are written here. No simulation, training, inference, install, GPU job or source-test rerun is launched.

## Question and immediate study

Does the project's whole-tree method improve local ancestry from genotype-inferred genealogies in a plausible ancient-European history, compared with actual author AncestralPaths, current-release ARGMix and native-encoder ARGformer?

The primary outcome is physical-span four-source ancestry error in present-day European queries. Bronze, Neo, Yam and BAA queries provide prespecified secondary strata. Short tracts, mixed inferred intervals, dosage, calibration and boundary localization matter alongside average accuracy. A positive result on this condition supports a declared simulation finding. General biological utility additionally needs longer/error/misspecified conditions and a public real-data concordance study.

The next action is **four new 2 Mb development parents, two TRAIN and two VAL**, with genuine genotypes→Relate inference and actual native default model fits. This is a biological execution pilot, not final training/evaluation. Its job is to measure baseline headroom, ancestry switches, variation between parents, runtime and storage. Final TEST is neither generated nor opened at this stage.

Root reports the real TensorFlow source-native Dense adapter passed a CPU qualification with 26,630 parameters, one handmade update and exact reload. That is completed engineering evidence, not biological training. Previously completed source/CPU checks need no repetition unless a concrete execution failure warrants it.

## Evidence and native reproduction boundaries

Use the already reviewed sources in docs/ancestralpaths_native_protocol_v1.md, docs/official_argmix_execution_v1.md, docs/argmix_paper_revision_search_v1.md and docs/official_argformer_execution_v1.md. Their sources and receipts stay unchanged.

| Method | Required common-task execution | Separate exact/native reproduction |
|---|---|---|
| AncestralPaths | Native 5×9 cohort-shell features and six-output source architecture; full six probabilities; disclosed C4 collapse | Author476-diploid/952-haplotype diagnostic panel is distinct from our proposed140-diploid panel. Supplement describes a CNN, 200 Mb parents and50,000 training pairs; byte-identical published/current scripts use Dense512, 2 Mb,487,500 rows and overlapping train/test parents5/6. Published numerical reproduction remains unresolved |
| ARGMix | Actual current author transformer+GRU at pinned c2481248eb0de5fda659d68f59c51e0e27becd20, native population tokens/TMRCA and exports | April paper uses25 nearest reference leaves overall+3 ancestors+query. Available release uses15 references, zero ancestors, log2 cap14 and a GRU. Bounded revision search found no April archive/config package. Do not claim exact29-node paper reproduction |
| ARGformer | Unchanged native encoder/tokenizer/pooler, within-ARG observed-only training and an explicitly downstream C4 readout | Native paper task is three-way AmericanAdmixture_4B18,30 Mb, tsinfer+tsdate versus FLARE,10,000 unique query paths and within-ARG90/10 path split. Four-source Europe/Relate is a new task. Ploidy, dates/defaults and native PCA/nearest-neighbor rule must be resolved for numerical reproduction |
| Whole-tree candidate | Freeze the actual project architecture and its observed feature/config contract before biological fitting | A generic graph Transformer, toy encoder or engineering F47/F48 adaptation cannot stand in for an author method |

Current ARGMix source preserves native integer-time rounding, key-index behavior,15-reference scope, bidirectional GRU/chunking and probability averaging. Replacing them with paper equations or richer features creates a labelled adaptation. Source padding/repeated last references do not count as distinct references. The paper's25 means haplotype/tree leaves overall, not25 individuals or25 per class.

ARGformer uses graph-specific node IDs. Its released contrastive forward trains encoder/pooling similarities, not the existing class-logit layer. Global averaged haplotype/individual embeddings cannot replace local path predictions.

AncestralPaths' author94.6% is simulated accuracy on four non-BAA paths within a six-path model, not our result or real-data accuracy. Removing BAA and retraining is different from summing six probabilities into four. The existing stdpopsim C4/24-reference/64-query engineering cohort reproduces none of these papers and cannot supply25 distinct reference leaves.

## Proposed common demography and map

Name the condition **european_paths11_fixed_events_v1**, derived from the pinned AncestralPaths example. It retains BAA and differs explicitly from the BAA-free ARGMix paper model and stdpopsim C4 catalogue condition.

Population order/sizes: present_bronze50,000; Yam20,000; Neo50,000; EHG3,000; WHG2,000; NE15,000; BAA20,000; CHG3,000; Ana5,000; WA15,000; trunk15,000. Backward events:

- present_bronze→Neo/Yam at166 generations, fractions0.5/0.5.
- Neo→WHG/Ana at259, fractions0.25/0.75.
- Yam→CHG/EHG at177, fractions0.5/0.5.
- BAA→CHG/Ana at169, fractions0.25/0.75.
- WHG/EHG merge at600; CHG/Ana at800; NE/WA at1500. External census350.

Freeze these events independently of the reduced sample panel. Passing its age file directly to author simulate.py would change max-age-derived pulse times. Source example gives BAA169, while published Figure S7 gives170; this proposal selects169 transparently.

Mutation rate is1.25e-8. The four-parent execution pilot may use the source's uniform recombination1e-8, labelled uniform-map development. The main study proposes a pinned sex-averaged GRCh37 chromosome22 map, region[20,000,000,22,000,000), rebased with a saved offset. Exact map product/hash/build/interpolation must be resolved before freezing; no integrated cM length is invented. A later10 Mb region[20,000,000,30,000,000) tests longer context.

Expected present/Bronze source fractions(Ana,CHG,WHG,EHG)=(0.375,0.25,0.125,0.25) are descriptive. Realized fractions/switches remain recorded. Do not select easy parents, force held-out ancestry proportions or exclude mixed trees.

## Proposed common panel:140 diploids/280 haplotypes

This is a scaled **new common-task panel**, not the authors'476 diagnostic diploids. It contains48 pure-source reference diploids/96 distinct haplotypes,32 admixed reference-only diploids/64 haplotypes, and60 query diploids/120 haplotypes.

| Observed cohort | Pure references, diploids | Mixed reference-only, diploids | Queries, diploids/haplotypes | Query age, generations |
|---|---:|---:|---|---:|
| Ana | 12 | 0 | 0/0 | — |
| CHG | 12 | 0 | 0/0 | — |
| WHG | 12 | 0 | 0/0 | — |
| EHG | 12 | 0 | 0/0 | — |
| Bronze_Age | 0 | 8 | 8/16 | 135 |
| BAA | 0 | 4 | 4/8 | 155 |
| Neo | 0 | 16 | 8/16 | 200 |
| Yam | 0 | 4 | 8/16 | 170 |
| present_day | 0 | 0 | 32/64 | 0 |
| Total | 48 | 32 | 60/120 | |

The96 pure-source leaves exceed25 and any declared nearest-drop requirement. A future paper-style25+3 control still needs its correct model/source configuration; sufficient references alone do not establish reproduction.

Use heterogeneous reference dates from the already inspected author example age file. For each cohort, sort its m ages and select n dates at floor((j+0.5)×m/n), j=0..n-1. Save original date provenance and final order. This reuses age templates, not empirical subjects. Example ranges: Bronze100–165, BAA136–168, Neo139–258, Yam153–176, Ana217–360, CHG184–473, WHG210–529 and EHG155–402 generations. References older than census350 are valid while their source population remains active; scored queries must be younger than350.

All simulated individuals have distinct parent-qualified identities, both haplotypes share an observed cohort/date, and both query haplotypes stay outside the pure-reference pool. Mixed references have observed cohort labels, never an invented single pure ancestry. AncestralPaths retains its native eight-cohort feature order and ancient focal self-inclusion. ARGMix ancestry-token and ARGformer ancestry-contrastive pools use the96 pure leaves. Whole-tree/cohort-aware methods may consume mixed observed context. Report consumed-information differences and add a source-only-cohort ablation before attributing gains solely to the operator.

## Ontology, splitting and inference

Common class order: **0 ANA,1 CHG,2 WHG,3 EHG**; ANA is an explicit alias for author Ana. External truth is the true lineage's flagged ancestor at census350 at the scored physical coordinate: population8/7/4/3 maps to these four classes. Missing/wrong-time targets are errors.

Native AncestralPaths preserves **six path classes**:1 Ana_nonBAA,2 CHG_nonBAA,3 WHG,4 EHG,5 Ana_viaBAA,6 CHG_viaBAA. Export all six probabilities. Common C4 probabilities are(p1+p5,p2+p6,p3,p4), summed before argmax. Report native six-path accuracy/confusion alongside the common-source score. A correct source with the wrong via-BAA path is not a correct native path.

Assign independent genetic parents and individual identities before window/site/path extraction. The first four parents are development only. Propose master seed20261010 with domain-separated ancestry/mutation/error streams and model seeds101/202/303; freeze the exact seed derivation before execution. Any real-founder mosaic robustness needs disjoint founder individuals/pools across splits and is a separate family.

All genealogy methods share the same phased genotypes, observed panel, inferred tree sequence and coordinates, while retaining native representations. Main inferred observations use a pinned Relate build: VCF conversion, observed SNP annotations, mutation1.25e-8, initialN30,000, ordered ages/map, five-iteration population-size estimation and conversion to tskit. The post-size tree sequence is the common main input. ARGMix's no-size-re-estimation chain is a labelled secondary condition.

Save original-site/VCF/Relate coordinate and sample/haplotype joins. No internal population, census flag, ancestry truth, purity, donor identity or truth boundary enters inference/model input or determines prediction coordinates. Oracle trees are a separate condition with census-only nodes/segmentation removed. Native first-SNP/majority/purity policies belong to separate native-data scores; they cannot delete mixed intervals from the common full-span task.

First run reviewed source-default models. Only after real development results, freeze training examples, native-compatible tuning/stopping/calibration and budgets. Retain a source-default anchor and report all tried settings, updates, failures, parameter counts and cost. Do not truncate an expensive baseline to an uncompetitive toy. Final holdout sample size follows prespecified power for a proposed minimum meaningful absolute error improvement of1 percentage point,80% power and familywise alpha0.05 across three comparators, using development paired parent variation and measured cost. Planning may conservatively use alpha0.05/3 per comparison; no TEST outcome supplies its variance. These power assumptions require parent review; no final N is asserted here.

## ARGformer observed-only transductive scope

For each ARG, construct its vocabulary from that parent's observed topology. MLM may use unlabelled query/reference paths over the declared chromosome. Contrastive labels are observed pure-reference population labels, never query census ancestry, including on TRAIN parents.

For a future TEST ARG, run the pre-frozen within-ARG vocabulary/training schedule without its ancestry targets. Its90/10 path split concerns observed MLM, not independent biological ancestry testing. Do not reuse arbitrary node embedding meanings across independent ARGs. Keep namespace/checkpoint identity, native tokenizer/pooler, full unique-path keys, truncation settings and every physical occurrence.

Use the actual native encoder with a named downstream C4 readout: classwise mean exp(cosine/temperature), normalized across classes, with reference weighting and VAL-only temperature frozen. Preserve embeddings/native retrieval diagnostics; these soft scores are not calibrated ancestry proportions by default. Native PCA/nearest-neighbor/top-five evaluation remains separate.

Report transductive test-time training cost and access to the whole observed chromosome distinctly from inductive AncestralPaths/whole-tree inference. If the actual encoder cannot run, the comparator is missing; a toy encoder or feature-tokenized path model cannot be renamed ARGformer.

## Full-span score contract

Primary: equal-parent mean physical-span C4 error on64 present-day query haplotypes. Weight query haplotypes equally within a parent, then physical bases over the prespecified observed-only coverage mask. Retain failures/coverage against the full simulated span.

Define a shared100 bp lattice before inference. Predict at each bin midpoint and hold the score constant across that bin. Intersect this fixed prediction partition with full-resolution external truth for exact physical-length weighting, including truth changes inside a bin. This is a disclosed100 bp prediction resolution, not sampled-point accuracy or continuous-output exactness. A20 bp sensitivity and native observed-tree/interval-resolution score are predeclared additional outputs.

Preserve full class scores, physical support, sample/haplotype order, parent/config/checkpoint identity and native stitching semantics. Secondary metrics:

- Genetic-span error; span-weighted class precision/recall/F1 and balanced recall. Common confusion has true rows/predicted columns. Native AncestralPaths retains predicted rows/true columns.
- Brier score, raw NLL and VAL-only calibration/reliability. Preserve exact zeros; zero true-class probability gives infinite raw NLL. Any epsilon diagnostic records its fixed floor/count.
- Diploid source-dosage MAE/RMSE/correlation, ancestry-fraction bias, tract-length distributions and fragmentation, without truth-informed smoothing/deletion.
- One-to-one same-haplotype/same-transition boundary matching at100 bp and1 kb tolerance, localization errors and accuracy within1 kb of true boundaries versus interiors.
- Coverage, failed outputs, SNP density, tree/tract counts, inference/adaptation/training time, memory and disk.

Uncertainty resamples independent genetic parents, not sites/trees/windows/seeds. Future primary comparisons are candidate versus actually executed current-native ARGMix, native-architecture AncestralPaths common readout and native-encoder ARGformer common readout, with Holm adjustment. Record every effect; do not report only significant strata. Successful-parent subsets are diagnostic when required outputs fail. Beating current ARGMix cannot establish superiority against an unavailable exact April version.

## Staged biological progression and specialists

| Stage | Concrete next scale | Decision |
|---|---|---|
| Genuine execution pilot | Four2 Mb parents,2 TRAIN/2 VAL, proposed280-haplotype panel; source defaults and actual inferred observations | Measure biology/headroom and native runtime; no TEST or superiority claim |
| Main empirical-map study | Same2 Mb region/panel; TRAIN/VAL/TEST counts to be powered/frozen after development | Complete actual comparator fits and full-span scores under a new biological freeze |
| Longer/OOD transfer |10 Mb region; independent parents/counts decided in that freeze | Test context, tract/boundary generalization and errors without further tuning |
| Public real-data pilot | Desired48 pure proxies, at least16 admixed ancient references,20 held-out GBR diploids, optionally20 dated ancient queries | Accession/QC inventory,2 Mb feasibility then full chr22 concordance; no per-site real ancestry accuracy |

One fixed proposed demographic OOD profile doubles all four source sizes and shifts pulses to Bronze190,Yam210,Neo300,BAA200. The existing reference/query dates remain valid and census350 remains beyond all pulses. Paired observation-error conditions use allele flips0.001 at retained biallelic SNPs and phase-switch Poisson rate0.1 per cM/diploid. These are explicit perturbations, not exact paper error reproduction. Save phase truth externally, align observed haplotype targets or prioritize dosage; paired errors do not add independent genetic parents.

Include at least one executed genotype specialist, preferably **FLARE and GNOMix**, on the same observed phased genotypes/map/pure-reference labels. FLARE is ARGformer's native comparator; GNOMix is AncestralPaths' four-path comparator. The authors'0.02 cM GBR versus0.2 cM ancient windows are native comparison settings, not automatically optimal for this task. Versions/settings and any ancient-proxy preprocessing require a separate freeze using TRAIN/VAL only.

Use actual pilot costs to set storage/sample sizes. No biological N is chosen because a baseline looks inconveniently accurate. Store truth once, stream features, count environments/intermediates in total repository usage, warn at450 GB and stop before500 GB. Root owns contained installs/Git/GPU scheduling; at most two eligible physical2–5 GPUs via the existing launcher. This proposal claims no measured resource footprint.

## Public real-data endpoint and handoff

The paper states ancient genomes are previously published and listed in Supplementary Table1, aligned to GRCh37. Public raw reads do not guarantee accessible phased/imputed products. The archive supplies no complete phased cohort or trained model. Its476 diagnostic subjects, broader1015 ancient selection samples and large UKB ChromoPainter analysis are different datasets/tasks. UKB is unnecessary for the first public pilot.

Inventory exact accessions, phased products, quality/relatedness, dates, map/build and subject IDs before claiming the desired12 pure proxies per source exist after QC. If not, publish a distinct lower-reference real condition. Never duplicate haplotypes into extra slots. Query subjects/close relatives remain outside reference sets. Apply simulation-trained models first, with ARGformer's observed-only adaptation disclosed.

Real endpoints are independent LAI concordance, ancestry-fraction/chromosome patterns, second-reference-panel/leave-out sensitivity and genotype/phasing uncertainty. Model agreement is not truth, and population history must not force observed proportions to the simulation prior. The goal is stable, interpretable biological findings with external evidence.

**Immediate handoff:** review the proposed140-diploid panel versus author476, fixed demographic events/ages, runtime/config selection and Relate/sample-coordinate exports; then execute four genuine development parents and native fits. Use those observations to power/freeze the main study and start public-data inventory. No new generic gate, runner or synthetic test suite is requested.
