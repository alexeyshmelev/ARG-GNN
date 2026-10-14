# ARGformer published American-admixture benchmark audit

Audited 2026-10-15. DOI: [10.1093/bioinformatics/btag438](https://doi.org/10.1093/bioinformatics/btag438). This is a source/publication audit and an inactive experiment proposal. No simulations, scientific arrays, truth, models, GPUs, admissions, installations or Git operations were accessed or run.

The published benchmark is a three-source **EUR/EAS/AFR simulated American-admixture task**. H038 uses AncientEurope-derived European four-source ANA/CHG/WHG/EHG ancestry, different genealogy preparation, samples and score support. Its C4 results cannot be inserted into Table1, compared numerically to these percentages as an equivalent task, or converted to this ontology by merging classes.

## Exact Table1

The source is the publisher article reproduced in [Europe PMC full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13501305/fullTextXML), table btag438-T1. XML SHA256: 53ab26710804f0b9ca6f355095893b365413caf56cf7bb722b39907f6db0619f. It matches the earlier local literature receipt.

All values below are percentages. Precision and recall are published. F1 is **derived** as 2PR/(P+R) from the rounded printed percentages, so it is approximate and is not an author-reported F1.

| Method | EUR P | EUR R | EUR derived F1 | EAS P | EAS R | EAS derived F1 | AFR P | AFR R | AFR derived F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLARE |98.41|98.54|98.4750|98.85|99.74|99.2930|99.70|96.70|98.1771|
| ARGformer PCA |98.47|97.91|98.1892|98.81|99.07|98.9398|97.46|97.81|97.6347|
| ARGformer retrieval |98.47|98.00|98.2344|98.88|99.04|98.9599|97.46|97.90|97.6795|

Table1 has **FLARE**, not FLAIR, and has no RFMix, G-Nomix or other method row. It reports neither F1, standard deviation, confidence intervals nor a replicate count. The text says retrieval improves AFR “accuracy”; the table specifically shows higher AFR recall than FLARE but lower precision. The derived AFR F1 is also lower. Do not transform that statement into a claim of superior AFR F1.

Machine-readable rows are in research/reviews/ARGformer_published_benchmark_audit_20261015_v1/paper_table1.csv and paper_table1.json. New runs must compute F1 from each run's confusion counts, then aggregate across runs; never derive a run-mean F1 from averaged P and R.

## What the paper specifies

Section2.2 specifies msprime simulations using stdpopsim HomSap/AmericanAdmixture_4B18: 1,000 admixed samples and200 each European, African and East Asian; one30Mb chromosome; flat recombination1e-8 per bp per generation; mutation1.25e-8; tsinfer+tsdate inferred genealogies from error-free phased simulated genotypes; unique marginal-path pretraining split90%/10%.

The catalog model uses AFR, EUR, ASIA and ADMIX populations, an admixture event12 generations ago, with source fractions1/6 AFR,1/3 EUR and1/2 Asian. Catalog source ASIA maps to paper class EAS. ADMIX is the query population, not a fourth prediction class. The installed stdpopsim0.3.0 model's default mutation rate is2.36e-8, so it must be explicitly overridden with the paper's1.25e-8.

“Samples” is not explicitly resolved as haploid genomes versus diploid people in the article. This matters: stdpopsim's old positional/get_samples API counts haploids, while0.3.0 dictionary and named-population CLI counts diploid individuals. Passing1000/200/200/200 unchanged to the new API creates3,200 haplotypes. The inactive reconstruction proposal explicitly chooses1,600 haploid genomes (800 paired diploid subjects) and marks this as an assumption to confirm, not a fact proven about Table1.

Section3.1 evaluates10,000 unique marginal paths sampled from admixed individuals. PCA is fitted on unadmixed reference embeddings, retains10 PCs, and query ancestry is assigned by reference-cluster membership. Retrieval uses nearest reference-haplotype path embeddings and their reference labels. FLARE consumes simulated genotypes; ARGformer consumes the inferred genealogy. This is a native-method/input comparison, not proof that genotype and inferred-ARG inputs are identical information modalities.

ARGformer encodes ARG-specific node identifiers along leaf-to-root marginal paths. Shared internal IDs across trees are meaningful. MLM predicts masked node tokens, not ancestry; paper masking probability is0.30. Contrastive learning uses a learned pooling head, L2-normalized embeddings, supervised reference-population positives, in-batch negatives and inverse-frequency anchor weighting (Eq5). The paper states a default temperature0.05.

## Release and missing exact-reproduction evidence

The audited official source is [AI-sandbox/ARGformer](https://github.com/AI-sandbox/ARGformer), current/tag v1.0.0 commit f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596. It matches the immutable local archive .runtime/author_sources/argformer_f7545be7bca0. The release was published2026-09-07, after the paper's2026-08 publication. It is advertised as the paper implementation, but its date is not an exact paper-training commit certificate.

GitHub release assets are an empty list. The60-file tree contains model/training/preparation/embedding/retrieval code, YAMLs, tests and the logo. It contains no American simulator, tsinfer+tsdate benchmark generator, Table1 scoring/PCA-classifier script, released benchmark data, Table1 model weights or saved predictions. The supplement retrieved through the [Europe PMC supplementary-files endpoint](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13501305/supplementaryFiles) contains one3-page PDF and article figures. It confirms12 layers,12 heads, RoPE, FlashAttention2, and provides real-data path counts and representation analyses; it does not resolve the American Table1 pipeline. The data-availability sentence refers broadly to article/supplement/code. This audit found no exact benchmark dataset/checkpoint link in those audited locations; it does not assert that no such assets exist anywhere.

The release is directly useful, with these limits:

| Component | Audited implementation | Consequence |
|---|---|---|
| Path preparation |src/data/prepare_data_pretrain.py; deduplicate full raw leaf-to-root tuples, sorted node vocabulary, special tokens, Python shuffle seed42,90/10 split | Preserve full-path identity; do not deduplicate truncated inputs/embedding values or transfer checkpoints across ARG vocabularies. Script creates directories/loads trees at top level and uses placeholder paths. |
| MLM |yamls/mlm.yaml:12 layers/heads, length128, RoPE/window128/global every3, RMSNorm, unpadded FA2/fused CE,5 epochs, seed17, batch512, train mask.30/eval mask.15, bf16 | Existing isolated native runtime is already documented as qualified. Preserve native model/optimizer/scheduler. |
| Contrastive |yamls/contrastive.yaml:3 labels, learned mean pooler, normalized dot similarities, temperature.30,1 epoch, seed36, full batch4096, FP32 | YAML temperature differs from paper default.05; actual Table1 setting is not supplied. Do not silently change it and claim an exact release reproduction. |
| Weighting |sequence_contrastive.py calls build_contrastive_model(cfg.model); the weighting code looks for cfg.train_loader and catches failure with a uniform-weight warning | Released configuration lacks that nested loader, so the stock call can take uniform weighting rather than paper Eq5. Record actual warning/weights. Fixing it is a disclosed implementation correction, not unchanged code. |
| Embeddings |embeddings.py --pooling head_norm for contrastive checkpoints, --include_splits train,val, --max_samples0 | Use actual trained encoder/pooling-head output. The nominal3-class classifier head is not the ancestry classifier. Keep one checkpoint/vocabulary namespace per ARG. |
| Retrieval |retrieve.py normalizes embeddings and streams cosine/dot top-k; default --topk1 | Main text does not specify simulated-LAI k. Its explicit real-data top5 protocol cannot automatically be assigned to Table1. Default1 is a prespecified current-release reconstruction choice. |
| PCA classifier |Not present as an integrated Table1 evaluator | Ten PCs are specified, but clustering/label-assignment method, fit details and ties are unresolved. Do not invent “k-means” or “nearest centroid” and call it exact. |

Other unresolved details are simulation/mutation seeds, tsinfer/tsdate versions/options, error-free VCF creation, original ploidy/count convention, donor/query path curation, exact checkpoint/step and pooling setting, which10,000 unique paths were chosen, and the true ancestry/FLARE call coordinate for a path recurring over multiple intervals. The latter determines the score unit: a unique inferred path can recur over several disjoint spans and cross true-ancestry changes. A hidden purity filter or majority-label convention would materially change the benchmark.

**Exact numerical Table1 reproduction is not yet established.** Faithful execution of the current native model and a registered paper-inspired C3 reconstruction are feasible with existing code. Exact reproduction requires the missing data/evaluator/config or author clarification. Unchanged current release, paper-equation reconstruction and any bug-corrected variant must be named separately.

## Concrete native plan from existing code

The audit folder contains three full YAML drafts and native_commands.draft.json. Their model/loader/optimizer/scheduler/epochs/seeds/batches/precision match the released YAMLs. Only local input/output/run/checkpoint paths, fresh-resume behavior and external logging are changed. A pinned existing local BERT-base config replaces the network config lookup while preserving768/3072 width and12 layers/heads. Native GPU child argv are ready for root's existing qualified lease/title machinery; they are **not direct unadmitted GPU launch commands**.

For each independently simulated ARG:

1. Generate the declared American demography and external truth. Preserve mutation/ancestry seeds, sample/population mapping and migrations needed to recover source ancestry at generation12. Export perfectly phased genotypes and observed reference labels. Keep genealogy truth/migration ancestry out of every predictor's observed data.
2. Infer one ARG with tsinfer+tsdate from those genotypes, preserving actual sample/node identity. The existing .runtime/envs/biological_tsinfer_v1 metadata records tsinfer0.5.1 and tsdate0.2.7. Those are available current versions, not established paper versions. Existing H038 inference drivers are fixed to European280/952/2Mb policies and cannot be reused unchanged merely by renaming their config.
3. Apply the released full-path extractor/vocabulary/90-10 shuffle in a contained I/O copy; never run the placeholder source in place. Create reference-only contrastive and query-only embedding views with that SAME full ARG vocabulary. Reference labels are EUR/EAS/AFR; admixed ancestry truth never enters MLM/contrastive preparation. The release does not contain this three-way role-view materializer.
4. Run native main.py with mlm.current_release.io_only.draft.yaml, then sequence_contrastive.py with contrastive.current_release.io_only.draft.yaml. Bind each actual checkpoint SHA/epoch/step. Use the release's checkpoint lifecycle; do not select a checkpoint using outer ancestry performance or silently claim a final-epoch checkpoint when latest was saved earlier.
5. Run embeddings.py with the contrastive YAML and --pooling head_norm --max_samples0 for reference paths, then the query I/O YAML for query paths. The command manifest contains exact existing CLI flags, including --use_split_labels only for the reference corpus.
6. Run retrieve.py with --topk1 and seal original indices/scores plus query full-path IDs before ancestry scoring. Do not substitute the existing H038 class-mean exp-cosine C4 readout for native nearest-neighbor retrieval.
7. Score the prespecified common evaluation records with one3x3 confusion matrix per method and replicate. For a reconstruction, the proposal chooses a path's first-occurrence interval midpoint before reading truth. This is explicitly an invented but deterministic evaluation contract, not a recovered author rule. Score FLARE/RFMix at the same haplotype and coordinate; disclose their marker interpolation/support. Do not remove difficult or missing calls after looking at labels.

The two principal missing adapters are a new American observed-data/role preparation pipeline and a common path-level ancestry evaluator. Neither is supplied by the author release or existing European frozen drivers. A thin I/O adapter may preserve all native algorithms; it cannot make an unknown published sampling/scoring rule known.

## Baselines

FLARE is the published comparator, a phased-genotype local-ancestry method based on reference haplotype copying/HMM inference, with its own model/EM estimation and marker filters. The audited current release README is0.6.0 and GitHub commit d6a6c8fd4479ebdb68529824acd11d2b6bbac4a1. The paper's exact FLARE version/options were not supplied.

Its documented invocation is: java -Xmx16g -jar <pinned flare.jar> ref=<reference.vcf.gz> ref-panel=<sample_to_source.txt> gt=<admixed.vcf.gz> map=<PLINK_cM_map> out=<new_prefix>. Use a pinned binary, perfectly phased/nonmissing genotypes and a flat map corresponding to1e-8 per bp per generation (30cM over30Mb). Defaults such as EM and min-MAC/min-MAF must be recorded; no held-out truth-derived ancestry proportions may be supplied.

RFMixv2 is an **additional new comparator**, not a Table1 replication row. The audited source commit is09599c19c69f81571c7a7aab4cd2d847a9e0633a. Its manual specifies random-forest discriminative classifiers followed by a chromosome conditional random field, with optional EM/reference reanalysis. Documented inputs are phased query/reference VCFs, sample/source map and a cM genetic map. Its quickstart argv are rfmix -f <query.vcf.gz> -r <reference.vcf.gz> -m <sample_map> -g <map> -o <new_prefix> --chromosome=1. Pin all defaults, RF/CRF windows and optional EM, and retain its academic-research license notice. No binary was installed or run in this audit.

G-Nomix or other named methods would also be newly registered competitors with separately audited native code/config and common evaluation units. No numbers for them are inferable from Table1.

## Legitimate three-class GNN adaptation

Train a fresh C3 model on newly generated American simulations. European ANA/CHG/WHG/EHG labels do not map to EUR/EAS/AFR, so neither collapsing those classes nor relabeling a trained C4 checkpoint is valid.

Existing generic models.build_model and AnchorKernelModel support num_classes=3, and ReferenceFeatureAdapter accepts three reference columns. The frozen H038 mutation-routed PyG code hardcodes four reference one-hot channels; H039 hardcodes four emissions,4x4 transitions,12 ordered off-diagonal witnesses and686 queries. Those production paths need NEW typed C3 versions.

Keep the same observed whole-tree message passing, mutation-support routing, temporal/interval aggregation and loss equations. Change only declared class/population-dependent dimensions: three donor channels, three emitted ancestry probabilities,3x3 pair-state transitions and6 ordered off-diagonal entries. Replace fixed952/686 sample contracts with the new frozen American roster. Freeze the feature layout and report the resulting parameter count. Preserve sum-product normalization and apply the same probabilities/call rule to all test records; do not truncate/renormalize a four-class output.

Use actual PyTorch/PyG graph computation and a fresh initializer. Keep reference source labels observable, query ancestry and true simulated genealogies external, and checkpoint selection on development validation only. The generic C3 constructors are evidence that adaptation is supported, not evidence that the frozen H039 training driver already runs this task.

## Prospective splits and standard deviation

protocol.proposal.json freezes20 independent GNN TRAIN simulations (seeds21001-21020),5 development VAL simulations (31001-31005), and10 paired outer benchmark simulations (41001-41010). Split replicates and diploid subjects before genomic windows. Mutations use ancestry seed+1,000,000; query-path selection uses simulation seed+51,000 and is label blind.

For each outer ARG, ARGformer fits its own native vocabulary/checkpoints transductively on observed paths, with reference-only contrastive labels. Native90/10 path validation is not an independent-ARG or held-out-subject split. It must not be presented as inductive transfer. GNN outer ancestry labels are never used for fitting or checkpoint selection. Native baselines receive the same frozen source/reference/query populations while retaining their intended genotype or inferred-ARG inputs.

Report per-class P/R/F1 and macro-F1 per outer replicate; then mean and **sample SD(ddof=1)** across the10 independent simulated ARGs. Use paired per-replicate method differences and an ARG-level paired bootstrap95% interval. Keep all per-run confusion matrices and seeds. Do not treat10,000 correlated paths as10,000 independent replicates, and do not use path bootstrap variability as a replacement for between-ARG SD.

A separate sensitivity may repeat5 model initializers on3 fixed ARGs to expose training variance. Report that separately, rather than counting15 fits as15 new genealogies. One pipeline canary is useful for correctness but cannot supply SD. Publication Table1 has no recoverable SD to copy.

Before launch, root must register the new task, resolve or explicitly freeze reconstruction assumptions, and obtain normal existing resources/runtime/storage admission. This audit creates no new resource guards and authorizes no experiment.

## Files and verification

The owned audit folder contains the full-text XML/Crossref/GitHub evidence, extracted/rendered supplement, Table1 CSV/JSON, inactive protocol, full native I/O-only YAMLs, exact child argv and source receipt. Public assets only were downloaded. YAML numeric/model fields were compared to the pinned release. Original author, H038/H039 sources and artifacts remain untouched.

The paper supplement source is research/reviews/ARGformer_published_benchmark_audit_20261015_v1/supplement.pdf. Its relevant S1 page was rendered and visually checked. The exact numerical results above come from article XML, rather than OCR.
