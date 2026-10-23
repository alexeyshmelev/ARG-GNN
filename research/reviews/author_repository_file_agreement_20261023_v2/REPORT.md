# File-by-file audit of the four author repositories

Completed 24 October 2026 on anogena. Three reviewers and the primary agent divided ARGMix, ARGformer/FLARE, AncestralPaths and GST. Reviewers subsequently shared the remaining ARGMix Rust files. This packet extends the earlier public-history audit and corrects incomplete or overly broad statements in it.

## Actual coverage

| Repository | Current official files | Review depth |
| --- | ---: | --- |
| ARGMix | 85 | Complete source/configuration reading, with structured dependency-lock review |
| ARGformer | 60 | Complete Python bodies, configuration/prose and structured lock review. One logo has hash/header review |
| AncestralPaths | 17 | Complete text review and visual review of its demographic schematic |
| GST | 166 | Complete text/record scans with scientific route review. Twelve PNGs have hash/header review |
| FLARE v0.6.0 | 105 | All 96 Java class contracts, declarations and option/data flow, plus critical method bodies. All nine other files scanned, including full structured scans of both VCF fixtures |

**All 433 current official files have a coverage record.** Integration compared exact path sets with the retained official trees and rechecked every payload hash. There are no omitted current paths or hash mismatches. This does not mean every Java utility was mathematically verified, every image was visually reviewed or every historical dataset was read.

Reachable public histories remain bounded at 9 ARGMix, 12 ARGformer, 27 AncestralPaths and 18 GST commits. Relevant historical source variants and differences were reviewed. Thirty AncestralPaths historical tree/VCF contents remain outside full review, including one header-only VCF. Its example checkpoint has metadata-only review, without numerical weights or optimizer values. Private and unreachable history are outside this claim.

## Material findings

| Method | Details recovered or verified | Conflicts or missing details |
| --- | --- | --- |
| ARGMix | Low-level 25-reference/three-ancestor components, manual export path, paper sampling budgets and distinct donor/output vocabularies | Current attention selects key i where the equation says key j. Fast raw-linear distance narrows before clipping. Supported full pipeline still blocks the paper ancestor geometry, restricts TRAIN replicate handling and rejects extra admixed donor tokens at the inference CLI. Original roster/map/error mechanism, exact architecture configuration and paper checkpoints remain missing |
| ARGformer | Full historical environment specification recovered. Native path, masking, pooling, objectives, checkpoint and retrieval routes traced through our wrapper | Release temperature, weighting fallback, corruption and extant pooling differ from paper descriptions. Original PCA10 ancestry classifier and several simulation/inference/evaluation details are not released |
| AncestralPaths | Historical CNN topology, exact Keras Flatten order, named 476-person roster and sample/age ordering recovered. Native feature bridge preserves ancient focal self-inclusion | Current Dense model is a different released condition. Native sampling and some train/test splits differ from the supplement. Broken painting interval output and incomplete downstream date aggregation remain. Exact empirical QC/genotype/map manifest and paper-run association are unavailable |
| GST | Synthetic per-model/cell/seed totals and every published BH training-set count explained by the correct source branch. Deleted GSTv2/CIFAR sources recovered | Gate architecture, learning rates, optimizer and uncertainty labels differ. Saved USPTO results are a 1,000-training-reaction pilot. Candidate adapters and full paired original-task experiments remain unfinished |
| FLARE | Official source identity, multiallelic input/MAC rules, ancestry order and native posterior/call handling verified. gen=10 initializes EM rather than fixing generations | Focal-site projection is an explicit study choice. Optional FLARE2 clustering is separate. Toy fixtures contain no ancestry truth and cannot stand in for the American benchmark |

The register contains 122 source-backed dispositions. Its categories distinguish resolved definitions, conflicts, explicit study choices, missing upstream detail and unfinished project work. A resolved row does not establish reproduction of a printed paper cell.

## Repairs and decisions made concrete

Two exact inactive ARGMix patches are included: `argmix/attention_key_j.inactive.patch` and `argmix/fast_linear_clamp.inactive.patch`. Both pass `git apply --check` against their exact pinned bases. The attention patch preserves directed psi(i,j) while selecting key j. The linear patch clips before narrowing. Small independent standard-library examples verify those logical effects. No author model was executed, and no accuracy effect was measured.

The historical AncestralPaths CNN has a concrete inactive specification, including its channels-first Conv2D, Keras 2.3.1 permutation before Flatten, six outputs, native features and final-checkpoint training rule. A naive NCHW flatten would permute Dense inputs. Runtime and numerical qualification remain required before calling a new implementation faithful.

Every method has a remediation matrix separating unchanged released controls from proposed equation/configuration corrections, historical compositions and new evaluators. Missing original settings are recorded explicitly. They are not filled by arbitrary defaults. Deterministic fixes do not identify unknown publication-run choices.

The FLARE non-Java supplement closes the earlier four unread test fixtures: 2,000 markers in each VCF, 300 reference samples in three panels and two query samples, all phased/nonmissing, with identical marker order and an exact panel/sample join. The map extends to 300 Mb while observed toy variants stop near 0.613 Mb. The launcher downloads unpinned latest files and was read, never run. It is a smoke example, not an accuracy benchmark.

## Consequence for our comparisons

The existing shared-data queue can still yield useful paired pipeline comparisons under its declared conditions. This audit **does not certify that those conditions reproduce all original paper benchmarks**. In particular, released ARGMix on the AP panel differs from its paper's 1,611-person, chromosome-22, genotype-error and 25+3 scenario. Released Dense AncestralPaths differs from the historical CNN. Native ARGformer retrieval is not the unreleased original PCA10 classifier. GST candidate experiments remain absent from that queue.

Reports must name the released and reconstructed conditions separately, retain shared sample/site/truth joins, disclose information access and training budgets, and keep corrected variants separate from untouched author baselines. No all-method superiority claim or fresh accuracy metric follows from this static review.

No running source, job, policy, frozen data, model or scientific setting was changed. All new files, source acquisitions, checks and Git operations for this audit stay inside the project on anogena. The Mac is only the SSH transport.

## Inspect the evidence

- [ARGMix report](argmix/REPORT.md), [agreement matrix](argmix/agreement_matrix.json), [patch review](argmix/patch_review.md).
- [ARGformer and FLARE report](argformer/REPORT.md), [agreement matrix](argformer/agreement_matrix.json), [FLARE non-Java supplement](argformer/flare_non_java_coverage.json).
- [AncestralPaths report](ancestralpaths/REPORT.md), [agreement matrix](ancestralpaths/agreement_matrix.json), [historical CNN specification](ancestralpaths/historical_CNN_INACTIVE_spec.json).
- [GST report](gst/REPORT.md), [agreement matrix](gst/agreement_matrix.json), [sample-count derivations](gst/count_derivations.json).
- [Unified gap register](gap_register.json), [coverage validation](coverage_validation.json), [logical checks](static_counterexamples.json).

Full acquired sources and verbose source displays remain on NFS2. Compact reports, file coverage, receipts and hashes are versioned. The report deliberately exposes the limits of each review mode instead of equating a file inventory with complete semantic or runtime validation.
