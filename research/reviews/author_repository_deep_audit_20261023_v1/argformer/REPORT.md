# ARGformer and FLARE author-repository deep audit

Audit date: 23 October 2026. Public requests began at 20:21:36 UTC. Source inspection only. This packet neither changes nor reruns the benchmark.

## Conclusion

The available author release supports a substantive native ARGformer pipeline comparison on our newly simulated, common American-admixture observations. It does not establish an exact reproduction of the original data or ARGformer Table 1. A fresh examination of every reachable ARGformer source snapshot recovered no missing Table 1 evaluator, original checkpoint, simulation script, focal-coordinate convention or ancestry-scoring implementation. The important paper/release differences remain real and disclosed.

No new source defect was found that by itself invalidates the planned comparison on shared data. That assessment is limited to the inspected source and interfaces. It is not a certificate of completed execution, final scores or all numerical behavior.

## Coverage and fresh evidence

- The current GitHub branch `main` and sole tag `v1.0.0` still point to `f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596`. All 60 official current files match the local pin by Git blob hash. There are 59 official text files and one logo image. The additional local acquisition receipt is identified separately. Every text file, including the dependency lock, was read, hashed and searched. Every Python file was parsed as syntax data to inventory functions/classes. Core data, model, training, extraction, retrieval, configuration and test interfaces were manually traced.
- Fresh branch, tag, release, commit-list, complete tree, all-issue and issue-comment responses are retained. There is one branch, one tag, one release with zero attached assets, and one closed issue. The issue asks for pretrained checkpoints; the only author response remains “Resolved via email,” without a download URL.
- All 12 commits form one complete linear chain to the parentless initial commit. All 12 source snapshots were fetched afresh through public codeload URLs and every text member was walked. Their source archives totaled 9,993,837 bytes read. Every historical file has a census/hash; 23 distinct core source/document/config versions are preserved as plaintext. No notebook or historical ancestry evaluator was recovered. The only historical-only filename is `environment.yaml`.
- Twelve public GitHub API requests were used across ARGformer and FLARE. Every request returned 200. URLs, UTC timestamps, response hashes and rate-limit/ETag/date headers are retained in `public_requests.json`. The final response reported 16 shared requests remaining. Codeload/raw source requests are recorded separately and did not use the API quota. No public genotype test payload was opened.

The broad file walk detects code/config/doc candidates and follows the scientific pipeline; it is not an independent numerical proof of every inherited transformer kernel.

## Recovered ARGformer pipeline

### Data and preprocessing

`src/data/prepare_data_pretrain.py:17–36,40–80,127–168` accepts an already dated tree sequence at a placeholder path. It does not simulate genotypes, run tsinfer/tsdate or define American-admixture samples. It enumerates samples and trees, assigns IDs to complete raw leaf-to-root tuples, retains the set of tree indices for each path, builds a sorted node-ID vocabulary with four special tokens, and applies a seed-42 random 90/10 split after deduplication.

This is a within-ARG path split. It is not an independent-individual or independent-ARG split. `README.md:100–105` expressly says node IDs and vocabulary belong to the ARG used in training. `src/arg_data.py:244–280` removes the extant leaf when requested, keeps up to 126 node tokens between CLS/SEP for a 128-token input, then pads. Distinct full paths can therefore yield identical truncated model inputs. Deduplicating by those inputs or by embedding values would change the author representation.

The preparation script does not create ancestry `labels.pkl`, sample a Table 1 focal set, or choose a genomic coordinate for a recurring path. The contrastive dataset merely loads an external row-aligned labels file and checks its length (`sequence_contrastive.py:229–271`). The class ontology/reference-label bridge and path-coordinate choice consequently remain external responsibilities.

### Configuration inheritance and MLM

`main.py:537–546` merges `yamls/mlm.yaml`, the supplied YAML and CLI overrides. The maintained worker instead loads that same release YAML and calls the author function directly, changing I/O, logging, retention and initialization paths (`american_competitor_worker.py:90–103,126–163`).

The MLM YAML fixes 5 epochs, batch/microbatch 512, BF16, seed 17, AdamW learning rate 5e-4, dropout choices, 12 layers/heads, RoPE, and a 0.30 selected-token probability. `src/flex_bert.py:255–281` uses `bert-base-uncased` to obtain base configuration and constructs a fresh ARG model unless an actual Composer checkpoint is supplied. The base-model name is not evidence of publicly supplied ARGformer pretrained weights.

`src/arg_data.py:329–335` delegates masking to the installed Hugging Face collator. It does not override corruption probabilities to replace every selected token with MASK. The inherited generic `tests/test_mlm_masking.py:39–57` explicitly expects the 80/10/10 rule. This supports the existing release-versus-literal-paper masking disclosure; it does not prove which private runtime produced Table 1.

### Contrastive training

`sequence_contrastive.py:411–495` implements learned pooling, unit normalization, all nonself in-batch negatives, same-class positives, and exclusion of anchors without positives. `yamls/contrastive.yaml` fixes temperature 0.30, one epoch, batch/microbatch 4096, FP32, seed 36, learning rate 2e-5 and unbalanced sampling. The `num_hard_negatives` fields do not activate the unused pair-dataset route: the actual trainer uses `ARGClassificationDataset` and the supervised in-batch collator (`637–710,821–848`).

The class-weight fallback is verified through configuration scope, not just a project flag. `train()` passes `cfg.model` into `build_contrastive_model` at line 832. Its inverse-frequency branch then seeks `cfg.train_loader` at lines 603–610, although the released loader is top level. The exception is caught at lines 621–622 and weights remain `None`. Our worker explicitly verifies temperature 0.30 and `class_weights is None`. Adding `model.train_loader` would change this release condition.

The repository also retains two different runtime recipes: `requirements.txt` pins Torch 2.3.0, Transformers 4.40.2 and Composer 0.22.x, whereas `pixi.toml` pins Torch 2.4.0, Transformers 4.44.1 and Composer 0.24.1. The history shows an environment-to-pixi migration, not a unique historical Table 1 environment. Our runtime must therefore remain separately versioned and disclosed.

The public contrastive CLI reads `yamls/defaults.yaml` at lines 922–927, but that file does not exist in any of the inspected snapshots. The documented untouched CLI would need this external file or an adaptation. Our wrapper bypasses that CLI by supplying the release YAML directly to unchanged `train(cfg)`. This is an executable entry-point adaptation, not recovery of the missing original defaults.

The contrastive YAML, loss implementation and retrieval implementation are byte-identical across all 12 commits. The only historical MLM YAML change removes the diagnostic UniqueTokenCoverage callback. There is no alternative historical temperature/class-weight/masking correction hiding in another branch or configuration. The source default temperature 0.05 remains available when no YAML temperature is supplied; it is not the release YAML value or evidence for the original Table 1 run.

### Embeddings, retrieval and PCA

`embeddings.py:14–96` reconstructs the architecture from the supplied training YAML, clears the training initializer, loads the requested checkpoint with prefix mapping and rejects missing encoder/pooling-head weights. `117–178` supports final-layer extraction and learned `head_norm`, which applies the trained pooling head followed by L2 normalization. The head itself is mean pooling plus its learned transformation (`src/bert_layers/model.py:748–769`).

The extractor can interleave multiple loaders randomly (`embeddings.py:311–322`). Its default 10,000-row cap is an extraction cap, not the paper's focal-sampling definition. Our four separate single-split extraction calls, unlimited extraction and full-path sidecars are appropriate for preserving row identity. Query-label loading is disabled; only reference labels are passed to retrieval.

`retrieve.py:13–16,78–97,117–168` normalizes corpus/query vectors and runs native streaming dot-product top-k. The default is k=1 with query batch 1024 and corpus chunk 100,000. It returns indices/scores and optional supplied labels. It does not implement Table 1 class voting, truth lookup, precision/recall or PCA classification. It has no explicit self-exclusion or stable lowest-index tie rule. Our reference/query sample rosters are disjoint, and the native tie behavior is preserved.

`README.md:135–181` and `aggregate_embeddings.py` average unique paths into haplotype/individual vectors and show a centered two-component visualization of all supplied means. This changes the unit from a local path and supplies neither reference-fitted PCA10 nor cluster-to-ancestry assignment. It cannot substitute for the paper's missing classifier.

### What the tests establish

Tests cover generic padding, rotary attention, FlashAttention/SDPA behavior, sequence packing, initialization/tiling, and recent extraction/aggregation edge cases. Several inherited tests reference absent `yamls/defaults.yaml`, model YAMLs or `sequence_classification.py`. No test implements American simulation, ancestry-label generation, PCA10 clustering, path-coordinate sampling or Table 1 scoring. No tests were run in this audit.

## FLARE source, options and current public changes

The official tag remains `v0.6.0`, commit `ec0ad295aeebfacf10d8cbfb6486ab78981de1b1`, and there are no GitHub releases/assets. A fresh tag archive independently confirms exact byte equality of all 96 Java sources used by our existing build. The local directory also contains 102 compiled class files, which were hashed but not executed. The current public master contains the same 96 Java files; only the AdmixMain version banner differs, from 31 October to 3 November 2025. No relevant Java algorithm/default-option update was found. Fresh README changes clarify wording and Java 11 requirements.

`AdmixPar.java:78–103` and the fresh documentation confirm native defaults for sequence data: min-maf 0.005, min-mac 50, EM true, update-p false, panel-probs false, initial gen 10 and seed -99999. Our command keeps those settings and explicitly requests probabilities and four threads. `README.md:191–195` and `ParamEstimator.java:56–84` establish that gen 10 is the initial value for EM; it does not force 10 generations against our generation-12 simulation. Without a supplied model, native parameters are estimated from the observed inputs. No query ancestry file is supplied by our command.

`create_model_file.py`, present both at the tag and current master, fits a Gaussian mixture to FLARE panel probabilities for the separate unassigned-panel workflow. It is not ARGformer's PCA10 evaluator, and our known-source-panel comparison does not use it. The small example test runner is likewise present at both versions. Its public VCF test data were not opened or executed.

Our FLARE adapter retains native AN1/AN2 hard calls. Its nearest-marker projection to frozen path focals, midpoint ties to the right and edge carry are project conventions. Exact original Table 1 FLARE binary/options and this projection remain unbound by the paper, even though the native algorithm is preserved.

## Scientific consequences and remaining ambiguity

The primary article's §2.2 specifies the named demography, 30 Mb, rates, population counts and tsinfer+tsdate observation. It does not resolve whether 1,600 samples means haploid genomes or diploid individuals, nor provide the original mutation model, versions/options or seeds. Our explicit haploid/JC69/discrete/unary-processing choices are reconstructed. The paper's stated default contrastive temperature 0.05 and inverse-frequency weighting differ from the unchanged release condition above.

Section 3.1 specifies 10,000 unique admixed marginal paths, reference-fitted PCA retaining 10 PCs, cluster membership, and a separate nearest-reference strategy. Original cluster algorithm, fit population, cluster-label mapping, retrieval k, checkpoint, focal seed, recurring-path truth coordinate, FLARE projection and score aggregation remain unrecovered. Current real-data top-5 examples do not resolve American Table 1 k. No public asset found in this pass closes those gaps.

The maintained common scorer checks identical record/sample/position/parent/class identity for all 15 evaluation parents and all three methods before resolving truth, and uses complete half-open truth intervals with original query columns (`american_scoring.py:184–220,287–324`). Native preparations compare their focal arrays to the shared registry exactly. These are meaningful protections for a paired pipeline comparison, distinct from paper-protocol equivalence.

The proper claim is an author-release reconstruction on common observations and common evaluation support. The methods have different training roles: TreeLoom uses supervised TRAIN ancestry, native ARGformer fits observed paths within each evaluation ARG and known references, and FLARE estimates ancestry from reference panels. This does not isolate architecture alone. Older audit packets' seeds 92001/93001 and 20×2 Mb GNN description are historical C3 facts; the fresh campaign has new streams and full 30 Mb parents. They must not be copied into current dataset claims.

## Evidence map and limits

Start with `evidence.json`, `source_census.json`, `current_tree_local_verification.json`, `historical_snapshots.json`, `historical_core_variants.json`, `core_history_diffs.json`, `route_source_excerpts.json` and `benchmark_interface_source_refs.json`. Public metadata responses, historical core plaintext files, FLARE current changed text, source comparisons and request logs are retained alongside them.

This bounded negative finding covers all currently advertised ARGformer refs and their complete accessible history. It does not exclude private/email-supplied assets, unreachable commits, forks or unadvertised external storage. FLARE's entire commit history was not downloaded; its relevant current source and pinned tag were independently compared. The existing primary JATS paper snapshot was read as data; publisher/preprint bodies were not fetched anew. No scientific arrays, model checkpoints or embedding payloads were opened, no author module imported, no installs/GPU/tests/jobs/process changes/Git operations performed, and all writes remained inside this new packet.
