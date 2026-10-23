# AncestralPaths per-file paper/source agreement review

Date: 23 October 2026 UTC (completed after midnight on 24 October local time).

## Finding and review boundary

The historical CNN is recovered and its layout, parameter shapes and optimizer configuration are supported by independent author source and HDF5 metadata. It should be prepared as a **separate inactive condition**, preserving the current released Dense condition and the exact shared frozen observations, features, labels and evaluation registry. Neither condition can currently be certified as the precise implementation that generated the paper's published cells.

All 17 current upstream files received content review: 16 complete text files and the complete one-page demographic schematic, rendered through SSH and inspected as an in-memory image. The 27 reachable commits span 64 paths. This pass reconstructed 71 text change events containing 69 unique byte contents and reviewed complete source bases plus all verified differences, rather than treating hashes or AST inventories as content review. Every reconstructed file is saved in full and authenticated by Git blob identity. All numeric rows in historical age, population-label, accuracy and confusion receipts were inspected or checked against a biologically meaningful order/totals contract.

Historical binary coverage has a narrower mode: 29 tree/VCF contents remain unread; one additional VCF has its header reviewed only, and the historical checkpoint has HDF5 configuration/names/shapes/dtypes reviewed only. No numerical model weights or optimizer values were read. These 31 assets are explicitly outside full-content coverage. Private, unreachable or omitted external history is outside this bounded public-history review.

The five missing source files at the known 9d808393 snapshot, one missing 36-value confusion receipt and the Keras 2.3.1 Flatten implementation were the only new public raw requests. No GitHub API calls were made in this pass. All other evidence was reused from the existing audit.

Deliverables are file_coverage.json/CSV, agreement_matrix.json, remediation.json and evidence.json. The matrix has 30 findings: 13 resolved definitions or behavior, 7 direct conflicts, 6 explicit new-run choices and 4 upstream-unavailable items. “Resolved” means the evidence resolves the stated definition, not that a model has been run or numerically validated.

## Historical CNN: complete concrete specification

Canonical author source is training.py at e6f5d9c6be628007c662e1c71a07358cf350efaf, retained in the prior audit as history_source/previous_training.py, SHA256 fc90bb333e3589f3e4f76098dfc591222de1e40c7382a3f367e876270393b4eb. Its topology already appears in the initial d0c2b075 training_balanced_class_BAA.py, then persists across the October 2022 variants. Commit 77db3cfa, “Simplified neural net,” replaces it with the accepted Dense model.

The recovered topology is input [N,1,5,9], Conv2D with 32 filters and a 5-by-1 valid kernel, ReLU, Flatten, Dense 64/ReLU and Dense 6/softmax. There is no pooling or dropout. The feature axis contains eight ancient-cohort proportions plus ancestor age. The kernel spans the five ordered informative ancestors and is shared across feature positions; it does not convolve directly on the genealogy.

The historical HDF5 configuration independently confirms Keras 2.3.1 with TensorFlow backend, channels_first on both Conv2D and Flatten, kernel shapes [5,1,1,32], [288,64] and [64,6], plus biases of 32, 64 and 6. The parameter total is 19,078. Numerical values remain unread, and the checkpoint's actual training updates, data identity and paper-result association remain unknown.

**Flatten ordering matters.** The complete Keras 2.3.1 Flatten class was read at keras_2_3_1_core.py:462-521, particularly 507-516. It permutes [N,32,1,9] to [N,1,9,32] before flattening. Thus flattened index is 32 times feature position plus filter. A naive NCHW reshape would permute the first Dense layer's inputs even though shapes and parameter count still look correct. Treating [1,5,9] as channels-last would also change the model.

Source lines 203-213 define six-class categorical cross-entropy, Adam, batch 30, five epochs and final-model saving, without validation selection or early stopping. HDF5 metadata records learning rate approximately 0.001, beta1 0.9, beta2 0.999, epsilon 1e-7, decay zero and amsgrad false, with Glorot-style uniform kernels and zero biases. Original initialization seeds, exact TensorFlow version and runtime closure remain unavailable.

historical_CNN_INACTIVE_spec.json makes the proposed baseline concrete without creating a runnable port. It retains the historical topology, channel and Flatten order, optimizer settings, batch 30, five epochs, fresh initialization and final checkpoint. On the same frozen 50,000 TRAIN examples as the Dense comparison, the complete expected fit is 8,335 updates. A fresh declared seed and explicit shuffle semantics are new-run choices. The old example checkpoint is configuration evidence only and is not a warm start. Runtime qualification, including CPU NCHW support and legacy optimizer/initializer semantics, remains required before separate execution authority.

## Sampling and train/test history

The CNN-era runner uses TRAIN parents 0-4 and TEST parents 5-9. This resolves the former inference that all released variants suffer the accepted version's train/test overlap. The accepted Dense runner instead uses TRAIN [3,5,6], while testing still uses 5-9, so two parents overlap in that specific release.

Neither native runner exactly implements the supplement's European numeric recipe. Historical CNN preallocates 812,500 TRAIN rows, accepted Dense 487,500, and native testing allocates 250,000 rows: 10,000 per cohort per TEST parent. The paper specifies three 200 Mb TRAIN parents, 10,000 training pairs per five admixed cohorts (50,000 total), then 10,000 testing GNNs per each of five new parents.

Native sampling uses site.id modulo a derived stride, traverses samples in source order and caps accepted counts using true path labels. If quotas are unmet, it traverses again after increasing the stride by half. It may revisit examples; stride zero and later fractional-stride behavior are unguarded, and there is no explicit pass bound. The preallocation is not evidence of realized unique training rows. Early hardcoded training variants also use an exclusive Yamnaya endpoint of 685, excluding one of 26 Yamnaya haplotypes; the accepted version's 686 endpoint restores that sample.

The maintained european_sampling.py:25-137 declares a different, observation-only paper-count reconstruction: exactly three independent TRAIN parents and five TEST parents, 50,000 TRAIN rows, 10,000 per cohort, and 10,000 TEST rows per parent with 2,000 per cohort. It selects retained SNP-rank bin midpoints, freezes a per-parent/cohort haplotype permutation, and uses no label-based class caps. These choices are disclosed and preserve shared support. Restoring the historical capped sampler would constitute a separate condition and must not change this frozen comparison.

A recovered 2bd4bd7c simulator indeed states three 200 Mb parents, but the complete source uses args.ts for output paths although its parser defines only -out and -ages, and emits numeric population labels. Its paired training runner requests five TRAIN parents. This snapshot is valuable evidence for the scale, not a coherent, untouched executable paper recipe. Later CNN-era and accepted simulators use ten 2 Mb parents.

## Demography, roster and query eligibility

The source defines eleven populations, ordered present_bronze, Yam, Neo, EHG, WHG, NE, BAA, CHG, Ana, WA and trunk. It uses census time 350; splits at 600, 800 and 1,500 generations; admixtures Neo=(WHG .25, Ana .75), BAA=(CHG .25, Ana .75), Yam=(CHG .5, EHG .5), and present_bronze=(Neo .5, Yam .5). Recent event times are maximum supplied group age plus one. Source sizes, rates and all event definitions are recorded in the coverage/matrix.

The 476-diploid roster has 91 GBR, 81 Bronze Age, 7 BAA, 151 Neo, 13 Yam, 51 WHG, 43 EHG, 25 Ana and 14 CHG individuals. The simulator concatenates samples in that order. The physical roster rows are in a different order. There are 952 haplotypes, of which 686 admixed-cohort haplotypes are targets and 266 belong to the four pure reference groups. Some reference samples are older than census 350; they are not native ancestry-inference targets.

The historical three-column example/mesoneo_ages_pop.txt recovers 476 unique public subject names, including ancient IDs and 91 GBR names. Every cohort/age pair matches the anonymous current roster. Every one of the 952 example/test.ages values and 476 example/test.poplabels records matches the simulator concatenation order. These are concrete newly recovered identity leads. They do not establish the complete empirical genotype/QC/phasing/map artifact manifest.

The source-derived BAA event is 169 generations because the maximum BAA age is 168. Historical test.yaml confirms 169. The complete current schematic and the supplement instead say 170. The maintained recipe explicitly chooses paper 170 while retaining the rest of the archived graph. This is a disclosed scientific choice shared by all methods, not something this audit should silently reverse. Similarly, the supplement's inconsistent “15,000 years (1500 generations)” wording does not justify converting the source split to 500 generations.

The linked current stdpopsim AncientEurope_4A21 model has ten populations, no BAA, different sizes and recent times of 140/180/200, and exponential present-day growth. It cannot silently replace the archived eleven-population six-path graph merely because the main article links it.

## Feature and truth agreement through maintained adapters

Historical training lines 149-177 and current testing lines 160-186 collect previously unseen ancient descendant leaves at successive parents, skipping empty shells without consuming a row. The order is Bronze Age, BAA, Yam, Neo, WHG, EHG, Ana and CHG, followed by int(age)/1500; remaining rows are padded with -15.

Ancient focal self-inclusion is present in native training/testing: the first parent contains the focal ancient leaf, and there is no leaf != query condition. Observed ancient admixed cohorts are allowed reference information. This does not expose true local ancestry labels. Query role limits which samples receive supervision, not which observed ancient cohort labels enter the feature matrix.

The maintained path is european_native_models.prepare_features:127-174 -> european_execution.iter_native_sampled_features:186-234 -> ancestralpaths_native_bridge.extract_native_features:254-299. It preserves ancient focal self-inclusion, all eight ancient groups, repeated-add Float64 arithmetic, integer training ages, empty-shell skipping and exact per-query padding, then casts the final row to Float32 for Keras. No discrepancy was found on those native semantics. Removing the focal sample or retaining only the four pure reference groups would be a named methodological ablation, not an automatic correction.

The empirical painter uses Float32 accumulation and float(age)/1500, unlike training/test. It pools identical feature shells for sibling samples but pads only sam, rather than every member of sam_rel. Counted siblings can therefore retain zero padding. The bridge's independent per-query extraction avoids this empirical optimization defect while agreeing with native training/test. An empirical repair must explicitly choose and document age/precision semantics.

Native labels are 1=Ana non-BAA, 2=CHG non-BAA, 3=WHG, 4=EHG, 5=Ana through BAA and 6=CHG through BAA. Census population and observed query cohort determine these labels. The maintained bridge and external target join preserve this ontology. european_sampling.py:140-184 uses exact original positions, half-open truth intervals and semantic (sample_id,haplotype) joins. Inferred sample IDs, VCF one-based positions, original zero-based positions and clipped local coordinates have explicit mappings. Truth is resolved separately from observed features and only after the required prediction barrier for evaluation. Original source assumes matching sample indices and walks until int(node time)==350 without equivalent validation.

## Relate, maps and runtime

The complete relate.sh:5-9 performs VCF conversion, SNP annotation, Relate with mutation 1.25e-8, initial N 30,000, sample ages and a uniform map, then five population-size iterations and final tree-sequence conversion. The current three-row map covers 2 Mb at 1 cM/Mb; the historical example map covers 200 Mb. A long map alone does not enlarge historical 2 Mb genotype data. The maintained recipe generates a full declared-length map and keeps the post-five-iteration inference condition. Its retained SNP coverage and coordinate offset are explicit.

Exact historical Relate/TensorFlow versions are not pinned. Current or future runtime pinning can support a named new-seed comparison, not prove identity with an unknown old paper runtime. The empirical painter expects tab-separated fine-scale map columns Position(bp), Rate(cM/Mb) and Map(cM), which differ from the simulation map's Relate headers.

Current pipeline.sh passes -poplab to accepted training.py even though that parser no longer defines the option. Native testing and painting retain CNN-shaped rank-four inputs while the accepted Dense model requires rank three. Legacy private backend imports, predict_classes and nb_epoch also need qualification. The maintained Dense adapter correctly preserves its named final architecture, uses rank-three predictions and records five epochs, batch 32, 7,815 updates and final checkpoint selection under a declared modern runtime.

README -sample versus parser -samples is **not** itself a current parser failure: argparse's default unique-prefix abbreviation accepts it. A real older args.sample attribute mismatch was fixed in e6f5. Future commands should use the canonical full option. README multiline examples also need explicit continuations, and the pipeline does not collect individual background Relate failures or enforce fail-fast shell behavior. No author pipeline or model-building entrypoint was invoked here.

## Metrics, SD axes and historical receipts

Native cvscores has shape [parent,cohort], with percentages. kfold.log applies mean/std over axis 1, yielding within-parent differences across cohorts, not each cohort's variation across five parents. np.std defaults to ddof 0. README's claimed axes/file description is inconsistent. The full raw matrix remains usable if the summary is explicitly defined.

Native confusion calls pass predicted labels first, so matrices have predicted rows and truth columns. This agrees with the supplement's row-normalized precision description. The maintained bridge keeps that convention. The common comparison scorer instead explicitly records truth rows and prediction columns, with precision computed using prediction-column totals. This is a representation convention, not an accuracy change. Transpose before native-style precision visualization.

The maintained summary uses equal-parent means and sample SD with ddof 1 across five independent parents, in fractions. The paper explicitly names mean and SD but does not state the divisor. Report units and distinguish percentage points, fractions, SD and SE.

All six historical confusion files were read. Each cohort totals 50,000 examples, and the pooled matrix totals 250,000 and exactly equals their elementwise sum. Its diagonal gives 91.4504% accuracy. All 25 kfold values agree with this mean to stored floating precision; sample SD across five equal-cohort parent means is approximately 0.8393 percentage points. This differs from the paper's 93.12 +/- 0.29%. It is arithmetic on old receipts, not a new evaluation and not certification that the checkpoint produced them.

## Separate BAA, GNOMix and empirical endpoints

Supplementary Note 3.1's BAA-excluded four-path classifier is a separate fit, reported at 94.3 +/- 0.27%. The six-path classifier restricted to the four non-BAA paths is reported at 94.6 +/- 0.46%, which is the relevant context for the main article's 94.6% headline. The project's derived founder view, (p1+p5,p2+p6,p3,p4), is neither of those conditions. No complete BAA-excluded runner was recovered among the 64 reachable paths.

The GNOMix comparison uses EHG/WHG/CHG/Ana references, excludes BAA queries, applies logistic-regression base and xgboost smoothing, uses 0.02 cM windows for GBR and 0.2 cM for Bronze Age/Neo/Yam, and summarizes five 200 Mb parents at spaced sites. The complete pinned executable evaluator, query-specific command configuration and site manifest are unavailable in the bounded source scope. Published GNOMix values remain cited results, not newly reproduced results.

The paper describes fine-scale-map Relate inference and painting of every autosome in a 476-person diagnostic panel. Broader 963-person admixture-time, 1,015 ancient-genome and 1,518-person selection/PCA scopes must not be conflated. GRCh37 and published sample sources are specified, but complete final empirical VCF/QC/phase/map/sample-order/model joins remain unavailable. The named historical roster is partial progress on identity, not that full manifest.

## Every current downstream file and its practical limit

paint.py writes a populated [trees,haplotypes,3] _painted array. Its _intervals array is actually [haplotypes,5,trees], not README's stated dimensions, and never assigns right bp, path or confidence. Previous-left values propagate the still-zero right slot. The downstream interval output therefore does not implement its stated contract.

admixture.py consumes those intervals, combines homologous chromosomes and calls the R fitting functions for Bronze Age, Yam and Neo. It excludes GBR and other specified groups; the advertised present_day branch is absent and unhandled groups can retain zero initialization. Broad exception handling omits fit-failure reasons. The final weighted combination of the two ancestry estimates and addition of sample age described by the paper are not present in the supplied runner.

exp_fit.R and exp_fit_bronze.R use inverse variance weights, genetic distance in Morgans and alpha*exp(beta*x)+theta, returning signed beta and theta. Positive time-since-admixture conversion and final date aggregation need an explicit downstream definition. tracts_prob.py samples a random unseeded phase for each of 1-50 cM and steps 1 cM; its unused Python fitter differs from R's weighting and prints rather than returns a fit. These are not ancestry-accuracy metrics.

chunk.py can omit a final singleton tract. find_nearest.py implements right-endpoint search and end clamping rather than closest-distance lookup. painting.R uses the genetic-distance column while labeling the multirow x axis Position(bp). All these files received full content review and remain unchanged. Future repairs require isolated adapters and named downstream conditions, not silent changes to author algorithms.

Generic 20 Mb demographic/path-count/model/sampling sensitivity runners described in Notes 3.2-3.3 were not recovered. The complete historical example/pipeline_slurm.sh contains a commented call to external paint_num_paths.py, which is absent from the reachable path union. This is a precise lead, not recovered executable material.

## Remediation, inactive baseline and commit scope

The agreement matrix binds every finding to exact paper quotes, code files/hashes/lines, author behavior, maintained behavior and a concrete action. It distinguishes source conflict, explicit reconstruction choice and unavailable provenance. Future integration should correct stale “CNN not recovered” metadata, add the separately named inactive historical CNN condition, retain existing Dense identity and shared frozen support, and require an independent source/runtime review before any execution.

This pass made no production or author-source edits, installed nothing, imported or executed no author code, built no model, read no checkpoint numerical values, ran no simulations/training/predictions/scoring/tests, used no GPU, signalled no process, and made no queue/policy/Git changes. All writes are exclusively under research/reviews/author_repository_file_agreement_20261023_v2/ancestralpaths.

Suggested compact commit scope for ROOT: this exclusive folder only, including the five requested primary review files, CSV coverage, inactive CNN specification, exact reconstructed text and raw-source receipts, data consistency evidence and the plaintext audit builders. Exclude no historical evidence by deletion and change no production configuration in this evidence commit. ROOT owns any eventual commit/push.
