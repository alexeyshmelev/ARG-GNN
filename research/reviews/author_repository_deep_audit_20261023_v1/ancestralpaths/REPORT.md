# AncestralPaths author repository deep audit, 23 October 2026

## Main finding and correction

The public history contains a genuine convolutional classifier. The earlier categorical claim that the paper CNN definition was unavailable is incorrect. Current main and the paper-cited accepted release contain a Dense classifier, but commit `77db3cfae100641c35218b87e1451e14f3074666`, entitled **Simplified neural net**, explicitly replaces the prior CNN. The previous commit `e6f5d9c6be628007c662e1c71a07358cf350efaf` supplies its complete source. Historical example checkpoint and data assets are also still publicly addressable, although they are absent from the accepted archive. These discoveries recover concrete author materials without proving that any particular historical artifact generated the paper tables.

No author module was imported or executed. No training, simulation, scoring, GPU use, installation, Git action, process control, policy edit or queue edit occurred. Writes are confined to this exclusive audit directory. The only genotype-file read was a VCF header, stopping at `#CHROM` before any genotype row. After the initial source audit, ROOT explicitly authorized downloading the one255,064-byte historical checkpoint as data and reading only HDF5 metadata. No runnable model was deserialized and no numerical weight or optimizer dataset value was read.

## Fresh repository and release census

At approximately 20:21–20:26 UTC on 23 October 2026, fresh public requests established the following:

- AliPearson/AncestralPaths has one visible branch, `main`, at `77db3cfae100641c35218b87e1451e14f3074666`, no tags and no releases. Its branch history lists 27 commits without a next-page Link.
- The paper-cited publication fork, ekirving/AncestralPaths, has the same sole branch and 27 commits. Its `accepted` tag points to that same commit. Its sole release, v1.0.0, was published on 5 September 2023 with the description “Published version of the LAI software”; the release assets array is empty.
- Both recursive current trees are untruncated. All 17 upstream files match the unchanged local pinned bytes by Git blob SHA1 and separately recorded SHA256. No notebook, requirements/environment file or separate four-path training runner appears in those current trees. `testing_bygroup.py` is the native evaluator, not a unit-test suite.
- Zenodo record8319452 lists one 96,075-byte accepted ZIP. Its versions endpoint reports one version total, and its files endpoint lists no additional model/data file. Local ZIP SHA256 is `4d58adf9633d96ff9ef3b7f600d6192b4207c2e1551c1175367596e83b774e85`; MD5 matches the publisher. All17 ZIP payload files are byte-identical to the pinned current source.
- The complete reachable-history path union has64 paths, obtained from25 successful plaintext commit patches plus complete metadata for the two commits whose patch/diff routes returned403. This includes deleted example assets. This census makes no claim about private repositories, unreachable commits, deleted branches or unpublished author files.
- Fourteen GitHub API calls were used: an initial publication-fork repository request, eleven branch/tag/release/history/tree/issue requests, and two decisive commit-detail requests after the corresponding patch and diff endpoints returned403. All14 API requests returned200. Rate-limit headers, HTTP status, timestamp, exact URL, response bytes and SHA256 are retained. The counter is shared with other clients, so its change is not a request-count measurement.

Evidence: `source_census.json`, `current_tree_local_verification.json`, `zenodo_archive_comparison.json`, `reachable_history_path_census.json`, `historical_file_census.json` and the individual response/receipt files.

## The recovered historical CNN

Canonical recovered source: [training.py at e6f5d9c6](https://github.com/AliPearson/AncestralPaths/blob/e6f5d9c6be628007c662e1c71a07358cf350efaf/training.py). Saved as `history_source/previous_training.py`, SHA256 `fc90bb333e3589f3e4f76098dfc591222de1e40c7382a3f367e876270393b4eb`.

Decisive lines are:

- Line23 sets `backend.set_image_data_format('channels_first')`.
- Lines199–208 reshape to `(N,1,5,9)`, apply `Convolution2D(32,(5,1),activation='relu',input_shape=(1,5,9))`, Flatten, Dense64/ReLU and Dense6/softmax.
- Lines203 and210 encode `labels-1` into six one-hot classes and use categorical cross-entropy with Adam.
- Line211 fits batch30 for five epochs using the historical `nb_epoch` spelling. There is no validation split, early stopping, validation checkpoint selection or explicit random seed in this runner. The final model is saved to HDF5.

This is a real CNN, not Dense terminology disguised as convolution. Under the explicitly declared channels-first convention, its input is one channel, five ordered informative ancestors and nine features. The5×1 kernel spans the five ancestors while retaining nine feature positions; default valid padding yields32×1×9 activations before Flatten. Assuming ordinary historical Keras defaults, the topology has19,078 parameters. This is an analytic count, not a runtime measurement. Reinterpreting the input as channels-last would change the model and must not be a silent port.

The network already occurs in `training_balanced_class_BAA.py` at first source-bearing commit `d0c2b07502901a43e0ac80583616c942983a93de` (3 May2022), SHA256 `ddb2868de6bb71f988eac5077f079cd63fe3bd6634d833b1374383b8c75f9ba6`. It survives the October2022 versions, including the poplabel-driven version added around `9d808393`, and remains unchanged in topology through `e6f5d9c6`. The June2023 head patch removes the CNN and replaces it with Flatten(5,9)→Dense512/ReLU→Dropout0.2→Dense6/softmax. It simultaneously removes training’s `-poplab` argument, hardcodes sample boundaries, changes TRAIN sequence IDs and changes model fitting. Therefore head/accepted is a materially different scientific condition.

The recovered source alone does not pin Keras/TensorFlow versions. The historical example checkpoint subsequently confirms Keras2.3.1 with TensorFlow backend, while the TensorFlow version and complete runtime closure remain unpinned. The old code uses `keras.backend.tensorflow_backend`, a private backend hook, `nb_epoch` and `predict_classes`; current API/runtime compatibility is unqualified. Source recovery does not establish an immediately runnable environment.

## Exact simulator, cohort and Relate route

The current simulator and its e6f5 CNN-era counterpart are byte-identical. They define eleven populations: present_bronze, Yam, Neo, EHG, WHG, NE, BAA, CHG, Ana, WA and trunk. Population IDs are respectively0–10 in that order. Census truth is recorded at generation350. Admixture proportions are Neo=(WHG0.25,Ana0.75), BAA=(CHG0.25,Ana0.75), Yam=(CHG0.5,EHG0.5), present_bronze=(Neo0.5,Yam0.5). WHG/EHG split at600, CHG/Ana at800 and the two ancestral branches split at1500 generations. Recent admixture times are maximum supplied ancient age plus1.

The age file has476 rows: GBR91, Bronze_Age81, BAA7, Neo151, Yam13, WHG51, EHG43, Ana25 and CHG14. The simulator creates91 present-day diploids explicitly and reads ages for the eight ancient groups. Its concatenation order is present-day, Bronze_Age, BAA, Neo, Yam, WHG, EHG, Ana, CHG, yielding952 haplotypes. The file’s physical row order is different; simulator concatenation, not a naive age-file row assumption, determines sample IDs.

`simulate.py:119–129` currently creates ten2Mb replicates with constant recombination1e-8, mutation1.25e-8 and no explicit simulation/mutation random seeds. A historical `simulate.py` at `2bd4bd7c...` does contain `num_replicates=3` and `sequence_length=200000000`, matching the paper’s stated scale for training simulation. However that same historical training source requests five sequence IDs, so this is not a self-consistent complete paper recipe. At the recovered e6f5 CNN snapshot the simulator is already ten2Mb.

`relate.sh:5–9` performs VCF conversion, SNP annotations, RelateParallel with mutation1.25e-8, starting N30000, sample ages, a constant map, eight threads, then EstimatePopulationSize with five iterations, and only then converts the post-population-size result to a tree sequence. Its exact Relate build is not pinned. The constant map in current main is0–2Mb at1cM/Mb. An older example map is0–200Mb at1cM/Mb; using that map does not by itself change the simulated sequence extent.

The current `pipeline.sh` passes `-nn5 -poplab` to training. That matches the historical CNN parser, but the accepted Dense parser rejects `-poplab`. Both current and historical testing retain the CNN-shaped input. The current training default `-nn4` also conflicts with its hardcoded5×9 reshape unless the caller explicitly supplies5. Do not treat the current pipeline as a qualified end-to-end command.

## Features, labels and native sampling/evaluation

Historical CNN training uses eight ancient population-proportion columns in this order: Bronze_Age, BAA, Yam, Neo, WHG, EHG, Ana, CHG. The ninth column is the ancestor age. For each focal haplotype, it walks upward from its parent and gathers previously unseen ancient reference leaves. Empty reference shells are skipped without consuming a row. Proportions normalize by newly encountered reference-leaf count. Training/testing store `int(age)/1500`; missing informative rows are padded with−15. This is an ordered5×9 feature matrix, not a graph convolution on the original genealogy. Query truth is obtained separately from matching-position simulation census nodes.

The six native path labels are:

| Native path | Census population and query condition |
| --- | --- |
|1 | Ana, non-BAA query |
|2 | CHG, non-BAA query |
|3 | WHG |
|4 | EHG |
|5 | Ana-origin BAA query |
|6 | CHG-origin BAA query |

The CNN-era runner obtains roster boundaries from the poplabel file, but assumes each group occupies a contiguous sample block. Its training sequences are0–4. Each sequence targets50,000 GBR rows,50,000 Bronze_Age,12,500 BAA,25,000 Neo and25,000 Yam, totalling162,500 per sequence and812,500 preallocated rows overall. Per-path caps are12,500 or6,250 for BAA. The sampler visits sites selected by `site.id % step ==0`, with initial step derived from observed site count and sample/quota counts. If quotas are unmet it traverses again and increases step by half; this can revisit source examples and has no explicit pass bound. These are source quotas, not verified realized-row or unique-row counts.

The accepted Dense version instead uses TRAIN IDs[3,5,6], with487,500 preallocated rows. Native testing uses IDs5–9, so that accepted source has train/test parent overlap5 and6. The recovered historical CNN uses0–4 versus5–9, eliminating that specific overlap. This distinction materially improves fidelity and must be preserved.

`testing_bygroup.py` targets10,000 rows per query population per TEST sequence: five populations×five sequences=250,000 allocated rows. It uses the same census labels and feature extraction, reshapes `(N,1,5,9)`, evaluates accuracy and saves the raw5×5 `cvscores` matrix. Its confusion call passes predicted classes first and truth second, so saved matrices have predicted rows and true columns. The `.kfold.log` code applies mean/std over axis1, which is the population axis of each sequence row, while its surrounding loop is described by population. That log cannot silently be used as the paper’s per-population variation across sequences. The raw matrix permits separately specified summaries. No GNOMix command or standalone BAA-excluded model runner was found in the enumerated current files and64 reachable-history paths.

A common C4 output `(p1+p5,p2+p6,p3,p4)` means Ana,CHG,WHG,EHG and is a derived endpoint. Supplement3.1 instead removes BAA reference contributions and labels5/6 before fitting a four-path model. Merging six-path probabilities does not reproduce that separately trained condition.

## Historical checkpoint and example data recovered as leads

[Commit9d808393](https://github.com/AliPearson/AncestralPaths/commit/9d808393b1b0662e65158e390d92936b954ca1b8) adds `example/model_test.h5`, ten true `.trees`, ten VCFs, ten post-popsize Relate `.trees`, sample ages/poplabels/demography and confusion/accuracy receipts. [The next commit4c79af9a](https://github.com/AliPearson/AncestralPaths/commit/4c79af9acf2b5c83de67a5201316b706e7d6bb33) removes42 example assets. Their exact Git blob identities and URLs are retained in `historical_file_census.json`.

The checkpoint’s Git blob SHA1 is `1243c02e9a6b61b304e5104430933c6669004935`. After a fresh HEAD returned200 and Content-Length255064, ROOT authorized the exact small-file download. The downloaded255,064 bytes match that Git blob exactly. Full-file SHA256 is `f5b32e639f7dce5db00bed48296b9a386aacca7c02364e3e6283f49dac951898`.

Existing project interpreter `.runtime/conda/envs/ancestralpaths_cpu_v1/bin/python3.11` supplied h5py3.16.0. A read-only HDF5 inspection parsed only the `model_config` and `training_config` JSON attributes plus layer/group names and dataset shape/dtype metadata. No Keras/TensorFlow import, `load_model`, pickle, model construction, runnable-object deserialization or numerical dataset-value read occurred. `historical_checkpoint_hdf5_metadata.json` confirms:

- Keras2.3.1, TensorFlow backend.
- Sequential input`[null,1,5,9]`, Conv2D32 with5×1 kernel, stride1, valid padding, channels_first and ReLU; Flatten also channels_first; Dense64/ReLU; Dense6/softmax. No dropout layer.
- Model-weight metadata has kernel shapes5×1×1×32,288×64 and64×6, plus biases32,64 and6. These match the recovered source and19,078 scalar trainable parameters by shape arithmetic.
- Categorical cross-entropy and accuracy; Adam learning rate approximately0.001, beta1≈0.9, beta2≈0.999, decay0, epsilon1e-7 and amsgrad=false. Initializer seed metadata is null.
- Adam optimizer groups exist, including an iteration dataset, but no optimizer value was read, so completed updates/epochs cannot be inferred from this inspection.

This is a verified historical author CNN checkpoint, not a certified paper-trained checkpoint. Its exact paper result association, training data, realized quota, optimizer history and inference performance remain unproven.

The historical `example/test_0.vcf` header returned200 and explicitly states `length=2000000` and `tskit0.4.1`. The historical example inputs therefore cannot simply be described as the paper’s200Mb inputs. The retained `test.kfold.txt` is5×5; its `.kfold.log` is empty. Simple descriptive arithmetic on that old receipt gives91.4504% equal-population mean and0.8393 percentage-point sample SD across five sequence means. This is not new evaluation, and it differs from the paper’s93.12±0.29%, so it does not certify paper result identity. The receipt’s original numbers remain unchanged.

## Comparison with primary supplement and linked sources

Cached primary supplement SHA256 is `0558218aacc6438427e2d41226b88a58e5dadc82d02b97fe019ad3aa84237a3c`; its full text was reread around Note3 and its source links. Exact excerpts with original text line numbers are in `supplement_decisive_excerpts.txt`.

- Lines998–1020 explicitly call the method a convolutional neural network and specify five informative ancestors, ancient-group proportions, ancestor age and−15 padding. Historical CNN source substantively matches this description. The supplement does not itself specify filter count, hidden width, optimizer, batch size or epochs; those are recovered code settings whose exact linkage to a paper experiment remains unproven. The separately inspected historical example checkpoint confirms the CNN topology and records Keras2.3.1 and its optimizer configuration, but does not identify the paper experiment.
- Lines1034–1049 specify three200Mb TRAIN sequences,50,000 total training pairs (10,000 per five query populations), then five additional TEST sequences with10,000 GNNs from each and93.12±0.29% accuracy. The recovered e6f5 runner’s five TRAIN parents,812,500 preallocated training rows, ten2Mb simulator and250,000 testing rows do not match that complete numeric recipe.
- Lines1090–1107 compare GNOMix on GBR, Bronze_Age, Neo and Yam, excluding BAA queries; references are EHG/WHG/CHG/Ana, with default logistic-regression base plus xgboost smoother. Window sizes are0.02cM for GBR and0.2cM for the other three. The five200Mb comparison parents and precise site spacing/command settings are not bound by the inspected code.
- Lines1409–1427 describe a distinct BAA-excluded, four-path training condition. Reported94.3±0.27% and six-path-model restricted four-path94.6±0.46% are separate endpoints. They cannot be replaced by the project’s collapsed C4 span endpoint.
- Lines884–916 describe a476-diploid cohort,91 GBR, age-matched simulation and200Mb chromosomes. Lines1133–1136 describe painting every autosome with fine-scale maps, mutation1.25e-8 and starting N30000. The main article identifies GRCh37 and published ancient data in Supplementary Table1.

The main article’s code availability explicitly assigns Note3 to AliPearson/AncestralPaths and Zenodo8319452. Other listed repositories concern Note2/9 chromosome painting and risk scores, Note4 selection trajectories, modified CLUES and Note7. The linked `ekirving/mesoneo_paper` README was freshly fetched and confirms its selection-analysis role; it is not evidence of a second ancestral-path CNN implementation.

The linked current stdpopsim `AncientEurope_4A21` source was also fetched without import. It has ten populations and no BAA, different population sizes, admixture times140/180/200, and exponential present-day growth. The archived AncestralPaths simulator has eleven populations including BAA and different sizes/times. The catalogue link therefore does not license silently substituting that model for the six-path archived recipe. Its full SHA256 and numbered function excerpt are saved.

## Real-data painting route and outstanding implementation issues

`paint.py` loads a named model and a pre-inferred tree sequence, matches diploid sample names to a supplied sample-order file, expands them into haplotypes, maps eight ancient-group labels into feature columns and treats other labels as present-day. For each marginal tree it builds the ordered shell features, uses CNN-shaped `(N,1,nn,9)` inputs, writes class+1 and maximum softmax to `_painted.npz`, and records the tree’s right coordinate. Relabeling ancient groups therefore changes model features. The sole public issue, “Impact of poplabel File on paint.py Results and Real Data Label Preparation,” asks about this exact dependence and had zero comments in the fresh issue response. It supplies no author resolution.

Static code also exposes concrete details needing qualification:

- Painting uses floating ancestor ages divided by1500, while historical training/testing use integer-truncated ages.
- In `paint.py:164–185`, `length_dist` starts at zero, left boundaries are copied from its prior right-boundary slot, and only the genetic-distance slot3 is subsequently assigned. The loop reads `path` and `softmax` into scalars but never assigns right bp slot1, path slot2 or softmax slot4. Thus the written `_intervals.npz` does not implement the documented populated interval record. This is a source-level defect; no runtime was performed. `_painted.npz` has explicit assignments and is a different output.
- The real-data caller requires exact externally assigned ancient-group labels and sample order. The roster age file provides groups/ages, not a full real subject-ID/QC/phase/map/version manifest. It is insufficient alone to reconstruct the exact empirical cohort.
- Admixture helpers consume interval paths, combine homologues and fit R exponential curves. These downstream analyses are distinct from ancestry-accuracy evaluation and do not supply ancestry truth.

## Resolved and unresolved items

Resolved by new evidence: historical native CNN definition, channel convention and code training settings; the explicit Dense replacement commit; disjoint CNN-era TRAIN/TEST IDs; previously deleted author example checkpoint/data/accuracy routes and verified CNN HDF5 configuration; precise six-path label semantics; source-level evaluator and real-painting defects; current published repository/release/version identities.

Still unresolved: exact historical commit/checkpoint/input/RNG/runtime identity behind each paper result; complete matching three200Mb/50,000-row/five-TEST execution recipe; original paper seeds and exact subject IDs; successful runnable loading and evaluation of the historical checkpoint, without treating metadata inspection as performance evidence; observed validity of historical data/checkpoint pairs; complete BAA-excluded runner and GNOMix evaluator; qualified legacy Keras/Relate environment; exact all-autosome empirical inputs/QC/phase/map joins. These are bounded-search conclusions, not universal absence claims.

The next scientifically defensible source-preparation step is to pin and review the recovered CNN as a historical author condition, keep the accepted Dense workflow separately named, and explicitly bind any paper-scale simulation/sampling changes. No replacement, integration or execution was undertaken in this audit.
