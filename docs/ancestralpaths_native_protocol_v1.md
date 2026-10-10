# AncestralPaths author-native biological protocol v1

## Scientific scope and status

The biological objective is stronger local ancestry inference on a plausible ancient-European benchmark, compared with AncestralPaths and separately qualified ARGMix/ARGformer controls. The native AncestralPaths condition must preserve its author-defined demographic populations, observed cohorts, inferred genealogies, ordered GNN features and path labels. Existing project experiments using 47 features, a four-source head, or the stdpopsim K4 oracle are different conditions. They cannot establish an exact AncestralPaths reproduction.

This document distinguishes the published protocol, the publicly archived runnable scripts, and our explicitly declared bridge changes. No upstream simulation, Relate inference, model training, cohort scoring or real-genome analysis has run as part of this bridge. Passing the CPU tests is implementation evidence only. The native feature/model adapters remain subject to parent review before upstream execution and dependency changes.

## Sources, provenance and licensing

The paper is Irving-Pease et al., **The selection landscape and genetic legacy of ancient Eurasians**, Nature 625, 312–320 (2024), DOI [10.1038/s41586-023-06705-1](https://doi.org/10.1038/s41586-023-06705-1), published 10 January 2024. The authoritative methodological description is Supplementary Note 3, especially printed pp25–34 and 43–44. Figures S7, S10, S11 and S16 provide demographic, feature and confusion-matrix definitions.

Source snapshots, acquired on 2026-10-10 without code execution:

| Material | Repository-relative path | SHA256 |
|---|---|---|
| Article XML | .runtime/author_sources/ancestralpaths_paper_PMC10781624.xml | 1a8ecd1ff10d3a181bf6d59edc23d2d58cc0c3a2c0cde21d3c7646f36ad43580 |
| Supplement PDF, 38,478,745 bytes | .runtime/author_sources/ancestralpaths_supplement_41586_2023_6705_MOESM1_ESM.pdf | 0558218aacc6438427e2d41226b88a58e5dadc82d02b97fe019ad3aa84237a3c |
| GitHub source archive, 90,010 bytes | .runtime/author_sources/ancestralpaths_77db3cfae100.tar.gz | 9624135ddb0d0ea3e2275a4ef463cbd0b62ecd13625cf2b2f3de653d778ba3a7 |
| Paper-cited Zenodo archive, 96,075 bytes | .runtime/author_sources/ancestralpaths_published_zenodo8319452.zip | 4d58adf9633d96ff9ef3b7f600d6192b4207c2e1551c1175367596e83b774e85 |

The article XML came from https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10781624/fullTextXML. The supplement came from the Europe PMC supplementaryFiles service for PMC10781624. Only the relevant supplement PDF was retained from that 100,186,120-byte bundle. No supplementary cohort spreadsheet was extracted or read.

The current source pin is [AliPearson/AncestralPaths](https://github.com/AliPearson/AncestralPaths), commit **77db3cfae100641c35218b87e1451e14f3074666**, extracted under .runtime/author_sources/ancestralpaths_77db3cfae100. Its source receipt records the codeload URL and archive hash. The paper's Code Availability instead cites [Zenodo 8319452](https://doi.org/10.5281/zenodo.8319452), titled *ekirving/AncestralPaths: v1.0.0*, version accepted, dated 2023-09-05, related to https://github.com/ekirving/AncestralPaths/tree/accepted. The downloaded Zenodo MD5 bd73e02921a2724991cdf9f74e8703dc matches its published metadata.

**All 17 original files in the current pin and paper-cited archive are byte-identical.** This comparison includes README, simulation, training, testing, painting, Relate scripts, helper scripts, example ages and schematic. The extra ARG_GNN_SOURCE_RECEIPT.json is project provenance, not an author file. Thus script/publication differences below are demonstrated differences between text and available scripts, not an assumed later-code change. The checker pins every original file separately and can check both archives.

The article is CC BY 4.0. Neither pinned code archive contains LICENSE/COPYING, and Zenodo metadata identifies only **other-open**, not a named code licence. Do not infer an MIT/GPL licence or code redistribution permission from article licensing. There is no supplied dependency lock or trained H5 model in either archive. Source provenance and licence uncertainty must accompany any baseline claim.

## Published protocol versus public scripts

| Item | Supplementary Note 3 | Pinned public source |
|---|---|---|
| Simulation length | Three 200Mbp training tree sequences, plus five additional test sequences, printed p32 | Ten 2,000,000bp replicates in simulate.py |
| Training examples | 10,000 GNN/true-label pairs per five admixed cohorts, 50,000 total, p32 | 487,500 preallocated rows over parents 3,5,6, with class-dependent quotas |
| Parent independence | Five **more** test sequences, p32 | Train 3,5,6 and test 5,6,7,8,9, overlap 5 and6 |
| Representation | First five informative ancestor GNN rows, cohort proportions and node age, -15 root padding, pp30–32 | 5×9, eight observed ancient cohorts plus age/1500 |
| Network | Text calls it a convolutional neural network, p31, without enough layer detail to reconstruct one | Flatten, Dense512/ReLU, Dropout0.2, Dense6/softmax, five epochs |
| Main result | Overall 93.12% with across-parent SD0.29%, p32 | No result generated by this project |
| Four-European-path result | BAA-excluded model94.3%±0.27%. Six-path model averaged over four non-BAA paths94.6%±0.46%, pp43–44 | Script trains six paths. Our probability collapse is an extra endpoint, not the BAA-exclusion experiment |

The paper's headline94.6% is an author simulation result, not accuracy on real genomes and not a result of this project. Its GNOMix comparison on five200Mbp sequences reports AncestralPaths accuracies GBR91.8%, Bronze93.5%, Neo95.5%, Yam95.2%, printed p34. These are different evaluations from the overall six-path result. Published quantitative reproduction requires resolving the training-example and architecture discrepancies, not silently inheriting whichever script is convenient.

## Native demography and observed sample order

The public generator declares 11 populations in this order:

| ID | Name | Initial diploid population size |
|---|---|---:|
| 0 | present_bronze | 50,000 |
| 1 | Yam | 20,000 |
| 2 | Neo | 50,000 |
| 3 | EHG | 3,000 |
| 4 | WHG | 2,000 |
| 5 | NE | 15,000 |
| 6 | BAA | 20,000 |
| 7 | CHG | 3,000 |
| 8 | Ana | 5,000 |
| 9 | WA | 15,000 |
| 10 | trunk | 15,000 |

Backward admixture present_bronze→Neo/Yam has proportions0.5/0.5. Neo→WHG/Ana has0.25/0.75. BAA→CHG/Ana has0.25/0.75. Yam→CHG/EHG has0.5/0.5. WHG/EHG merge into NE at600 generations, CHG/Ana into WA at800, and WA/NE into trunk at1500. The generator derives each admixture time as max age of that cohort plus1, so input dates define part of its demographic configuration. For the supplied example, Bronze166, Yam177, Neo259 and BAA169 generations. Published Figure S7 shows BAA170, so substituting the figure value would be an explicit change.

The source uses constant recombination1e-8 and mutation1.25e-8, census350, msprime default ancestry model/ploidy behavior with explicit diploid SampleSets, and no explicit seeds. It exports a Demes YAML model. This11-population/BAA setting is distinct from the project's unchanged AncientEurope_4A21 K4 catalogue condition. Do not replace the source demography with that catalogue model while retaining the native-reproduction label.

The age file's row order is **not** the simulation sample order. simulate.py collects cohort-specific age lists and concatenates SampleSets as below. The provided GBR age rows are ignored: 91 present-day diploids at time0 are hardcoded separately.

| Simulated cohort order | Diploids | Haplotypes, inclusive IDs | Example age range, generations |
|---|---:|---|---|
| present_day, reported GBR in training | 91 | 0–181 | 0 |
| Bronze_Age | 81 | 182–343 | 100–165 |
| BAA | 7 | 344–357 | 136–168 |
| Neo | 151 | 358–659 | 139–258 |
| Yam | 13 | 660–685 | 153–176 |
| WHG | 51 | 686–787 | 210–529 |
| EHG | 43 | 788–873 | 155–402 |
| Ana | 25 | 874–923 | 217–360 |
| CHG | 14 | 924–951 | 184–473 |

This totals476 diploids/952 haplotypes. Within a cohort, original age-file order is preserved. The exported .ages file follows haplotype-node order and duplicates each diploid's date. .poplabels is one row per diploid, header sample population group sex, with names tsk_0 etc. The example includes references older than census350, which is permissible: native scored admixed query cohorts are younger than350. No rule should incorrectly reject all ancient references older than350.

Training hardcodes these bounds even though the README says sample numbers are configurable. Testing derives min/max ranges from .poplabels, which assumes cohorts remain contiguous. Real painting instead joins each diploid name to the sample-order file. The bridge requires an explicit ordered diploid/haplotype manifest and joins names rather than guessing inferred node IDs. Both haplotypes of a diploid must share their observed cohort. Unknown population strings require explicit declaration as an other/present-day cohort. A typo cannot silently become group8.

## Observed features and truth-only labels

Feature columns are **Bronze_Age, BAA, Yam, Neo, WHG, EHG, Ana, CHG, ancestor age/1500**. This is different from the simulator's cohort concatenation order, especially Yam versus Neo and Ana versus CHG.

Starting at the focal haplotype's parent, collect descendant ancient-cohort leaves not included in any previous informative row. Present-day/group8 leaves are excluded globally. Skip any ancestor adding no new ancient leaf without consuming a feature row. Record normalized cohort counts and node age for the first five informative ancestors. At the root, fill every unvisited9-column row with-15. Normalization uses all new ancient leaves in the shell, not per-cohort normalization and not nearest-source votes.

The source includes an ancient focal sample itself in its first shell if its observed cohort is one of the eight ancient groups. This is an observed cohort covariate, not local ancestry truth, and the bridge retains it. All sampled groups, including admixed Bronze, BAA, Yam and Neo, contribute as observed cohort identities. Pure-source-only panels are an explicitly different reference condition.

training.py/testing_bygroup.py use int(age)/1500, which discards fractional estimated generations. paint.py uses float(age)/1500 and float32 storage. extract_native_features exposes training_int and painting_float explicitly. It reads only parent links, leaves, observed times and the observed cohort manifest. It has no target argument and never reads internal population, flags or metadata. Inference must pass the Relate-derived observation. Passing a census-bearing truth tree would expose artificial unary-node/interval information even if population labels were ignored, so truth files must remain outside the inference capability.

The author truth path is the population of the focal sample's simulated ancestor at census350, conditional on the observed BAA query cohort. The bridge derives the following table from the simulation's population declaration order and the training/testing AST, without importing or executing source:

| Native path,1-based | Query condition | Census ID/name | Four-source class,0-based |
|---|---|---|---|
| 1 | non-BAA | 8/Ana | 0/Ana |
| 2 | non-BAA | 7/CHG | 1/CHG |
| 3 | non-BAA | 4/WHG | 2/WHG |
| 4 | non-BAA | 3/EHG | 3/EHG |
| 5 | BAA | 8/Ana | 0/Ana |
| 6 | BAA | 7/CHG | 1/CHG |

This matches the path descriptions in printed p27/Figure S7:1 travels through Neo/Ana,2 through Yam/CHG,3 through Neo/WHG,4 through Yam/EHG,5 and6 begin in BAA and join the respective Ana/CHG routes. Internal inferred-population labels cannot provide targets. Native targets come from the paired **true** simulated tree at exactly the same physical coordinate.

The robust truth helper checks a flagged census ancestor, exact time350, author population order and query dates. It rejects missing/unexpected ancestors rather than using the source's unbounded while loop or stale path variable. This guard is an explicit correctness extension. Truth-to-inferred node joins must be saved externally by sample name/haplotype, not presumed from IDs. Census nodes/flags/metadata must not enter observations, and truth-only segmentation must not define inference windows or candidate coordinates.

## Relate observation chain and required outputs

relate.sh converts the phased VCF using ConvertFromVcf, generates SNP annotations with .poplabels, runs RelateParallel with mutation1.25e-8, initial-N30000, eight threads, ordered .ages and genetic_map_simulated.txt, then EstimatePopulationSize with five iterations/.poplabels, then ConvertToTreeSequence. Features are extracted from the **post-population-size-estimation** *_relate_popsize_i.trees output, not the simulated tree or initial Relate trees.

There is no author Relate version pin. A reviewed runtime must record exact Relate executable/version/source, dependency versions, CLI arguments, map, mutation rate, SNP annotation inputs, sample-age order, conversion/site mapping, ancestral polarization and retained SNPs. Keep .haps, .sample, .annot, .anc/.mut or equivalent intermediate outputs, post-size trees, model checkpoint, full softmax output and provenance. Generating SNP annotations from observed cohort labels is allowed. Supplying census/local ancestry labels as annotations is forbidden.

VCF positions are integer/1-based while tree coordinates are physical floats. Freeze and validate an explicit site-coordinate map before comparing inferred and true trees. The author's testing uses inferred SNP positions to look up true labels. Do not change this convention silently or score a nearest-truth-tree surrogate. Record complete inference coverage and failures before scoring.

## Native model, probabilities and metrics

The source's actual model is Flatten(5×9), Dense512/ReLU, Dropout0.2, Dense6/softmax, Adam and final categorical cross entropy, trained for five epochs. An initial compile with sparse categorical loss is immediately overwritten before fitting. Labels are one-hot(path-1,6). Batch size32 and shuffle true come from Keras fit defaults, not literal source arguments. No source training seed is specified. native_model_spec records this distinction. build_native_keras_model accepts an explicitly reviewed Keras API and does not import TensorFlow itself.

Training quotas per parent are present/GBR50,000, Bronze50,000, BAA12,500, Neo25,000 and Yam25,000; three parents total487,500. Non-BAA per-path caps12,500 and BAA caps6,250 enforce class balance within the pertinent admixed cohorts. Candidate sites depend on the inferred site count and site ID modulo a step. The while-loop rescans and increases step by1.5 if quotas are unmet, potentially reusing examples. Zero steps/infeasible quotas and duplicate tuples need a reviewed guard. Label balancing is a **training** choice, not a permitted held-out parent or interval selection rule.

Testing collects10,000 examples per cohort/parent, five cohorts×five parents, and pools within-cohort confusion counts. Testing's class caps25,000/50,000 exceed its10,000 target so do not balance its class frequencies. Native accuracy is unweighted categorical sample accuracy at selected SNP/focal pairs. Its confusion call passes predicted labels first and truth second, so rows are **predicted paths**, columns **true paths**. Figure S11 normalizes each predicted row to show precision. The bridge retains this orientation and six fixed axes, padding absent classes with zero counts. Zero-row precision must be marked undefined if computed later.

cvscores has tree-sequence rows and cohort columns, although README describes the opposite. The public .kfold.log averages axis1, which averages cohorts within a parent, while later indexing is described per population. Preserve the raw parent×cohort table and label axes explicitly. A corrected per-cohort summary across independent parents is an explicitly named reduction fix.

The native six-path result is primary. Preserve every softmax probability, position, ordered query/sample identity, parent identity, feature convention and source pin. predict_native_probabilities replaces removed predict_classes with predict on[N,5,9]; argmax preserves the author's hard-class semantics. The author painting file retains only the winning label and max probability. The bridge adds a complete[N,Q,6] probability export instead of pretending that winning probability reconstructs the omitted classes.

The **secondary four-source endpoint** sums probabilities **p1+p5, p2+p6, p3, p4**, then takes argmax if needed. It does not collapse the native argmax, remove BAA examples, retrain with four outputs, or reproduce the author's BAA-excluded4-path experiment. Ana can have the highest summed source probability even when CHG has the highest individual native path. Any additional uppercase ANA alias used by the common benchmark must be explicit in its ontology receipt.

Physical-span accuracy, calibration, class-balanced scores, dosage, tract and boundary metrics may be useful common endpoints. Report them separately from native sample accuracy and precision. Estimate uncertainty by independent parent replicate, not sampled-site count. Keep errors, low-confidence predictions and mixed intervals in evaluation.

## Mechanical compatibility fixes and execution review

Source inspection alone establishes that pipeline.sh passes -poplab to training.py, which lacks that parser argument. Training depends on hardcoded952-haplotype bounds. Testing and painting reshape[N,1,5,9] despite the model's[N,5,9] input. They import keras.backend.tensorflow_backend and call removed predict_classes, and attempt private _LOCAL_DEVICES GPU inspection. paint.py calls legacy msprime.load. All sources import the old Convolution2D alias although the Dense model does not use convolution.

No single compatible runtime has been verified. Package metadata establishes a conflict: installed tskit1.0.3 requires NumPy>=2, whereas TensorFlow2.15.1 requires NumPy>=1.23.5,<2. An isolated candidate is Python3.11, NumPy1.26.4, TensorFlow2.15.1, Keras2.15.0, tskit0.6.4, msprime1.3.4 and demes0.2.3. PyPI metadata for tskit0.6.4 requires NumPy>=1.23.5; msprime1.3.4 requires NumPy>=1.23.5, tskit>=0.5.2 and demes>=0.2. These are metadata-compatible bounds, not installed-wheel ABI/API validation. Do not downgrade the current project environment. This candidate can express the native Dense/Dropout architecture, but this does **not** imply unmodified scripts run. Root must review an external adapter and verify relevant APIs before choosing an environment. Do not monkeypatch private GPU APIs or modify the pinned author files. The bridge uses direct per-focal extraction, proper rank3 inputs, predict+argmax and tskit-owned observed tree loading in callers. Per-query extraction also avoids paint.py's sibling shortcut, whose -15 padding is only applied to the first sample in a batched sibling group.

paint.py allocates an interval array but does not populate the right physical bound, path or confidence columns before saving. Its README's interval dimensions also disagree with source. That broken interval export must not feed a scientific tract/admixture evaluation. Our export carries explicit positions/full probabilities; a future properly specified interval export must obtain half-open observed tree intervals from the observation and preserve every native probability.

Required dependency/API review: numpy, tskit, msprime, demes, tqdm; TensorFlow/Keras for training/inference; pandas for real painting map inputs; Relate executable plus named helper scripts. scipy, rpy2 and R4+ are relevant to author admixture-time analysis only, not the core local ancestry bridge. R/rpy2 is not imported here. check_ancestralpaths_source.py reads installed package metadata without importing TensorFlow, Keras or author scripts. It reports runtime_compatibility_verified:false until real API checks qualify a separate runtime receipt. It never downloads dependencies or launches a pipeline.

Before execution, review source hashes, dependency/environment writes, parent/sample split, demography, map/site convention, target isolation, labels, architecture and any adapter diff. All execution/storage remains inside /workspace-SR003.nfs2/ARG-GNN. Primary agent owns installs, commits, Relate jobs and GPU scheduling. This bridge requests none of them.

## Independent-parent and common-benchmark conditions

The current-source diagnostic split contains two overlapping parents and cannot support an independent test claim. validate_parent_partition rejects any train/validation/test parent overlap and duplicate IDs. Split independent simulation parents and subjects before site selection/windowing. Record new seed streams, separate training/validation/test parents, eligibility failures and all retained parents before accuracy is known.

An independent-parent fix must be named as such. The paper's three training/five additional test scheme is the natural native-protocol target, while a validation cohort is an additional transparent tuning condition. Do not recycle source parent5/6 predictions as held-out evidence. Preserve the original source split only as a quarantined source-behavior diagnostic, not a leaderboard result.

For a common paired biological evaluation, all methods may share observed inferred trees, sample dates/cohorts, query identities and predefined coordinates, while using their own native input representations. AncestralPaths remains an inductive native5×9/six-path classifier trained on separate simulated parents. The new method may use graph features but may not inject those47 features into the baseline or replace its head and call it exact. ARGMix/ARGformer controls require separate paper/source qualification. ARGformer's graph-specific node embeddings, masked-node learning and contrastive retrieval may require a declared transductive observed-only adaptation regime and an explicitly downstream readout. That is distinct from the AncestralPaths inductive task.

If a common panel uses only24 pure-source references/64 queries, absent native Bronze/BAA/Neo/Yam reference cohorts reduce the available native information. Report that as an altered-panel robustness condition, not the author-native European benchmark. Keep the fully specified476-diploid cohort/ages condition as the native-reference benchmark. A smaller diagnostic may test execution correctness but cannot inherit the paper's numerical claim.

## Real-data availability and evaluation feasibility

The article's Data Availability says all ancient genomes are previously published and listed in Supplementary Table1, aligned to GRCh37. It lists modern genomes from1000 Genomes, SGDP and HGDP, and GWAS inputs from GWAS Catalog, FinnGen and UK Biobank. Supplementary Note3 describes imputed/filtered MesoNeo genomes and476 diagnostic diploids including91 GBR samples. The main article's broader1015-genome selection analysis and >400,000 UKB chromosome painting are different scopes. In particular, the UKB large-scale painting uses ChromoPainter/GLOBETROTTER components and cannot be attributed automatically to AncestralPaths.

Existing public genome availability makes a phased/dated ancient-plus-GBR Relate study plausible, but the code archive alone supplies neither the complete phased input dataset nor trained checkpoint nor fully reconciled diagnostic sample manifest. Raw published reads, imputed/phased products and reuse terms must be checked accession by accession. No actual cohort/held-out target files or Supplementary Table1 were opened during this implementation. UKB access is controlled and unnecessary for an initial public pilot.

A real-data pilot should first reconcile a small public GRCh37 phased cohort with observed ages, sample IDs, coverage/quality criteria, genetic map, ancestral polarization and population labels, and run a reviewed Relate chain. Apply the trained classifier using the same feature conventions and export full six-path probabilities and uncertainty. Compare ancestry proportions and tract patterns against prespecified independent biological/LAI evidence. There is no per-site ancestry ground truth for real genomes, so such analyses assess concordance, plausibility and robustness, not supervised real-data accuracy or an inherited94.6% claim. Keep diagnostic source-proxy cohorts separate from evaluation subjects when possible.

Include imputation/genotype errors and phase-switch perturbations as separately named simulation conditions. The paper reports imputation and phase limitations. Phase swaps change haplotype identity, so evaluate diploid dosage or preserve an external phase-truth alignment; do not pass that hidden alignment as a model covariate.

## Bridge API and CPU validation

src/arggnn/ancestralpaths_native_bridge.py provides source byte/AST inspection, immutable observed panel parsing, native5×9 features, model specification/reviewed-Keras builder, six-probability prediction/export, secondary source collapse, native accuracy/confusion, independent-parent checks and separate truth-only census helpers. scripts/check_ancestralpaths_source.py is a read-only CLI. It can check both source trees and archives. Model construction is exercised only against a fake Keras API in CPU tests, not real TensorFlow.

tests/test_ancestralpaths_native_bridge.py uses handmade tree tables rather than simulation execution. Checks cover exact shell/column order, skipped ancestors, non-recounting, ancient self-inclusion, root padding, age/dtype variants, no truth reads, hidden census counterfactuals with a merged observation tree, malformed census rejection, explicit sample-name/order joins, source drift, derived six-path mapping, probability collapse before argmax, asymmetric native confusion orientation, all-probability export without pickle/overwrite, rank3 adaptation and rejection of source parent overlap.

Recorded test counts, logs and validation hashes belong in the separate CPU receipt. CPU checks cannot qualify GPU metadata, runtime compatibility, native numerical reproduction or a biological superiority claim. Older custodian/audit source and frozen evidence remain untouched.
