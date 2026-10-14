# ARGMix published-benchmark audit, 15 October 2026

H038 is a declared common-input development comparison, not a reproduction of the April ARGMix Tables 1–2. Current native ARGMix and native AncestralPaths preserve their released model components with disclosed input/runtime adaptations. Their panel, genealogy preparation, training units, checkpoint selection and scoring differ materially from the paper. Similar percentages would not establish reproduction or absolute fairness.

## Primary sources and versions

The inspected primary paper is *Graph transformer for ancient ancestry inference*, Shanks et al., DOI [10.64898/2026.04.05.714076](https://doi.org/10.64898/2026.04.05.714076), bioRxiv v1 dated 7 April 2026, archived as [PMC13082007](https://pmc.ncbi.nlm.nih.gov/articles/PMC13082007/). Europe PMC supplied the same-version [JATS](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13082007/fullTextXML) and [supplementary archive](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13082007/supplementaryFiles). Direct bioRxiv/PDF requests failed or returned challenge HTML. The JATS and public supported supplementary endpoint sufficed, without bypassing challenges.

Saved JATS SHA256: `383bf243185583820dbb35cb8bbf8b680be7f06197d1e26d01e29db1488c63fe`. Supplement PDF SHA256: `9216125de41efa714550a8a7673512f83e831462225613ab3fc36513c7626e72`. The supplement has seven pages: title and S1–S6 figures. Text extraction and visual inspection of S1 identify a PCA, not the sample-age/count table referred to in §4.5. It does not resolve exact ages, seeds, sequence length or the uncertainty statistic.

A fresh GitHub API check on 15 October found author HEAD [c2481248eb0de5fda659d68f59c51e0e27becd20](https://github.com/AI-sandbox/argmix/tree/c2481248eb0de5fda659d68f59c51e0e27becd20), dated 11 August 2026, and empty tags/releases. The bounded public history search found nine commits with an initial 5 August release, no parents and no April archive. This is absence of a publicly found archive, not proof that private or unindexed archives do not exist. See `docs/argmix_paper_revision_search_v1.md`.

The evidence directory is `research/reviews/ARGMix_published_benchmark_audit_20261015_v1/`. It includes literal numbered author-code excerpts, local source excerpts, API/HTTP receipts, exact table extraction, completion metadata, cohort mapping, a machine-readable comparison, a prospective OriginalRecipe plan and SHA256 inventory.

## Exact published accuracies

The paper reports population accuracy percentages. Differences below are arithmetic percentage points. The inspected main text, table captions and supplement do **not define the meaning of ±** as SD, SE, CI or a particular ddof.

| Table / evaluation | Cohort | AncestralPaths (%) | ARGMix (%) | Δ (pp) |
|---|---|---:|---:|---:|
| 1 / three IID test replicates | Present Day | 86.61±0.59 | 94.84±0.20 | +8.23 |
| 1 / three IID test replicates | Post-Neolithic | 86.72±0.53 | 95.18±0.32 | +8.46 |
| 1 / three IID test replicates | Neolithic Farmer | 97.29±0.65 | 98.31±0.07 | +1.02 |
| 1 / three IID test replicates | Yamnaya | 92.23±2.15 | 95.85±0.56 | +3.62 |
| 2 / five misspecified test replicates | Present Day | 88.52±1.18 | 93.85±0.31 | +5.33 |
| 2 / five misspecified test replicates | Post-Neolithic | 88.52±0.99 | 94.31±0.26 | +5.79 |
| 2 / five misspecified test replicates | Neolithic Farmer | 97.01±0.86 | 98.55±0.15 | +1.54 |
| 2 / five misspecified test replicates | Yamnaya | 87.95±1.89 | 94.41±0.75 | +6.46 |

These are published results, not results recomputed in this audit.

## Published recipe: what is explicit

§2.2 states: “We trained both models on a simulated demography from Irving-Pease et al. 2024 containing 500,000 examples (subgraphs), generated from 5 replicates”.

§4.2 says “We modified the msprime simulations to exclude Bronze Age Anatolians”, added genotype error 0.001 to all samples including present-day samples, used the chromosome 22 recombination map, and ran Relate with effective population size 30,000 and mutation rate 1.25×10−8. Misspecification perturbs population sizes by 1–50%, makes admixture timings older by 1–25%, and changes divergence times by ±1–25%. Exact perturbation distributions, joint draws and raw seeds are unspecified.

§4.3 says “We reimplemented AncestralPaths in PyTorch” and “To mitigate the redundancy from local correlation between trees we skipped every ten trees.” Validation uses a separate replicate, 100,000 examples and every 100 trees. ARGMix uses 25 nearest references by TMRCA and three inferred ancestors, giving 29 nodes including the query. Both models use the best validation-loss checkpoint. Testing uses three additional replicates of 100,000 samples each, and five misspecified replicates of 100,000 samples without skipped trees.

The paper does not expose a complete executable recipe: exact sequence length/build, raw seeds, optimizer/hyperparameters/epoch count, individual age/ID table, target coordinate within sampled trees, cohort quotas, reference eligibility/focal exclusion, or ± statistic are unresolved in the inspected sources.

§4.5 reports 1,611 diploids / 3,222 haplotypes: Ana22, CHG13, EHG42, WHG48, Neolithic Farmer159 (including Iceman), Yam13, Post-Neolithic654 and Present660. The four pure-founder groups give 125 diploids / 250 haplotypes. Neolithic Farmers and Yamnaya also serve as informative references. Including all those groups could yield 594 reference haplotypes before eligibility and focal exclusion, but that is a derived upper pool, not a verified eligible count. The four query cohorts give 2,972 haplotypes. H038 instead has 686 queries and 266 pure donors among 952 observed haplotypes.

## Current public author code

The current README describes forward-in-time simulations. The forward template uses two replicates (train0, val1), 15 references, no ancestor features, hidden128, four layers/four heads, FFN factor4, output16, dropout0.25 and attention dropout0.1, log-distance cap14, and a bidirectional GRU128. Chunks are 1024 with stride512, batch16, max20epochs, LR2e−4, AdamW decay1e−4, label smoothing0.1 and patience2. Template sample seed is0, source fit calls seed_everything(50), training drops one nearest reference while validation drops none. Edge trees are excluded during native training/validation.

The loader explicitly rejects `model.num_ancs`: “ancestor features were removed”. Native validation averages batch means and selects strict lower validation loss. The truth-label default is first SNP, with optional majority/purity behavior. Final inference averages softmax probabilities over overlapping chunks. Current final metrics include SNP-level hard accuracy, cohort accuracy, truth-denominator class accuracy, diploid soft-dose Pearson correlation squared, coverage and jackknife intervals. Those do not equal H038 physical-bp scoring.

The forward simulator creates founder mosaics with Poisson tract breakpoints and absolute event generation times. Its required seed is incremented by replicate. This does not establish the original msprime/coalescent paper recipe. The paper Eq3 relative-key indexing and the current implementation also require author reconciliation before an OriginalRecipe claim.

## All observed differences relevant to comparability

| Dimension | Paper vs H038/current baseline |
|---|---|
| Subjects and query distribution | Paper1611diploids/3222haps versus H038476/952, Present1320queryhaps versus182; query2972 versus686. Exact matching ages/IDs unproven. |
| Cohort/path ontology | Paper excludes BAA. H038 includes BAA14, six native paths collapsed as p1+p5,p2+p6,p3,p4 to Ana/CHG/WHG/EHG. |
| Demography | H038 released AP roster and events166/169/177/259, census350, splits600/800/1500. Exact correspondence to the paper's modified demography is unproven. |
| Sequence and recombination | Paper empirical chr22 map, unspecified length/build. H038 synthetic2Mb, uniformr1e−8. |
| Mutation and observation error | Both nominalmu1.25e−8. H038 JC69 discrete Hudson simulation writes mutations directly to VCF with no genotype-error injection; paper error0.001. |
| Inferred genealogy | H038 actually reads Relate `post_size.observed.trees`, not tsinfer trees. H038Relate1.2.4, N30000, five size iterations and final branch reestimation. Paper Relate version and size iteration settings unresolved. |
| Reference pool | Paper pure founder groups plus informative NF/Yam. H038 ARGMix uses only pure266 donors. |
| ARGMix model | Paper25refs+3ancestors; public/H03815refs,noancestors,log14,GRU. Current release is a separate condition. |
| AncestralPaths model | Paper PyTorch reimplementation, bestVALloss. H038 native Keras Flatten→Dense512ReLU→Dropout0.2→Dense6softmax, nativeAdam, five epochs, finalepoch. |
| Training and validation units | Paper500k sampled subgraphs/5TRAIN and100k/1VAL, skip10/100. H03832TRAIN/8VAL wholeforests, up to960complete-parent updates. NativeAP realizes5,190,874rows. |
| Candidate selection | PaperbestVALloss. H038ourmodel maximum equal-parent Present182 accuracy, then NLL/earlierpass. NativeARGMix selects unweighted smoothed nativeVALbatchloss. |
| Labels and inference products | CurrentARGMix native firstSNPtrainlabels and overlappingsoftmaxmean. Paper exact label point/quotas unspecified. H038100bp originalcenters and exactintervalphysicalmass, coverageclip, notail extrapolation. |
| Test and uncertainty | Paper3IID/5OOD100k each. H038observedresults are eight internalVAL simulations and one modelseed. Reserved outer products are separate. Paper±undefined; H038sampleSDddof1 across simulations is explicitly a different statistic. |
| Metrics and aggregation | Paper sampled-subgraph populationaccuracy. H038physicalquery-bp accuracy,NLL,Brier,ECE,boundary/confusion, equal-parent aggregation; richerSNP/mutation/wholetreeinputs. |
| OOD robustness | No completed H038 replication of the paper's five misspecified table cells is established. |
| Versions | Aprilpaper versus Augustcurrentrelease; no retrievedpaper-era archive/config/checkpoint establishes equivalence. |

The numbered source evidence establishes the important genealogy correction: `prepare_expanded_pyg_observations_v4.py:259–268` fixes the native tree to Relate post-size output, and `mutation_routed_full_panel_observations_v1.py:125–161` verifies and reads it. Reader names containing tsinfer are not evidence of tsinfer topology.

## Cohort correspondence

| Paper cohort | H038 cohort / haplotypes | Status |
|---|---|---|
| Present Day | present_day182 | Same conceptual category, different subjects/counts. |
| Post-Neolithic | Bronze_Age162 | Approximate only. Equality of cohort definitions is unproven. |
| Neolithic Farmer | Neo302 | Analogous, different panel: paper159diploids versus151. |
| Yamnaya | Yam26 | Both nominal13diploids; age/ID identity unproven. |
| No table row | BAA14, ancient504, all686 | Extra H038 cohorts/aggregates. |
| PFP | unresolved | No literal definition found in inspected primary main text or supplement. |

Do not silently rename Bronze_Age to Post-Neolithic or guess PFP.

## What actually completed

This audit read completion JSON, cache metadata and generation/inference JSON only, without experimental-array reads. All40development generation receipts exist. H038aligned and zero-concordance completion receipts record32TRAIN/8VAL, one seed101,30passes/960updates, and zero outer accesses. The aligned selected pass is30; the zero-concordance selected checkpoint ispass25. Both receipts retain exact final diagnostic refs. NativeAP completion records32TRAIN/8VAL, five fixed epochs, finalepoch5, noVALcheckpointselection, and all8sealedVALprediction manifests. ARGMix preparation completed and its fit completion records32TRAIN/8VAL, native source seed50 and all8VAL seals. These support current-release development baselines, not OriginalRecipe/real-PCA completion.

Use `completion_evidence.json` and the actual pinned receipts instead of old planned-stage flags in baseline contracts. This audit does not execute or restate new diagnostic results.

## Real-data PCA and minimal OriginalRecipe plan

The real-data analysis in §2.3/Fig2B masks non-Anatolian **genotype haplotypes to missing**, then applies ancestry-specific PCA. A PCA of four ancestry-fraction vectors would be a different analysis. The paper reports 3,720,746 final variants, 1,611 individuals, Beagle5.4, ancient QC/imputation and INFO>0.8, Ötzi GLIMPSE and HGDP+1KGP references, plus snputils ancestry-specific analysis without exact version/settings. Accuracy on simulations cannot establish this real-data result.

`OriginalRecipe_plan.json` is a prospective plan with unresolved gates. First obtain paper-era source/config/checkpoints or author-certified25ref/3ancestor implementation, the exact roster/ages/reference eligibility, map/build/length/error/Relate settings, generation/sampler/training seeds and target/sampling rule, and clarify the uncertainty statistic. Then freeze the fiveTRAIN/oneVAL/threeIID/fiveOOD products, native selection and reporting before execution. Keep OriginalRecipe, currentAuthorRelease and H038CommonC4 as separate conditions. With new seeds or an independently designed split, call the study a prospective recipe comparison, not exact numerical table reproduction. If reporting our chosen sampleSD, state ddof1 and the replicate count explicitly; do not infer optimizer-seed stability from one fit.

No GPU, fit, prediction, resource admission, installation, commit, frozen-source edit or experimental-array read was performed for this audit.
