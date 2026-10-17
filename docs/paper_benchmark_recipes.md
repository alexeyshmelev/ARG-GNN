# Paper-based paired benchmarks

Execution remains on hold. This preparation adds source-backed recipes and shared-data interfaces. It does not report new ancestry scores or claim that every published method is already reproduced.

For each independent simulation, generate the genotypes and known ancestry once. Keep ancestry truth separate. Infer the genealogy once where graph methods require it. Every method receives its required representation of those same observations and reference/query panel. Freeze the same evaluation queries and coordinates before predictions, then report each method against the same known ancestry. Different internal data formats are expected.

## Prepared comparisons

| Recipe | Task and paper settings encoded | What still needs resolution |
|---|---|---|
| [ARGformer American](../configs/benchmarks/recipes/argformer_american.json) | AmericanAdmixture_4B18, 30 Mb, stated population counts, mutation/recombination rates, error-free phased data, tsinfer/tsdate. TreeLoom, native ARGformer and FLARE. | Whether stated samples mean haploids or diploids, material inference and evaluation rules, and paper/release training conflicts. A caller choice is not proof of the paper's convention. |
| [ARGMix European](../configs/benchmarks/recipes/argmix_european.json) | 1,611 people, BAA excluded, chr22 map, genotype error 0.001, 5 TRAIN/1 VAL/3 IID TEST, published sampling budgets and tree skips, four founder classes and cohort metrics. | Exact age/eligible-donor roster, map domain/build, error mechanism, sampling rule and missing training settings. Released ARGMix differs from the preprint. Its AP comparator also differs from the released Keras model. Five OOD simulations need a separately resolved perturbation recipe. |
| [AncestralPaths European](../configs/benchmarks/recipes/ancestralpaths_european.json) | Complete 476-person age roster, 3 TRAIN and 5 TEST sequences of 200 Mb, paper demographic parameters, native six-path endpoint, Relate settings from the author script. TreeLoom and released AncestralPaths. | Paper CNN description versus released Dense network, and exact sampled-example selection details. These affect training/comparison, not the configurable generator's ability to use the paper demographic recipe. |

The AP generator explicitly selects the paper's BAA admixture at 170 generations. Released code uses max(sample age)+1, which is 169 for this roster. The seven BAA individuals have ages 136 or 168, so the paper's 170 is consistent with every sampled age. This parameter decision is recorded, rather than silently inheriting the released value. The native six-path labels retain the two BAA routes. Their collapse to four sources is a separate endpoint.

The American 20 TRAIN/5 VAL/10 TEST counts are our prospective design for supervised TreeLoom fitting and evaluation. They are not attributed to the ARGformer paper. Native ARGformer still fits its vocabulary and representation per observed ARG, including the evaluation ARG, without its query ancestry truth. Record that transductive computation separately from TreeLoom's training across independent simulations.

## Code boundaries

- `paired.py` creates independent parent/seed plans, joins actual family preparation helpers and binds shared observations. Product identities change with the random streams. Renaming a genealogy cannot make it an independent TRAIN or TEST parent.
- `european_recipes.py` contains the configurable AP producer. It reuses the existing roster, truth separation and guarded output helpers without editing their pinned source. Length, map endpoints and auxiliary coordinates come from the configuration. Strict mode checks the registered parameters and evidence before execution.
- `targets.py` supplies sampled-example cross-entropy with explicit parent, split, sample registry and class order. Repeated draws retain their weight. It does not silently substitute the earlier physical-span objective. Native class quotas may use TRAIN truth. Evaluation sampling stays label-blind.
- `treeloom.py` exposes the frozen model and the existing native-six output head. Task bindings change input/output dimensions, not the selected architecture.
- Original author implementations and historical model variants remain in place. [Model history](model_history.md) records which ideas worked, failed or remain untested.

Inspect with `paper-recipes` and `paper-plan RECIPE_ID` through `python -m arggnn.benchmarking`. Planning verifies the cited local source pins, creates no data and starts no jobs. The plans expose remaining differences instead of substituting the previous reconstruction settings.

The code is not an approved end-to-end launch package yet. Per-family runtime integration, complete sampling rules and material comparator choices must be resolved before a paper-aligned campaign can run. Source and CPU smoke checks establish code behavior only. They do not establish ancestry accuracy, full-size GPU fit or reproduction of printed table values.

## Source attribution

The pinned [ARGformer article XML](../research/reviews/ARGformer_published_benchmark_audit_20261015_v1/paper.fulltext.xml)
is the authors' open-access article, © The Author(s) 2026, published by Oxford University Press,
[doi:10.1093/bioinformatics/btag438](https://doi.org/10.1093/bioinformatics/btag438).
It is redistributed unchanged with its original attribution and embedded
[Creative Commons Attribution 4.0 licence](https://creativecommons.org/licenses/by/4.0/).
Author runtime checkouts are separately acquired from the pinned repository revisions.
