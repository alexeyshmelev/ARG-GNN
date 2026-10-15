# C4 biological interpretation

**The main Present182 comparison is 96.8270 ± 0.5870% for H045 versus 94.4461 ± 1.1491% for current native ARGMix: +2.3809 percentage points.** These are equal-parent development means, with sample SD across eight simulations and one fit per method. They compare pipelines with different inputs, training and selection. H045's matched own-state arm is 96.8175 ± 0.6142%, so its specific foreign-episode increment is only +0.009568 pp. The high ancient scores do not come from scoring the clamped pure references: all 504 ancient scored haplotypes are queries with unknown input ancestry.

## What C4 and the seven rows mean

C4 is the simulated population of a query's ancestor at **350 generations before present**: 0 ANA/Ana (Anatolian-farmer source), 1 CHG (Caucasus hunter-gatherer source), 2 WHG (Western hunter-gatherer source), 3 EHG (Eastern hunter-gatherer source). It is neither the query's sampling population nor an empirically established person's exact historical population. In particular, the admixed Neo cohort and the pure Ana reference source are different populations.

The external native six-path encoding is Ana_nonBAA, CHG_nonBAA, WHG, EHG, Ana_viaBAA, CHG_viaBAA. Code adds four to paths 1/2 for BAA queries and collapses with [0,1,2,3,0,1] for C4. Native AP is trained on six labels; our model and current ARGMix predict C4. BAA is retained here as the released code's cohort/branch name; no unsupported expansion of that acronym is needed.

There are **five disjoint query groups**, plus Ancient504 and All686 overlapping aggregates—not seven independent biological cohorts. Every one of the 32 TRAIN and eight exposed VAL parents uses this same roster and demographic recipe, with independent simulation streams.

| Query group | Haplotypes | Sampling population; age range in generations BP | Population-level expected C4 proportions, ANA/CHG/WHG/EHG | Generations from sampling back to first mixture pulse | Actual true C4 changes per haplotype across the scored span |
|---|---:|---|---|---:|---:|
| Present182 | 182 | present_bronze; 0 | .375/.25/.125/.25 | 166 | 2.753 |
| BronzeAge162 | 162 | present_bronze; 100–165 | .375/.25/.125/.25 | 1–66 | .773 |
| BAA14 | 14 | BAA; 136 or 168 | .75/.25/0/0 | 1 or 33 | .205 |
| Neo302 | 302 | Neo; 139–258 | .75/0/.25/0 | 1–120 | .528 |
| Yam26 | 26 | Yam; 153–176 | 0/.5/0/.5 | 1–24 | .067 |

The proportions are analytic expectations from the demographic graph, not assigned individual targets or observed mixture fractions. Present/Bronze first split .5 Neo/.5 Yam at 166; Neo splits .25 WHG/.75 Ana at 259; Yam splits .5 CHG/.5 EHG at 177; BAA splits .25 CHG/.75 Ana at 169. Source merges occur later backward in time, at 600 and 800, beyond census350.

Crucially, original author simulate.py:51–54 sets each mixture time to **max(the corresponding cohort's sample ages)+1**. The frozen recipe yields the four times above. Thus the oldest query in each ancient group is only one generation after its first mixture. Calendar age, sampling population, possible source classes and tract complexity are coupled by construction. Present-day samples have had many more generations for ancestry-changing recombination; their two-stage history also supplies all four sources. Independent parents do not break that coupling.

Ancient504 = Bronze162+BAA14+Neo302+Yam26; Neo alone supplies 302/504=59.92% of it. All686 is 73.47% ancient by haplotype count. The true-change column comes solely from the completed H042 aggregate metric JSON: Present4008, Bronze1002, BAA23, Neo1275 and Yam14 changes over eight parents, divided by each group's 8×haplotypes. These are counts on existing common coverage, not new truth reads or a fitted tract model.

The main completed scores reflect that easier structure:

| Scoring row | H045 accuracy %, mean ± sample SD | Native ARGMix accuracy %, mean ± sample SD |
|---|---:|---:|
| Present182 | 96.8270 ± .5870 | 94.4461 ± 1.1491 |
| BronzeAge162 | 99.2047 ± .3199 | 97.7051 ± .7466 |
| BAA14 | 99.1511 ± 1.2545 | 98.4370 ± 1.0186 |
| Neo302 | 99.6063 ± .0850 | 99.2955 ± .2623 |
| Yam26 | 99.9039 ± .1292 | 99.2218 ± .8966 |
| Ancient504 | 99.4799 ± .1347 | 98.7566 ± .3114 |
| All686 | 98.7761 ± .2486 | 97.6130 ± .4975 |

These are not merely age-only predictions: a cohort's largest expected class fraction is at most .75, far below 99%. Conversely, near-perfect span accuracy is not near-perfect tract recovery. H042 has only 14 true Yam changes across 208 haplotypes and just four matched changes at 1 kb; its BAA score is 99.02% while matching only 3/23 true changes. H045 Present boundary F1 is 18.1558 ± 1.3790%. Sparse transitions and long interiors make span accuracy forgiving.

## Reference proxies and method-specific information

The other 266 haplotypes are source proxies, excluded from all seven query-score rows: ANA50, CHG28, WHG102, EHG86. Their sampling ages are ANA217–360, CHG184–473, WHG210–529, EHG155–402. They are sampled directly from unmixed source branches, with their source class treated as known observed metadata. References older than350 have no query census target; their labels are source-proxy assumptions, not labels obtained by traversing a query to the census.

| Method | Explicit sample/time/population information after inference | Direct genotype input and objective |
|---|---|---|
| Our H038→H042→H045 family | Four source-class one-hots only for pure references; unknown-label flag for all686 queries; explicit log sample age, all-node ages/root/sample flags, branch durations and descendant geometry. No query-cohort one-hot or numeric sample-ID feature. H042 flags/H045 episode context are observation-derived. | Observed SNP allele/reference-frequency/callability/panel-frequency/entropy features and mutation–branch concordance, plus complete trees. Supervised TRAIN query C4 exact-span loss. |
| Current native ARGMix | Query token <SAMPLE>; 15 selected pure-reference source tokens; no ancestor features in the current configuration. Directed time-to-MRCA entries abs(t_i−t_MRCA) are log2-scaled, quantized and clipped at14. Query age affects these distances and reference ranking. No explicit Bronze/BAA/Neo/Yam query token. The genomic decoder uses raw row span in cM; preceding-distance input is disabled. | Tree-derived representation, without the GNN's direct SNP channels. Supervised simulated TRAIN C4 labels at native first-SNP interval labels, label smoothing .1 and native early stopping; current fit ended at11 epochs. Training already drops the nearest reference before selecting15; VAL does not. |
| Released native AncestralPaths Dense | Five successive nonempty ancestor shells; proportions of eight observed ancient groups Bronze/BAA/Yam/Neo/WHG/EHG/Ana/CHG, plus ancestor age/1500, root padding -15. Ancient focal self-inclusion is retained, so an ancient query's observed cohort can enter its own shell summary. No separate query-age scalar in this 5×9 feature array; age is represented by ancestor times and upstream inference. | No direct SNP feature channel after inference. Supervised native six-path TRAIN labels; five epochs on5,190,874 accepted rows, then sum the six probabilities into C4 before argmax. |
| Native ARGformer C4 transfer | Leaf-to-root node-ID/topology tokens, not explicit node dates, cohort labels or genotype matrices. Pure-reference labels enter contrastive learning and retrieval. Dates affect the genealogy upstream. | Observed-only MLM and reference-only contrastive/retrieval objective, including within-ARG unlabelled query paths. It receives no simulated query-ancestry training labels. The measured reduced C4 run is a different panel/genealogy from these full952 scores; full-panel source contracts do not supply a completed same-panel result. |

All three main scored pipelines share the post-five-size-iteration Relate observations. Relate itself receives known sample ages and observed cohort labels through SNP annotations and population-size estimation; clearing population tables afterward does not mean cohort information was absent upstream. Our/ARGMix direct predictor inputs lack a query-cohort one-hot, while native AP explicitly aggregates those labels. The ARGformer tsinfer/tsdate source route instead excludes source labels from genealogy inference. These are substantive pipeline differences.

H045 adds30 passes/960 updates from the completed H042 pass30 checkpoint. It is not trained for the same work or objective as native AP/ARGMix. AP and ARGMix are **also supervised simulation methods**; “our supervised model versus native reference-only learning” applies to ARGformer, not to all native competitors. Its within-ARG transductive training additionally differs from inductive cross-parent learning. No comparison here isolates architecture or reproduces the papers' published tables. The AP primary paper and ARGMix preprint report their own demographic robustness conditions; those outcomes cannot be inherited by this adapted panel.

## Confounding versus leakage

Known age/cohort/source-proxy metadata can legitimately predict ancestry under this task. Their correlation with hidden targets is **demographic confounding/limited transfer**, not evidence that VAL truth leaked into a model. The exact native AP cohort self-feature is an observed-covariate assumption that should be disclosed, rather than called hidden ancestry. Intended contracts give queries unknown source labels, separate TRAIN targets from feature preparation and require prediction seals before VAL scoring. The completed summaries record that barrier.

These source/metric reads establish those definitions and intended boundaries, not a new runtime certification of forbidden artifacts. No new data were opened. The current evidence does not quantify how much accuracy comes from age priors versus alleles/topology, and it does not establish transfer when age/pulse/proxy composition changes. H047 already tested a different issue—equal Present/ancient risk—and failed its benefit criterion; age-coupled data structure is not fixed merely by reweighting that loss.

## One bounded, presently unresolved biological condition

**Withdraw the source labels of the single ANA reference diploid aged360, while retaining its genotypes, sampled age, tree nodes and all query inputs.** Do this only in a newly registered future input contract: current fixed-count validators must not be bypassed. This is a concrete missing-proxy condition, not another generic reference-geometry experiment.

Public age metadata show that the GNN's eight SNP donor channels contain younger/older counts ANA48/2, CHG18/10, WHG90/12, EHG76/10 at the fixed350 split. Thus the entire older ANA frequency channel rests on two haplotypes from one diploid. Current simulations always supply its correct pure label. The resulting biological vulnerability is *unknown*: reported accuracy on that perfect fixed panel cannot establish robustness when an ancient proxy is unclassified or its source annotation is uncertain.

This is not an undiscovered general idea. H002 addressed reachability; H021/H026/H027/H029 used observed proxy evidence, ages, census placement and carrier routes; H032 tested donor-support dependence. The earlier mechanism plan already proposed a reference-age/composition perturbation, and native ARGMix already trains with nearest-reference dropout. Neither makes generic reference robustness a new idea. The specific condition here removes a whole older class channel by withholding one declared diploid's labels, rather than dropping one query-dependent nearest reference while retaining the panel. census_lineage_memory_v1.py changes only boundary_rho for current/genuine/sham census-edge retention and explicitly leaves reference features and labels unchanged. None of those inspected conditions records this specific full952 one-diploid label-withdrawal outcome. No universal absence claim is made beyond the inspected ledger/configuration scope.

A single fixed-checkpoint future comparison against the unchanged panel could measure Present accuracy, NLL, ANA recall and boundary F1, with paired parent effects. Negligible changes would reject this targeted fragility hypothesis; a material decline would motivate explicit reference-availability/reliability conditioning. Existing masks and whole-tree inference might already suffice. A reliability gate or uncertain-reference representation is a conditional engineering direction, not a demonstrated gain or novelty claim. No new fit, perturbation, prediction or scoring was run here.

## Evidence and scope

source_refs_v1.json pins the configuration, author age/simulation sources, exact feature/label builders, prior-reference conditions, primary-paper XML and completed metric summaries. biology_summary_v1.json stores only public-source demographics and aggregates derived from those completed JSON receipts. The primary sources include the AncestralPaths paper [10.1038/s41586-023-06705-1](https://doi.org/10.1038/s41586-023-06705-1), ARGMix preprint [10.64898/2026.04.05.714076](https://doi.org/10.64898/2026.04.05.714076), and ARGformer [10.1093/bioinformatics/btag438](https://doi.org/10.1093/bioinformatics/btag438), read from existing public XML copies.

Only this owned review folder was written. No private truth/tree/genotype/probability/checkpoint payload, actual model, GPU, guard/accounting, Git, installation, test or new scientific execution was used.
