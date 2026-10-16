# AncestralPaths biological protocol audit

2026-10-16. This source-only audit closes the previously stated Supplementary Note3 reading gap. It does not run a method, evaluate ancestry, or propose another architecture.

The paper defines ancestry as a route through a demographic history, including ancient admixed queries. Its six-path task, its BAA-excluded retrained four-path task, and the project's derived four-source span endpoint are different experiments. Current European development scores cannot be compared numerically with the paper's94.6% headline as if the labels, data and metric matched.

## Source and page convention

Primary source: Irving-Pease et al., The selection landscape and genetic legacy of ancient Eurasians, Nature625,312–320 (2024), DOI https://doi.org/10.1038/s41586-023-06705-1 , Supplementary Information, Note3.

Existing source, read in place:
.runtime/author_sources/ancestralpaths_supplement_41586_2023_6705_MOESM1_ESM.pdf
SHA256:0558218aacc6438427e2d41226b88a58e5dadc82d02b97fe019ad3aa84237a3c.
It is38,478,745 bytes. No PDF download or copy was made. The known author-source text cache was used with fresh selected-page extraction checks. Only small excerpts/report/receipts are written to this dedicated directory. The supplied Springer CDN hostname failed DNS in a read-only HEAD attempt; the cached primary source removes that acquisition dependency.

Citations below use PRINTED supplement pages. For these pages the PDF viewer page is printed page+1. Full pages27,32,34,43 and44 were rendered to memory and visually inspected, including table/figure axes and footers. Relevant text was read on printed pp24–35 and43–60. This is a review of the relevant methods, not every page of the supplement. No cohort spreadsheet or genomic payload was accessed.

## Exact biological facts

| Item | What the primary supplement states | Page |
|---|---|---|
| Label meaning | A haplotype is labelled by the path it takes backwards through successive populations, rather than one timeless population identity. Paths may overlap and coalesce. |24,27 |
| Paths1–4 | 1: present European -> Neo -> Ana -> WA; 2: present European -> Yam -> CHG -> WA; 3: present European -> Neo -> WHG -> NE; 4: present European -> Yam -> EHG -> NE. Each continues to the model root. |27, Fig.S7 |
| Paths5–6 | Both begin in Bronze Age Anatolians (BAA);5 joins the Ana route of path1 and6 joins the CHG route of path2. These preserve different historical routes even where the upstream source is shared. |27 |
| Diagnostic analysis panel |476 diploid individuals/952 haplotypes, including91 GBR1000Genomes individuals. Simulation samples match the real individuals' populations and radiocarbon/context dates. |27–28 |
| Larger selection panel |A separate expanded panel has1,015 ancient genomes relevant to European history plus503 present-day European1000Genomes samples, total1,518 diploids. Do not substitute these counts for the476-person analysis panel. |28 |
| Inferred inputs and labels |The predictor extracts GNNs from Relate-inferred trees. True path labels are obtained separately from the corresponding simulated trees at the SAME genomic sites. Internal population identities are unknown in Relate. |30–32 |
| Feature definition |Traverse from the focal haplotype towards the root. Record proportions of sampled ancient groups below successive informative ancestors, excluding leaves already counted at earlier ancestors. Use the first five informative nodes, ancient-group proportions plus node age, and -15 padding when the root is reached early. |30–32 |
| Ancient queries |The classifier also trains on admixed ancient samples. Pure diagnostic CHG/WHG/EHG/Ana samples can be assigned from their population identity alone. “Ancient sample” therefore does not automatically mean “pure reference.” |32 |
| European simulation |200Mbp, constant recombination1e-8 per bp per generation and mutation1.25e-8. The paper notes that a real recombination map can be used as a separate option. |28 |
| Main train/test description |Three200Mbp TRAIN sequences with different seeds.10,000 training pairs from each of five admixed cohorts,50,000 total, across evenly spaced sites. Five MORE simulated/inferred sequence pairs are used for testing. Relate mutation1.25e-8 and initial haploid population estimate30,000 are stated. |32 |
| Main reported result |93.12% mean accuracy, SD0.29% across five testing tree sequences. The text says testing on “10,000 GNNs from each”; preserve this wording rather than inferring a more specific denominator from the prose alone. |32 |
| Confusion matrices |Pooled testing examples are shown with rows normalized to display precision. The inspected Fig.S16 explicitly labels vertical axis Predicted labels and horizontal axis True labels. Precision is not recall or overall accuracy. |33,44 |
| Biological error mechanism |Present-day GBR has more generations since admixture and shorter tracts. The text attributes boundary difficulty partly to Relate's flanking-SNP use and inertia in changing tree topology; small correlated false tracts remain. |33 |

The source calls the classifier a convolutional neural network on p31. The archived training.py actually defines Flatten(5,9), Dense512/ReLU, Dropout0.2, Dense6/softmax and five epochs (lines173–177). Reading this method section does not recover an exact published CNN specification. The current baseline can properly be called the archived/released Dense workflow with disclosed adaptations; it cannot silently become a numerical reproduction of the paper's stated CNN.

### The94.6% headline is not today's C4 span score

Printed pp43–44 describe two conditions:

- Remove BAA from GNN distributions AND remove path labels5/6, retrain a smaller-feature four-path model: reported94.3% +/-0.27%, over five testing tree sequences in GBR.
- Keep BAA and a six-path classifier, then average over the four non-BAA paths: reported94.6% +/-0.46%. Fig.S16 still allows six predicted path labels and shows predicted-row precision.

The source calls these quantities accuracy, while the adjoining figure is a class-precision display. This audit preserves the author's numerical labels and does not invent an exact aggregation formula where the primary prose does not supply one. In particular, neither condition is established to equal physical-span accuracy after summing path probabilities into source classes.

The project's collapse (p1+p5,p2+p6,p3,p4) merges routes that the original six-path task distinguishes. It must occur before argmax for the defined derived C4 endpoint. Collapsing a six-way hard argmax instead is a different rule. A correct source prediction can still be a wrong historical path. Removing BAA features/retraining is also different from retaining BAA and collapsing outputs.

### The paper's actual same-data GNOMix comparison

Printed p34, Table15, and p35:

- References are the four single-path source populations EHG, WHG, CHG and Ana.
- Query cohorts are GBR, Bronze Age, Neo and Yam. Only GBR is present-day. BAA is excluded in this comparison, although the paper says it could be tested with Ana/CHG references.
- Both methods are scored at evenly spaced sites on the same five200Mbp sequences.
- GNOMix uses the default logistic-regression BASE and xgboost SMOOTHER together, not two alternative variants. The window is0.02cM for GBR and0.2cM for Bronze/Neo/Yam.
- Mean/SD are taken across the five sequences. The paper uses two-sided two-sample t-tests. The table displays the following mean accuracies, not a new project result:

| Query cohort | GNOMix % | AncestralPaths % |
|---|---:|---:|
| GBR |74.7 |91.8 |
| Bronze Age |87.1 |93.5 |
| Neo |94.4 |95.5 |
| Yam |98.6 |95.2 |

GNOMix wins Yam in this published comparison. “AncestralPaths is better for every cohort” would be false. Exact original seeds, site spacing/identities, trained weights and complete executable defaults remain unresolved in the existing project contract. A newly specified paired comparison is valid as a new study, but those choices cannot be presented as recovered original bytes/settings.

### Robustness evidence is bounded, not universal

The supplement separately tests four-path demographics with different path numbers/divergence, population sizes, sample imbalance, admixture timing/fractions, drifted reference branches, extra samples and ghost lineages (pp45–58). The generic scenarios typically use20Mbp and five training/five test sequences, rather than the principal3x200Mbp European training recipe (p45).

Not every misspecification is harmless: altered population size can materially reduce accuracy (p53), and ghost-lineage precision depends on whether other populations along that path are observed (pp57–58). Future biological claims should name the tested condition. Current development success on one fixed European demography does not establish robustness to these errors, absent matched tests.

## What the current development comparisons establish

Reviewed sources: README.md, research/current_scoreboard.md, docs/ancestralpaths_native_protocol_v1.md, docs/ancestralpaths_full_panel_discovery_v1.md, docs/paper_dataset_evaluation_contract_v3.md, the frozen expanded-panel/H038/native-baseline configs, and the documented C3 proposal. Their reviewed scopes/hashes are recorded in material_reviewed.v1.json. No predictions, truth arrays or model checkpoints were opened.

The expanded European condition uses32 TRAIN/eight VAL simulations,2Mb, the476-diploid author roster,952 observed haplotypes,686 query haplotypes and266 pure-source reference haplotypes. Those role counts are a project/frozen-roster fact, not a new count inferred from the paper's prose:

| Cohort | Diploids | Current query/reference role |
|---|---:|---|
| Present/GBR |91 |182 queries |
| Bronze Age |81 |162 queries |
| BAA |7 |14 queries |
| Neo |151 |302 queries |
| Yam |13 |26 queries |
| WHG/EHG/Ana/CHG |51/43/25/14 |266 pure reference haplotypes total |

Native AP's GNN covariates also use observed ADMIXED ancient-cohort identities. These are not local-ancestry truth, but they are legitimate observed side information that must be described. A fair comparison must document each method's observed metadata and native feature use rather than call all952 leaves pure reference labels or strip these features and still claim the unchanged native method.

The common endpoint is explicitly derived Ana/CHG/WHG/EHG C4, with BAA included among queries, equal-parent physical-span metrics on common100bp-centered support clipped to inferred coverage. The native AP six-path SNP/focal metric remains a separate endpoint. The32/8 parent split,2Mb length, larger source-workflow fit, present-day-primary reporting, support and probability collapse differ from the paper's protocol.

As displayed in the reviewed scoreboard, native AP derived C4 has93.7830 +/-0.8385% Present182 span accuracy; H045 foreign_episode has96.8270 +/-0.5870%. These are existing development scores with SD across eight VAL simulations and one optimizer seed, not fresh measurements by this audit. H045's matched own-state control has96.8175 +/-0.6142%, and its0.009568 percentage-point paired gain fails the declared0.10pp practical threshold. Native AP retains a five-epoch Dense fit; the GNN has a different cumulative training lineage and validation checkpoint selection. Our GNN also consumes observed SNPs. Therefore a descriptive pipeline ranking is possible, but neither architecture-only attribution nor published-paper reproduction follows.

The current American C3 condition is a different problem: EUR/EAS/AFR ancestry, an American-admixture model and1000 query/600 reference haplotypes under a disclosed reconstruction convention. European Ana/CHG/WHG/EHG outputs cannot be merged or renamed into those classes. The reviewed scoreboard still marks its held-out ancestry scores pending. European development scores do not establish performance on the original ARGformer/FLARE task. This audit does not re-check live job status.

## Actionable biological corrections for root integration

1. Keep three AP endpoints labelled separately in prose/tables: native six-path; BAA-removed/retrained four-path; project-derived C4 probability collapse. Cite pp27 and43–44 for the distinction. Do not use the94.6 headline as a matched baseline for current span accuracy.
2. Keep present-day and ancient query strata separate, with observed age/cohort metadata and true local path labels distinguished. Report all five admixed European cohorts, including BAA in the current derived-C4 condition, plus the appropriate BAA-excluded endpoint if reproducing the published GNOMix comparison.
3. For a new fair comparison, derive every native representation from the SAME observed genotypes, observed dates/cohorts, reference/query identity list and inference condition where applicable; seal all predictions before opening held-out ancestry truth. Native VCF, path/GNN and PyG formats can differ. Use the SAME haplotypes, coordinates/coverage and truth for each paired endpoint, with missing mass/attrition explicit.
4. Preserve sample/site accuracy for a paper-recipe assessment and report physical-span/tract/boundary metrics as separately named common endpoints. High span accuracy cannot establish accurate short-tract or switch inference; p33 already identifies this failure mode. Use true-short retention and false-short burden alongside boundary precision/recall on prespecified support.
5. Keep the archived Dense model, source sampler quotas and runtime/selection adaptations disclosed. Source scripts use TRAIN parents3,5,6 and TEST5–9, unlike the supplement's five MORE test sequences; current independent-parent fixes are appropriate new-study corrections, not unchanged reproduction. Exact original CNN details, raw streams, site sampling and trained weights remain missing.
6. Complete independent untouched-parent confirmation and training-seed uncertainty before a superiority claim. Eight repeatedly reused development VAL simulations are not an untouched test; optimizer repeats do not create new biological replicates. The reviewed current table contains no native GNOMix reproduction of p34, and the American C3 benchmark is separately unfinished.
7. Empirical paintings have no known true local ancestry labels. The current small empirical demonstration is descriptive; it is not the paper's476-person/expanded1518-person analyses or an empirical accuracy result. PCA/Fst agreement of simulations with real data (pp28–30) supports plausibility, not per-tract truth.

Most of these distinctions already appear in the repository's frozen native-protocol and v3 comparison contract. The correction is to carry them consistently into claims and future evaluations, not to rewrite frozen sources or retrospectively change the development estimand.

## Verification and limits

The cached PDF hash matches the existing source pin. Relevant page text was cross-checked against fresh pdftotext output. Full-page visual checks verified page numbers, Table15 and Fig.S16 axes. The retained relevant-page extract is supplementary_note3_relevant_pages.txt. No large source was copied; all new artifacts together are well below10MiB.

This audit supports documentary biological/protocol fidelity only. It does not validate any source implementation at runtime, actual causal payload joins, model score, completed native reproduction, or statistical superiority. Root owns edits to the manuscript/README/protocol integration and any future experiments.
