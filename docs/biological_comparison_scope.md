# Biological endpoints and comparison scope

Our European development endpoint is Ana/CHG/WHG/EHG source ancestry. For the archived six-output AncestralPaths workflow, the derived probabilities are `(p1+p5, p2+p6, p3, p4)` before argmax. This merges historical routes. It is a separately named endpoint, and its physical-span accuracy is not interchangeable with the paper's published path-classification results.

| Endpoint | Labels and cohort condition | Relation to current development score |
|---|---|---|
| Native six-path ancestry | Four routes through present European/Neo/Yam history plus two BAA routes | Different labels |
| BAA-excluded, retrained four-path model | BAA removed from covariates and paths 5/6 removed | Different training inputs and fit |
| Six-path model, four non-BAA paths reported | Original six-path classifier retained | Different aggregation |
| Project-derived four-source span endpoint | Probability collapse above, including BAA queries, common physical support | Current European development comparison |

The observed panel has 952 haplotypes. Its 686 queries comprise Present182, BronzeAge162, BAA14, Neo302 and Yam26. The other 266 haplotypes are pure-source references. Native AncestralPaths also uses the observed cohort identities of admixed ancient samples as covariates. These identities are distinct from unknown local ancestry labels.

The present European experiment uses 32 TRAIN and eight reused VAL parents of 2 Mb. Supplementary Note 3 describes three 200 Mb training sequences and five further testing sequences for its principal experiment. The archived Dense workflow, training budgets, SNP inputs, validation selection, scoring coordinates and source-probability collapse must remain disclosed. Current rankings establish descriptive pipeline development performance, without architecture-only attribution, an untouched test or exact published-table reproduction.

The American EUR/EAS/AFR benchmark is a separate task with 600 reference and 1,000 query haplotypes. European source labels cannot be renamed into those classes. Native ARGformer/FLARE predictions and our GNN must share frozen query identities, focal coordinates and truth support. The original prediction barrier remains in force.

The Nature supplement already identifies flanking-SNP/topology inertia as a source of short-tract errors. Report boundary and tract metrics alongside span accuracy. Empirical paintings and genotype concordance provide descriptive biological evidence without known local ancestry truth.

[Primary-source page excerpts and full audit](../research/reviews/AncestralPaths_biological_protocol_audit_20261016_v1/REPORT.md). The source audit is documentary evidence, not a new model evaluation.
