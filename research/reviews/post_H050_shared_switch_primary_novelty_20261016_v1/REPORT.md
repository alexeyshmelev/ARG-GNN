# Shared-switch architecture claim rejected

The proposed shared-switch layer does not currently support an architectural novelty claim. Its central inference machinery is established, and its biological support construction is not justified. Do not implement or register it on the strength of the previous design. Preserve that design as an explicitly superseded, unregistered proposal. No H050 result is inferred here.

## Primary-source overlap

| Primary source read | Exact overlap and distinction |
|---|---|
| [Baran et al.2012, LAMP-LD/HAP](https://doi.org/10.1093/bioinformatics/bts144), Methods on trios | Family constraints in LAI are established. The paper discusses the factorialHMM for joint parental paths but implements a faster trio-phasing step followed by independent haploid ancestry inference. It does not implement our exact factor, and known Mendelian transmission is stronger evidence than an inferred changed clade. |
| [Rasmussen et al.2014, ARGweaver](https://doi.org/10.1371/journal.pgen.1004342), Fig1B and threading model | Genealogical transitions/recombination histories, HMM threading and posterior ARG sampling are established. Marginal tree endpoints are not the complete transition history. |
| [Hubisz, Williams and Siepel2020, ARGweaver-D](https://doi.org/10.1371/journal.pgen.1008895), population-path HMM states | A threading state includes branch,time,population path. Sampled ARGs already induce introgression probabilities while accounting for genealogical uncertainty. This is a different task/scale from European C4LAI, not a head-to-head benchmark result. |
| [Steinrücken et al.2018, diCal-admix](https://doi.org/10.1111/mec.14565), Methods2 | Demography-conditioned coalescent HMM ancestry is established. It analyzes a focal haplotype against reference lineages rather than our shared-query factor. |
| [Bleakley and Vert2011](https://arxiv.org/html/1106.4199v1), Sections3.1-3.2 | Borrowing strength across sequences through shared change points is established. Group fused Lasso is not our categorical factor algebra, but shared boundaries alone are not new. |
| [Kohli, Ladicky and Torr2009](https://doi.org/10.1007/s11263-008-0202-0), Eq17-23 | Robust higher-order consistency, exceptions in imperfect groups and weighted membership are established. Our mixture is not literally their truncated Potts energy and inherits none of their graph-cut guarantees. |
| [Tarlow et al.2012](https://arxiv.org/abs/1210.4899), auxiliary-count construction | Cardinality dependence and efficient auxiliary-state sum-product are established. Their exactness does not apply automatically to overlapping loopy factors. |
| [Kuck et al.2020, BPNN](https://arxiv.org/html/2007.00295v1), Sections2-3 and [Garcia Satorras and Welling, NEBP](https://arxiv.org/html/2003.01998v5), Sections4.1-4.3 | Learned factor-graph operators, log-domain messages, neural/BP feedback and end-to-end unrolled inference are explicitly established. A whole-tree unary encoder and three fixed inference rounds do not invent this layer. |

These are verified primary articles/manuscripts with cached relevant full text and retrieval hashes. Exact locators, versions and overlap limits are in verified_primary_evidence.v1.json. Springer returned a challenge page, so the Potts argument uses the verified institutional author postprint. LAMP and diCal XML endpoints failed, so their accessible PMC full text was read instead. This is a focused audit, not an exhaustive absence-of-prior-art proof. Rejection already follows without assuming the exact domain formulation is absent from the four supplied papers.

## Literal algebraic overlap

Write X_q=(Y_q,left,Y_q,right). The proposed factor is

    F(X_D) = sum_z phi(z) product_(q in D) psi_q(X_q,z).

Introducing z simply makes it a star-shaped latent-variable factor graph with unary phi and pairwise psi factors. Eliminating z is ordinary sum-product. In energy form it is a log-sum-exp over sums of local group costs. If psi depends only on whether X_q matches a candidate transition, equal-member potentials depend only on transition-label counts. Weighted participation gives a weighted group-consistency variant.

Thirteen states, optional query gates, a neutral no-change state and an MLP conditioned on tree embeddings are parameterization choices. They do not establish a new kernel, new message-passing principle, new posterior algorithm or A*architecture contribution. The matched independence/sham controls are sensible controls, not novelty evidence.

## The biological counterexample

Consider one genuine subtree-prune-and-regraft transition:

    left tree:  ((a,b),(c,d))
    operation:  prune only a, then attach its lineage beside c
    right tree: (b,((a,c),d))

The newly appearing proper clades include{a,c}and{a,c,d}. Both contain samples whose ancestral route was not recombined. Thus the full descendant set of an appearing/disappearing clade is not the set of descendants sharing the recombination event. Coalescence times can be older than the ancestry-label horizon, allowing a's source ancestry to change while c's and d's do not. The same structural counterexample exists even with perfectly known marginal trees. Inferred topology and date uncertainty only add ambiguity.

Moreover, recombination can join lineages with the same source ancestry. Under the simple source-redraw pulse model used to motivate LAIHMMs, the probability of a source change at a recombination is1-sum_c(alpha_c^2), not1. This is algebra under that model, not an estimate for our multi-pulse simulations. An inferred topology boundary can also be an inference artifact. Conversely a source switch can be weakly represented or hidden in a point estimate of local trees. Graph edges are not observed ancestry boundaries.

Optional participation can soften these errors, but then the candidate is a generic learned robust group regularizer. It does not repair the biological claim that changed-clade support identifies a shared ancestry-switch event. Products over correlated descendants can also create unjustified confidence unless dependence and repeated support are modeled or calibrated. No accuracy/calibration benefit has been shown.

## Would genuinely SPR-compatible support marginalization rescue it?

No current architectural claim is rescued. Let S(T,T')be the set of compatible recombination/recoalescence explanations of two trees. Averaging factors over s in S is another latent mixture. ARGweaver/ARGweaver-D already provide the central machinery of probabilistic recombination and population-path histories.

Compatibility is weaker than identification. There may be many allowed prune times,recoalescence branches and source-population paths. A pair of inferred trees may need multiple events or may violate the chosen single-SPR approximation. Uniform weighting over compatible explanations is an arbitrary prior, not a posterior inferred from sequence data. A scientifically qualified weight needs an explicit population/recombination/mutation model and uncertainty in the inferred trees. Relate's stored internal IDs or an unqualified branch association do not supply that posterior automatically.

One narrow biological question could remain: given an independently qualified event posterior, does using its uncertainty-aware affected-lineage support improve ancestry boundary inference beyond an equally informed independent-query model? That would be a conditional utility test of an established graphical modeling bias. It is not a new transport/operator claim. The present project has no such validated support posterior in this proposal. Reject this hypothesis as a next architecture experiment until that prerequisite exists and a distinct contribution is demonstrated. Do not manufacture posterior weights merely to keep the idea alive.

## Exact edge-diff efficiency is a separate engineering question

[Ralph, Thornton and Kelleher2020](https://doi.org/10.1534/genetics.120.303253), AlgorithmB, already updates cached subtree statistics along edge-removal/insertion ancestor paths. [Naman and Simmhan2025, Ripple](https://arxiv.org/html/2505.12112v1), Sections4.1-4.3, already incrementally propagates exact GNN embedding changes under edge/feature updates using cached linear aggregates and fixed trained weights.

Our whole-tree up/outside sweeps can have global dependencies. A single changed child can alter an ancestor state, whose outside messages then alter many other descendants. In the worst case Omega(N)materialized outputs change. Tree-sequence edge diffs therefore do not imply O(number of changed edges)exact neural work. Genomic recurrence and optimizer updates add dependencies and cache invalidation. These observations do not prove caching is useless, only that a sublinear generic claim is unsupported.

A useful implementation question is whether structurally unchanged subtrees and preprocessing can be reused exactly on actual benchmark inputs. It needs profiling, an explicit dependency contract, ordinary numerical-equivalence checks and end-to-end measured benefit. It does not yet establish a distinct architecture or an efficiency theorem. No implementation or efficiency experiment is proposed for immediate registration here.

## Recommendation

Close the shared-switch novelty direction in its present form. Preserve the biological counterexample and exact prior-art overlaps. Continue the faithful benchmarks and await H050's measured outcome. The useful residue is a sharper modeling constraint: observed clade changes may supply uncertain context, but cannot be treated as identified shared ancestry-switch events. Any later claim must earn its support model, utility and originality separately.

This audit runs only literature retrieval,text extraction and source reasoning. It reads no genotype,truth,probability,checkpoint or model payload, executes no scientific model/fixture, registers no condition, changes no live source and performs no admission,guard or Git operation.
