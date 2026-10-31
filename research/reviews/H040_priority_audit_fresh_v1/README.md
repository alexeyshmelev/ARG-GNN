# H040 source and priority audit

**Verdict:** H040 is a plausible tree-specific correspondence heuristic, but novelty and biological usefulness are unestablished. Its defensible candidate is the exact rule that converts a changed rooted triplet into a shared-sample-conditioned message between two anonymous within-tree LCAs. A new antisymmetric tensor, a new temporal GNN family, or a first neural ancestry method on inferred genealogies is not supported. The registered tensor-versus-eventMLP comparison tests the interaction kernel while holding correspondence fixed. It cannot establish that correspondence improves over generic local mixing.

This is an independent source/related-work audit, not an empirical review or a venue verdict. I read AGENTS.md, the two assigned H040 sources and the execution-source README, plus the inherited GRU declaration. I opened no project observations, labels, targets, checkpoints, results or journals and ran no model. Source hashes and online verification details are in literature_verification.json. No frozen source was modified.

## What the implementation actually adds

The witness chooses four stable sampled-haplotype keys without reference class or age, then examines every unordered triple containing at least two landmarks (5,692 for this panel). For each adjacent pair of observed trees it retains a triple only when both closest-pair states are resolved and differ. The LCA of each winning pair supplies an endpoint. The sample shared by the old and new winning pairs supplies an anchor. Internal-node indices remain local to each native tree. Degree normalization is computed from this observed event multigraph (witness lines 28–49, 83–96, 124–160).

For endpoint states b,c and shared-sample contexts q_t,q_(t+1), the tensor arm computes, componentwise,

    a = (U q_t + U q_(t+1))/2
    m = eta * a * [(V b)*(W c) - (V c)*(W b)]
    residual[b] -= O m
    residual[c] += O m

The implemented blocked rank-space scatter is algebraically equivalent because O is linear and bias-free. A second shared whole-tree PyG sweep propagates the injection, followed by the inherited genomic bidirectional GRU (GNN lines 35–77, 133–159). This is a sample-anchored observed association, not an estimate of ancestral-node identity or proof of a recombination event.

The operation that might be distinctive is the *combination* of restricted changed-triplet incidence, the shared sampled genome as message conditioner, and injection into winning-pair LCAs before another whole-tree sweep when inferred internal nodes lack persistent identity. The examined papers do not establish priority for that exact combination. They also do not prove it is absent elsewhere.

## Verified overlap with primary work

- **ARGMix [1] is the nearest biological precedent.** Its primary abstract explicitly describes a graph transformer using inferred ARG marginal coalescent trees and ancient reference samples for local ancestry. Thus marginal-tree ancestry learning with ancient references is already occupied. A subsequent source-only follow-up verified cached primary XML PMC13082007. Methods 4.1 uses relative TMRCA positional self-attention and population-label tokens. Methods 4.3 uses 25 nearest references plus three inferred ancestors and the query, a 29-node subgraph. Those methods describe a per-marginal-tree predictor and training subsampling of trees, not H040's adjacent-tree change incidence or genomic GRU. This closes the full-methods access gap for that version but does not establish exhaustive priority.
- **ARGformer [2] already learns genealogy-based ancestry representations.** Its methods encode leaf-to-root paths, use shared internal-node identifiers across paths/intervals, pretrain by masked-node prediction and finetune by reference-population contrastive learning. H040's anonymous-tree incidence is a different representation from the documented path/token mechanism. That distinction motivates a source comparison, not a superiority claim.
- **The tensor kernel is established technology.** ComplEx [3], equation (11), includes the antisymmetric term Im(w)_k[Re(e_b)_k Im(e_c)_k - Im(e_b)_k Re(e_c)_k]. Set Re(e)=Vh, Im(e)=Wh and Im(w)=a: H040 uses precisely this componentwise algebra, with a context-derived relation vector and learned output mixing. Data conditioning and graph placement may be useful adaptations. They do not create kernel priority.
- **Dynamic graphs plus recurrence are established.** EvolveGCN [4] explicitly handles changing or disjoint node sets by recurrently evolving GCN parameters. TGN [5] models dynamic graphs as timed event sequences using memory and graph operators. H040 uses different correspondence and does not implement either model, but cannot claim the general combination of graph change, learned interaction and recurrence. Its opposite endpoint scatter and second sweep fit ordinary message passing [6].
- **Rooted-triplet disagreement is classical.** Sand et al. [7] define the same three resolved closest-pair topologies, the unresolved case, and triplet disagreement between leaf-matched rooted trees. Their operation counts differences. H040 selects a landmark subset and turns differences into learned LCA-to-LCA incidence. The distinction is the routing construction, not discovery of a new tree-change statistic.

As a general ML claim, this remains a structured, anchor-conditioned message function on an event/factor graph. No expressivity separation from generic message passing or a sufficiently flexible event network follows. The matched eventMLP has the same 1,024 added weights and can contain mixed interactions through GELU. Parameter matching does not imply equal hypothesis classes or optimization difficulty.

As tree-specific engineering, avoiding fictitious internal-node identities while exploiting stable sampled genomes is sensible. It could help carry reference-related context across local topology changes. However, inferred topology changes can be errors, branch topology need not change at every ancestry transition, and a shared descendant does not make two LCAs homologous. The worthwhile biological question is whether this routing improves ancestry boundaries and short tracts beyond ordinary sequential context while remaining robust to inferred-tree errors.

## Properties and a concrete limitation

For a fixed event, swapping b,c negates m and also reverses the scatter signs, leaving the injection at each physical endpoint unchanged. The rank residual sums to zero, as does its linear projection. Neither is a new conservation principle.

This local orientation invariance differs from reversing chromosome coordinates. Reversing the complete hold sequence and transforming all incidence/geometry consistently makes the coupling and per-tree processing reverse equivariantly. The inherited GRU is an ordinary bidirectional nn.GRU with separately learned direction weights (whole_tree_gnn_v1.py:277), followed by a classifier. It provides ordered genomic context but guarantees neither coordinate-reversal equivariance nor invariance to arbitrary hold permutation. Chromosomal position is not an evolutionary time arrow. Reversal equivariance could be biologically appropriate without making unordered trees appropriate.

There is also a latent cancellation limit: if Vb=Vc and Wb=Wc, both the tensor and eventMLP messages are zero for every anchor a, even with valid changed incidence. Equal upstream states suffice. Equally colored/aged symmetric clades can exhibit this condition, although heterogeneous SNP/reference features may distinguish them. Exact local flags retain a structural indicator but do not universally prevent their adapter from vanishing. This is distinct from missing incidence and is detailed in latent_cancellation_addendum.md.

There is an exact K4 blind spot. Let all four landmarks form subtree L. Compare (L,((x,y),z)) with (L,(x,(y,z))), where x,y,z are nonlandmarks. Every eligible triple has at most one of x,y,z, so all eligible rooted-triplet states are unchanged. H040 emits no event, although a proper clade has changed. This is an analytic counterexample, not a measured experiment. The base tree GNN may still detect the local change, but the proposed cross-tree channel cannot address it.

## Ranked next scientific checks

1. **Test correspondence itself.** In addition to a continuation with the same two sweeps, GRU and training exposure but no cross-tree message, compare the chosen kernel on true incidence against within-boundary, shared-anchor-stratified degree-preserving rewiring. Two-edge swaps (b1,c1,s),(b2,c2,s) -> (b1,c2,s),(b2,c1,s) preserve endpoint degrees, boundary counts, anchor usage and endpoint feature marginals while disrupting exact endpoint pairing. Retain only nontrivial swaps and report the fraction actually changed. A largely immobile null is not decisive. The registered tensor-versus-MLP pair cannot replace this control. True incidence beating an effective rewired null would support useful correspondence conditional on this architecture.
2. **Challenge coverage with a simpler exact feature.** Mark each observed proper clade as lost to the next tree or gained from the previous tree by comparing exact descendant-leaf sets. This catches the counterexample without enumerating all triples or inventing ancestor identity. Use a local, parameter-matched adapter with the same warm start, two sweeps and GRU, but no endpoint pairing. If such flags explain the benefit, the useful contribution is tree-change feature engineering. If H040 adds reproducible benefit beyond them and the rewired null, a correspondence claim becomes more credible. A separate implementation contract accompanies this audit.
3. **Close the priority and biological gaps before broader claims.** The cached ARGMix methods now distinguish its 29-node TMRCA-attention predictor from H040's incidence. Compare the authors' actual implementation and ARGformer's node-sharing assumptions against this precise rule before a broad priority claim. Then evaluate the surviving mechanism on untouched simulated replicates with boundary/short-tract measures and controlled inference/phase error. Freeze these checks before reading their truth. Reused development validation cannot establish general biological usefulness. A general ML contribution would additionally need a property or transferable mechanism beyond this genealogy task.

## Primary references

[1] Shanks C, Bonet D, Comajoan Cara M, Ioannidis AG. *Graph transformer for ancient ancestry inference*. bioRxiv, 2026-04-07, DOI 10.64898/2026.04.05.714076. [Preprint](https://doi.org/10.64898/2026.04.05.714076). [Accessible primary metadata and abstract](https://api.biorxiv.org/details/biorxiv/10.64898/2026.04.05.714076). [Verified primary full methods](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13082007/fullTextXML).

[2] Bonet D, Shanks C, Cara MC, Abante J, Ioannidis AG. *ARGformer: learning on ancestral recombination graphs with transformers*. Bioinformatics, 2026, DOI 10.1093/bioinformatics/btag438. [Primary full text](https://europepmc.org/articles/PMC13501305). [Verified full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13501305/fullTextXML).

[3] Trouillon T et al. *Complex Embeddings for Simple Link Prediction*. ICML/PMLR 48:2071–2080, 2016. [Paper](https://proceedings.mlr.press/v48/trouillon16.html). [Verified PDF, equation 11](https://proceedings.mlr.press/v48/trouillon16.pdf).

[4] Pareja A et al. *EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs*. AAAI 2020. [Primary preprint](https://arxiv.org/abs/1902.10191).

[5] Rossi E et al. *Temporal Graph Networks for Deep Learning on Dynamic Graphs*. 2020. [Primary preprint](https://arxiv.org/abs/2006.10637).

[6] Gilmer J et al. *Neural Message Passing for Quantum Chemistry*. ICML 2017. [Primary preprint](https://arxiv.org/abs/1704.01212).

[7] Sand A, Brodal GS, Fagerberg R, Pedersen CNS, Mailund T. *A practical O(n log² n) time algorithm for computing the triplet distance on binary trees*. BMC Bioinformatics 14(Suppl 2):S18, 2013. DOI 10.1186/1471-2105-14-S2-S18. [Primary full text](https://europepmc.org/articles/PMC3549851). [Verified full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3549851/fullTextXML).
