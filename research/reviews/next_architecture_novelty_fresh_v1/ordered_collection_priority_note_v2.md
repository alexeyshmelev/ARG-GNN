# Ongoing novelty assessment: ordered collections of complete local genealogies

Source-only update. This note inspects definitions, not current fits or data. No priority finding follows from absence in the inspected source. The three distinct ingredients are complete tree context, observed sample correspondence and genomic ordering; each already has substantial prior art.

**Identity and coordinates.** A persistent sample ID identifies the same observed haplotype within this panel. A qualified internal ID identifies an inferred object under its producer's rules. Neither certifies a true ancestor, census population or ancestry tract. Internal indexes without a qualified namespace are merely graph-local storage. Descendant-set overlap is observed relational support, not proof of ancestral identity. Genomic position x orders trees in physical bp or declared genetic distance; genealogical age t locates ancestors in generations before the present. Recurrence along x is not evolution along t. A spatial decay length is not an admixture age. Older proxies have no fabricated descendant at350.

## Nearest-prior-art matrix

| Closest mechanism | Established capability relevant to this request | Boundary for a prospective contribution |
|---|---|---|
| [ARGMix](https://www.biorxiv.org/content/10.64898/2026.04.05.714076v1.full), inspected current-release kernel/GRU source | Reference-weighted TMRCA votes, genealogical attention and genomic recurrent filtering | Ordered ancestry prediction and reference affinity are already covered. Greater whole-tree input access is a disclosed information difference, not itself a new operator. |
| [ARGformer](https://doi.org/10.1093/bioinformatics/btag438), inspected methods | Shared inferred ancestral-node path tokens, masked-node learning, contrastive reference retrieval | Shared ancestral vocabulary and genomic co-occurrence are established. A complete-tree GNN must explain what path-token/retrieval representations lose. |
| [AncestralPaths](https://doi.org/10.1038/s41586-023-06705-1), project native-source audit | Ancestor-shell cohort composition and dates for local ancestry | Reference-labelled genealogical neighborhoods and age-conditioned ancestry classification are established. |
| [Graph Set Transformer](https://arxiv.org/abs/2606.05116), project primary/source audit | Local propagation, graph pooling, collection attention and feedback into nodes | Full-tree plus collection feedback, even before query compression, is not a fresh contribution. Position/span encoding alone is weak. |
| [TGN](https://arxiv.org/abs/2006.10637v3), §3.1; [DySAT](https://arxiv.org/abs/1812.09430) | Event-updated node memory, message/updater interfaces and structural/temporal embeddings | Qualified-node memories and whole-tree reads are domain choices inside established temporal GNNs. Matrix rather than vector storage does not establish novelty. |
| [GraphSSM](https://arxiv.org/abs/2406.00943v2), Eqs5/21, AppendixB.2; [EvolveGCN](https://arxiv.org/abs/1902.10191) | Irregular graph sequences, graph-conditioned filtering, changing node sets and state/parameter recurrence | Variable ancestral vertices and persistent samples are supported cases. GraphSSM explicitly retains intersecting-node state and initializes new nodes. Exact ZOH/semigroup behavior is inherited. |
| [Graph Matching Networks](https://arxiv.org/abs/1904.12787), project prior-art audit; Relate branch association | Joint graph-pair reasoning, cross-graph correspondence; descendant-set branch comparison | Soft or clade-based correspondence among anonymous ancestors is not intrinsically new. |
| [Tree-LSTM](https://arxiv.org/abs/1503.00075v3), §3.1; pruning/rerooting algorithms | Learned child aggregation and efficient subtree/outside messages | Complete-tree sweeps, cavities and exclusion are established. Complete pre/post differences are contextual changes, not causal recombination attribution. |
| [DeltaNet](https://arxiv.org/abs/2406.06484v6), §2.2; [Gated DeltaNet](https://arxiv.org/abs/2412.06464v3), Eq10 | Associative matrix lookup and prediction-error replacement with gated retention | H029/H030 reuse these primitives. Their covariance/cavity key choice is a narrow coupling, not new recurrent algebra. |
| [Tree OU covariance](https://arxiv.org/abs/2003.13004v2), Eqs8–9; [second-order GNN pooling](https://arxiv.org/abs/2007.10467v1), Eq3 | Correlated tree observations and second-order feature statistics | H032's donor-support Gram uses this neighborhood. No posterior ancestry covariance, new pooling primitive or ordered-architecture claim follows. |
| [Continuum Attention](https://arxiv.org/abs/2406.06486v4), GraphSSM and project Kron/heat/SPDE audits | Interval integrals, exact hold composition, Gaussian/diffusion state exchange | Physical-span weighting, serial subdivision properties, covariance validity and ordered noncommuting actions are inherited mathematical properties. |

The new inspection revisited cached GraphSSM Eq21 and current project definitions. TGN/tree/DeltaNet/OU/pooling primary texts were inspected in the earlier fresh-review packet. Other rows use existing project source audits and are not newly exhaustive operator comparisons.

## Routes already present here

- H021: observed census350 query partitions, ordinary block diffusion and sample-set Jaccard retention. Replacing the name, adding a recurrence or using several age scales does not invent partition diffusion.
- H029: complete-tree reference cavities, atomic attachment/source events, qualified child-carrier GRU or delta matrix, pre-query whole-tree read. Its completed history controls did not establish an ancestral-continuity benefit.
- H030: class-balanced unit log-cavities and learned telescoping age potential; the same conservative carrier-history policy. Normalization and positive rerooting are established.
- H032: full/subtree/outside donor-support Gram gates in a static GNN, independence and marginal-outer controls. No carrier scan. First-moment parity is exact at equal cost parameters; independent fitting can change p. Alignment sham is explicitly omitted for unresolved cost. This is the previously proposed evidence route, now defined, so it is not an unexplored mechanism.
- Earlier IntervalLineageGNN: all-node source/target attention with same-sample and descendant-Jaccard biases, interval integration and context feedback. Generic all-node clade attention must not be proposed again.

## One low-confidence unexplored coupling

A remaining representational question is **ancestral-age-indexed branch history transferred by persistent-sample overlap**, rather than inherited native internal IDs. It is not represented by the inspected H021/H029/H030/H032 definitions. It does not address a demonstrated cause of H031's weak SNP readout.

At a given age t, a complete dated tree partitions eligible samples among native branches crossing t. A sample is eligible only where its real sample-to-root path exists; no older sample is projected downward. Let P_g(t) be sample-to-crossing-branch incidence and h_g(t) the branch-state vector. At a genomic boundary, on common eligible samples define

T[new←old](t) = diag(P_new(t)ᵀ1)^(-1) P_new(t)ᵀP_old(t),
h_new(t) = T(t)h_old(t).

For unchanged eligible rosters this averages historical state according to actual intersection counts. It preserves constants and sample-count-weighted mass and is a contraction in the corresponding weighted norm. These are ordinary conditional-expectation/projection facts. If eligibility/path coverage changes, missing mass must be explicitly reset and those conservation statements no longer apply unchanged. Neither a common inferred clade nor the transfer asserts a common biological ancestor.

Each complete-tree encoder supplies observed-only forcing f_g(e,t). Standard genomic hold evolution could be

h(x+Δ,t) = exp(-Δ/ell_bp)h(x,t) + (1-exp(-Δ/ell_bp))f_g(e,t).

Both directions use positive genomic distances. Transported branch fields enter a full-tree decoder before the census query read, keeping x and t distinct. All native branches remain encoder inputs, including branches not crossing350. No query truth controls the incidence, forcing or transport.

This proposes a specific history representation and boundary correspondence. It does **not** invent averaging, functional transport, recurrence or a semigroup. In sample space the transfer is another partition projection, making a multi-horizon H021 model a particularly close alternative. Exact continuous-age history can accumulate old breakpoints inside current branches; one constant state per current native edge is not automatically exact. A bounded fixed-age basis changes the operator and must be disclosed, with total state/activation counts matched. No efficiency claim follows.

**Novelty and falsification condition.** A distinct coupling claim needs an explicit operator-level difference from multi-horizon H021, sample-correspondence supra-graph/GraphSSM and existing clade attention. If the same operator is recovered by an ordinary incidence lift/project construction, only the task formulation/implementation may be distinct. Historical priority remains unresolved even if the operator differs.

Predictive usefulness would require gains over those equally informed controls with matched age information, actual live-state budget, encoder/decoder depth, optimization and selection. A transport-only overlap sham can retain instantaneous trees/labels/fields while replacing intersection alignment by a coupling with the same row/column sample-count marginals. A tie weakens the overlap-history explanation. Collapse to a350-only field tests information beyond H021's horizon, but must not confound that with extra states or age features. Native-ID renaming, genealogy serialization and splitting unchanged genomic holds should not change the modeled signal under the stated forcing contract. Inferred dating/topology perturbations and independent parents are necessary for biological interpretation.

Current evidence does not justify calling this a useful novel GNN. Its value is a precise unresolved comparison. A narrow supported representation mechanism or biological finding would be more defensible than a broad claim about the first ordered whole-ARG GNN.
