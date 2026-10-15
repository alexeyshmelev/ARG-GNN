# Cut memory novelty and whole tree scaling

The present H049/H050 memory has a defensible domain-specific inductive bias, but distinct generic operator novelty is unsupported. H050 reuses H049's operator on a different inherited predictor. The strongest next direction is a controlled, mathematically specified test of **which conditional history should survive uncertain inferred-tree changes**, together with exact whole-tree scaling. Adding another GRU would not address that question.

## What the operator contributes

Let A, B and C be successive observed cut partitions of the same H sampled leaves. Let L_X be the hard leaf-to-cell incidence matrix and D_X its cell-count diagonal. The source carries old pair-cell states AB into new pair cells BC through

\[
P=D_{BC}^{-1}L_{BC}^{T}L_{AB},
\qquad
P_{BC,AB}=\frac{n_{ABC}}{n_{BC}}.
\]

The adjacent-marginal arm instead uses n_AB/n_B. The stable-leaf arm avoids this intermediate pooling. Thus triple support can preserve within-B information that the adjacent null deletes, while the leaf arm retains more detailed history. Triple memory has no general information or expressivity advantage over stable leaf memory.

The implementation uses a rank-eight projection of first-sweep upward plus outside states at real cut representatives, exponential decay in physical bp, forward/reverse histories, pooling back to current cut groups and a second complete tree sweep. Its 513 added scalars are two 32-by-8 maps and one decay-length parameter. Cut membership is observed at 350 generations. These partitions do not establish persistent ancestor identity, and the learned bp decay length is not an estimated demographic rate. Counts include the full observed panel, so this is transductive pooling rather than a reference-only ancestry flow.

H049 retains its common grid decoder. H050 injects the same memory into H045's own-state predictor, preserving its full-parent BiGRU and direct native-hold readout. H050 therefore tests utility on that predictor, rather than introducing a second transport primitive. Exact sources and function locations are in [source_refs_v1.json](/workspace-SR003.nfs2/ARG-GNN/research/reviews/H050_novelty_scaling_fresh_20261016_v1/source_refs_v1.json).

## Closest established mechanisms

| Evidence | Consequence for the novelty claim |
|---|---|
| [Whitehouse et al. 2024](https://doi.org/10.1093/molbev/msae223), GCNs on Tree Sequences and architecture methods | Six tree GAT layers followed by node/tree GRUs already combine genealogical topology and genomic order. Their sequential-window cap of 128 trees leaves room for an all-tree scaling contribution. |
| [Rosvall et al. 2014](https://www.nature.com/articles/ncomms5630), Memory networks, and [De Bruijn goes Neural 2022](https://arxiv.org/html/2209.08311v1), Sections 2–3 | Previous/current pair states, conditional path counts, higher-order neural propagation and projection back to original nodes are established. Conditioning the same incidence relation in the reverse direction does not alone establish priority. |
| [TGN 2020](https://arxiv.org/html/2006.10637v3), Section 3, [DyGFormer 2023](https://arxiv.org/abs/2303.13047), and [GraphMixer 2023](https://arxiv.org/abs/2302.11636), abstracts | Graph history, memory, co-occurrence features and even simple mean-pooling alternatives are established. These are related mechanisms, not algebraically identical implementations. |
| [ARGformer 2026](https://doi.org/10.1093/bioinformatics/btag438), Section 2.1, and [ARGMix 2026 preprint](https://doi.org/10.64898/2026.04.05.714076), Sections 4.1 and 4.3 | Path tokens and 29-node reference/query subgraphs differ materially from retaining every native vertex. Full-tree, mutation-supported per-leaf inference remains a useful comparison axis, rather than a new generic recurrence claim. |

The current PyG backbone implements upward aggregation and recipient-message-excluded downward recursion through complete native trees. This gives a full-tree receptive field without all-pairs attention. PyG supplies aggregation machinery, not new learned layers. The cavity calculation is an architectural bias, not a theorem of exact probabilistic inference. This focused comparison supports withholding a generic novelty claim; it is not an exhaustive priority search.

## Scaling of the current source

Write T for native holds, G for distinct consecutive cut episodes, K for the total cut representatives over all holds, N for retained vertices, S for sites, d=32 and r=8.

| Component | Leading work or storage |
|---|---|
| Site encoder | O(SH(19d+d²)) work |
| Complete upward/downward sweep pair | O(Nd²) work; H049/H050 use two pairs |
| Inherited genomic GRU | O(THd²) work and serial dependence along T |
| Cut memory | O(Kdr+THr) work |
| Cut-plan tensors | Exactly 13K+8TH+68GH+H bytes with the source dtypes |

There are at most H occupied AB cells and H occupied ABC triples at an event. The source lifts to leaves and reduces into occupied cells. It does **not** require an H³ contingency tensor or H² attention matrix. Its layouts are nevertheless padded to H, and its histories remain indexed by every hold even when G is small.

For illustration, T=10,000 and H=952 give a worst-case cut-plan tensor bound of 847,280,952 bytes. One FP32 [T,H,8] history alone uses 304,640,000 bytes. The code constructs left, right and combined histories, in addition to full-bank neural buffers. These are analytic sizes, not measured peak memory. Python metadata, original observations, training activations, I/O and H049's common decoder add costs. The exact derivation is in [mathematics_and_scaling_v1.json](/workspace-SR003.nfs2/ARG-GNN/research/reviews/H050_novelty_scaling_fresh_20261016_v1/mathematics_and_scaling_v1.json).

## Efficient processing that retains whole trees

1. **Pack complete trees into contiguous blocks and batch their depth levels.** Encode each whole tree once per required pass, retain all leaves/unary vertices, and reuse its contexts across query readouts. Block boundaries are scheduling boundaries. A future exact checkpoint/offload schedule must preserve both directional recurrent boundary states and all gradient paths, without resets, detachment or truncated history. The current H050 interface still requires the complete bank on one device.

2. **Keep the succinct tree-sequence representation and cache observation-only recipes.** [tskit edge_diffs](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence.edge_diffs) exposes inserted/removed edges. It can reduce topology parsing and structural recomputation. It does not make this nonlinear neural encoder O(number of changed edges): one edge can change many upward ancestors and outside states, and per-hold site evidence can change forcing throughout a tree. Exact learned-state caches must be invalidated after parameter changes. Producing all per-hold/per-leaf outputs itself costs Ω(TH).

3. **Remove padding before inventing a denser recurrence.** Ragged occupied-cell layouts can reduce the 68GH term. Within an unchanged partition, the affine updates compose associatively as (a₂,b₂)∘(a₁,b₁)=(a₂a₁,a₂b₁+b₂), permitting segmented scans. Distinct per-hold forcing must remain. Across partition changes, composing transport matrices can densify to H-by-H, so a cheap logarithmic-depth scan does not follow. The inherited nonlinear GRU also remains a separate sequential bottleneck.

These are exact scheduling/representation opportunities. None requires subtree sampling, path-only replacement or a claim of free incremental neural inference.

## One falsifiable improvement direction

**Shrink only the conditional innovation that is unsupported under inferred-tree uncertainty.** In leaf space define the orthogonal averaging projector Π_X=L_X D_X⁻¹L_Xᵀ. For an old state x in the AB subspace, the current triple and adjacent carries are Π_BC x and Π_B x. Their difference is

\[
Jx=(\Pi_{BC}-\Pi_B)\Pi_{AB}x.
\]

Because B is coarser than BC, Π_BC−Π_B is an orthogonal projector. Hence ‖J‖₂≤1, Π_BJ=0, and J vanishes for every old state exactly when the empirical A and C memberships are conditionally independent given B. Nonzero J establishes witness dependence, not ancestry predictiveness.

A specific prospective family is T_α=Π_B+α_B(Π_BC−Π_B), with 0≤α_B≤1 constant inside each B. For fixed observation-derived α, it preserves each B mean and is a contraction in x. α=0 and α=1 recover the current marginal and triple endpoints. These are exact mathematical constraints, not a new priority claim for shrinkage or conditional expectation, and they do not certify stability of the complete learned predictor.

The biological hypothesis is that observation-only confidence in the conditional innovation can reduce spurious short ancestry tracts caused by tree-inference errors while retaining real short tracts. Test it on independent admixed parents across prespecified inference-error levels against fixed α, the existing triple/null/leaf arms and continued training, with capacity and history matched. Require fewer false switched bp/false short tracts, without sacrificing true ≤10 kb tract recall, boundary accuracy or calibration. If fixed shrinkage works equally well, or gains erase true short tracts, reject the adaptive-innovation claim. Score unavoidable within-hold boundary misses separately. Keep current H050 unchanged while it answers its registered utility question.

## A defensible cross domain diagnostic

[TGB tgbn-trade](https://arxiv.org/html/2307.01026v2), dataset description and Section 5.2, supplies a natural order: annual agricultural trade between nations, with next-year destination trade proportions as the target. A future adaptation could build a full deterministic dendrogram from each year's observed trade profiles and retain a fixed training-defined country panel with an explicit missing-data policy. Country identities provide witnesses and calendar years provide order. The target is future observed trade, rather than a label manufactured from the same clustering.

This would expose whether regrouping history adds predictive information beyond current groups. Use only past observations, a forward-only memory, official chronological evaluation and NDCG@10. Include persistence and moving averages, which were strong in TGB's original Table 4, plus a matched whole-tree encoder with causal GRU and the leaf/marginal controls. The roughly three-decade record is a small mechanism diagnostic, not strong evidence of broad cross-domain generalization or large-T scalability. Dendrogram construction may also require quadratic preprocessing in panel size. This requires a separate adapter/protocol: the present H050 bundle hard-codes 952 leaves and is not directly reusable.

[GST 2026](https://arxiv.org/html/2606.05116v1) instead models unordered sets of graphs through graph tokens and set attention. Sorting those graphs into an artificial chain would change the problem and supply neither a biological coordinate nor valid shared-leaf witnesses. It is not a justified ordered-memory benchmark.

Source and public-literature analysis only. No model/data execution, architecture changes, experiment, installation, GPU/job, accounting or Git action was performed. Claim-level primary locators and hashes are in [public_reference_evidence_v1.json](/workspace-SR003.nfs2/ARG-GNN/research/reviews/H050_novelty_scaling_fresh_20261016_v1/public_reference_evidence_v1.json).
