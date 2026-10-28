# Closest prior art and novelty boundary

28 October 2026. Primary papers and selected author implementation sections were inspected. This is a targeted comparison, not a systematic evidence-of-absence search. Download URLs, timestamps and SHA256 hashes are in `sources/*receipts.json`. Retrieved author code was read as evidence, never installed or executed.

## What the sliding 3D analogy already means in the literature

**MS-G3D is an exceptionally close conceptual antecedent.** Liu et al., *Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition*, CVPR 2020, explicitly introduces a sliding temporal window containing the union of node sets from several frames, adds cross-space-time edges, applies joint graph convolution, and calls it analogous to a classical 3D convolution block. Section 3.3, equations 5–7, constructs its block adjacency by tiling the spatial adjacency into all window blocks. It also supports dilated windows and multiple window scales. This directly precludes claiming sliding graphs, joint within/across graph propagation, absence of recurrence, multiscale windows, or the 3D analogy as our invention.

- Paper: https://arxiv.org/abs/2003.14111
- Author repository: https://github.com/kenziyuliu/MS-G3D
- Pinned inspected commit: `2f927f47143c7242e4875ddb08727bf5396aab46`.
- `model/ms_gtcn.py`: `UnfoldTemporalWindows` uses `nn.Unfold`; `build_spatial_temporal_graph` tiles `A + I`; `SpatialTemporal_MS_GCN.forward` applies adjacency with `einsum` followed by an MLP.
- `model/msg3d.py`: `MS_G3D` unfolds windows, performs the joint convolution, and uses a `Conv3d` to collapse the window dimension. Its multiple-window module supplies windows 3 and 5 by default.

Its skeleton joints have stable semantics and a common graph roster. Anonymous internal genealogy nodes do not. Our branch-overlap matrix supplies a different, observed correspondence using shared sampled descendants and overlapping ancestral ages. The primitive neural update is still familiar relational graph convolution. A dense tiled skeleton adjacency would be inappropriate for anonymous genealogies and expensive at 952 leaves, but these practical differences do not establish mathematical priority.

**STSGCN independently gives the simple persistent-node construction.** Song et al., *Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting*, AAAI 2020, constructs localized graphs from three consecutive spatial graphs and models spatial and temporal effects synchronously. Its middle-frame output avoids treating all overlapping windows as separate targets.

- Primary PDF: https://ojs.aaai.org/index.php/AAAI/article/download/5438/5294
- DOI: https://doi.org/10.1609/aaai.v34i01.5438
- Author repository: https://github.com/Davidham3/STSGCN
- Pinned inspected commit: `3f825f6421afb84fb69ef560179bf799fbf4481c`.
- `utils.py:construct_adj`: repeats A on the diagonal, links each node to itself in adjacent time slices, and adds self-loops.
- `models/stsgcn.py:stsgcm`: stacks graph-convolution operations, crops nodes N:2N from each result, and takes a maximum over depths.
- The individual/shared layer implementations slide three-frame windows and reshape B×3×N×C to 3N×B×C before this module.

This rules out novelty for merely placing complete trees next to each other, joining corresponding leaves, applying one GNN, and retaining a central tree. Our genealogy-specific branch edges and continuous-age measure distinguish the proposed instance, not the general schedule.

**STGCN already removes the recurrent requirement.** Yu, Yin and Zhu, *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*, IJCAI 2018 (https://arxiv.org/abs/1709.04875), sections 3.2–3.3, sandwiches graph convolution between gated temporal convolutions. It explicitly discusses parallelization and avoiding recurrent step dependencies. This is factorized rather than our simultaneous relation sum, but is a mandatory simple comparator concept. We read its paper architecture/method sections, not its source implementation.

## Product graphs and mathematical convolution

**GTCNN formalizes the general graph-time operator.** Sabbaqi and Isufi, *Graph-Time Convolutional Neural Networks: Architecture and Theoretical Analysis*, inspected preprint https://arxiv.org/abs/2206.15174v1, sections 2–3, defines Cartesian, Kronecker and strong product supports, then filters as sum_k h_k S_product^k, followed by nonlinearities. It discusses parameterized product coupling and sparse recursive evaluation with cost O(K |E_product|). Its antecedent is Isufi and Mazzola, *Graph-Time Convolutional Neural Networks*, 2021, https://arxiv.org/abs/2103.01730v1 and https://doi.org/10.1109/DSLW51110.2021.9523412 (bibliography/abstract verified here, full 2021 methods not separately read).

Our changing trees do not share a stable internal-node graph and therefore do not form a literal fixed Cartesian product. They form a block graph with observed inter-slice correspondences. Replacing identity blocks with biologically motivated couplings is a specialized support construction. It does not create a new shift-and-sum convolution principle. A generic typed-edge GNN can implement the operator exactly. The exact-title GitHub repository search returned no results; we do not claim this means GTCNN author code is unavailable.

**Space-Time GNNs cover joint operators and continuous temporal variables.** Hadou, Kanatsoulis and Ribeiro, *Space-Time Graph Neural Networks*, inspected https://arxiv.org/abs/2110.02880v2, sections 3.1–3.3, builds filters from graph and time shift operators and composes them with pointwise nonlinearities. Its perturbation analysis does not transfer automatically to changing genealogy supports, substochastic branch couplings, or near-zero branch normalization. We inspected the operator definitions and finite-filter network construction, not every stability proof. The appendix names an existing GNN library; no specific author implementation of its space-time operator was audited here.

Genomic coordinate is not temporal evolution of a single physical network, and ancestral age is another distinct coordinate. These papers motivate computations and comparison; their physical interpretations and stability claims cannot simply be imported into an ARG model.

## Existing genealogy and internal lineage

The current `research/redesign_20261028/literature/REPORT.md` already checks GST, Whitehouse et al., ARGMix, DySAT, GraphSSM, GraphSAINT and Cluster-GCN. We use that source audit for those findings rather than claiming a new primary reread. Whitehouse et al. and ARGMix establish earlier joint use of genealogical and genomic information, including recurrent released models. GST establishes another graph-set architecture. They rule out a broad first joint-genealogy learning claim; they do not establish that recurrence is necessary.

The same internal audit derives normalized descendant-intersection transport. `research/reviews/continuous_clade_transport_novelty_20261017_v1/REPORT.md` shows that lifting old group states to persistent leaves and pooling into new partitions is normalized overlap. `research/reviews/fresh_ARG_operator_research_20261017_v1/REPORT.md` records earlier H035 signed graph/alignment compositions and H040 antisymmetry. A commutator or a changed name would therefore repeat existing internal ideas.

Our exact branch coupling is an integral of partition incidences over shared ancestral age:

    M_ts[u,v] = sum_i integral 1{i descends from u at a in t}
                              1{i descends from v at a in s} d eta(a).

It is an ordinary overlap/Gram construction on sample identity × ancestral age. Positive semidefiniteness of the combined unnormalized Gram, when interpreted on this common domain, is elementary and not a new kernel theorem. Normalized directional B_ts and the learned residual network need not be symmetric or PSD. Its substochastic normalization expresses observed branch support, not conserved biological ancestry.

The selected-cut frozen model already includes branch dates in its within-tree encoder. Showing that continuous overlap distinguishes two date configurations missed by a coarse cut table does **not** show that the full frozen model cannot distinguish those trees. The actual difference is where and how date-resolved foreign states are coupled, and whether doing so gives a better inductive bias at an acceptable cost.

## Recommendation

Do not present the proposal as a novel sliding 3D graph convolution. A defensible prospective claim is narrower:

> We investigate a bounded joint convolution over complete inferred genealogies whose cross-tree edges integrate persistent-sample overlap across contemporaneous ancestral branches, and test whether this correspondence improves local ancestry and boundary recovery relative to same-leaf and selected-cut context.

This remains an unvalidated hypothesis. The targeted search did not identify this exact complete-tree branch-overlap ancestry network, but it also cannot establish priority. The formulation is sufficiently different from the frozen implementation to test separately; it is not sufficiently different from general spatiotemporal graph methods to claim a new operator family. Its continuous-age correspondence is a refinement of prior cut transport, not an independent discovery. A negative comparison should retire the added complexity while preserving the useful bounded minibatch execution.
