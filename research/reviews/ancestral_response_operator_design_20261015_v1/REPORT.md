# Ancestral-response operator: no-go as the next architecture direction

**Recommendation: do not implement or fit this candidate now.** A differentiable sample response obtained by eliminating internal Gaussian tree states is mathematically sound. It is precisely a Kron/Schur-complement operator; its local edge sensitivity is already in the Kron-reduction literature. The project already implements the corresponding anchored Gaussian tree solve in H033. Carrying its outputs along the genome would create a different utility condition, but neither a new operator principle nor a demonstrated computational advance follows. No architecture, hypothesis, experiment or priority claim is registered here.

## The precise construction, and what it marginalizes

Let B contain every persistent observed sample in the same order at each locus, and I the current internal nodes. Samples can be sampled ancestors; B need not literally be terminal leaves. For a current observed forest define

    Q = incidence * diag(c_e) * incidence^T + diag(a_v),
    c_e > 0, a_v >= 0,
    E(z) = .5 z^T Q z - f^T z.

Require K=Q_II positive definite: for an ungrounded forest every eliminated component must touch B; positive interior grounding is a stronger sufficient condition. Set

    S = Q_BB - Q_BI K^-1 Q_IB,
    b = f_B - Q_BI K^-1 f_I.

Eliminating I gives .5 h^T S h - b^T h plus a constant. Gaussian integration and quadratic minimization have the same S and b. Actual Gaussian marginal likelihood additionally contains determinant/normalization terms; a predictive mean computation does not by itself evaluate that likelihood. For SPD Q, S^-1 is the B-by-B block of Q^-1. S is a marginal precision/Dirichlet-to-Neumann map, not the covariance or automatically a stochastic transition.

For ancestry, the channels must be described as continuous reference-affinity or logit surrogates. Exact categorical ancestral-state marginalization is ordinary tree sum-product and generally does not become a linear Schur complement. This construction does not prove that ancestral population labels diffuse as a Gaussian trait.

A confidence-anchored response is

    h_new = (S + T)^-1 (b + T h_old),  T positive diagonal.

This is the sample restriction of a full-tree Gaussian solve with sample precisions T. H033 does exactly this with interior absorption .01, observed-reference sources, frozen-emitter-derived query precision, and differentiable tree elimination. Its source explicitly calls the states predictor surrogates. See interleaved_tree_tv_gnn_v1.py:236–289,356–392. Merely materializing S changes representation, not that map.

## Exact local edit update

On one fixed vertex namespace and fixed B/I partition, let all coefficient changes be represented by

    Q' = Q + U C U^T,

where an edge insertion/deletion/weight change contributes an incidence column to U and its signed conductance change to diagonal C. Changed grounding contributes unit-vector columns. Omit zero changes. Define

    X = K^-1 U_I,
    Z = U_B - Q_BI X,
    M = (C^-1 + U_I^T X)^-1.

Provided both K and K'=K+U_I C U_I^T are nonsingular,

    S' = S + Z M Z^T.

For unchanged f, writing v=K^-1 f_I also gives

    b' = b - Z M U_I^T v.

Thus updating the precision alone while retaining an old internal-source response can be wrong. Source changes require their own exact update or new solves. Signed deletions are allowed in this batched identity; intermediate deletion-only graphs need not meet the final invertibility condition. Nearly singular denominators remain a numerical issue. These are real-arithmetic identities, not FP32 stability certificates.

Dörfler and Bullo, arXiv:1102.2950v1, §3.4, Theorem 17 and equation (35), derive the rank-one interior-edge case explicitly, including loss/appearance of a link. Their proof uses the same inverse-update machinery and states its extension to more complex symmetric perturbations. The formula above is its standard block/signed extension, not a new theorem.

Two limits defeat an unconditional fast-update claim:

* Local topology changes need not give local coefficient changes. A whole-tree GNN can change many conductances or sources after one edit; degree normalization can change other incident weights. The rank must include every changed coefficient. Small rank requires deliberately restricted edge-local, persistent coefficient semantics.
* Woodbury updates an inverse or Schur complement. It does not supply an exact rank-r update of exp(-d S). The Duhamel integral spreads a low-rank precision perturbation through changing eigendirections; the exponential difference can have full rank. A resolvent is implementable by tree solves, but generally R(d1+d2) differs from R(d2)R(d1), so it lacks exact hold-subdivision consistency.

## Inferred-node validity

**Algebraic validity does not require true ancestral identities. Computational sparsity does require an explicit common coordinate system.** Any relabeling of I leaves S and b unchanged. Therefore the safe persistent interface is all actual samples. Matching anonymous internals is unnecessary for defining the response.

Within one immutable tskit table, node IDs and edge endpoints are valid storage coordinates. edge_diffs returns all removed/inserted table edges; it does not certify one biological SPR, constant edit rank, or calibrated ancestor continuity. tsinfer's primary paper describes synthetic path-compression ancestors and inference errors that can increase edge counts. Its default relative-age ordering is not automatically generations; a separately dated product must use its own preserved provenance. This note does not assume the project's tsinfer/tsdate products use default tsinfer times.

New/disappearing internals make an active-only matrix change dimension. A union namespace with zero-grounded inactive nodes makes K singular, invalidating ordinary Woodbury. One valid convention is positive grounding on all union internals: disconnected inactive nodes then contribute no S or b when their source is zero. This changes an ungrounded model on active nodes unless the same grounding was already part of its definition. A union can also be much larger than a current tree. Active-set factorization is the simpler exact route.

A concrete public-source obstacle exists for Relate. The pinned author converter at commit b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74 sets node_const=tree_count*(N-1), appends N-1 new internal table rows for every tree and offsets their edge endpoints (ConvertToTreeSequence.cpp:383–434). This representation shares samples but provides tree-specific internal IDs. Consequently an adjacent-tree table diff can replace essentially the whole internal tree even if unlabeled topology changes little. The project's paired_inferred_dataset sanitation preserves supplied parent/child IDs rather than merging equivalent internals (relate_pipeline.py:301–329). The retrieved author commit is **not asserted byte-identical to installed v1.2.4**; no scientific files or installed binary were inspected. It nevertheless disproves treating “Relate tskit output” alone as a small-persistent-edit contract. Matching by exact clades/ages would introduce a new explicit representation rule, overlapping H035/H043, rather than recover verified biological identity.

## Cost and ordered behavior

For N active tree nodes, H retained samples and D channels, a fresh tree factorization plus response/anchored-solve action costs O(ND) work and O(ND) working space. This includes arbitrary current sample evidence and avoids dense S. H033 already uses this structure.

For r signed sparse coefficient changes, forming X by tree solves costs O(Nr), forming its small Gram matrix costs O(r^2), and factoring M costs O(r^3). Storing X and Z costs O(Nr+Hr+r^2). An already-built correction acts on D probes in O(HrD+r^2D). Explicitly updating dense S costs O(H^2 r+Hr^2+r^3) and stores H^2 entries. Keeping exact low-rank corrections indefinitely does not keep r bounded: cumulative rank can grow to H. The old base response must still act on each new arbitrary h. There is no proved O(number_of_ARG_edits) complete training algorithm here.

Dynamic-tree structures already support link/cut and local cluster maintenance. Alstrup et al.'s top trees use clusters with at most two boundary vertices; constant-size Gaussian quadratic/source summaries can be joined by elimination. That Gaussian-summary observation is a deduction here, not a claimed implementation or a theorem quoted from their applications. Maintaining a few selected responses can be attractive. Reading all H changed sample outputs has an Omega(HD) output cost; changing all sample evidence also has an Omega(HD) input cost. For ordinary N=O(H) trees this alone matches the order of a fresh tree pass. tsinfer unary/path-compression structure means N must not be assumed to equal 2H-1.

Response matrices on shared samples can genuinely fail to commute. For example, three-sample weighted stars with conductances (1,1,2) and (2,1,1) yield

    4 S1 = [[3,-1,-2],[-1,3,-2],[-2,-2,4]],
    4 S2 = [[4,-2,-2],[-2,3,-1],[-2,-1,3]],
    (S1 S2 - S2 S1)[1,2] = -1/16.

This is a hand-derived algebraic witness, not a scientific/synthetic execution result. A carried product of symmetric resolvents or heat kernels can therefore depend on genomic order. That is established time-varying linear dynamics. In contrast, updating S itself by exact edits is path-independent: returning to the same Q returns the same response. Edit-update arithmetic supplies no extra ancestral memory. Genome position is not evolutionary time, and a bidirectional or explicitly reversal-equivariant readout is still needed.

## Distinction from prior work and the project

| Prior | Exact relation |
|---|---|
| Kron reduction / Gaussian BP | S is Kron reduction; Gaussian BP/tree elimination computes the same quadratic marginal and mean. This is the closest identity, not an analogy. |
| Graph Gaussian processes | A grounded Laplacian is a standard GMRF precision. The graph Matérn paper explicitly uses (2 nu/kappa^2 + Laplacian)^nu and sparse precision. Nu=1 includes this basic family; arbitrary GP kernels are not all identical to this tree response. |
| Dynamic graph updates | Rank-one Kron sensitivity and inverse updates are established; top trees already exploit local forest edits. No new exact dynamic all-sample bound is supplied. |
| H005 | Changing sources using current predictions alone revisits feedback, whose distinct contribution was unestablished. |
| H025 / H029 / H030 | Persistent ancestral traces, whole-tree cavities and retained carrier history were already tried without a distinct history accuracy advantage. A fixed-sample interface avoids internal identity claims, but does not establish new information. |
| H033 | Exact anchored Gaussian tree solve with internal absorption and learned precisions, with backward solves. The static response is the same mathematical object. Its interleaved/blocked accuracy was essentially tied. |
| H035 / H043 | Aligned operator compositions and foreign changed-clade states already test neighboring-tree interactions. A leaf response avoids anonymous-node alignment; adding role matching to make Relate updates sparse reintroduces it. |
| Earlier hierarchical-semigroup/commutator designs | Different operator family, but shared-sample ordered products/noncommutativity were already specified and traced to prior mathematics. Swapping in a Schur response does not create priority. |

No project outcome proves this entire family useless. A full H042-plus-Gaussian-response recurrence would be a changed empirical condition, not an unchanged H033 run. There is currently no measured error tied specifically to missing Gaussian ancestral responses, no sparse coefficient/update contract for the actual representations, and no new algorithmic bound that justifies selecting it over the completed and live directions.

**No new predictive fit or broad qualification campaign is recommended.** Reopening would require a concrete new estimand or exact algorithmic capability with a bound stronger than the tree pass under a specified all-sample workload. Merely timing Woodbury or observing a nonzero commutator would not establish either. This no-go is the completed output, not a request for another audit.

## Primary evidence and scope

* Dörfler & Bullo, [Kron Reduction of Graphs With Applications to Electrical Networks](https://arxiv.org/html/1102.2950), §2.1 and §3.4/Theorem 17/eq.(35); journal DOI [10.1109/TCSI.2012.2215780](https://doi.org/10.1109/TCSI.2012.2215780). Full primary HTML read.
* Borovitskiy et al., [Matérn Gaussian Processes on Graphs](https://proceedings.mlr.press/v130/borovitskiy21a.html), [full primary HTML](https://arxiv.org/html/2010.15538), §3 precision/sparsity. Full relevant section read.
* Alstrup et al., [Maintaining Information in Fully-Dynamic Trees with Top Trees](https://arxiv.org/html/cs/0310065), §1–2 and cluster/update interface. Full relevant sections read; no Gaussian implementation claimed.
* Malioutov, Johnson & Willsky, [Walk-Sums and Belief Propagation in Gaussian Graphical Models](https://www.jmlr.org/papers/v7/malioutov06a.html). Primary abstract only; detailed tree formulas above are directly derived and present in H033 source.
* Kelleher et al., [Inferring whole-genome histories in large population datasets](https://doi.org/10.1038/s41588-019-0483-y), full author article via Europe PMC PMC6726478; age ordering and synthetic path compression read.
* [tskit data model](https://tskit.dev/tskit/docs/stable/data-model.html), [edge_diffs API](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence.edge_diffs), and [pinned Relate converter source](https://raw.githubusercontent.com/MyersGroup/relate/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/include/file_formats/ConvertToTreeSequence.cpp). Relate's primary journal page was retrieved; Europe PMC full-text retrieval failed, so no unobserved full-text claims depend on it.

AGENTS, experiment ledger, earlier verdicts and the cited project sources were read. source_refs_v1.json pins source/ledger snapshots. Literature successes and failures, exact URLs and content hashes are preserved. No actual model, observation payload, private truth, checkpoint, GPU, guard/accounting, installation or Git action occurred. Only this owned review folder was written.
