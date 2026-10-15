# One candidate: sample-witness order-2 cut memory

**Decision:** generic operator priority is a no-go, but a narrowly defined domain inductive-bias utility condition remains untested. This is one concrete source-only candidate, not a registered experiment or recommendation to launch before root's decision. No observed-data support distribution or accuracy was measured. H048 and all live sources remain unchanged.

The candidate retains a latent state for each nonempty intersection of the two most recent distinct observed census-cut groups, rather than immediately averaging all incoming histories into the current group. The final event/marker-grid contract in section 9 governs unchanged-cut holds and the prospective common output grid. The next transport is weighted by the number of identical observed leaves supporting the entire three-tree itinerary. It can preserve a dependence that adjacent overlap averages discard. Its mathematical primitive is exactly ordinary incidence lift/project on an expanded state space, with direct memory-network and De Bruijn GNN precedents. A useful, honestly scoped contribution would therefore have to be the **observed dated-tree cut, sample-witness path weighting and demonstrated denoising/prediction bias under inferred genealogies**, not invention of higher-order temporal message passing.

All 48 existing decisions and the H040/H042–48, raw-overlap, commutator and Kron-response audits were consulted. This packet does not revive those mechanisms under new names.

## 1. Biological selection: a genuine partition, not all overlapping clades

Inputs are one whole ordered collection of complete observed dated marginal trees, their original positive physical spans, all actual sampled-leaf correspondences, and the unchanged observation-only H042 whole-tree/SNP encoder. Graph-local internal indexes have no identity across trees. Every original vertex, edge, query and reference remains in that encoder and the output roster.

Fix the existing C4 task horizon c=350 generations, without using query ancestry labels. For each sampled leaf i:

- If its actual sampling age is at most c, ascend to the highest existing ancestor u with observed age(u) at most c. Assign i to that u's group.
- If its actual sampling age is greater than c, make it a singleton state represented by its actual sampled node, with an above-horizon mask. This is sample memory, **not a fabricated ancestor at 350**.
- If a root is already younger than c, the extant root is the representative and the group is marked root-saturated. No extrapolated node above the observed root is inserted.

This selects disjoint blocks P_t covering every observed leaf. For eligible leaves, i and j share a block exactly when observed age(MRCA(i,j)) <= c. Monotone node dates make the cut an equivalence relation. Inclusive equality is explicit here; the old H021 Q-only API required sample ages strictly below c. This boundary convention is not a novel mechanism.

All descendant leaves of a selected at-or-below-c node are themselves at or below c, so those cut groups are full sampled-descendant clades. Above-c sampled leaves are separate singleton blocks. Equal-age unary chains use the highest existing node at or below c, defined by topology and dates, never numeric internal ID. Branch/node features are taken from that actual representative; this sketch does not interpolate a neural state along an edge. It does not claim unary-insertion invariance.

Let g_t(i) be the group of leaf i at hold t. P_t's native clades are disjoint by construction; do not substitute the full nested proper-clade collection into the normalized formulas below. A cut is a structural hypothesis on inferred trees, not certified common ancestry, ancestral individual identity or population membership. Older reference labels remain ordinary observed proxy features and receive no invented census truth. Genomic order x is not genealogical time c.

## 2. Exact layer

Use D=32 native first-sweep states and rank r=8. For cut representative v, take q_t(v)=upward_t(v)+outside_t(v), both already supplied by the first complete-tree sweep. Define f_t(B)=U q_t(rep(B)), with one bias-free U:32->8 shared across directions. Every feature and cut is observed-only.

For adjacent groups A in P_(t-1), B in P_t, let

    n_AB = |A intersect B|,       E_t = {(A,B): n_AB>0}.

Keep state m_t(A,B) in R^8 for every nonempty intersection. At the next boundary, for C in P_(t+1), define

    n_ABC = |A intersect B intersect C|,
    n_BC  = sum_A n_ABC,
    alpha_t(A | B,C) = n_ABC/n_BC.

For existing pair state m_t, its transported value at pair (B,C) is

    z_(t+1)(B,C) = sum_A alpha_t(A | B,C) m_t(A,B).

With current hold width w_(t+1) in physical bp,

    rho_(t+1) = exp(-w_(t+1)/ell_bp),
    m_(t+1)(B,C) = rho_(t+1)*z_(t+1)(B,C)
                   + (1-rho_(t+1))*f_(t+1)(C).

A shared learned ell_bp is bounded 100..100000bp and initialized 5000bp, using the already established scalar sigmoid parameterization. Initialize m_0(B)=f_0(B); the first pair update simply inherits that old group field. The reverse branch applies exactly the same definitions in reverse genomic order, with positive widths and shared U/ell. This is ordinary exponential forgetting, not an admixture-age estimate.

Read pair states back to each current group without destroying them in the recurrence:

    r_t^F(B) = sum_A (n_AB/n_B) m_t^F(A,B),

with the analogous reverse read. Add

    O[0.5*(r_t^F(B)+r_t^R(B))-f_t(B)]

at the actual representative before the existing second complete-tree sweep, alongside the unchanged H042 local flag adapter. O:8->32 is bias-free and initialized to zero. Retain the original full-panel leaf context, BiGRU, classifier, loss and scoring support. No state or old/new source gradient is detached.

U256 + O256 + ell1 adds 513 learned scalars, totaling 31,301 on the existing 30,788-weight H042 body. Both directions share these weights. These are design counts only, not an instantiated model or source qualification. The existing body must actually expose both first-sweep fields during implementation.

The layer is well-defined over all original samples, including references and non-scored context leaves. It must not propagate only 686 queries or infer a new sample roster. Hidden truth, migration labels, population annotations of internal nodes, trainable internal-ID embeddings, native-ID equality and query labels never enter it.

## 3. What information is retained, and the precise limit

Consider six observed leaves. At the old hold, let A1={0,1}, A2={2,3}, R={4,5}; at the middle hold B={0,1,2,3}, R; at the new hold either:

    X: C1={0,1}, C2={2,3}, R
    Y: C1={0,2}, C2={1,3}, R.

These are valid partitions induced by dated rooted trees at c=350: within-pair nodes may have age100, old/new four-leaf roots age500, the middle four-leaf node age200, and the overall root age800. X and Y have identical adjacent overlap matrices:

    old -> middle      [[2,0],[2,0],[0,2]]
    middle -> new      [[2,2,0],[0,0,2]].

Within B, X has two supported triples with counts2, while Y has four with counts1. If incoming pair values are1 on(A1,B) and0 on(A2,B), transport to(C1,C2) yields(1,0) inX and(1/2,1/2) inY. Adjacent overlap means yield(1/2,1/2) in both. Conditional mutual information within B is log2 versus0, a standard statistic, not a new estimand.

This demonstrates only that **adjacent overlap marginals are insufficient for this conditional transport**. The full leaf-labelled trees differ. It does not prove information is absent from an injective whole-tree embedding, a stable-leaf BiGRU or a sufficiently informed supra-graph model. The existing H042 source performs whole-tree encoding, then independent temporal scans along each leaf, followed by pointwise classification. The proposed route explicitly regroups temporal fields through current cut nodes *before* the second spatial sweep. That bypasses a particular spatial/temporal bottleneck. A leaf-memory-plus-regather layer can expose the same observations and is the required stronger control.

Nor is order-2 memory exact unbounded preservation of a single leaf's entire history. Once distinct older itineraries merge inside one current pair cell, their separate memories are averaged. Future triple weights cannot recover correlations already lost inside that cell. “Exact sample-witness history” would therefore be an overclaim. The transport is exact for the specified pair-constant field; its representation deliberately retains only the previous/current group index plus an averaged recurrent value. This finite-memory pooling may denoise inferred topology, or may discard useful ancestry information.

Pure transport preserves constants and count-weighted total value. Jensen gives

    sum_(BC) n_BC ||z_BC||^2 <= sum_(AB) n_AB ||m_AB||^2.

These are conditional-expectation properties. They do not make learned forcing, residual O, the full GNN or ancestry probabilities contractive. Shared two-direction averaging is reversal-equivariant conditional on correspondingly reversed input fields/partitions; no new reversal theorem is claimed for the inherited H042/BiGRU body. Repeating nonlinear encoder/scan steps after splitting a hold can change its fields, so there is no whole-model hold-subdivision theorem.

## 4. Exact generic reduction and verified primary art

Let L_t[i,e]=1 when leaf i belongs to adjacent intersection-cell e=(A,B), and D_t=diag(L_t^T1). Then

    z_(t+1) = D_(t+1)^(-1) L_(t+1)^T L_t m_t.

The overlap between old cell(A,B) and new cell(B,C) is precisely n_ABC. This is the **already audited incidence lift/project kernel**, now on intersection cells instead of native clades. It is also an ordinary message-passing layer on the time-layered memory graph whose vertices are adjacent-group pairs and whose edges are supported three-group itineraries.

The forward probability transition on that graph is n_ABC/n_AB. Our value pull uses n_ABC/n_BC; it is the count-weighted adjoint/Bayes reverse, not the same normalizer. This distinction matters. Both are standard representations of the same weighted path-count graph. Attaching a learned forcing, readout or exponential hold update does not create a new generic primitive.

Verified primary comparisons:

1. **Rosvall et al. 2014, “Memory in network flows and its effects on spreading dynamics and community detection,” Nature Communications 5:4630, DOI 10.1038/ncomms5630.** Results/second-order Markov dynamics explicitly replace physical nodes with memory nodes representing directed links, then model second-order dynamics as first-order dynamics on that expanded network. The full primary HTML explains previous/current state dependence and higher-order extensions. This is direct state-expansion priority.
2. **Scholtes et al. 2014, “Causality-driven slow-down and speed-up of diffusion in non-Markovian temporal networks,” Nature Communications 5:5024, DOI 10.1038/ncomms6024.** Methods “Constructing higher-order time-aggregated networks” and “Higher-order Markov models” define weighted two-path networks and compare their transitions with a maximum-entropy Markov null preserving first-order edge weights. That directly precedes the matched triple-count versus adjacent-marginal-independent distinction below.
3. **Qarkaxhija, Perri and Scholtes 2022, “De Bruijn goes Neural: Causality-Aware Graph Neural Networks for Time Series Data on Dynamic Graphs,” arXiv 2209.08311v1.** Section 2 defines walk-indexed higher-order vertices and path-count edges. Section 3/Eq. 1 performs neural message passing on that graph; Eq. 2 projects higher-order representations back to original nodes through a bipartite aggregation. The paper explicitly permits other message-passing schemes. Our time-layered cut-intersection graph is not literally the same dataset, normalization or published GCN; its learn-on-memory-graph/project-back principle is established.

The inspected primary methods do not describe this exact 350-generation tree-cut construction and full-panel ancestry model. That is a bounded comparison, not a historical-priority certificate. A domain architecture contribution may still consist of a useful combination of established methods if the cut/path choice and measured benefit are substantive. It must not claim the first order-sensitive GNN, first anonymous-lineage transport, first non-Markovian neural graph propagation, first incidence kernel or new exponential recurrence.

Primary HTML/text, URLs, hashes and exact section locators are retained in this folder. Failed/irrelevant discovery responses are preserved and excluded from affirmative evidence.

## 5. The two necessary matched controls

**Adjacent-marginal independence control.** Keep identical actual cuts, n_AB, n_BC, pair-state addresses, cut features, flags, masses, weights, initialization, loss, optimizer and budget. Replace only triple count by

    n_tilde_ABC = n_AB*n_BC/n_B,
    alpha_null(A|B,C) = n_AB/n_B.

This preserves every adjacent overlap marginal and group size; its fractional expected counts need not be actual leaf intersections. Compute it implicitly by pooling old pair values by B, then broadcasting to outgoing(B,C) pairs. Do not materialize a dense A×C table. The comparison tests useful conditional dependence beyond adjacent pair overlaps, not usefulness of extra parameters or an unrelated topology corruption.

**Stable-leaf memory plus current-cut regather.** Use exactly the same first-sweep cut forcing, shared U/O/ell, two directions, cut masks, flag injection, second whole-tree sweep and body. Keep r=8 state per observed leaf:

    v_t(i)=rho_t*v_(t-1)(i)+(1-rho_t)*f_t(g_t(i)),
    r_t^leaf(B)=mean_(i in B) v_t(i).

Apply the same shared bidirectional residual and current-cut injection. It retains the full leaf-specific past rather than projecting into current/previous cut intersections. It can represent every pair-cell field and can retain more history. It has no trainable leaf-ID embedding; identity is only a checked correspondence index.

Allocate H slots per direction in both arms, masking unused pair slots, and report the actual active pair-cell count E_t versus H. Equal allocation and parameter count do not mean equal effective state independence or equal runtime. Compare measured end-to-end time/memory. Do not slow the simpler baseline artificially to conceal candidate cost.

All three arms would need identical starting backbone, new weights, independent reset optimizers, full-parent training schedule, checkpoint selection and common predictions-before-truth barrier. An H042-based sketch has 31,301parameters/arm. If root later grafts it onto H048, the chain, marker inputs and selected initialization must be identical in every arm and separately frozen; the present packet supplies no such integration and makes no H048 changes.

Interpretation: beating only the independence control but tying leaf memory supports sample-history retention/regrouping, not the particular order-2 cut constraint. Beating leaf memory but tying the independence null supports a different pooling/regularization effect, not triple-witness dependence. A claim for this particular mechanism needs both comparisons, adequate independent parents and useful per-cohort/span/boundary behavior. No current score supplies that evidence.

## 6. Efficient PyG implementation sketch and honest cost

Preparation is topology/date/identity-only. Derive each cut with parent-before-child traversal in O(N_t+H), assigning each eligible leaf its highest existing below-cut representative. No descendant bitset bank is necessary for partitions. Construct one local group ID per leaf and adjacent pair IDs from tuples(g_(t-1)(i),g_t(i)). At most H pair cells and H distinct supported triples occur at any boundary. Native IDs are used only to gather the current tree's representative features; they are never matched across trees.

For each step, a PyG MessagePassing(aggr="add",node_dim=0) can use one directed edge per actual leaf:

    old source node = pair_t(i)
    new target node = pair_(t+1)(i)
    edge weight = 1 / count_new_pair[pair_(t+1)(i)].

Repeated edges are intentional: their multiplicity is n_ABC, yielding the exact normalized transport without a three-way tensor or a dense all-node matrix. An equivalent scatter mean first gathers old pair values to leaves, then sums into new pair cells. This is the exact reduction above and a natural implementation path.

Read pair values to current-cut groups with count-weighted scatter sums. Leaf baseline needs only elementwise affine updates on H rows followed by the same group scatter mean. Both use the genuine existing whole-tree PyG sweeps; this sketch does not replace the backbone with a leaf-only predictor. Every temporal state remains in autograd. Preparation should be sealed independently before targets and reused unchanged for all controls.

Beyond the shared backbone, rank-r projection/transport work is O(sum_t |P_t|Dr + THr), plus O(sum_t |P_t|Dr) residual projection. Pair construction via sorting/torch.unique costs O(TH log H), or expected O(TH) with bounded hash tables. The recurrence is sequential over native holds. A generic parallel affine scan would compose sparse matrices that can densify; no cheap parallelization theorem is claimed. GPU kernel-launch/level overhead may dominate, as prior H046 cost cautions.

For H=952, T=1000, r=8, one FP32 state bank is 30,464,000bytes; two directions are 60,928,000bytes. A full forcing bank is at most another 30,464,000bytes; two retained cut-read banks at most 60,928,000bytes. Representative gathers, residual tensors, pair/group indexes, model/optimizer, full native backbone and autograd workspaces are additional. A single int32[1000,952] index bank is 3,808,000bytes; several are needed. Compact theoretical fields do not certify total RSS/VRAM or throughput. No payload was opened to estimate actual group counts, no timing was measured, and no resource admission is supplied.

## 7. Exact relationship to previous conditions

- **H019 / raw-overlap source no-go:** the generic normalized incidence kernel is the same family. This candidate changes the *state space* to two-cut intersection cells and poses a domain utility contrast; it does not reopen a generic mathematics claim.
- **H021:** executed/qualified cut diffusion is already the biological cut predecessor. Source defines query-only block diffusion E_g, sample-wise Jaccard retention J_b and numerator/mass impulse scans. It has no adjacent-intersection pair-state index and no n_ABC versus n_AB*n_BC/n_B contrast. Its late pilot was interrupted; no completed result answers this condition. Its persistent sample state and partition operators are close, and a broad “census cut plus recurrence” claim would be a repeat. Candidate includes all full-panel cut/singleton states and a direct first/second spatial-sweep injection, with no donor-kernel/covariance claim.
- **H025/H027:** whole joint sample/qualified-ancestral trace refinement and reference-only variants found no distinct genuine-trace benefit. They did not retain cut-pair intersection states or this matched independent-marginal null.
- **H029/H030:** source keys inherited carriers by qualified native/sample identities, with creation/parent/source event writes and GRU/delta-matrix memory. Completed controls did not establish an ancestral-history accuracy benefit. They do not implement sample-count triple coupling among anonymous cut cells. They strongly motivate the leaf-memory baseline and caution against interpreting any retained memory as useful.
- **H032:** static donor-support Gram uses second moments within a tree, not ordered triple group intersections. Its negative donor-dependence result is relevant caution, not the same intervention.
- **H033/H035 / commutator audit:** anchored tree-TV interleaving and two-tree signed spatial/alignment routes were tested without useful mechanism gains; chronology commutators have direct prior art. This candidate has no signed commutator, inverse/Kron solve or claimed low-rank heat update.
- **H040/H042/H043:** changed rooted-triplet witnesses (three leaves, two trees), local gained/lost flags, and nearest changed-clade partner exchange are not three-successive-tree path-count memory. H043's negative meaningful partner effect does not answer this dependence condition. Flags and complete-tree observations must remain identical in all arms.
- **H045/H046:** unchanged exact proper-clade episodes reset when membership changes and pool/integrate episode fields. Candidate pair cells can bridge changing cut membership and tests retention of one additional support label. The small negative foreign-versus-own effects do not establish that this representation works, and adding another weak context module without the two stronger controls would repeat their evidential weakness.
- **H047:** risk allocation is a different objective intervention. **H048:** marker-grid reset-chain persistence is a known class-chain utility experiment, currently running. Candidate changes latent whole-tree context rather than adding another copying or class HMM. It does not replace H048's within-hold emissions or resolve missing interior boundary support on its own.

No exact executed condition in the 48 decisions was found that compares this specified pair-cell triple transport with both the same adjacent-marginal null and equal-width leaf-memory/regather on the full current C4 body. That factual absence is not positive utility evidence or priority proof.

## 8. The still-untested factual condition

The prospective biological question is whether inferred dated-tree cut changes contain **useful three-hold sample-itinerary dependence**, beyond instantaneous complete-tree fields/local flags and ordinary persistent-leaf history, and whether averaging histories within recent cut-pair cells usefully denoises that signal rather than destroying it.

We have not even measured the eligible opportunity: cut sizes, root saturation, above-horizon masks, E_t/H, stable-cut runs, n_ABC deviation from the independent-marginal null, or nonzero residuals on actual observed data. If cuts are mostly unchanged/singletons or their triple coupling is conditionally independent, the mechanism has little or exactly no opportunity. Such metadata can be checked later under root's normal authorization without query labels; it is not a substitute for prediction evidence.

The biological rationale is continuity of reference-conditioned genealogy support through transient changes of inferred cut membership, not real ancestral identity or inferred recombination truth. Errors in dates/topology, population-structure misspecification and ancient-proxy drift may instead make those links harmful. The small/easy-tract development panel and repeatedly exposed eight VAL parents cannot establish a new population-history conclusion. This layer adds no observed SNP within a native hold, and does not by itself repair the known within-hold boundary-resolution limit.

Root may retain this as one conditional domain hypothesis after H048's result, or discard it for cost/biological opportunity. This packet supplies no new hypothesis ID, implementation, active source/configuration, admission or launch request. Generic priority rejection is affirmative; predictive utility remains unresolved.

## 9. Final event and marker-grid contract

**Cut events, not arbitrary native-record count.** The definitive pair-state index uses the two most recent *distinct cut partitions*. Consecutive native holds with the same full sampled-leaf partition form a maximal cut episode. Within that episode, keep the same pair layout and apply only the width-aware affine forcing update for each native hold. At a genuine cut-partition change, perform the triple-support transport once, using the previous two distinct cut partitions and the new one. No projection is repeated merely because an unchanged cut was serialized again. Current representative-node fields may still change between native trees inside an episode and are integrated as separate piecewise forcing. The first episode has one-group state; the first change creates pair states.

This supersedes reading t in the overlap formulas as “every serialized native hold”: in those formulas t indexes successive distinct cut episodes, while each episode's forcing is integrated across all its actual native holds. Native widths, graph-local features and the full marker-grid clock remain unchanged. For unchanged prepared fields, splitting a hold adds no cut event and the affine semigroup composes correctly. The upstream encoder can change when original SNP/span pooling changes, so this is not a whole-model subdivision theorem.

This event rule is necessary: rebuilding a pair as(B,B) on an unchanged-cut duplicate would average older within-B differences and make future predictions depend on artificial record count. The candidate explicitly avoids that defect. The memory-graph walk follows a chosen genomic ordering for computation; it is not a biological causal trajectory or an inferred recombination path. Event detection must compare actual sampled-leaf partitions modulo local group labels, not equal internal vertex IDs or arbitrary group numbers. A reappearance after a genuinely different cut remains a new event.

**Stronger leaf-equivalence qualification.** In leaf coordinates the pair update is Q_R applied after identity carry, where R is the intersection partition of the previous/current distinct cuts and Q_R is its ordinary block projection. Current-cut mean readout cancels the final projection because R refines the current cut. Since the first pair cut refines both initial forcing partitions, the true pair and leaf-memory readouts agree in real arithmetic through the first three distinct cut episodes for the stated linear forcing/recurrence. Differences can appear after older histories have been pooled and a later cut regroups them. Thus the three-episode witness distinguishes true versus marginal-independent transport, but it does not distinguish true pair memory from full leaf history. The prospective utility is precisely **historical pooling/denoising versus preserving the full leaf field**, not an information gain over leaf memory.

**Common H048 marker-grid wrapper for any prospective fit.** The H042-only body above defines the insertion location and parameter arithmetic. It must not be used to hide the already measured native-hold readout limitation. The prospective comparison uses all three context arms under the **same current H048 graph_to_transition marker-grid decoder**, with no comparison of unlike output grids:

1. Keep the complete H042 first/second PyG sweeps, leaf context and genomic BiGRU. Insert only the selected pair/null/leaf memory residual before the second sweep.
2. Feed the resulting same-shape complete-hold logits and width32 leaf contexts into the unchanged marker-grid decode_block_v1 logic. Reuse the same original grid_hold, disjoint-bin features/mask, pair_left/pair_right/edge_pair, positive delta_bp, marker head, graph-to-transition rate placement, destination simplex and exact chain scan.
3. All arms return raw normalized log marginals on the complete original[P,686,4]100bp-bin grid, with the same original exact-span target mass, callability/missingness, coverage/tail policy, query blocks and prediction-before-truth barrier. Marker observations and readout grid are identical. No new marker windows, truth-derived breakpoints or posthoc smoothing are introduced.
4. The common 31,933-parameter marker-grid model plus 513 memory scalars gives 32,446parameters per arm. New U/O/ell and every inherited marker/conditioner/chain/body parameter are matched. Initial O=0 makes all three models identical before training. Use the same H042 initialization and fresh common marker-grid-head initialization if this design is ever registered; no H048-trained checkpoint is needed by the source sketch. Root can revise the common decoder choice before any new freeze, but must apply it equally and record that choice.
5. Implement this in a new owned module/wrapper with explicit cut-plan input. The current MarkerGridResetBundleV1 and backbone_state_v1 signatures do not carry that plan; this is not a drop-in launch command and requires a separately reviewed observation-only bundle and a narrow new encoder wrapper. Existing H048 source stays frozen.

This tests *which latent history pooling supplies useful current graph context*, with decoder placement fixed across arms. Current H048 tests *whether the same graph conditioner belongs in transition rate versus local confidence*, with no cut-memory module. These are distinct scientific interventions. No extra H048 transition-versus-confidence factorial sweep is proposed.

The marker channel and chain can still call switches strictly inside a native hold. The memory layer supplies no new within-hold SNP or tree observation, and does not by itself create that resolution. All incremental cost statements above exclude the already substantial common marker-grid feature/chain work. Actual full-body FP32 work, memory, gradients and common-grid joins require future source qualification and root admission; none occurred here.
