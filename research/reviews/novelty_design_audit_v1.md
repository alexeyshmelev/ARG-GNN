# Novelty and research-design audit v1

Date: 2026-10-10. Scope: implementation and mechanism audit, with fresh context. This is not a manuscript peer review or an acceptance prediction. I read the remote AGENTS.md before inspecting the repository. I did not read paper/methods_working.tex, run training, score final tests, install packages, or change implementation files.

## Recommendation

The current implementations support useful, testable representation contracts. They do not yet establish a sufficiently distinct architectural contribution to justify an AAAI-level novelty claim. Gated local/context interleaving is already central to the supplied Graph Set Transformer (GST). Interval integration, normalized continuous attention, exact zero-order-hold evolution, identity alignment, dissipative generators, and input-dependent state transitions also have direct prior art.

The narrowest current differences are (i) fixed-coordinate, node-level structural context from independently encoded source graphs without GST's single graph-token bottleneck, and (ii) graph-conditioned mixing of latent state coordinates in an exact affine hold system. Neither difference alone establishes broad novelty.

The additional persistent-interface Kron-flow proposal is the more defensible next mechanism. It couples actual query-sample memories through changing observed genealogy geometry, with reference samples as clamped observed-source forcing, and can enforce both interval and serial-unary subdivision contracts. Prototype those contracts on small CPU fixtures, then run the common-invariant-input pilot defined below against no-memory, bag, independent-leaf and Kron+standard-sequence controls. Its established mathematics must remain attributed, and empirical merit remains unknown. The forcing-free switching probe below remains a diagnostic for the current channel-mixing hypothesis. This audit asserts no performance gain.

## Evidence inspected

Directly inspected implementations:

- src/arggnn/models.py, particularly log_interval_kernel_mass at line 107, LineageCrossAttention at line 254, and IntervalLineageGNN at line 311.
- src/arggnn/reference_encoder.py, ReferenceFeatureAdapter at line 44.
- src/arggnn/semigroup_model.py, DissipativeGenerator at line 61 and SemigroupGraphModel at line 115.
- Supporting graphssm_baseline.py, study_models.py, and chickenpox.py.

Directly inspected supplied prior-art audits:

- docs/dissipative_transport_audit.md.
- docs/lineage_kernel_operator_audit.md.

Directly inspected cached public sources:

- Escrig Molina, Chen and Probst, [Graph Set Transformer, arXiv:2606.05116v1](https://arxiv.org/abs/2606.05116v1), especially Section 3.1 and Eqs. 1–12, from .runtime/cache/gst-baseline/paper.txt and its source manifest.
- Li et al., [GraphSSM, arXiv:2406.00943v2](https://arxiv.org/abs/2406.00943v2), especially Theorem 1, Eqs. 5–8, Appendix B.2, and its stated piecewise-dynamics limitation.
- Calvello, Kovachki, Levine and Stuart, [Continuum Attention for Neural Operators, arXiv:2406.06486v4](https://arxiv.org/abs/2406.06486v4), Definitions 4–5 and 10–11, and the discrete-to-continuum approximation statements.
- [Stable Port-Hamiltonian Neural Networks, arXiv:2502.02480v1](https://arxiv.org/abs/2502.02480v1), Eq. 3 and its positive-semidefinite dissipation parameterization.
- [Liquid Structural State-Space Models, arXiv:2209.12951](https://arxiv.org/abs/2209.12951), its input-dependent state-transition description.

The two supplied audits provide a wider equation-level crosswalk to linear attention, retention, graph kernel attention, antisymmetric recurrence, and controlled differential equations. I agree with their central caution. This is a targeted audit of the closest mechanisms, not a claim that the entire literature has been exhausted.

## What is implemented and what is distinct

### IntervalLineageGNN

For a fixed query coordinate x, source graph g occupying I_g and source node i in that graph, the implemented attention is proportional to

\[
\frac{m_g(x)}{n_g}
\exp\!\left(
q(s)^\top k(h_{gi})/\sqrt H
+\alpha\,{\bf1}[{\rm id}_q={\rm id}_{gi}\ge0]
+\beta\,{\rm Jaccard}(c_q,c_{gi})
\right),
\qquad
m_g(x)=\int_{I_g}e^{-|x-u|/\tau}\,du.
\]

Here n_g is the number of source nodes, h comes from a graph-local source bank, and the correspondence biases are included only in the correspondence variant. A sigmoid gate and LayerNorm fuse the weighted source message into the target state. In feedback mode, a target-tree MPNN processes that contextual state at the next depth. The source banks remain independently encoded at every depth.

GST already interleaves local graph propagation, set attention, and gated write-back at every layer. GST Eqs. 3–7 first reduce each source graph to a token and contextualize graph tokens. GST Eq. 9 allows node queries into the post-transformer graph tokens. The current model queries all source nodes instead, with physical interval mass and observed correspondence biases. This is a real architectural difference from GST. It is not evidence that gated feedback or removal of late-only context is newly invented.

The independent source-bank choice also means the implementation is not a fully joint GST update over all graphs. It is query-conditioned target processing against local source banks. That restriction makes its refinement contract easier to maintain and may be useful, but no general expressivity superiority follows automatically. Pooling can be sufficient on particular tasks, and direct node attention can cost substantially more.

The exact interval weight is a correct closed-form integration of this chosen piecewise-constant source model. Continuum attention already defines normalized attention through integrals/expectations over function domains. Exact subdivision consistency is the additivity of those integrals when the graph-local features and query coordinate stay fixed. It is a valuable implementation contract, not a new continuum-attention theory.

The current content exponential plus Jaccard bias is not the positive finite-feature operator proposed in lineage_kernel_operator_audit.md. Replacing it with a clade-incidence dot-product feature map would change the kernel. It would not exactly linearize the existing softmax/Jaccard calculation. Positive kernel features, normalized key/value moments, and exponential retention are established in the sources crosswalked by that audit.

With P query coordinates, T_p target-tree nodes, N total source nodes, depth L and hidden width H, the attention work contains roughly O(L sum_p T_p N H), plus clade-overlap work proportional to its incidence dimension. The method does not have linear total-node complexity in its current form.

Important interpretation limits:

- A permutation of packed graph records with intervals and target mapping permuted together is serialization invariance. Moving graph signals to different physical intervals is a different input. The latter can change this distance-weighted operator.
- The absolute-distance kernel is reflection symmetric. It supplies contextual arrangement, not a causal chronology.
- The node-count normalization chooses a uniform within-graph source measure. It limits changes in graph mass as internal-node count varies, but does not make the model invariant to adding or duplicating internal nodes.
- Query leaves outside the reference-landmark set can have zero landmark-clade vectors. Their direct Jaccard route is then absent. The all-node target attention still has an indirect route through target ancestors and target-tree message passing. Do not describe the zero query-leaf clade as directly selecting its ancestors.
- The equal_tree_weights implementation still has a midpoint-distance factor. It removes interval-span mass, rather than removing coordinate weighting entirely.

### ReferenceFeatureAdapter

This is a fixed, zero-learnable-parameter domain control. It appends all-node summaries of observed reference categories and observed MRCA times, including nearest-reference votes and fixed-scale kernel evidence. The upward/downward tree passes already provide information from references beyond a shallow local MPNN's receptive field.

Sharing the identical prepared tensor across every model is the right attribution control. A gain from these informative summaries cannot be credited to IntervalLineageGNN or SemigroupGraphModel. Feature-only MLP, independent GNN, and simple leaf-kernel smoothing controls are therefore especially important. An optional raw-feature sensitivity analysis can measure the adapter's contribution, but cannot replace the common-input primary comparison.

### SemigroupGraphModel

For every persistent query leaf separately, the model produces graph-conditioned A_g and b_g from a local query-leaf embedding:

\[
A_g=\rho\left[-D_g-U_gU_g^\top+V_gW_g^\top-W_gV_g^\top\right],
\quad
D_g=\operatorname{diag}(\operatorname{softplus}(d_g)+\epsilon),
\quad
b_g=\rho\tanh(f_g),
\]

with rho>0 and epsilon>0. It evolves the affine augmented state by exp(delta M_g), where

\[
M_g=\begin{pmatrix}A_g&b_g\\0&0\end{pmatrix}.
\]

The implementation sorts by physical coordinates, uses zero boundary states, evaluates partial holds at fixed query coordinates, and scans backward with positive durations. Those choices are coherent with its stated refinement contract.

There is no temporal exchange between different leaves' memory reservoirs. Topology affects a leaf's generator through its local GNN embedding. The off-diagonal entries mix latent channels belonging to that same leaf. Calling this inter-graph ancestry transport should not imply that ancestor states or memory from another sample are carried across trees.

GraphSSM Eq. 5 already has a full fixed HiPPO matrix, graph-dependent forcing, and piecewise LTI evolution. Its Eq. 8 gives ZOH evolution, and Appendix B.2 aligns changing node memories. The implemented candidate's precise distinction is generating the homogeneous state matrix from graph content and evaluating an exact continuous readout at a supplied coordinate. The project's GraphSSM comparator is a disclosed adaptation, with snapshot-end readout and a default approximate injection factor. Winning against that comparator alone cannot show that its known discretization/readout choices are a general limitation of GraphSSM.

The J-R form is an established quadratic-energy port-Hamiltonian construction. With H(s)=||s||^2/2, -D-UU^T is dissipation and VW^T-WV^T is the conservative skew component. Input-dependent transition dynamics also predate this model. The candidate is a structured graph-conditioned controlled ODE, so full-matrix mixing does not establish superiority in expressivity over generic controlled-ODE/CDE models.

The implemented positive minimum damping strengthens the contraction statement relative to the unimplemented proposal in the audit. For two states under the same graph signal and forcing,

\[
\frac{d}{dt}\|e\|_2^2
\le -2\rho\epsilon\|e\|_2^2.
\]

This is a homogeneous state-perturbation bound. It is not a claim that the forced state norm never grows or that the entire encoder/readout network is globally nonexpansive. The augmented affine matrix has a constant-coordinate zero eigenvalue.

Diagonal affine transport already distinguishes order through forcing. Indeed,

\[
[M_1,M_2]_{\rm affine}
=
\begin{pmatrix}
[A_1,A_2]&A_1b_2-A_2b_1\\0&0
\end{pmatrix}.
\]

Even if [A_1,A_2]=0, the upper-right term can be nonzero. Consequently, order-shuffle degradation cannot identify homogeneous noncommutativity. Neither can ordered-versus-span_set alone: that intervention changes the complete affine flow.

The symmetric bidirectional readout uses f+b and (f-b)^2. With reflection of the supplied intervals/query, the two scans exchange roles and the output is unchanged. The model thus distinguishes some arrangements modulo this reversal symmetry. This is a reasonable chromosome prior, but should not be presented as unconstrained chronological order modeling. For external forecasting, directional readout or a forward-only scan is an important declared control.

The span_set variant is order blind in its context. When paired with symmetric bidirectional readout its squared-difference feature is identically zero, and the implementation correctly reports the inactive head parameters. homogeneous_set is a stronger comparison for ordered transport because it retains left/right query-to-boundary distances while replacing the changing generator by its interval-weighted average.

Dense matrix exponentials still cost roughly O(G Q d^3 + P Q d^3) for G holds, Q leaves and state width d, with saved transition memory and backward cost to measure. Low-rank parameterization of A does not make exp(A) low rank. There is no implemented parallel-scan kernel.

## Additional structural candidate: persistent-interface Kron flow

This candidate is more substantively connected to ARG graph collections than within-leaf channel mixing. It moves the time-dependent homogeneous generator into persistent sample space, where it can exchange historical information between samples according to each observed genealogy. It also supplies a second, useful representation contract: suppressing or inserting serial unary vertices should not alter the boundary response.

It is still a proposed mechanism. Kron reduction, electrical boundary-response operators, series-resistance equivalence, graph coarsening and diffusion are established. I verified the public metadata/abstract of Dörfler and Bullo's [Kron Reduction of Graphs with Applications to Electrical Networks](https://arxiv.org/abs/1102.2950) / [DOI](https://doi.org/10.1109/TCSI.2012.2215780). The verified abstract of Bianchi, Grattarola, Livi and Alippi's [Node Decimation Pooling](https://arxiv.org/abs/1910.11436) / [DOI](https://doi.org/10.1109/TNNLS.2020.3044146) explicitly uses Kron reduction to connect retained nodes and sparsifies the resulting dense graph for GNN computation. These additional abstract/metadata reads were made on 2026-10-10 without caching new files. The full NDP implementation was not audited.

The plausible contribution would be a correspondence-preserving boundary-memory architecture for a collection of changing graph realizations, with both interval and serial-edge subdivision contracts and task evidence. It would not be a newly discovered Kron equivalence or a theorem that reduction preserves arbitrary tree dynamics. Relative to the sources inspected, the fixed sample interface carried through a sequence is a concrete distinction from selecting pooling nodes independently for each graph. A broad first-of-kind claim still requires a dedicated search for temporal Kron models, boundary-response neural operators, and persistent-interface graph dynamics.

### Precise minimal formulation

Let S be a fixed persistent interface of all observed sample leaves, including reference-panel leaves where available. Queries can address a subset of S. Match identities as keys only. If the interface contains targets alone, eliminating reference leaves removes their explicit memory carriers. That is a separate, weaker design choice to disclose and ablate.

For a connected observed tree g with positive edge resistances r_e=kappa times observed branch length, define its undirected weighted Laplacian

\[
L_g=B_g\operatorname{diag}(r_e^{-1})B_g^\top.
\]

Partition vertices into S and anonymous interior vertices I_g. If every component has an interface vertex, the interior principal matrix is positive definite, and

\[
C_g=(L_g)_{SS}
-(L_g)_{SI}(L_g)_{II}^{-1}(L_g)_{IS}
\]

exists. Use a linear solve, not an explicit matrix inverse. C_g is symmetric positive semidefinite, has zero row sum, and has nonpositive off-diagonal entries in exact arithmetic. Its Dirichlet characterization is

\[
v^\top C_gv
=\min_{u\in\mathbb R^{|I_g|}}
\begin{pmatrix}v\\u\end{pmatrix}^{\!\top}
L_g
\begin{pmatrix}v\\u\end{pmatrix}.
\]

Build invariant leaf features X_g from observed reference roles, leaf ages, and observed TMRCA summaries. A source encoder uses only X_g and C_g, producing F_g in R^{|S| by d}. A minimal coupled memory evolves as

\[
\dot Z=-K_gZ+F_g,\qquad
K_g=\epsilon I+\beta_g C_g,\quad
\epsilon>0,\ \beta_g\ge0,
\]

where beta_g is fixed or generated from a permutation-invariant summary of the same invariant inputs. Its exact hold flow is

\[
Z(t+\delta)=e^{-\delta K_g}Z(t)
+K_g^{-1}(I-e^{-\delta K_g})F_g.
\]

Use a stable eigenvalue/expm1 evaluation of the forcing factor or an augmented affine exponential. Read the target sample at the fixed query coordinate, with partial-hold evolution and a disclosed forward/reverse policy. This state couples different persistent samples. It differs materially from the currently implemented independent-leaf reservoirs.

For identical graph signals, ||Z_1-Z_2||_F contracts at least as exp(-epsilon t). Distinct C_g need not commute, even though each is symmetric. The constant leaf mode is annihilated by C_g, so coupling alone does not dissipate that mode. The explicit epsilon controls it. Diffusion can erase useful ancestry differences between nearby samples, so usefulness and oversmoothing must be measured rather than presumed.

### Preferred feasible prototype: query memory with clamped reference sources

The full-interface formulation above clarifies the geometry. A more feasible first prototype first defines the full-leaf response C_g, then uses an exhaustive, fixed partition S=Q union R into query samples and observed reference samples:

\[
C_g=\begin{pmatrix}
C_{QQ,g}&C_{QR,g}\\
C_{RQ,g}&C_{RR,g}
\end{pmatrix}.
\]

Store dynamic memory only for Q. Let U_{R,g} be graph-local reference latent features derived from observed reference roles, ages and invariant summaries, and let F_{Q,g} be graph-local query forcing. Both are fixed over the hold. For latent channel h define

\[
\dot z_h
=-(\lambda_{g,h}C_{QQ,g}+d_{g,h}I)z_h+b_{g,h},
\qquad
b_{g,h}=F_{Q,g,h}-\lambda_{g,h}C_{QR,g}U_{R,g,h},
\]

with nonnegative scalar lambda and d>=epsilon>0. Scalars may be conditioned on invariant graph content, but must be shared across query nodes within a channel. Reference roles are observed inputs, never query ancestry truth. This is a clamped-source boundary model. It is different from eliminating the references as freely equilibrating vertices, which would take another Schur complement and define a different operator.

C_QQ is a principal positive-semidefinite submatrix. In a connected graph with at least one clamped reference it is positive definite. If C_QQ=V diag(mu_k)V^T, define r_{g,h,k}=lambda_{g,h}mu_k+d_{g,h}. Exact state and constant-force action is

\[
z_h(t+\delta)
=
V\operatorname{diag}(e^{-\delta r_{g,h,k}})V^\top z_h(t)
+
V\operatorname{diag}\!\left(
\frac{-\operatorname{expm1}(-\delta r_{g,h,k})}{r_{g,h,k}}
\right)V^\top b_{g,h}.
\]

The eigenvectors/eigenvalues of the observed C_QQ can be cached per graph because the learned channel scalars do not change them. Full and partial hold readouts then cost O(|Q|^2 H) online, without forming a full exponential or augmenting a |Q|H-dimensional system. The O(|Q|^3) eigendecomposition is preprocessing when conductances are fixed observed quantities. Learned node-specific damping/rates or learned changes to individual conductances generally invalidate this cached diagonalization. A small stable quotient/eigenvalue numerical routine is sufficient here, and this spectral action is established linear algebra.

This improves transport feasibility at query cohorts of 8, 16 and 64, but does not erase other costs. Materializing C_QR costs O(|Q||R|) per graph, and multiplying it by reference features costs O(|Q||R|H). A source GNN on the full reduced graph can still have O(|S|^2 H) cost if implemented densely.

Full C need not be materialized merely to obtain the required blocks. C_QQ needs interior solves for query basis columns. The reference forcing can be applied directly as
\[
C_{QR}U_R
=
L_{QR}U_R
-L_{QI}\operatorname{solve}(L_{II},L_{IR}U_R),
\]
using a cached factorization of the sparse tree interior block. A tree solve can be linear in tree size per right-hand side. This is a feasible mathematical route, not a claim of an implemented scalable kernel.

Use a particularly strong independent-query control: retain the exact same source encoder and b_g, but replace C_QQ in the dynamic term by diag(C_QQ). It preserves reference geometry/forcing and each query's diagonal decay while removing exchange of temporal memory between queries. That comparison directly addresses the proposed cross-query mechanism. The baseline still may be order sensitive through forcing.

For a mean-operator bag, if graph-conditioned rates are used, averaging C alone does not average the actual generator. Define the span-weighted mean of lambda_g,h C_QQ,g+d_g,h I and, for the fully order-blind affine control, the corresponding mean b_g,h. Evaluate it at the same query-to-boundary distances. Also report a mean-C/changing-forcing control, but do not call the latter order blind.

Keep the complete Q reservoir fixed when sampling readout leaves or query coordinates for training. Moving unscored target leaves into R, removing them, or changing Q for every minibatch changes the modeled input and its coupled predictions. Varying cohort size should be an explicitly designed experiment, with primary cohorts/subjects split before windowing. A larger 64-query cohort is needed to test meaningful cross-query ancestry sharing rather than reporting a coupling advantage solely on tiny groups. Report target heterogeneity and relatedness as well as overall error.

The full-tree counterexample also applies in this clamped-source formulation. In the three-node example above, take the first leaf as Q, clamp the second leaf in R to zero, and initialize the interior harmonically at 1/3. Full-tree query diffusion has initial derivative -2/3 and second derivative 2/3, whereas reduced query dynamics has second derivative 4/9. The equality of initial boundary response therefore still does not imply equal trajectories.

### What the two invariances require

Replacing an edge of resistance r by a serial chain with positive resistances summing to r leaves the minimized boundary energy unchanged. Eliminating the degree-two nodes therefore leaves C_g unchanged. With X_g, beta_g and F_g unchanged, the complete model is invariant to this graph representation change. Likewise an identical graph interval split into subintervals preserves the exact affine flow at each unchanged query coordinate.

Both statements are exact in real arithmetic under a fixed-interface, piecewise-constant-input contract. They require all of the following:

- The interface set, sample features, queries, and source measure stay unchanged. Inserted unary nodes are anonymous interior vertices, not newly observed samples.
- Resistances are additive under subdivision. A globally fixed multiplier of physical branch length satisfies this. A difference of a fixed monotone age potential also telescopes but defines a different genealogy metric. Applying log1p to each separate edge length does not preserve additivity.
- Source encoding uses the reduced response and invariant leaf features exclusively. An original-tree MPNN can change its receptive field when a unary node is inserted. Raw log_parent_branch at a leaf changes when its parent edge is subdivided. Those routes invalidate the full-model property.
- Damping is applied after reduction, on the same persistent interface. Grounding or damping every original interior vertex before reduction introduces representation-dependent leakage.
- No fresh stochastic mask, nonlinear boundary jump, or node-count normalization is introduced merely because a graph interval or edge was subdivided.
- A reduced-graph approximation/sparsifier, if used, preserves the intended equivalence under the same inputs and its numerical error is reported. Mathematical exactness should not be claimed from approximate exponential actions.

The existing feature metadata includes log_parent_branch. Drop it from every model in the primary invariant-feature comparison, after a documented common preparation stage. The fixed reference summaries evaluated at leaves preserve MRCA relations under insertion of a non-sample unary vertex at an intermediate age. Do not allow the proposed model alone to receive a selectively favorable feature set. A separate raw-feature sensitivity comparison can retain that parent-branch field with its property limits disclosed.

### Optional learned age-resistance density (not implemented)

A shared monotone age warp is a coherent extension after the fixed-length prototype. For observed parent/child ages t_p>t_c, define

\[
R_{pc}=\phi_\theta(t_p)-\phi_\theta(t_c)
=\int_{t_c}^{t_p}\rho_\theta(s)\,ds,
\qquad \rho_\theta(s)>0.
\]

For example, on nonnegative ages with a fixed declared scale T,
\[
\phi_\theta(t)=a_1t+\sum_{k=2}^K a_k T(t/T)^k,
\qquad a_1\ge\epsilon_\rho>0,\quad a_k\ge0,
\]
is monotone. A monotone spline or a positive-density neural integral can also be used. Coefficients may use softplus constraints, with the explicit lower bound on the linear density term. The shared warp learns which age ranges carry greater resistance while preserving the serial representation contract:
\[
R_{pu}+R_{uc}
=\phi_\theta(t_p)-\phi_\theta(t_c).
\]

This is scientifically better connected to observed genealogical time than arbitrary latent-coordinate mixing. It remains learned geometry, not a biological law or an established novelty result. Weighted tree/electrical metrics and monotone transformations are established ingredients. The verified public abstract of [Unconstrained Monotonic Neural Networks, arXiv:1908.05164](https://arxiv.org/abs/1908.05164) explicitly constructs monotone neural transformations from strictly positive derivatives. That read was abstract/metadata only on 2026-10-10. A limited metadata search did not establish an exact matching age-warp/Kron-memory construction, but this does not support a first-of-kind claim.

The exact contract requires a single phi shared over all relevant edge endpoints and unary insertions. Evaluate phi once at node ages, then take differences. Independent per-edge numerical quadrature can produce different approximate resistances after an edge is split. A graph-conditioned warp would also need its conditioning to be invariant to inserted unary vertices and fixed within the hold. A node-specific or arbitrary edge-specific warp need not telescope. Equal endpoint ages still give zero resistance, so the warp does not resolve the zero-length-branch case.

Fix the warp's scale, for example by a declared anchor normalization phi(t_anchor)-phi(0)=t_anchor or an equivalent coefficient normalization. Otherwise multiplying all resistances by a>0 scales C by 1/a, which can be exactly offset by multiplying the coupling lambda by a. That nonidentifiability makes the learned age function difficult to interpret. Declare age units, the anchor, the training range, and the extrapolation rule for older samples. Positive polynomial coefficients ensure monotonicity outside the training range but can produce rapidly growing, poorly conditioned resistances.

During training C_theta and its eigenbasis depend on theta. A cached fixed eigenbasis no longer supports the earlier training-time cost claim. The Schur complement must be recomputed/differentiated, and gradients can be ill conditioned near short edges. One useful exact differential is
\[
dC_\theta
=H_\theta^\top(dL_\theta)H_\theta,
\qquad
H_\theta=\begin{pmatrix}I\\-L_{II,\theta}^{-1}L_{IS,\theta}\end{pmatrix},
\]
under the same fixed-interface, invertible-interior assumptions. It follows from the minimized Dirichlet energy and avoids an incorrect claim that the harmonic extension is independent of theta.

A small-query generic matrix exponential is a defensible initial differentiable route. Stable matrix-function/Fréchet derivatives can also be used. Differentiating individual eigenvectors is delicate at repeated or near-repeated eigenvalues even though the matrix exponential itself is smooth. After training freezes phi, inference may again cache the resulting eigensystems. Do not report that inference cache as a training acceleration.

If this extension is investigated, give the same learned-geometry family and fitting budget to Kron local-only, Kron+GRU/GraphSSM, and the interface-memory candidate. Include fixed linear phi, a low-dimensional monotone phi, and an unconstrained edge-weight learner as separate controls. The last can test whether subdivision consistency regularizes the task usefully, but does not inherit the exact contract. Keep the fixed-length resistance model as the first prototype and reserve this extension for evidence that learned age geometry, rather than generic extra parameters, adds value.

### Counterexamples and restrictions

1. **Per-edge regularization can break serial invariance.** If r_e=length_e+r_0, an edge of length 1 has resistance 1+r_0, while two half-length edges have total resistance 1+2r_0. Their boundary response differs. max(length_e,r_min) and an additive per-edge conductance floor have analogous failures. Exact additivity cannot be asserted after arbitrary edgewise clipping.

2. **Zero-length branches need a contract.** The 1/r formula requires positive resistance. Zero-resistance interior edges may be contracted without changing the interface when that contraction does not identify distinct interface samples. A zero-resistance path between distinct persistent samples imposes a voltage-equality constraint and does not yield an ordinary finite C on unconstrained separate sample coordinates. Such inputs need rejection, a stated constrained treatment, or a stated regularization that weakens exact invariance. Very short positive edges also require condition-number and float64 checks. The existing monotone-time tree contract is not automatically a strictly positive-branch contract.

3. **Kron reduction preserves boundary response, not arbitrary full-tree diffusion.** Consider two interface leaves and one interior vertex, with conductances 1 and 2:
   \[
   L=\begin{pmatrix}1&0&-1\\0&2&-2\\-1&-2&3\end{pmatrix},
   \qquad
   C=\frac23\begin{pmatrix}1&-1\\-1&1\end{pmatrix}.
   \]
   Set the leaves to (1,0) and initialize the interior harmonically at 1/3. Full-tree diffusion dot z=-Lz and reduced diffusion dot v=-Cv have the same initial leaf derivative (-2/3,2/3). Their second leaf derivatives are respectively (2/3,-4/3) and (8/9,-8/9), so their trajectories immediately diverge. This counterexample persists after adding common uniform damping. Kron elimination assumes an algebraic, instantaneously equilibrated interior. In a DAE, L_IS v+L_II u=0 gives the reduced dynamics, but ordinary capacitive/dynamic interior states do not satisfy that constraint at every time. No strong-lumpability claim is justified.

4. **Reduction can remove information intentionally.** Unary-vertex placement is quotiented out. Non-interface pendant branches have no boundary response when they terminate without a boundary condition. With only one interface vertex, C=0 and all ungrounded genealogy geometry disappears. Fixing the full observed sample interface is therefore important. For a positive weighted tree, interface effective resistances encode its leaf-to-leaf path-length metric, which is a meaningful ancestry geometry, but not a replacement for every observed ancestral attribute.

5. **This is an inductive prior, not a biological dynamics law.** Genomic coordinate is not elapsed population time. The memory equation propagates evidence across a chromosome and between related samples. It should not be described as physical movement of ancestry along electrical wires. Closely related samples can still have different local ancestry, and diffusion can oversmooth the very boundary/heterogeneity signal the task needs.

### Feasibility and the decisive pilot

A reduced response on all tree leaves is often dense, even when the original tree is sparse. The method is feasible first at a modest interface size, with exact double-precision CPU property fixtures and carefully measured training cost. Prefer the cached query/reference spectral action defined above. For the unpartitioned dense version, materializing C costs O(|S|^2) storage per distinct tree, and a straightforward dense eigendecomposition/exponential costs O(|S|^3) per tree. Reuse the same eigensystem for full and partial holds within an interval. Shared generator across value channels permits O(|S|^2 d) transition application after that factorization, rather than exponentiating a |S|d by |S|d matrix.

One can apply C to vectors through a sparse interior solve:
\[
Cv=L_{SS}v-L_{SI}\operatorname{solve}(L_{II},L_{IS}v).
\]
Tree-specific elimination can avoid an explicit dense matrix for linear message operations. Nonlinear attention or arbitrary edge-conditioned messages on the reduced complete graph can still require quadratic work. Krylov/exponential-action methods can improve feasibility, but they introduce an approximation whose residual and refinement errors must be measured. NDP's sparsification prior art is relevant precisely because the reduced graph becomes dense.

The decisive experiment should use the same invariant leaf features and persistent interface in each comparison:

- Kron local GNN with no temporal memory: tests whether the reduced representation alone explains any gain.
- Exact independent-leaf affine flow: tests whether temporal cross-sample memory exchange is needed.
- Mean-reduced-operator bag: replace changing C_g by its interval-weighted mean. In the strongest control also use corresponding mean forcing while retaining the same boundary/query-distance readout. An additional changing-forcing/mean-C control separates forcing order from evolving spatial coupling.
- Kron local GNN plus GRU or GraphSSM: tests whether the useful reduced geometry requires the proposed exact flow. Keep the same invariant features/reduction.
- Ordered interface flow: the candidate above.
- Full-tree exact diffusion projected to leaves, with clearly declared interior initial/boundary and reset policies: tests an alternative use of tree geometry. Treat it as a distinct dynamical model, not an expected mathematical reproduction of Kron flow.

Use both fixed-capacity and active-parameter/compute comparisons. Separately test exact interval splitting, serial unary splitting at random legal intermediate ages, sample relabeling, and ordinary storage permutation. Include intentionally failing original-tree/raw-parent-branch and resistance-floor controls to show that the tests detect the asserted representation changes.

The property tests should report C error and prediction error separately. For unary insertion, recompute all invariant features from the modified observed tree and verify equality rather than simply copying features. Check the Dirichlet-energy identity, off-diagonal signs, row sums, eigenvalue tolerances, finite gradients, and the projected-diffusion counterexample on small CPU fixtures.

A useful domain result would be a reproducible training/validation improvement of ordered interface memory over the no-memory, independent-leaf, bag, and Kron+standard-sequence controls, with ancestry boundary/heterogeneity performance and memory/runtime reported. That would support the specific boundary-memory hypothesis. Passing the two algebraic invariance checks alone does not establish useful learning or novelty. This is the more defensible next mechanism to prototype before investing further in arbitrary latent-channel noncommutativity.

## Current semigroup diagnostic: transport identifiability

Run a predeclared training/validation-only pilot that asks whether graph-conditioned homogeneous mixing adds value beyond continuous forcing and smoothing. Keep final test sets untouched. Use identical prepared reference features, physical query coordinates, boundaries, source encoder depth, and leaf matching in the primary comparisons. This is a decision experiment, not a new novelty claim.

### Exact switching probe

First include a small controlled probe in which the only informative change is the order of two graph-conditioned homogeneous generators. Keep the query in a common final sentinel graph, so the local readout cannot reveal which graph was last. Remove forcing after a common initialization, disable reverse/symmetric readout, and avoid nonlinear per-record jumps.

In an instrumented transport fixture set s_0=e_1 in dimension at least three. This is equivalent to a common zero-initialized prefix that injects a multiple of e_1. Let

\[
J_{12}=\begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix},
\qquad
J_{23}=\begin{pmatrix}0&0&0\\0&0&-1\\0&1&0\end{pmatrix},
\]

\[
A_A=-\epsilon I+\omega J_{12},\quad
A_B=-\epsilon I+\omega J_{23},\quad
A_C=-\epsilon I,
\qquad b_A=b_B=b_C=0.
\]

Compare A,B,C against B,A,C, with equal A/B hold widths delta and the same fixed query a distance c inside C. For readout e_3^T s, the analytically defined paired difference is

\[
\Delta
=e^{-\epsilon(2\delta+c)}\sin^2(\omega\delta).
\]

The graph multiset, interval mass, query graph, query coordinate and boundary condition are the same. Choose delta and omega in advance so this expression is nonzero. The labels are teacher outputs for a diagnostic, not ancestry truth features.

For any diagonal or simultaneously diagonalizable homogeneous A_A,A_B, the two middle flows commute, so their paired prediction difference must be exactly zero in real arithmetic under this probe's assumptions. The best identical prediction for a pair with targets 0 and Delta has pair-mean squared error Delta^2/4. This supplies a meaningful negative control. A full model that also gives zero contrast fails to use the proposed mechanism. At small widths the response is proportional to delta^2[A_B,A_A]s_0.

A GRU or generic full controlled ODE can solve this probe as well. Passing it confirms a specific capability absent from the restricted commuting controls. It does not demonstrate new order expressivity or an AAAI contribution. The initialization/forcing mask is a disclosed diagnostic intervention, not an assertion that current production forward implements that probe unchanged.

### Domain pilot and ablations

Then evaluate the following models under the same ARG common-input protocol:

1. Feature-only MLP, independent GNN and exact leaf-kernel smoothing.
2. Fixed-A exact affine ZOH transport with learned graph forcing.
3. Input-conditioned diagonal exact affine ZOH.
4. A commuting-basis control with a single learned orthogonal Q shared across all graph holds. A useful even-dimensional form is
   \[
   A_g=Q\,\operatorname{blockdiag}(-d_{gk}I_2+\omega_{gk}J_2)_k\,Q^\top,
   \quad d_{gk}\ge\epsilon,
   \quad J_2=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
   \]
   All homogeneous A_g commute, while graph-dependent damping, stable rotation rates, forcing and a globally mixed latent basis remain available.
5. Symmetric-only full dissipative transport, with the skew term removed. Different symmetric matrices can still fail to commute, so this tests whether the skew component itself is necessary.
6. The current full generator.
7. homogeneous_set using the same full generator construction, and a documented GraphSSM adaptation. Include an exact-ZOH GraphSSM control where practicable, and distinguish it from its paper-update setting.
8. A generic graph-conditioned controlled-ODE/CDE comparator before any broad claim over continuous graph models.

For domain pilots keep b_g free in all transport variants. Their affine forcing can remain order sensitive even in the commuting-basis control. Use the forcing-free probe to establish which distinction is being tested, and avoid interpreting a domain performance difference as pure commutator causation.

Report two fairness views: identical hidden/state dimensions and depth, then a predeclared comparable active-parameter budget. Do not pad restricted models with unused projection heads. Report allocated/inactive parameters, wall time and memory. Saved capacity can be assigned to a forcing or readout MLP in a matched-budget control, with that architectural change disclosed.

Record the validation effect across independent simulation replicates/seeds, with subjects/replicates split before windows, together with fixed-query refinement error and the controlled probe contrast. Predictions at actual ancestry boundaries and interiors should be summarized separately using a predeclared metric, because simple smoothing can improve overall accuracy while degrading boundary localization. Do not select cutoffs from reserved tests.

Go forward only if the full model gives an informative, reproducible advantage over diagonal/commuting transport and simple smoothing, and its compute cost is acceptable. If only exact interval mass helps, the contribution is an interval-aware representation/evaluation result. If only reference summaries help, the proposed architecture has not demonstrated added value. If full and commuting models are tied, discard homogeneous noncommutativity as the central hypothesis.

## External temporal-graph benchmark

Hungarian Chickenpox is a defensible secondary external task: observed weekly node signals on a supplied geographic graph, persistent county correspondence, 52-week observed histories, and one-week-ahead regression. The inspected preparation code uses training-only county scaling, chronological target boundaries, an explicit CC BY 4.0 source manifest, and no target-week value in the input. Later validation/test histories may contain previously observed values from those periods under the declared rolling-forecast protocol. That is normal forecasting information, not future-target access.

It tests temporal feature processing and usefulness of geographic coupling outside local ancestry. For the proposed interface model every county can be an interface vertex, with no eliminated interiors, so C is the geographic Laplacian. This is a valid special case, but it does not exercise Kron elimination or changing interface geometry. Its fixed adjacency and small 20-node graph do not test changing graph topology, anonymous-ancestor correspondence, or large-graph scaling. Do not use this dataset to claim those properties. The right justification is external temporal node forecasting, with the changing-tree ARG task providing the variable-topology setting.

Use persistence, 52-week seasonal persistence, training-only ridge, and learned graph/no-geography controls. Fit choices on training/validation only. A history shuffle that fixes the last observed graph can test dependence on earlier chronology while retaining the local endpoint and history multiset. It cannot, by itself, distinguish diagonal forcing from full homogeneous mixing. Bidirectional processing is permissible only over the already observed history, although a forward-only/directional control better matches chronological interpretation. Keep the scalar-head adaptation disclosed.

Final testing remains reserved. No result or test score was inspected or asserted in this note.

## Audit snapshot

Source hashes at 2026-10-10T02:21:46Z:

| File | SHA256 |
|---|---|
| src/arggnn/models.py | 8429bd02a40950a3dc4d76046ab4b31b366871499d95e295a16ade730f567df5 |
| src/arggnn/reference_encoder.py | 039f3b25383ea92f243090c2eaf592df590768fea7abf41d15fd7c34a7715138 |
| src/arggnn/semigroup_model.py | 4a7a909b214d02f929c569907e97d72aa53e08a42c1b7843a75997448463ca3c |
| docs/dissipative_transport_audit.md | 8f69cc07d5a6e75c268adcc1ccdbaf8e9eb497ad47fc11ceabba3313b449d261 |
| docs/lineage_kernel_operator_audit.md | 129a883ce0a5c4406bcbc7a7b093a0cbf4d12718ea73dae4889bb1ff91c1fab9 |
| GST cached extracted text | 5f13f70a345f654fa68a7240e6d97eefa535de2caccde5d9a4058193ace9c869 |
| GraphSSM v2 cached extracted text | 8a036aae07be4a6409a8cf2b80f97cab4b14e431bb8eaddec1ba2382f8eb1168 |
| Continuum attention v4 cached extracted text | 55a569b623b5cabe67026b145476a60651050648d3b9820e8eb003a2d4a01d11 |

The cache source manifests retain the public PDF versions/hashes. Primary-agent changes after this snapshot are outside this audit.
