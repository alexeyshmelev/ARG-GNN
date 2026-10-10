# Age-conditioned boundary kernel GNN design v1

Date: 2026-10-10. Status: proposed design and critical audit. No implementation, training, GPU job, package installation, Git operation, or final-test evaluation was performed for this note. All reads/writes occurred through anogena, and this note is the only new file written by this task.

## Decision and current evidence

Do not claim an advantage for the current coupled boundary flow. The parent reports the following Q8 inferred-tree training/validation pilot values. I did not independently re-score their predictions.

| Reported pilot control | Validation NLL | Accuracy |
|---|---:|---:|
| Coupled full flow | 0.93783 | 0.65851 |
| Gain-matched independent flow | 0.93411 | 0.67046 |
| Gain/leak-matched independent flow | 0.93725 | 0.65949 |
| Homogeneous bag | 0.92963 | 0.65949 |
| Scalar boundary kernel | 0.89087 | 0.68173 |
| Boundary Transformer | 0.92188 | 0.69495 |

These values give no evidence that changing-operator cross-query memory helps this pilot. A larger independent Q64 oracle/inferred study is already underway. Its results should be assessed before interpreting an extension as a remedy.

There are two defensible, separable hypotheses:

1. **Geometry:** full-age genealogy conductance shares evidence through branches older than the ancestry target's relevant source-separation era. A positive integrated age-resistance family may reduce inappropriate sharing.
2. **Depth:** one linear memory layer fed by graph-local independent sources may be too restricted. Fixed physical anchors permit nonlinear, repeated kernel message passing while preserving both representation contracts.

A narrow proposed next architecture is a two-layer fixed-anchor boundary-kernel GNN with a small fixed age-operator bank and shared source encoding. It must be evaluated factorially against the same architecture using plain geometry and one anchor layer. Do not attribute a combined-model gain to age geometry, depth, rate optimization, or cross-query history without the corresponding independent control.

Before implementing it, inspect label-free Q64 coverage and operator diversity. If age operators are nearly scalar copies of plain Kron, or the cut forest consists almost entirely of singletons, the biological extension is weak or redundant. The log-rate optimization control is an inexpensive alternative explanation to check first.

## Biological meaning and target horizon

The inspected ancient_simulate.py defines the target as the source population of a query's census ancestor at 350 generations. It computes labels outside model input, then removes census vertices, census-only interval boundaries, population fields, migration fields and metadata before constructing the observed genealogy. Inputs retain observed topology/times, observed reference roles, and fixed reference/TMRCA summaries.

The inspected training metadata for the pinned AncientEurope_4A21 demography has admixture events at 140, 180 and 200 generations, source mergers at 600 and 800, and a zero migration matrix. Reference ages are ANA 260, CHG 300, WHG 250 and EHG 250. These ages are younger than the 350-generation census.

At a fixed coordinate in a correct genealogy, two query samples with local MRCA age strictly below 350 lie on the same ancestral lineage at the census. They necessarily have the same census source class. This is an agreement constraint, not disclosure of which class it is. It uses no census node IDs, population annotations, migration paths or query labels.

In this particular pinned no-migration demography, source populations are distinct throughout the open era (200,600). Two queries that coalesce during that era have the same source population at 350 even if their coalescence is older than 350. Thus 350 is a sufficient common-census-ancestor condition, not the maximal safe class-sharing horizon for this model. A 600-generation cut, with a strict boundary convention, can capture additional compatible sharing but relies on the supplied model's separation/no-migration assumptions. Do not generalize it to migration-rich histories or real ancient panels without separate justification.

For inferred trees, a reported MRCA below the horizon can be false. Inferred ages and topology do not establish the biological agreement condition. Hard component tying can force genuinely different-source queries to share predictions. Any cut-based prediction should therefore be a separately named oracle-sensitive control, or a soft observed-geometry hypothesis, rather than a guaranteed-correct inference rule.

A known target age is legitimate task specification when fixed before fitting. It is not equivalent to reading each query's actual ancestry state. In a deployment where the target horizon is unknown, choosing 350 from the simulation label generator is a task-specific prior that must be disclosed and separately estimated or externally specified.

## Fixed positive age-resistance family

For each observed edge with decoded physical endpoint ages t_c<t_p, use a shared age potential Phi_s and resistance

\[
r_e^{(s)}
=\kappa\int_{t_c}^{t_p}w_s(t)\,dt
=\kappa[\Phi_s(t_p)-\Phi_s(t_c)],
\qquad w_s(t)>0.
\]

Times come from the observed tree only, decoded before subtraction. No truth tree is substituted into an inferred-tree bank.

The example w_s(t)=1+(t/s)^p has the exact potential

\[
\Phi_s(t)=t+\frac{t^{p+1}}{(p+1)s^p}.
\]

It is mathematically valid for nonnegative ages and p>0, but can produce very large resistance ratios on old inferred branches. A bounded family is a safer first prototype:

\[
w_s(t)=1+a\frac{(t/s)^2}{1+(t/s)^2},
\qquad
\Phi_s(t)=(1+a)t-a s\arctan(t/s).
\]

Predeclare a=3 and two scales s=350 and s=1400 generations, together with plain w_0=1. These are proposed settings, not fitted findings. A strict 600-generation forest and a 600-scale sensitivity can be checked later as distinct controls for the known separation era.

This bounded family has 1<=w_s<=4. Hence r_0<=r_s<=4r_0 on every edge. Conductance energies and their Kron-minimized boundary energies obey

\[
\frac14 C_0\preceq C_s\preceq C_0.
\]

The bound follows from comparing weighted edge energies before taking the same interior minimum. It limits how aggressively the initial geometry can change, although it does not guarantee improved learning or numerical conditioning on an already ill-conditioned tree.

For a legal inserted unary age t_c<t_u<t_p,

\[
r_{cu}^{(s)}+r_{up}^{(s)}=r_{cp}^{(s)}.
\]

Kron reduction onto the same persistent sample interface therefore preserves C_s exactly in real arithmetic. Evaluate Phi_s once at endpoint ages and take differences. Per-edge learned weights, arbitrary floors, and independent approximate quadrature can break telescoping. Equal-age endpoints remain zero resistance and require the existing explicit zero-branch policy, rather than a silent floor.

Older resistance weakens conductance through deep shared branches. It does not delete all older evidence or guarantee that every strengthened/attenuated entry of C_s corresponds to same-class queries. Kron reduction is global, and interpreting individual off-diagonal changes requires actual operator measurements.

### Normalization matters

The current model divides each graph's plain C_0 by its full-boundary mean diagonal sigma_g. A weighted response independently normalized by its own mean diagonal can remove precisely the global age attenuation that motivates w_s.

For the primary bank use one shared scale per graph:

\[
\sigma_g=\operatorname{mean}\operatorname{diag}(C_{g,0}),\qquad
\widehat C_{g,s}=C_{g,s}/\sigma_g.
\]

Use sigma_g=1 only for an analytically zero plain response. This convention is invariant to sample permutation and unary subdivision. It retains both relative branch-shape changes and attenuation relative to plain geometry. Record raw responses and all normalization choices.

Include the strong scalar-age control

\[
\widehat C^{\rm scalar}_{g,s}
=\frac{\operatorname{mean}\operatorname{diag}(C_{g,s})}{\sigma_g}
\,\widehat C_{g,0}.
\]

It has the weighted operator's mean diagonal but the plain operator's relative geometry. If it explains the gain, the extension acts as graph-dependent scalar age gating rather than a new structural coupling kernel.

A nearly star-shaped tree with a very old common ancestor can make all resistances approximately equal across leaves. Then age weighting can be almost a scalar rescaling, which independent per-scale normalization erases. Measure this possibility rather than assuming multiple age scales create useful distinct geometry.

## Horizon-cut forest and reference grounding

A conceptual forest cut at tau retains only paths younger than tau and introduces unlabeled virtual cut points where edges cross tau. Keep only observed sample identities as the persistent interface. Virtual component identifiers are graph-local equality information and are never numerical embeddings or truth census IDs.

Queries are age zero here. For eligible samples younger than tau, two leaves share a below-cut component precisely when their MRCA is younger than the chosen strict horizon. State a consistent convention for exact ties. A unary insertion exactly at tau must not merge distinct below-cut branches merely because the representation now contains a vertex on the cut. Test the convention independently.

Reduce each forest component onto its existing sampled boundary. Every retained interior component must touch a sample boundary. A dangling virtual cut root is not a clamped known source. It is eliminated as an unobserved interior endpoint.

After the exhaustive query/reference partition,

\[
C_\tau=\begin{pmatrix}C_{QQ,\tau}&C_{QR,\tau}\\C_{RQ,\tau}&C_{RR,\tau}\end{pmatrix},
\]

reference signals can provide grounding only to query components that actually contain a reference sample. A query-only component has C_QR=0 and a zero constant mode in its C_QQ block. Positive learned damping still makes its memory dynamics stable, but it supplies no source-class information. A singleton query component has zero forest response. Do not invent a horizon-root class, clamp the virtual root, or reuse truth ancestry to resolve it.

A reference older than tau has no sampled lineage in the younger retained forest and must be isolated/excluded from that cut channel's connectivity while retaining its observed static features in the common input. Even references younger than tau can be disconnected from every query because their MRCA occurs later. For the actual ages 250–300 and tau=350, only 50–100 generations separate reference sampling from the cut. With the inspected source sizes of 10,000–50,000, reference-grounded components may be sparse. This is a qualitative coverage concern, not a measured coverage result.

In the correct pinned oracle genealogy, a query component containing an observed source reference younger than 350 inherits that reference's source at 350 because those source populations have no intervening migration or merger. This category is an observed panel input, not a query truth annotation. Inferred trees can create incompatible reference categories in one component. Report those conflicts; do not silently select one or use query labels to resolve them.

The hard forest is not the first recommended learned operator. It can lose nearly all reference bridges while duplicating agreement already obvious from young kinship. It should first be measured and compared with a simple observed component-pooling control. For query component incidence E_tau, the kernel

\[
K_\tau=E_\tau E_\tau^\top
\]

is PSD. The normalized component average
\[
P_\tau=E_\tau\operatorname{diag}(n_c^{-1})E_\tau^\top
\]
is a PSD projection. Threshold/cluster and tree-kinship kernels are established constructions. If simple component pooling is sufficient, a Kron forest flow adds complexity without demonstrating a needed mechanism.

## Cached multiscale boundary flows and their kernel

First retain the current fixed query-memory/reference-source partition. Each fixed C_s is reduced offline, and its query block has a cached eigensystem

\[
\widehat C_{QQ,g,s}=V_{g,s}\operatorname{diag}(\mu_{g,s})V_{g,s}^\top.
\]

For each layer ell, scale s and latent channel h use positive global scalars

\[
K_{\ell,s,h}(x)
=\lambda_{\ell,s,h}\widehat C_{QQ,g(x),s}
+d_{\ell,s,h}I.
\]

They are constant over graph holds and shared across query nodes. The eigenbasis remains fixed. Positive damping handles ungrounded/zero modes. A channel's homogeneous partial action is

\[
T_{\ell,s,h}(x+\Delta,x)z
=V_{g,s}\operatorname{diag}
(e^{-\Delta(\lambda_{\ell,s,h}\mu_{g,s}+d_{\ell,s,h})})
V_{g,s}^\top z
\]

inside one graph interval. Across different graphs use the actual chronological product. Do not replace it by exp(-integral K) unless the interval generators commute.

Separate persistent states are required for each scale. Define global nonnegative mixture weights pi_{\ell,s,h}=softmax_s(alpha_{\ell,s,h}) and mix the scale-specific kernel messages/readouts. This preserves cached eigenpairs. It is not the same model as evolving a single state under sum_s pi_s K_s. Those matrices generally have different eigenvectors, and their sum would require another eigensystem. Nor may one average per-hold transition matrices and assume a semigroup: cross-scale products appear when such an averaged action is composed after an artificial interval split.

### The bidirectional block kernel is PSD

For one channel/scale, define the causal propagator T(x,y) by dT/dx=-K(x)T for x>=y and T(y,y)=I. T itself need not be symmetric for noncommuting interval operators. Define the matrix-valued space-time kernel

\[
\mathcal H(x,y)=
\begin{cases}
T(x,y),&x\ge y,\\
T(y,x)^\top,&x<y.
\end{cases}
\]

This complete block kernel is PSD. The auxiliary SDE

\[
dX=-K(x)X\,dx+\sqrt{2K(x)}\,dW,\qquad
\operatorname{Cov}(X(\ell))=I,
\]

with the initial state independent of future Brownian increments, has covariance equation P'=-KP-PK+2K and solution P=I. Its cross-covariance is exactly H. This argument allows symmetric positive definite noncommuting K. The positive-duration reverse scan reverses the order of symmetric hold factors and thus computes the transposed causal block. This is the technical result inspected in Prop.1 of boundary_kernel_working.tex and the independent implementation audit.

Global nonnegative mixtures sum_s pi_s H_s remain PSD. If the scale-specific sources differ through C_QR or lambda, the complete message is a sum of PSD-kernel actions on different source fields, not one mixed H applied to a single shared source. The rectangular reference-to-query source map is not itself a PSD Gram matrix. These are conditional within-sequence sample/coordinate kernels. They are not automatically PSD similarities between entire graph collections, Gaussian-process predictions, or uncertainty estimates.

For conductance C_QQ and positive damping, local exp(-Delta K) is entrywise nonnegative and substochastic. Products retain nonnegativity. This permits positive row-mass normalization in the message operator. A row-normalized action generally is not a symmetric Gram matrix, even though its underlying H is PSD. Distinguish those statements.

## Fixed-anchor nonlinear kernel GNN

Choose P physical anchors y_1<...<y_P from a predeclared grid on each fixed tile, independent of tree midpoint/record count. A proposed first grid is P=32 equal-width cell midpoints, with Voronoi/cell weights omega_p equal to the physical cell widths. Use the same grid and weights for every control.

Compute the ordinary shared plain-boundary local encoder once for each observed graph. Sample its invariant leaf embeddings at the containing tree of each anchor, giving U^0_p. The common leaf input remains the existing 47 observed fields with raw parent-branch omitted. No ancestry target, horizon-root label, inferred census ID or extra reference summary is appended only to the candidate.

For the first prototype, query embeddings can receive context at anchors while reference embeddings remain the independently encoded observed-source bank. For each layer, use shared node projections/gates to build source S^ell_p from query U^ell_p and the fixed local reference embeddings. One explicit source matches current grounding semantics:

\[
S_{\ell,s,h,p}
=q_{\ell,h}(U^\ell_{Q,p})
-\lambda_{\ell,s,h}
\widehat C_{QR,g(y_p),s}
r_{\ell,h}(U^0_{R,p}).
\]

The same q/r maps are shared across scales. Each source is evaluated once at its physical anchor. Neither source nor generator is recomputed because a graph record is duplicated. Both maps use only observed inputs and previous-layer anchor context.

The kernel message for anchor p is

\[
M_{\ell,h,p}
=\sum_s\pi_{\ell,s,h}
\sum_q\omega_q\mathcal H_{\ell,s,h}(y_p,y_q)
S_{\ell,s,h,q}.
\]

This is an exact finite anchor-kernel action. It is not exact integration of a continuously varying nonlinear source field. The anchor masses are a chosen discrete measure, and changing P changes the model's resolution. No anchor-refinement invariance is asserted.

For each scale/channel it can be evaluated without materializing a P|Q| by P|Q| Gram matrix. Maintain an inclusive forward sum a_p and an exclusive reverse sum b_p:

\[
a_p=T(y_p,y_{p-1})a_{p-1}+\omega_pS_p,
\]

\[
b_p=T(y_{p+1},y_p)^\top
(b_{p+1}+\omega_{p+1}S_{p+1}),
\]

with a_1=omega_1 S_1 and b_P=0. Then a_p+b_p equals the message and counts the self term once. Homogeneous actions between anchors use every intersected physical graph hold, including partial holds, with cached spectral factors. There is no per-tree nonlinear update between anchors.

For the primary row-mass normalization, run the same scans with both query and reference source projections replaced by ones. The scale-specific mass source is
\[
m_{\ell,s,h,q}
=\mathbf1_Q-\lambda_{\ell,s,h}
\widehat C_{QR,g(y_q),s}\mathbf1_R.
\]
Because C_QR has nonpositive entries, this source is at least one. The denominator is
\[
D_{\ell,h,p}
=\sum_s\pi_{\ell,s,h}\sum_q\omega_q
\mathcal H_{\ell,s,h}(y_p,y_q)m_{\ell,s,h,q}>0.
\]
The diagonal self term at an anchor is at least omega_p. At a non-anchor readout x there is no self impulse: the mass is still strictly positive mathematically for finite rates because each sample has positive same-sample propagation from the anchors, but the omega_p lower bound does not apply. Check numerical mass positivity and underflow explicitly rather than silently adding a denominator floor. Use M/D elementwise as an attention-like normalized message over the actual query/reference source coefficients. This avoids omitting reference-source mass from the normalization while changing its coupling rate. A query-only kernel-mass denominator using ones_Q is a different declared choice. Use the same choice in every matched control. Neither normalization is a record-count normalization, and neither makes the row-normalized operator itself a symmetric PSD Gram matrix.

Update only at the fixed anchors, for example

\[
U^{\ell+1}_{Q,p}
=\operatorname{LayerNorm}
\left[
U^\ell_{Q,p}
+\sigma(G_\ell[U^\ell_{Q,p},M^\ell_p])
\odot W_\ell M^\ell_p
\right].
\]

Two such layers permit later sources to depend on earlier contextual messages. This creates repeated contextualization missing from the present single independent-source linear memory layer. It is also closely related to established gated graph/kernel message passing and GST-style interleaving. Depth alone is not a new GNN principle.

For an unambiguous two-layer decoder, also evaluate each layer's anchor-source kernel message at every requested fixed coordinate x. Initialize U^0(x) from its independently encoded local containing-tree embedding. Apply the same layer gate/update to U^ell(x) and M^ell(x), then classify U^2(x). Only the fixed anchors produce source impulses for later layers; evaluation coordinates do not become additional source nodes. This avoids a hidden third kernel layer and makes predictions independent of which other readout coordinates were requested.

Do not interpolate an interval-end contextual state and silently claim exact physical-coordinate processing. Every query action uses actual partial hold products. A nearest-anchor decoder is a separate approximation and is not the proposed first decoder.

### Why the two subdivision contracts survive depth

For identical interval splitting, each hold generator is unchanged and its homogeneous factors compose by the semigroup law. Therefore every T between fixed anchors/queries is unchanged. For serial unary splitting, telescoping resistance preserves each C_s and all selected leaf features. Therefore each T and base anchor embedding is unchanged.

By induction over layers, unchanged preceding anchor embeddings yield unchanged sources, kernel messages, gates and subsequent anchor embeddings. This proves the model contract in exact arithmetic for fixed anchors, interface, partition, tile endpoints, queries, features and deterministic evaluation.

It does not prove invariance to changing anchor count, moving tile boundaries, adding samples, changing source roles, arbitrary age perturbations, or introducing stochastic masks at newly serialized records. Anchor grid resolution and evaluation breakpoints need the same declared containing-tree convention as the current model.

### Optional all-sample memory

All persistent samples can instead carry anchor memory, including references. Use K=lambda C_s+dI on B=Q+R samples and source projections on all observed embeddings. This gives references history-bearing context while their observed roles remain fixed inputs. It contains no query truth leakage.

That change removes the original clamped-reference forcing model and enlarges the dynamic interface. It must be a separate ablation, not bundled with age/depth as the first prototype. Cache full-boundary eigensystems and report O(B^2 H) actions instead of O(Q^2 H). At Q64/R24 the increase may be manageable, whereas it is much larger relative to a Q8 state. Do not treat known reference category as guaranteed ancestry at arbitrary horizons or add a reference ancestry loss without a separately declared biological target.

## Rate optimization is a control, not a novelty mechanism

Current coupling uses lambda=softplus(theta) initialized at lambda=20, so theta is approximately 20 and d lambda/d theta is approximately 1. For a small parameter update Delta theta,

\[
\Delta\log\lambda\approx\Delta\theta/20.
\]

Thus Adam's approximately additive parameter steps can move a large rate slowly in relative terms. This is a local parameterization observation, not proof that the current training is stuck or that better optimization will produce a gain. Inspect recorded rate trajectories and gradients.

Use a positive log parameterization

\[
\lambda=\lambda_{\rm init}e^\eta,\qquad \eta_{\rm init}=0,
\]

and likewise for damping. A small Delta eta gives Delta log lambda=Delta eta. Forward at initialization is identical, while optimizer trajectories differ. At the same nominal small Adam step, the initial relative coupling change can be about twenty times larger than softplus-at-20. Different gradient scales, moments, clipping and schedules matter, so do not convert this into a promised convergence rate.

Run plain geometry with log rates as an independent optimization control. Apply identical rate parameterization/optimizer groups to full, gain-matched, gain/leak, bag and scalar controls used for attribution. Exclude positive-rate/log-mixture parameters from weight decay in both compared parameterizations, or explicitly report the different induced priors. Do not compare a new age/depth/log-rate candidate only against an old softplus baseline.

Record initial/final rates, relative movement, modal Delta*(lambda mu+d), gradient norms and selection behavior. Reject nonfinite rates rather than hiding overflow behind an undocumented clamp. A predeclared bounded log range is an alternative modeling choice, not a silent repair.

## Narrow prototype and decision experiment

Recommended prototype scope is limited to:

- Fixed bounded age bank: plain, s=350 and s=1400, a=3, common plain normalization.
- Shared existing plain local encoder and q/r source maps, Q-only memory and observed reference forcing.
- P=32 fixed physical anchors, two nonlinear anchor layers, global nonnegative scale mixture and log-positive rates.
- No learned resistance warp, online eigensolve, horizon-root labeling, all-sample-memory expansion or additional query features.

Use the same total latent width/active budget and anchor inputs in the factorial comparison:

| Comparison | Question resolved |
|---|---|
| Plain existing flow, softplus versus log rates | Could rate optimization explain the earlier negative result? |
| One versus two anchor layers, repeated plain geometry bank | Does nonlinear context feedback help independently of age weighting? |
| Plain repeated bank versus age bank, fixed depth | Does age geometry help independently of multibranch memory/capacity? |
| Scalar-age-scaled plain bank versus warped bank | Does relative branch geometry matter beyond scalar attenuation? |
| Full versus gain-matched/gain-leak independent states, same age bank | Is cross-query history needed beyond instantaneous coupled equilibrium? |
| Ordered versus homogeneous bag | Does actual graph arrangement add useful information? |
| Static/local multiscale GNN and scalar genomic kernel | Is temporal boundary memory needed? |
| Same-anchor Transformer and original boundary Transformer | Is the proposed invariant kernel/depth useful against attention on the same inputs and against the existing stronger control? |
| Horizon component pooling/forest control | Is the target-aware signal reducible to simple observed cluster averaging? |

A repeated-plain bank must have the same number of states, global mixture parameters, source maps and output channels as the age bank. Predeclare distinct genomic coupling initializations if multiscale temporal capacity is intended, and use the identical initialization schedule for the corresponding age/plain branches. Otherwise identical branch symmetry can make the repeated-plain bank an artificially weak comparison.

Holding total readout dimension fixed does not eliminate the extra hidden memory capacity of multiple branches. Report it and match it in the repeated-plain control. Depth comparisons need a wider or deeper local/source-map control with comparable active parameters, in addition to same-width comparisons. Record preprocessing/cache bytes, encoder cost, forward/backward training time and memory.

For label-free admissibility/coverage diagnostics, report on train/validation observed inputs only:

- Fractions of queries in non-singleton horizon components and in components with any reference, plus component-size distributions.
- Cross-source observed reference conflicts within cut components.
- Isolated/older reference counts, without filling their missing grounding using targets.
- Weighted-versus-plain response residual after best scalar rescaling, mean-diagonal attenuation, spectra and reference-grounding row sums.
- Oracle versus inferred operator/cut stability where paired observed trees exist, without substituting oracle geometry into inferred model input.

Using validation labels, separately evaluate whether inferred young-coalescence pairs disagree in true ancestry. That is an error diagnostic, not an input feature or a gate chosen per query from truth. Keep parent replicates and subjects split before windowing, preserve fixed scoring coordinates, report ancestry boundary/interior performance and calibration, and reserve final tests.

Implementation contract checks should verify all selected features are recomputed after unary insertion, age-potential telescoping, Kron response equality, fixed-anchor message equality after interval splitting, self-term counting, PSD of small block H/mixed H fixtures, equivalence of scan and materialized anchor action, correct reverse product orientation, finite source/rate gradients, and no response to changes in external label tensors. These tests establish the implementation and mathematics, not a predictive contribution.

Do not launch the architecture merely because it can be written. Proceed when label-free diagnostics show nontrivial geometry/coverage and the factorial pilot can distinguish the hypotheses. If repeated-plain depth or scalar-age gating explains all gains, simplify to that result. If the stronger Transformer/scalar kernel remains better, report that limitation and stop treating boundary memory as an established winning architecture.

## Prior-art boundaries and scope

Directly inspected repository sources for this design include ancient_simulate.py, boundary_memory.py, kron_interface.py, boundary_baselines.py, boundary_study_controls.md, boundary_memory.md, the Q64 configurations, training demography metadata, and the specified technical covariance sections of boundary_kernel_working.tex and boundary_implementation_audit_v1.md. I did not read methods_working.tex or produce a manuscript verdict.

Established predecessors already include:

- [Kron reduction](https://arxiv.org/abs/1102.2950) and [Node Decimation Pooling](https://arxiv.org/abs/1910.11436): static boundary response, equivalent series edges, and Kron/GNN coarsening.
- [Diffusion kernels on graphs](https://openalex.org/W14071851): matrix-exponential graph kernels. The linked bibliographic record/title/2002 abstract was verified through OpenAlex in this task; the full paper was not equation-audited here.
- [The Multiscale Laplacian Graph Kernel](https://arxiv.org/abs/1603.06186): multiscale graph-kernel representation. Its verified abstract uses nested subgraphs and differs from age-warped resistance, but precludes treating “multiscale Laplacian kernel” as a newly invented category.
- [Phylogenies and the Comparative Method](https://doi.org/10.1086/284325): classical tree-based phylogenetic dependence. Bibliographic metadata was verified here. Positive branch-weight/shared-lineage covariance and threshold-cluster kernels are established mathematical ingredients, not claimed new results.
- [Non-separable Spatio-temporal Graph Kernels via SPDEs](https://proceedings.mlr.press/v151/nikitin22a.html), [dynamic graph kernel reconstruction](https://arxiv.org/abs/1612.03615v2), and linear-SDE covariance constructions: graph space-time kernels and the covariance framing have direct precedents, as independently source-audited in boundary_implementation_audit_v1.md.
- [GraphSSM](https://arxiv.org/abs/2406.00943v2), [Message-Passing State-Space Models](https://arxiv.org/abs/2505.18728v2), and [GST](https://arxiv.org/abs/2606.05116v1): persistent graph memory, message-passing SSMs, and gated repeated local/context processing already exist.

The fixed age bank is a choice of weighted genealogy metric. Nonnegative mixtures are standard multiple-kernel construction. The fixed-anchor extension is nonlinear graph/kernel message passing on a conditional covariance kernel. Its potentially useful distinction is the jointly specified observed boundary interface, biological age geometry, and exact two-subdivision contract through multiple nonlinear layers.

A broad first-kernel or first-GNN claim is unsupported. Scientific value would come from a needed ancestry-aware mechanism, independently attributed gains, inference robustness, and an informative failure analysis. No such gain is established by the current pilot or by this design note.

## Design source snapshot

- src/arggnn/ancient_simulate.py: 4098a1c8a43f74355e248c91d3491f6b8d4cd8f1c7193c526ac1eba56df33290
- src/arggnn/boundary_memory.py: a056cd10d1c848d7dabdc1daf646548876fd90e8f489667c46a374ac28cd11dc
- src/arggnn/kron_interface.py: 09ec6286c185e7aa2b13d76dcbe8fea24b3a1c785a6619ac9bc5464e0de66d69
- docs/boundary_study_controls.md: e1fc261da2109a27c5cd8f5108a6b1dd4935464bbb9f8975eab6b3ffe92b8182
- paper/boundary_kernel_working.tex: a70323ffa11f6bcca100a7157469f350b9f6bc5d7b957147f3ba47f4aa91eba4
- research/reviews/boundary_implementation_audit_v1.md: 5c10a71cfda3aafc1a89d2dd8f24feb0db2e970712f677e97451f6c0a0e8288b
- data/ancient_common_input_pilot_v2/train/rep000/metadata.json: b14fa770c0c2afd3bd7d2f3de7a241ae7312e57b8b602f51d21f2ff4a8199c36
