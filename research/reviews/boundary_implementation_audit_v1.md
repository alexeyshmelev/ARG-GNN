# Independent implementation audit: invariant boundary query memory

Audit date: 2026-10-10. Scope: mathematical and implementation correctness, observable-input separation, two refinement contracts, mechanism controls, and targeted prior art. This is not a manuscript acceptance review.

All inspection and CPU computation occurred on anogena in /workspace-SR003.nfs2/ARG-GNN. No prior reviewer verdicts were read. No packages, training, GPU jobs, commits, or shared implementation edits were performed. The only file written by this auditor is this report. The root subsequently supplied proposed controls and the covariance interpretation for independent checking.

## Assessment and claim gates

The tested main regime has correct Schur-complement reduction, harmonic extension, exhaustive query/reference partition, and exact constant-generator affine flow. The serial-edge and interval-subdivision contracts follow under the stated positive-resistance, unchanged-feature, fixed-boundary and fixed-tile assumptions. No algebraic blocker was found for the actual 32-boundary, Q8/R24 pilot bank.

The initial audit found a concrete public-API numerical defect at an exactly zero theoretical response: singleton-boundary reduction could leave Schur roundoff that normalization amplified or rejected. The root subsequently fixed this per connected component, and the final 90-test verification passes. The original counterexample is preserved below. See the final amendment for integrated source hashes and resolved status.

The original full-versus-diagonal comparison is insufficient for a performance claim specifically about useful cross-query temporal evidence transfer. It also changes static gains and modal decay. The added gain-matched diagonal control is mathematically correct and removes the equilibrium-gain confound, while retaining a modal-timescale difference. A small causal-route probe establishes that the implemented exchange route exists. It does not show that learned ancestry improvement, if observed, uses that route.

Kron reduction followed by a GNN, graph diffusion, exact zero-order-hold flow, matrix-valued graph kernels, and persistent hidden-state exchange each have direct precedents. A defensible formulation is a candidate application-specific synthesis of a fixed sample-boundary response, observed invariant source bank, exact continuous genomic flow, and jointly audited serial-edge/interval subdivision contracts. No first-in-literature conclusion follows from this bounded search.

## Source snapshot and evidence

Initial boundary-memory source SHA256 at the focused suite:
- kron_interface.py: 69ead9015c3d6edda5208b3e1807da46bb9fc0e650ba9852da93bd98b9a21f5c
- boundary_memory.py: 56bf00edf3aa223dc26a76e36b2d840a1c41622fa3f214a83aeccd5f276bbe36
- test_kron_interface.py: 1df7efa996d7ad861c7f0f6ed5111acfce170fe0fe0a58b018f8ff8490c0ec71
- test_boundary_memory.py: f68349a12404a3e58ccc51f6fe9ee7f9d1b873524a5fc9ca48c175ad01da3192
- inspected prepared input: 92d1f4e438de6fe4792096a0af096064cd8ae632b7cb9d8ea77925ff6e9daa6a

The focused suite passed **46 tests in 3.44 seconds**:
```sh
source scripts/project_env.sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$ARGGNN_PYTHON" -m pytest tests/test_kron_interface.py tests/test_boundary_memory.py -q -s -p no:cacheprovider
```
Capture and pytest cache were disabled. Additional diagnostics were executed as in-memory scripts, with no diagnostic files or bytecode written.

During the audit, the root added gain_matched_diagonal. Its source SHA256 during this auditor's additional checks was 8c747ccd8df045850f8b6766e9691b033f894b9f4c87c8bcdddd3fd43cd92809. The three aggregation modes of that addition independently produced finite gradients and identical source forcing/parameter counts to copied full models. Combined unary/root-cap/interval refinement changed gain-control outputs by at most 2.78e-16. Full and gain-matched equilibrium logits were exactly identical at copied weights. These are supplemental in-memory checks, not a claim that the original 46 tests covered the new variant.

The missing boundary_memory.md docstring target at initial inspection was in-progress and now exists. It is not treated as an unresolved blocker.

## Mathematics and observable-input separation

For positive conductance and every interior component anchored to a boundary, L_II is positive definite and the implementation correctly computes:
```
C = L_BB - L_BI solve(L_II,L_IB)
X_I = -solve(L_II,L_IB) X_B
```
The Cholesky solve, energy/current identities, boundary ordering, deduplication of bidirected simple edges, and explicit rejection of unanchored/ill-conditioned interiors are appropriate. Physical parallel edges are deliberately not represented by duplicate serialization. Zero-resistance contraction is deliberately unsupported. Those restrictions must accompany the contract.

With per-channel global positive lambda and delta, K_h=lambda_h*Cqq+delta_h*I is positive definite. Cached eigensystems justify exp/expm1 actions and inverses. Eigenvector signs and rotations in repeated eigenspaces cancel because functions of the eigenvalues are scalar within each eigenspace. Graph-varying operators need separate bases, which are present. No global common eigenbasis or exp(-integral K) shortcut is justified when operators do not commute.

The source GNN reads only leaf_x and normalized response. Neither temporal state, record count, width, midpoint, coordinates, numerical IDs nor clades enters it. Weighted neighbor means are a standard message-passing construction. Integer IDs only select/aligned rows. Full-boundary mean-diagonal normalization removes an absolute common resistance scale from the temporal operator. The local neighbor mean already cancels any scalar response rescaling. Absolute age information can still enter through the supplied sample/TMRCA features.

The prepared input has eight original observed fields: four reference categories, query/leaf flags, log time, and log parent branch. Its adapter adds forty reference-summary columns. Query leaves have exactly zero original reference-category indicators in the inspected replicate. The primitive excludes raw parent-branch column 7, leaving 47 columns. The adapter accepts only observed one-hot reference labels and rejects query labels in those columns. Its summaries use observed sample-pair MRCA structure and observed time, not census targets.

ancient_simulate.py computes targets externally from the census-bearing genealogy, simplifies away unary census nodes and census-only boundaries, and removes population/individual/metadata annotations before packing observations. GraphSequence has no ancestry-target field. The boundary model does not read external labels or truth. This is an oracle observed-genealogy benchmark, not evidence of inferred-tree robustness.

The generic interface cannot certify arbitrary upstream x: excluding one declared column does not prevent a caller from putting labels or other non-invariant features in remaining columns. Preparation/loading provenance and audited feature selection remain necessary. A frozen dataclass also does not prevent in-place tensor mutation. from_dict validates schema fields, not source authenticity or cache/eigensystem consistency. _validate_bank is a consumer sanity check, not a complete untrusted-cache/provenance validator.

## Both refinement contracts

Serial-edge refinement holds when an edge r is replaced by positive r1,r2 with r1+r2=r and the inserted node is not a sample boundary. Branch resistance proportional to decoded physical duration satisfies the series law. Log branch encodings cannot be added as physical resistances. The inserted age must remain strictly between endpoint ages at stored precision.

At sampled leaves, the forty adapter summaries depend on the same sampled MRCA/reference relations. Inserting a non-sample unary ancestor introduces no new reference sibling shell and leaves those summaries unchanged. Sample age/role also stays fixed. The immediate parent-branch feature changes and is correctly excluded.

This auditor reconstructed the full reference adapter on the actual first tree from train/rep000 after a positive unary insertion. The selected leaf-feature maximum difference was exactly 0, raw response difference 3.55e-15, normalized response difference 4.44e-16, and random double-precision model logit difference 1.28e-15. This extends the synthetic fixture evidence to actual input construction.

Interval refinement holds because identical copies induce identical local sources, K and forcing, and the affine flow composes by the semigroup law. Width-weighted bag averages preserve total mass. Coordinates retain float64 while matrix/feature transfer can use float32. Fixed tile endpoints are part of the guarantee: changing reset positions, adding sampled boundary nodes, changing reference roles/metric or introducing zero branches is outside it.

The reflection readout is valid away from discontinuity tie ambiguities. At an exact original graph breakpoint x=.4, reflection maps the retained original target graph to its transformed right endpoint, which the half-open containment contract rejects. This auditor reproduced the ValueError. State/kernel transitions are continuous at such points, but the graph-local readout can jump. State a breakpoint tie convention or qualify the current reflection test/claim. The tests use interior scoring positions.

Numerical subdivision equality is within floating-point tolerance, not bitwise identity or a uniform bound independent of the number of subdivisions. Complete low-precision scan error has not been established by the single-action or small-fixture checks.

## Concrete zero-response normalization defect (pre-fix snapshot)

A connected graph with exactly one boundary and no shunts has C=0 analytically. The implementation's source-relative backward checks can admit Schur cancellation residual:
```python
L = resistance_laplacian(
    3, torch.tensor([[0,1],[1,2]]),
    torch.tensor([.2,.3],dtype=torch.float64))
reduction = kron_reduce(L,torch.tensor([0]))
# response == [[1.7763568394002505e-15]]
# response_row_sum_relative_to_response == .48
```
A one-edge resistance .7 instead returned -2.220446049250313e-16. The first case is normalized to [[1]], whereas the second fails the positive normalization-scale check. An isolated single boundary with no edges normalizes its exact zero to zero. Thus adding an unobserved dangling tree can change/reject the zero-response bank despite theoretical equivalence.

Use an explicitly documented analytic policy for singleton-boundary components, or restrict/reject the degenerate consumer contract with a specific explanation. Do not treat arbitrary weak responses as zero using undocumented epsilon clipping. Source-relative backward stability and L_II condition number alone do not establish relative accuracy of a weak boundary response. The response-relative diagnostic should be bound to the preparation protocol. The actual B32 bank's reported residuals are small.

## Mechanism identification and feasible same-bank controls

Define K=lambda*Cqq+delta*I, D=diag(K), F the existing learned forcing, and e=K^-1 F.

A normalized three-leaf star with Q2/R1 gives Cqq=[[1,-.5],[-.5,1]]. At lambda=20,delta=1, the full homogeneous decay rates are 11 and 31, whereas the original diagonal control has rates 21 and 21. Even identical constant F=[1,1] produces full equilibrium [1/11,1/11] versus diagonal [1/21,1/21]. At distance .1 from zero their entries are .0606481 and .0417878. This difference needs no heterogeneous other-query past evidence.

On the actual 972-tree pilot, the ratio of slowest original-diagonal decay to slowest full decay at the initialization was median 1.00947 and maximum 4.12582. The confound is not uniformly large, but some graphs have a substantial difference.

| Control | Same-bank construction | Interpretation |
| --- | --- | --- |
| Original diagonal | z'=-Dz+F | Removes off-diagonal state coupling, changes equilibrium and modal times |
| Added gain-matched diagonal | z'=-Dz+D e | Matches each local full equilibrium, has no persistent cross-query state transfer |
| Optional gain/leak matched diagonal | d_q=delta-lambda*(Cqr*1)_q; z_q'=-d_q(z_q-e_q) | Also matches full decay of a spatially constant homogeneous state, retains no persistent cross-query exchange |
| Full equilibrium | read e at target graph | Tests whether static spatial gain/source evidence already explains the result |
| Full homogeneous bag | average K and F by exact widths | Tests arrangement beyond the homogeneous average, conditional on local target readout |
| Same-bank local and temporal neural controls | same features/response with local GNN, per-query GRU/Transformer, feature-only controls | Distinguishes representational/input advantage from the particular exact-flow design |

The gain-matched variant is correct as added. For the homogeneous bag, its forcing must be D_bar*K_bar^-1*F_bar, which the current implementation computes. Averaging D_g*e_g would be a different control. The optional leakage version uses empty-reference leakage zero and requires finite positive numerical d_q. It matches one common spatial mode, not the entire spectrum. Assigning cached eigenvalues arbitrarily to query IDs to create a 'spectrum-preserving diagonal' would generally violate node-permutation equivariance.

Use both copied-checkpoint interventions and independently trained same-budget controls. Copied checkpoints preserve exact sources and isolate an operator change. Separate training can compensate through sources/head/rates and is a model comparison. The final learned forcing is not identical merely because two independently trained models have the same architecture.

Equilibrium currently uses a smaller readout than bidirectional flow: hidden+H versus hidden+2H. Disclose this. A same-size symmetric static head [local,2e,0] can keep schema size, but its zero block is inactive and must not be counted as active capacity.

The existing synthetic causal-route and gradient tests establish that another query's earlier evidence reaches the target through persistent coupled state when later forcing is fixed. Retain them, ideally alongside held-forcing interventions on trained checkpoints. They are implementation evidence, not predictive benefit. Coupling can also diffuse ancestry across unrelated or different-source query pairs. No implemented census-age/kinship gate guarantees beneficial exchange.

## Conditional channelwise covariance-kernel interpretation

The root's proposed interpretation is correct, including noncommuting K. Condition on one supplied observed graph signal and learned positive parameters. For each channel h, let T_h(x,y) solve:
```
dT_h(x,y)/dx = -K_h(x) T_h(x,y),  x>=y
T_h(y,y)=I
H_h(x,y)=T_h(x,y)                  if x>=y
H_h(x,y)=T_h(y,x)^T                if x<y
```
On a finite domain [a,b], construct the auxiliary linear SDE:
```
dX_h(x) = -K_h(x) X_h(x) dx + sqrt(2 K_h(x)) dW_h(x)
X_h(a) independent of future W_h, mean 0, Cov(X_h(a))=I
```
Its marginal covariance P obeys P'=-K P-P K+2K. P=I is the unique solution from P(a)=I, including continuity across piecewise-constant jumps. For x>=y, independent future innovations give Cov(X_h(x),X_h(y))=T_h(x,y). Therefore H_h is a symmetric positive-semidefinite matrix-valued covariance kernel, even though individual T_h(x,y) need not be symmetric.

The forward context is integral_a^x T_h(x,y)F_h(y)dy. Reversed positive-duration scan is integral_x^b T_h(y,x)^T F_h(y)dy: each local exponential is symmetric and reversing their order gives the transpose of the chronological product. Thus f_h+b_h equals integral_a^b H_h(x,y)F_h(y)dy. Diagonal and gain-matched controls have diagonal homogeneous kernels, with different source weighting. Exact partial interval integration and both refinement contracts carry over when K,F are unchanged.

The auxiliary covariance process is used to prove PSD. The implemented memory starts from deterministic zero at each integration endpoint and contains no simulated SDE noise. f+b is not an OU posterior smoother, GP likelihood, uncertainty estimate, or kernel between entire graph collections. It is a conditional within-sequence channelwise kernel action. F depends on learned lambda and graph-local nonlinear sources. The final predictor also includes the local GNN embedding and squared directional difference, so its complete logits are not solely kernel convolution.

Matrix exponentials/inverses are smooth in positive lambda,delta. With fixed observed interval endpoints, the action is differentiable in learned parameters. Coordinate derivatives exist inside graph intervals and can jump at their boundaries. Local readouts can be discontinuous. Do not claim global smoothness in coordinate or gradients through the discrete observed topology.

Independent tiny diagnostic: three symmetric noncommuting K segments with intervals [0,.4),[.4,.7),[.7,1] and five coordinates [0,.15,.45,.9,1]. The first commutator Frobenius norm was 4.52548 and T(.8,.1) asymmetry .0416249. The resulting 10x10 block H was exactly symmetric with minimum eigenvalue .1393454. Independently integrated ordered matrix-exponential kernel action agreed with cached-spectral forward+reverse sum to 1.39e-16. This is a diagnostic corroboration of the proof, not a stochastic training result.

## Verified nearest primary references

The bibliography below distinguishes direct equation/full-text inspection from metadata/abstract verification. No paper is claimed to implement the exact complete current synthesis. Absence from this limited title/abstract search is not evidence of absence from the literature.

1. **Florian Dörfler and Francesco Bullo.** "Kron Reduction of Graphs With Applications to Electrical Networks." IEEE Transactions on Circuits and Systems I: Regular Papers 60(1):150–163, January 2013. DOI [10.1109/TCSI.2012.2215780](https://doi.org/10.1109/TCSI.2012.2215780). Primary preprint [arXiv:1102.2950v1](https://arxiv.org/abs/1102.2950v1), 2011. Crossref bibliographic metadata and arXiv abstract verified. Direct precedent for the static Schur response, electrical interpretation, resistive identities and graph-theoretic properties.

2. **Filippo Maria Bianchi, Daniele Grattarola, Lorenzo Livi, and Cesare Alippi.** "Hierarchical Representation Learning in Graph Neural Networks With Node Decimation Pooling." IEEE Transactions on Neural Networks and Learning Systems 33(5):2195–2207, May 2022. DOI [10.1109/TNNLS.2020.3044146](https://doi.org/10.1109/TNNLS.2020.3044146). Primary [arXiv:1910.11436v3](https://arxiv.org/html/1910.11436v3), updated April 20, 2024, original preprint October 2019. Crossref metadata and full-text Sec.III-B Eq.7 verified. It uses offline Kron reduction to generate coarsened graphs for GNNs, followed by sparsification. A fixed chosen sample boundary and exact unsparsified contract differ, but "Kron plus GNN" is not a new construction.

3. **Andrea Ceni, Alessio Gravina, Claudio Gallicchio, Davide Bacciu, Carola-Bibiane Schönlieb, and Moshe Eliasof.** "Message-Passing State-Space Models: Improving Graph Learning with Modern Sequence Modeling." Explicit version [arXiv:2505.18728v2](https://arxiv.org/html/2505.18728v2), updated May 26, 2026, original preprint May 24, 2025. No venue was verified. Metadata and full-text Eq.2 verified: X_(t+1)=A X_t W+U_(t+1)B. It exchanges persistent hidden state across nodes, permits other graph shift operators, and develops exact sensitivity/spectral calculations. Broad claims of first cross-node memory exchange or first equivariant graph SSM are untenable.

4. **Jintang Li, Ruofan Wu, Xinzhou Jin, Boqun Ma, Liang Chen, and Zibin Zheng.** "State Space Models on Temporal Graphs: A First-Principles Study." Primary [arXiv:2406.00943v2](https://arxiv.org/html/2406.00943v2), October 29, 2024, original June 2024. Metadata and Eqs.5,8,21 inspected. Eq.5 is U'=U A^T+(I+alpha L(t))^-1 X B^T: graph diffusion acts on incoming evidence, whereas current full K also couples persistent node state. Appendix B.2 handles varying node sets by identity-aligned memory. Interleaved multi-layer adaptations can create further historical exchange and cannot be dismissed from the single-layer equation.

5. **Alexander V. Nikitin, St John, Arno Solin, and Samuel Kaski.** "Non-separable Spatio-temporal Graph Kernels via SPDEs." Proceedings of the 25th International Conference on Artificial Intelligence and Statistics, PMLR 151:10640–10660, 2022. Verified primary proceedings [paper and BibTeX](https://proceedings.mlr.press/v151/nikitin22a.html). Full text [arXiv:2111.08524v3](https://arxiv.org/html/2111.08524v3), updated December 27, 2024. Sec.4.4 derives graph stochastic-heat covariances, including general spatial noise and a Lyapunov relation in Eq.19. It includes Hungary Chickenpox experiments in Sec.5.3. This is a close direct predecessor for graph diffusion covariance/kernel framing. Its inspected closed forms use fixed graph operators. Time-varying noncommuting operators and the exact sample-boundary refinement contracts require separate explanation, not a first graph-kernel claim.

6. **Daniel Romero, Vassilis N. Ioannidis, and Georgios B. Giannakis.** "Kernel-based Reconstruction of Space-time Functions on Dynamic Graphs." IEEE Journal of Selected Topics in Signal Processing, 2017. DOI [10.1109/JSTSP.2017.2726976](https://doi.org/10.1109/JSTSP.2017.2726976). Primary [arXiv:1612.03615v2](https://arxiv.org/abs/1612.03615v2). Authors/title/DOI/year verified from Crossref and arXiv abstract. It explicitly treats space-time kernels on changing topology and online kernel Kalman filtering. The specific kernel constructions were not fully equation-audited here. No first time-varying graph-kernel claim is justified.

7. **Simo Särkkä and Arno Solin.** Applied Stochastic Differential Equations. Cambridge University Press, 2019. Verified publisher DOI [10.1017/9781108186735](https://doi.org/10.1017/9781108186735). Bibliographic metadata verified. General background for linear time-varying SDE transitions, covariance/Lyapunov equations and OU processes. The self-contained PSD proof above is an application of standard linear-SDE covariance facts.

8. **Jouni Hartikainen and Simo Särkkä.** "Kalman filtering and smoothing solutions to temporal Gaussian process regression models." 2010 IEEE International Workshop on Machine Learning for Signal Processing, pages 379–384, August 2010. Verified DOI [10.1109/MLSP.2010.5589113](https://doi.org/10.1109/MLSP.2010.5589113). Crossref metadata/abstract verified. Establishes the state-space/temporal-GP connection and efficient Kalman solutions for common covariance families.

9. **Simo Särkkä, Arno Solin, and Jouni Hartikainen.** "Spatiotemporal Learning via Infinite-Dimensional Bayesian Filtering and Smoothing: A Look at Gaussian Process Regression Through Kalman Filtering." IEEE Signal Processing Magazine 30(4):51–61, July 2013. Verified DOI [10.1109/MSP.2013.2246292](https://doi.org/10.1109/MSP.2013.2246292). Crossref metadata/abstract verified. Precedent for interpreting spatiotemporal GP models through dynamical state-space systems. It does not imply the current deterministic readout supplies Bayesian filtering or uncertainty.

10. **Viacheslav Borovitskiy, Iskander Azangulov, Alexander Terenin, Peter Mostowsky, Marc Deisenroth, and Nicolas Durrande.** "Matérn Gaussian Processes on Graphs." AISTATS 2021, PMLR 130:2593–2601. Verified primary proceedings [paper and BibTeX](https://proceedings.mlr.press/v130/borovitskiy21a.html). Precedent for SPDE-derived graph covariances and spectral graph GP constructions.

11. **Sina Y. Caliskan and Paulo Tabuada.** "Kron Reduction of Generalized Electrical Networks." [arXiv:1207.0563v2](https://arxiv.org/abs/1207.0563v2), April 1, 2013 revision, original July 2012. **Manish K. Singh, Sairaj Dhople, Florian Dörfler, and Georgios B. Giannakis.** "Time-domain Generalization of Kron Reduction." [arXiv:2203.12084v1](https://arxiv.org/abs/2203.12084v1), March 22, 2022. Metadata/abstracts verified. They concern dynamic electrical-network reductions under their own structural assumptions. The present static response plus newly specified boundary ODE must not be described as inventing dynamic Kron reduction.

12. **Laurent Pagnier, Robin Delabays, and Melvyn Tyloo.** "Nontrivial Kron Reduction for Power Grid Dynamics Modeling." [arXiv:2409.09519v1](https://arxiv.org/abs/2409.09519v1), September 14, 2024. Metadata/abstract verified. It discusses effects of eliminated-node dynamics and Mori–Zwanzig memory, reinforcing the need to separate static equivalence from projected full-tree dynamics.

## Required interpretation at the next experimental stage

Preserve the valid implementation statements and the criticism together. Report same-bank comparisons as conditional tests of a candidate inductive bias. Include gain-matched and static controls, source/parameter/protocol hashes, parent-level split accounting, and copied-checkpoint route probes before attributing gains to useful cross-query history. Address the singleton response policy and breakpoint reflection qualifier. The conditional covariance-kernel interpretation is valid mathematics and a helpful description of the summed context, with direct graph-GP and SDE predecessors. It does not independently establish novelty, learned predictive benefit, calibrated uncertainty, inferred-genealogy robustness, or a PSD similarity between entire graph collections.


## Final verification amendment after root integration

The root incorporated the audit findings while this audit was running. This amendment supersedes the unresolved implementation status above while preserving the counterexamples and original concerns.

The exact singleton-boundary response policy now applies per connected component. Components are identified from strictly negative source-Laplacian off-diagonal entries. Only a component with exactly one boundary has its analytically zero response rows/columns canonicalized, after raw Schur backward checks. Diagnostics retain the pre-canonicalization residual. No magnitude-based clipping of a genuine multi-boundary response is performed. This auditor rechecked the positive and negative B1 roundoff cases, inspected the per-component extension, and ran the final focused regression suite, including disconnected all-singleton and mixed genuine-response tests. The reported degeneracy is resolved for the stated graph contract.

The root also implemented gain_leak_diagonal, the previously optional equilibrium/leakage control. Its independently evolving state is driven toward full cached equilibrium, with rate delta+lambda*rowsum(Cqq). Accepted negative row sums within the explicitly recorded 4*Q*eps*sum(abs(row)) roundoff bound are set to zero. Material negative rows are rejected. This is a declared numerical policy for a theoretical nonnegative leakage, not an arbitrary positive floor. It preserves the full homogeneous generator's action on a spatially constant state, not the whole modal spectrum or the original affine forcing.

The scan now prepares decay, full-interval exp/expm1 factors and forcing projections in batches before recurrence. This auditor inspected the implementation. It applies the same spectral/diagonal affine coefficients, selects full holds in physical order and computes partial target holds in coordinate precision. A single sorted integer metadata transfer replaces per-hold scalar extraction. Tests compare the optimized scan against public validated actions across all four coupling choices. These changes preserve the mathematical kernel/refinement interpretation.

The final focused command above passed **90 tests in 4.03 seconds**, CPU only:
- kron_interface.py SHA256: 09ec6286c185e7aa2b13d76dcbe8fea24b3a1c785a6619ac9bc5464e0de66d69
- boundary_memory.py SHA256: a056cd10d1c848d7dabdc1daf646548876fd90e8f489667c46a374ac28cd11dc
- test_kron_interface.py SHA256: 744b67522cf4a98a83ac8e928ad6caee781cc737ec2317fda378314eaaa1359b
- test_boundary_memory.py SHA256: fb41570b4e6fedd75ebde479c941fa1176cf9b5d3ce2c01b87c1c43ded41aeff

The added breakpoint test makes the local half-open reflection limitation explicit. The root working TeX theorem statements were independently checked. Proposition 1 correctly restricts K to symmetric positive definite piecewise-constant matrices and separates the auxiliary covariance process from zero integration boundary states. Proposition 2 now states exact arithmetic, fixed tile endpoints/interface/query-reference partition and fixed scoring coordinates, followed by a finite-precision tolerance qualification. The noncommuting covariance proof and directional kernel-integral identity are correct. A small wording clarification remains useful: the auxiliary *initial state* has covariance I and is independent of subsequent Brownian increments. Independence is not a property of the deterministic covariance matrix.

No blocking implementation or theorem defect remains in the checked intended regime. Preserve the scientific concerns: full versus original diagonal remains multiply confounded, gain/leak controls still differ in their full modal dynamics, separately trained networks need not retain identical learned forcing, low-precision complete scans require bounded empirical verification, and exact representation guarantees do not prove ancestry performance or architectural novelty. The conditional within-sequence matrix kernel framing has direct graph-SPDE, dynamical-kernel and linear-SDE prior art. Nikitin et al.'s official PMLR second-author metadata is **John, St**, displayed **St John**; arXiv displays **ST John**. Do not invent an expanded name.
