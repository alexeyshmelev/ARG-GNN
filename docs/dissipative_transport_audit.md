# Graph-conditioned dissipative state transport: prior-art audit

Status: proposed operator and mathematical audit, 2026-10-10. No implementation or benchmark result is asserted. Source versions and hashes are retained in `docs/operator_audit_sources.json`. The primary agent owns any implementation decision.

## Assessment and proposed narrow hypothesis

The construction combines established stable-system parameterizations, graph encoders, input-conditioned state transitions and exact affine hold evolution. Matrix exponentials, the semigroup law, dissipativity and order-dependent noncommuting products are not new contributions. A potentially useful hypothesis is that **graph-conditioned mixing between latent state coordinates improves fixed-coordinate inference on piecewise graph signals while preserving a specified interval-subdivision contract**. Its merit remains empirical.

The proposed state for each persistent sample obeys

\[
\dot s(t)=A_gs(t)+b_g,
\quad t\in[a_g,b_g),
\]

with graph-local conditioning h_g and

\[
A_g=-D_g-U_gU_g^T+J_g,
\quad D_g=\operatorname{diag}(\operatorname{softplus}(d_g)),
\quad J_g=V_gW_g^T-W_gV_g^T.
\]

Use another symbol such as the right endpoint r_g in implementation to distinguish an interval endpoint from forcing b_g. Anonymous internal nodes can contribute to the local GNN while the transported state belongs only to persistent sample keys. If internal-node states are also transported, their correspondence/reset semantics must be separately specified.

## Closest verified references

| Reference | Equation-level overlap | Meaningful distinction to investigate |
|---|---|---|
| Roth, Klein, Kannapinn, Peters and Weeger, [Stable Port-Hamiltonian Neural Networks](https://arxiv.org/abs/2502.02480v1), 2025, Eqs. 3–5 and Section 3 | `xdot=(J(x)-R(x)) grad H(x)+G(x)u`, with skew J and symmetric positive-semidefinite R. Neural outputs are constrained to skew J and a Cholesky-based R. Energy dissipation is explicit | With quadratic `H(s)=0.5||s||^2`, the proposal is exactly a J-R linear-energy form. It conditions J and R on an external graph encoding rather than the evolving state. Do not claim a newly invented dissipative parameterization |
| Chang, Chen, Haber and Chi, [AntisymmetricRNN](https://arxiv.org/abs/1902.09689), ICLR 2019, Eqs. 10–13 | Hidden dynamics use `W-W^T-gamma I`, with damping and optional input gates | Stable/skew recurrent state mixing predates the proposal. Its nonlinear Euler update is not an exact affine hold flow |
| Erichson, Azencot, Queiruga, Hodgkinson and Mahoney, [Lipschitz Recurrent Neural Networks](https://arxiv.org/abs/2006.12070), ICLR 2021, Eqs. 1–4 and Theorem 1 | Adds a linear hidden term to a Lipschitz nonlinear recurrent term. Eq. 2 explicitly combines symmetric and skew matrix parts with damping | Stability-constrained recurrent dynamics and symmetric/skew decomposition are established. Its stability conditions differ from the proposal's direct negative symmetric part |
| Hasani et al., [Liquid Structural State-Space Models](https://arxiv.org/abs/2209.12951), 2022 preprint, Eq. 8 | Linearized liquid dynamics have `xdot=[A+B u(t)]x+B u(t)`, described as coupled bilinear input-dependent transitions. Eq. 9 uses bilinear/trapezoidal discretization | Input-dependent continuous state generators and multiplicative interactions are established. The proposal uses graph conditioning, a dissipative matrix constraint and exact affine hold evolution. The Liquid-S4 equation's elementwise/broadcast input convention should not be silently equated with an arbitrary full input-conditioned matrix |
| Krause, Lu, Murray and Renals, [Multiplicative LSTM for Sequence Modelling](https://arxiv.org/abs/1609.07959), Sections 1.1–1.2, Eqs. 6–10, describing Sutskever et al.'s 2011 mRNN | A tensor RNN uses `W(x)=sum_n W_n x_n`. A multiplicative RNN factorizes `W(x)=W_hm diag(W_mx x) W_mh` | Different input-conditioned transition matrices and potentially noncommuting matrix products are old recurrent-model mechanisms. They need not obey continuous-time hold consistency |
| Kidger, Morrill, Foster and Lyons, [Neural Controlled Differential Equations](https://arxiv.org/abs/2005.08926), NeurIPS 2020, Eq. 2 | `z(t)=z(0)+integral f(z(s)) dX(s)` incorporates incoming continuous data through matrix-valued vector fields | The proposed input-conditioned affine dynamics are a structured CDE/controlled ODE subclass. A finite driver can integrate the entries of A and b and use linear coordinate vector fields. Do not claim to exceed generic full-matrix CDE expressivity merely by using noncommuting generators |
| Choi, Choi, Hwang and Park, [STG-NCDE](https://arxiv.org/abs/2112.03558), AAAI 2022, Eqs. 4–7 and 12 | Temporal and spatial CDEs combine as `Zdot=g(Z) f(H) Xdot`, with graph-aware spatial processing and an augmented ODE | Graph-conditioned continuous recurrent processing is already established. Its native traffic setting has fixed vertices/topology, whereas ARG graphs need explicit partial correspondence |
| Li et al., [GraphSSM](https://arxiv.org/abs/2406.00943v2), NeurIPS 2024, Eqs. 5, 6 and 8, Appendix B.2 | Before diagonalization, Eq. 5 has a full HiPPO state matrix with graph-diffused forcing. The system is piecewise LTI on unchanged-graph intervals. Eq. 8 has exact ZOH evolution, and B.2 gives changing-vertex alignment | The proposal generates a full state matrix from graph content rather than retaining a fixed HiPPO/diagonal matrix with graph-conditioned forcing. This is a concrete difference from this comparator, not proof of general architectural novelty |
| Sun et al., [RetNet](https://arxiv.org/abs/2307.08621v4), Eqs. 1–6, and Dao/Gu, [SSD](https://arxiv.org/abs/2405.21060v1) | RetNet begins with a full state matrix before diagonalizing a fixed transition. SSD connects attention and structured state transitions | Full matrix state equations and attention/state-space connections have direct prior art. Varying A across graph holds is the relevant narrower modeling choice |

A newly posted related paper, Heilig et al., [Port-Hamiltonian Neural Networks for Systems with Multiple Asymptotically Stable Equilibria](https://arxiv.org/abs/2610.01356v1), was available on 2026-10-01. It extends Hamiltonian parameterization to multiple attractors. It reinforces that stability-by-construction in port-Hamiltonian neural models is an active established line. Its multiple-equilibrium result is not needed for the affine proposal and is not inherited by it.

## Exact affine evolution and the actual subdivision claim

Define the augmented generator and state

\[
M_g=\begin{pmatrix}A_g&b_g\\0&0\end{pmatrix},
\qquad \widetilde s=\begin{pmatrix}s\\1\end{pmatrix}.
\]

For an unchanged hold of width delta,

\[
\widetilde s(t+\delta)=e^{\delta M_g}\widetilde s(t).
\]

This avoids inverting A when computing the forcing integral and works for singular matrices. It is the standard block-matrix representation of an affine flow. Equivalently, `s_next=exp(delta A)s + integral_0^delta exp(u A) b du`.

For the same M on two consecutive subintervals,

\[
e^{\delta_2M}e^{\delta_1M}=e^{(\delta_1+\delta_2)M}.
\]

A query inside a hold at x uses `exp((x-a_g) M_g)` from the state at its left boundary. Reading only right-end states is insufficient for exact fixed-coordinate predictions after arbitrary interval splitting. The semigroup property becomes a complete-model subdivision guarantee only if:

1. The graph encoder and generator depend on the observed graph signal, not split widths, midpoints, sequence ordinal or duplicate-record count.
2. Generator A and forcing b are held constant throughout the underlying unchanged graph interval. They must not be recomputed from the evolving recurrent state once per serialized subinterval.
3. Persistent identity alignment preserves the same sample state at artificial split boundaries. Anonymous states must not gain a new reset/event merely because an unchanged graph was duplicated.
4. Queries keep their physical coordinates and use fractional hold evolution.
5. Nonlinear readouts or a target-graph MPNN are evaluated on the same states at the same coordinates. Nonlinear per-record state jumps, record-count normalization, or feeding interval-end states into a second piecewise-constant contextual layer generally violate the contract.
6. Any dropout/randomness is disabled or coupled to the underlying graph signal rather than independently sampled per duplicate.

It is reasonable to test this model contract at finite precision. Call it exact in real arithmetic under the specified constant-hold assumptions, with numerical tolerances reported for the implementation. Do not claim arbitrary neural CDE trajectories are evaluated exactly by one matrix exponential.

## What dissipativity actually proves

The symmetric part of A is

\[
\frac{A_g+A_g^T}{2}=-D_g-U_gU_g^T.
\]

Consider two states driven by the **same observed graph signal and forcing**. Their difference e obeys `edot=A_g e`, giving

\[
\frac{d}{dt}\|e\|_2^2=-2e^TD_ge-2\|U_g^Te\|_2^2\le0.
\]

Thus each exact homogeneous transition is non-expansive in the Euclidean norm. If `D_g >= epsilon I` for one uniform positive epsilon, arbitrary switching between these generators contracts at least as `exp(-epsilon t)`. Positive softplus values alone do not provide a uniform lower bound over all possible graph inputs. Adding a predeclared epsilon is a useful parameterization choice if a uniform theorem is desired.

This argument concerns initial-state perturbations under identical inputs. It does not imply that the absolute forced state norm never grows. With bounded forcing `||b_g||<=B` and uniform epsilon, a standard bound is `||s(t)|| <= exp(-epsilon t)||s(0)|| + B(1-exp(-epsilon t))/epsilon`. Without a bound on b, absolute states can grow. It also does not imply all parameter/input gradients are bounded, because A and b depend on a learned encoder. Changing graph inputs changes the vector field itself.

The augmented M has a constant-coordinate zero eigenvalue and is not strictly dissipative as a whole. Apply the contraction statement to differences in s, where the two constant coordinates cancel. If A depends on the recurrent state, derivative terms from that dependence invalidate this simple contraction proof. A following unconstrained nonlinear target MPNN may amplify perturbations, so the state-transport proof should not be advertised as a full-network Lipschitz bound.

## Noncommutativity and order: necessary correction

For two graph holds, their affine generators have commutator

\[
[M_1,M_2]=
\begin{pmatrix}
[A_1,A_2]&A_1b_2-A_2b_1\\0&0
\end{pmatrix}.
\]

The products `exp(delta2 M2) exp(delta1 M1)` and the reversed product can differ. For sufficiently small widths, their leading difference is proportional to `delta1 delta2 [M2,M1]`. Noncommuting A matrices enable additional graph-conditioned mixing between latent state coordinates. This is a standard time-ordered-flow mechanism.

**Diagonal SSMs are already order-sensitive.** Even if all diagonal A matrices commute, the affine forcing term can make the augmented generators fail to commute. For a scalar example with A=-1, two unit-width holds, initial state zero, and forcing values 1 then 0, the final state is `exp(-1)(1-exp(-1))`. Reversing the forcing gives `1-exp(-1)`. The inputs have the same multiset but different results. Selective forcing, gates and nonlinear stacked layers provide further order dependence.

Therefore the supported distinction is **input-conditioned cross-coordinate state mixing**, not the ability to encode order. Nonzero commutators also do not guarantee every input or readout distinguishes every ordering. A zero initial state with zero forcing remains zero, and a readout can project out the changed direction. A two-dimensional skew matrix is a scalar multiple of one rotation generator, so purely skew generators in two dimensions commute if damping is a shared scalar. Varying anisotropic damping or using larger state dimension can introduce noncommutativity, but the property should be measured rather than assumed from the model name.

For a useful mechanistic intervention, remove the skew term, compare input-dependent diagonal transitions against full dissipative transitions, and separately restrict all A_g to a shared eigenbasis so their homogeneous generators commute. Keep forcing/readout capacity comparable. An order-shuffle result alone cannot identify noncommutativity as its cause.

## Reverse context, correspondence and compute

A bidirectional offline model should integrate a second stable system in the reverse coordinate `r=1-t`, using **positive** hold lengths and the same dissipative sign convention. It should not invert the forward flow by `exp(-delta A)`, which reverses damping and can be expansive. It may share parameters or use independent reverse parameters, but this choice and added compute must be reported. For a one-week-ahead forecast, both directions may inspect only the already observed history window, not the future target or a full-series state bank.

Stable sample IDs are correspondence keys. Anonymous internal ancestors with ID -1 must not be merged into a persistent reservoir. Their observed features can influence each sample's local conditioning through the GNN. A sample disappearing and reappearing needs an explicit reset/resurrection policy consistent with the data contract. Artificial subdivision of an identical interval must not create an identity event.

Let L be the number of transported samples, G graph holds, d state width, r the low-rank generator factor width and M fixed leaf-coordinate queries. A straightforward exact implementation computes batched exponentials of `(d+1) by (d+1)` matrices, costing approximately `O(G L d^3)` for transition construction, `O(G L d^2)` for application, plus partial-query exponentials up to `O(M d^3)`. Local graph encoding is additional. Forming the low-rank parameterization does not make its **matrix exponential** low-rank or linear in d. Dense exponentials and their backward pass can dominate runtime even when asymptotic cost is linear in graph count at fixed d and L. Ordinary autograd storage can include `O(G L d^2)` saved transitions. Report actual memory/runtime rather than promising Mamba/GraphSSM-like throughput.

## Recommendation

A small state dimension is a defensible implementation experiment if the parent judges this hypothesis valuable. Begin with exact affine holds and fixed-coordinate readout in one layer, with no per-record nonlinear state jumps. Validate the full subdivision contract, storage permutation, identity relabeling, finite gradients, same-input contraction and the diagonal-forcing counterexample on CPU fixtures before any benchmark allocation.

Compare against: fixed-A exact ZOH transport, input-conditioned diagonal exact ZOH, symmetric-only dissipative full matrices, skew-plus-damping with U=0, the full proposed generator, and the documented GraphSSM adaptation. Match widths/parameters and input features as far as possible, expose the extra compute, and test changing internal-node counts separately. Add a generic graph-conditioned CDE or controlled-ODE comparison before claiming broad superiority to continuous graph models.

Proceed only as a **graph/interval hypothesis test**. The strongest likely defensible contribution is a well-specified correspondence-aware interval architecture with measured advantages and limitations. The matrix exponential, J-R construction, semigroup law and noncommuting order mechanism remain established tools.
