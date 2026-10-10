# H019: low-rank whole-tree Markov interface

**Untested candidate proposal. No implementation, numerical test, training or superiority result.** The algebra permits positive, contracting query-history transport without a dense query matrix. The operator is standard normalized bipartite/anchor diffusion. A possible contribution would require a concrete biological constraint and a measured advantage over strong existing operators, rather than a new-kernel or first-graph-SSM claim. The current dense Q64 gains are small and feedback is unsupported; neither motivates a predicted positive result.

## Construction and guarantees

At each observed marginal-tree hold, let a whole-tree encoder produce nonnegative leaf-to-landmark assignments `P∈R^(n×r)`, with `P1=1`. Here n includes every transported unknown leaf Q and every fixed reference source R. Use an actual tree-wide upward/downward message sweep or explicitly global tree conditioning; running two local GNN layers on the full tree does not give whole-tree receptive fields.

Define
\[
d=P^T1,\quad D=\operatorname{diag}(d),\quad
B=PD^{-1/2},\quad S=BB^T,\quad C=I-S.
\]
All active columns require `d_a>0`. Then S is entrywise nonnegative, symmetric and PSD, and `S1=PD^{-1}d=1`. Jensen's inequality gives
\[
v^TP^TPv=\sum_i(P_iv)^2\le\sum_{ia}P_{ia}v_a^2=v^TDv,
\]
hence `B^TB≼I`, `0≼S≼I` and `0≼C≼I`. C has zero row sum and nonpositive off-diagonal entries. Equivalently,
\[
x^TCx=\min_z\sum_{ia}P_{ia}(x_i-z_a)^2.
\]
Thus C is exactly the Schur complement of the **learned bipartite** Laplacian
\[
\begin{pmatrix}I&-P\\-P^T&D\end{pmatrix}.
\]
It is not automatically the electrical/age Kron response of the original observed genealogy, nor an exact projection of that genealogy's heat dynamics. Calling it Nyström is appropriate only after specifying a kernel/cross-block approximation; a learned anchor graph is the more precise description here.

Partition B into Q and R rows. With scalar nonnegative rates λ,δ,
\[
A=(\lambda+\delta)I-\lambda B_QB_Q^T,\qquad
\dot h_Q=-Ah_Q+f,\quad
f=\lambda B_QB_R^Ty_R+\delta u_Q.
\]
The reference values y_R are admissible observed-source signals; u_Q is a graph-conditioned prior/source independent of query truth and evolving memory. Let `a=λ+δ`. A is symmetric, has nonpositive off-diagonals and spectrum in `[δ,a]`. Therefore `exp(-τA)` is entrywise nonnegative and has Euclidean norm at most `exp(-δτ)`.

Positive δ ensures a nonsingular M-matrix. At δ=0 and λ>0, strict grounding requires every query communicating component to reach a reference, equivalently `ρ(B_QB_Q^T)<1`; PSD alone does not guarantee an invertible grounded generator. If every query row leaks at least η>0 to R, the smallest eigenvalue is at least `δ+λη`. Uniform damping bounds give contraction across arbitrary holds for perturbations under the **same inputs and forcing**, not a full-network/input-gradient robustness guarantee.

If h_Q, y_R and u_Q are class distributions, then `f1=A1`, so positivity and unit class mass are preserved. This requires the full leaf partition Q∪R: omitted unknown leaves cannot become unforced grounded sinks. One-hot reference values are justified only by the target/source definition; observed cohort names are not automatically perfect per-locus path-ancestry labels. Arbitrary hidden features retain the linear positivity/contraction properties of the transport operator but are not class distributions.

## Exact rank-r action and gradients

For a hold width τ≥0 in a declared physical or genetic coordinate, set `G=B_Q^TB_Q∈R^(r×r)`. Its eigenvalues lie in [0,1]. Define
\[
\psi_\tau(z)=
\frac{e^{-(a-\lambda z)\tau}-e^{-a\tau}}{z},\qquad
\psi_\tau(0)=\lambda\tau e^{-a\tau}.
\]
Then
\[
E_\tau=e^{-\tau A}=e^{-a\tau}I+B_Q\psi_\tau(G)B_Q^T.
\]
E is generally full rank: its correction to scalar decay has rank at most r. This form uses nonpositive exponents and avoids an overflowing intermediate `exp(λτG)`.

With `gτ(x)=(1-exp(-τx))/x`, `gτ(0)=τ`,
\[
F_\tau=\int_0^\tau e^{-sA}\,ds
=g_\tau(a)I+B_Q\theta_\tau(G)B_Q^T,
\quad
\theta_\tau(z)=\frac{g_\tau(a-\lambda z)-g_\tau(a)}{z}.
\]
Use the analytic limit `θτ(0)=-λgτ'(a)`; for a>0 it is
\[
\lambda[1-(1+a\tau)e^{-a\tau}]/a^2.
\]
The exact update is `h_next=Eτ h+Fτ f`, valid even when A is singular. Reference forcing simplifies to
\[
F_\tau\lambda B_QB_R^Ty_R
=\lambda B_Qg_\tau(aI_r-\lambda G)(B_R^Ty_R).
\]
No Q×R forcing matrix or Q×Q transition need be formed. Fixed-coordinate outputs use fractional hold widths. Identical-hold subdivision obeys the affine semigroup only when P, rates and sources are unchanged; no per-record feedback, width-conditioned encoder, fresh dropout or nonlinear state jump is included in this proposal.

All matrix functions above are analytic at zero/repeated eigenvalues. A future implementation should use stable matrix-function/divided-difference or block-exponential actions, not divide by tiny eigenvalues or rely on naive eigenvector gradients. The Fréchet derivative is
\[
DE_\tau[\Delta A]=-\int_0^\tau
e^{-(\tau-s)A}\Delta A e^{-sA}\,ds,
\]
where
\[
\Delta A=(\Delta\lambda+\Delta\delta)I-\Delta\lambda B_QB_Q^T
-\lambda(\Delta B_QB_Q^T+B_Q\Delta B_Q^T),
\quad
\Delta B_{ia}=\Delta P_{ia}/\sqrt{d_a}
-\tfrac12P_{ia}d_a^{-3/2}\sum_j\Delta P_{ja}.
\]
These give gradients through assignments, global column mass, rates and forcing. Differentiation remains ill-conditioned when landmark mass approaches zero; contraction does not cure that. Empty columns must be removed or prevented by a declared stochastic parameterization. Clamping D with an epsilon changes row sums and forcing identities. Hard biological masks must leave each row supported; any smoothing must respect the masks and retain stochasticity.

## Cost and actual limitations

Let V,E_tree be tree nodes/edges, q=|Q|, k=|R|, h feature width and T holds.

| Work per hold | Indicative cost, excluding constants |
| --- | --- |
| Whole-tree encoder + landmark scores | Encoder-specific; a two-sweep dense message cell is O(Vh²), plus O(Vrh) pooling and O(nrh) assignments |
| Gram and small matrix functions | O(qr²+r³) |
| Transport/source action for h channels | O((q+k)rh+r²h) |
| Factor storage | O(nr+r²), with O(qh) persistent memory |
| Dense cached interface action | O(q²h), dense geometry/eigensystem storage O(n²+q²); generic dense preparation O(n³), with tree-specific elimination potentially cheaper |

Learned P and its r×r functions are recomputed during optimization; the current fixed dense geometry can cache its eigensystem. Whole-tree backpropagation and saved states can dominate storage: an ordinary scan may retain O(T(nr+r²+qh)) plus encoder activations. These are asymptotic accounts, not measured speed/memory claims. At q=64, overhead can erase the advantage; r near q loses it.

Persistent memory is q×h, not r×h. Changing column spans leave history in directions outside the current span, which future trees can reuse. Products of successive scalar-plus-rank-r transitions grow correction rank, so a constant-r parallel scan is not automatic. No nonlinear memory-to-P feedback is proposed. Column permutations within a hold leave S/forcing unchanged, while changing the actual leaf panel changes d and the operator; normalization must use the fixed full observed panel, not the current minibatch/readout subset.

## Biological hypothesis and prior-art boundary

A generic GNN softmax to r learned tokens is insufficient differentiation. The concrete candidate is an **observed-age-constrained ancestral interface**:

- Each landmark represents an observed ancestral branch/clade or declared age band associated with reference lineages. Assignment support follows observed ancestry paths; landmark times cannot precede assigned tip/reference sample ages.
- Only observed tree times/topology, known reference metadata and genotype/inference-derived correspondence are allowed. No true census ancestor, internal population, query ancestry or target tract enters P.
- Corresponding reference lineages/shared ancestral clades constrain neighboring-tree landmark support across genomic breakpoints. This is a testable structural regularizer, not the assumption that ancestor IDs persist everywhere or encode ancestry as learned tokens.
- Any unary/root-cap invariance needs canonical observed structure or compositional edge messages. A generic nonlinear tree encoder does not inherit the dense electrical invariances. Inferred-age/topology errors and ancient-source admixture must be part of the declared robustness conditions.

Whether this adds useful biology rather than redundant constraints remains open. The constraint, reference-source definition and target-class mapping require a separate biological protocol before implementation.

[Zhou, Huang and Schölkopf, NIPS 2006](https://papers.nips.cc/paper_files/paper/2006/hash/dff8e9c2ac33381546d96deea9922999-Abstract.html), Sections 4–5, defines normalized hypergraph walks and `I-Dv^-1/2 H W De^-1 H^T Dv^-1/2`; the present soft-incidence formula is the corresponding row-normalized bipartite construction. Its transductive label diffusion is an additional direct precedent. [DiffPool](https://arxiv.org/abs/1806.08804) already learns GNN soft assignments. Anchor-graph SSL, Nyström approximation and diagonal-plus-low-rank matrix functions are further prior-art threats; the attempted anchor-graph primary retrieval was unsuccessful, so that literature check remains incomplete.

[GST](https://arxiv.org/abs/2606.05116) already interleaves graph/set processing and feedback. [GraphSSM](https://arxiv.org/abs/2406.00943) supplies graph-conditioned forcing, continuous holds, ZOH and changing-node alignment; interleaved graph layers can exchange histories. H019's proposed node-axis positive low-rank generator is a specific constraint, not evidence of a first history-exchange mechanism. Ordered products and semigroups are standard; diagonal affine systems are already order-sensitive.

Before any performance claim, the proposed comparisons are: identical whole-tree encoder with generic anchor diffusion versus biological masks; fixed versus learned factors; static/local and uncoupled/diagonal controls preserving source gain; span-set versus ordered flow; dense electrical/age interface; matched GST and GraphSSM; and the genuine author biological baselines under their stated task scopes. Report rank, parameters, encoder/preparation/backward cost, source panels, matched tuning, parent-level NLL/calibration/dose/tract metrics and failures. The existing ledger is untouched; root may register H019 from the matching proposal JSON if it merits a later experiment. **No execution is authorized by these files.**
