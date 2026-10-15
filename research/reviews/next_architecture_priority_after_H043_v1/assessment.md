# Exact tree-conditioned Markov semigroup across a graph collection: source-only assessment

Date: 2026-10-15. Scope: mathematical assessment, finalized/public sources, and literature only. No data, targets, new outcome metrics, runtime/process/GPU inspection, project imports, scientific execution, admission, scheduling, or Git operations. This note does not authorize implementation or a run. All notes and retrieval receipts are confined to this review directory. The C3 capacity closure and previous audit closures were not changed.

## Decision

**The proposed operator and the DFS prefix/range-add action are mathematically valid under the stated assumptions. No sufficient new-operator or general graph-collection novelty is established.** The closest mathematical prior is exact, rather than merely analogous: Bendikov and Krupski define the same hierarchical Laplacian and the same cumulative ancestor-rate spectrum in 2013/2014. A GNN choosing nonnegative coefficients, followed by exact heat evolution, is a useful restricted neural architecture; exact learned diffusion and efficient hierarchy-based neural spectral filtering also have strong precedents.

The defensible immediate hypothesis is narrow: an exactly integrable, leaf-preserving hierarchical diffusion may be a practical inductive bias for ordered marginal-tree observations. That is an architectural/application hypothesis, with no accuracy or publication-priority promise. A substantive general contribution would still require a result or capability beyond composing known hierarchical heat operators in genomic order. The current construction does not compress arbitrary persistent leaf states across all trees and does not furnish a single pass linear in the shared ARG size.

## Candidate and mathematical check

Retain the same H observed leaves in a persistent order at every locus. Let T be one rooted tree whose nonempty clades are nested or disjoint. For each internal clade C, E_C replaces values inside C by their uniform mean and is the identity outside C. Let

    Q_T = sum_C r_C (E_C - I),     r_C >= 0.

Rates are functions of fixed current-tree observations, independent of the propagated history h. They are constant over a physical interval of length d >= 0. Units of r are inverse physical distance in the chosen coordinate; they are not ancestry generations without an additional biological model.

E_C is a symmetric orthogonal projector. Its off-diagonal entries within C are 1/|C|, and E_C-I has zero row sums, nonnegative off-diagonal entries, and nonpositive spectrum. Nested/disjoint projectors commute. In particular, when C is contained in D, E_C E_D = E_D E_C = E_D. Thus Q_T is a symmetric continuous-time Markov-chain generator. Its heat kernel P_T(d)=exp(d Q_T) has nonnegative entries, is symmetric and doubly stochastic, preserves the leaf mean, and contracts centered Euclidean energy. Uniform leaf measure is stationary; uniqueness requires irreducibility and must not be asserted merely from the construction. An excluded root can be represented by rate zero; the process can then be reducible.

Write mu_u for the mean of the input h on the descendant leaves of node u. Define

    lambda_u = sum_{C ancestor-or-self of u} r_C,
    w_u = exp(-d lambda_u)                     (internal u).

For a leaf ell,

    [P_T(d)h]_ell = mu_root
                   + sum_{edges u -> v on root-to-ell path}
                         w_u (mu_v - mu_u).

The multiplier belongs to the **parent u**, not the child v. A child-parent contrast is a tree martingale difference with eigenvalue -lambda_u: strict-descendant averaging fixes it, while u and its ancestors annihilate its mean. Expanding h into these differences proves the formula. This works with polytomies if subtree means are weighted by leaf count; binary structure is unnecessary. Singleton generators vanish. For H=1 the operator is the identity and should be handled directly.

The parent's alternative exact expansion is also correct:

    P_T(d)h = diag(a_leaf) h
              + sum_internal C a_C 1_C mu_C,

    a_root = 1 - w_root,
    a_C = w_parent(C) - w_C                    (nonroot internal C),
    a_ell = w_parent(ell)                      (leaf ell).

All coefficients are nonnegative because lambda increases along a root-to-leaf path. For each leaf they telescope to one. This expansion also exposes symmetry: every clade contribution is a_C 1_C 1_C^T / |C|, plus a diagonal term. Stable evaluation is

    a_root = -expm1(-d r_root),
    a_C = w_parent(C) * [-expm1(-d r_C)].

This avoids subtracting nearly equal exponentials. d=0 returns identity. Zero rates are allowed. Duplicate unary clades are harmless only if their total rate is preserved; inserting a unary node into an arbitrary GNN changes observations and predicted rates, so whole-model unary invariance does not follow. A forest can be covered by an artificial zero-rate root, preserving separate components; it must still retain the fixed H-leaf interface.

## Efficient Torch/PyG outline, without implementation or speed claims

A fixed tree's DFS leaf permutation makes each clade contiguous, [left_C,right_C). Precompute these leaf intervals, clade sizes, parent indices, and inverse leaf permutations. A separate preorder over all tree nodes also gives a contiguous node-subtree interval for every ancestor.

1. Run the existing/current-tree GNN to encode observations and predict scalar nonnegative r_C, for example with a positive output map. Encode rates once for a fixed observed graph collection, rather than recomputing them from h at every propagation substep. Predicting a rate from h would invalidate this exact fixed-generator action.
2. Compute cumulative lambda for every internal node differentiably. Range-add each r_C over C's interval in the all-node preorder, then cumsum; every node receives the rates of exactly its ancestors and itself. This avoids a Python loop over tree depths. Gather parent lambdas and evaluate w and stable a.
3. Gather h into the tree's DFS leaf order. A length-H prefix sum with an initial zero gives all clade sums with two indexed gathers; divide by right_C-left_C to obtain mu_C.
4. Form a_C mu_C. Scatter-add this vector at left_C and its negative at right_C into a zero difference array of length H+1. Its cumsum at the first H positions adds each clade contribution to precisely its leaves. Add a_leaf times h and scatter back to the persistent order.
5. Process physical intervals in their chosen chronology. Multiple independent sequences can be batched, while each sequence retains its interval recurrence. PyG supplies the encoder/batching; ordinary Torch gathers, scatter-adds, cumsums, exp/expm1 implement the action. No dense H-by-H matrix, ancestor-membership matrix, eigendecomposition, or enumerated root-to-leaf path expansion is required.

With N tree nodes and D channels, this is O((H+N)D) arithmetic for the action and O((H+N)D) working tensors; rates/cumulative lambdas add O(N). It remains linear in the materialized tree size even for unbalanced trees, not in an assumed balanced-tree depth. For T intervals the forward arithmetic is O(D sum_i(H+N_i)), in addition to the encoder, with T sequential interval actions. Storing all recurrent activations for training can cost the corresponding sum, and checkpointing/recomputation changes that tradeoff. CUDA kernel launch overhead, scan efficiency, scatter contention, permutation traffic, autograd storage, and mixed-precision cancellation remain empirical questions. There is no measured speed or fit claim here. Floating-point scatter/scan is not an exact-real-arithmetic equality; normal tolerances and gradient checks would be needed in an authorized implementation.

The prefix/difference expression is a useful implementation derivation. I did not verify a prior containing this identical GPU kernel sequence. That absence is not a priority result: the exact hierarchical eigendecomposition and fast tree/Haar transforms are already known, and linear-time bottom-up/top-down action follows naturally from tree means and contrasts.

## What ordered graph collections add, and what they do not

For column states, the interval endpoint state is

    h_m = exp(d_m Q_m) ... exp(d_1 Q_1) h_0.

This is the exact fundamental solution of a linear system with piecewise-constant generator. Different tree hierarchies can yield noncommuting Q_i, so order can matter. The leading discrepancy from the order-blind integrated generator is governed by commutators, as in classical time-ordered exponentials/Baker-Campbell-Hausdorff/Magnus theory. This fact is not a new graph-collection expressivity theorem. All Q_i share uniform stationarity, so even an ordered product remains doubly stochastic, preserves the global mean, and contracts centered energy. If h_0 is constant over leaves, the unforced model stays constant. If the readout only uses the global mean, it discards the entire propagation effect.

The root averaging component commutes with every clade generator and cannot by itself encode tree order. Noncommutativity comes from incompatible nontrivial clades across trees. For example, with three leaves, a positive rate on {1,2} in one tree and on {2,3} in another gives noncommuting generators. Very weak diffusion makes order effects second order and potentially tiny; strong common mixing can erase useful contrasts. Neither regime supplies an accuracy prediction. Symmetry of each factor does not give symmetry of the product: reversal transposes the ordered product. A genomic model needs an explicit choice about orientation or bidirectional readout; whole-model reversal invariance is not automatic.

A useful graph-collection contribution would need something more substantial, such as an exact incremental algorithm that exploits shared tree edits while retaining the unrestricted persistent leaf state, with proved total complexity beyond expanded marginal trees; or a rigorously specified task/representation separation from suitable existing neural dynamical models together with matched evidence. Merely learning the hierarchical coefficients, observing a nonzero commutator, or writing the ordered product does not establish that contribution. No such additional result has been supplied or proved here.

This is distinct from the earlier compressed copying-HMM proposal: retaining all H leaves avoids unjustified donor-belief lumping. It is also distinct from H044's SNP/reference-frequency boxes and original-100bp readout. That application experiment should not be relabeled as evidence for this operator's priority.

## Exact refinement: conditional guarantee only

For fixed Q, exp((d1+d2)Q)=exp(d2 Q)exp(d1 Q). Thus redundant subdivision of an interval leaves the propagated endpoint unchanged **only when the generator, observations, coordinate, state/interface, and any forcing remain identical**. GNN inputs using stored row spans, row endpoints, artificial-boundary indicators, row-specific normalizations or adjacency, or state-dependent rates can change under subdivision. Encoder dropout independently resampled after splitting also breaks samplewise equality. Encoding a canonical physical observation once is one way to satisfy the premise. A per-row reset or nonlinear update introduces an extra operation every time the row is split and defeats the guarantee. This note makes no claim that the existing whole encoder automatically satisfies these conditions.

Current-tree generator locality is different from output locality: a carried state intentionally contains evidence from earlier loci. It must be described as genomic memory, rather than as a prediction depending only on the present tree. Exact propagation on a constant tree can vary a state smoothly with physical distance, but it introduces no new SNP evidence inside that hold. SNP-resolved decisions still need explicit site evidence, physically specified forcing/events, or a site readout. The supplied H042 boundary weakness motivates investigation but does not identify missing ordered diffusion as its cause. H043 at 18/30 passes with no supplied VAL cannot support a retrospective conclusion about its outcome.

Optional constant forcing h'=Qh+b has exact action

    h(d) = P_T(d) h(0) + G_d(Q)b,
    G_d(Q) = integral_0^d exp(sQ) ds.

Use the same tree contrast decomposition with multiplier

    g_d(lambda) = -expm1(-d lambda)/lambda,    lambda > 0,
    g_d(0) = d.

The zero mode therefore accumulates d times the global mean of b. Omitting it loses the forcing or creates division by zero; unconstrained repeated forcing can drift. The same prefix/range-add grouping works with root coefficient d-g_d(lambda_root), internal coefficient g_d(lambda_parent)-g_d(lambda_C), and leaf coefficient g_d(lambda_parent). Exact affine split consistency holds for fixed Q and b. Leaf-specific reaction/damping generally destroys the commuting hierarchy; it must not be added while retaining this diagonalization claim. No forcing extension is required to establish the current mathematical verdict.

## Closest verified priors and precise overlap

1. **Bendikov, A. and Krupski, P. (2014), On the spectrum of the hierarchical Laplacian**, Potential Analysis 41, 1247–1266. [Primary arXiv v2, 2013](https://arxiv.org/abs/1308.4883v2); [DOI](https://doi.org/10.1007/s11118-014-9409-6). The full primary source defines

       L_C f(x) = -sum_{B containing x} C(B) [P_B f - f(x)],
       P_B f = (1/m(B)) integral_B f dm,
       lambda(B') = sum_{T containing B'} C(T),

   with child-parent contrast eigenfunctions 1_B/m(B)-1_B'/m(B'). Taking counting measure and C(B)=the proposed r_B gives **exactly L_C=-Q_T**, including the cumulative-rate spectrum used by the action. Its positive choice-function setting extends to zero coefficients in the finite case by the elementary formula/limits. Introduction equations `hlaplacian`, `eigenfunction`, `eigenvalue`, and Section 3 are the decisive passages. This is not just prior work on a different generic graph Laplacian.

2. **Bendikov, Grigor'yan, Pittet and Woess, Isotropic Markov semigroups on ultra-metric spaces**. [Primary arXiv:1304.6271v3](https://arxiv.org/abs/1304.6271v3), revised 2014; journal metadata on the primary record: Uspekhi Mat. Nauk 69 (2014), no.4,3–102, [DOI](https://doi.org/10.4213/rm9602). Constructs symmetric Markov semigroups on ultrametric measure spaces and studies their generators, heat kernels, and pure-point spectra; relates these to tree-boundary jump processes. Full HTML retained. Supports the established hierarchical heat/averaging interpretation.

3. **Chamberlain et al. (2021), GRAND: Graph Neural Diffusion**. [Primary arXiv:2106.10934v2](https://arxiv.org/abs/2106.10934v2), full HTML retained. Section 3.1 explicitly gives the fixed-attention linear case x(t)=exp(t Abar)x(0); Section 3.4 and the linear model discuss time-independent learned attention. Thus learned coefficients, fixed-generator diffusion, analytic matrix-exponential interpretation, and diffusion-based neural layers are existing ideas. GRAND does not establish the same clade hierarchy or the proposed exact O(ND) tree action. The precise distinction is the restricted hierarchical operator family and execution, rather than learned diffusion itself.

4. **Li, M., Ma, Z., Wang, Y. G. and Zhuang, X. (2020), Fast Haar Transforms for Graph Neural Networks**, Neural Networks; [DOI](https://doi.org/10.1016/j.neunet.2020.04.028), [verified primary arXiv:1907.04786](https://arxiv.org/abs/1907.04786), full HTML retained. Uses sparse/localized hierarchical Haar bases, fast transforms, and neural spectral filtering. Section 4.3 states O(N) for the adjoint transform and O(N(log N)^2) for forward transform/convolution in its construction; do not quote both as unconditional O(N). This is a close efficient-neural-hierarchy prior, but not verified as the identical Markov kernel, rates, or prefix/range-add realization. Haar Graph Pooling, [arXiv:1909.11580](https://arxiv.org/abs/1909.11580), ICML 2020, is additional hierarchy/neural pooling context, not the same leaf-preserving semigroup.

5. **Gasteiger (published as Klicpera), Weißenberger and Günnemann (2019), Diffusion Improves Graph Learning**, NeurIPS 2019. [Primary arXiv:1911.05485](https://arxiv.org/abs/1911.05485). GDC includes the heat kernel as generalized diffusion for graph learning. This is broader prior and less exact than items 1–4.

6. **Baake and Schlägel (2011), The Peano–Baker series**, Proc. Steklov Inst. Math.275,167–171, [primary arXiv:1011.1775](https://arxiv.org/abs/1011.1775). A self-contained account of the standard fundamental solution for time-dependent linear systems. Ordered interval products and their distinction from exp(integrated generator) are classical consequences, rather than a new source of mathematical priority.

7. **tskit's tree-compressed Li–Stephens HMM**: current public [haplotype_matching.c](https://raw.githubusercontent.com/tskit-dev/tskit/main/c/tskit/haplotype_matching.c) and header retained from main as retrieved 2026-10-15. Existing detach/graft tree updates, mutation-subtree state changes, probability compression, and forward/backward/Viterbi are prior for genealogy-compressed sequence-state computation, not this generator. Configurable rounding (default precision6) means this source alone must not be cited as exact real-arithmetic compression. This matters if the proposal later turns into a compressed donor-state/HMM architecture; it does not negate the present all-leaf algebra.

A mistaken initial arXiv identifier 1907.13483 retrieved an unrelated geometry paper; those receipts are explicitly marked excluded. It is not a citation. Unsuccessful FLARE/RFMix full-text/API retrievals and broad search receipts are not used to substantiate the operator verdict. Prior literature review artifacts on neural operators, sheaf diffusion, ARGformer, full-ARG representations, and incremental tree statistics remain available, but none is relabeled as the present contribution.

## Strongest matched falsifier and rejection criteria

**Primary mechanistic falsifier:** compare the ordered action with the order-blind integrated generator using exactly the same persistent leaves, tree observations, rates/parameterization, durations, initial state, readout, conditioning, and fixed training/evaluation protocol:

    ordered(m) = exp(d_m Q_m) ... exp(d_1 Q_1) h_0,
    blind(m) = exp(sum_{i<=m} d_i Q_i) h_0.

Use corresponding prefixes m for locus outputs; pooling all trees before a local readout can give the control future evidence that the causal ordered arm lacks. If both arms are bidirectional, match their observation access explicitly. Both functions should be trained with the same parameter budget and fixed optimization/selection protocol; a same-checkpoint intervention additionally diagnoses whether an ordered fit actually uses chronology. A claim about model class and a claim about a trained solution are different.

This is parameter/observation matched, **not automatically compute matched**. A sum of incompatible hierarchical generators generally has no common tree eigendecomposition. Its exact matrix exponential can require dense H-by-H work or a separately controlled Krylov/polynomial approximation. Runtime cost and numerical error must be reported honestly; a costly control is not evidence the ordered arm is scientifically better. With forcing, use augmented affine generators A_i=[[Q_i,b_i],[0,0]] in both ordered and exp(sum_i d_i A_i) actions, or derive an equally precise forcing match. Arbitrary pooled emissions are not that control.

Before any authorized scientific run, mathematical/software checks would need to establish dense-small-tree agreement, derivatives, positivity/row sums within tolerance, d=0, polytomies, zero rates, fixed-Q refinement, and a fixed-rate noncommuting example. These are future verification requirements, not tests executed in this assessment.

Reject a *new-operator* claim now: the exact hierarchical generator and cumulative eigenvalues have an explicit 2013/2014 primary prior. Reject the stronger *general graph-collection contribution* unless a substantive additional result beyond known switching/composition is supplied. Reject an *ordered-context mechanism* if a well-matched prefix order-blind control performs equivalently under a predefined paired criterion, if the learned action collapses to effectively commuting/global averaging, or if controlled order interventions make no material difference. Reject the *application hypothesis* if it fails a predefined useful gain on independent evaluation with appropriate uncertainty and cost accounting. A numerical improvement on selected existing validation parents alone cannot establish novelty or independent confirmation.

The source-only outcome is therefore: keep this as a mathematically clean optional architecture experiment, if the parent chooses, but do not present it as an established novel general graph-collection method. H044 remains a separate, explicitly non-novel accuracy hypothesis.


## One deferred question; no new audit or proposal

Could clade-specific donor measures that are not restrictions of one global measure support a useful directed alternative? Reset E_C to the mean under a fixed pi_C rather than the uniform mean. Each E_C is still row-stochastic/idempotent and Q=sum_C r_C(E_C-I) is still a CTMC generator, but inconsistent nested measures can remove internal commutation and the guarantee of a common reversible measure. The present exact tree eigenaction then no longer applies. A finite positive-feature form pi_C(j) proportional to 1_{j in C} sum_k a_Ck phi_k(j), with nonnegative coefficients/features and positive normalizers, could permit support-exact Q actions using weighted leaf-prefix means and clade range-adds, in O(K(H+N)D). Uniformization with a valid dominating exit rate would approximate exp(dQ) using Markov powers; the truncation error is bounded by the corresponding Poisson tail in sup norm (times the input norm). Cost may grow with the dominating rate and physical span. This is related to directed hypergraph diffusion, Markov reset operators, positive-feature attention, and neural ODEs. It is one deferred mathematical/application question, with no originality, advantage, or efficient-fit claim and no separate primary-priority audit undertaken here. The parent's current work remains H044 accuracy and benchmarks.
