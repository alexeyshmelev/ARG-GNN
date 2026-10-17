# Continuous clade transport: bounded novelty assessment

**Verdict: reject a new-operator claim. Retain, at most, an untested genealogy-specific utility hypothesis.** The proposed exponential is exactly graph heat diffusion on disjoint normalized cliques. Stable sample identities, exponential retention, incidence lift/pool, temporal graph state and ordered composition already have primary and local precedents. Finite-rate diffusion of the *stored* stable-leaf state is not numerically identical to the completed H050 stable-leaf arm; that is a precise possible changed condition, not evidence of mathematical priority or biological benefit. Do not register, implement or launch another condition from this assessment. H058 and native comparisons remain separate ongoing work.

## What the existing work already answers

The current decision/ledger rows were read; they are historical records, not newly observed process status.

| Condition | Actual mechanism and outcome relevant here | Consequence |
|---|---|---|
| H048 | Four-class, original100bp-grid rank-one reset CTMC, conditioned by whole-tree features, versus the same graph signal used for confidence. Completed utility criteria failed: accuracy difference -0.008640pp and boundary-F1 difference -0.072765pp. | An idempotent reset generator already has the same exponential algebra. Changing its state space from classes to stable leaves is a domain/architecture distinction, not a new semigroup. |
| H050 / European TreeLoom | Fixed350-generation cuts, projected whole-tree upward+outside states, stable-leaf bidirectional physical-distance EMA, clade pooling, injection before the second shared tree sweep, then the original full-parent BiGRU. All four arms completed. Stable-leaf accuracy97.077844% versus continuation97.000028%, but boundary F1 fell from19.048847% to18.380657%; triple memory failed both criteria. | Stable identity, clade observations, exponential history and feedback are already implemented and evaluated. No unchanged memory replay. |
| H058 | Same-marker reference source/age/allele path alignment before the second tree sweep; matched alignment-null and original continuation. Ledger has no completed efficacy answer; ROOT reports its full trial is moving. | Different question. This proposal neither replaces its trial nor inherits an H058 result. It introduces no new observed data modality. |

The v14 manuscript and actual H050 model agree on the stable-leaf EMA plus final BiGRU. The American full model adapts this to1600 sampled haplotypes,600 references,1000 queries and3 classes; it is not the completed European952/266/686/four-class result. These data/model bindings must never be interchanged.

The actual reused CutMemoryV1 stable arm computes a clade representative forcing f_i, lifts it to stable leaves as z_i=R_i^T f_i, and updates

    m_i = rho_i m_(i-1) + (1-rho_i) z_i,   rho_i=exp(-width_i/ell),

with a special first-state initialization, separately in each direction. It pools the resulting leaf histories back to current clades, subtracts current forcing, and applies the learned output projection. Its transition on the old leaf state is scalar rho_i I. Therefore replacing the input with another clade mean while retaining this scalar transition is the existing mechanism, up to the disclosed feature, initialization and readout choices.

## Correct operator, dimensions and exact limits

Let i index genomic intervals and x be genomic coordinate in bp; tau denotes a *coalescent age cut*, not the genomic evolution coordinate. For N stable observed sampled-haplotype identities and K_i nonempty cut groups, take R_i in {0,1}^{K_i x N}, exactly one membership per column, and D_i=R_i R_i^T=diag(group sizes). For H in R^{N x r},

    P_i = R_i^T D_i^(-1) R_i in R^{N x N},
    Q_i = I-P_i,
    E_i(w) = exp[-lambda w Q_i] = P_i + exp(-lambda w) Q_i.

The proof is P_i^T=P_i and P_i^2=P_i. P_i averages features within a cut group, not between ancestry classes; r can be a small latent width such as8. For lambda>=0 and w>=0, E is a convex combination of I and P: symmetric, positive semidefinite, doubly stochastic, and nonexpansive in Euclidean norm. It preserves each current-group mean while damping within-group disagreement. lambda has units bp^-1 and is a model smoothing rate, not a biological recombination, migration, ancestry or coalescence rate.

This is exactly heat flow for the weighted clique Laplacian L_i=I-P_i: off-diagonal conductance1/group_size within each group, zero outside. For the *same* P,

    E_i(a) E_i(b) = E_i(a+b),   E_i(0)=I.

Refinement means splitting one physical interval into pieces with the same total width. Duplicating its full width twice represents twice the physical length and must not leave the result unchanged.

Different partitions need not commute. In particular,

    E_j(b)E_i(a)-E_i(a)E_j(b)
      = (1-exp(-lambda a))(1-exp(-lambda b))(P_jP_i-P_iP_j).

The partitions {1,2}|{3} and {1}|{2,3} already have noncommuting projectors. Thus order matters, but switched/temporal graph diffusion already has this property. It does not identify an enduring inferred ancestor across recombination.

There are two important reductions:
- lambda=0 gives no spatial mixing; with a common leak/forcing rule it reduces to independent stable-leaf state.
- The strong-mixing limit gives hard clade averaging. Lifting old group values to leaves and pooling to a new cut produces D_new^(-1) R_new R_old^T z_old: normalized descendant-overlap transport, already derived and rejected as operator novelty in the earlier local no-repeat audit.

Finite-rate relaxation retains within-clade leaf residuals between topology changes; this differs from H050's scalar old-state transition. If one first projects the state into the current clade subspace and discards all residuals, applying E has no effect. Likewise P_i E_i(w)H=P_i H: a current-clade-mean-only readout cannot see this interval's homogeneous relaxation directly; any effect must survive into a later different partition or use leaf-resolved state.

## Forcing is essential, and cannot be guessed

The supplied exponential defines only homogeneous evolution. To be driven by a fixed whole-tree GNN observation field U_i in R^{N x r}, an explicit affine ODE and its exact source integral are needed. For

    dH/dx = -lambda Q_i H + U_i,

the exact constant-field update is

    H_out = E_i(w) H_in
          + [w P_i + (1-exp(-lambda w))/lambda Q_i] U_i,

with the continuous lambda=0 limit wU_i. Adding one unscaled U_i at every serialized interval breaks refinement consistency. The tempting update E H+(I-E)Z never drives the clade means at all: I-E=(1-rho)Q.

The pure generator also has undamped mean modes; constant forcing can accumulate with chromosome length. A possible *explicit modification*, not the original homogeneous proposal, is

    dH/dx = -[gamma I + lambda Q_i]H + gamma Z_i,   gamma>0.

Then

    A_i(w) = exp(-gamma w)P_i + exp[-(gamma+lambda)w]Q_i,
    H_out = A_i(w)H_in
          + (1-exp(-gamma w))P_i Z_i
          + gamma/(gamma+lambda) [1-exp(-(gamma+lambda)w)]Q_i Z_i.

This is ordinary stable driven graph-state dynamics. It could avoid2Mb-to30Mb mean drift, but its forcing/rates would need freezing prospectively. A GNN depending nonlinearly on the evolving memory is not a constant external forcing; the displayed closed form then does not solve that feedback ODE.

## Sample ages, reference masks and multiple cuts

Use stable haplotype columns, not internal-node IDs. The whole observed roster stays present across trees; missing SNP calls are feature masks, not permission to remove a sample or reset its state. Preserve actual phase/sample/query/reference mappings. As in existing code, a sample older than tau remains its actual-sample singleton until the relevant cut exists. Isolated samples/multiple roots need explicit nonempty groups. Root saturation can give nearly global mixing and must be disclosed.

The unmasked projector may mix *observation-derived* query and reference latent states; that is transductive graph computation, not permission to use query ancestry truth. Reference class indicators remain known-reference-only; query classes use the existing unknown mask. TRAIN/VAL/TEST truth capabilities remain unchanged.

If specifically demanding reference-only broadcasts, let S be a diagonal reference indicator and D_ref=R S R^T. Then

    A = R^T D_ref^dagger R S

broadcasts reference-clade means. A is idempotent on supported groups but generally not symmetric/orthogonal; it is not the claimed Euclidean projector P. A no-reference clade has zero rows, so blindly using I-A damps its state to zero. Explicit identity/no-transport blocks for unsupported clades restore an idempotent row-stochastic map, but still do not yield the original symmetric contraction proof. Conversely S R^T D_ref^dagger R S has zero query rows and cannot broadcast to queries. These alternatives must not be conflated. Start with one clearly stated mask policy, not an unregistered choice among them.

Multiple age cuts are not justified by operator novelty. Under consistent dated-tree semantics they give nested partitions, so their orthogonal projectors commute within an interval. For fine-to-coarse P_1,...,P_C, P_a P_b=P_max(a,b). A nonnegative sum L=sum_c lambda_c(I-P_c) is just a multiresolution spectral smoother. Its eigenspaces are I-P_1, P_1-P_2,...,P_(C-1)-P_C,P_C, with cumulative scalar decay rates; exact exponentiation remains simple. This is a known hierarchical filter, not multiple independent ancestral identities. Without a separate age-scale biological hypothesis and matched control, retain the inherited single350-generation cut rather than add cuts.

## What full-predictor refinement invariance would require

The theorem concerns a fixed partition AND a fixed observation field. Identical topology alone does not imply identical GNN forcing: splitting SNPs and recomputing hold mean/max pooling changes it.

A whole-predictor claim would require:
1. The same physical observation field, cut membership, masks and boundary state under subdivision; any source integration is width-scaled and exact.
2. Leaf-resolved state evaluated at the same physical query coordinates, using partial-interval evolution where needed. Assigning the right-end state to every point of a hold changes predictions after splitting it.
3. A pointwise readout of local GNN features and forward/backward continuous states, or another separately proved compatible operation. Both directions must have fixed physical boundary conditions.
4. No extra per-piece GRU/nonlinear recurrence, resets, independent dropout draws or serialization-dependent normalization. Geometry must not change solely because interval centers/neighbor widths were redefined.
5. The same physical/focal prediction and loss support, with physical weights rather than equal weight for each artificial piece.

The existing BiGRU is not such an operation: in general GRU(GRU(h,z),z) is not the original one-step update, even if interval-length features are provided. Keeping it yields, at most, invariance of the isolated diffusion submodule. Keeping a fixed original native-hold grid while refining only numerical integration can protect that grid's outputs, but is not invariance to re-encoding or subdividing the native tree sequence. These are real-arithmetic identities; no bitwise FP32 claim follows.

## Primary literature: known, and exactly what is different

- **Chamberlain et al., GRAND, ICML2021, section3.1.** Defines graph diffusion; explicitly gives analytic matrix-exponential evolution for a fixed linear diffusion operator. The proposed L=I-P is a particularly tractable clique Laplacian. [Proceedings](https://proceedings.mlr.press/v139/chamberlain21a.html); [full text inspected](https://arxiv.org/html/2106.10934v2).
- **Li et al., State Space Models on Temporal Graphs: A First-Principles Study, NeurIPS2024, section3.1 theorem1/equation5 and section3.2 theorem2/equation8.** Fixed node identities, piecewise graph mutation, Laplacian-regularized memory, and oracle ZOH integration are explicit. Crucial distinction: their equation5 smooths the *forcing*, (I+alpha L(t))^-1 X(t)B^T, while memory evolves through the feature-state matrix A. The candidate instead diffuses *old state across leaves*. It is not correct to call the displayed candidate exactly that GraphSSM layer. Their practical equation9 also approximates the source factor, so do not attribute an exact whole-model refinement theorem to it. [Primary paper](https://papers.nips.cc/paper_files/paper/2024/hash/e5ba3d6d93213db6b1d1931c6517fe1a-Abstract-Conference.html).
- **Poli et al., Graph Neural Ordinary Differential Equations, arXiv1911.07532v4, sections2 and3.2.** Graph-dependent vector fields and autoregressive/hybrid graph dynamics already cover the general continuous/temporal GNN concept. [Full text](https://arxiv.org/html/1911.07532v4).
- **Li et al., DCRNN, ICLR2018.** Diffusion convolution plus graph recurrent temporal modeling is established; it is not the same clique generator. Only its primary abstract was used for that broad comparison. [Primary abstract](https://arxiv.org/abs/1707.01926).
- **Gu, Goel and Re, S4, ICLR2022, sections2.1–2.4.** Continuous state-space models, step-size-dependent recurrence and sequence/convolution views are established. Its reported section2.3 discretization is bilinear, not this exact exponential ZOH rule. [Full text](https://arxiv.org/html/2111.00396v3).

This bounded primary comparison rejects broad operator priority; it is not an exhaustive publication-priority certificate. No paper inspected demonstrates this exact genealogy-specific masked readout and matched biological utility. Absence from this small set does not establish novelty.

## Cost and one representative future utility test

Given group IDs, apply P H by group sums/counts and a gather in O(Nr+K_i r) time and O(Nr+K_i r) scratch; never materialize N-by-N projectors. Serial evolution is O(TNr), or O(CTNr) for C cuts, in addition to cut construction and the unchanged whole-tree GNN. Streaming one direction uses O(Nr) state; full-parent bidirectional outputs, gradients and cached forcing commonly require O(TNr) histories plus the existing GNN/autograd footprint. Tree traversal/cut construction is additional, not compressed-ARG edit complexity. Associativity alone gives no cheap S4-style parallel scan: products across changing partitions generally fill in, so a fixed-size sparse projector representation is not closed under composition. No speed, memory fit or resource gain was measured.

The biological hypothesis would be: **correct alignment of latent genomic history to the inferred current clade partition improves ancestry tract discrimination beyond equally strong but misaligned partition smoothing, without erasing genuine short tracts.** It is not that clades equal ancestral populations. Inferred-tree errors, proxy-reference bias, root saturation, old-sample singletons and smoothing across a genuine ancestry switch can all defeat it.

If later work is independently justified, one representative *new*, prospectively fixed pair would use the same whole-tree observation encoder, one inherited350 cut, the same exact stable driven transition and a common refinement-safe pointwise readout (no final hold-count BiGRU). Compare true P_i against S^T P_i S with one fixed parent-wide sampled-haplotype permutation S. Keep references within observed source/age strata and queries within observed age strata; never use query ancestry in the permutation. Leave GNN inputs and targets on their true sample IDs. This null preserves group sizes, reference-mask counts, partition overlap dynamics, rates, spectra, parameters, spans and arithmetic, while disrupting clade-to-feature/state alignment. Record moved support and unchanged/immovable groups before training.

Both arms need the same encoder warm start, new-head initialization, full original observations, independent matched optimizer streams/budgets, checkpoint rule and scoring. A representative European discovery schedule is the original32TRAIN parents x30passes =960updates per arm, with all96 checkpoint x arm x8VAL seals before either truth. This is only an illustrative future matched test, not a registration or authority; it is not the old H050 model because the common readout changed. Do not reuse old H050/H058 scores as matched controls.

Freeze the utility rule before outputs: for example >=1.00pp paired mean1kb boundary-F1 gain over the conjugated null, <=0.05pp span-accuracy loss and no mean NLL regression, while reporting true-short-tract recall and false switches/switched bp perMb. Failure, or a gain achieved only by deleting short tracts, rejects that biological use. A win establishes clade-alignment utility conditional on the common continuous core; it does not prove superiority to TreeLoom/H058, a new general operator, causal ancestry transport, real-data validity or independent-test generalization.

Only this review directory was written. No implementation, numerical model, data/truth/checkpoint payload, resource/process/GPU inspection, training, admission, authority, ledger registration, live source or paper modification occurred.
