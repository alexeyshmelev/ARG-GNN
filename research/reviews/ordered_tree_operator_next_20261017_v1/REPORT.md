# Shared census labels across ordered trees: an inference-ready utility proposal, no operator novelty

2026-10-17. Source-only design for ROOT review. **No defensible new general operator is established.** This note develops the already recorded categorical-census hypothesis in [the prior addendum](../TreeLoom_next_mechanism_novelty_audit_20261016_v1/ADDENDUM_census_lineage_states_v1.md), rather than presenting it as a new discovery. The unresolved biological mechanism is final joint ancestry consistency among queries that share a census-crossing lineage, with genomic links carried only by persistent sampled haplotypes. Its useful contribution, if demonstrated, would be biological structured prediction. Implementation, registration and experiments remain deferred.

## Why this question remains open

H050 stable-leaf memory reaches Present182 span accuracy 97.077844 ± 0.567395%, versus matched continuation 97.000028 ± 0.497841%, a +0.077816 percentage-point difference. Boundary F1 at 1 kb is 18.380657% versus 19.048847%, a −0.668190-point difference. These are eight reused development VAL simulations, one continuation seed101 and 960 added complete-parent updates per arm. The difference does not establish that memory caused oversmoothing, nor that this proposal repairs the responsible errors.

The inspected ledger and audits already cover cut/history transport, SSM/semigroup flow, conditional-innovation shrinkage, cross-tree partner/episode exchange, graph-conditioned reset chains, shared-switch factors, uncertain attachments, donor copying and marker feedback. H043/H045/H046/H048/H050/H051 do not establish useful distinctive cross-tree exchange. H058 already tests same-marker source/age/allele alignment and must not be duplicated. H057's TRAIN native-hold/grid oracle gap, 0.1044431 ± 0.0098382 accuracy points, is limited representational headroom, not a forecast of gain.

The census addendum identifies an unanswered restriction: current census layers pool hidden context, but do not impose a shared categorical variable in the final ancestry distribution. That is the sole mechanism developed here. This is an inference specification for an existing hypothesis, not another memory proposal or an answered cell.

## Biological relation and exact conditional model

Fix the existing European estimand, census age \(\tau=350\) generations, and retain the entire observed-tree/SNP encoder, both complete sweeps, own-state adapter, genomic BiGRU, all observed haplotypes, original query outputs and native-hold support. Let \(f_{tq}(y)\) be its four-class logits on hold \(t\). No input is a query ancestry label.

For a query sampled younger than \(\tau\), define \(g_t(q)\) by the actual observed branch crossing the census: lower age \(\leq\tau<\) upper age. Queries sharing the crossing branch form \(D_{t\ell}\). The group name is local to a hold. It never denotes a persistent internal ancestor. Missing crossings or ineligible ages receive the unchanged backbone output and an explicit unavailable flag, rather than an invented lineage. Older references remain at their actual dated nodes and supply the original encoder evidence. Their proxy labels do not pin a census state.

On the correct dated genealogy, members of one census lineage have the same population label at that site. On an inferred tree this relation can fail through topology or dating error. Therefore use a *soft* relation, and allow every source change along every sampled-haplotype path.

Let \(Y_{tq}\in\{1,\ldots,K\}\), \(K=4\), and introduce local \(C_{t\ell}\) with
\[
 a_{t\ell}(c)=\operatorname{softmax}_c\left(|D_{t\ell}|^{-1}
                    \sum_{q\in D_{t\ell}} f_{tq}(c)\right),
 \quad
 \psi(y,c)=\exp\{\lambda\,1[y=c]\}.
\]
Both arms below receive exactly this group summary. Define \(d_t\) as the physical distance between consecutive original hold centers, \(\ell_x=5000\) bp, and
\[
 T_t(y,z)=\exp\{\kappa e^{-d_t/\ell_x}1[y=z]\}.
\]
The proposal uses only two new learned scalars, \(0\leq\lambda,\kappa\leq2\). They are compatibility strengths, not uncertainty probabilities or demographic rates. These bounds and 5 kb scale are prospective modeling choices, without efficacy evidence.

The conditional model is
\[
 p(Y,C\mid O)\propto
 \prod_{t,q}e^{f_{tq}(Y_{tq})}
 \prod_{t,\ell}a_{t\ell}(C_{t\ell})
 \prod_{t,q\ {\rm eligible}}\psi(Y_{tq},C_{t,g_t(q)})
 \prod_{t,q\ {\rm eligible\ at\ both}}T_t(Y_{tq},Y_{t+1,q}).
\]
Here \(O\) is the frozen observed input. Internal identities never cross holds. Integrating out one local variable gives
\[
 F_{t\ell}(Y_D)=\sum_c a_{t\ell}(c)
                    \exp\{\lambda\sum_{q\in D}1[Y_{tq}=c]\}.
\]
This is an ordinary latent higher-order factor. It is discriminative: the logits and pooled lineage potential already share observations and genomic context. Their product is not a collection of independent genotype likelihoods and has no automatic posterior-calibration interpretation.

At \(\lambda=\kappa=0\), all added factors are constant and the original predictor is recovered. Initialize both scalars at zero with the same explicitly specified one-sided projected update in both arms. The nonuniform group potential avoids the uniform-latent symmetry that can make the association coefficient's initial gradient vanish. If a group potential is uniform, the shared effect begins at second order in \(\lambda\). This mathematical initialization requires source-level numerical qualification before use.

The hard limit \(\lambda\to\infty\) recovers local label equality in the exact model, but it is deliberately outside the proposed finite bounds. Hard tying on inferred trees can make true mixed-source assignments impossible. Finite-round approximate inference below must not be described as enforcing exact hard equality.

## Efficient Torch/PyG path and what it cannot guarantee

Use normalized variable-to-group cavities \(r_q(y)\). For one group, compute
\[
 A_q(c)=\sum_y\psi(y,c)r_q(y)
       =1+(e^\lambda-1)r_q(c),
 \qquad
 B_q(c)=a(c)\prod_{j\in D\setminus q}A_j(c).
\]
Then its exact local factor message is
\[
 m_{F\to q}(y)=\sum_c B_q(c)+(e^\lambda-1)B_q(y).
\]
Thus no \(K^{|D|}\) table or dense query-by-query matrix is needed. Compute \(\log B_q=\log a+\operatorname{scatter\_sum}(\log A)-\log A_q\), normalize log messages, and evaluate the finite \(K=4\) log-sums stably. Temporal Potts messages have the same \(1+(e^\eta-1)r(y)\) form, with \(\eta=\kappa e^{-d/\ell_x}\). Recipient exclusion is algebraic, not statistical independence.

A future implementation can keep the existing PyG whole-tree body and use a separate incidence graph: variable rows \((t,q)\), local-group rows \((t,\ell)\), one group membership edge per eligible row and temporal edges for the same observed sample in adjacent holds. Torch scatter reductions and gathers implement all group messages in batches. There is no per-query Python traversal or internal-ancestor alignment. Use four synchronous sum-product rounds, probability-space damping 0.5, normalized uniform initial messages, and the resulting final variable beliefs as output. Retain gradients through every round and the complete inherited encoder. A projected AdamW scalar update must specify the right derivative at zero rather than accidentally installing a dead ReLU gate.

Let \(M\) be eligible variable memberships, \(J\) temporal links and \(C\) groups. Added forward work is \(O(RK(M+J+C))\), with \(R=4\), and message storage is \(O(K(M+J+C))\). Naive unrolled training activation storage is \(O(RK(M+J+C))\), in addition to the unchanged whole-tree model. A tree pass deriving crossing groups costs \(O(\sum_t |V_t|)\) if ancestor representatives are memoized in traversal order. No repeated descendant enumeration is necessary. These are analytic costs, not measured CPU/GPU fit or an edge-edit complexity claim.

Changing groups and multiple sample trajectories usually create loops. Four rounds are an approximation, with no convergence, exact marginal or exact likelihood guarantee. Only nearby graph paths are reached by those rounds, although the inherited BiGRU already supplies global genomic context. If solver residuals or normalization/gradient checks fail, this specified condition is not executable as written. Do not silently increase rounds after observing accuracy. Small acyclic fixtures should later compare local messages and final beliefs with exhaustive enumeration, including zero-strength and singleton limits. No fixture is run here.

The full inherited predictor is not made invariant to redundant tree serialization by this layer. Preserve the original canonical holds. Outputs remain constant within a native hold, so unrepresented internal switches remain unresolvable.

## Precise prior-art distinction and novelty verdict

[Garcia Satorras and Welling, *Neural Enhanced Belief Propagation on Factor Graphs*, arXiv:2003.01998v5](https://arxiv.org/html/2003.01998v5), §2.1 and §2.2 Eqs1–3, define arbitrary factorization, recipient-excluding sum-product and marginal readout. §4.1/Table2 builds GNNs on arbitrary-variable factor graphs; §4.2 combines learned refinements with BP. The construction above is directly representable by that machinery, with ordinary Potts-factor messages rather than an additional learned correction network. Its precise extra specification is the *census-crossing membership relation and cloned-dependence falsifier*. It is more restricted than the generic factor-graph framework, not a new inference operator.

[Kohli, Ladicky and Torr, *Robust Higher Order Potentials for Enforcing Label Consistency*, 2009](https://doi.org/10.1007/s11263-008-0202-0), Eq17, already imposes robust group consistency through a truncated minimum of class-dependent linear disagreement counts. The proposed marginalized factor instead gives a log-sum-exp energy. In the uniform-\(a\), equal-\(\lambda\) case,
\[
 F(Y_D)=K^{-1}\sum_c e^{\lambda n_c}.
\]
For four classes and six members, count histograms \((4,1,1,0)\) and \((4,2,0,0)\) have the same majority size and therefore the same equal-class Robust \(P^n\) Eq17 energy, while the second factor exceeds the first by \((e^\lambda-1)^2/K>0\). This distinguishes these two particular potentials. It does **not** distinguish the proposal from known latent-mixture factors, sum-product, or soft consistency models as a class. Kohli's exact graph-cut move guarantees also do not transfer to this loopy marginal-inference schedule.

The cited full primary passages, not titles or abstracts alone, were checked in existing public-text caches. The bounded review does not establish absence of an exact prior census composition. **No defensible operator-level novelty or priority claim survives these reductions.**

## One matched falsification, only if ROOT elects to proceed

Compare **shared lineage** with **cloned lineage**. In the latter replace \(C_{t\ell}\) by \(C_{t\ell q}\), with
\[
 F^{\rm clone}_{t\ell q}(y)=\sum_c a_{t\ell}(c)\psi(y,c).
\]
Every clone receives the same actual group summary \(a_{t\ell}\), observations, scalar parameters and temporal factors. It has no private learned parameter. Run the same four-round/damped schedule with the same incidence support and its other-member product blocked. This removes joint categorical dependence among sample paths while retaining ordinary group evidence and chain smoothing. Neither a topology permutation nor a comparison with NoMem alone answers this question.

Prospective pilot: identical completed H045 own-state pass30 initializer, independent reset AdamW using the exact existing H050 continuation recipe, FP32, all32 original TRAIN parents, all686 query outputs and original exact-span supervision. Complete30 passes/960 parent updates per arm, seed101, six checkpoints at5/10/15/20/25/30. Seal all96 new checkpoint×arm×eightVAL predictions before scoring targets. Preserve the original equal-parent Present182 accuracy/lower-NLL/earlier-pass selection rule. This is a proposal for1,920 new updates, not a registration or authorization.

The completed H050 continuation may serve as the practical no-addition reference only after its source, initialization, ordering, optimizer, supervision and scoring equivalence is authenticated. Do not rerun that answered condition simply to add a baseline. Historical scores alone are descriptive if equivalence is not established.

Before outputs, freeze the joint utility gate: shared must gain at least1.00 percentage point Present182 boundary F1 at1kb over cloned, lose at most0.05 point span accuracy, and show no mean NLL regression. It must also satisfy those practical noninferiority requirements against an authenticated equal-budget continuation. Report paired parent effects/sampleSD, precision/recall/event counts and all existing cohorts. Use the existing fixed, uncensored true≤10kb tract-recall definition and fixed false-island/false-switched-bp diagnostics. A mean loss of true-short-tract recall rejects the claim even if boundary precision improves. Do not select checkpoints by these boundary outcomes.

If shared ties cloned, loses either practical comparison, erases real short tracts, or learns \(\lambda=0\), reject useful shared-lineage dependence. If both new arms improve equally, attribute no gain to categorical sharing. Passing would justify separately designed independent confirmation, not generic novelty. Do not repurpose H059's frozen independent confirmation, tune on outer outcomes, change H058, or interfere with H060/current admissions. No automatic replay or broadened sweep follows failure.

## Recommendation and scope

ROOT may review this as a fully specified refinement of an existing unanswered biological restriction. It should **not** be registered as a fresh general ordered-tree operator. The default scientific conclusion remains no defensible new operator novelty. On another ordered graph collection the same factorization applies only when locally defined groups have an externally justified common-label relation, stable observed entities connect snapshots, and that relation can tolerate observation error. Arbitrary graphs do not inherit the census argument, and no cross-domain efficacy is claimed.

Only source, concise ledger/review metadata and cached primary literature were read. This directory contains prose/evidence only. No scientific arrays, truth payload, predictions or checkpoint contents were opened. No model import, implementation, registration, numerical fixture, job, process/resource probe, guard/admission/accounting action, installation or Git operation occurred.
