# Physical exposure signatures of complete ARG trees

17 October 2026. Source-only architecture research for ROOT.

**Recommendation: retain one precise, low-confidence application hypothesis, withhold a new-general-operator claim, and defer implementation or fitting.** The candidate replaces tree-count recurrence with fixed-distance, second-order signatures of a cumulative whole-tree evidence field. Its meaningful question is whether ordered *exposure* to reference-affinity evidence improves tract discrimination and length/density transfer beyond equally sized commutative statistics. Signature layers, graph/signature combinations and order-sensitive graph-sequence kernels already exist. That fact does not rule out a useful task-specific composition, but the composition and its biological benefit are unvalidated.

## Problem and current evidence

TreeLoom's complete upward/outside traversals, structural residuals, sample EMA before the second shared traversal and final BiGRU are present in actual source. The original readout advances its GRU once per native tree and produces constant-per-hold logits. Supplying interval lengths does not make a GRU an integral over genomic exposure. The H049/H050 audit also gives explicit full-predictor subdivision counterexamples involving the memory readout, episode-size eligibility and span-dependent SNP pooling. These are source/algebra facts, not measured failure frequencies.

Selected European H050 stable-leaf memory scores 97.077844 ± 0.567395% Present182 span accuracy versus 97.000028 ± 0.497841% continuation. Its 1 kb boundary F1 is lower, 18.380657% versus 19.048847%. Eight reused development parents, one optimizer seed and separately selected checkpoints do not identify an oversmoothing cause or establish independent generalization. H043 exchange, H044 localization, H045/H046 foreign context, H048 reset persistence and H050 conditional-triple history did not establish their intended practical benefit. H058, ordered-marker placement, H059 and H060 retain their own questions.

Unequal tree dwell lengths are real input geometry. A path through embedding *vertices* treats a representation change differently from time spent in one representation. The proposed object is instead a cumulative integral of the complete-tree evidence field. It adds no inferred ancestor correspondence, recombination-event posterior, clade diffusion, attachment mixture, donor-copying state or reference-covariance shrinkage.

## One explicit architecture

**Input quotient is part of the architecture.** Start from the complete observed marginal metric-tree field, stable sampled-haplotype IDs, known reference labels/dates and coordinate-anchored observed variants/mutation annotations. Coalesce adjacent intervals representing the same rooted, dated, sample-labelled tree, independent of internal node numbering. Preserve every sampled node and union coordinate-marked observations exactly once. Do this *before* computing span, SNP pooling, mutation-support gates, clade flags or episode eligibility. Equality means equality of the attributed metric tree, not merely an equal age-cut partition. A digest can index candidates, but exact canonical forms must resolve equality. Do not merge changed dates/topologies or overlapping copies that double physical exposure. No unary-suppression claim is made.

Let the resulting maximal intervals be \(J_t\), and \(e_{tq}\in\mathbb R^{32}\) the sampled-node embedding from the same two shared complete-tree GNN sweeps and local structural adapters. Both passes include every observed node and sample. Remove the sample-memory module and final BiGRU in *all* proposed arms. Local weights may share the source-defined H045 own-state initializer, while the new common readout has one shared initialization. This is a changed predictor, not a zero-initialized addition that exactly recovers the selected TreeLoom.

Use a single bias-free projection
\[
 u_{tq}=\tanh(Ue_{tq})\in[-1,1]^4,\qquad U\in\mathbb R^{4\times32}.
\]
On \(x\in J_t\), define \(v_q(x)=(1,u_{tq})\in\mathbb R^5\) and the continuous path
\[
 X_q(x)=\int_{x_0}^{x}v_q(s)\,ds .
\]
The first channel measures physical bp. Feature channels measure bp times latent evidence. This is **not** linear interpolation through \((x_t,e_{tq})\), nor a path of topology-edit operators. On a constant-evidence piece of width \(w\), the path increment is \(wv\), even without any embedding change.

Fix \(L=5000\) bp prospectively. For every original fixed 100 bp evaluation-bin center \(x\), use the left/right windows
\[
 W^-_x=[\max(x_0,x-L),x],\qquad
 W^+_x=[x,\min(x_1,x+L)] .
\]
Their clipping depends only on the connected observed-coverage component containing the focal point. For either window \(W=[a,b]\), with \(\ell=b-a>0\), calculate
\[
 S_1=\int_a^b dX,\quad
 S_2=\int_{a<s<t<b}dX(s)\otimes dX(t),\quad
 \mu={1\over\ell}\int_a^b u(s)\,ds ,
\]
\[
 A={S_2-S_2^\top\over2\ell^2},\qquad
 M={1\over2\ell}\int_a^b u(s)u(s)^\top ds .
\]
Use the ten upper-triangular entries of the \(5\times5\) antisymmetric \(A\). The commutative control below uses the ten upper-triangular entries, including the diagonal, of the \(4\times4\) symmetric \(M\). Both have entries bounded in absolute value by \(1/2\). Zero-length windows return zero statistics and coverage zero.

At each fixed \(x\), the classifier receives
\[
 [\,e_q(x),\mu^-,\mu^+,\operatorname{ut}(A^-),
              \operatorname{ut}(A^+),\ell^-/L,\ell^+/L\,]\in\mathbb R^{62}.
\]
Use a shared-form 62→32→4 MLP and softmax. No hold index, tree count, raw sample ID or chromosome length enters this head. Output locations and loss weights remain the original fixed physical support. Probabilities can now vary within a native hold through neighboring context, although within-hold marker arrangement discarded by pooling remains unavailable. Every control has the same output resolution.

Two constant pieces of widths \(a,b\) and lifted evidence \(v,w\) contribute
\[
 S_2={a^2\over2}vv^\top+ab\,vw^\top+{b^2\over2}ww^\top.
\]
Swapping their order preserves \(\mu,M\) but reverses their antisymmetric term \(ab(vw^\top-wv^\top)/2\). This is the specific information being tested. It is familiar second-level signature algebra, not an expressivity theorem over arbitrary GNNs. Truncation at level two is not injective and cannot distinguish every short excursion from every sustained transition.

## Exact property, and its limits

For a constant piece with increment \(z=wv\), its truncated signature is \((1,z,z\otimes z/2)\). Concatenation is
\[
 (a,B)\star(c,D)=(a+c,B+D+a\otimes c).
\]
Splitting the same piece into widths \(w_1+w_2=w\) gives exactly the unsplit signature. Window endpoints can cut pieces using the same formula.

At fixed weights in deterministic evaluation, the **whole proposed predictor** is invariant in real arithmetic to redundant split/merge serialization of an identical attributed tree interval, provided the observation measure, canonicalization, fixed query coordinates and window clipping are unchanged. The quotient first restores identical canonical tree inputs and pooled features. The encoder therefore returns identical \(e\). Exact interval composition yields identical window statistics, and the pointwise decoder returns identical probabilities. Training randomness must likewise attach to canonical objects if training-path equivalence is ever claimed. Floating-point tolerance needs later qualification.

This does not hold if SNP pooling is recomputed on artificial fragments, episode counts use raw records, the decoder advances per fragment, or intervals are split while retaining their full original widths. It does not give invariance to genuine extra topology/date changes or to another inference method's trees. **Canonicalization alone could also protect a GRU from redundant serialization.** Thus a split/merge test is a necessary implementation check, not signature-specific novelty or biological utility evidence. Bounded physical windows avoid a chromosome-wide accumulating signature, but do not prove length transfer of the encoder or full classifier.

## Feasibility and decisive controls

For \(T\) canonical trees, \(H\) sampled haplotypes, \(B\) fixed output positions, encoder width \(d=32\) and rank \(r=4\), the extra projection costs \(O(THdr)\). Signature products cost \(O((T+B)H(r+1)^2)\) for a fixed number of windows, in addition to all complete-tree/SNP work and the classifier's output cost.

A two-pointer sliding window with a two-stack ordered-product queue gives amortized constant signature products per insertion/removal. Complete middle pieces use the queue aggregate; clipped head/tail pieces use analytic partial increments. Vectorize over all \(H\) samples. Four equal-physical-quarter subwindows for the null below multiply work by a fixed constant. This avoids an \(H^2\), tree-pair or chromosome-prefix subtraction tensor. It is a standard associative sliding-aggregation construction, not a new algorithm. Forward queue storage scales with the maximum number of active tree pieces times \(H(r+1)^2\); full training activations, GNN states, output arrays and checkpoint/recomputation costs remain additional. No throughput, numerical stability, memory fit or resource admission was measured.

Three independently optimized arms isolate the question:

1. **True exposure order:** the specified \(A^-,A^+\).
2. **Commutative statistics:** replace only each ten-coordinate area slot with \(\operatorname{ut}(M)\). Retain the same \(\mu\), local tree embeddings, coverage, projection/head dimensions and parameter counts. This controls first-order exposure plus ordinary second moments.
3. **Order null:** divide each actual clipped window into four equal physical quarters, retain each quarter's complete signature and permute the quarter order with a fixed, outcome-independent nonidentity permutation for each parent/position/side. Use the same permutation across samples, never shuffle labels or sample correspondence, and do not supply the permutation to the head. Compose the permuted quarters to obtain its area. This preserves total exposure, \(\mu,M\), all complete-tree evidence and within-quarter order, while disrupting the specified between-quarter chronology. It is a partial order null, not removal of all temporal information.

Every arm can calculate all three slots before selecting its own to match arithmetic. Equal parameter count is not a claim of identical effective capacity. A tie with the commutative arm rejects useful second-order chronology beyond exposure moments. A tie with the order null rejects the tested between-quarter ordering explanation. Beating only historical TreeLoom is uninterpretable because the readout/resolution and training changed.

A later source qualification should test unchanged fixed-coordinate probabilities under arbitrary identical-tree split/merge, including splits through a SNP-supported hold and clipped windows, plus internal-ID/child-order relabelling with corresponding inputs. No such test was run here.

## Biological hypothesis, benchmark and falsification

A short misleading genealogical affinity excursion and a sustained transition can have similar average evidence but different temporal arrangement. Ordered physical exposure might help their discrimination while reducing sensitivity to tree-record density when moving from 2 Mb to 30 Mb. The hypothesis is plausible, not diagnosed by the current scores. Real short ancestry tracts can also look like excursions, and aggressive discrimination can erase them. Tree reconstruction errors, proxy-reference mismatch, source-age effects, lost within-hold SNP order and level-two aliasing remain failure modes.

The existing European 32 TRAIN/eight reused VAL full-952-haplotype benchmark can compare the three arms with all686 query outputs, original100bp physical supervision and Present182 dense boundary diagnostics. A representative future discovery schedule is the original30passes/960complete-parent updates per arm, fresh matched AdamW and parent ordering, common local initializer and common head seed. Preserve the original accuracy/lower-NLL/earlier-checkpoint rule at passes5/10/15/20/25/30, and seal all144 arm×checkpoint×VAL products before any scoring truth. This is a prospective design, not a registration or execution request.

Freeze the practical gate beforehand: true order must gain at least1.00 percentage point equal-parent 1kb boundary F1 over **both** controls, lose at most0.05 point span accuracy, show no mean NLL regression and no loss of uncensored true≤10kb tract recall. Report paired parent effects/SD, correctly typed transitions, matched/missed/extra boundaries, false islands and false-switched bp/Mb. Gains solely from deleting real short tracts fail. One seed/reused VAL remains exploratory.

The current American20×2Mb TRAIN/five30Mb VAL benchmark can separately test length/panel transfer at its original1600haplotypes/10000fixed focals with a three-class shared head. It cannot establish dense boundary fidelity. Prespecify density strata using TRAIN-only canonical-hold quantiles and retain all predetermined focals within each stratum. Density correlates with biological difficulty, so this stratification is descriptive, not a causal intervention. Artificial split/merge is the pure representation intervention. Do not repurpose H059, alter live H058/marker/H060 conditions, or tune on independent outcomes. A subsequent independently frozen dense-label panel and multiple optimizer seeds would be needed for generalization.

## Primary antecedents and novelty boundary

| Primary literature inspected | Overlap and exact distinction |
|---|---|
| Whitehouse, Ray & Schrider, *Tree Sequences as a General-Purpose Tool for Population Genetic Inference*, [DOI10.1093/molbev/msae223](https://doi.org/10.1093/molbev/msae223), GCN Architecture and Training/Fig1 | Complete marginal-tree GAT processing, a node-sequence GRU and tree-sequence GRU already combine genealogy structure with genomic order/segment geometry. The new question is a particular physical-exposure readout, not priority for whole-tree GNN plus genomic context. |
| Bonnier etal., *Deep Signature Transforms*, NeurIPS2019, [v2](https://arxiv.org/html/1905.08494v2), §§1.3–1.4,3.1–3.3, AppendixA | Learned feature transforms, window lifts, differentiable signatures, resampling properties and composition are established. The candidate is a specialized instance, with an ARG observation quotient and a cumulative physical-exposure lift. |
| Király & Oberhauser, *Kernels for Sequentially Ordered Data*, JMLR20(31),2019, [inspected preprint](https://arxiv.org/html/1601.08169v1), §§3–5,8 | Signature-based sequentialization explicitly includes sequences of graphs; ordered moments and efficient low-rank computation are established. A GNN encoder and local ancestry readout do not create generic graph-sequence priority. |
| Kidger etal., *Neural Controlled Differential Equations for Irregular Time Series*, NeurIPS2020, [v2](https://arxiv.org/html/2005.08926v2), §3/Eq3, AppendixC | Time-augmented continuous controls and irregular observations are established. Their interpolated observation path differs from integrating a piecewise-constant evidence density. The latter is an explicit modeling choice, not a new CDE principle. |
| Morrill etal., *Neural Rough Differential Equations for Long Time Series*, ICML2021, [v2](https://arxiv.org/html/2009.08295v2), §§2.1,3.1/Eqs9–10 | Interval log-signatures, level-two signed area and length reduction are explicit. Here a finite signature is decoded directly at fixed genomic coordinates; no numerical RDE solver or new rough-path theorem is proposed. |
| Choi etal., *Graph Neural Controlled Differential Equations for Traffic Forecasting*, AAAI2022, [v1](https://arxiv.org/html/2112.03558v1); Choi & Park, *Graph Neural Rough Differential Equations for Traffic Forecasting*, TIST2023, [v2](https://arxiv.org/html/2303.10909v2) | Spatial graph processing with continuous controls and subpath log-signatures already exists. The latter uses a fixed sampled-location roster, time-interpolated node features and learned adaptive adjacency. This candidate processes changing complete trees first and joins only actual sampled identities; that task-specific distinction is not general operator priority. |
| Qin etal., *Learning Dynamic Graph Embeddings with Neural Controlled Differential Equations*, [v2](https://arxiv.org/html/2302.11354v2), §III/Eq4 and theorem5 | Time-varying graph paths, graph-enhanced controlled dynamics and signature-based universality arguments already cover changing-graph continuous architectures. The candidate neither interpolates anonymous ARG ancestor adjacency nor evolves a GNN state through that control. |
| Cass, Lyons, Salvi & Yang, *Computing the full signature kernel as the solution of a Goursat problem*, [inspected v2](https://arxiv.org/html/2006.14794v2), §2.1/theorem2.2 | Signature inner products and their exact kernel interpretation are established. This design uses explicit rank-four level-two features, not a new kernel/PDE or all-pairs signature computation. |

Temporal Motif Signatures (arXiv2606.01176v1) was also checked: its candidate-local recency/star/triadic count features are a different use of “signature,” not evidence of this iterated-integral construction. Schäfl etal., *G-Signatures: Global Graph Propagation With Randomized Signatures*, [v1](https://arxiv.org/html/2302.08811v1), abstract/Introduction, also lifts graph structure into latent paths and applies randomized signature layers. That is intra-graph latent traversal, rather than genomic exposure, and further excludes a generic “signature GNN” priority claim. Its unavailable v2 HTML and rate-limited discovery requests supply no absence evidence.

Internally, H040 already uses an antisymmetric trilinear interaction between changed-clade states; H035 used a signed difference of graph/alignment compositions. This design's antisymmetry itself is not new. Its variables are integrated sampled-node evidence over a fixed physical window, not H040 events or H050 sample-witness partition triples. The 16October interval-tree audit already mentions path-signature expansions of switched linear dynamics and graph-sequence kernels. This report does not reopen its rejected heat/semigroup direction.

**The concrete additional composition is canonical observed-tree/SNP encoding → stable-sample physical exposure path → fixed-coordinate signed-area readout, with exact split/merge semantics and a chronology-versus-moment falsifier.** Those commitments resolve a real representation ambiguity and define an unanswered utility condition. They do not presently establish publication priority or a new general GNN operator. Known primitives alone are not a rejection criterion; the limiting evidence is the close graph/signature antecedents and absence of demonstrated distinct biological benefit. Retain the specification for review, but do not add another full fit solely to manufacture architectural novelty. Await the current matched marker evidence and a separate ROOT decision.

Only instructions, manuscript/source text, decision/ledger metadata, relevant prior reviews and public primary literature were read. No scientific tree/genotype/truth/checkpoint/prediction payload, guard import, numerical fixture, simulation, fit, test, registration, job/resource/process inspection, installation or Git action occurred. Writes are confined to this new review directory. All claims above are symbolic or source/literature assessments.

