# Independent H051 literature and mathematical priority audit, v2

Date: 2026-10-16. This is a research-method audit, not an acceptance/rejection recommendation. It starts from the named candidate sources and primary literature, with no requested favorable verdict. No H051 model accuracy result exists in the material reviewed. The report distinguishes an exact algebraic classification, plausible domain specialization, and untested empirical utility.

## Main finding

H051's fixed transport is an empirical conditional-expectation operator. Its innovation is the difference of nested conditional-mean fits, represented sparsely by a centered three-way contingency table. Carrying states on AB cells and sending them to BC cells is a second-order memory/line-graph construction in the broad, established sense. The particular biological cut carrier, occupied-cell implementation, and precise matched test can be useful contributions, but the present evidence does not establish a new general operator, a new principle of graph memory, or a new principle of exploiting persistent ancestral identity.

It would also be inaccurate to call H051 exactly a published nonbacktracking GNN, exactly a Mori–Zwanzig memory equation, or exactly GST. Those models have different constraints and learned operations. Established ingredients strongly limit generic novelty, while leaving a narrower implementation or biological-method contribution possible if decisive utility and scaling evidence are obtained.

The current Relate representation supplies persistent sample witnesses, not authenticated persistent internal ancestors. The inspected official exporter creates new internal tskit rows per marginal tree, and the current loader also discards original internal keys. A cut, clade, node time, packed row number, or inferred similarity is not evidence of persistent biological identity. The biological edge-equality arm is therefore unavailable on this condition as specified.

## Candidate and exact material reviewed

All work and writes for this audit occurred through SSH on anogena under /workspace-SR003.nfs2/ARG-GNN. The only created/edited directory is research/reviews/H051_independent_priority_audit_20261016_v2. No GPU, model forward, model training, scientific benchmark, install, guard/accounting call, admission, process change, ledger edit, commit, or push was performed. No real observation payload, ancestry truth, or checkpoint was read.

The exact file inventory, SHA256 values, and reading scope appear in material_reviewed.v1.json. The candidate sources read were:

- AGENTS.md.
- H051_conditional_innovation_design_20261016_v1/source_handoff.v1.json, REPORT.md, API_INTEGRATION.md, prospective_protocol_v1.json, source_evidence_v1.json, and the complete conditional_innovation_v1.py.
- TGB_trade_H051_training_source_20261016_v1/source_handoff.v1.json and protocol.source_frozen.inactive.v1.json, plus complete model_v1.py and preparation_v1.py. The suggested filename protocol_v1.json was absent. The actual protocol path was resolved from the existing handoff. This is source review, not runtime qualification.
- TGB_trade_PyG_dataset_source_20261016_v1/API_HANDOFF.md, complete trade_tree_core_v1.py and whole_country_tree_encoder_v1.py. It constructs anonymous annual single-linkage trees from causal outgoing profiles, aligns country leaves, and supplies no persistent internal identities.
- Relate ConvertToTreeSequence.cpp lines351–440 and the native loader extract_native_tree_holds_v1 lines176–228, read directly rather than accepting a prior report's conclusion.

For the user-supplied biological papers, the locally acquired primary XML versions were inspected directly: ARGformer abstract and section2.1, ARGMix abstract, graph-transformer/TMRCA methods and training methods, and Irving-Pease et al.'s ancestry-stratified inference section. Whitehouse et al.'s graph architecture and node/sequence GRU methods were also read. The Nature paper's Supplementary Note3 was not retrieved, so this audit does not claim a complete AncestralPaths architecture audit.

For generic operators, full primary sections were read from EDMD, Rosvall et al.'s memory networks, DBGNN, TGN, LGNN, and GST. New public retrievals are retained with URL/status/hash receipts. The EDMD v2 HTML request returned404, after which v1 full HTML was obtained. LGNN v3 is an earlier paper version with different title/author front matter, so the comparison uses the v6 full text and cites that version explicitly. An initially tried PMC URL for Chorin resolved to an unrelated MYC paper and is explicitly rejected as evidence. DOI lookup identified the correct PMC16175, whose accessible page contains the abstract only. Chorin is used only as broad projection-memory context, not as proof that H051 implements the Mori–Zwanzig equation. A trial OpenReview HTML retrieval supplied no usable bibliographic evidence and is unused.

## Exact mathematical comparison

Take a uniformly sampled persistent leaf i in {1,...,H}. A(i), B(i), C(i) are observed partition memberships. L_X is the occupied-cell indicator matrix and D_X=L_X^T L_X. Define P_X=L_X D_X^{-1} L_X^T. This is the orthogonal projector and empirical conditional expectation E[ . |X] in the leaf inner product H^{-1}sum_i u_i v_i. AB and BC denote joint memberships, not products of arbitrary graph adjacencies.

For an AB-cell state x_ab and n_abc=number of leaves in that triple, the implemented output is exactly

    (T x)_bc = sum_a [n_abc / n_bc] x_ab
    (M x)_bc = sum_a [n_ab / n_b] x_ab
    (J x)_bc = sum_a [n_abc/n_bc - n_ab/n_b] x_ab
    carry_alpha = M x + alpha_b J x.

Terms only connect cells with the same b. On leaf-expanded states,

    T = P_BC P_AB
    M = P_B
    J = (P_BC - P_B) P_AB
      = (P_BC-P_B)(P_AB-P_B).

The last equality follows from nesting. It identifies J as cross-propagation between the centered A-detail and C-detail subspaces conditional on B. The sparse weights are a conditional contingency table centered by the conditional-independence table n_ab*n_bc/n_b. This is a descriptive property of the empirical memberships. It says nothing by itself about conditional independence of true ancestry, observation errors, or labels.

One can derive T without a neural-network argument: minimize ||L_BC y-L_AB x||_2^2 over y. The normal equations give y=D_BC^{-1} L_BC^T L_AB x, exactly the triple transport. Restrict the fit to the coarser B indicator basis and lift back to BC to obtain M. J is the difference of these two nested least-squares fits. Thus the conditional-expectation/Galerkin interpretation is exact for these finite indicator spaces. Calling it a newly discovered conditional innovation would overstate the mathematical novelty. EDMD's established dictionary/Gram-matrix least-squares construction is relevant prior art [R5], although H051 is a change of empirical carrier and does not estimate a general unknown dynamical Koopman operator.

P_BC-P_B is an orthogonal projector. J generally is not: it need not be symmetric or idempotent because the two detail projectors need not commute. In the count-weighted cell norm, T and the stated B-constant convex gates are contractions. This does not prove contraction of the GRU, the full trained sequence model, predictive calibration, or a bound on classification error. J=0 as an operator iff the occupied empirical A,C variables are conditionally independent given B. Jx=0 for one learned state is weaker and can occur even when J is nonzero.

For example, with B constant and A=C a nontrivial two-cell partition, J=P_A-P_B is already nonzero without any genealogy or internal identity. Demonstrating that alpha1 differs from alpha0 therefore demonstrates an information-retention effect relative to the coarse fit, not biological novelty.

### Higher-order memory and line graphs

Rosvall et al. explicitly represent a second-order process by memory nodes (previous,current), with directed transitions to (current,next) [R6]. DBGNN already places learned message passing on higher-order De Bruijn graphs whose nodes represent walks and whose edges represent longer walks, with a learned return to original node space [R7]. At the carrier level, H051 AB->BC is the same state-lifting pattern. Its edge weight n_abc is the number of aligned witnesses that realize that triple.

There are useful distinctions. A, B and C are time-typed partition labels, so equal anonymous numbers in different episodes do not identify the same node. H051's weights count simultaneous aligned witnesses across three partitions, whereas temporal-walk papers count occurrences of causal interaction paths. The transport normalizes by n_bc because it pulls old features into each destination cell. It is not literally the forward random-walk normalization n_abc/n_ab. These distinctions define a concrete specialization but do not make pair-state lifting or higher-order weighted message passing new.

A directed line graph connects (a,b) to (b,c). The nonbacktracking matrix used by LGNN additionally requires c != a [R8, section4.2]. H051 has no such exclusion and generally cannot impose it by comparing arbitrary labels from different partition namespaces. Its observed n_abc support can also be stricter than support inferred by freely combining two pairwise overlap edges. It is therefore a weighted witness-supported second-order transport, not exactly Hashimoto nonbacktracking propagation. Adding a nonbacktracking mask would neither be generically novel nor biologically justified by bookkeeping equality alone.

### Dynamic graph memory

TGN already combines event messages, persistent node memories, memory update functions such as GRUs, and graph-based embeddings [R9, section3.1]. H051's topology-derived carrier change is a more specific linear operation than a generic TGN memory update, but adding chronology, persistent leaf memory, a GRU, or a decay scale is established territory. Its exponential fixed-forcing relaxation is the exact solution of dh/dd=(f-h)/ell on one unchanged carrier. That semigroup alone does not prove invariance of a full nonlinear graph model to splitting or reordering observations.

The projection/resolved-detail decomposition also resembles the starting point of reduced-order and Mori–Zwanzig reasoning [R10]. An exact Mori–Zwanzig closure contains evolution-dependent memory/noise terms. H051 has no underlying fine generator, orthogonal dynamics, derived memory kernel, or exact closure identity. It should not be sold as an exact Mori–Zwanzig GNN.

### Graph collections and the user-supplied GST

GST interleaves per-graph message passing, graph pooling, self-attention over graph tokens, return of collection context to nodes, and learned per-channel gates at each block [R4, section3.1, equations1–12]. This is strong prior art against a generic claim that graph-local and collection-wide computation must be interleaved, or that gating graph context is new.

GST treats a set of graphs and does not specify H051's ordered aligned-leaf conditional-overlap transport. Permuting set elements is different from preserving a genomic/year order, and a pooled graph token can lose leaf correspondence. Those differences can motivate a comparison, but merely attaching H051 to a GNN+set-transformer does not establish an advance. A causal adaptation of GST for forecasting would need a disclosed time/causality encoding and equal observed history. An unmodified unordered architecture is not a fair purported test of sequence-order reasoning.

## Biological specialization and the four supplied papers

ARGformer [R1, section2.1] explicitly uses unique inferred-node identifiers as tokens and learns from repeated ancestral-node co-occurrence across samples/intervals. Shared inferred identity and across-interval context are prior art. The paper's representation does not authenticate an identity field that the current Relate input lacks, and a shared inferred ID is not proof of a true ancestor.

ARGMix [R2] is already an ARG-derived graph transformer for local ancestry with ancient reference labels, TMRCA relative encodings, selected reference/internal tokens and held-out replicates. Its methods use a 29-token subgraph in that condition and skip correlated training trees. H051 could be a distinct full-tree ordered carry specialization, but cannot claim that ARG topology, ancient references, graph attention, or deep local ancestry prediction are new.

Irving-Pease et al. [R3] already train a neural classifier on Relate-inferred trees from simulated demographic histories to infer local ancestral paths and apply it to ancient/present-day data. Its reported accuracy belongs to that paper's dataset and protocol, not to H051. This audit did not inspect the supplementary neural-network specification and makes no claim that AncestralPaths lacks every form of smoothing or cross-position processing.

Whitehouse et al. [R11] already combine graph attention with a per-tree node GRU and a tree-sequence GRU. Their methods explicitly discuss exploiting consistency in Relate's node ordering. Treating this baseline as devoid of ordering/identity proxies would be misleading. It does not specify H051's exact contingency-table transport, but graph-plus-GRU processing of ordered tree collections is already established.

A defensible biological contribution would instead be: a precisely specified, source-authenticated bias for transporting ancestry evidence through uncertain changing observed partitions, shown to reduce false short tracts while retaining real short tracts under the same full-tree inputs and inference pipeline. That is a mechanistic utility claim requiring data. The edge-continuity gate currently lacks its prerequisite input on the named Relate condition. Switching to a shared-ID producer is a new observation/pipeline condition that must be shared across all compared arms.

The source audit is decisive about representation, not about biological continuity. The inspected Relate exporter sets node_const=tree_count*(N-1), adds internal node rows inside the marginal-tree loop, and offsets their endpoints. The loader reconstructs compact rows from sorted retained keys but emits only parent/time/sample arrays. This audit found no lossless native persistent-ancestor channel in those reviewed paths. It did not search every possible Relate mode or alternative producer.

## What the pending trade experiment can establish

The frozen source uses 255 catalogue-aligned country leaves, a full anonymous annual hierarchy, a leaf GRU, and a fixed persistence-profile residual prior common to all arms. Its forecast for year tau uses complete edge years <tau. Those are appropriate causal source-level decisions, subject to root runtime/authentication review.

The alpha0 versus alpha1 difference is specifically P_B versus P_BC P_AB at genuine changes of the last two DISTINCT cut episodes. Unchanged annual holds retain leaf state and perform further GRU updates. Thus 'first_order' and 'conditional_triple' are carry-rule names. Both complete models have recurrent history, and the contrast is not a proof that a first-order full model loses to a second-order full model.

Every arm stores [255,64] temporal state. The current test cannot support a compressed-cell memory advantage. All arms share a full profile in the readout prior, so a gain cannot be credited simply to introducing persistence or to preserving the outgoing profile outside the hidden bottleneck.

The primary three-arm campaign is a useful descriptive utility diagnostic, but it omits the fixed-half/count and stable-leaf carry arms needed to determine whether alpha1 wins because of particular second-order overlap rather than generic loss reduction or smoothing. The source already notes half/count as future, not primary, arms. This audit does not change that frozen protocol or authorize adding cells. A follow-on mechanism claim must prespecify those controls independently of TEST outcomes. Existing independent target years remain few and serially dependent. Extra optimizer seeds cannot create new independent years. Country forecasts are a portability diagnostic, not evidence for ancestry, recombination, ancestor identity, or short-tract retention.

## At most two substantive architecture directions

Neither proposal below is claimed unprecedented. Adaptive model reduction and uncertainty propagation have extensive prior art, and would need a separate focused priority search before a novelty claim. They are concrete changes that move beyond simply renaming J, increasing memory order, or adding a learned gate. Each supplies a falsifiable capability target and a stop rule.

### 1. A carrier that preserves a specified amount of predictive detail

The present operator irreversibly deletes within-AB leaf detail before transfer. Replace the fixed AB carrier with a causal adaptive refinement S_t of AB. Store h_hat=P_St h and choose extra splits using the current hidden-state residual e=h-P_AB h plus observed full-tree detail, never labels or future observations. Impose a frozen state-slot budget K and an explicit discarded-energy target ||h-h_hat||_leaf <=epsilon, declaring when K makes it infeasible. Transport the refined leaf function into the next carrier by occupied intersections, then apply the same update/readout. Persistent leaves provide alignment; no internal identities are required.

This changes what information can be represented. It must include a causal selection rule, deterministic tie handling/equivariance, stored metadata, training-gradient semantics, and a cost accounting that counts residual computation and carrier selection. Using all leaves merely recovers the established stable-leaf baseline, so that is not a success.

The decisive experiment is a budget curve rather than one favorable accuracy number. First use a synthetic coarsen/reappear sequence in which a label-relevant within-cut mode vanishes from fixed AB and reappears later. Compare fixed H051, fixed multiscale/higher-order carriers with the same slots, the adaptive carrier, and persistent-leaf memory. Require the advertised per-step residual certificate to hold, and test reconstruction/future prediction on unseen partition sequences. Then use the SAME frozen biological observations and full-tree body with prespecified false-short burden, true-short retention, peak state+activation bytes, and runtime. A possible theoretical contribution is an actual propagation bound from measured projection losses under stated update Lipschitz assumptions. It must not assume a GRU is a contraction without establishing it.

Reject the direction as a contribution if adaptive selection adds no benefit over a fixed multiscale basis at equal total cost, if the certificate fails, if it simply allocates nearly H slots, or if retained signal is purchased by unacceptable short-tract loss. An adaptive refinement algorithm with a verified error/cost advantage would be a substantive improvement over the present hard projection, while 'a multiscale GNN' alone would not.

### 2. Transport correlated structural uncertainty, not just an averaged tree

If a separately authenticated observation-only uncertainty ensemble is available, represent paired histories (T_t^r,h_t^r) with weights w_r. The correct ensemble carry satisfies

    E[T h] = E[T] E[h] + E[(T-E[T])(h-E[h])].

A model using only the mean operator and mean hidden state deletes the second term. Develop a bounded low-rank representation of this operator-state covariance in leaf-aligned detail coordinates, and update it with a specified approximation to nonlinear recurrence. The target is to match a full ensemble at lower state cost while exposing uncertainty in ancestry probabilities. Arbitrary tree perturbations or a bootstrap must not be called a calibrated genealogy posterior.

The first decisive test is a paired-ambiguity construction with equal mean transports and equal mean states but different operator-state covariance. A mean-only model must be unable to distinguish the two cases from those summaries, while the covariance model and full paired ensemble can. All methods receive the same underlying ensemble histories, and controls include an equal-budget generic extra-memory model and a full paired ensemble. Next test realistic prespecified inference-error strata using the same observations, comparing hard H051, mean-operator propagation, the covariance model, and the full ensemble at measured equal cost. Report NLL/Brier/calibration alongside short-tract metrics, not just ancestry accuracy.

Reject the direction if its approximation cannot track the paired ensemble, if improvement vanishes against equal-budget generic memory, or if uncertainty information is fabricated/unavailable. In particular, this cannot proceed on the assumption that current Relate supplies persistent internal identities or a ready calibrated posterior. The proposed advance would be a demonstrated approximation to a specific lost covariance term, not the already-established idea of ensembling or Bayesian uncertainty.

## Evidence limits and claim discipline

This is a targeted priority audit of the exact operator and closest named families. It is not an exhaustive systematic search, a patent search, or a theorem of non-novelty. No absence-of-prior-art claim follows from failure to locate an exact named H051 formula. Conversely, the algebra above is sufficient to classify the operator without finding a paper with identical notation.

Local primary-source snapshots were used for the biological comparisons. Publication titles, versions, relevant passages and hashes are retained in evidence.v1.json. No author code was executed or independently reproduced. The 61 CPU checks in the candidate handoff are engineering evidence reported by that source, not checks repeated here and not model-performance evidence. Runtime behavior, real identity availability, causal payload lineage, full-model stability, gains, parameter/runtime comparisons, and biological calibration remain unverified in this audit. No H051 accuracy, NDCG, tract score, or improvement is invented.

The source review may justify continuing a carefully scoped utility diagnostic. It does not yet justify an architectural priority claim. Future experiments remain root-owned and require existing registration/admission rules. Under the latest user policy only physical GPU0 can be considered; this report grants no execution authority.

## Primary references

[R1] Bonet et al. (2026), ARGformer: learning on ancestral recombination graphs with transformers. Bioinformatics. https://doi.org/10.1093/bioinformatics/btag438 . Read abstract and section2.1 in cached primary XML.

[R2] Shanks et al. (2026), Graph transformer for ancient ancestry inference, bioRxiv v1. https://www.biorxiv.org/content/10.64898/2026.04.05.714076v1 . Read abstract, graph transformer/TMRCA methods and training methods in cached primary XML. Preprint findings are not treated as H051 results.

[R3] Irving-Pease et al. (2024), The selection landscape and genetic legacy of ancient Eurasians. Nature625,312–320. https://www.nature.com/articles/s41586-023-06705-1 . Read main-text ancestry inference section, not Supplementary Note3.

[R4] Escrig Molina, Chen and Probst (2026), Graph Set Transformer, arXiv:2606.05116v1. https://arxiv.org/abs/2606.05116 . Read abstract and full methodology section3.1/equations1–12.

[R5] Williams, Kevrekidis and Rowley (2015), A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition. Journal of Nonlinear Science25,1307–1346. https://doi.org/10.1007/s00332-015-9258-5 . Primary full preprint https://arxiv.org/html/1408.4408v1 , especially introduction and section2 least-squares/Galerkin derivation.

[R6] Rosvall, Esquivel, Lancichinetti, West and Lambiotte (2014), Memory in network flows and its effects on spreading dynamics and community detection. Nature Communications5,4630. https://www.nature.com/articles/ncomms5630 . Read memory-network representation and transition discussion.

[R7] Qarkaxhija, Perri and Scholtes (2022 preprint), De Bruijn goes Neural: Causality-Aware Graph Neural Networks for Time Series Data on Dynamic Graphs. https://arxiv.org/html/2209.08311v1 . Read definitions of higher-order De Bruijn graph and section3 message passing/bipartite projection.

[R8] Chen et al., Supervised Community Detection with Line Graph Neural Networks. ICLR2019; full arXiv revision v6 (2020). https://arxiv.org/html/1705.08415v6 . Read section4.2 nonbacktracking edge operator and joint node/edge GNN. The current arXiv abstract metadata and fulltext disagree on the second author's name; this audit avoids resolving that discrepancy by inference.

[R9] Rossi, Chamberlain, Frasca, Eynard, Monti and Bronstein (2020), Temporal Graph Networks for Deep Learning on Dynamic Graphs, arXiv:2006.10637v3. https://arxiv.org/abs/2006.10637 . Read full cached section3.1 core modules.

[R10] Chorin, Hald and Kupferman (2000), Optimal prediction and the Mori–Zwanzig representation of irreversible processes. PNAS97(7),2968–2973. https://doi.org/10.1073/pnas.97.7.2968 . Verified bibliographic record and accessible abstract https://pmc.ncbi.nlm.nih.gov/articles/PMC16175/ . Full mathematical text was not available in that HTML view, so this is contextual only.

[R11] Whitehouse et al. (2024), Tree Sequences as a General-Purpose Tool for Population Genetic Inference. Molecular Biology and Evolution. https://doi.org/10.1093/molbev/msae223 . Read cached primary XML section GCN Architecture and Training, especially the node and tree-sequence GRUs.
