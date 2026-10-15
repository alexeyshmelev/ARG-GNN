**H048: independent novelty and cross-domain assessment, 15 October 2026**

**Assessment.** H048 is a coherent, registered local-ancestry utility experiment. It does not establish a new general graph-collection operator, and no real H048 training/accuracy result was available to this review. The defensible question is whether tree context works better as marker-grid persistence than as local confidence. The registered decision/current producer supersede the older note’s “unregistered” status. [Decision snapshot](decision_snapshot.json), [producer protocol](../../../src/arggnn/train_marker_grid_reset_chain_v1.py#L45).

**Actual operator.** The inherited whole-tree PyG/SNP/BiGRU body supplies ancestry logits. Its first 32 classifier-input channels are second-sweep tree context, while the remaining channels contain genomic GRU history. H048’s rate head uses those first 32 channels, not the GRU history directly. It symmetrically combines the two containing-hold contexts as `[mean(cL,cR), abs(cR-cL)]` and produces one scalar reset rate. Disjoint-bin SNP evidence supplies a common emission correction. Both arms start identically with 31,933 parameters, but their independently trained contexts need not remain numerically identical. Equal capacity and available information are controlled, not an immutable shared learned signal. [Core, lines 187–255](../../../src/arggnn/marker_grid_reset_chain_gnn_v1.py#L187), [inherited state, lines 297–310](../../../src/arggnn/marker_resolution_readout_v1.py#L297).

The rate is constant for a repeated containing-hold pair. Within-hold marker localization comes from bin emissions and chain inference, not from a rate head seeing distinct local marker features at every edge. The chain nevertheless permits within-hold switches.

**Exact RFMix relationship.** RFMix’s Methods subsection “Learning the θT Parameters” gives the unnumbered adjacent-window joint formula, HTML equation `ufd2` / MathML `M11`:

    J[j,k] = q[j] * { exp(-d*G)*1[j=k] + (1-exp(-d*G))*q[k] }.

Its row marginal is q[j]. Dividing by that marginal gives

    P[j,k] = r*1[j=k] + (1-r)*q[k], r=exp(-d*G).

This is exactly H048’s **conditional transition family**, with `q=pi` and `d*G=lambda_bq*delta_b`. [RFMix primary subsection](https://pmc.ncbi.nlm.nih.gov/articles/PMC3738819/#sec2.1.3.2), [formula](https://pmc.ncbi.nlm.nih.gov/articles/PMC3738819/#ufd2), [H048 lines 158–184](../../../src/arggnn/marker_grid_reset_chain_gnn_v1.py#L158). The complete CRFs are not identical: RFMix presents a joint pair potential including q[j], uses forest/window evidence and genetic distances, while H048 uses conditional matrices, initial pi, graph-conditioned rates and physical bp. Those differences do not create a new reset law.

There is a further expressiveness reduction. Let `Pi=1*pi^T`, so `Pi^2=Pi`. All generators are `lambda_b*(Pi-I)`, hence the transitions commute and

    product_b P_b = exp(-sum_b lambda_b*delta_b)*I
                    + (1-exp(-sum_b lambda_b*delta_b))*Pi.

The transition process is therefore reset smoothing under a graph-dependent cumulative coordinate/time change. It shares one destination mix, obeys detailed balance with pi, and cannot express independently directed class-pair dynamics. Interleaved emissions still make the full posterior depend on observation order. This reduction neither permits arbitrary sequence permutation nor proves whole-GNN reversal or grid-refinement invariance.

**Nearest prior art.**

| Primary evidence | Consequence |
|---|---|
| [RFMix, Methods / Theory, Inputs and Windowing, LAI, θT](https://pmc.ncbi.nlm.nih.gov/articles/PMC3738819/) | Reference emissions, disjoint windows, reset-family coupling and forward-backward ancestry smoothing are established. |
| [Reeves & Bhat, Neural Continuous-Time Markov Models, abstract and §3](https://arxiv.org/html/2212.05378v1) | Neural nonlinear rate dependence on external covariates is established. Their fully observed system-identification task differs from this latent-label decoder. |
| [ARGMix, Methods §4.1 / §4.3](https://doi.org/10.64898/2026.04.05.714076) | Reference-informed marginal-tree ancestry with TMRCA attention exists. Their 29-node representation differs from H048’s complete-tree body. |
| [ARGformer §2.1](https://doi.org/10.1093/bioinformatics/btag438) | Learned shared genealogical context across samples/intervals exists through path tokens, masked pretraining and contrastive learning. |
| [AncestralPaths, “Ancestry-stratified selective sweeps” / Supplementary Note 3](https://www.nature.com/articles/s41586-023-06705-1) | Supervised ancient path ancestry from inferred genealogies predates H048. It does not establish this marker-grid decoder. |
| Completed [H039 decision](decision_snapshot.json), [source lines 152–181](../../../src/arggnn/directed_clade_exchange_crf_v1.py#L152) | Project precedent already conditions native-hold CRF transitions on graph structure, distance and directed clade witnesses. It completed 320 updates/arm. Selected accuracies: directed 95.7119%, plain 95.5635%, generic 95.4964%, eight reused VAL parents / one optimizer seed. |

The unresolved distinction is **marker-resolution emissions plus tree-conditioned reset persistence versus matched own-confidence use**. The exact composition’s publication priority is unverified. A successful pair supports specialized utility, not a new transition principle. The source’s existing no-priority boundary is appropriate.

**GST reaction-center data is incompatible with this chain hypothesis.** GST §3.1.1 “Reaction centre prediction” defines an *unordered set of reactant molecular graphs*, with a binary target *per atom*: positive if its hydrogen count changes or it touches a formed/broken bond. The released parser builds labels from edit lists and input features from reactants. [Primary definition](https://arxiv.org/html/2606.05116v1#S3.SS1.SSS1.Px4), [author code, lines 22–47 / 190–256](https://github.com/daenuprobst/gst-conference/blob/main/scripts/eval_rxn_center.py#L190), [cached code](gst_eval_rxn_center.py).

There is no native time/chromosome order, gap metric, stable per-query categorical trajectory or persistence target. Sorting atoms/reactants by SMILES, atom-map ID or input index would invent a chain and violate the intended permutation semantics. Product-derived edits are supervision and cannot determine an input ordering or gate. Molecular bonds define a graph, not H048’s unique physical chain. GST is relevant prior art for collection-context exchange (§3.1 Steps 2–5), but USPTO-15K is not a justified H048 transfer benchmark.

**Cross-domain recommendation.** No new **whole-GNN** non-ARG transfer run is justified for H048 as implemented. The abstract K-state decoder applies to ordered observations with persistent categorical states. The complete implementation binds ancestry-specific trees, the 952/686 panel, reference alleles, four classes and 100-bp geometry. Replacing these with a domain encoder would test a new encoder plus an established sequence decoder, not H048’s whole ARG GNN or a general graph-collection mechanism.

The sole plausible **future decoder-only** benchmark examined is **PKU-MMD Phase I, skeleton-only continuous action detection**. Native inputs are 30-FPS streams of 25-joint 3D skeletons for up to two subjects. Supervision is `(action label, start, end, confidence)` intervals. This provides observed body graphs, real time and persistent action states. The author project publicly links data/evaluation resources. [Paper §§3–4](https://arxiv.org/html/1703.07475v1#S3), [author Data Format / Metrics](https://struct002.github.io/PKUMMD/), [author README](https://github.com/ECHO960/PKU-MMD#data-format).

That benchmark could support only context-conditioned temporal smoothing of action detections. **It cannot support the proposed whole-GNN claim. Do not build an unrelated skeleton model for a cross-domain table.** If an independently motivated skeleton project already exists, use the identical encoder in both arms, generalize decoder/state count/rate units, retain native interval AP and author splits, and establish that interval overlap/multiplicity fits one categorical chain without dropping labels. The public schema alone does not prove exclusivity. No annotation payload was read or data download exercised. No transfer run is recommended before that semantic condition is resolved. Manufacturing frame accuracy from conflicting intervals would change the native task.

**Two concrete scientific deficiencies / research requirements.**

1. **Scalar persistence is not directed interaction.** Complete the registered pair and report paired parent effects, boundary/short-tract effects, and whether learned rates vary or merely shift global smoothing. If useful, a separately frozen constant-rate/rate-reassignment diagnostic could test the importance of rate placement. A full learned generator alone is not a distinct operator: H039 already has directed class-pair potentials, and neural CTMCs already learn covariate-dependent rates. Any extension needs an observed graph-structural failure and a non-reduction beyond that substitution.

2. **The new route is one-way context→decoder.** Graph states precede chain inference. Posterior task-state uncertainty never modifies tree computation through the new mechanism. A genuinely coupled graph-collection proposal would need an explicit update of graph/task-state uncertainty, evidence that this feedback resolves an observed failure, and a matched one-way encoder+chain comparison. Raw descendant overlap is already reduced to an existing incidence kernel in [the no-repeat audit](../internal_tree_transport_hypothesis_source_v1/equivalence_no_repeat.md). Appending GRU or cross-graph attention is not novelty either: [GMN §3.2](https://proceedings.mlr.press/v97/li19d.html) and [GST §3.1](https://arxiv.org/html/2606.05116v1#S3.SS1) establish foreign-node/collection-context messages. No distinct new coupled operator is established or proposed as publication-ready here.

Neither deficiency warrants changing the frozen H048 pair. Apply its registered ≥0.10 percentage-point accuracy gain/no mean boundary-F1 loss criterion, or separately labeled boundary-only alternative. Eight reused development parents and one seed remain selection evidence. Arithmetic checks are not efficacy or biological-rate validation.

**Limits and receipts.** Bounded independent assessment, not exhaustive priority search. Existing cached primary sources and [retrieval receipts](../ordered_operator_novelty_independent_20261015_v1/retrieval_receipts.json) were read directly. [New receipts](retrieval_receipts.json) preserve failed URLs as well as successes. [Evidence manifest](evidence.json) binds inspected sources and decision snapshot. Reads were instructions, sources, decision fields and public papers/author code only. No scientific observation/checkpoint/truth payloads, model execution, GPU/admission/guard actions, installs, Git, or other workers’ source edits occurred. Only this review directory was written. Root owns active execution and publication.
