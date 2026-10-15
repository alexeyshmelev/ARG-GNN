# H044 marker-resolution readout

Source-only implementation. Root registered H044 on2026-10-15T03:01Z before implementation. Resource admission and execution remain root-owned. The scientific module is src/arggnn/marker_resolution_readout_v1.py. Separate source-only preparation/training/root integration lives under research/reviews/H044_marker_resolution_root_v1 and new dedicated arggnn modules. Original H042 scientific sources and worker pins remain unchanged. No real observations, targets, checkpoint payloads, runtime/GPU/process state or guards were read or executed. The fixed H042 completion JSON was read only for exact pass30 metadata.

The hypothesis is that observed SNP evidence localized within native trees can improve boundary behavior that a per-tree constant readout cannot represent. H042's reported96.5778% span accuracy and18.0451% boundaryF1, plus the fixed H038 geometric diagnostic, motivate this development hypothesis. They do not establish an H044 gain. H044 is designed after those diagnostics and must not be described as an independent confirmatory use of the same eight VAL parents. Outer remains unread.

## Scientific contract and corrections before admission

The initial binary-emission idea was corrected before any H044 corpus access: the frozen bank permits categorical alleles0..3. We retain every valid allele and use0.5 Jeffreys pseudocount per original per-site allele cardinality, preferably supplied from observed.allele_count. Explicit all-observed-panel support derivation is also implemented; it is a named encoding choice, not a silent fallback or fitted filter. For two alleles the formula is exactly the Bernoulli rule.

For called query allele a and known-reference class c, evidence is log((n_ac+.5)/(n_c+.5*A))+log(A). A class with no called references contributes exactlyzero evidence and count; missing queries do the same. Known reference labels are the existing observed reference_class values, and every scored query must have reference_class=-1. No ancestry target, oracle, learned emission table, imputation, purity filtering or SNP dropping enters preparation.

Two prefix sums produce average evidence and effective marker counts inside prespecified closed physical boxes±1kb/5kb/20kb at every original100bp center. The six per-class channels are three averages and three log1p counts. Physical box indexes are computed once by searchsorted. Counts contract over the fixed four allele/category channels; there is no marker-by-marker Python recurrence or dense marker affinity.

The tree_pooled control takes the physical-bin-width weighted mean of these exact same six features over centers having the same original native readout_hold, then broadcasts before the identical decoder. Thus both arms have the same SNP observations, feature definitions, parameters, initialization and training schedule; only intra-tree localization differs. Pooling allocates only scored holds, while the backbone retains every original tree.

The shared classwise decoder is Linear38to16 with bias, SiLU, Linear16to1 without bias:32 leaf-context channels plus six evidence/count channels,640 parameters. Its zero final projection makes both arms initially identical to H042pass30. Absent-class/all-missing feature channels cannot receive a residual solely from context. The complete original H042 body, local clade adapter, two full PyG sweeps, classifier and genomic GRU remain. Both arms fine-tune all H042 weights; frozen refers to source/architecture, not frozen learned weights. There is no detached history, pruning, recurrent reset or new temporal model.

## Concrete API

- MarkerCoordinatesV1 transports exact original physical SNP positions, scoring centers/edges, native intervals, optional original allele cardinality, and a root-verified identity binding.
- prepare_marker_evidence_v1(bank, sidecar) requires the unchanged full952/686 CPU bank, original loader source ref, original observed-entry ref and exact ordered SHA/dtype/shape identities. The module computes actual identities over supplied arrays without a payload-sized bytes copy.
- MarkerEvidenceBankV1.features_for_queries gives[P,B,4,6] feature blocks. pool_within_trees is the matched control. No caller should cache every full-grid feature for every parent blindly.
- bind_marker_bank_v1(bank,markers) verifies exact original CPU genotype/reference/query/site/readout identities and actual shared marker tensor objects. It returns H044BoundMarkerBankV1, whose typed .to(device) carries this pairing to the device. Forward requires this bundle. Torch version stamps detect normal in-place mutation/replacement; they do not detect NumPy-alias or .data writes. Root/consumer must preserve non-mutating tensor aliases and metadata.
- MarkerResolutionReadoutV1(arm=localized or tree_pooled) emits original[P,686,4] logits. Query/center chunks and nonreentrant readout checkpointing preserve full gradients and never truncate the genomic GRU.
- make_matched_pair_v1 loads the exact SHA-bound H042pass30 model state and deep-copies the complete backbone/decoder initialization into independent arms. Optimizer states are not loaded.
- original_grid_span_nll_v1 takes external complete100bp target span masses only at the TRAIN loss boundary. Targets are never model/feature inputs.
- logical_bounds_v1 reports explicit tensor-byte formulas from actual counts and genotype dtype. It is not runtime/allocator feasibility evidence.

The original observation bank does not retain SNP positions or per-site cardinality. Root must build a separate sidecar from unchanged load_full_parent_v1 outputs positions_bp0, intervals_bp, genotype, allele_count and the original scoring geometry, aligned with bank.site_hold/readout_hold and persistent query/reference columns. Original allele_count is mechanically converted to int64 for the sidecar and its identity recorded. Do not infer positions from tree membership or reuse a different VCF row order. sidecar_binding_inactive.json shows the required metadata; all unknown identities remain null.

## Fixed training and selection

Warm start is completed H042 exact_clade_flags.pass_30.pt SHA520fcffa1d5e85603620a7f4688b7fdf46669686fea7960fc311852ff2f7e366. Both arms start from this same fixed checkpoint, with independent reset AdamW moments/steps, FP32 seed101, LR.001, weight decay.0001, betas(.9,.999), epsilon1e-8 and norm1 clipping.

Each arm gets30 complete passes over original32TRAIN parents,960 full-parent updates, checkpoint passes5/10/15/20/25/30, no early stopping and original eight-VAL selection: earliest maximum equal-parent Present182 exact span accuracy, then lower NLL, then earlier pass. BoundaryF1 remains a secondary diagnostic; do not choose a checkpoint on it. All96 arm/checkpoint/VAL probability products must be immutably sealed before any VAL truth in the new evaluation phase. Outer stays unread.

Localized logits require FullPanelExternalTargetV1.target_mass on the original100bp bins. H042 native_hold_train_mass_v1 is invalid here because it erases intra-tree target localization. Both matched arms use the same complete-parent mass and normalization. Root approved streamed or bounded-LRU targets: construct one complete parent mass and share it between arms for that step; do not retain32 dense masses or compress into holds. Full target construction remains bookkeeping with transient FP64, while model, gradients and final masses are FP32.

## Exact parameter and RAM bounds

There are30,788 existing H042 parameters plus640 new parameters,31,428 per arm (125,712 FP32 parameter bytes). The tiny test measured640 new parameters; full H042 wrapper construction is not run in this source task. Source asserts the exact full count on future construction.

At P<=20,000, Q686 and C4, each dense FP32 logits/probability/target tensor is219,520,000bytes. All2arms×6checkpoints×8VAL predictions require21,073,920,000bytes before container/transient overhead. The root-proposed32,000,000,000byte prediction cap must be separately admitted. H042's old8GiB prediction cap is insufficient even for one full-grid48-product arm. No coverage cut or native-hold compression is used.

A complete-parent FP32 target is at most219,520,000bytes, with439,040,000byte transient FP64 construction. Keeping32 FP32 targets would require7,024,640,000bytes, exceeding the old4GiB target cap; streaming avoids that residency.

A charged marker entry has genotype_bytes*S*H +76*S+76*P+8*Q tensor bytes, including borrowed genotype, class evidence/call masks and exact box/pooling indexes. Root can deduplicate genuinely shared storage, otherwise charge it. The proposed32–48GB bounded feature cache needs actual S, dtypes and retained-entry counts; none were read here. The observation-only evidence prefix and window subtraction use FP64 to preserve weak late windows after large prefix totals, then cast average evidence to FP32. Counts remain exactly represented FP32 integers. Mixed evidence/count prefixes require12*(S+1)*B*C bytes, and a full query feature block requires4*P*B*C*6 bytes. One-time coordinate lookup costs O(P*R*logS); evidence and all blocks cost O(S*H*C + S*Q*C + P*Q*C*R), fixed C4/R3.

Readout checkpoint saved selected-state/feature inputs can themselves reach4,829,440,000bytes at the full grid, plus output/concat, prefix/pooling temporaries, complete H042 body/autograd and allocator costs. Original exported state is4*T*952*64 bytes. GPU residency must have a separate actual cap and qualification; these exact named tensor formulas do not establish total peak memory or training time.

## Tiny verification and limits

Bounded CPU synthetic checks used5 sites,6 haplotypes,2 queries,60 centers and2 native holds. It verifies direct categorical sum parity, binary equivalence, allele3 preservation, missing/absent zeros, within-tree variation, pooled constancy/weighted-mean conservation, declared versus observed support, empty sites, zero residual initialization,640 parameters, finite final-projection gradients, expected zero first-layer gradients at zero output projection, original-grid loss and RAM formulas. An isolated100,000site/6haplotype prefix algebra case additionally checks cancellation correction, final FP32 features, and ordinary Torch mutation-stamp rejection. No native H042 model or actual data was constructed.

Full original-sidecar reconstruction, strict checkpoint payload restore, H042 zero-residual end-to-end parity, actual complete-parent gradients, performance and resource peaks remain future admitted qualifications. The separate root source package adds minimal preparation/paired worker/context/driver integration using the existing current guard. This scientific source task performs no admission or launch and changes no guard.

There is no novelty claim for emissions, pseudocounts, prefix sums, box filters or residual heads. The intended empirical test isolates whether intra-tree SNP localization improves this existing backbone under matched observation and optimization budgets. Exact architectural/literature novelty requires a separate assessment; this source task establishes no performance or biological mechanism result.
