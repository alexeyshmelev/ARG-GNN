# Packed same-marker successor: independent-review handoff

Status: **SOURCE ONLY, unexecuted, unregistered.** This replaces the unusable layout/dispatch engineering of the preserved `source_handoff/` artifact. It retains the same pending scientific proposal, query-allele-before-pooling statistic, reference/leaf identities,714 parameters, H045 splice and full-budget protocol. It introduces no new experiment ID or operator claim. `PROTOCOL_source_only_v2.md` is the compact scientific protocol. H057 remains untouched.

## API for the separately root-owned one-hold qualification

Core source: `same_marker_packed_csr_v2.py`. It has no filesystem reader, optimizer or job logic. The qualification runner first extracts one complete original TRAIN hold and remaps its parent/sample rows locally, then calls:

    packed = prepare_packed_parent_v2(
        parent=parent[V], sample_index=samples[1,952],
        site_hold=zeros[B], genotype=gt[B,952],
        reference_class=observed_class[952], sample_age=observed_age[952],
        query_column=actual_queries[686], route_permutation=identity_or_null,
        allele_count=original_cardinality[B])
    hold = packed.holds[0]
    evidence = PackedSameMarkerEvidenceV2(marker_block_size=16)
    features, mask = evidence.marker_features_v2(log_gamma[V], gt[B,952], hold)
    pooled = evidence(log_gamma[V], gt[B,952], hold)

These are illustrative signatures, not executed code. Per-marker outputs are `[B,686,24]` and a same-shaped boolean mask. Channels0–7 are matching-frequency contrasts,8–15 relative log called masses,16–23 availability, in youngerC4 then olderC4 order. The first16 channels require a called query and called reference group. Availability itself is averaged over called query sites, using its own mask. Pooled output `[686,48]` is mean24 followed by max24, excluding invalid values per channel. Public marker access is limited to1..16sites. Group absence is separate from numerical underflow. All four allele slots remain explicit. Root parent=-1 and its log attenuation is ignored.

`PackedSameMarkerAdapterV2.forward_native_hold_v2(cpu_hold, global_upward[N,32], geometry[N,4], support[N,5], genotype[S,952])` returns `[686,32]`. For a remapped one-hold qualification, N=V and S=B. Whole-parent arguments in eventual integration are existing shared observation/state references. The optional learned gate initializes exactlygamma=.95, with bias about-5.74, NOT the midpoint of the numerical log bounds. This deliberately starts with broad related-reference support. It is a hop kernel, not an evolutionary posterior, and remains unary/depth dependent. Zero output preserves the initial H045 predictor.

The previously reviewed null builder remains `source_handoff/same_marker_allele_support_v1.py`, SHA256 `a933d39033312acbccc74f63e5c76789b220ae530d983d2935d81fd681e37490`, function `build_alignment_null_v1(reference_class,sample_age,diploid_id,phase,seed=101)`. Only call its null builder after review, never its old full-parent layout. Observed sample IDs can be mapped deterministically to integer diploid joins. The mapping is not a model input. Keep whole diploid sequences, source/age strata and the same permutation at every SNP. The actual olderAna group contains one immovable diploid.

## Packed dispatch and checkpoint lifetime

CPU preparation uses contiguous int32 child CSR and contiguous height/edge-group arrays. There is no Torch tensor per graph node. The source returns15 index tensors per hold, not one per native vertex. Upward PyG sums and downward cavity updates are batched by height. Prefix/suffix sibling exclusion uses a grouped parallel log-add scan, avoiding subtraction and per-node GPU dispatch. For degreeD it costs O(E*F*logD); the recorded original trees have1903vertices, but no binary-only restriction is introduced. Equal evidence is summed without selecting a winning branch. Batched grouping changes floating reduction order relative tov1, not the declared all-path real-arithmetic formula.

There is deliberately no whole-parent GPU `.to` method. Only one hold’s11 packed index vectors and source-group one-hot tensor are created as a transient device view. CPU child CSR is not copied to GPU. Both whole-hold AND marker-block nonreentrant checkpoints are mandatory in gradient-enabled use. Their closures capture CPU layouts, not device views. Whole-hold checkpointing discards gathered gate inputs across holds; inner block checkpoints prevent backward recomputation from retaining every node×marker block at once. Root qualification must verify this intended lifetime in the actual Torch version. Checkpointing preserves gradients and changes recomputation cost, not the scientific observations.

## Exact integration binding

`observation_binding_v2.py` supplies a future root setup step. `bind_current_observations_v2` requires the CPU route inputs, the exact original CPU/device bank objects, original CPU/device episode tensors, observed roster, authenticated entry/cache/site-sidecar refs and a root full-parent layout-resource-proof ref. It compares original observation tensors in chunks of at most1,048,576elements, without a full duplicate GPU bank. It checks source/date/query columns against the actual roster, complete diploid phase assignments, all topology/site/genotype/branch-feature fields, original episode bytes and immutable stamps. The root reader must authenticate the external refs. A metadata path/hash alone is not authority.

`integration_hooks_v2.py` requires that binding and validates the exact bank/episode/route identities before using a payload. Same holdcount/span is insufficient. Construct the payload AFTER original bank/episode device transfer and binding. Its `.to` intentionally raises, because moving an already-bound payload would invalidate its exact device identity. New whole-parent route metadata remains CPU resident. The inherited backbone still uses its original single-parent GPU bank; this successor adds no duplicate treebank layout or message bank.

## Storage ledger, including backward

These are source/metadata calculations and engineering allowances, not measured peaks. The maximum TRAIN metadata are T1005/N1,912,515/V1903/H952/Q686/R266/S13801. The original H050 continuation first step measured14.621GB allocated/20.451GB reserved. That historical figure does not bound a later step or this model.

| Allocation | Count / approximate upper metadata size |
|---|---|
| New packed CPU indexes, all holds of ONE parent | Exactly `4*(7*N + T*(2*H+3*Q+3*R-1)) + 4*T*Q` bytes,75,439,320bytes at these bounds. Default numeric cap128MiB. |
| CPU Tensor descriptors |15*T+1 objects,15,076 at T1005. Python level metadata and preparer integer objects also count toward RSS. No million-object child-tensor bank. |
| Named CPU preparation allowance | `80*N+32*T*H`, about184MB at bounds, cap512MiB. This is not a measured complete Python RSS certificate. Cache at most ONE parent’s new packed route. |
| Transient one-hold device indexing | About0.11MB of11 int64 vectors plus group one-hot; gate and inner-block recomputation can briefly hold two such views. |
| One FP32 marker message `[V,B,8,4]` |3,897,344bytes at B16/V1903. Seed, up, outside, frontier, scan and gradient temporaries coexist. |
| Numerical forward/backward working allowance |32 message-sized FP32 buffers is about124.7MB, plus masks, PyG scatter workspace/indices and query/gate tensors. Plan0.25GB for the isolated numerical route pending measurement. This is NOT a hard peak proof. |
| New query residual outputs and concatenation | Two `[T,Q,32]` fields can coexist transiently, about176.5MB total. |
| Additional functional scatter and row join | One extra `[N,32]` copy can coexist with the old injection, about244.8MB, plus5.52MB int64 query rows. |
| Checkpoint/backward bookkeeping | Shared original input references do not copy the bank. Reduced block/gradient/autograd graph metadata and recomputation workspaces still count. Nested checkpointing does not make these zero or prove end-to-end RSS. |
| New weights/grad/AdamW |714scalars, negligible beside the fields above, but included in actual process peaks. |

The visible route arrays suggest an additional GPU planning margin on the order of1–2GB above the backbone, rather than a node×whole-parent-site tensor. This is an estimate with explicit unknowns, not an admission amount, measured saving or guaranteed fit. Python/autograd metadata, allocator fragmentation and backend workspaces can invalidate it. Root must establish the full-parent layout/resource proof and measure full first scheduled forward/backward/optimizer peak before continuing. The current original16GB CPU-bank and8GiB GPU-residency caps concern particular stored components, not complete model peaks. Only physicalGPU0 is eligible later.

The independently specified bounded qualification is one CPU FP32 observed TRAIN hold, no truth/optimizer/VAL/checkpoint,600s/8GiB under root authority, as in `same_marker_allele_support_independent_review_20261016_v1/QUALIFICATION_PLAN.md`. It cannot authorize full-parent training or supply ancestry utility. After source/numerical/layout gates, the actual utility pilot remains two new seed101 arms of30passes/960updates EACH, with certified H050 continuation reused; no extra seeds before a useful pilot and no two-update substitute.

Only AST parsing and source/JSON-metadata reasoning were performed here. No Torch/PyG/project import, tensor operation, numerical test, dataset-array/checkpoint/truth load, job, accounting, Git or live-source edit occurred. Every source file needs independent review and no success result is claimed.
