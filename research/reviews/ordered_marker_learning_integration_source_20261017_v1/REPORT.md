# Minimal ordered-marker learning integration

17 October2026. **Complete forward-path source is implemented for a scientifically narrow matched pair. It is unexecuted and not registered or ready to launch.** The positive/log kernel from the accepted FP32 feasibility work now has an observation-plan bridge to the actual H058 packed layout. The new model reads unpooled marker evidence before reduction and places an identical learned residual either before or after the second full-tree pass. This is not an ancestry, speed or novelty result.

## Exact model and scientific comparison

Both arms retain the current H058 own-state backbone:31812 H045 weights, the714-weight same-marker gate/rank residual, its common mean/max injection, both complete PyG tree traversals, every sampled identity, the chromosome-wide genomic BiGRU and the original classifier. No original tree is replaced by a pooled graph token.

A new1440-weight module reads each hold's raw marker evidence in its existing chronological order: Linear24→8(no bias), tanh, one bidirectional GRU8→8 per direction, concatenate its two final states16, Linear16→8(no bias), tanh and a zero-initialized Linear8→32(no bias). Its ordered32-vector is zero when a query has no available source evidence in the hold. Both models have **33966 parameters analytically**, pending actual instantiation checks. All weights remain trainable.

| Stage | Shape |
|---|---|
| Complete first tree state | [N_total,32] |
| Original site-independent hold gate | [N_t] |
| Unpooled kernel evidence/mask | [S_t,686,24] |
| Marker GRU input / bidirectional states | [S_t,686,8] / [2,686,8] |
| Existing H058 pooled mean/max and common residual | [686,48] / [686,32] |
| New ordered residual | [686,32] |
| Complete second sweep sampled context | [T,952,32] |
| Original local+genomic classifier input / native logits | [T,952,96] / [T,952,4] |
| Original query/scoring lookup | [R,686,4] |

`ordered_marker_tree_feedback` adds the new vector at the same query-node rows as the common H058 residual before sweep2. `ordered_marker_sample_readout` adds the identical vector to the corresponding sampled contexts immediately after sweep2, before the original genomic BiGRU. Capacity, marker evidence, inherited weights, source/date masks, output support and training budget match. Placement is the only new arm-specific operation. Because the graph path is nonlinear, equal initial module weights do not imply equal learned representations later; that is the intended comparison.

This tests whether ordered information helps more *inside* the complete tree computation than at a one-way sampled readout. Both arms already have ordered evidence. A win does not isolate the general value of marker order itself. Existing H058 aligned can be a lower-capacity representation reference after exact ROOT attestation/completion; its714-only architecture is not a matched-capacity control. The frozen H058 condition is not altered or replayed by this source.

## Biological and output boundaries

The new route retains order of reference/query identity-by-state support before a hold reduction. It does not retain a latent donor-copy path, infer genuine mutation/recombination identity or produce a calibrated population-path posterior. The group age350 threshold and proxies remain modeling choices. Ordered query evidence can influence other sampled nodes through sweep2, but correlated evidence is not treated as independent likelihood factors.

The marker GRU resets at each genuine native hold. It uses observed row order, not physical SNP gaps or new coordinates. The original genomic BiGRU continues across all holds for every sampled identity. This deliberately minimal experiment keeps native-hold-constant predictions and cannot localize a switch inside a hold. It can only change which hold edges receive different calls. A finer marker readout would require a separately frozen coordinate/target/support interface and is not claimed here.

## Exact kernel and interface bridge

`prepare_positive_parent_v1(route,cpu_bank)` accepts only supplied original observation tensors and the exact existing packed route. It checks their original stamps, reads all routed known-reference calls, counts called alleles by the original8 source×age groups and builds positive homogeneous-reference subtree covers. Singleton sampled references handle differing internal-node observations. Missing regions seed no allele. No query genotype chooses a cover and no query ancestry enters any object. Forced-dense sites do not retain unused covers; the dense branch still reads every original call.

`attach_positive_plan_v1` preserves the actual H058 v3 bank/roster/site binding. Forward rechecks that binding and plan versions before adding residuals. The same learned73→1 gate sees the same edge geometry/support and first-sweep endpoint states. The sparse branch uses positive group/path log sums; a16-marker block containing a structurally expensive cover uses the actual H058 parameter-free PyG dense evidence API instead. Sparse/dense semantics include:

- Same child gamma in both directions, without a degree divisor in this added kernel.
- Known pure reference seeds only; group=class+4×1[age>=350]; original reference sequence routing.
- Original unpermuted query genotype and **outside[query_node]**, excluding reference descendants of an internally sampled query.
- All-reference-group called allele counts for ordinary frequency and log-mass normalization, four explicit called allele slots and missing masks.
- Current H058 safe_logD=0/valid=1 behavior when group calls exist but a query has zero outside mass. It remains an unusual edge semantic, not a calibrated likelihood.

The new payload is a subtype of the actual `PackedSameMarkerPayloadV3`, so existing readout shape checks remain applicable. It adds no file reader, truth adapter, optimizer, launcher, resource guard or authority. The current accepted model's stage checkpointing is retained; the added whole-hold checkpoint returns only common/ordered query residuals and recomputes the full marker route without detaching gates.

## Costs and concrete remaining blockers

Raw reference access remains O(S·R), in addition to the existing complete S×H observations. Straightforward exact cover preparation costs O(sum_t S_t N_t), with transient CPU child/depth lists. The persistent plan retains parent tuples, only sparse-eligible covers, count tensors and original layout stamps. Python-object overhead and full-parent peak are unmeasured. The learned fields are regenerated from current first-sweep states/weights; changing trees or gates does not permit an unqualified cache.

The dense route has the existing per-marker full-tree cost. Sparse positive evaluation still materializes query evidence and walks query paths; it has no proved practical speed advantage. The structural fallback proxy may select dense for most actual holds. Marker recurrence adds O(S·Q) state work at fixed width8; whole-hold [S_t,Q,24] features and GRU activations coexist during checkpoint recomputation. Each tree still receives both original neural traversals. Costs include preparation, transfers, both sweeps, fallback and backward. No lower sampling, fewer trees, calls or queries is permitted if fit fails.

This integration has **not** inherited a full-parent numerical or memory PASS from the earlier kernel toy. Actual source import/instantiation,33966 state count, warm-start equality to practical FP32 tolerance, new-payload joins, checkpoint recomputation and complete forward/backward lifetime remain unverified. The prior toy tested the kernel algebra, not this learning interface. ROOT needs independent source review and the ordinary complete resource/execution process before any representative run. This is a concrete implementation boundary, not an unresolved biological input definition.

## Representative TRAIN/VAL proposal

The inert protocol proposes the original32 TRAIN2Mb parents, all952 haplotypes/686 queries/266 references, every node/hold/SNP and the original external native-hold exact-span loss. Both arms start from the same supplied completed H045 own-state pass30, identical714+1440 new weights and zero residual outputs. Complete30passes/960 parent updates each, fresh independent AdamW with the existing learning/clip/ordering recipe, seed101 and FP32. No early stopping or disposable numerical-training updates. Checkpoints5/10/15/20/25/30 produce all96 new arm×checkpoint×VAL products before either new arm's VAL truth. Selection remains equal-parent Present182 accuracy, thenNLL, then earlierpass.

The specific prospective utility gate is feedback versus sampled-readout: at least1.00boundary-F1 percentage point at1kb, no more0.05accuracy-point loss, no meanNLL regression and no loss in uncensored true-short-tract recall. Report all cohorts, paired parent effects/SD, typed/count/short-tract diagnostics and actual fallback/preparation costs. Failing that contrast rejects feedback utility even if both models improve against the lower-capacity pooled reference. Eight reused VAL parents and one seed remain development evidence. ROOT owns registration, real jobs and later confirmation.

Source-only authoring and standard-library AST/compile inspection occurred. No actual data/payloads, model imports/instantiation, numerical campaign, CPU/GPU training, optimizer steps, installations, guards/admissions, live jobs or Git were touched. All authored files are inside this new directory.
