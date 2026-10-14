# H040 execution source v1

Scientific source is ready for root admission and execution. This source task did not prepare actual observation banks, load real checkpoint weights, train a model, create a CUDA context or read any outer/VAL truth.

Worker: src/arggnn/H040_latent_clade_execution_v1.py, SHA256 **28973fe0a9dae2c950c07dd463a23edcb6b45ab6659813f5ef31b9dc766bf84d**.

The existing anonymous_clade_witness_v1.py and anonymous_clade_gnn_v1.py are unchanged at their frozen hashes. No H039 scientific source, guard, source pin or scheduler was edited.

## Fixed scientific comparison

Each invocation/output trains exactly one of tensor or event_mlp. Both use the actual selected H038 aligned seed101 pass30 checkpoint, identical CPU initialization of all learned states, O=0, fresh AdamW, FP32/PyG and the same seed101+pass parent permutation. Each receives 30 full passes over the original 32 TRAIN parents, 960 complete-parent updates. Checkpoints are saved at passes 5/10/15/20/25/30. There is no early stopping or VAL-driven optimization. Each arm has 30,788 weights = 29,764 original + 1,024 new.

The tensor core is the established antisymmetric trilinear form. The matched MLP is GELU(a+vb+wc)-GELU(a+vc+wb), using the same projections, incidence, normalization, scatter and second PyG sweep. It can model three-way interactions. The candidate contribution is the sample-anchored reassociation incidence and latent rerouting, not kernel priority or an established expressivity advantage.

This is exploratory development on reused VAL parents. Tensor-versus-MLP matches optimizer exposure. Comparing either continuation with the original H038 checkpoint does not isolate added optimization. There is no outer or original-paper superiority claim.

## Observations and RAM witnesses

The worker uses original fast-read complete native Relate banks, all 952 haplotypes, the original 686 query columns and observed pure-source labels. No tsinfer substitution, target-derived features or true ancestral-node alignment is introduced. Legacy branch_support_true names the aligned observed mutation evidence, not ancestry ground truth.

Four distinct haplotype landmarks are selected uniformly by fixed seed101 hash priorities over stable sample keys, independent of class, age and outcomes. Two landmarks can belong to one diploid individual; no distinct-individual promise is made. All 5,692 unordered triples containing at least two landmarks are considered. Events are strict changes in resolved rooted-triplet closest-pair state across adjacent inferred native trees; ties/sampled interiors are unresolved. Such changes can reflect inference/phase error as well as recombination and are not certificates of true ARG events. Shared sample identity anchors old/new local LCAs without claiming ancestral correspondence.

All 32 observation/witness recipes are built before any TRAIN target is opened. Runtime events are held only in a bounded CPU cache and one shared read-only GPU event payload per parent. Rebuilt arrays, if eviction ever occurs, must match their original sealed hashes. No dense event NPZ is written.

## Exact raw storage bounds and runtime planning

For parent T holds and E changed events:
- Candidate events E<=5692*(T-1).
- Runtime event tensors: exactly 20*E bytes (four int32 indexes plus FP32 weight).
- Complete audit arrays: 26*E+68,320 bytes. They are transient during construction/verification.
- Permanent event arrays on disk: **0 bytes**.
- For the existing TRAIN T metadata (852–1005 holds, total 30,531), worst-case cached runtime tensors over all 32 parents: **3,472,006,160 bytes**, below the proposed **4,294,967,296-byte** cache. Maximum one-parent event payload: **114,295,360 bytes**. No sparsity/compression saving is assumed.
- The explicit conservative ndarray preparation formula is in witness_bounds. For the planning binary Relate bound N=1903*T, maximum T=1005 and depth<=952, its allowance is **1,087,476,580 bytes**. Actual N/depth are checked from the observed bank before preparation. The proposed preparation cap is 8 GiB, separate from retained cache and bank storage.
- Named coupling forward tensors U/V/W/rank residual/O injection cost 224*N+32*T*952 bytes, plus a conservative 18,874,368-byte event-block allowance. Extra full-tree/autograd buffers and allocator overhead are additional and remain under the root's actual allocator/RSS guards.
- Proposed GPU observation/target residency is 8 GiB, GPU event cap 1 GiB. Existing H039 metadata records 8,422,390,772 bytes for all 32 old banks+mass, under 8 GiB; H040 does not attach the old H039 sidecar fields.
- Checkpoints/evidence only are durable in TRAIN. Root chose a 20-MB output scope per arm; six ~0.4–0.6 MB FP32 model/AdamW checkpoints plus compact receipts fit. The source proposal's 32-MB recommendation is not an execution requirement.

One bounded synthetic CPU timing probe used complete balanced 952-leaf forests with independently permuted leaves. At 32 holds, build took 0.0508 s and verification 0.1156 s. Linear scaling to the existing TRAIN hold counts is about **159 seconds per worker once**. Plan roughly **3–10 minutes** including real depths, CPU contention and allocation; this is not a production benchmark. Existing bank/sidecar read time was 38.8 s in H039's metadata receipt. With the worst-case 4-GiB cache, all 32 runtime witnesses fit, so the 30 training passes do not require rebuilding them. The driver records actual build seconds/counts and update/pass timing.

No GPU training runtime is measured for H040. Two full PyG sweeps and event backpropagation add work to H038. Root should use actual first-pass timing, without shortening the fixed30-pass scientific schedule silently.

## Frozen evaluation

A separate admitted evaluate context accepts both train_completion_refs. It requires six actual checkpoints per arm, matching initial-state hashes and original 960-update provenance. It first predicts all **6 passes × 2 arms × 8 VAL parents = 96** native-hold arrays into immutable CPU RAM. Every array gets a JSON content/geometry/checkpoint/source seal before the first VAL truth read. No dense probability arrays are written to disk.

Native RAM probabilities use exactly 12*10976*sum(T_VAL) bytes. The original readout_hold mapping expands them only for the original H038 scorer. Selection is the unchanged equal-parent Present182 exact-span accuracy, lower NLL tie-break and earlier pass. Full cohort/calibration/boundary diagnostics are computed for the selected checkpoints only, using the original implementation. No probability floor, posthoc renormalization or outer access is added.

## Root integration and runnable commands

The root-owned context_factory.py supplies all required fields: root_authorizer, admitted, phase, arm, cfg, config_ref, reader, output, guard, device, registry, entry_refs, observation_cache_ref, observation_source_refs and phase-only bindings. Evaluation additionally supplies train_completion_refs. The factory owns the current resource successor, lease/title checks, qualified device, current source closure, logical job and guarded writer. The scientific worker does not replace these.

Important integration detail: configure_fit requires a base config containing gpu_runtime/nvtop_helper. Keep proposed_config's original configs/census_tsinfer_fit_v1.json base reference, rather than replacing it with the top-level H038 training config, which lacks gpu_runtime. This correction was reported to root before admission.

Source-only description, with no data/scientific runtime:
```sh
source scripts/project_env.sh
.runtime/env/bin/python -B -m arggnn.H040_latent_clade_execution_v1
```

Root-owned admission and launch entrypoints (not run by this source task):
```sh
.runtime/env/bin/python -B research/reviews/H040_root_execution_v1/root_driver.py --admit tensor
.runtime/env/bin/python -B research/reviews/H040_root_execution_v1/root_driver.py --admit event_mlp
.runtime/env/bin/python -B research/reviews/H040_root_execution_v1/root_driver.py --launch tensor --gpu 2
.runtime/env/bin/python -B research/reviews/H040_root_execution_v1/root_driver.py --launch event_mlp --gpu 3
```

The two launch commands should run in independent root supervisor sessions because each waits for the actual launcher exit. The root launcher supplies --execute --input and --expected-input-sha256 to this module. Direct execution without a separately pinned active root context factory is rejected.

## Source evidence

proposed_config.json and registration_proposal.json prospectively link H040 to H038 and distinguish it from H039. The ledger was consulted before proposal creation; root owns journal/ledger integration and commits. source_CPU_check.json records the single bounded synthetic timing/parameter check, including identical initial state hashes and no optimizer updates. source_closure.json pins scientific sources. completion.json records the final source-only outcome.
