# Static TreeLoom continuation and corrected American retention

The selected module is research.benchmark_rebuild_20261024.american.treeloom_speed.prediction_v2.
Its fit_factory and prediction_factory return module/function/spec/python/source_refs.
The dependency-resolved callable run_task(request, resource_check=...) launches no process.

## Sixteen-node task contract

task_builder.build(resume_checkpoint_ref=..., resume_evidence_ref=...,
output_root="experiments/results/paper_american_20261024") builds one fit plus15
held-out predictions from actual original scientific inputs and focal receipts.
It reads JSON/source metadata only and imports no Torch.
The inert task16_inert_template.json has explicit nonlaunchable placeholders.

Fit arguments: parent_registry_ref, train_inputs, output_directory,
max_rss_bytes=512GiB, preparation_headroom_bytes=128GiB,
training_headroom_bytes=128GiB, observation_cache_bytes=352GiB,
max_truth_uncompressed_bytes=8GiB, resume_checkpoint_ref, resume_evidence_ref.
Fixed science:20TRAIN parents x30epochs=600full-parent updates, FP32,
49trainable tensors/32132parameters, AdamW lr.001 wd.0001 betas.9/.999 eps1e-8,
gradient clip1. Fit output is training_completed.json with completion_ref.

Each prediction request has stage=predict, arguments holding one registered
held-out parent, and training_completion_path resolving final checkpoint600.
Prediction memory: maxRSS256GiB, preparation reserve128GiB, prediction reserve64GiB,
cache0, output16MiB. All10000 canonical focal identities and softmax probabilities
survive in predictions_completed.json and individual parent seals.

ROOT applies its /native rewrite recursively inside scientific_spec, including
arguments.output_directory and training_completion_path. Outer task
output_directory remains supervisor-owned. No held-out truth selects checkpoints.

## Safe migration

The actual current live successor writes checkpoints/latest.pt and
latest_checkpoint.json after EVERY complete parent optimizer update.
CPU inspection authenticated a saved58-update snapshot, all49finite Adam moment
pairs/counters58, all model tensors and the exact last/next parent schedule.
The live latest path may advance after this inspection.

1. ROOT verifies its exact owned PID/session/start and coordinates stopping the
   live producer through its existing lifecycle rules. This worker sends no signals.
2. After genuine exit/drain, preserve old source/results/failures and copy the
   complete latest durable checkpoint into a fresh immutable capture path.
   Hash actual copied bytes. Preserve latest/progress/source records and ROOT's
   capture/lifecycle receipt. Never modify the historical file.
3. Write evidence JSON: schema=treeloom_durable_optimizer_boundary_v2,
   checkpoint_ref=the exact immutable capture ref, parent_registry_ref=the
   original registry, complete_parent_updates=actual serialized count,
   durable_complete_parent_optimizer_boundary=true,
   no_inflight_update_in_serialized_state=true,
   VAL_truth_read=false, TEST_truth_read=false,
   source_refs=actual original live job call.source_refs.
   Include ROOT's lifecycle/capture receipt as additional provenance.
4. Pass those exact checkpoint/evidence refs into task_builder.build. The worker
   authenticates sources, every model key/shape/FP32 tensor, all49Adam states and
   counters, original observed barrier/plans, exact schedule identity/last parent/
   next index, and saved RNG state before continuing.
5. Continue at the saved state's next parent and stop at exactly600updates.
   No optimizer reset or replayed completed update is allowed. Python/NumPy/Torch
   RNG states and all saved Adam moments/counters are strictly restored.

The candidate retains the actual current route's compact preparer, streamed
decoder backward and uninterrupted1024-hold GRU chunks. New execution partitions:
128native tree holds per GNN block,64haplotypes per decoder tile,1024history holds.
Affine-prefix history changes FP32 arithmetic order only. Global carry/VJP stays
connected across windows. No TBPTT, local reset or focal reduction is introduced.

Latest snapshots preserve cumulative periodic checkpoint event hashes.
Resumed loops preserve inherited events. New-scope redundant periodic files rotate,
retaining event metadata. No inherited checkpoint is removed.

## Qualification

The selected production FP32 CPU transition starts from a COPY of the actual
58-update trained model and49Adam moment pairs. A synthetic2-hold parent has all
1600samples,10000focals,fourSNPs and nonzero history theta gradient0.0006128967.
Parallel history plus the production64hap streamed CE/contextVJP gives maximum
logit error0, gradient error3.28e-7 and one-update parameter error1.49e-8 versus
the serial model. Candidate saved59 -> strict reload -> step60 is bitwise identical
to uninterrupted candidate step60. These synthetic updates never count as
production training. Receipt: model_transition_smoke_v5/completed.json.

Local and cross-window FP32/VJP and native GRU tile checks also passed.
No production GPU speed, full-parent GPU peak or accuracy gain is claimed.
No GPU job was launched by this worker. Preserved earlier synthetic attempts
incorrectly asserted a nonzero first-step theta gradient despite the native
zero-initialized memory output projection. The selected trained-state test
exercises a nonzero memory path.

## Corrected native preparation and retirement

All13new preparation nodes must call
arggnn.benchmarking.paper_american.preparation.run_prepare.
Keep their original scientific specs/limits. The static source copies of the
three native author functions have exact independently verified AST identity.
This avoids the old american_c3_paths_v1 AST/exec loader.
Bind preparation.py, paths_static.py and author_path_functions.py in outer
task source_refs BEFORE execution; native_source_pins.json includes them.

retirement_task_mapping.json adds15CPU retirement nodes and dependency additions
serializing the next preparation/MLM after the prior retirement.
paper_american.retirement.run_task requires genuine successful ROOT producer waits,
canonical prediction validation, bitwise-verified compact final model tensors,
and an exact SHA/size/mtime inventory before deleting any enumerated new corpus/
sidecar/checkpoint payload. Reused TEST000/001 preparations are protected.
Historical source/data/results are never deletion targets.

Producer completions, executed specs, resolved configs, canonical focal/observation
references, vocabulary, statistics/logs, final prediction/PCA files and all final
model tensors survive. Full-budget checkpoint counters are preserved in the
retirement receipt. The all45 score barrier reads surviving metadata lineage and
canonical predictions, never retired datasets or optimizer payloads.
Compact final models contain state.model with every exact original tensor and
no quantization.

retirement_bounds.json derives model caps from actual inferred node counts.
All15 final models:11,447,147,520bytes. Including128MiB metadata per parent:
13,460,413,440bytes. Worst active-parent transient:39,453,221,888bytes.
At ROOT's914GB measured-base/live-growth/margin envelope, serialized American
peak is966,913,635,328bytes. After retirement, adding40GiB ARGmix bulk peaks at
970,410,086,400bytes. Simultaneous full American transient plus40GiB ARGmix reaches
1,009,863,308,288bytes and cannot fit1TB. ROOT must sequence bulk phases or prove
tighter actual remaining sizes. Capacity admission remains ROOT-owned.
