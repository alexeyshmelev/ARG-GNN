# ARGMix continuation and common-score audit, 26 October 2026

All paths below are relative to `/workspace-SR003.nfs2/ARG-GNN` on anogena. This work only read bounded metadata/source files, statted explicit payloads, and wrote this audit plus an inert scoring task. It did not launch jobs, change queues/policies, open truth payloads, remove data, or scan the project/datasets recursively.

## Current state and reuse

The old controller PID 198309 is absent. Its log ends with `Controller received explicit termination signal`. Its final task rows contain 32 succeeded, 213 pending, 4 failed and 2 blocked tasks. The cached `counts` field still says two running and is stale. Both former active American workers have genuine controller-ended waits and recorded owned-group absence. Continuation must use a new reviewed controller attempt, not claim the former queue remains active.

The original plan and all six applied repair requests were hash-verified and replayed in memory. Effective revision 6 has digest `b5ce86462261a97fb27d77d8ae42806b15ebc43367d1ec951cf37023726c407e`.

ARGMix has 11 succeeded and 139 pending tasks. Every succeeded result metadata reference was verified. Reuse all nine TRAIN000 stages: generation, Relate inputs, native Relate, inferred conversion, support, features, labels, TreeLoom observation, and completed scratch retirement. TRAIN001 generation and Relate input preparation also succeeded. Every pending ARGMix output directory is absent. The exact evidence references are in `AUDIT.json`; no expensive successful native stage needs repeating.

The next unfinished stage is `paper_argmix_published_recon_train_001.native_relate`. Its existing input is `experiments/paper_argmix_1611_rebuild_20261024_v1/published_recon_train_001/relate_work`. It requires the unchanged 24 GiB native-work ceiling, 264 MiB outer terminal/log allowance, 256 GiB host-memory allowance, eight CPU threads, and 24-hour wall cap. The stage is CPU native inference; GPU pressure is not its immediate blocker.

## Storage conclusion

The final running-loop reservation was 966,422,717,552 bytes under a 990 GB operational limit. Its last reviewed TRAIN001 request needed 24,020,545,375 additional bytes, producing 990,443,262,927 reserved bytes. This request fits a 995 GB successor ceiling with 4,556,737,073 bytes remaining. The 995 GB ceiling retains 5 GB below the owner's 1 TB limit. This is historical conservative admission arithmetic, not a fresh physical measurement; root must carry forward preserved output/archive bytes and account intervening new work.

The original one-time measurement was 895,594,346,997 bytes, recorded in `research/benchmark_rebuild_20261024/integration/project_capacity_once.json`. The plan baseline is 916,082,164,159 bytes after margins and legacy caps. Use these existing accounting records and bounded scope evidence. No new whole-project scan is needed for this audit.

Recommend changing the successor ARGMix local scope limit from 40 GiB to **128 GiB (137,438,953,472 bytes)** while retaining the authoritative global **995,000,000,000-byte** admission limit, every stage cap, and the protected American pipeline floor. The scope limit is a secondary failstop, not a new reservation or storage credit. With the preserved 916 GB baseline, the global ceiling is already tighter than this local limit.

Do not claim that the whole ARGMix campaign is guaranteed to fit. TRAIN000's retained sanitized tree alone is 5,005,173,252 bytes, versus the earlier sub-1-GiB per-parent forecast. Fourteen equal-sized trees would occupy about 70.1 GB before other retained data and native transient files. Future sizes are unmeasured. All fourteen inferred-stage allowances are now 16 GiB plus 8 MiB outer overhead. Existing authorized scratch retirement preserves these sanitized trees. The old 40 GiB limit can therefore stop valid work for an obsolete local forecast, but raising it does not remove genuine future global admission waits. Do not reduce native scratch caps from final retained sizes, manufacture accounting credit, or delete historical evidence.

The American floor remains 39,469,999,104 bytes and is released only after all 15 retirement tasks succeed. Three had succeeded at the stop. Intermediate retirements do not necessarily release net reservation because the floor grows to protect the next native parent. Preserve this behavior.

## Earliest truthful matched scores

Prepared `european_three_method_interim_task.json`, task ID `paper_ap_three_methods_all_test_interim_score_20261026`. It compares the historical native-six CNN, Dense control, and TreeLoom on all five frozen European TEST parents, 10,000 rows per parent. It depends on the all-five TreeLoom prediction task and all five existing native CNN completion tasks. Both CNN and Dense prediction metadata were verified, including identical registry references. Its output is the fresh `experiments/results/paper_campaign_resume_20261026/ap_three_methods_interim_score/native` directory.

The task uses the existing `arggnn.benchmarking.paper_ancestralpaths.workflow.make_call` and its all-method/all-parent prediction barrier before truth. It keeps the original four-method/GST score task intact. A static combined-DAG validation and inert factory resolution passed; 68 task source references and 11 factory source references were verified. No scorer was run. `INTERIM_PREFLIGHT.json` records these checks. Publish this as a three-method interim comparison; the full four-method result remains pending GST prediction.

ARGMix cannot truthfully score from TRAIN000. It requires all five TRAIN parents, one validation parent for the author-model fit, and all eight evaluation parents (three IID plus five misspecified), both fitted methods, all 16 prediction products, the paired barrier, then the eight evaluation label products. The existing scorer covers all 800,000 shared examples per method and reports mean/sample SD across independent parents. Preserve its paired barrier and denominator.

The American scorer requires all 15 parents for all three methods before truth access. Its current task also depends on every American retirement. Do not report partial-training loss or pretraining accuracy as ancestry accuracy.

## Exact scheduler and runtime touchpoints

- Sealed original plan: `configs/benchmarks/paper_campaign_20261024_v2.json`.
- Frozen stopped state/log: `.runtime/benchmarking/paper_campaign_20261024_v2/state.json` and `controller.log`.
- Revision history: that directory's `repair_receipts/*.json` and their exact `request_ref` files under `repair_requests/`. Replay in increasing `resulting_plan_revision`; verify prior/resulting plan digests.
- `src/arggnn/benchmarking/paper_campaign/scheduler.py`: `validate` enforces GPUs [3,4,5] and <=1 TB; `pipeline_floor` begins at line 156; `run` reads the plan at line 182 and rejects an existing state at line 190; admission uses output plus auxiliary growth at lines 309-312; final reserved accounting is at line 379. The finally block drains owned children and marks controller-ended failures. The original source has no continuation or global-policy reload API.
- `src/arggnn/benchmarking/paper_campaign/repairs.py`: failed/blocked-only changes, exact original-plan/revision/attempt binding, exact output/auxiliary scope preservation, and no root policy fields. The old API cannot add this new interim node or alter global/scope ceilings.
- `src/arggnn/benchmarking/paper_campaign/runtime.py`: task source pins, actual project-contained output, process title, resource checks, and concrete-call completion wrapper.
- `src/arggnn/benchmarking/paper_campaign/build.py`: `argmix()` maps the source DAG; `american_retirement()` builds the floor; `assemble()` attaches the full scores. A successor must preserve original completed evidence and account reused outputs exactly once when changing them into external tasks.
- Source ARGMix recipe/DAG: `research/benchmark_rebuild_20261024/argmix/paper_argmix_recipe.json` and `paper_argmix_jobs.json`. Revision-6 replacements are in `.runtime/benchmarking/paper_campaign_20261024_v2/repair_requests/argmix_downstream_capacity_v1.request.json`; do not rebuild solely from the pre-repair source DAG.
- Native/ARGMix execution: `.runtime/env/bin/python -B -m arggnn.benchmarking.paper_argmix.paper_argmix_cli`; author model runtime `.runtime/official_argmix_abi210_v1/cu129/env/bin/python`; TREELOOM runtime `.runtime/env/bin/python`.
- European interim scorer runtime: `.runtime/conda/envs/ancestralpaths_cpu_v1/bin/python3.11`; source `src/arggnn/benchmarking/paper_ancestralpaths/workflow.py`, lines 100-214.
- ARGMix scoring/barriers: `src/arggnn/benchmarking/paper_argmix/paper_argmix_score.py`; American common scoring: `src/arggnn/benchmarking/american_scoring.py` with corrected native adapter `src/arggnn/benchmarking/paper_american/scoring.py`.
