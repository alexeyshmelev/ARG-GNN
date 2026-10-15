# Independent20×2Mb American C3 TRAIN preparation

Source-only H041 preparation condition `C3_independent20x2Mb_TRAIN_1600haps_inferred_unary_v1`. Root registers the new condition, owns every admission/launch/credit and may begin with two parents. This task read source/config metadata only. It ran no simulation, inference, payload read, guard configuration, overlay state, GPU, installation or Git action.

## Fixed parent registry

The new20 parents are american_c3_short_train_000 through019, each independent2,000,000bp, split TRAIN. Ancestry seeds are22001–22020, mutation seeds1022001–1022020 and future path-sampling seeds92201–92220. All60 values are unique and disjoint from the original registry.

Every parent uses exactly1,600 haplotypes: ADMIX1,000 plus AFR/EUR/ASIA200 each, the original AmericanAdmixture_4B18 model,12-generation pulse, mutation rate1.25e-8, recombination rate1e-8 and original JC69/discrete-genome reconstruction. Classes are EUR/EAS/AFR. The inactive configs copy the frozen source schemas and explicitly record the new TRAIN condition.

Original five VAL and ten TEST30Mb IDs/seeds are copied exactly as deferred metadata; this driver cannot select or create them. american_c3_train_000/seed21001 remains30Mb TRAIN, is never relabelled TEST and is not counted among the new20. No parent/window/path split is changed after generation.

The prior meaningful1–2Mb proposal motivates this new prospective20×2Mb data condition. It does not assert that whole-parent GNN training fits a GPU. Frozen GNN/author sources remain unchanged. This wave authorizes no native paths, model training, predictions or evaluation, and makes no exact published Table1 claim.

## Unchanged producers and truth boundary

Simulation uses exact american_c3_simulate_v1. Observed inference uses exact american_c3_observe_v3, tsinfer num_threads8 and the registered direct all-sample unary collapse/node-map policy before unchanged tsdate defaults. Each inferred parent retains raw inferred topology, old/new node mapping and all-site/sample checks.

The simulation stage retains private truth under simulation/truth and observation-only genotype/allele/position/roster/manifest under simulation/observed. Inference consumes only the observed manifest. A parent preparation receipt exposes inferred_model_observation_ref and external_TRAIN_truth_ref separately; it does not decode query truth or insert it in model inputs.

One successful actual v5 short qualification gates the entire wave. Each admission revalidates that existing metadata evidence; it does not rerun20 qualifiers or any model test.

## Scope and storage design

Existing unchanged Writer/overlay contracts require each producer output to equal its reservation primary output. Therefore every parent uses two sequential exact reservations: simulate, then infer after actual simulation exit0. Each also has its own initially absent worker_aux log/cache/temp scope.

Product paths are:

    experiments/results/American_C3_short_train2mb_wave_v1/PARENT/simulation
    experiments/results/American_C3_short_train2mb_wave_v1/PARENT/inference

Compact root packets are admitted/PARENT/simulate and admitted/PARENT/infer inside this source directory. Their worker_aux directories are exact auxiliary scopes. All80 product/auxiliary scope strings are distinct and nonoverlapping. Scientific writes remain inside those scopes. Compact bookkeeping and four scheduler lock files remain in this owned source namespace under the overlay's charged metadata allowances.

| Stage | Count bound | Planned bytes with work/log/metadata | Cap | Walltime |
| --- | ---: | ---: | ---: | ---: |
| Simulation |752,640,256 |904,683,776 |1,000,000,000 |3,600s |
| Inference |1,550,331,648 |1,702,375,168 |2,000,000,000 |10,800s |

Simulation rejection bounds are100,000 sites,250MB tables and20,000 truth intervals. Inference uses100,000 sites,250MB tables,750MB SampleData store and250MB node map. These are pragmatic source bounds, not measured2Mb results. No clipping/subsampling/shortening or substitute parent is allowed on failure. Root preserves partial artifacts and versions/reviews any revised bound.

Per-parent cap is3GB. Four open parents cap at12GB; all20 would nominally total60GB, which is not reserved. Before reserving a fifth distinct parent, the driver requires root to close/credit enough completed wave-parent stage keys. Root owns genuine completion certificates and supervisor/process drain; no credit is applied by this driver. Root may run two-parent batches alongside existing full/native work.

RAM planning is16GB per active parent and is neither measured nor enforced. Four inference parents use at most32 declared tsinfer worker threads; BLAS/OMP/NUMBA environment thread counts remain1. The wave is CPU-only, with CUDA hidden and GPU_authorized false. Only physical GPUs4/5 remain eligible for other root-owned project work.

## Root interface

Root driver is root_wave_driver.py. APIs:

- admit_stage_v1(parent_id,stage,policy_ref)
- launch_stage_v1(parent_id,stage,policy_ref)
- prepare_parent_v1(parent_id,policy_ref)

Stage is simulate or infer. Only exact registry IDs are accepted. From the repository root, begin with:

    .runtime/env/bin/python -B research/reviews/American_C3_short_train2mb_wave_source_v1/root_wave_driver.py --run-parents american_c3_short_train_000 american_c3_short_train_001 --policy research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json --policy-sha a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85

For manual stages use --admit simulate or --launch simulate with --parent ID and the same policy arguments, then infer. --run-parents accepts one to four explicit IDs, with no automatic20-parent loop. Default/--source-report is inert metadata only.

The four parent-slot locks are cross-process and held through each parent pipeline. Their FD is inherited by each worker, so a dead supervisor cannot release a still-running worker's slot early. An admission mutex serializes this wave's quota operations; root coordinates external credit/admission operations. Exact predecessor authority failures preserve packets and are not automatically replayed.

Reservations use reviewed creditoverlayv3 and exact policy a0885b...; every state[3] guard/policy/marker/repair binding and actual producer/common/helper/config/runtime/driver closure is reserved as .py/.sh/.json. Receipt initial_ledger_pins are preserved, permitting unrelated append-only descendants. The common worker receives exact ARGGNN_COMPLETED_CREDITS_POLICY_REF JSON.

## Actual waits and credit evidence

Each stage launches its direct Python worker in a new process group and records supervisor PID, worker PID/PGID and the complete supported owned roster. These producer modules create no helper launcher; C worker threads share the process PID. Worker exit is obtained from proc.wait, then killpg0 must report ESRCH for complete group drain. Exit0 without producer completed.json or with a surviving group fails the stage.

actual_wait.json retains admission, actual completion, direct PID/exit, owned roster, group drain and log refs. Root must additionally observe the recorded root supervisor absent after the whole batch exits and revoke write/resume capabilities before any credit. A live supervisor or uncertain descendant prevents credit. This task performs no process inventory/observation itself.

Successful stages are validated and may be reused by prepare_parent_v1 without rerunning simulation/inference. Failed stages are never automatically retried or replaced. Every original20 parent remains in the prospective registry.

## Source checks

generate_wave_configs.py deterministically created40 inactive config files and the registry; it refuses overwrite. source_generation_receipt.json records exact seed formulas/source refs and metadata-only generation. source_qualification_receipt.json records successful compilation, no undefined globals, exact seed/config/rate/heldout checks,80 nonoverlapping paths and an actual inert source-report exit0. No scheduler locks or scientific output were created by those checks.

Root's next actions are condition registration, source review, then one initial two-parent batch under its live resource budget. There is no additional scientific qualifier campaign in this handoff.
