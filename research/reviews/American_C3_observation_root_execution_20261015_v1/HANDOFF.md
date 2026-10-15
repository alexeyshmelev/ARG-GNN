# Root C3 observation inference handoff

Source-only integration for the unchanged `american_c3_observe_v1.py` worker. Root owns all admissions and execution. No inference, GPU, install, Git, live guard configuration or reservation was performed here. Read-only preflight hashes only the four declared observed files; genotype/position arrays are not loaded. Private truth and the sanitized oracle-tree payload are never opened.

## Exact stages

| Stage | Existing observation | Sites × haplotypes | Table cap | SampleData store cap | Output reservation |
| --- | --- | ---: | ---: | ---: | ---: |
| qualification100kb | american_c3_qualification_000, 100,000 bp | 244 × 64 | 250,000,000 | 250,000,000 | 1,000,000,000 |
| full30mb | american_c3_train_000, 30,000,000 bp | 226,185 × 1,600 | 2,000,000,000 | 4,000,000,000 | 12,500,000,000 |

Qualification counts are AFR/EUR/ASIA 8 each and ADMIX 40. Full counts are AFR/EUR/ASIA 200 each and ADMIX 1,000. All are haploid and retain their original diploid pairing, genotype columns, parent, sites, coordinates and allele strings. Both configs remain inactive. Mutation rate is 1.25e-8 and current tsinfer.infer/tsdate.date defaults remain unchanged. No trimming, resampling or alternative dating fallback is added.

Serialized count bound is `max_sample_store_bytes + 2*(max_tables_bytes + 16MiB) + 16MiB`: 800,331,648 qualification and 8,050,331,648 full bytes. Add 128MiB private temporary/cache planning allowance, 16MiB log cap and 1MiB metadata planning allowance: total 952,375,168 and 8,202,375,168 bytes respectively. The actual root packet is an auxiliary reservation scope, so its logs/cache/temp files are charged with the worker output. The private work allowance is monitored while the child runs, with sampled enforcement; a library write can overshoot before termination. Explicit worker serialization uses the existing capacity-v2 guard and SampleData max_file_size. No RAM upper bound is enforced; the existing planning estimate is 32GB for full inference. Qualification walltime is 600s and full walltime 75,600s, inside a fresh 22-hour root admission. Root must version bounds if they are insufficient, preserving failed partial outputs.

## Existing runtime and source checks

`inference_cpu_runtime_metadata.json` is a fresh metadata-only report from `.runtime/envs/biological_tsinfer_v1/bin/python`, byte-value equal to the earlier runtime report: Python 3.11.17, numpy 2.4.6, tskit 1.0.3, tsinfer 0.5.1 and tsdate 0.2.7. No installation is needed. This metadata check does not establish live tsinfer/tsdate interoperability; the short inference is the operational qualification. Imports of simulation helpers do not import absent msprime/stdpopsim until simulation execute is called.

`source_qualification_receipt.json` records successful in-memory compilation and both bounded read-only preflights. The corrected simulation qualification wait is `American_C3_root_execution_20261015_v1/admitted_qualification100kb_v2/retry2_actual_wait.json` with exit 0. The initial exit 1 remains preserved. Full simulation also has direct wait exit 0.

## Root CLI and API

Run from `/workspace-SR003.nfs2/ARG-GNN`. No-flag and `--source-report` commands are inert read-only reports. Root should now perform one short actual inference and proceed directly to full inference if it succeeds, without another model test campaign.

    .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v1/root_observation_driver.py --preflight qualification100kb
    .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v1/root_observation_driver.py --admit qualification100kb
    .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v1/root_observation_driver.py --launch qualification100kb

Then use the same `--preflight`, `--admit` and `--launch` sequence with `full30mb`. Full admission requires this exact driver's qualification direct wait exit 0 and inferred manifest verification of all original sites/sample identities. No automatic retries or overwrites exist. Import API is `preflight_v1(stage)`, `admit_v1(stage)`, `launch_v1(stage)`. The returned launch receipt is an actual wait on the direct worker process and includes its admission, driver, log and completion refs.

Root must serialize admissions with other live launches. The driver refreshes initial ledger pins immediately before first worker validation. A concurrent accounting change can still require a fresh root-reviewed launch attempt, and no automatic replay is added. Root must confirm H041 registration covers observation inference before execution.

The common worker invokes:

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/envs/biological_tsinfer_v1/bin/python -B -m arggnn.american_c3_observe_v1 --execute --admission ADMISSION_JSON --expected-admission-sha256 EXACT_SHA256

GPU authority is false and CUDA is hidden. This CPU source proposes no GPU device; any later GPU proposal must honor the user's GPUs 4/5-only instruction.

## Guard and bindings

The driver configures current capacity-v2 with the root marker-repair authority only inside admit/launch. It preserves the historical journals, failed artifacts and existing authority. Direct consumers are .py/.sh/.json refs only, per frozen v4. `inputs_STAGE.json` holds the SHA-bound NPY/JSONL/roster/manifest closure transitively; every consumed file is checked again in preflight. Neither truth nor the manifest's oracle-tree reference is a consumed payload.

Frozen worker closure is common + observe + simulation helpers. Consumer closure additionally binds activated config/runtime, root driver, frozen metadata/input manifests, original inactive config, every current guard binding, marker, repair helper/authority, capacity-v1/v2 and activation barrier. The root packet and fresh output are exclusive scopes. Any guard/policy/launcher source changes during the GPU4/5 migration fail frozen-reference checks; root must review and version the affected source closure before admission. This task does not change those policies.

Failures retain partial output and actual wait evidence. If a post-exit guard check rejects the normal wait write, a compact `actual_wait_guard_failure.json` is retained in the already reserved packet for root follow-up. This is observation inference only, and grants no native path, training, prediction, PCA or evaluation admission.
