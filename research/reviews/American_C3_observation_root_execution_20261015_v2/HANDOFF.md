# C3 observation driver v2: fresh auxiliary-scope repair

The v1 reservation failed before any journal event because it tried to reserve the already-created root packet as an auxiliary scope. V1 and its failed packet remain intact. This v2 source changes only root admission/output plumbing.

The root packet remains admitted_<stage>_v2, containing compact config/runtime/admission/launch/wait JSON. The reservation now names only its genuinely fresh worker_aux child. Admission leaves that child absent; launch verifies the exact admitted auxiliary path, creates it after successful reservation, and places private work under worker_aux/work and bulk output under worker_aux/launch.log. The scientific output directory remains the original input-closure pointer.

Both inactive inference configs, both observed input-closure JSONs and inference runtime metadata are byte-identical copies. Direct frozen refs point to those copies; original transitive source/input references remain unchanged. Worker, helper, common CLI, guard, mutation rate, inference options, seeds, cohort, sites and runtime are unchanged. V2 uses a fresh job key and v2 packet names. Full30Mb still requires the successful v2 short direct wait and retains the original TRAIN parent identity.

Only source compilation and source/JSON reference checks were performed. No source-report, preflight, live guard configuration, admission, launch or scientific arrays were executed/read by the implementation agent. No new guard or credit overlay was activated.

Root commands from /workspace-SR003.nfs2/ARG-GNN after source scripts/project_env.sh:

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v2/root_observation_driver.py --source-report
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v2/root_observation_driver.py --preflight qualification100kb
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v2/root_observation_driver.py --admit qualification100kb
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v2/root_observation_driver.py --launch qualification100kb

The scientific child still uses the already-installed .runtime/envs/biological_tsinfer_v1/bin/python, tsinfer0.5.1 and tsdate0.2.7, with explicit empty CUDA visibility. Qualification retains its original600-second operational limit,1GB cap,64 haplotypes and244 sites.

Root must coordinate admissions/launches as before because this repair retains the existing guard's exact initial-pin semantics. After actual short exit0/completion and available full-stage admission, the same commands accept full30mb. This source preparation makes no claim of successful inference or capacity admission.

root_driver_repair.diff and source_qualification_receipt.json provide the concrete source diff and frozen-reference checks.
