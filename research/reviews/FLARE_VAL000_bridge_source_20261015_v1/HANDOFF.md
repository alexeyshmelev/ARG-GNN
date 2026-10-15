# Original American VAL000: late-bound FLARE target bridge

Source only. No target, prerequisite completion, reservation or execution was
created by this agent. The simulation genuinely completed with 228477 sites and
actual root direct wait0. Inference and native-path completion are late-bound
prerequisites. Root owns their execution and the later FLARE admissions.

build_target_v1.py reads bounded JSON metadata and frozen source files only.
It requires actual original VAL000 inference and native-path completions, their
actual direct waits0, exact actual Popen/admission/minimal-wait lineage and owned
group drain. It checks the independent VAL seeds 31001/1031001, full 30 Mb,
the shared 1600-haplotype roster, pure-source references R600, queries Q1000,
original class order and absence of query ancestry truth in inputs.

The minimal FLARE root count manifest derives num_sites and retained sample
count from actual completed inference unary_preprocessing.json. It cross-checks
simulation.sites and the explicit no-filter/no-trim/sample-preservation evidence.
It does not copy TRAIN counts, open the tree or invent unknown tree/node counts.
Native focal preparation must be the unchanged generic original source, using
the exact inferred observation, seed 93001 and 10000 query focals. The builder
copies its actual focal ref. No new focal sampling or competing registry occurs.

The target and prerequisite receipt record this one shared observation,
R600/Q1000 identity split and 10k focal registry for FLARE, native ARGformer and
GNN. The frozen shared evaluation contract remains authoritative. GNN fitting
on VAL000 is forbidden. Truth scoring still requires the complete prespecified
method predictions and the original GNN checkpoint/VAL barrier. Preparing this
one parent grants no early truth access, checkpoint selection or test access.

The existing FLARE980 stage driver, converter, author native inference,
sealer and scorer remain unchanged. All original sites and native outputs,
options and projection are preserved. Caps remain 950 MB/9GB/32MB including the
original auxiliary allowance. The builder checks the original pure full-site
formulas against these fixed caps. It provides no scoring execution route.

## Root late-binding command

Run from /workspace-SR003.nfs2/ARG-GNN on anogena only after genuine inference
and native-path completion. All four hashes below are read from actual terminal
JSON files. Missing or failed prerequisites stop before the target directory
is created. No future hash or completion is supplied in this source packet.

~~~bash
.runtime/env/bin/python -B research/reviews/FLARE_VAL000_bridge_source_20261015_v1/build_target_v1.py --build   --expected-inference-completion-sha256 "$(sha256sum experiments/results/American_C3_val000_full30mb_shared_v1/inference/completed.json | cut -d ' ' -f 1)"   --expected-inference-wait-sha256 "$(sha256sum research/reviews/prospective_980GB_capacity_source_v1/admitted_infer_v1/actual_wait.json | cut -d ' ' -f 1)"   --expected-focal-completion-sha256 "$(sha256sum experiments/results/American_C3_val000_full30mb_shared_v1/native_paths/completed.json | cut -d ' ' -f 1)"   --expected-focal-wait-sha256 "$(sha256sum research/reviews/prospective_980GB_capacity_source_v1/admitted_native_paths_v1/actual_wait.json | cut -d ' ' -f 1)"
~~~

The fresh output packet is
research/reviews/FLARE_VAL000_bridge_source_20261015_v1/bound_target_v1.
It contains prerequisite_receipt.json, root_counts.json and target.json.
The printed target SHA is the actual target pin. Default invocation without
--build prints only an inert source status and creates nothing.

## Root FLARE conversion

After the actual target exists, use fresh stage scopes and the existing active
980GB route. Example expiry 2026-10-17T18:00:00Z may be replaced by root with a
suitable future absolute UTC expiry. Existing scopes cannot be replayed.

~~~bash
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver --admit --stage convert   --target research/reviews/FLARE_VAL000_bridge_source_20261015_v1/bound_target_v1/target.json   --expected-target-sha256 "$(sha256sum research/reviews/FLARE_VAL000_bridge_source_20261015_v1/bound_target_v1/target.json | cut -d ' ' -f 1)"   --packet research/reviews/FLARE_val000_convert_980_root_v1 --output data/FLARE_val000_convert_980_root_v1   --job-id FLARE/VAL000-convert-980-root-v1 --expires-at-utc 2026-10-17T18:00:00Z

CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver --launch   --admission research/reviews/FLARE_val000_convert_980_root_v1/admission.json   --expected-admission-sha256 "$(sha256sum research/reviews/FLARE_val000_convert_980_root_v1/admission.json | cut -d ' ' -f 1)"
~~~

## Root native inference and sealing

Only after actual preceding completion and successful real outer/direct wait
evidence, continue with the actual returned predecessor. The already completed
official installed-JDK FLARE build is reused as the existing pinned default.

~~~bash
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver --admit --stage native   --target research/reviews/FLARE_VAL000_bridge_source_20261015_v1/bound_target_v1/target.json   --expected-target-sha256 "$(sha256sum research/reviews/FLARE_VAL000_bridge_source_20261015_v1/bound_target_v1/target.json | cut -d ' ' -f 1)"   --conversion data/FLARE_val000_convert_980_root_v1/completed.json   --expected-conversion-sha256 "$(sha256sum data/FLARE_val000_convert_980_root_v1/completed.json | cut -d ' ' -f 1)"   --packet research/reviews/FLARE_val000_native_980_root_v1 --output data/FLARE_val000_native_980_root_v1   --job-id FLARE/VAL000-native-980-root-v1 --expires-at-utc 2026-10-17T18:00:00Z

CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver --launch   --admission research/reviews/FLARE_val000_native_980_root_v1/admission.json   --expected-admission-sha256 "$(sha256sum research/reviews/FLARE_val000_native_980_root_v1/admission.json | cut -d ' ' -f 1)"

CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver --admit --stage seal   --target research/reviews/FLARE_VAL000_bridge_source_20261015_v1/bound_target_v1/target.json   --expected-target-sha256 "$(sha256sum research/reviews/FLARE_VAL000_bridge_source_20261015_v1/bound_target_v1/target.json | cut -d ' ' -f 1)"   --native-run data/FLARE_val000_native_980_root_v1/completed.json   --expected-native-run-sha256 "$(sha256sum data/FLARE_val000_native_980_root_v1/completed.json | cut -d ' ' -f 1)"   --packet research/reviews/FLARE_val000_seal_980_root_v1 --output data/FLARE_val000_seal_980_root_v1   --job-id FLARE/VAL000-seal-980-root-v1 --expires-at-utc 2026-10-17T18:00:00Z

CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver --launch   --admission research/reviews/FLARE_val000_seal_980_root_v1/admission.json   --expected-admission-sha256 "$(sha256sum research/reviews/FLARE_val000_seal_980_root_v1/admission.json | cut -d ' ' -f 1)"
~~~

Preserve all native outputs, prerequisite metadata, target/count refs, real waits,
launch monitor receipts and prediction seals. Scoring stays pending until the
original common truth barrier is satisfied under separate root authority.

source_static_receipt.json records 13 source/metadata checks. The actual existing
simulation and genuine wait0 metadata were validated. No future completion was
fabricated, no synthetic completion fixture was created, and no focal, genotype,
tree, checkpoint or query truth payload was read. No guard/admission/launch,
GPU/native/model execution or Git operation was performed by this agent.
