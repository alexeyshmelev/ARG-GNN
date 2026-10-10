# ARGMix biological metadata repair v3

This is a narrow adapter repair for unchanged current-native ARGMix commit c2481248eb0de5fda659d68f59c51e0e27becd20. It adds the source-required paths.references metadata pointer to the paired training backend. Author model, trainer, objectives, hyperparameters, Rust preparation, supervised labels, archive inputs and inference semantics remain unchanged. It is separate from an unavailable exact April-paper reproduction.

Root reported that v2 completed actual GPU qualification, observed frame preparation and native archive preparation, then stock training stopped at predict-haplotype counting before optimizer updates. This source-only repair inspected the two completed product JSON files, observed reference TSVs/roster metadata and source. It did not inspect old GPU/fit receipts, archives, tensors, truth arrays, native scores or live runs. It executed no GPU, native fitting, simulation, installation or Git operation. Root owns independent review, registration and execution.

## Source cause and scientific boundary

The unchanged joint_interval_model.py:719-726 calls get_paths_config(config), requires paths.references, loads observed node metadata, and counts entries whose population belongs to predict_pops. The paired backend previously supplied train_pairs, val_pairs and save_dir, while references were present only inside each pair. Stock train_joint_model.py:329 calls the singular counting function before loading its archives.

The resulting available_predict is used only at line347 in the startup log and line599 as an argument to write_joint_training_data_log. That writer emits predict_haplotypes_available_in_references in joint_training_data.txt at joint_interval_model.py:1193. It does not feed a model input, sampler, loss, optimizer, scheduler, early stopping or archive selection. The native model builder and trainer source remain unchanged. The new pointer also becomes factual checkpoint/config metadata.

All four observed references.tsv files are byte-identical at SHA256341b99dd59d65fd9947b38738685aacefe58541ead9524ace7fe8b5a428639a3. Each has280 unique node IDs0-279, QUERY120, UNLABELED_REF64, and96 pure observed references with24 each ANA/CHG/WHG/EHG. Every row's ID, role-derived population and age exactly matches that parent's bound observed roster. No query ancestry target contributes to counting or eligibility. The source reader truncates ages to integer generations, but this count uses only population membership.

The pointer is the first TRAIN parent's already-bound references.tsv. It reports120 available predict haplotypes per panel, rather than the sum of240 across two TRAIN parents. The four panels have equivalent eligibility. The archives and pair-specific reference pointers continue to supply all native scientific inputs. During inference the existing explicit --references argument remains the appropriate VAL parent's pointer, so the representative TRAIN metadata pointer cannot replace inference reference input.

This is a metadata adaptation to the stock paired-config interface. It does not relax the user's scientific-fidelity requirement, substitute a model, introduce data cleaning, alter reference selection, or disguise an altered objective as native behavior.

## Strict inherited product amendment

Updated policy retains kind official_argmix_biological_pilot_v1 and every previous scientific field. Its compatibility_amendment is fixed in source and config and admits only these original products:

| Product kind | Exact path | SHA256 |
| --- | --- | --- |
| official_argmix_observed_frames_v1 | experiments/results/official_argmix_biological_pilot_v2/observed/observed_frames.json | b8de4b503bfebcf7eb71b6861879826f24731deb37aabe68b5283f6f07f07531 |
| official_argmix_native_biological_archives_v1 | experiments/results/official_argmix_biological_pilot_v2/native/native_prepared.json | 2c15bd908ae89b81e1a8c9d07809e82c46d6b1b41a2031ac93013dc54534961c |

Their original config hash is beb49bccf0f4780d6d76a3e738e1cba85dc5a1de4c846811aea2a20962029819. Their original owned module hash is cde961f2a9ed7aa98b6d24af644ef83d5657abed7ac5d8029afe46cebafc1665 and script hash is4261fb1195d16f73370b5fa0144ba0f59afe6a67726af6567634b0bd282e223d.

Legacy acceptance requires the exact two-field path/SHA record, correct kind and both exact original config/source declarations. A copied path, changed bytes, extra record fields, wrong kind, another old product, old GPU admission or old fitted artifact is rejected. Current-code/current-config products retain the existing same-policy admission. Nothing rewrites the original JSON producer declarations.

verify_frames and verify_prepared retain all original checks. They revalidate bound root manifests, observed entries, tree/reference/sample/grid/roster/map files, copied roster consistency, backend/pair files, archive hashes, external truth provenance pointers and native label-pointer hashes. This repair does not broaden source validation or skip a stage. The authoring task did not execute these actual-array/archive hash checks. Root execution will perform them normally.

The new reference_metadata_amendment additionally verifies every observed TSV against its own bound roster and strict counts/IDs. It verifies the original train/val YAML pair lists against all four bound tree, references, sample and native truth pointers before adding paths.references. The loader accepts only the original path schema or that schema plus exactly references. If references is present it requires an explicit observed path/SHA record and exact pointer consistency. Fitted/inference configuration calls require the entry to exist and match that record.

stock_train writes a new references_metadata_amendment.json in its fresh output, then runs unchanged stock train_joint_model.py with the unchanged prebuilt train/val archives. Returned stock_fit.json records the amendment, all four role-count summaries, chosen observed pointer, reused original product records and their original config/owned-source declarations. It does not claim v2 products were generated by repaired code. infer_and_seal recomputes the amendment, checks exact fitted provenance and passes the bound record through backend validation.

## Fresh same-code qualification remains mandatory

gpu_receipt is unchanged and still requires actual full-source joint GPU forward/backward and stock Rust TorchScript GPU fixture admission with the current config and owned-code hashes. The old v2 actual GPU receipt remains factual historical evidence, but cannot admit new fitting under the repaired code/config. It receives no compatibility exception.

Run a new qualify-gpu stage first, using the existing actual completed native build and CPU FFI admission records. Then supply that new successful receipt to train and infer-and-score. Do not repeat prepare-observed or prepare-native. The existing products are reused by their exact records above. Fresh bootstrap output, storage guard, isolated ABI210/cu129 interpreter, author/source inventory, driver UUID, root lease, exact process title, NVTOP own PID and final native/source/policy checks remain unchanged.

## Root commands after independent review

From /workspace-SR003.nfs2/ARG-GNN:

~~~sh
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" \
  scripts/train_official_argmix_biological_v1.py --phase policy \
  --expected-config-sha256 25829cdb811da918db4cb7fad5ee607a44610bd1f04bc40dbe2f5564a6d58eb2
~~~

Resolve ARGMIX_V3_BUILD/ARGMIX_V3_BUILD_SHA and ARGMIX_V3_CPU/ARGMIX_V3_CPU_SHA to the same factual successful build and CPU FFI records already used by v2. They are required original execution inputs, not newly fabricated receipts. ARGMIX_V3_GPU is one free root-approved physical GPU2-5. Root provides those values and registers the mechanical correction before execution. All output directories below must be absent, with their parent present.

~~~sh
ARGMIX_V3_CONFIG_SHA=25829cdb811da918db4cb7fad5ee607a44610bd1f04bc40dbe2f5564a6d58eb2
ARGMIX_V3_RUN=experiments/results/official_argmix_biological_metadata_repair_v3
mkdir -p "$ARGMIX_V3_RUN"

PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" scripts/launch_gpu.py \
  --gpu "$ARGMIX_V3_GPU" --job argmix-biological-v3-qualification \
  arggnn.official_argmix_biological --execute --phase qualify-gpu \
  --expected-config-sha256 "$ARGMIX_V3_CONFIG_SHA" \
  --build-receipt "$ARGMIX_V3_BUILD" --expected-build-sha256 "$ARGMIX_V3_BUILD_SHA" \
  --cpu-admission "$ARGMIX_V3_CPU" --expected-cpu-admission-sha256 "$ARGMIX_V3_CPU_SHA" \
  --output "$ARGMIX_V3_RUN/gpu_qualification"
~~~

Root reviews the new passed GPU receipt and sets ARGMIX_V3_GPU_SHA to its factual SHA256. Its exact path is below. Then fit directly from the inherited prepared record:

~~~sh
PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" scripts/launch_gpu.py \
  --gpu "$ARGMIX_V3_GPU" --job argmix-biological-v3-stock-fit \
  arggnn.official_argmix_biological --execute --phase train \
  --expected-config-sha256 "$ARGMIX_V3_CONFIG_SHA" \
  --build-receipt "$ARGMIX_V3_BUILD" --expected-build-sha256 "$ARGMIX_V3_BUILD_SHA" \
  --cpu-admission "$ARGMIX_V3_CPU" --expected-cpu-admission-sha256 "$ARGMIX_V3_CPU_SHA" \
  --gpu-admission "$ARGMIX_V3_RUN/gpu_qualification/receipt.json" \
  --expected-gpu-admission-sha256 "$ARGMIX_V3_GPU_SHA" \
  --input-receipt experiments/results/official_argmix_biological_pilot_v2/native/native_prepared.json \
  --expected-input-sha256 2c15bd908ae89b81e1a8c9d07809e82c46d6b1b41a2031ac93013dc54534961c \
  --output "$ARGMIX_V3_RUN/fit"
~~~

After actual fitting, root reviews fit/stock_fit.json and sets ARGMIX_V3_FIT_SHA to its factual hash. Inference uses this new fit and the same fresh GPU admission:

~~~sh
PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" scripts/launch_gpu.py \
  --gpu "$ARGMIX_V3_GPU" --job argmix-biological-v3-stock-inference \
  arggnn.official_argmix_biological --execute --phase infer-and-score \
  --expected-config-sha256 "$ARGMIX_V3_CONFIG_SHA" \
  --build-receipt "$ARGMIX_V3_BUILD" --expected-build-sha256 "$ARGMIX_V3_BUILD_SHA" \
  --cpu-admission "$ARGMIX_V3_CPU" --expected-cpu-admission-sha256 "$ARGMIX_V3_CPU_SHA" \
  --gpu-admission "$ARGMIX_V3_RUN/gpu_qualification/receipt.json" \
  --expected-gpu-admission-sha256 "$ARGMIX_V3_GPU_SHA" \
  --input-receipt "$ARGMIX_V3_RUN/fit/stock_fit.json" \
  --expected-input-sha256 "$ARGMIX_V3_FIT_SHA" \
  --output "$ARGMIX_V3_RUN/inference"
~~~

No bypass flag, old GPU substitution, extraction retry, lower validation threshold or author-source edit is required.

## Source qualification

19 handmade/source tests passed in2.56 seconds. They cover missing-pointer stock count failure and exact120 with the unmodified source function definitions, strict count/roster/pair consistency, bound optional path schema, inherited record/kind/config/source restrictions, old GPU rejection, source-use sites and unchanged scientific adapter bodies/constants. Tests use contained handmade metadata, not biological arrays, native archives, targets or fitting.

Only policy, stage, verify_backend, stock_train and infer_and_seal changed. reference_metadata_amendment is the sole new function. NATIVE_MODEL and JOINT_MODEL constants, frame/native extraction, external supervision, author execution, GPU qualification and all other original function bodies are unchanged. The immutable86-file author inventory matches its frozen hash. Source-only policy invocation passed without a native model, data or GPU.

The prior owned source/config are preserved exactly as new audit snapshots. Their old hashes remain attached to inherited products. The audit separately records observed TSV proof, source-use evidence, final new hashes and test logs. This is author qualification awaiting independent review, actual new-code GPU admission and native fitting. It establishes no repair runtime outcome, accuracy, superiority or novelty claim.

## Frozen repaired hashes

| Artifact | SHA256 |
| --- | --- |
| src/arggnn/official_argmix_biological.py | 4a6c44dfd81ba3f75e74c5287a4b55776bb9fe1612732c457748643cd75d7b1f |
| configs/official_argmix_biological_pilot_v1.json | 25829cdb811da918db4cb7fad5ee607a44610bd1f04bc40dbe2f5564a6d58eb2 |
| scripts/train_official_argmix_biological_v1.py | 4261fb1195d16f73370b5fa0144ba0f59afe6a67726af6567634b0bd282e223d |
