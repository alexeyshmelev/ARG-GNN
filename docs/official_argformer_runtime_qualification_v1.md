# Official ARGformer native runtime qualification v1

This is a narrow full-model infrastructure qualification on handmade inputs. It does not train a comparator, execute a biological cohort, read planned held-out inputs/targets, reproduce reported scientific performance, or recover an author-trained checkpoint. Installation, compiler/ABI admission, FA2 build and GPU scheduling remain primary-agent responsibilities.

The new runner is src/arggnn/native_argformer_exec.py. The new qualifier is scripts/qualify_official_argformer_runtime.py. Existing launcher, author sources, bridge and protocols remain unchanged. Dependency provenance and the earlier source-only lock candidate remain in docs/argformer_native_environment_candidate_v1.md and configs/official_argformer_native_environment_v1. Actual root installation found incompatibilities in that complete lock. The qualification now explicitly labels packaging_repairs_v2, preserving author source/model while admitting actual tskit0.6.4, multiprocess0.70.16 and ninja1.13.0 metadata against the primary-owned repair manifest. It recursively freezes all recipe hashes and records nominal conda records separately from actual pip overlays. It does not claim exact reproduction of the author dependency lock.

Root adopted the official FlashAttention2.6.3 cu123/Torch2.4/C++ABIFalse/cp311 binary wheel, whose recipe pins SHA25625405479af3f6865c873ee3bbdfadcfccea9055355de78e7cd6b93170e9d4377. This is a disclosed prebuilt-wheel route, not a source build. The qualifier records actual package metadata and loaded extension hash, then requires real native GPU execution. Earlier failed install/probe records and source-only candidate snapshots remain intact.

## Fixed interpreter and scheduler route

The runner accepts only mode mlm/contrastive, fresh in-project output directory and optional backward. It admits only the assigned qualification script and isolated .runtime/conda/envs/argformer_native_v1/bin/python. It validates the scheduler's canonical eligible physical index 2/3/4/5, GPU UUID, job, descriptor/path/device/inode, actual held lock and current process title without CUDA inspection. It makes the actual lease descriptor inheritable and execs native Python with the same PID/lease. It resets Python path and redirects runtime caches into the qualification directory.

The native child sets and verifies exact title AI Auto Research Pipeline Test before importing Torch or creating CUDA contexts. It rechecks exec provenance, interpreter prefix and held lease, requires the pinned runtime versions, then validates the real isolated CUDA runtime UUID against the reservation. Torch2.4 does not expose a UUID property, confirmed against its pinned C++ binding source. CUDA12.4 cudart also has no cudaDeviceGetUuid export, confirmed by static symbols/headers. After creating the titled leased Torch context, the qualifier calls cudaGetDevice/cudaRuntimeGetVersion through isolated libcudart.so.12, then cuCtxGetDevice/cuDeviceGetUuid through the actual driver already present in its own process maps. It records both library paths/hashes, actual call statuses and canonical context UUID, requires runtime logical device0/version12040, and cross-checks any available Torch property UUID. It does not guess the cudaDeviceProp structure layout. After creating its first actual CUDA tensor, it invokes .runtime/env/bin/nvtop --snapshot. It saves complete stdout/stderr and command status, and requires its own compute PID at the reserved physical list index with cmdline.strip() equal to the title. It also records the actual nvidia-smi physical index/UUID/driver mapping. NVTOP and CUDA evidence are actual native-mode operations only.

Root launches each mode through the existing scheduler, using a new job name/output directory and an eligible GPU verified free. Do not invoke the isolated script directly for GPU work.

~~~bash
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
.runtime/env/bin/python scripts/launch_gpu.py \
  --job argformer-native-mlm-v1 arggnn.native_argformer_exec \
  --mode mlm --output-dir .runtime/tmp/argformer_native_mlm_v1 --backward

.runtime/env/bin/python scripts/launch_gpu.py \
  --job argformer-native-contrastive-v1 arggnn.native_argformer_exec \
  --mode contrastive --output-dir .runtime/tmp/argformer_native_contrastive_v1 --backward
~~~

Root must retain outer launcher stdout/stderr and actual terminal statuses. Run separate fresh processes for the two modes. The scheduler continues to enforce the at-most-two-project-GPU policy. The isolated runner ancillary recipe pins setproctitle 1.3.8. Installation success and source checks alone do not establish native execution.

## Actual author model and fixture

The qualifier verifies the unchanged 60-file source tree at author commit f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596 before and after execution. It verifies the frozen BERT base config hash, copies it and both released YAMLs to the run directory, records primary-owned recipe hashes, and uses real imports/factories whose module origins are asserted under the source directory. Relevant Torch/FA/Composer/Transformers/Triton modules must resolve inside the isolated prefix. No AST definition loading, module substitutions, dependency stubs or toy encoders are used in native mode.

The only inputs are freshly written trusted dictionary pickles for six unequal-length handmade node-ID paths, including a full128-token model input, and toy paired labels 0,0,1,1,2,2. The actual author ARGDataset skips each extant token, truncates/pads to128 and supplies native tensors. MLM masks two explicit ancestor tokens per row and reconstructs their original IDs, with other labels=-100. This is a disclosed handmade masked input, not a reproduction of the training collator.

MLM calls actual create_flex_bert_mlm with the released model_config, local base config/vocabulary and seed17. It retains all12 layers, hidden768/intermediate3072/heads12, RoPE, RMSNorm, FA2, fused FlashAttention CE and released per-layer windows. Forward uses bf16 autocast. Optional backward executes native gradients through every layer without an optimizer step.

Contrastive calls actual build_contrastive_model(cfg.model), seed36 and released fp32 behavior, then actual encoder/pooler/similarity loss and encode method. It preserves the author model-only configuration's caught missing train_loader class-weight lookup and uniform fallback, recording constructor output, actual weights and YAML temperature0.3. Optional backward requires pooler/encoder gradients and confirms that the classifier is unused by that loss.

The two resolved configs are never merged. MLM windows are global at zero-based layers 0/3/6/9 and(64,64) elsewhere, with unpad_embeddings=True. Released contrastive defaults to all-global windows and unpad_embeddings=False. Each mode saves its own fully resolved model/invocation configuration and checks real classes/layers/backend/loss before forward.

## Evidence and failures

Each fresh run produces source/config/fixture/code/runtime manifest hashes, copied configs and actual batch tensors, initial/final complete model-state tensor manifests, native output tensors/hash, loss, actual process/lease/device/NVTOP evidence, parameter/layer/backend identity, timing/memory and optional gradient summary. State hashes must remain identical because no optimizer step occurs. No large random checkpoint is needed for this state-hash qualification.

receipt.json and exit_status record actual terminal outcome. The receipt preserves completed stage flags and a concrete exception/traceback on failure. Forward outputs are saved before an optional backward, so a backward failure does not erase successful forward evidence. A failure before import/device creation cannot claim those later stages. Root-created result directories, source receipts or real snapshots must not be substituted for one another.

## Source-only preflight

The subagent may use source-only mode in the existing CPU environment with empty CUDA_VISIBLE_DEVICES:

~~~bash
source scripts/project_env.sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
.runtime/env/bin/python scripts/qualify_official_argformer_runtime.py \
  --source-only --mode mlm \
  --output-dir .runtime/tmp/argformer_runtime_source_only_v1
~~~

This parses Python ASTs, verifies pinned bytes/configs/recipes and materializes the small trusted fixture. It imports no author model/Torch, obtains no lease, creates no CUDA context and runs no NVTOP command. Its receipt explicitly reports passed_source_only_no_native_execution and false actual import/forward/CUDA/NVTOP flags. It is a preflight, not positive native evidence.
