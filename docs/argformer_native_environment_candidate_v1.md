# ARGformer unchanged native environment candidate v1

Status: source-only candidate prepared 2026-10-10. No candidate package installation, full author-model import/forward, GPU launch, training, planned held-out input/target read or biological evaluation has been performed. The primary agent owns installation, native execution, scheduler integration and experiment-ledger entries.

## Objective and pin

The next qualification must import the actual author modules and execute a complete released 12-layer encoder. Earlier AST-loaded component probes do not establish this result. Source remains immutable at .runtime/author_sources/argformer_f7545be7bca0, official AI-sandbox/ARGformer commit f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596. The archive SHA256 is 44c1b76f7806dfc3063d707eae1950601eb4caa3ebe2fa20277bb98e77f1890c. The sorted 60-file manifest, excluding the generated source receipt, is 2295d86ace359ac1a7d0522c1001b6f2822b30cf9b16200517ea3473f234ca5b.

Use the released Pixi lock, SHA256 e14f152b61e624fb926b86c9f8d6125d8e998aacf896f13c96f6e2a2804dcda3, as the coherent source route. The separate requirements.txt has an older Torch2.3/Triton2.3/Transformers4.40/Composer0.22 stack. Do not combine the two dependency manifests. Preserve the separate shared project runtime (Torch2.6.0+cu124/Triton3.2/NumPy2.4.6).

This qualifies the released source/config route, not all paper-versus-release discrepancies or an author-trained checkpoint. No checkpoint/data asset was present in the pinned archive.

## Complete prepared materials

All candidate materials are repo-contained under:

    .runtime/tmp/official_argformer_native_environment_candidate_v1/

- conda-explicit-linux-64.txt: all 442 referenced conda URLs with lock MD5 fragments, consumable by existing conda as an @EXPLICIT spec.
- pypi-locked-requirements.txt: all 29 referenced PyPI distributions, direct URLs and SHA256 hashes. Install with --no-deps --require-hashes because the complete dependency closure is already in the lock.
- lock-package-manifest.json: each referenced URL, conda SHA256/MD5, PyPI SHA256, artifact identity and available compressed size.
- flash-attn-source-requirements.txt: additional FlashAttention2.6.3 source URL/hash.
- public_sources/flash_attn-2.6.3.tar.gz: downloaded and SHA256-verified 2,631,958-byte public source distribution, inspected as an archive only.
- public_sources/flash_attention_v2.6.3/: reviewed upstream setup, README, rotary and fused-loss snapshots, inspection inputs only.
- hf_base_config/config.json: frozen BERT base configuration only, no BERT model weights.
- inspection_receipt.json: provenance, counts, source-only boundary and inspection statuses, including unsuccessful read/command-construction attempts.
- candidate_generation.exit_status, source_inspection.exit_status and documentation_write.exit_status: terminal status0 for their successful stages.
- handoff_receipt.json: final candidate and preservation hashes.

The extraction checked one-to-one correspondence between all default Linux package references and top-level lock package records, with no duplicate or unused records. The compressed conda archive total is 4,349,048,973 bytes. Extracted environments, retained caches and compilation require extra storage. Run the existing storage gate before installing and respect the 450GB warning/500GB repository maximum. This is a full author-lock route, so it includes author tooling not strictly needed for one forward.

The URL lists can be regenerated from the immutable lock: select environments.default.packages.linux-64 references, join each with its top-level package record, write @EXPLICIT conda URL#md5 lines and hashed PyPI direct URLs. Re-solving version ranges defines another candidate.

## Exact core tuple

| Component | Exact locked artifact filename |
|---|---|
| python-3.11.9 | python-3.11.9-hb806964_0_cpython.conda |
| pytorch-2.4.0 | pytorch-2.4.0-py3.11_cuda12.4_cudnn9.1.0_0.tar.bz2 |
| pytorch-cuda-12.4 | pytorch-cuda-12.4-hc786d27_7.tar.bz2 |
| torchtriton-3.0.0 | torchtriton-3.0.0-py311.tar.bz2 |
| torchvision-0.19.0 | torchvision-0.19.0-py311_cu124.tar.bz2 |
| torchaudio-2.4.0 | torchaudio-2.4.0-py311_cu124.tar.bz2 |
| cuda-toolkit-12.4.1 | cuda-toolkit-12.4.1-ha804496_0.conda |
| cuda-nvcc-12.4.131 | cuda-nvcc-12.4.131-hcdd1206_1.conda |
| gcc-12.4.0 | gcc-12.4.0-h236703b_2.conda |
| gxx-12.4.0 | gxx-12.4.0-h236703b_2.conda |
| c-compiler-1.7.0 | c-compiler-1.7.0-hd590300_1.conda |
| cxx-compiler-1.7.0 | cxx-compiler-1.7.0-h00ab1b0_1.conda |
| numpy-1.26.4 | numpy-1.26.4-py311h64a7726_0.conda |
| transformers-4.44.1 | transformers-4.44.1-pyhd8ed1ab_0.conda |
| tokenizers-0.19.1 | tokenizers-0.19.1-py311h6640629_0.conda |
| huggingface_hub-0.24.6 | huggingface_hub-0.24.6-pyhd8ed1ab_0.conda |
| omegaconf-2.3.0 | omegaconf-2.3.0-pyhd8ed1ab_0.conda |
| torchmetrics-1.4.0.post0 | torchmetrics-1.4.0.post0-pyhd8ed1ab_0.conda |
| einops-0.8.0 | einops-0.8.0-pyhd8ed1ab_1.conda |
| tskit-1.0.0 | tskit-1.0.0-py311h0372a8f_0.conda |
| setuptools-82.0.1 | setuptools-82.0.1-pyh332efcf_0.conda |
| wheel-0.46.3 | wheel-0.46.3-pyhd8ed1ab_0.conda |
| packaging-24.1 | packaging-24.1-pyhd8ed1ab_0.conda |
| pip-24.2 | pip-24.2-pyh8b19718_1.conda |
| mosaicml-0.24.1 | mosaicml-0.24.1-py3-none-any.whl |
| mosaicml_streaming-0.8.0 | mosaicml_streaming-0.8.0-py3-none-any.whl |
| torch_optimi-0.2.1 | torch_optimi-0.2.1-py3-none-any.whl |
| ninja-1.11.1.1 | ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl |


Host GCC/G++ is12.4.0. Locked libgcc/libstdcxx runtime libraries15.2.0 do not imply using GCC15 as NVCC's host compiler. Respect compiler activation and record compiler executable/version. CUDA Toolkit is12.4.1, NVCC12.4.131 and the Torch build CUDA12.4/cuDNN9.1.0. Record actual runtime ABI/import results rather than treating this source compatibility review as execution.

Static public-wheel inspection confirmed wheel0.46.3 still exposes its bdist_wheel compatibility shim and setuptools82.0.1 exposes the integrated command. FlashAttention's wheel.bdist_wheel import is therefore not an established blocker. Setuptools82 lacks pkg_resources; the reviewed exact Composer/Streaming wheels had no pkg_resources references. Actual imports remain the gate. Do not alter lock entries preemptively. Record any real incompatibility and identify a separate candidate if a dependency change is needed.

## FlashAttention is necessary for released MLM

The author README recommends flash_attn==2.6.3 --no-build-isolation after verifying CUDA-enabled Torch, but omits it from the base lock. Although described as optional, it is necessary for the released MLM configuration:

1. FlexBertUnpadRopeAttention asserts the actual FlashAttention rotary implementation exists.
2. The author rotary module imports FlashAttention's Triton rotary kernel.
3. Released sliding-window attention raises in the SDPA branch.
4. Released fa_cross_entropy is registered only after the actual FlashAttention loss imports.

Global non-RoPE SDPA/ordinary CE would be a disclosed architecture variant, not qualification of this route. FA3 is outside the candidate.

FlashAttention2.6.3 publishes Linux/Torch>=1.12/CUDA>=11.6 requirements, supports Ampere/Ada/Hopper fp16/bf16 and head dimensions<=256. Released BERT base head dimension is64. Its build source emits SM80 and, with NVCC>=11.8, SM90 code. The pinned Torch2.4/CUDA12.4/GCC12 tuple meets these source requirements. Actual architecture/kernel execution is still unqualified.

Exact source distribution:

    https://files.pythonhosted.org/packages/cd/1e/21321670cd3c2f225026d9fd89272cae661529238850e79f48ccf2f9e4e3/flash_attn-2.6.3.tar.gz
    SHA256 5bfae9500ad8e7d2937ebccb4906f3bc464d1bf66eedd0e4adabd520811c7b52

The verified archive has2,092 members,670 CUTLASS include members and the expected flash_api.cpp. Its setup.py matches the reviewed tag bytes, SHA2566f9cb14dddf8a65811033d9ec5ab4c5abe1b8a46c008d97ef24b8815801cf8ef.

Upstream setup unconditionally runs git submodule update --init csrc/cutlass and ignores its status. With repository-contained TMPDIR, Git could otherwise discover enclosing ARG-GNN. Set GIT_CEILING_DIRECTORIES=$ARGGNN_ROOT/.runtime for build subprocesses to prevent discovery outside the build area. The verified sdist already contains CUTLASS. A logged "not a git repository" from this ignored upstream subprocess is compatible with building the intact sdist. No setup, Git command or extraction was executed by this subagent.

For a source build use FLASH_ATTENTION_FORCE_BUILD=TRUE, FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE, MAX_JOBS=2 initially, --no-build-isolation and --no-deps. Do not set FLASH_ATTENTION_FORCE_CXX11_ABI; inherit Torch's measured ABI. A prebuilt route needs exact Torch-minor/Python/CUDA/ABI matching and a persisted binary hash. Upstream2.6.3 lookup maps CUDA12 Torch to cu123 wheel names, which does not establish that such a binary was downloaded/executed.

## Primary-agent installation route

Use existing /home/jovyan/miniconda3/bin/conda and scripts/project_env.sh. Pixi was absent from the inspected remote PATH; the explicit spec avoids installing a new manager. The following commands are a proposal, not executed by the subagent.

Create a fresh prefix .runtime/conda/envs/argformer_native_v1. Persist install stdout/stderr and actual exit status per stage, and stop on nonzero. Never relabel a partial failed install as successful.

~~~bash
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export WANDB_DIR="$ARGGNN_ROOT/.runtime/logs/wandb/argformer_native_v1"
export WANDB_CACHE_DIR="$ARGGNN_ROOT/.runtime/cache/wandb"
export WANDB_CONFIG_DIR="$ARGGNN_ROOT/.runtime/config/wandb"
export GIT_CEILING_DIRECTORIES="$ARGGNN_ROOT/.runtime"
export MAX_JOBS=2
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
ARGFORMER_PREFIX="$ARGGNN_ROOT/.runtime/conda/envs/argformer_native_v1"
ARGFORMER_CANDIDATE="$ARGGNN_ROOT/.runtime/tmp/official_argformer_native_environment_candidate_v1"

/home/jovyan/miniconda3/bin/conda create -y \
  --prefix "$ARGFORMER_PREFIX" \
  --file "$ARGFORMER_CANDIDATE/conda-explicit-linux-64.txt"

/home/jovyan/miniconda3/bin/conda run --no-capture-output \
  --prefix "$ARGFORMER_PREFIX" \
  python -m pip install --no-deps --require-hashes \
  -r "$ARGFORMER_CANDIDATE/pypi-locked-requirements.txt"
~~~

Before FlashAttention build, activate/use conda run so locked compiler/NVCC activation hooks apply. Record sys.executable, Torch version, torch.version.cuda, torch.cuda._is_compiled(), Torch C++ ABI, nvcc/GCC/G++/Ninja versions and torch.utils.cpp_extension.CUDA_HOME, without initializing CUDA. CUDA_HOME must locate the isolated toolkit. Do not select system CUDA or stale PATH compilers. Split conda CUDA packaging may put headers/libraries in targets/x86_64-linux; verify through actual activation/build logs.

~~~bash
/home/jovyan/miniconda3/bin/conda run --no-capture-output \
  --prefix "$ARGFORMER_PREFIX" \
  python -m pip install --no-deps --no-build-isolation --require-hashes \
  -r "$ARGFORMER_CANDIDATE/flash-attn-source-requirements.txt"

/home/jovyan/miniconda3/bin/conda run --no-capture-output \
  --prefix "$ARGFORMER_PREFIX" python -m pip check
~~~

The actual process-title helper is ancillary project plumbing. Current project distribution metadata is setproctitle==1.3.8. If used inside native Python, separately pin/hash/record its isolated install. Do not add the shared runtime's site-packages directory. Export full installed conda/pip records, compiled FA binary hash, build log, interpreter/prefix and environment manifest hash. An install exit0 is not native model execution.

## Base config and handmade qualification input

Prepared hf_base_config/config.json resolves author alias bert-base-uncased to canonical google-bert/bert-base-uncased revision86b5e0934494bd15c9632b12f734a8a67f723594:

    https://huggingface.co/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/config.json
    SHA256 7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20

Verified base width768/intermediate3072/heads12/layers12. Providing that local directory to pretrained_model_name freezes the same config values without BERT weights. Authors did not pin this config revision, so disclose this provenance pin. Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 for qualification.

Root creates a trusted handmade fixture only in its qualification directory. Plain dictionary pickles can contain special tokens PAD0/MASK1/CLS2/SEP3 plus node_0...node_11 mapped to4...15 and unequal-length raw_sequences. Put identical vocabulary dictionaries in train/raw_train_sequences_and_vocab.pkl and val/raw_val_sequences_and_vocab.pkl. Optional contrastive labels [0,0,1,1,2,2] are explicitly toy labels.

Use actual ARGDataset with length128/skip_extant_tokens=True for inputs. MLM qualification clones original unmasked ancestor tokens, masks selected actual ancestor positions with token1, uses original token IDs as reconstruction targets, and sets other targets=-100. These are synthetic MLM targets, not biological labels.

Do not import prepare_data_pretrain.py, which performs path/data/log writes at top level. Do not run author tests blindly: test_rotary.py queries GPU capability at import. Intended normal imports are src.flex_bert and sequence_contrastive with contained WANDB variables set first. Put author root first on sys.path in fresh isolated processes and assert actual module files resolve under the immutable author directory. Do not reuse a different previously imported src namespace.

## Full model execution gates

Use real imports/factories with no AST extraction, toy encoders, layer substitutions, monkeypatches, fake dependencies or author-source edits.

MLM gate: load released yamls/mlm.yaml and pass its model_config to actual src.flex_bert.create_flex_bert_mlm, tokenizer_name="argformer", recompute_metric_loss=False. Redirect only dataset/base-config local paths. pretrained_checkpoint=None is the random-init default. Preserve all12 layers, widths768/3072, length128, released initialization, RoPE, window128/global-every3, RMSNorm, unpadded embeddings, FA2 and fused CE. Record seed17, fully resolved final config and actual vocabulary resize/padding. Use a small batch through actual wrapper/model forward under torch.autocast("cuda", dtype=torch.bfloat16), matching MLM precision. Require finite native logits/loss and persist actual output shapes/dtypes/hash. Optional one backward without an optimizer step qualifies native gradient kernels.

Contrastive gate: fresh native process, released yamls/contrastive.yaml, local path redirects and pretrained_checkpoint=None for separately labeled random-init qualification. Call actual sequence_contrastive.build_contrastive_model(cfg.model), seed36, a six-sample batch with toy positive pairs and fp32 as released. Require finite actual embeddings/similarity/loss. If backward runs, inspect encoder/pooler gradients and absence of classifier loss gradients. Random-initialized qualification is not a trained comparator.

Resolve these architectures separately. MLM explicitly sets sliding_window128/global_every3/unpad_embeddingsTrue/fa_cross_entropy. Contrastive omits them and actual FlexBertConfig defaults yield sliding_window=-1/global_every=-1/unpad_embeddingsFalse/cross_entropy, alongside its own pooling/dropout and initialization defaults. Loading MLM state does not copy window/config behavior. Do not silently merge YAML settings.

The unchanged contrastive entry point passes cfg.model. Its class-weight computation then tries cfg.train_loader in that model-only section; author code catches the missing field and falls back to uniform weights. Record actual warning/behavior rather than repairing it. Use actual released temperature0.3, distinguishing paper/code-default0.05.

Assert actual source FlexBertForMaskedLM/contrastive FlexBertForSequenceClassification, twelve real encoder layers, FlexBertUnpadRopeAttention, UnpaddedRotaryEmbedding, IMPL_USE_FLASH2=True, IMPL_USE_FLASH3=False, config.use_fa2=True, and actual flash_attn.losses.cross_entropy.CrossEntropyLoss for MLM. Record actual per-layer window tuples: zero-based MLM layers0/3/6/9 global, others(64,64). Record parameter counts/state hash.

Later trained-checkpoint qualification needs actual Composer state.model schema, checkpoint-specific vocabulary/config and all mapped/missing/unexpected keys, existing bridge's missing encoder/head guards, and reload-equal handmade outputs. Root-created random checkpoint replay remains synthetic. No author-trained checkpoint was recovered.

## Scheduler and final receipt

scripts/launch_gpu.py only accepts arggnn.* modules and hard-codes shared .runtime/env/bin/python. Direct isolated-Python GPU invocation bypasses leases. Root integrates a narrow project runner, launched through that scheduler, which invokes native Python and preserves inherited physical UUID/index, lease environment and lease descriptor. This is project plumbing, not an author change.

With exec, preserve the inheritable lease descriptor. With subprocess, pass it and keep parent alive until completion. Native child must set exact title AI Auto Research Pipeline Test before any CUDA context, since a title set before exec can reset. Verify title in NVTOP. Eligible physical GPUs2/3/4/5 and at most two project GPUs remain the existing policy. Root owns availability checks/scheduling.

Terminal receipt must include source/config/fixture/environment/log hashes, actual isolated interpreter, physical GPU UUID/index, actual driver/Torch/CUDA/Triton/FA versions, title verification, device architecture, actual layer/backend/loss classes, native forward/optional backward outcome, timings/peak memory, output hash and process exit status. Persist failures without substituting CPU/mocked CUDA evidence.

The source-only launch proposal is ready. Actual native execution, biological performance, trained-checkpoint reproduction and real-data conclusions remain separate unqualified work.

## Preservation

No author source, existing bridge/source-check script/tests, prior execution document or workflow audit was edited. Final handoff hashes verify that boundary. Original Apache2 notices and active RMSNorm source notices referring to Llama2 Community License stay in place. This route imports source in place, without creating a redistributable vendor copy.
