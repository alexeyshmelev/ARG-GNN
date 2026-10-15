# H043 scientific paired execution API handoff

Source-only handoff. No real bank, target, warm checkpoint, H040 VAL, GPU, admission, guard configuration, journal, process inspection, Git or installation action was performed. The H043 prototype, design review, all H040 sources and H042 sources were preserved.

## Owned files

- src/arggnn/train_anonymous_exact_clade_transport_v1.py
- research/reviews/H043_exact_clade_transport_execution_source_v1/fixture_check.py
- This folder's HANDOFF.md, protocol_contract.json, fixture_receipt.json, source_manifest.json and completion.json.
- tmp/ and cache/ are disposable CPU code-generation caches. Do not stage them.

This is a scientific library only. It has no executable main, root driver, admission layer, resource journal, lease manager or launcher. Import is stdlib-only and inert. Root should pin its exact source in the actual root source closure before importing it. Root retains registration, current credit-v3 guard/policy, storage reservation, permitted GPU4/5 assignment, process title, direct launcher wait and failure/completion integration.

## Entry points and root context

Preferred entry point:
    run_context(ctx, input_ref)

It validates the fixed science and root context, initializes the existing lazy scientific serializers, then dispatches run_train or run_evaluate. The phase callables are also exported for an already validated/initialized root driver. The input_ref is carried into every checkpoint and completion. Root must independently validate that exact input capability and its transitive source/config/observation/target bindings.

Required ctx keys:
- root_authorizer="root_agent", admitted=True, phase="train" or "evaluate", arm="joint_pair".
- cfg: frozen scientific protocol plus separately root-reviewed positive caps. See protocol_contract.json.
- config_ref: the exact current cfg JSON ref, read through the supplied hash-verifying reader.
- reader: existing project reader, whose root resolves to the repository.
- output: existing guarded writer exposing save(name, value) and save(name, value, "torch"), returning canonical path/SHA refs. It must preserve partial artifacts and enforce admitted writes.
- guard: existing no-argument root resource/runtime callback. It is called around preparation and every update.
- device: root-qualified leased torch CUDA device, after the required launcher/title setup.
- before_transport_build(pid, original_cpu_bank, raw_upper_bounds): required root callback. It must review actual observed dimensions and construction/RSS headroom, and raise before the builder if the preparation is not qualified. Return values are ignored. It is called on every build/rebuild, before any exact-clade construction.
- observation_cache_ref, registry, entry_refs, observation_source_refs: the exact original observation-only fast-reader capabilities. entry_refs must contain the original32 TRAIN and8 VAL parents and no outer entries.
- bindings: exactly32 original TRAIN target bindings for training, or exactly8 original VAL bindings for evaluation. The scientific PhaseReader denies truth outside that phase and keeps the gate closed until the observation or complete prediction barrier.
- train_completion_ref: evaluation only, the actual H043 paired training completion. Root must also verify successful direct launcher wait/lineage evidence before admitting evaluation.

TRAIN and evaluation may have different values for the seven explicitly named resource caps. Every other config field, including the practical threshold and any additional root metadata, must match the frozen training config. Original training checkpoint config_ref remains exact.

The wrapper intentionally does not create a new accounting framework. Raw upper bounds are a scientific interface, not a Python allocator/RSS proof. Guard admission and external device use remain root responsibilities.

## Fixed science

- true_partner and own_state in that order, independent models and fresh independent AdamW optimizers.
- Both models are the actual inherited whole-parent Torch/PyG body with30,788 learned parameters:29,764 original plus the same1,024 projection parameters.
- Exact original aligned, teacher-free H038 seed101 pass30 warm completion/checkpoint refs and original MODEL kwargs.
- Seed101, full32 TRAIN/eight VAL parent roster,30 passes,960 updates per arm.
- Each pass uses the same original shuffle Random(101+pass). Each parent is loaded once for both arm steps.
- Checkpoints at5/10/15/20/25/30, separately serialized model/optimizer states and exact input/config/observation/warm/source provenance.
- Learning rate0.001, weight decay0.0001, betas(0.9,0.999), epsilon1e-8, global norm clip1, FP32, no early stopping.
- Event block32768, checkpointed event blocks, all H042 flags retained.
- No VAL influence on optimization, no conditioning on H040 VAL, no outer target binding.
- Original primary metrics, native support, all population diagnostics and original earliest maximum equal-parent Present182 accuracy, then lower NLL, then earlier-pass selection.
- Prespecified practical_accuracy_gain_threshold=0.001 absolute accuracy (0.1 percentage point), selected without known H040/H042 scores. It is a practical interpretation threshold, not significance or novelty evidence.

protocol_contract.json is an inactive source contract. It deliberately contains no usable resource-cap values, admission or launch capability.

## Shared residency and gradient lifetime

All original CPU observation banks and external TRAIN masses are loaded once. One bounded RAM-only transport cache keeps complete deterministic payloads and exact recipes. Each recipe binds stable observed sample keys plus original topology, sample-index and query-column hashes. Cache hits revalidate input and payload identities. Rebuilds must exactly equal the pre-truth recipe. There is no disk transport or bitset cache.

For each parent, original FullParentResidencyV1 supplies one shared bank/target. One complete transport payload moves to the same device. true_partner runs forward/loss/backward/clip/step, clears all gradient storage and drops activation references, then own_state does its independent step on the exact same bank/target/payload. Only Python scalar diagnostics survive. No two activation graphs coexist.

After the paired parent finishes, original residency references are explicitly cleared. No second parent bank or second event payload is retained. The original transfer/stream implementation is reused unchanged. CUDA allocator-reserved memory is still accounted by the root guard.

Applied optimizer steps are counted immediately after each actual optimizer.step, before subsequent finite checks or the next arm. Thus a first-arm success followed by second-arm failure preserves truthful partial logical counts. No automatic replay or resume is provided.

## Continuous first-step qualification

The first actual full-parent update of each arm is the qualification update and is optimizer step1 of its normal960-step trajectory. There are no extra updates, no standalone two-update experiment and no reset after qualification.

That step checks finite FP32 logits, loss, gradients, updated parameters and AdamW state, records per-arm CUDA peak/current allocated and reserved bytes, and invokes the root guard after gradient cleanup. It writes first_real_parent_step_qualification.json only after the paired first parent completes. The per-arm qualification also appears in training completion. If a later guard/step fails, the run stops with partial artifacts and step counts.

This source handoff does not qualify any real full-parent forward/backward memory. The first real step can still fail. Root must reserve the complete training scope before that continuous run, review observed CPU construction/RSS costs, and treat a failed qualification as a failed partial run without unauthorized automatic replay.

## Frozen evaluation

Training provenance is checked for all twelve prescribed checkpoint cells before inference. Models load raw finite FP32 states strictly. Both arms use one shared current parent/payload and sequential no-grad inference.

TransportRAMMatrix is a small H043 adapter: it supplies H043 seal provenance and the true_partner/own_state arm set, while inheriting the unchanged original H040 current-grid mapping and immutable-array verification semantics. The original H040 globals are never monkeypatched.

All96 (six checkpoints x two arms x eight parents) native-hold probabilities remain in CPU RAM with immutable arrays and JSON hash seals. Every seal is reverified before the VAL truth gate opens. Dense predictions are not saved. The unchanged FullPanelPrimaryValidationV1 scorer then applies the original metric and selection rules, with one selected full diagnostic call per arm.

Outputs include eight paired selected parent effects, mean, sample SD, defined/undefined counts, and positive/zero/negative counts for accuracy, NLL, Brier and calibration where those fields exist. All seven diagnostic populations are retained. Infinity is preserved as undefined finite NLL difference, not floored. Independent ARG parent is the statistical unit. No leaf/window pseudoreplication, significance or novelty claim is made.

## CPU qualification

Command:
    env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .runtime/env/bin/python -u -B research/reviews/H043_exact_clade_transport_execution_source_v1/fixture_check.py

All12 checks passed with Torch2.6.0+cu124, PyG2.7.0, CPU thread1 and CUDA hidden:
- Inert import, frozen protocol/source pins and prospective threshold.
- Actual compact builder/cache on a handmade full952-leaf star-to-cherry forest, complete payload caps, deterministic rebuilds and bank/key/content drift rejection.
- Both actual full PyG model constructors, equal initial learned states, separate parameter storage and fresh AdamW states, using synthetic source-initialized warm-schema weights only.
- Toy actual FP32 AdamW steps with an autograd lifetime tripwire proving sequential independent graphs and shared bank/payload identity.
- First-arm-applied/second-arm-failed step accounting and invalid-target pre-update rejection.
- All96 immutable H043 seals before any synthetic truth, duplicate/missing/current-parent/writeability rejection.
- All twelve synthetic checkpoint provenance cells, rejecting wrong arms/passes/initial hashes/threshold, while permitting separately reviewed cap changes.
- Unchanged original primary and selected full scorers on small in-RAM synthetic truth, earliest tie selection, all seven population effects, SD and negative outcomes.

No actual whole-parent model forward or training was run. The only optimization fixture uses tiny toy logits. No real observations, H038 checkpoint bytes, private scores or H040 VAL were read.

Root can integrate the scientific callable after prospective registration and independent resource review. This handoff itself authorizes no real build, fit, evaluation or launch.
