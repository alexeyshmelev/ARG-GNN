# Root ARGMix same-support scoring handoff

Source-only ready. Owner folder: research/reviews/expanded_native_argmix_H038_score_root_v1.

The small driver reuses the SHA-pinned AP scoring infrastructure for its existing retired guard, CPU runtime metadata,10MB admission, JSON writer and actual child-wait launcher. No new guard, ceiling, training, inference, probability adaptation or resource protocol was introduced.

Root commands after sourcing scripts/project_env.sh, from the repository root with CUDA_VISIBLE_DEVICES explicitly empty:

    CUDA_VISIBLE_DEVICES="" .runtime/env/bin/python -B research/reviews/expanded_native_argmix_H038_score_root_v1/root_scoring_driver.py --admit --tag v1
    CUDA_VISIBLE_DEVICES="" .runtime/env/bin/python -B research/reviews/expanded_native_argmix_H038_score_root_v1/root_scoring_driver.py --launch --tag v1

--launch inherits the actual waited CPU child and publishes admitted_v1/root_wait.json with the real return code. Direct --execute (alias --run) accepts --input and --expected-input-sha256 for the exact activated root packet. --admission-draft, --runtime-report and default source report are available.

Prospective output: experiments/results/expanded_native_ARGMix_H038_VAL_score_v1, reserved10,000,000 bytes plus existing journal metadata charge. Source report verifies53 source/JSON consumer references before runtime/draft and original budget bindings are added. No binary payload enters reservation consumer_bindings.

Scoring reads the exact completed native ARGMix run with eight VAL seals. It reuses the AP scoring-only H038 cache loader. Before any external target opens it verifies all eight original grids, complete686 query support, native sampled identities, the fixed physical positions/bin edges/retained indexes, native checkpoint/config lineage, and the recorded probability estimand. It loads one dense grid at a time. The bound completion JSON is the compact complete-eight seal certificate passed to the existing external-target reader after this gate.

No probabilities are recomputed from logits, floored, calibrated, normalized or averaged again. Scoring uses existing FP64 physical target mass construction, FP32 Present182 mass cache and NumPy primary scorer, followed by the existing equal-parent mean. True-mass probability zeros retain infinite NLL as null plus NLL_is_infinite. TRAIN/outer truth and models are never opened by this scoring path.

The summary records that this is the current author-release ARGMix with full952 input/slot metadata adaptations, source seed50 and unchanged native inner VAL checkpoint selection. It is not an exact paper-original reproduction, and common-support external VAL scores do not revise native checkpoint selection.

Authoring validation: syntax, pinned AP infrastructure compilation, exact source/JSON metadata closure, inactive draft and unchanged H038 metric function calls. No scoring was run; no arrays, truth, live inventory, admission or launch was performed. receipt.json records these limits. The deferred H039 capacity successor remains inactive.
