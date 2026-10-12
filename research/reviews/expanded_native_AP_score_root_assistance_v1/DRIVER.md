# Root-only AP scoring driver

Source-only delivery, 2026-10-12. Neither driver nor helper has been imported or executed by the author. No admission, numerical score, runtime report, observation/prediction array, truth access, GPU action or Git operation occurred. Syntax parsing is the only code verification.

Root can review/pin root_scoring_driver.py and root_score_helper.py, then invoke:

    cd /workspace-SR003.nfs2/ARG-GNN
    CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B research/reviews/expanded_native_AP_score_root_assistance_v1/root_scoring_driver.py --admit --tag v1
    CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B research/reviews/expanded_native_AP_score_root_assistance_v1/root_scoring_driver.py --launch --tag v1

The first command creates one10,000,000-byte retired-guard CPU scope. The second refreshes initial ledger pins, launches one CPU child and directly waits for its actual exit. It publishes the fixed admitted_v1/root_wait.json receipt against the fixed completed.json output. Failures produce stdout traceback evidence and a guarded failed.json when possible. No process title or GPU lease is required. This is the already registered H038 native-baseline VAL comparison, not a new hypothesis.

Default invocation prints source/metadata only. --admission-draft and --runtime-report are metadata-only CLI modes. The static admission_draft.json here is unadmitted and has no live pins/runtime or execution authority. Root admit reconstructs it from current bound sources and metadata.

The driver uses aligned/input.json solely to inherit the original H038 source pins, registry, entry refs, cache-index ref and target-binding metadata. It does not invoke the expired/active GPU fit capability, training inputs_v1 or any training runner. Existing fastreader read_full_cache_index_v1 validates lineage. root_score_helper.read_h038_scoring_only_v1 loads only metadata/scoring arrays from eight cache products, without model-input tensors or backbone/bank construction. Model definitions may be imported by the existing reader, but no model is instantiated.

Consumer bindings are Python/JSON only. The expected union has54 entries at delivery, below64. Data arrays, trees, checkpoints and actual target NPZs are not consumer bindings. Their transitive immutable refs remain in the existing cache/AP completion/target metadata and are SHA-verified by the reader only during root execution.

Output scope: started.json, eight per-parent reports, summary.json and completed.json, plus failed.json only if needed. Every actual output write uses the existing terminal-retired guard. No copied predictions, new inference, TensorFlow checkpoint load, tuning, calibration, tract dumps or outer score. Primary is the exact existing H038 Present182 FP32 mass scorer. Native AP6 SNP focals remain separate.

All eight AP C4/native6 products are verified before the target gate, with per-parent stdout preflight traces. One dense C4 parent is retained during scoring. Scope arithmetic and current free headroom must be checked afresh by root admission. The author has not called the accounting utility or claimed that a reservation exists.
