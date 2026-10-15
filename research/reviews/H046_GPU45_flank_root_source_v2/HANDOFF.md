# H046 canonical launcher entry correction

This is an integration-only successor. The v1 launcher rejected its review-package module before starting the scientific worker. The actual pinned launcher accepts only canonical arggnn module names and reserves the corresponding src/arggnn file.

The successor entry is arggnn.H046_GPU45_flank_worker_v2, implemented by src/arggnn/H046_GPU45_flank_worker_v2.py. It delegates directly to the unchanged train_continuous_clade_flank_v1.run_context_v1 API. The three copied root integration sources point to this fresh v2 directory. Runtime output and logical job names use v2 so root can allocate fresh scopes.

Model, scientific worker, every prior frozen source pin, scientific protocol, reviewed resource caps, guard, policy, memo, original v4, launcher and reservation semantics are preserved. The config differs from frozen v1 only in source_manifest_ref. The inherited reviewed-cap/status fields remain root's prior values. No active input, reservation or admission is created by this correction.

integration_only.diff records the exact source changes. integration_correction.json records the source-only syntax/rule/pin review. No scientific qualification was repeated; no actual count, data, checkpoint, GT/VAL, process/GPU probe, guard, admission or launcher operation was executed. All v1 files, failed-scope evidence and reservation remain untouched. Root owns any fresh allocation and execution.
