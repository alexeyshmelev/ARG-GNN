# H024 v3 author bundle

Exact two-gradient-guard source correction after sealed V2 F4 REJECT. This bundle is author source/handmade CPU qualification only. It admits no real data/producer/profile/fit or resources.

Final scripts: core_checks.py, adapted_oracle_checks.py, correction_checks.py, callback_gradient_checks.py. Source scripts/project_env.sh; use ARGGNN_PYTHON -B, empty CUDA_VISIBLE_DEVICES, OMP/OPENBLAS/MKL threads1, and TMPDIR set to this bundle's existing tmp directory. All arrays and optimizer objects are handmade/disposable CPU units.

All926 checks use the final source/config pins in author_receipt.json. Callback producers must leave all intended optimizer grads None; V3 rejects contamination after each cell and immediately before aggregate backward without clearing it. Genuine connected zero gradients after the aggregate backward and explicitly declared inactive parameters remain allowed. F4 tests reproduce both sealed V2 bugs and compare clean clipped SGD/AdamW parameters and every internal state entry over two steps.

source_delta_evidence.json proves inverse removal of exactly the two guards plus version/config SHA restoration yields the entire V2 module AST. No other algorithmic source change or numerical tolerance relaxation occurred. All V1/V2 source/config/author/reviewer seals and negative evidence remain immutable.
