# H024 v2 author bundle

Final source-only correction for preserved F1–F3 v1 REJECT. Exact final source/config and all artifact references are in author_receipt.json. This author bundle is not an independent review, actual source-model integration receipt, CUDA profile or fit/resource admission.

Run all final checks with scripts/project_env.sh, CUDA_VISIBLE_DEVICES empty, OMP/OPENBLAS/MKL threads 1, TMPDIR set to this bundle's existing tmp directory and ARGGNN_PYTHON -B. Final scripts are core_checks_attempt03.py, adapted_oracle_checks_attempt02.py and correction_checks_attempt03.py. They contain only handmade CPU arrays and disposable model/optimizer unit operations.

Core checks preserve predecessor reference math and use the production unique-tree semantic entry; input VJPs use a shared original-tree base. The adapted reviewer oracle preserves its matrix-exponential/reference math and is author corroboration only. Focused checks explicitly cover all three rejection findings, production unique encoding and optimizer connection/binding domains. All numerical tolerances remain unchanged.

Preserved failed attempts: core_checks_attempt01.py/.log and core_attempt01_status.json record the constrained-input derivative fixture mismatch; correction_checks_attempt01.py/.log and correction_attempt01_status.json record the denied lazy PyTorch TMPDIR probe. A pre-final source/report snapshot is retained before the final backward binding check. All final suites ran against the final source. No predecessor or REJECT was modified.
