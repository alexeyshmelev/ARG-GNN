# Compact spectral action v1 author bundle

Source-only engineering implementation and handmade CPU qualification. Root owns integration and later runtime/model admission.

- [Author receipt](author_receipt.json) records ownership, source hashes, explicit qualification reason and decision-ledger integration status.
- [Final test result](pytest_attempt_03.txt) reports 36 passed in 7.27 seconds. This is unit-test duration, not a speed benchmark.
- [Final test command/runtime receipt](pytest_attempt_03.json) binds exact tested module/test hashes.
- [Initial failed run](pytest_attempt_01.txt) and [its receipt](pytest_attempt_01.json) preserve four strict float64 oracle failures.
- [Independent oracle diagnostic](oracle_diagnostic.json) compares the small-norm dense exponential with scalar exp and SciPy. Tolerances were unchanged.
- [Intermediate passing run](pytest_attempt_02.txt) reports 34 passed after the oracle correction.
- [Parity measurements](parity_metrics.json) include output, matching mass and gradient errors for all controls and a 1,025-readout hold.
- [Metric collector](collect_handmade_metrics.py) reproduces those measurements using explicit synthetic tensors only.

No actual data or holdout arrays, CUDA jobs, package installation, Git operation, storage scan or frozen dependency edits occurred. No scientific novelty, predictive outcome, bitwise equivalence or measured efficiency is claimed.
