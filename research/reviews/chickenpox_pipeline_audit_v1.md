# Chronological forecasting pipeline audit

Reviewer: independent prior-art subagent, existing project context, not a manuscript verdict.

The reviewer inspected the regression adapter, train_chickenpox.py and frozen config. No leakage or scalar-head defect was found in the inspected paths. Per-county scaler fitting uses raw weeks0–364 only. Training target indices52–364 use histories[t-52:t]. All checkpoint and ridge selection uses validation weeks365–442. No final test targets are constructed or scored by the runner. Bidirectional context contains only already observed history. The legacy classification output layer is replaced with a single unconstrained scalar head before optimizer construction.

The source has one fixed-topology time series, so its temporal windows/counties are not independent simulation replicates. Applicability here does not establish changing-topology benefit or biological generalization. Final uncertainty should use contiguous temporal blocks. Nonnegative clipping is disabled for every method in the frozen config.
