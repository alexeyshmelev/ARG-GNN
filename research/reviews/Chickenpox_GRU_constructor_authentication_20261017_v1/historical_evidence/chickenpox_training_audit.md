# Chickenpox regression pilot review

Static source review on 2026-10-10. The reviewed files belong to the primary agent and were not edited by this review. No real-data model was fitted or assessed during this audit. The separate preparation suite previously passed 14 CPU tests. The primary agent owns integration-suite execution and subsequent train/validation results.

The reviewed v1 configuration preserves history52, train_end365, validation_end443 and observations522. The source archive hash matches the frozen UCI archive. Training uses standardized MSE. Epoch/model and ridge selection uses chronological validation MAE in original case units. Prediction clipping is disabled for every method. Test assessment is explicitly excluded.

## Leakage and scalar-output findings

No leakage or scalar-head defect was found in the reviewed code.

- `TrainingScaler.fit(data.counts, protocol.train_end)` fits only observations0–364. Its transforms are elementwise with frozen training parameters. Transforming the complete loaded array does not calculate aggregate statistics from validation/test or pass future values into an input.
- `protocol.targets("train")` returns52–364. Only these target labels enter MSE/backpropagation. Randomizing training-window order is permitted because the target boundary remains chronological.
- `forecast_sequence(data, scaler, t, 52)` slices exactly observations `[t-52:t]`. Sequences are constructed only for concatenated training and validation target indices. No test sequence or test metric is constructed by this runner.
- Classical controls and ridge candidate scores use only validation targets365–442. Ridge coefficient fitting remains limited to training targets. Early stopping/checkpoint selection uses validation MAE, as declared.
- Models process one history sequence per call and do not retain recurrent state across calls. Bidirectional GRU, temporal Transformer and reverse GraphSSM scans therefore inspect only the already observed history. They do not scan the complete522-week series before selecting validation outputs.
- `make_regression_model` builds the documented encoder family and replaces the whole terminal Linear output with one scalar per county. Sequential heads retain their intermediate hidden layers but replace their final Linear. The temporary two-class output module is discarded. `num_classes` becomes1, and the replacement happens before AdamW parameter collection.
- Forecast training uses `[0,:,0]`, corresponding to P1,Q20,K1, against the standardized20-county target. There is no softmax or discretization of count labels at the output. Inverse scaling occurs before metrics.
- The no-geography ablations replace native geographic edges with identity/self edges. Stable county correspondence and, for attention, global county interactions remain available. Interpret the ablation as removing geographic message passing, not eliminating all cross-county context.

## Interpretation and follow-through

All county clade features are zero. The clade-overlap parameter consequently has no active overlap signal on this dataset. The attention mechanism still has content and county-identity routes. Results can support fixed-topology applicability, temporal/identity modeling and geographic-edge comparisons, but cannot support an ancestral-clade mechanism or changing-topology benefit.

The runner is an explicitly scoped train/validation pilot. It does not implement the fixed-last chronology-shuffle intervention, final-test evaluation, uncertainty intervals or complete hyperparameter-budget matching. These are future experiment/protocol steps rather than leakage in the pilot. Do not label its selected validation scores as final test performance.

The existing regression-head tests check scalar output shape and finite training gradients across nine families. This audit did not duplicate those model executions while the primary agent's integrated tests were already running. Source review does not establish numerical convergence or empirical forecast quality.

## Reviewed source hashes

- `src/arggnn/study_models.py`: `49006ce53d34055692273e41321af49c4ca834510ed3a33a4aa9b64582e4dd38`
- `src/arggnn/train_chickenpox.py`: `776b9eb5508133c4e7537e7a2ddcf726b5cfe519669553f738d09786293e4bd7`
- `configs/chickenpox_common_input_pilot_v1.json`: `0e2402f9f1aaf89c3a4488f233ca8d2803b80a830c41cb0aa734f4d1729a8a2f`
- `tests/test_regression_heads.py`: `fa888b160819c973ca6385d9311f233fd9b10dd1d9a85f9903a095dfa421f1e7`
