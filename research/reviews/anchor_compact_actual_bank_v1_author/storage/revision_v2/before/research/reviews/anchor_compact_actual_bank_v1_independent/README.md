# Independent source/handmade review, sealed v1

Verdict: **FAIL — repair required before qualification admission.** This verdict concerns the sealed adapter/protocol source boundary, not the compact mathematical core. Actual execution remains unadmitted.

The inherited encoder, gates, normalization, head, parameter/state_dict names, full Q184 anchor/target state, R96 donor source map and Q120 output-index provenance are retained. CPU alias views retain the original full CPU object. Diagonal CPU views alias full Cqq backing storage. Device conversion includes required neural arrays and float64 coordinate metadata only. CPU footprint reduction, actual GPU memory, measured speedup and trained trajectory identity remain unsupported.

## Blocking finding

`qualification_models_v1` constructs `coupling="full"`. `compare_action_VJPs_v1` chooses native geometry/Cqr and dispatches both actions with `model.coupling`, but never checks compact view `coupling` or `geometry_name`. Pairing an unchanged full model with scalar, diagonal or gain-matched-diagonal views returns PASS and runs full propagation in both branches. Three independent spy-backed negative tests confirm this. An age model with an unchanged age geometry array merely mislabeled plain also returns PASS. These tests pass because they reproduce the defect, not because the source qualifies.

Requested fix: fail closed on the direct model/view geometry and coupling contract before encoder/action work, and make intended non-full control construction explicit. Return enough cell identity for a runner to bind the performed control. Re-run independent negative and numerical checks on a newly sealed adapter/protocol/author binding. The later reviewed runner must validate complete frozen cell coverage, stage/factory boundaries and domain-specific runtime/storage/launcher receipt content.

## Independent evidence

32 independent handmade CPU tests passed in 13.87 seconds. Eight Q7/R4 cells cover all four couplings in float64/float32 with noncommuting grounded operators, unsorted graph serialization, duplicate/out-of-order readouts, an anchor at a graph boundary, near-boundary coordinates and fixed stress rates/mixtures. A separate direct weighted impulse-sum oracle uses dense `matrix_exp` and `solve`, rather than either source scan. Output and source/rate/damping VJPs pass frozen tolerances. Each cell also passes both full-model and independent signal qualification helpers with the intended coupling explicitly set.

Six handmade Q184/R96 candidate-sized cells cover plain initial, age initial and age stress in both dtypes. They compare all-Q logits/states/messages, feature and inherited parameter VJPs, active gradient clipping, private AdamW weights and optimizer moments. Caller model weights and CPU RNG remain unchanged. A last-donor feature perturbation changes Q120 mapped outputs. These are generated handmade arrays, not any observed bank.

A mocked handmade loader exercises exact Q184/R96/Q120 joins and rejects gather, flag, feature columns, positions, graph count, output mapping, role overlap, sample identity and mid-load hash drift. Deserialization is mocked and no actual array is opened. CPU mock profiling checks fresh factories, ABBA order, synchronization placement and the isolated live-baseline guard. The factory's CPU-only inputs and full-transfer-through-optimizer scope remain an explicitly unenforced caller contract. Mock samples are not runtime or memory measurements.

The author 44 tests/3488 comparisons are acknowledged evidence only. No source, author bundle, actual admission, runner, controller, runtime, launcher, ledger or Git state was edited in this review. No installation, actual data, native inference, CUDA context or GPU job occurred. All writes are under this independent review directory.
