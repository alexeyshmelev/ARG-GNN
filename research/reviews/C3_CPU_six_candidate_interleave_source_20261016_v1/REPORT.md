# CPU six-candidate feature-reuse successor

Implemented an inactive CPU scheduling optimization in two small production modules: six_candidate_scan_v1.py and prediction_bridge_v1.py. Each directional block is prepared once, then passed through six separate original models. Every model retains its original forward/reverse block order and private recurrent/readout state. Learned graph contexts are recomputed for each checkpoint, so no weights or learned embeddings are shared.

The same original full30Mb/1600sample/10000focal support, all six epochs, original masks/refs, CPU FP32, zero loader workers and existing caps are required by the bridge. Original strict checkpoint loading, feature generation, PyG/GRU equations, scoring and30cell pretruth barrier remain external unchanged dependencies. All14direct original scientific source hashes match the sealed existing package. Nothing in a live evaluation was edited or restarted.

## Measured tiny synthetic evidence

Actual qualified Torch2.6.0/PyG code ran on CPU with one Torch thread and no CUDA. The exact original serial inference function was compiled from its unchanged source AST, with only a handmade synthetic registry supplied. Two cases used six distinct model parameter sets, five handmade trees, six sample columns and four focal records.

| Synthetic case | Serial feature builds | Interleaved feature builds | Learned graph encodes in either schedule | Maximum compared FP32 tensor difference |
|---|---:|---:|---:|---:|
| Hidden32, four blocks |48|8|48|0.0|
| Hidden4, one block |12|2|12|0.0|

Contexts, forward/reverse recurrent histories and terminal states, logits, focal identities and raw posteriors match. Per-model numerical call order and every original arithmetic receipt field match. All120per-model checks of old/new input banks found no feature mutation, including unary/nonbinary trees, an isolated-sample forest, an empty-site block and a block without focal readouts. Model states remain unchanged. Five meaningful negatives reject: fewer than six candidates, duplicate model identity, shared candidate parameter storage, duplicate focal rows and an injected deadline failure.

The fixture exposed DataLoader base-seed RNG consumption. Successful group inference now preserves the original12iterator CPU RNG advancement, and the extra five model constructors are isolated while retaining the original single-constructor advancement. Actual model encoding, GRU and readout calls consume no RNG. The first fixture assertion failure and subsequent missing receipt-field failure are preserved in the attempt logs, alongside the final successful actual execution. No prior failure is relabelled.

The final numerical fixture itself took0.757seconds after imports. This is not a throughput comparison. Full-scale runtime,32thread numerical variation,peak memory and ancestry accuracy remain unmeasured. No real checkpoint, data or truth payload was opened and no training occurred.

## Expected work reduction and limits

For the existing VAL000 plan, this scheduling pattern would reduce checkpoint-independent block feature preparations from28,704to4,784, while retaining all28,704learned graph encodes. This is a source-derived work count, not a measured6x overall speedup. Only one shared observation block is held at a time. Six private focal buffers/models add a few MB over the original schedule, but actual full-scale fit must still be checked under unchanged caps.

The package intentionally does not install an active guarded worker or grant execution. ROOT can use the narrow integration recipe after independent review and a fresh effective source closure. All current serial jobs remain untouched. Partial-output timing changes because six seals follow the complete grouped scan, while the final file/identity/scorer support is preserved.

[Actual synthetic result](synthetic_result.v1.json) · [Successful fixture execution](synthetic_attempt4_execution.json) · [Source pin verification](source_static.v1.json) · [ROOT integration recipe](ROOT_RECIPE.md).
