# All-record observed-boundary temporal controls

These independent GRU and Transformer controls consume every observed hold in each sealed boundary tile, including holds that contain no scored readout coordinate. They address the information-access gap in anchor-only and target-graph-only temporal comparisons. They are standard temporal models; adding redundant subdivisions can change their predictions. No native paper reproduction, subdivision invariance, independent confirmation, predictive superiority, or novelty is asserted.

## Source and temporal architecture

`allrecord_boundary_temporal.py` accepts a `LeafInterfaceBank` directly. It uses the exact branch-excluded 47-field interface (`0..47` except field7), the normalized plain full-boundary response, and the same independently graph-local depth2 `BoundaryResponseGNN`. Query/reference IDs are correspondence keys only; numeric IDs, labels, truth and population annotations are not embedded or consulted. The source bank is sorted by original physical left endpoint before temporal processing.

Every hold supplies one token per query. A token contains the encoded query embedding, the all-boundary mean embedding, the plain `-Cqr` row-normalized reference embedding, the matching observed reference-role evidence, and physical left/right/midpoint/span. The reference role inputs are the four observed one-hot fields already available in the sealed47-field interface. The reference sum is zero for a disconnected row and never substitutes a target label. This explicit reference context differs from the candidate's learned multi-branch point-source rule; it does not claim parameter or operator identity with the candidate.

Coordinates use original global bp/1e6 units. Endpoints, midpoints and widths are formed from the original float64 intervals before casting the token features to the model dtype. Tiles do not reset this physical origin. Targets create no source anchors or temporal tokens.

| Family | Temporal source bank | Fixed-coordinate decoding |
| --- | --- | --- |
| `allrecord_gru` | Two-layer bidirectional GRU per query over every ordered hold; H recurrent channels per direction | Independent physical interpolation of the encoded record midpoint states, constant endpoint extrapolation, plus the target hold's independently encoded query feature |
| `allrecord_transformer` | Two Transformer encoder layers per query over every ordered hold; four heads, FFN multiplier4 | Independent coordinate query into the same query's full record keys/values, plus the target hold's independently encoded query feature |

Source input dropout and temporal dropout use the selected0/0.1 setting. The Transformer readout has zero attention dropout, so requesting more coordinates cannot change decoder-mask draws. Source/context encoding finishes before any readout. With the same source dropout realization, added/duplicated/permuted readouts preserve the original predictions in both families. The source graph encoder permits spatial exchange through the plain boundary response; temporal recurrence/attention is querywise rather than learned cross-query recurrent memory exchange.

The Transformer guards a forward dense-score estimate `heads * queries * records^2 * dtype_bytes` at268435456 bytes per layer. This is a conservative score-allocation bound, not a total training-memory guarantee: it excludes activations, backward storage, readout workspaces and library overhead. No online matrix inverse, linear solve, eigendecomposition or matrix exponential occurs in model forward.

The manifest reports source depth, temporal depth, dropout, physical units, source/input scope, decoder rule, record sensitivity, parameter counts, active recurrent channels and the attention guard. Shared source bindings establish source provenance; they do not establish equal capacities, equal information transformations, or identical comparison scope.

## Exact future development budget

The non-executable proposal `configs/allrecord_development_search_plan_v1.json` pins the reviewed common implementation dependencies and the three new source files. Both families prescribe exactly eight globally distinct configurations: width32/64 × dropout0/0.1 × LR0.0003/0.001. Source depth2, temporal depth2, four classes,64 queries,24 reference haplotypes, eight windows,48 epochs, patience8, gradient clip1 and ordinary AdamW weight decay0.0001 are fixed. Training seeds are101/202/303. All configurations share a completed fresh64 TRAIN /16 VAL /TEST0 preparation; the all-record controls use its plain boundary bank. Age bindings remain mandatory common preparation attestations, but age operators are not model inputs.

The actual counts below are reported for the prescribed four-class models. Dropout and learning rate do not change parameter count. Each configuration trains all its parameters with the same ordinary weight decay; these standard controls have no learned kernel-rate parameter group.

| Family | Width | Total parameters | Shared source-GNN parameters |
| --- | ---: | ---: | ---: |
| All-record GRU |32|43972|5824|
| All-record GRU |64|169860|19840|
| All-record Transformer |32|44484|5824|
| All-record Transformer |64|170884|19840|

`allrecord_development_bindings.py` does not simulate or prepare data. Its binder verifies the reviewed proposal hash and source/dependency hashes, then calls the frozen AD03/AD04 `verify_completion(..., with_snapshot=True)` helper. It verifies every64/16 parent and640 tile through that helper, adds its boundary47/query/reference/window guards, and assembles real configs carrying all seven completed source/preparation hashes. An absent, incomplete, changed or non-fresh preparation cannot produce configs. The optional output directory must be new; both config files are written exclusively. No executable fresh config is included with this proposal.

The binder API is:

```python
values, config_bytes, config_sha256 = bind_completed_development(
    completion_path, expected_completion_sha256,
    plan_path, expected_plan_sha256,
    output_dir=new_config_directory,
)
```

The CLI is `python -m arggnn.allrecord_development_bindings --help`. It requires explicit real completion and reviewed plan hashes. Bound settings are captured from one initial config-byte read, then validated as an exact family grid/schema. Live checks hash controls again for integrity without replacing the captured parsed config. The dedicated `train_allrecord_development` runner calls the mandatory shared `admit_run` startup path with common CORE plus its three safe extra `.py` pins. It preserves the captured plan in the runtime source lock, uses stat identity checks for full source tensors at epochs, and full hashes at model/run seals. Existing shared helpers, runners and active-job sources are unchanged.

The runner selects each seed's earliest exact minimum of equally weighted parent validation NLL using stable float64 log-softmax of uncalibrated logits. Targets are joined only after source-only predictions. Selected checkpoints and validation logits carry actual config/seed/completion/boundary/dependency bindings; output predictions preserve full ordered parent IDs, coordinate/query order and raw float64 logits. Every failure leaves an explicit failure record. A later generic selection ledger must classify/resolve such attempts rather than treating a partial family as completed. `--models` may schedule a subset, but generic family selection still requires all24 configuration×seed cells to resolve.

## Generic selector integration

The already extensible fresh selector needs no family-specific code change. A future combined frozen selection plan should add `allrecord_gru` and `allrecord_transformer` to both `families` and `required_baseline_families`, with descriptive roles such as `allrecord_boundary_gru` and `allrecord_boundary_transformer`, `grid_kind="width_dropout_lr"`, the real bound protocol refs/hashes, exact source pins, and the matching factual `input_scope` / `cross_tree_aggregation` strings. A strongest required baseline is unavailable until every listed family resolves and has an eligible three-success configuration. Pilot16/4 validation cannot be relabeled as fresh development by this binder or runner.

## CPU admission evidence

The new synthetic tests cover forward shapes and every parameter gradient, every hold's feature reachability, intervening plain-operator counterfactuals without scored targets, observed reference-role counterfactuals, external-label score changes with unchanged predictions, source encoding without any readout access, added/duplicated/permuted readouts, source dropout consistency, query/ID/serialization/unary representation contracts, state/bank roundtrips, and explicit subdivision sensitivity. They also check global interval features and differences formed before float32 casting, exact family budgets, safe source pins, pre-read path guards, captured-config drift, source-startup admission, stat/full seal modes, refusal to bind incomplete data, raw parent joins and earliest checkpoint tie behavior. All data and fitting in these tests are synthetic CPU-only. No real cohort is fitted, no GPU is used, and no final-test input is accessed.

The separate `research/protocols/input_conditioned_hold_rates_options.md` note records an evidence-independent positive per-hold rate-conditioning hypothesis. It is not implemented or added to this search.
