# Supplied-prior-art fresh development adapters

The required `gst_ca_span` and `graphssm_zoh` families use unchanged existing `make_study_model` implementations on raw observed trees and verified F48. They are independent paper-equation common-input adaptations, not exact native reproductions, independent confirmation, or novelty/superiority evidence. No architecture variant, real training run, simulation, GPU job or final-test access is added by these adapters.

## Factual information and aggregation scope

Both families receive the original observed F8 prefix, including branch field7, plus the existing40 reference-derived fields. The raw prefix and full-parent reference adapter are verified against the sealed inferred source before tile construction. Known observed reference roles are valid inputs; query ancestry targets, populations and truth annotations remain external to prediction. Raw topology and anonymous internal nodes are retained. There is no canonicalization or boundary47 projection. Parent positions and query identities are preserved from the original128-midpoint contract and scattered back in exactly their original order.

| Family | Existing architecture | Source aggregation and readout |
| --- | --- | --- |
| `gst_ca_span` | Two sequential native GCN / GraphNorm / attentional-pool / graph-set-Transformer / node-cross-attention blocks; four heads, FFN multiplier2, paper gate, self-loop weight2 | Every observed graph in a scored tile contributes one pooled token. Log physical span biases both graph-token key attention and node-to-token key attention. The set is unordered; no positional or numeric-ID embedding. Fixed coordinates select containing-tree query leaves for an MLP readout. |
| `graphssm_zoh` | Two GraphSAGE diffusion + real diagonal state-space blocks; state dimension16, representation interpolation in the first block only | Shared-weight forward/reverse scans follow physical genomic order through every observed hold. All nonnegative persistent query/reference leaf IDs align memory; anonymous internal nodes reset. Physical right-left is formed before casting. Readouts use containing-tree snapshot-end query states, not fractional within-hold decoding. |

GST provides global graph-token mixing inside a tile. GraphSSM has two **local** spatial diffusion blocks: perturbing an arbitrary distant raw reference-node field need not directly affect every query. Its verified global F40 summaries also carry observed reference information. Here global refers to reference information across a marginal tree: the existing adapter statistics and encoded-TMRCA kernels are per-tree, not genomic smoothing. The tests therefore distinguish temporal hold reachability from direct spatial node reachability and refresh the reference adapter when changing observed reference roles. Full raw input availability does not imply an unlimited spatial receptive field.

Each physical tile is a separate forward call; neither family aggregates across tiles. The existing observed tiler uses eight fixed physical windows, retains/clips every observed hold in scored windows, and skips windows containing no retained scoring coordinates. Adding readout coordinates does not create graph tokens or recurrent source updates. Evaluation-mode duplication, addition and reordering of readouts preserve their original predictions. Existing dropout behavior is preserved; GST includes stochastic MLP readout dropout during training. No whole-model redundant-subdivision invariance is claimed: span key mass or an exact single-hold ZOH cell alone is insufficient to establish it for stacked nonlinear models, representation mixing and snapshot-end readouts.

Factory audits report the exact class, layers, source scope, state/head/gate details, adaptation status, paper/source commit references, and actual parameter count. GraphSSM bidirectionality and snapshot-end interval readout are explicit task adaptations. The adapters do not replace these architectures or attach alternate temporal encoders.

## Frozen future grid and required-family policy

`research/protocols/development_scope_amendment_v2.json` requires both supplied-prior-art families before strongest-baseline selection, alongside the other seven required baselines. Its optimizer-start budget is216 main neural starts plus72 secondary-control starts,288 maximum total; the deterministic reference smoother has zero optimizer starts. The proposal pins the amendment's actual bytes and verifies its development-only counts, required-family membership and budget.

Each family has exactly eight ordered configurations: width32/64 × dropout0/0.1 × LR0.0003/0.001. The existing two-layer factory depth is fixed; GST has two graph/set feedback blocks, GraphSSM has two diffusion/scan blocks. Four classes,64 queries, eight windows,48 epochs, patience8, ordinary AdamW weight decay0.0001, gradient clip1 and seeds101/202/303 are fixed. No grid cell may be replaced based on a development outcome.

Actual capacities for the unchanged prescribed four-class models are:

| Family | Width32 parameters | Width64 parameters |
| --- | ---: | ---: |
| `gst_ca_span` |43332|166532|
| `graphssm_zoh` |15300|44932|

Dropout and learning rate do not change these counts. Capacities differ from the candidate and other families and are reported rather than treated as matched.

`configs/priorart_development_search_plan_v1.json` is a proposal only. It contains actual reviewed implementation and amendment hashes, with no fabricated completed-source digest and no executable placeholder config. A real config can be emitted only after actual completed64 TRAIN /16 VAL /TEST0 preparation and raw source admission succeed.

## Binding and source integrity

`PriorArtSnapshot` composes the unchanged ARGMix `IntegritySnapshot`. Its inner source snapshot retains the exact original ARGMix implementation pins required by the verified raw-prefix loader. The outer snapshot additionally pins the existing GST/GraphSSM files and all new binder/runner/script files, with an import-time code guard. It checks these extra pins before controls/inputs and around source admission. This avoids another raw-source catalog or a circular binder-to-runner dependency. The binder imports only the unchanged shared verified-parent utility when data admission is required.

The outer control reader captures config bytes once, verifies the supplied hash and before/after stat identity, rejects duplicate JSON keys and conflicting hashes, and reuses the captured parsed content. Shared runtime checks hash the frozen controls again without changing the parsed config. The composed checker validates control identity, common and own code hashes, raw source stat identity at epochs, and full source hashes at major boundaries/seals.

The binder API is:

```python
settings, exact_config_bytes, config_sha256 = bind_completed_development(
    completion_path, expected_completion_sha256,
    plan_path, expected_plan_sha256,
    output_dir=new_directory,
)
```

It verifies the reviewed plan and scope amendment, uses the unchanged ARGMix completed-preparation verifier, then admits every raw observed parent through the existing prefix/reference/target identity checks before writing either family config. All seven common completion/source/boundary/age/preparation hashes are carried factually. Output files use exclusive creation in a fresh directory.

`scripts/bind_priorart_development.py --help` describes the CLI. `capture_training_snapshot(config_path, expected_config_sha256)` returns `(settings, verified_parents, composed_snapshot)`. It verifies the original reviewed plan, amendment, successful completion, all source hashes and original prefix/query/target correspondence before returning. The runner then checks imported code pins before its first device query or model device transfer. Configs cannot relabel reused16/4 data as fresh64/16 development.

## Deterministic numerical policy

The plan and each executable config pin the shared `development_execution.py` and its exact `DETERMINISTIC_POLICY`. After complete source/prefix/target admission and before any device query or model transfer, the runner applies CUBLAS workspace4096:8, strict deterministic algorithms with warnings disabled, cuDNN deterministic mode and benchmarking off, TF32 off, and highest float32 matmul precision. The helper refuses initialized CUDA. The applied policy is checked during fitting and recorded in run/cell/checkpoint metadata. Determinism applies to a fixed runtime/hardware; cross-platform bitwise equality is not promised. Unsupported deterministic operations produce explicit failures rather than a relaxed policy.

Historical selected NLL, reloaded NLL and their replay delta are all retained. Exact selected-checkpoint replay remains mandatory; no extra replay tolerance or successful-seed averaging is introduced. Selector history/checkpoint/parent-score linkage remains unchanged.

## Training, evidence and failure records

`python -m arggnn.train_priorart_development --help` describes the dedicated runner. It calls the existing factory, preserves the original tile/query scatter, and joins separately stored labels only after predictions. Checkpoint selection uses equally weighted parent validation NLL from stable float64 log-softmax of uncalibrated raw logits, choosing the earliest exact epoch minimum. Three-seed family selection remains the generic selector's responsibility.

Selected checkpoints contain actual config, seed, completion, boundary and dependency metadata plus real model weights. Separate validation-prediction artifacts contain ordered source-qualified parent IDs, original coordinate/query vectors, finite raw float64 logits, seed and config/completion hashes. They reside in the ignored checkpoint namespace. Run records carry family/seed/source/initial and ending code pins, all ordered per-parent scores, full epoch history, selected checkpoint/prediction hashes, factual scope and amendment reference.

A fitting exception leaves an explicit family/configuration/seed/source-bound failure record and no success receipt. It is not silently dropped, reclassified as an infrastructure retry, or used to average successful seeds. The future combined selector plan must list both families in `required_baseline_families`, use `width_dropout_lr`, copy the exact bound `input_scope` / `cross_tree_aggregation` strings and real protocol/code pins, and preserve every failure/missing cell. Its existing configurations-list normalization preserves original config bytes/hashes. No strongest required baseline is available if either family is unresolved or wholly ineligible.

## Synthetic CPU verification and limits

The new suite performs tiny actual optimizer fits on synthetic tensors, finite gradient and original F48/internal-node checks, intervening-hold counterfactuals without scored targets, properly refreshed observed reference-role counterfactuals, label-only metric changes with unchanged predictions, readout/query/record serialization checks, real checkpoint/state roundtrips, full synthetic80-parent completed-source admission, prefix and target corruption rejection, code/control/source drift guards, device-ordering checks, explicit fitting-failure evidence and generic selector integration. No real study labels or development scores are used.

Capacity and per-tile throughput may differ across families. Existing GST has graph-token and node-to-graph attention; GraphSSM scans full raw-node banks with state dimension16. The adapters add no resource-driven architecture pruning, context truncation, approximation or fallback. Fresh-data feasibility/throughput remains unmeasured here and must be assessed before a final execution freeze; an infeasible required family must be recorded and resolved rather than omitted.
