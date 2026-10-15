# Fixed initial trade-tree carry comparison

This source-only packet defines a concrete initial scientific campaign. It grants
no execution authority, imports no numerical module while being authored, and
changes no old source, result, route or admission. Root owns qualification,
execution and physical GPU0. No model/data/fit was run by the author.

## Disclosed difference from H052

H052's fixed native baselines sealed all 1987–2016 predictions before opening any
label file. Supervised fitting necessarily differs: this trainer requests only
TRAIN target years 1987–2009 before optimization. Root agreed to this adaptation.
VAL 2010–2013 and TEST 2014–2016 labels/source masks are separate capabilities and
are absent from the trainer. Every arm predicts all 255 countries before held-out
source selection. All future learned parameters are fitted only through 2009.

The exact same complete observation prefix rule is preserved: forecast tau uses
edge years below tau. Earlier held-out edge observations are legitimate history
at later forecasts, just as in H052. This is distinct from fitting on held-out
labels. Final replay discards in-sample TRAIN outputs; no rolling historical TRAIN
forecast performance is claimed for parameters fitted through 2009.

The already-known strong H052 persistence baseline motivated the shared residual
prior before any new-model evaluation. This is an informed initial diagnostic,
not a claim that baseline TEST performance was unknown. The residual prior itself
is standard and is not presented as a contribution.

The existing H052 result/prediction-seal/terminal refs are pinned. Native
persistence and EMA are not rerun, changed or selected. Their official query
support and exact sklearn1.9.1 NDCG@10 (ignore_ties=False) are reused.

## Frozen primary budget

Authoritative configuration: protocol.source_frozen.inactive.v1.json.

- Three arms: tree_only, first_order, conditional_triple.
- Seed 17; width 64; frozen complete country-tree encoder and 255 destinations.
- 100 epochs, exactly one optimizer update after each full 23-year TRAIN sequence.
- AdamW lr .001, betas (.9,.999), eps 1e-8, weight decay .0001, gradient norm clip 1.
- FP32 with AMP/TF32 disabled. Shared initialization per seed, no dropout.
- Soft cross entropy: equal queried countries within year, equal weight across
  TRAIN years. Target weights normalize only for the TRAIN loss; exact raw FP64
  targets remain the native metric input.
- Full 23-year BPTT with no detach/truncation, reset to zero in 1987 each epoch.
- Chosen checkpoint is epoch 100. No early stopping, checkpoint hunt, tuning,
  automatic arm selection or TEST-driven branch.

Primary total is three fitted models, 300 optimizer updates, 6,900 TRAIN graph
forwards, plus 90 graph forwards for final-weight replay. Future confirmation
seeds 29/43/59/71 are prespecified metadata only, not automatically runnable in this
primary driver. Root can decide on paired unchanged confirmation arms while TEST
remains closed; a later phase needs its explicit binding/driver. Half-alpha and
count-prior16 are future hypotheses, not current selectable arms.

## Exact common body and carry semantics

Every arm shares the frozen full-tree encoder, leafwise GRUCell64 and
128 -> 64 -> 255 residual readout. Every arm adds this residual to fixed logits
log(latest_observed_outgoing_profile + 1e-8). The final residual layer weight and
bias initialize to zero, so the complete original profile bypasses hidden-width
compression. Positive additive smoothing is identical in all arms; never-observed
or observed-zero rows become uniform. It is fixed before new-model evaluation.
At zero residual the same-FP32 persistence ranks are preserved mathematically.
Actual NDCG agreement with native FP64 persistence still requires root
qualification of floating-point/log/softmax ties; no numeric agreement is claimed
here. The head receives current full-tree leaf forcing and
updated leaf memory. Country IDs are only persistent witness metadata.

tree_only supplies zero carry at every year, so output depends only on the current
causal tree. first_order uses H051 fixed alpha0 at genuine cut changes.
conditional_triple uses H051 fixed alpha1. Both call the frozen H051 transport with
basis="leaf", then broadcast BC-cell output to all country leaves.

Cuts are compared by country co-membership after anonymous label canonicalization.
Only genuine partition changes shift the last two DISTINCT cut episodes. Initial
state is zero; the first distinct change has a=b=initial cut, so the two memory
transports agree. Unchanged-cut annual holds retain full leaf state and each get
one GRU update with their own forcing; they do not cause an extra projection.
There is one discrete calendar-year update, with no bp/generation/ARG clock.

All memory arms store [255,64] leaf state. This source makes no compressed-state
advantage claim. The alpha0 carry equals Pi_B in exact arithmetic; alpha1 is
Pi_BC Pi_AB. Current-year forcing/readout and full tree information are identical.
TRAIN and all final forecasts follow increasing calendar year; no reverse scan,
teacher-forced target state or online held-out parameter update exists.

Expected nominal learned parameter count is 116,351 for every arm. In tree_only,
GRU temporal_cell.weight_hh has 12,288 structurally unused parameters because
carry is always zero, leaving 104,063 potentially active parameters. These are
source-derived counts, checked against actual model parameters only at a future
root run. Other data-dependent zero gradients are possible. Thus equal nominal
counts do not imply identical effective capacity. The main mechanism contrast is
conditional_triple versus first_order; history versus tree_only is descriptive.

## Interfaces and root adapter

All modules use package
research.reviews.TGB_trade_H051_training_source_20261016_v1.

TargetYearV1(year, values, source_present) is defined in targets_and_metrics_v1:
values is a CPU float64 [255,255] tensor, source_present a CPU bool [255] tensor,
in the exact official catalogue order. No target file reader is supplied.
Root must provide a real TRAIN-only reader/capability; returning a subset after
parsing held-out target objects is not this interface's target-closure contract.

DirectoryArtifactWriterV1(fresh_output_directory) creates a new output directory
only when begin_run is called. It writes exclusive common initialization/final
state files, all epoch losses, protocol, predictions, immutable seal and result
receipts. It reads no checkpoint and supports no resume. Raw primary prediction
payload is 5,462,100 bytes and final parameter payload 1,396,212 bytes before file
format overhead. This is not measured RSS/VRAM or an operational cap.

run_primary_v1 requires keyword arguments:
dataset, load_train_target_year, writer, runtime_check, device, protocol.
The Dataset must contain all 30 forecast years using the frozen official catalogue.
Preparation builds all CPU trees/schedules/event plans once. Every graph, schedule,
plan and TRAIN target then transfers once to the authorized device before epochs
and remains there across arms/epochs. No CPU tree construction or input transfer
occurs in the repeated epoch/year loop.

Root's separate adapter module exposes:

    def prepare_v1(protocol):
        # Authenticate actual source/runtime/admission and physical GPU0 first,
        # before numerical imports, payload materialization or CUDA work.
        return dict(
            dataset=causal_dataset,
            load_train_target_year=train_only_target_loader,
            writer=DirectoryArtifactWriterV1(new_output_directory),
            runtime_check=actual_root_runtime_check,
            device=authorized_logical_device)

The callback handles actual root guard/caps/drains; this packet contains no guard
implementation, lease, current pool, resource probe or device eligibility fallback.
The adapter must guard its own preparation. The scientific driver subsequently
calls runtime_check before scientific imports/allocation, each epoch/year,
checkpoint and prediction seal. No GPU4/5 assumption is present.

After actual root authorization, invoke the module:

    .runtime/env/bin/python3.11 -B -m research.reviews.TGB_trade_H051_training_source_20261016_v1.driver_v1 --execute-root --root-adapter ROOT_AUTHORIZED_ADAPTER_MODULE

Default CLI is inert and imports no Torch/model. --execute-root is an explicit
entry switch, not an authorization mechanism. The driver returns after sealing
predictions and never imports/calls the evaluator automatically.

## Sealing and split evaluation

Final epoch100 weights replay the entire 1987–2016 causal sequence from zero.
Scores for 2010–2016 are FP32 softmax destination probabilities, shape
[seed=1,arm=3,year=7,source=255,destination=255]. All are persisted before any VAL or
TEST target callback. The seal pins prediction bytes, canonical score payload
hash, protocol, common catalog/observations, initialization and final checkpoints.
Before scoring, the evaluator checks that in-memory payload still matches its
sealed hash.

evaluate_sealed_split_v1 requires:
bundle, split, load_target_year, baseline_results, baseline_result_ref,
writer, runtime_check, protocol.
Only the requested split's target years are queried. Root authenticates the
baseline JSON bytes against the pinned H052 result ref. NDCG@10 uses exactly the
official label-source support, averages queried countries within each year, then
equally averages years. It records year SD, model-minus-native and
conditional_triple-minus-first_order paired year differences.

Root explicitly evaluates VAL first, freezes any confirmation decision while TEST
is closed, and releases TEST separately. No TEST result selects epoch, optimizer,
arms, extra seed campaign or future gates. One initial seed cannot estimate
optimizer-seed variation; year SD is descriptive and serially dependent. Source
status, a tree-only gain or a single-seed metric is not a new-contribution or
original-paper-reproduction claim.

## Focused semantics fixture and remaining qualification

The supplied focused_semantics_fixture_v1.py is root-deferred:

    .runtime/env/bin/python3.11 -B research/reviews/TGB_trade_H051_training_source_20261016_v1/focused_semantics_fixture_v1.py --execute-fixture-root --output NEW_RECEIPT_OUTSIDE_PACKET.json

Its default is inert. Explicit CPU execution uses exact AST-extracted source
methods with model-free encoder/GRU/head stubs plus parameter-free H051 transport.
It checks cut-label invariance, distinct-cut event/hold semantics, common bootstrap,
expected alpha0/alpha1 carry, zero-history tree_only, exact TRAIN-only label calls,
query masks/loss, smoothed persistence rank preservation at zero residual,
residency order and source parameter arithmetic. It constructs
or forwards no trainable encoder/GRU/head and performs no optimization. No actual
fixture PASS is claimed by this author.

Root still owns actual model construction/counts, forward/autograd and strict
FP32 device qualification, authenticated actual data/target adapters, host/device
fit, source-bound admission, full epoch completion and outer terminal evidence.
The frozen Dataset's previous root fixture does not qualify this learned trainer.
