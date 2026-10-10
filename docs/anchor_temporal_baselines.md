# Same-source fixed-anchor temporal comparators

CPU-verified implementation for the current exploratory inferred Q64 pilot.
Root owns integration, independent audit, training, GPU scheduling, and later
fresh-development binding. No training, dataset scoring, GPU job, install, Git
operation, or existing shared source/cache/configuration edit occurred here.

## Common input and API

AnchorGRUModel and AnchorTemporalTransformerModel consume the existing
AnchorKernelBank from prepare_anchor_kernel_bank and return [P,Q,4] logits in
original readout/query order. Defaults match the candidate's input and anchor
scope: **47 observed fields**, independent plain BoundaryResponseGNN width32,
source depth2, latent8, S3 source coupling initializations [5,20,80], and
32 fixed physical midpoint anchors. Both temporal models have contextual depth2.
The plain source encoder is the exact existing class, independently trained in
each comparator. It has the same schema and initialization path; no weights are
shared between fitted models unless explicitly copied for a contract test.

At each layer and anchor, references remain the independently encoded observed
reference embeddings. Query sources use the current contextual anchor embedding.
The pre-temporal source and matching ones mass use the candidate's exact rule:

~~~text
signal_s = tanh(q(query)) - rate_s * Cqr_plain * tanh(r(reference))
mass_s   = 1 - rate_s * Cqr_plain * 1_reference
point_source = sum_s(pi_s * signal_s) / sum_s(pi_s * mass_s)
~~~

Tests copy the candidate's encoder, q/r maps and source rates, then verify exact
source and mass tensor equality. Couplings are positive global channel scalars
init*exp(raw), and mixtures are global channel softmaxes. Source rates/mixtures
receive zero weight decay. There is no unused damping parameter.

Pointwise branch reduction occurs before temporal encoding. The kernel candidate
instead propagates separate scale states and mixes their messages afterward.
Thus these are input/source-conditioned baseline families, not equal-state or
equal-parameter ablations. Source Cqr is **plain** in these comparators; an age
candidate changes its own propagated operators/source geometry. All retain the
same observed features and plain local encoder.

~~~python
import torch
from arggnn.anchor_kernel import prepare_anchor_kernel_bank
from arggnn.anchor_temporal_baselines import (
    AnchorGRUModel, AnchorTemporalTransformerModel,
    ObservedReferenceCoordinateSmoother,
)

prepared = prepare_anchor_kernel_bank(
    verified_plain_bank, num_anchors=32, num_scales=3,
).to("cpu", dtype=torch.float32)
model = AnchorGRUModel()  # or AnchorTemporalTransformerModel()
logits = model(prepared)
~~~

Neural encode_anchors exposes fixed anchor states, clamped references and
per-layer source diagnostics. encode_context additionally evaluates arbitrary
readouts. All anchor computations finish before any target decoder call.

## Temporal mechanisms and independent readouts

The GRU is bidirectional along each query's fixed anchor sequence. Each layer
takes the normalized source plus absolute coordinate and physical cell width.
It has H recurrent channels **per direction**, hence 2H recurrent channels per
query. A projection returns latent messages. Arbitrary readouts use physical
linear interpolation of those messages, with constant endpoint extrapolation,
then the same source-conditioned residual gate/norm form as the candidate.
Its temporal state does not exchange queries after the shared local encoder.

The Transformer uses joint self-attention over **A*Q anchor/query tokens**, allowing
temporal cross-query exchange. Tokens contain projected source and physical
coordinate fields, with no numerical sample-ID embedding. A separate querywise
cross-attention decoder reads each query's globally contextual anchor states.
Readouts do not attend to one another and are never encoder/source tokens.
The original query correspondence selects the appropriate decoder bank.

Coordinates remain globally **bp/1e6**. Physical differences/interpolation
fractions are formed in float64 before feature-dtype casting. The scale is not
reset per tile. Both families use noncausal known observed ARG context, including
anchors on both sides of a readout; no future ancestry labels are used.

Deterministic evaluation tests show that adding, duplicating, or permuting
readouts leaves existing predictions and anchor sources unchanged. Anchor-bank
independence also holds in training mode under the same RNG state with dropout:
target decoding occurs after all anchor draws. Training-time decoder dropout
is stochastic, so changing the target batch need not retain identical individual
decoder masks. Pilot dropout is zero.

## Simple observed-reference coordinate smoother

ObservedReferenceCoordinateSmoother uses the same verified plain bank and anchors
but **has no neural encoder and no trainable parameters**. It transparently uses
a subset of the available observed inputs: one-hot reference roles and plain
reference conductance evidence -Cqr. At each anchor, it sums conductance into
the observed reference categories. An exponential physical-distance kernel with
anchor-cell masses smooths that evidence. A positive prespecified pseudocount
gives finite normalized class scores. Stable log-normalization is used.

Bandwidth is specified in bp and converted by bandwidth_bp/1e6. The pilot
evaluates every combination of 10/30/100/300 kb and pseudocount0.001/0.01.
These **eight fixed settings** use the same equally weighted parent validation
raw-NLL criterion. No neural training or score/temperature fitting occurs for
this family. Any later fitted score or temperature requires a separately frozen
train-only procedure.

## Capacity and attention resources

Counts include the same selected47 input schema, source depth2, latent8, S3 and
temporal/context depth2. Width equality does not imply capacity equality.

| Model | Width | Source encoder parameters | Context parameters | Head | Total |
|---|---:|---:|---:|---:|---:|
| Existing S3 depth2 kernel | 32 | 5824 | 4464 | 132 | 10420 |
| Fixed-anchor GRU | 32 | 5824 | 22352 | 132 | 28308 |
| Fixed-anchor temporal Transformer | 32 | 5824 | 41744 | 132 | 47700 |
| Fixed-anchor GRU | 64 | 19840 | 73232 | 260 | 93332 |
| Fixed-anchor temporal Transformer | 64 | 19840 | 157072 | 260 | 177172 |
| Observed-reference smoother | n/a | 0 | 0 | 0 | 0 |

The Transformer checks the conservative dense attention-score estimate
heads*(A*Q)^2*dtype_bytes before joint attention. The default declared bound is
**256 MiB per layer for that score estimate alone**. It does not cap total
allocator memory. Feed-forward/embedding activations, gradients, optimizer
states, other attention tensors, caches, and workspaces need additional memory.
Efficient PyTorch attention may use less than the dense estimate.

For Q64/A32/four heads, float32 dense scores are 64 MiB per layer (float64:
128 MiB). Querywise decoder scores scale as Q*heads*P*A rather than (P*Q)^2.
Joint score memory grows quadratically in A*Q. The GRU recurrent state scales
linearly in A*Q*2H. Actual forward/backward time and allocator peaks must be
recorded during scheduled runs; none were measured on a GPU here.

## Runner, configurations and reproduction

New train_anchor_temporal.py validates the bound train/validation protocol,
loads the sealed plain dataset, prepares the same anchors, and keeps external
targets confined to training loss and post-forward evaluation. It uses
confirmatory_metrics.logit_metrics for **raw float64 stable-log-softmax
uncalibrated NLL**, with no hidden probability floor. Each checkpoint is selected
by minimum equally weighted parent validation NLL, with earliest epoch on exact
ties. Raw ordered validation logits, coordinates, query IDs, code/config hashes,
checkpoints, model manifests, learning curves and allocator peaks are retained.

The pilot configuration binds the existing16 TRAIN/four VAL parents, width32,
dropout0, LR0.001, 24 epochs/patience6 and ten model IDs: GRU, Transformer and
eight untrained smoother settings. It is exploratory reused-validation evidence.

The separate development search plan is **proposal-only** until root binds
fresh observed manifests. It follows confirmatory_design_v1: each neural family
gets eight width32/64 × dropout0/0.1 × LR0.0003/0.001 configurations, three seeds
101/202/303, max48 epochs/patience8, plus the eight simple smoothing settings.
That JSON does not dispatch work or promote the pilot to confirmation.

~~~sh
source scripts/project_env.sh
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$ARGGNN_PYTHON" -m pytest -q   tests/test_anchor_temporal_baselines.py tests/test_train_anchor_temporal.py   --basetemp=data/NEW_TEMPORAL_TEST_TMP

"$ARGGNN_PYTHON" -m arggnn.train_anchor_temporal   --config configs/anchor_inferred_cohort64_temporal_baselines_pilot_v1.json   --validate-config

# Root may run the pilot separately after integration/audit:
"$ARGGNN_PYTHON" -m arggnn.train_anchor_temporal   --config configs/anchor_inferred_cohort64_temporal_baselines_pilot_v1.json   --data data/boundary_inferred_cohort64_pilot_v1   --output experiments/NEW_TEMPORAL_PILOT --seed 101 --device cpu
~~~

GPU execution requires the existing project launcher/lease and explicit
--device cuda. It was not launched by this task.

## CPU verification and source freeze

**27 tests passed in 4.44 s**. They cover exact source/mass matching, all neural
parameter gradients and observed feature reachability, independent readouts,
dropout anchor-bank independence, unary/interval/storage invariance, numerical
ID/query alignment, external annotation guards, reference-input counterfactuals,
no online eigenproblem/solve, weights-only model-state round trips, physical
interpolation and smoother units, attention-budget rejection, protocol admission,
rate/mixture optimizer groups, parent scatter order, external label joining, and
raw-logit NLL1000 versus historical probability-floor NLL<30 on a synthetic case.
All ten pilot factories also passed configuration-only validation.

| New file | SHA256 |
|---|---|
| anchor_temporal_baselines.py | 3e88095eb466ea25db5f69553dfa31e6e002fa55ce4d31c4ddb17cb73115518b |
| train_anchor_temporal.py | 6cbce0c3e69aa7278213b4f8a7411c6ca9b86c9372ba9fb2627781b96810d63b |
| test_anchor_temporal_baselines.py | 9a8f06d615cac6467081fe73b568c2385611a1d11fa8477a39530686077084cc |
| test_train_anchor_temporal.py | 4d93a5bf9d327522b1eef14869f8f6e4ff0416ddb625104ce794b5da369aacc5 |
| Pilot config | a0e8e3806185926b083f7430a28d68ab3ea9ce26f1e231c5074f22e0ec378464 |
| Development search plan | 3e06918d5c3336c3227cb99afbd8eae20140e4e1a5507f9f1f1a711b963c462b |

Frozen anchor_kernel.py remains
2b8cc1c1726f746f961c0d63a4870e806a1a9d3d6e1fa6724dc0bda39967b083.
The reused metrics helper hash is
4e4f906cd6a302ffc422302c6f10694594856faab02d0da7565c508baf4c0a80.
The inspected confirmatory proposal hash is
b73321c273971619a5fa1bd5b865ad3e15f1a254eca2b36c08cda61ab4679d54.
The runner records its complete dependency snapshot before training and rejects
implementation drift before sealing results.
