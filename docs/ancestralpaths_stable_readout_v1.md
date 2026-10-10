# AP stable-logit auxiliary readout v1

This new numerical condition reuses the completed AncestralPaths biological v2
Dense checkpoints for seeds 101, 202 and 303 and the same two observed VAL caches.
It addresses float32 softmax underflow: preserved native six-class probabilities
can be exactly zero, making the raw C4 NLL infinite on truth-supported intervals.
It changes arithmetic after the trained final Dense layer. It does not train,
calibrate, change model weights or sample support, tune a probability floor, or
replace the original float32 predictions and scores. H014/H020 condition registration
and actual readout execution remain root actions.

The new runner/config integrity-bind the frozen AP v1/v2 definitions/configs,
completed v2 fit receipt, all three actual checkpoints, six original raw prediction
files and six original metric files. Metric files are hashed without score parsing
during preflight. Frozen v2 validation binds the qualified observed caches to exact
inferred-tree/sample/query/SNP/grid/bin/coverage identities. No feature extraction
is repeated and no original output is writable by this runner.

## Exact native logits and stable arithmetic

The loaded native Sequential architecture must be Flatten(5x9), Dense512 ReLU,
Dropout0.2, Dense6 softmax, with 26,630 parameters. Existing checkpoint weights
are loaded with compile=False; no optimizer or fit is created. Original seed
identities initialize the reload context, and inference explicitly uses
training=False, so dropout is disabled as in native prediction.

A Keras Model exposes the input to the final Dense layer. The exact unchanged
FP32 kernel and bias then produce logits using tf.matmul followed by tf.nn.bias_add,
matching the installed Keras2.15 Dense rank2 call before activation. Applying the
original softmax to those logits must agree with native model.predict within
absolute error 1e-6. Native model.predict keeps the original batch size 1024.
Checkpoint and model-weight hashes must remain unchanged.

FP32 logits are converted to float64. Subtract each row maximum and compute
logp6 = shifted_logits - log(sum(exp(shifted_logits))). C4 log probabilities use
logaddexp for native groups [0,4] and [1,5], and native columns 2 and 3 directly.
Exponentiating gives stable native/C4 probabilities; C4 probabilities must equal
the sums of native groups and normalize to one. There is no clipping, smoothing,
probability floor, temperature adjustment or fitted calibration.

Stable exp(logp6) must agree with actual native p6 within 1e-6, and recomputed native
p6 must agree with the preserved original p6 within 1e-6. Exact zeros are allowed
in both float32 and exponentiated float64 representations. Finite log probabilities
remain valid even if a float64 exponential itself underflows; NLL uses logp directly.
Native and C4 argmax disagreements, zero counts and probability errors are recorded.
Original benchmark accuracy and raw metrics remain preserved, including any
rounding/tie differences in the auxiliary argmax calculation.

## Sealing and unchanged physical score support

Every seed/parent exports a NEW NPZ with FP32 pre-softmax logits, float64 native6
and C4 log probabilities, stable probabilities, feature-bank joins and original
physical positions/bin edges/query identities/retained grid/coverage/offset.
A separate seal records hashes and numerical checks. All six prediction/logprob
files and the all-predictions seal are written before any VAL TruthReader is
instantiated or original metric scores are parsed.

Then the frozen v1 sealed_val_score truth capability and exact intersections
function join every complete truth interval against the same retained-center
clipped physical bins. Primary present-day64, secondary ancient56 and all120 query
groups remain identical, as do the exact query-bp denominator, original2Mb
normalization and exclusion of uncovered tails.

The auxiliary C4 NLL sums -logp4[true_class] times exact intersection/query spans,
then divides by exact scored query-bp. Native6 log-NLL is similarly reported as an
auxiliary quantity. Brier and descriptive 15-bin ECE use stable summed C4
probabilities with the unchanged source formulas and physical weights.
No calibration is fitted. The output retains the original raw NLL value and
infinite-NLL query-bp support beside finite direct-log NLL, plus Brier/ECE numerical
deltas. Original native confusion, accuracy, tract and other benchmark outputs
remain unchanged; this condition makes no new paper-CNN, quantitative reproduction
or final holdout claim.

## Root review and execution

The preparing agent performed only source inspection, metadata/hash binding and
handmade CPU arithmetic checks. It loaded no trained models or biological arrays,
read no VAL truth or VAL score values, and performed no training, GPU work,
installation or Git operation. Actual checkpoint/native-probability parity is
required during root execution; a source arithmetic check cannot establish it.

From /workspace-SR003.nfs2/ARG-GNN, source scripts/project_env.sh. Policy preflight
(default mode) hashes existing artifacts and reads only qualified observed cache
schemas; it imports no TensorFlow, loads no models and opens no truth:

    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python scripts/score_ancestralpaths_stable_logits_v1.py

After review/freezing the new source and config, use the existing qualified
TensorFlow CPU interpreter with a fresh separate output:

    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .runtime/conda/envs/ancestralpaths_cpu_v1/bin/python scripts/score_ancestralpaths_stable_logits_v1.py --execute --expected-config-sha256 <reviewed-policy-hash> --output experiments/results/ancestralpaths_stable_readout_v1

Keep outer stdout/stderr/exit code and the generated receipt/exit_status. Any failed
numerical parity or score stage is preserved in that fresh output; never overwrite
the source predictions, metric files or checkpoints to repair a readout failure.
