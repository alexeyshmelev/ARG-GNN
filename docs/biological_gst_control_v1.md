# Biological GST control v1

This is an ARG task adaptation of the existing, unchanged GST control in
src/arggnn/gst_baseline.py. It is not an original-domain numerical reproduction
of GST, and it does not import the official model implementation. The existing
control is an independent reconstruction of the paper equations, with the
explicit interpretation and source differences in docs/gst_baseline.md and
docs/gst_baseline_provenance.json. This integration changes input admission and
the biological training harness. It does not change GST layers or its existing
query-leaf readout.

The source-only handoff contains no biological predictions, model fit,
performance comparison, or TEST evaluation. The primary agent owns experiment
registration, GPU scheduling, actual execution, result interpretation, and Git.

## Prespecified configurations

The factual, score-free H009 selection binding is
experiments/results/biological_gst_control_v1_root_selection_metadata/metadata.json,
SHA256 c0896a8ef32bc6e882a547febcd809bf3197f79e792cdce9efad98dce248d99f.
It identifies the fixed gst_ca_span mechanism control in
configs/inferred_cohort64_gst_graphssm_controls_v1.json. It does not identify a
winner from the separate eight-configuration development tuning grid.
The old F48 width32 count was 43,332. Its old training budget and dataset are
not transferred to biology.

| Arm | Width | Depth | Heads | Dropout | F47 parameters | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gst32_primary | 32 | 2 | 4 | 0 | 43,300 | Actual H009 fixed configuration |
| gst64_capacity_sensitivity | 64 | 2 | 4 | 0 | 166,468 | Prespecified capacity sensitivity |

Both arms use cross-attention, span-weighted graph-token keys, the paper gate,
FFN multiplier 2, self-loop weight 2, and the existing query-leaf MLP with four
outputs. Learning rate is 0.001 and AdamW weight decay is 0.0001 on all learned
GST parameters. The analytic count for the exact F47/C4/depth2 configuration is
39 H² + 105 H + 4. Handmade model instantiations verified both counts.
Width64 was fixed before any biological GST outcomes and does not replace the
primary width32 arm.

## Frozen biological input

The integration reuses H020 preparation, without preparing another bank.

- Frozen H020 config is configs/biological_candidate_pilot_v1.json,
  SHA256 4151e7e156db3f188512bcc738b43597407d7707bc77c420509165c2e9bfc2f1.
- Completed observed producer is
  data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json,
  SHA256 6872222127c96b3defbfb1b9edb83ba1d5b11b0b60e6888956369bc507cefe28.
- Root-supplied completed prepared manifest is
  data/biological_candidate_pilot_v1/manifest.json,
  SHA256 3c0a3320fbc6dd9d421a2acfad398961c4912cdbc49d111faf02108ee4f8596d.

Actual fitting loads only the hashed tile_XX.whole_observed.pt objects. The
Kron bank records remain provenance and are never loaded or reconstructed.
The inference input consists of all 280 sampled leaves and all retained
internal nodes in every original marginal tree. It retains all 3,718 original
records, with parent counts 955, 897, 946, and 920, plus explicit copies at
physical tile boundaries. There are the same 32 existing physical tiles,
eight per development parent. All both-direction tree edges, interval spans,
node identity joins, and stored clade metadata are retained. GST ignores the
clade metadata.

Every stored node has the frozen 48 observed fields. source47 removes only
raw parent-branch column 7 from every node. It retains columns 0 through 6
and 8 through 47 in their original order. No summaries are recomputed.
Coordinates and window bounds remain float64 original physical bp/2,000,000.
There is no tile-local coordinate rescaling. Features and model weights use
float32. Readouts retain the original 100 bp centers and bins clipped to
observed physical coverage, with uncovered tails excluded.

Pure96 source haplotypes provide 24 observed class labels per C4 class in
the frozen order ANA, CHG, WHG, EHG. Mixed64 haplotypes remain unlabeled,
with actual-query flag zero and no reference class. The serialized Q184
query namespace is retained throughout the model forward. The existing
actual-query output indexes select query120 only after the forward for
loss, saved predictions, and scoring. Integer IDs are lookup keys, not
learned embeddings. The existing GST architecture has no positional encoding
and no genomic ordering mechanism beyond span key mass.

One complete existing physical tile is one GST set and one forward call.
The integration never batches subsets of marginal graphs, thins leaves,
resamples marginal records, selects a nearest subset of references, or
creates a reduced graph substitute. Resource exhaustion must preserve the
partial receipt and stop the cell. It must not trigger an architectural
fallback or a hidden change to context.

There is a method-specific input and aggregation difference. GST consumes
F47 on all internal nodes as well as all leaves, and operates on full observed
topology. H020's neural source encoder consumes leafF47 together with its
Kron response operators. The observed source feature construction, leaf roles,
support, target capability, and output task are matched, but the internal-node
feature access and aggregation mechanism are not identical.

## Loss and selection

Training uses the same frozen external C4 truth capability as H020. TRAIN
truth is admitted only for actual query120 class-span loss. Both VAL parents
and all active GST arms have their epoch predictions sealed to hashed NPZ
files before epoch scoring. VAL truth is admitted after sealed predictions,
and cached afterward only for the common selection and scoring definitions.
Neither query nor mixed truth enters model features.

The loss is the exact class-span mass times log_softmax at actual query120,
normalized by its query-bp denominator. Each complete TRAIN tile produces one
optimizer update per active arm. Tile order uses the fixed seed plus epoch.
The cap is 30 epochs with patience 8 and gradient clip norm 1. Seeds are
101, 202, and 303. This matches H020's cap and does not use H009's old budget.

Checkpoint selection is the minimum unweighted mean of the two VAL parent
present64 NLL values. Each parent value uses exact retained-center clipped-bin
query-bp support and raw softmax probabilities. Strict less-than comparison
retains the earliest epoch at an exact tie. The selected checkpoint is
restored before final VAL predictions and metrics. A changed-weight check
must demonstrate actual optimizer updates.

There is no probability floor, posthoc probability renormalization, or VAL
calibration fit. Positive target mass on a zero predicted probability gives
infinite NLL. Final metrics reuse the frozen H020 full_span_score definitions
for present64, ancient56, and all120. No TEST parent is present.

## Admission and root execution

Default invocation is source-only. It verifies source/config hashes and the
score-free H009 metadata with the Python standard library. It does not import
Torch or NumPy, deserialize observed/model tensors, open truth, or access a GPU.

From the anogena project root:

~~~bash
source scripts/project_env.sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" \
  scripts/train_biological_gst_control_v1.py \
  --expected-config-sha256 GST_CONFIG_SHA256
~~~

The final config hash is recorded in the adjacent source audit receipt.
Actual fitting is root-only and must follow substantive-experiment
registration and a fresh complete successful root storage snapshot. The
unchanged launcher enforces allowed physical GPUs, a single leased visible
UUID, the exact AI Auto Research Pipeline Test process title, and the
maximum of two concurrent project GPU jobs.

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" scripts/launch_gpu.py \
  --gpu ROOT_SELECTED_FREE_ALLOWED_GPU \
  --job biological_gst_control_v1_seed101 \
  arggnn.biological_gst_control_v1 \
  --train \
  --expected-config-sha256 GST_CONFIG_SHA256 \
  --expected-observed-manifest-sha256 6872222127c96b3defbfb1b9edb83ba1d5b11b0b60e6888956369bc507cefe28 \
  --expected-prepared-manifest-sha256 3c0a3320fbc6dd9d421a2acfad398961c4912cdbc49d111faf02108ee4f8596d \
  --seed 101 \
  --device cuda \
  --output experiments/results/biological_gst_control_v1_seed101 \
  --storage-snapshot ROOT_COMPLETE_STORAGE_SNAPSHOT_JSON \
  --expected-storage-snapshot-sha256 ROOT_STORAGE_SNAPSHOT_SHA256
~~~

The source-only auditor did not run this fit command. Root may explicitly
choose CPU fitting with empty CUDA visibility and the same frozen policy.
Actual GPU admission reuses H020 configure_fit, with title and deterministic
policy before CUDA, driver-context UUID checks, Torch properties, and an
own-PID/title/physical-device NVTOP receipt. It does not assume a foreign CUDA
runtime version or change the official ARGformer runtime.

Outputs must be fresh. The conservative two-arm reservation is 23.52 GB,
in addition to the 50 GB concurrent-write reserve and the root snapshot's
existing usage, under the 450 GB ceiling. No repeated root du scan is added.
New predictions, checkpoints, and metrics are directly accounted.
Failures retain receipts, traces, predictions, and checkpoints. Final source,
config, observed-manifest, prepared-manifest, and whole-graph hashes are
rechecked before completion.

## Source-only verification

tests/test_biological_gst_control_v1.py contains handmade CPU semantic checks.
The artificial fixture has three complete 280-leaf star trees with H020
role counts and original-domain coordinates. It is not a biological model
input. Checks cover all-node projection and tensor preservation, both
parameter counts, full Q184 forward with query120-only direct loss, mixed
role rejection, pure-class integrity, rejection of a thinned tree, unscored
graph context reaching readouts, raw zero-probability infinite NLL, and
a source preflight that forbids Torch and NumPy imports.

Audit logs, hashes, limitations, and source review are under
research/reviews/biological_gst_control_v1_source_audit. Their passing checks
do not establish an actual fit, biological score, runtime memory bound,
performance result, or original-domain reproduction.
