# Biological whole-tree candidate pilot v1

This is an unexecuted H020 handoff for two biological TRAIN and two VAL parents.
No TEST parent is admitted. Root first runs seed101; seeds202 and303 use the same
frozen protocol later. No actual bank preparation, label access, fitting, install,
GPU operation, or Git operation was performed by the producer agent.

The three conditions import existing models unchanged: AnchorKernelModel age
geometry/depth2, AnchorKernelModel plain geometry/depth2, and AnchorGRUModel with
matched source encoder/anchors and temporal depth2. These are pilot comparisons,
not a superiority, novelty, independence or holdout claim. All share the exact
same observed whole-tree inputs and precomputed plain/age banks.

## Observations and roles

The only preparation inputs are the completed hashed biological Relate manifest,
its observed source/derived rosters, coordinate grid, completion/command receipts,
and post-five-iteration inferred trees. The AP runner's audited observations
function is imported as a frozen definition-only helper; AP protocol, run and
TruthReader are never called. Preparation does not open the simulation's
target-bearing parent manifest or any truth/oracle tree or label file.

Actual sample_index-to-inferred-node mappings govern all280 sampled leaves.
Pure references are96 (24each Ana0,CHG1,WHG2,EHG3). Mixed reference-only64 and
actual query120 retain null source classes. Dynamic Q is every184 non-pure leaf;
R is the96 pure references. This exhausts the boundary partition required by
the unchanged bank implementation. Explicit actual-query output indexes select
only the120 supervised/predicted outputs from dynamic Q184. Mixed64 states may
participate in learned dynamics but never receive labels or contribute to loss,
selection, or metrics. Their query feature flag is zero.

No cohort feature is added. The48 observed fields are the existing base8:
four pure-reference class one-hots, actual-query flag (120only), sample flag,
log1p(age)/log1p(10000), and log1p(parent branch)/log1p(10000);
plus the unchanged ReferenceFeatureAdapter's40 source-major genealogy summaries
(4categories x10fields). These are descendant/panel reference fractions,
minimum/mean/std encoded TMRCA, minimum-reference fraction, nearest-reference vote,
and three fixed encoded-age kernel evidence values at0.05,0.1,0.2.
The raw parent-branch column is excluded by the existing leaf bank, yielding47
model input fields. Pure96 alone supply reference labels and16 observed reference
landmarks. Numeric node IDs are join keys and never feature values. All280 leaf
ages and all internal nodes/edges remain in each whole observed marginal tree.

## Physical tiles and frozen banks

The producer's cleaned inferred trees are LOCAL coordinates. Preparation adds the
declared coverage_left offset and uses original physical coordinates divided by
2,000,000, preserving the original2Mb normalization. It clips the eight original
250kb tiles to observed coverage. Every marginal tree, including trees containing
no retained center, remains inside its physical tile; boundary-crossing trees are
copied completely into adjacent tiles with only their genomic interval clipped.
Every declared retained100bp center appears exactly once. A positive tile without
a retained center, disconnected/missing-leaf tree, unsupported sample ancestor,
or invalid age orientation fails explicitly; no alternate packing or thinning is
invented.

Each tile stores a hashed whole observed graph and a hashed full bank. Unchanged
precompute_age_interfaces produces Q184/R96 operators with age350 and age1400,
strength3, time scale10000, resistance scale1e-4, and shared plain full-boundary
mean-diagonal normalization. All original plain/age cache fields, raw/normalized
matrices, eigenpairs, diagnostics and coordinate metadata are retained through
AgeInterfaceBank.as_dict. Source prepare_anchor_kernel_bank verifies the fixed
32-anchor/three-scale layout and operators before serialization.

PyTorch's ordinary lossless shared-storage serialization is used, with no field
compression, matrix reconstruction, quantization or eigensystem change. A full
per-field/tensor fingerprint is compared after a roundtrip, and the unchanged
AgeInterfaceBank.from_dict validation runs before sealing. The saved fixed
AnchorKernelBank object shares tensors with that full cache. Training loads its
hashed fixed prepared object and transfers it via unchanged
AnchorKernelBank.to(device,dtype=float32), using the existing consumer behavior
while coordinate tensors remain float64, without rebuilding eigenpairs or repeating offline matrix
verification inside neural forwards.

All conditions use width32, source depth2, latent8,32 anchors,3 scales and initial
couplings5,20,80. Candidate kernel depth is2, initial damping1 and full coupling;
the GRU has temporal depth2 and dropout0. The GRU's existing nominal configuration
text says bp/1e6; the biological receipt explicitly records actual coordinates as
original bp/2Mb for all conditions. Coordinates are never rescaled to local
coverage or independently normalized per tile.

## External supervision and validation

Separate ExternalTruthCapability objects admit only TRAIN loss or VAL scoring
after a prediction NPZ has been written and hashed. Only these capabilities open
the frozen external target manifest/full truth intervals; neither is passed to
observed packing, reference summaries, operators, feature banks or model inputs.

TRAIN loss integrates exact100bp prediction-bin/full-truth intersections into
class-span mass for actual query120 only. The existing classifier emits native
C4 probabilities directly. It applies no AP six-path collapse to these outputs,
and no probability floor or post-hoc renormalization is introduced.

Run30epochs maximum with patience8, AdamW lr0.001/weight decay1e-4, rate/mixture
weight decay0, gradient clipping1. Training is by physical tiles, with the same
seeded tile order for all active conditions. Each condition retains the minimum
mean parent VAL NLL checkpoint, using the64 present-day queries as primary.
Ancient56 are descriptive secondary endpoints. There is no calibration fitting
or query/bin selection based on VAL truth. All raw epoch probabilities and
selected checkpoint predictions are sealed before their score capabilities.
VAL early stopping is authorized for this development candidate, unlike AP's
fixed epoch5 fitting rule.

VAL NLL selection uses exact scored query-bp raw C4 probabilities per parent,
then an unweighted mean over the two parents. Zero true-class probability has
infinite NLL rather than a fabricated finite value. Selected checkpoint metrics
reuse the audited AP full_span_score implementation: exact intersections,
accuracy,NLL,Brier,confusion,spanIoU,ECE15,dosage and boundary/tract summaries.
Support is the union of declared retained-center bins clipped to coverage;
partial bins without retained centers and uncovered tails are excluded.
Denominators are exact represented query-bp. Physical coverage,represented
support,original2Mb normalization and conditional score scope are separate.

## Storage and execution bindings

Root supplies one complete successful project-root du snapshot before each
preparation or one-seed fit. Its hashed JSON has schema
biological_candidate_storage_snapshot_v1, status complete, project_root
/workspace-SR003.nfs2/ARG-GNN, command ["du","-s","-B1",project_root],
returncode0, used_bytes integer and timezone-aware completed_utc.
Root accounts for intervening toolchain/model writes within the reserved headroom;
the shared complete root snapshot can be reused without another recursive scan.

Admission uses a strict450GB ceiling and at least50GB concurrent-write reserve
beyond that snapshot. Preparation conservatively reserves12MB per producer tree
plus at most seven tile-boundary duplications per parent and a2GB fixed margin.
One-seed fitting reserves96MB per parent/epoch/condition for raw probabilities
and logits plus12GB for final predictions,metrics,checkpoints and receipts.
Direct writes are accounted against this reservation and filesystem free-space
checks use statvfs. There is no recursive storage scan per tile or epoch.
Insufficient admission fails without thinning trees or changing matrices.

The config freezes source/helper/model hashes and observed roster hashes.
Execution additionally requires exact config, completed Relate manifest and
prepared manifest SHA256 bindings supplied by root. Outputs must be fresh;
failed/partial artifacts are preserved. Observed preparation uses CPU and explicitly empty CUDA visibility.
Root FIT defaults to CUDA through the unchanged project launcher; explicit CPU
fitting remains possible with --device cpu and empty CUDA visibility.

## Root CUDA lease and process qualification

Before Torch imports or any CUDA context, module/script entry validates the real
inherited root-project lease and applies exact process title
AI Auto Research Pipeline Test through frozen native_argformer_exec helpers.
The unchanged launcher permits only physicalGPU2,3,4,5, reserves a free UUID,
passes the held descriptor and enforces at most two project GPU jobs. Root plans
GPU3 for candidate/controls while GPU2 runs the native ARGformer pipeline.

After frozen observed/config/prepared inputs and storage admission, fitting
applies the frozen deterministic execution policy before CUDA initialization.
It checks exactly one visible logical CUDA0, obtains the actual current driver
context UUID from its own loaded libcuda/proc maps using the stable driver API,
and compares it to the inherited lease and Torch property UUID when available.
It does not assume the isolated ARGformer's CUDA12.4 prefix or a device-property
ABI. The unchanged nvtop_snapshot helper records actual NVTOP output and verifies
own compute PID, reserved physical index and exact process title before labels
or scientific fitting. Source/helper/NVTOP executable bytes are frozen. Lease
identity is rechecked through epochs and completion. Receipts trace hardware,
actual title/UUID/NVTOP, original/updated parameter fingerprints, actual parameter
counts, optimizer groups/steps, parameter deltas and dynamic/rate controls.
No qualification or GPU call occurred during authoring.

From /workspace-SR003.nfs2/ARG-GNN, root's wrapper sources
scripts/project_env.sh and sets PYTHONDONTWRITEBYTECODE=1. CPU source-only and
preparation commands use CUDA_VISIBLE_DEVICES='' and script entry
scripts/train_biological_candidate_v1.py with --check-source-only or
--prepare-observed-cpu. Root CUDA fitting uses:

    .runtime/env/bin/python scripts/launch_gpu.py --gpu 3 --job biological-candidate-pilot-v1-seed101 \
      arggnn.biological_candidate --train --seed 101 --device cuda \
      --expected-config-sha256 CFG_SHA \
      --expected-observed-manifest-sha256 RELATE_MANIFEST_SHA \
      --expected-prepared-manifest-sha256 PREPARED_MANIFEST_SHA \
      --storage-snapshot ROOT_STORAGE_SNAPSHOT \
      --expected-storage-snapshot-sha256 ROOT_STORAGE_SNAPSHOT_SHA

Root fills the reviewed SHA values and root snapshot path before launching.
Required common binding is --expected-config-sha256.
Preparation additionally supplies --expected-observed-manifest-sha256,
--storage-snapshot and --expected-storage-snapshot-sha256.
Training supplies those plus --expected-prepared-manifest-sha256.
Default preparation manifest is data/biological_candidate_pilot_v1/manifest.json;
default seed101 output is results/biological_candidate_pilot_v1_seed101.

The completed inference contains955+897+946+920=3,718 marginal trees. Including
at most28 boundary copies, the12MB/tree conservative estimate plus2GB fixed
margin is46.952GB; shared root used58.976GB plus50GB reserve remains comfortably
below450GB. Actual serialized bytes are still accounted at preparation.

This handoff needs root review and actual runtime manifest/cache/fit evidence.
Syntax checking alone does not establish preparation, lossless real-bank
roundtrip, optimizer fitting or scientific performance.
