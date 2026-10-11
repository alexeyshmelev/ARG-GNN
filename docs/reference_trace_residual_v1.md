# Reference-driven persistence residual v1 (H027)

This information-route hypothesis reuses the established H025 finite persistence
operator. It is not a new operator or GNN claim. Compare sample_only,
genuine_ancestor and sham_ancestor with identical observed reference inputs,
parameters, initialization and training support.

The immutable selected H025 genomic smoother, epoch10, is the frozen output
skip. Its exact per250kb-tile computation is reused. Query unaries are never
propagated through the H027 graph and never enter the learned context head.
Only the observed96 pure-reference proxy labels supply graph signal.

## Observed sources and complete carrier

All original metric-tree vertices and one midpoint per original branch remain.
For a hold, derive an actual sample ID to local native sample row join. Reference
carrier rows are that joined row plus the cumulative counts
N_native + N_original_branches of all preceding holds. Native or actual IDs
are never assumed to be carrier rows. The original operator's Q source-row
descriptors are checked against the same joins for consistency; they are not
H027 signal or mass sources.

Partition reference sources by their direct observed native age:84 younger
than350 and12 at/older than350. Both cohorts enter every arm at their original
native sample nodes. There are no invented census states. The eight fixed
source channels are four observed classes for each cohort. Older labels are
proxies, not hard historical ancestry truth. The older cohort has no ANA donor;
its ANA numerator is zero. A cohort ones source is normalization mass, not an
ANA or other class label.

For each cohort/class present in the observations, a source has weight

    physical_hold_width / genomic_scale_bp / (number_present_classes * class_count)

Signal is this weight times its observed onehot class in the appropriate cohort.
Mass has the identical weight on the cohort's four ones channels. No signal or
mass is placed at an unlabeled query or context sample.

## Matched established operator

Each channel uses the existing screened system

    A_h = I + spatial_rate_h * L_spatial + temporal_rate_h * L_control

L_control is the sample continuity operator, plus genuine or sham ancestral
traces for the corresponding arm. Qualified parent-wide native endpoints,
actual sample continuity and the original physical readout interpolation are
unchanged from H025. Ancestral sham pairing remains the original age/duration/
endpoint-role matched control, including its reported singleton degeneracy.

Solve all eight signal and eight matching mass channels together with the same
channel rates, geometry, sources, weights and readout. Divide readout signal
by readout mass only after all masses are finite and strictly positive. No
floor, clamp, replacement query source or fictitious reference class is used.
The exact screened matrix is positive; its FP32 approximate readout mass is
not guaranteed positive on real data. Failure preserves partial outputs.

Spatial and temporal rates are learned, bounded in(1e-4,20). Fixed initial
rates are18 and1 respectively to provide stronger initial spatial spread.
The existing FP32 implicit solver uses true-equation relative tolerance1e-6,
absolute tolerance0 and at most128 iterations, in both forward and adjoint.
No FP64 sweep or matrix qualification is part of this pilot.

Center the resulting eight-channel context separately at every original
readout over all actual queries of the same exact observed age. This is
unlabeled transduction over the full Q120 roster, not a minibatch mean.
A zero-initialized bias-free8-to4 head supplies the correction. Baseline logits
are only the detached skip; there is no standalone pointwise calibration head.
The48 stored parameters are16 rate coordinates and32 context-head weights.

## Pilot and input contract

One seed101,10epochs and160 original TRAIN tile updates per arm. Reuse exact
TRAIN query120 class-span loss, all32 original tiles, full original support and
sealed equal-parent VAL Present64 scoring. Select earliest minimum development
VAL NLL and report both accuracy and NLL beside the frozen smoother. No TEST
claim is made. Keep compact epoch metrics and current/best predictions and
checkpoints only, under a2GB root reservation.

Driver CLI:

    python -m arggnn.train_reference_trace_residual_v1 --execute --input INPUT --expected-input-sha256 SHA

The pinned input schema is reference_trace_residual_input_v1. It has the same
explicit frozen-baseline source/config/completed/checkpoint refs, original
observed/emitter/target joins and runtime fields as H026, but new H027
source/module/config hashes and storage. Root owns registration, GPU lease,
launch and reservation. Resource checks occur at input, preparation and epoch
boundaries; the existing writer guards planned products. No filesystem budget
or source revalidation is performed inside each neural arm step.

One58-state, two-hold handmade CPU FP32 sanity case used arbitrary actual IDs
1000..9003 and verified reference carrier rows4,5,6,7,33,34,35,36. All three
controls reproduced the frozen baseline exactly at zero-head initialization,
had positive readout mass and finite backward, and received nonzero spatial
and temporal rate gradients after the head moved. Minimum mass in that case
was approximately5.7e-8 to1.4e-7. This is a small source sanity check, not
biological qualification or evidence of predictive benefit.
