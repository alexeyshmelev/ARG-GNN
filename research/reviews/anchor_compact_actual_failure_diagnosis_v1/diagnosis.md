# Compact actual-bank failure diagnosis v1

Read-only source and receipt diagnosis. No scientific array, bank, checkpoint or model was opened or replayed. No GPU, install, Git, ledger or decision operation occurred. Writes are restricted to this new evidence directory. The failed attempt and tolerances remain unchanged.

## Main finding

Recorded TF32/matmul-policy drift is unsupported. The runtime receipt and all 70 stage-boundary observations report `float32_matmul_precision=highest`, CUDA/CuDNN TF32 disabled and strict determinism. `development_execution.py:21–48` sets these values before CUDA and verifies actual getters. The bootstrap applies this policy before initialization, and the runner rechecks it at every boundary.

The leading source-supported hypothesis is accumulation from different finite-precision recurrences. Native `anchor_kernel.py:186–195,258–274` projects physical state with U-transpose and reconstructs with U at every event subsegment, including readouts. Its layout includes the original approximately 2,500 readouts per selected tile. Compact `anchor_spectral_compact_v1.py:300–353` retains spectral state, crosses bases at hold boundaries, and reconstructs grouped readouts. Both reuse the same eigenpairs, but CUDA transfer rounds them to float32. No post-cast orthogonality measurement is recorded.

For ideal orthogonal U, P(d2)P(d1)=P(d1+d2). With stored rounded U and E=U-transpose U−I, even exact arithmetic on those stored entries instead gives:

`P(d2)P(d1) = U D(d2) (I+E) D(d1) U-transpose`.

The compact semigroup step omits the intervening E. Native readout subdivision repeatedly applies this factor, while compact subdivision omits most readout-triggered roundtrips. Floating GEMM roundoff adds another effect. This is a concrete source-level reason that real-arithmetic equivalence does not imply finite-precision equivalence. It is not proof that native or compact is the more accurate approximation.

## Recorded pattern and localization

All 34 frozen cells executed: 20 PASS, 7 action FAIL, 7 model parity failures with optimizer not run. All 14 CPU float64 full-coupling probes pass. All 14 CUDA float32 full-coupling probes fail. Scalar, diagonal and gain-matched-diagonal direct controls pass on both devices. Gain-matched controls still perform eigenbasis source transforms but avoid repeated full-coupling propagation, which further implicates the recurrence rather than any eigenbasis multiplication alone.

Every failed independent action probe first disagrees in raw per-scale action, numerator, mass and normalized output, using an identical source bundle. CUDA raw-action maximum absolute errors range from 0.000283777714 to 0.000369250774. Action VJPs pass their frozen checks. Full-model first-layer source/mass-source checks pass, but downstream targets and several encoder VJPs fail. No selected optimizer mutation occurred in those seven model cells. Profiling is blocked, so no speed or memory gain is established.

The source layout/anchor/output adapter validates identical physical coordinates and source maps. All CPU float64 geometry/state cases pass strict checks, including the stress state. These observations argue against an obvious source-order,anchor-count,query-subset or chronology change. They do not prove absence of every implementation defect.

## Secondary mechanisms and limits

Native and compact also split exponential factors differently, add sources in different bases and use different GEMM shapes. Native state products have H×V=16 columns, while compact readout packing can use B×H×V up to 2,048 columns. Highest precision does not guarantee the same reduction order for these products. Scalar/diagonal controls pass with the same coordinate regrouping, so exponential splitting alone is a weaker explanation than repeated eigenbasis roundtrips.

The frozen matrix contains CPU float64 and CUDA float32 only. Dtype and device effects are confounded. Receipts store maximum errors rather than signed position trajectories or post-cast U-transpose U residuals. No causal intervention or independent high-precision oracle was run here. The failed qualification establishes an implementation-equivalence problem under the frozen CUDA float32 condition, not a demonstrated change to the scientific kernel definition.

## What would distinguish the hypotheses

Root may design a distinct, fully disclosed condition that preserves all original files, all 34 frozen cases, full query/readout support and the frozen tolerances. The useful additional evidence is post-cast orthogonality by every original hold, matched native/compact error trajectories against a frozen high-precision reference, and crossed CPU float32/CUDA float64 coverage. Any changed accumulation precision,readout recurrence or GEMM implementation requires new pins and independent qualification. This diagnosis authorizes none of those actions.

Do not rerun the same failed condition unchanged, loosen tolerances, select only passing controls or declare the compact result preferable without reference evidence. A numerical precision remedy must be named and disclosed, with complete coverage and the original failure preserved. TF32 is already disabled, so disabling it again is not a new recorded condition.
