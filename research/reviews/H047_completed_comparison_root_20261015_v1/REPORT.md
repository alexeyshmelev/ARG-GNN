# H047 cohort-balanced TRAIN risk

Both arms completed30 full passes/960updates from identical H042 initialization with independent reset optimizer state. Eight independent shared VAL simulations, one optimizer seed. All96 products were sealed before truth. Selection used the original frozen rule.

| Arm | Present accuracy % ± sample SD | NLL ± sample SD | Boundary F1 % ± sample SD |
|---|---:|---:|---:|
| balanced_risk | 96.757631 ± 0.528671 | 0.096819 ± 0.021357 | 18.557398 ± 1.779961 |
| original_risk | 96.777660 ± 0.596958 | 0.094808 ± 0.020849 | 18.277471 ± 1.749334 |

Balanced-minus-original accuracy was−0.020029percentage points. It improved boundaryF1 by0.279927points, but worsened NLL. Six of eight parents had lower accuracy. The registered criterion failed. This does not support the balancing hypothesis or a novel architecture claim.

All seven cohort metrics and per-class precision/recall/F1/IoU, with defined-parent counts and sample SD, are in summary.json. No unchanged repeat is planned.
