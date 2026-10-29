# Independent completed-development review, 29 October 2026

Verdict: PASS. The registered final-epoch accuracy rule selects **structure** for
the fresh 50,000-target benchmark fit. This is an internal-development selection
on one entirely held-out TRAIN chromosome; no TEST payload was read.

| Mode | Correct / 16,666 | Development accuracy | Cross-entropy | Macro F1 | Fit accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_cross | 16,068 | 96.4119% | 0.132655 | 0.962019 | 97.8580% |
| leaf_only | 16,075 | 96.4539% | 0.143026 | 0.962157 | 98.9710% |
| structure | 16,096 | 96.5799% | 0.155597 | 0.963435 | 99.4120% |

Structure gains 28 correct targets over no_cross and 21 over leaf_only, but has
worse cross-entropy than either. The registered primary metric still selects
structure; cross-entropy was only the exact-accuracy tie breaker. This is not
uniform metric improvement or evidence of general superiority. Targets within
a chromosome are correlated, and these small differences cannot establish
population-level significance. No architecture, learning rate or checkpoint
rule was changed after seeing these results.

The independent CPU audit authenticated every saved fit/development canonical
identity and label against the frozen cache and recomputed all metrics. Each
arm completed exactly 50 epochs, 2,600 updates and 1,666,700 target presentations.
All arms had identical initial-state and training-order hashes. Their active
optimizer states numbered 40, 49 and 55 parameter tensors; each active tensor
changed and recorded step2600, including the focal readout and classifier.
All model and optimizer states were finite. Final checkpoint model values were
identical to final_state.pt. CPU model/optimizer reload succeeded, and one
14-target development site's restored outputs per arm matched saved GPU logits
within the prospective tolerance (worst absolute error 3.3379e-6). The reviewer
performed no optimizer update. All eight owned workers per arm exited zero and
all loader pin threads stopped cleanly.

All six classes are represented in final predictions. Development recall:

| Class | no_cross | leaf_only | structure |
| --- | ---: | ---: | ---: |
| Ana_nonBAA | 98.5717% | 98.2146% | 98.2543% |
| CHG_nonBAA | 94.9269% | 93.9524% | 95.8441% |
| WHG | 91.6667% | 93.8690% | 94.4643% |
| EHG | 95.3235% | 96.2204% | 94.3306% |
| Ana_viaBAA | 99.0157% | 98.5039% | 98.2677% |
| CHG_viaBAA | 95.2141% | 96.0957% | 97.1033% |

These are uneven class tradeoffs, not class collapse. Gradient clipping occurred
in 39.65%, 40.15% and 40.96% of updates respectively. The maximum unclipped
structure gradient norm was 24.09; all checks remained finite.

Structure's 49 warmed training epochs took 218.888 seconds, or 7,462.1 useful
canonical targets/second and 11.641 optimizer updates/second. The first epoch
took 28.847 seconds, including 21.972 seconds starting its iterator and
23.140 seconds to the first batch. The entire run including development/final
fit evaluations took 293.127 seconds. Exact recorded epoch boundaries identify
436 warm-interval device samples averaging 85.01% GPU utilization. No foreign
compute PID was observed on GPU5. Peak allocated/reserved device memory was
3.335/25.296 GB; parent RSS was 2.728 GB and excludes worker memory. CPU loader
wait and CUDA event timings overlap and are not additive wall-time fractions.

No-cross and leaf-only shared GPU4, so their device utilization cannot be
attributed separately. Structure's sustained activity supports useful execution;
it does not establish scientific accuracy, comparative speedup, or convergence
advantage. The earlier offline export/preparation cost remains separate.

The machine-readable review is COMPLETED_DEVELOPMENT_REVIEW.json, SHA256
`d655811ff9da515a317133a60dbda986eb85168bbc05c29e4ee82176d5e8e52a`.
It binds result, prediction, activity and checkpoint hashes and includes the
full confusion matrices, per-class metrics, changed parameter names and timing
breakdowns. Benchmark initialization must be fresh seed101, with the selected
structure arm on all 1,245 TRAIN sites / 50,000 targets. At 16 sites per batch,
50 epochs mean 3,900 optimizer updates and 2,500,000 target presentations.
