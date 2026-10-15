# H045 completed paired development comparison

Both arms completed 30 FP32 passes and 960 whole-parent updates from the same H042 pass-30 initialization. All 96 validation products were sealed before validation truth. Genuine launcher and outer evaluation waits are 0. No outer training wait is invented.

| Arm | Present182 accuracy, mean ± SD (%) | NLL | Boundary F1, mean ± SD (%) |
| --- | ---: | ---: | ---: |
| foreign_episode | 96.8270 ± 0.5870 | 0.092980 | 18.1558 ± 1.3790 |
| own_state | 96.8175 ± 0.6142 | 0.092976 | 18.0557 ± 1.7761 |

Foreign-episode context gains 0.009568 percentage points accuracy and 0.100147 pp boundary F1. It does not reach the registered 0.10 pp accuracy threshold. Useful cross-tree exchange and novelty remain unestablished.

SD is across eight independent ARG validation parents, one optimizer seed. This reused validation set supports development, not independent test or original-paper reproduction. H044/H045 differ in readout and capacity, so their difference does not isolate the episode mechanism. Prior art and hold-subdivision dependence remain documented.

Exact completions, checkpoints, waits and all population metrics are bound in [summary.json](summary.json).
