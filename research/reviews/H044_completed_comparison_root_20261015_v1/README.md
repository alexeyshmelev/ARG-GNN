# H044 measured paired comparison

Both 31,428-parameter arms completed 30 FP32 passes and 960 whole-parent updates each from identical H042 initialization. All 96 prediction products were sealed before validation truth. No training or evaluation rerun occurred.

| Arm | Present182 accuracy, mean ± SD (%) | Mean NLL | Boundary F1, mean ± SD (%) |
|---|---:|---:|---:|
| Localized | 96.791852 ± 0.596713 | 0.094943787 | 15.537416 ± 1.177652 |
| Tree pooled | 96.786592 ± 0.616093 | 0.094962263 | 18.674979 ± 1.268731 |

Localization gains 0.005260 percentage points and loses 3.137563 boundary F1 points to matched tree pooling. This does not establish useful marker localization or novel architecture. Both continuations improve accuracy over H042, with additional training and readout confounding. SD is across eight validation simulations, with one optimizer seed. Outer data remain untouched.

The actual worker, launcher and root exit was 1 after completed.json was written. The CLI passes its result dictionary into SystemExit. The independent audit decoded the printed dictionary without eval or project imports and found it exactly equal to the written completion. The genuine wait 1 and drained descendants are preserved. There is no clean-exit claim or rerun. Raw probability arrays were not saved and were not independently rehashed after exit. Exact references and cohort metrics are in summary.json.
