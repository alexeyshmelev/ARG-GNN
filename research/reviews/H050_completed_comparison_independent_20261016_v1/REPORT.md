# H050 independent reporting-arithmetic audit

**PASS.** The four H050 arms in the inspected current_scoreboard.json match the actual completed scored metadata. CPU tool 61885 has a consumed exit 0 with completed_ref and result_ref bound to the same scientific completion. Original TRAIN CLI exit 1 remains distinct.

| Arm | Present182 accuracy (%) ± sample SD | NLL | Boundary F1 (%) ± sample SD | Selected pass | Selected cumulative updates |
|---|---:|---:|---:|---:|---:|
| Triple-support pair memory | 97.0640 ± 0.5595 | 0.088471 | 18.9265 ± 1.4187 | 30 | 3840 |
| Adjacent marginal null | 97.0504 ± 0.5796 | 0.088394 | 18.8498 ± 1.0520 | 30 | 3840 |
| Stable leaf memory | 97.0778 ± 0.5674 | 0.088115 | 18.3807 ± 1.3141 | 30 | 3840 |
| H045 continued training | 97.0000 ± 0.4978 | 0.088541 | 19.0488 ± 1.4275 | 20 | 3520 |

Independently recomputed equal-parent means and sample SDs for primary accuracy, NLL, macro F1, balanced accuracy, Brier, ECE and boundary precision/recall/F1 agree with reporting. SD uses divisor 7 across eight simulations, not training seeds. Accuracy/F1 values and their SDs convert from fractions to percentages. NLL/Brier/ECE remain dimensionless. Paired SD fields below are already in percentage points and are not multiplied twice.

| Triple minus control | Accuracy delta pp ± paired SD | Boundary F1 delta pp ± paired SD |
|---|---:|---:|
| Adjacent marginal null | +0.013679 ± 0.024336 | +0.076613 ± 0.641715 |
| Stable leaf memory | −0.013801 ± 0.038427 | +0.545802 ± 0.751794 |
| H045 continuation | +0.064015 ± 0.145348 | −0.122387 ± 0.784856 |

All stored per-parent paired differences, mean effects and paired SDs match direct scalar arithmetic. Every control comparison fails both the registered primary rule ( +0.10 accuracy pp without boundary loss) and secondary rule ( +1.00 boundary F1 pp with at most 0.05 accuracy pp loss). Stable leaf has higher accuracy than triple support. No useful higher-order superiority or overall top-1 claim follows.

Every arm has all seven correctly mapped cohorts in all eight parent records: Present182, BronzeAge162, BAA14, Neo302, Yam26, Ancient504 and All686. Selected rows exactly match the recorded history at 30/30/30/20. Each arm completed 960 new updates from a 2880-update warmstart, 3840 cumulative per model. Across four arms there were 3840 new optimizer updates, not 15360 new updates. The selected continuation uses 640 new updates, 3520 cumulative. Seed 101 and reset AdamW lineage are retained. CPU evaluation performs zero optimizer updates.

This was one metadata/scalar audit only. No model, scorer, truth/prediction/checkpoint payload, training, test suite, source edit, job, probe or Git action occurred. Only this report was written.

Evidence: current_scoreboard.json snapshot 2026-10-16T08:22:02Z SHA c26f7139e0221400bf7f9fbe35738338a0b8df03593faf898a37a8ed296e9b7d. Actual completed.json SHA 2e3dcb14affee5f87d9e0607745a0039647d0ca55daffa0e276bb5599a4a6763. Root actual_tool_terminal.root_v1.json SHA b842a3d33841332c705d5c61d3c08969de79249788e8caec27d934ad28a266fe. All refer to the H050_CPU_full192_savedfit_failedCLI_root_v1 evaluation.
