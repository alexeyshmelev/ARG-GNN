# H046 continuous foreign-clade flanks

H046 does not meet the preregistered development utility rule. Independent-test confirmation, training-seed variation and useful novelty remain unestablished.

Foreign-flank minus own-state Present182 accuracy: +0.011573 percentage points. Mean boundary F1 difference: +0.149225 points. The rule requires at least +0.10 accuracy points and no more than 1.00 point mean boundary F1 loss, with all eight primary boundary values defined.

Both arms completed 30 FP32 passes and 960 whole-parent updates from identical H042 pass-30 initialization, with independent reset AdamW optimizers (learning rate 0.001, weight decay 0.0001, seed 101), 31,165 parameters each. The original rule selected checkpoint passes 30 (foreign flank) and 30 (own state).

All values below are equal-parent means ± sample SD across eight independent shared VAL simulations. Undefined values are excluded with explicit defined-parent counts. This reused VAL supports development selection. It supplies no independent outer-test result, training-seed SD or significance claim. Raw six-metric values, exact scoring support, paired parent effects and receipt hashes are in [summary.json](summary.json).

Cohorts contain 182 present-day, 504 ancient and 686 total queries, with ancient strata Bronze Age 162, BAA 14, Neo 302 and Yam 26. Scores integrate exact external intervals over the frozen clipped 100 bp bin support. NLL uses natural logs, boundary tolerance is 1,000 bp, Brier sums squared error over four classes, and ECE uses 15 bins with unchanged raw probabilities and no probability floor.

| Cohort | Arm | Accuracy % ± SD | NLL ± SD | Macro F1 % ± SD | Boundary F1 % ± SD | Brier ± SD | ECE ± SD |
|---|---|---:|---:|---:|---:|---:|---:|
| primary_present_day_182 | foreign_flank | 96.789161 ± 0.604163 | 0.094685 ± 0.020530 | 96.626689 ± 0.634091 | 17.484669 ± 1.774105 | 0.048275 ± 0.008912 | 0.005786 ± 0.002550 |
| primary_present_day_182 | own_state | 96.777588 ± 0.635127 | 0.094792 ± 0.021067 | 96.614110 ± 0.654672 | 17.335443 ± 1.677687 | 0.048263 ± 0.009034 | 0.006065 ± 0.002511 |
| secondary_ancient_504 | foreign_flank | 99.493068 ± 0.141692 | 0.015892 ± 0.004914 | 99.343704 ± 0.230307 | 20.244655 ± 2.815959 | 0.008042 ± 0.002241 | 0.001156 ± 0.000648 |
| secondary_ancient_504 | own_state | 99.487503 ± 0.136824 | 0.015964 ± 0.004897 | 99.335310 ± 0.230006 | 20.281995 ± 2.114079 | 0.008096 ± 0.002183 | 0.001288 ± 0.000710 |
| all_queries_686 | foreign_flank | 98.775705 ± 0.253306 | 0.036797 ± 0.008486 | 98.508003 ± 0.328172 | 18.454347 ± 2.082066 | 0.018716 ± 0.003804 | 0.002133 ± 0.001017 |
| all_queries_686 | own_state | 98.768546 ± 0.257227 | 0.036877 ± 0.008586 | 98.498352 ± 0.334475 | 18.364768 ± 1.777981 | 0.018753 ± 0.003787 | 0.002197 ± 0.001112 |
| age_stratum_Bronze_Age | foreign_flank | 99.190194 ± 0.347763 | 0.026938 ± 0.014052 | 99.107582 ± 0.342576 | 19.567722 ± 2.842727 | 0.013129 ± 0.005878 | 0.002523 ± 0.001766 |
| age_stratum_Bronze_Age | own_state | 99.184960 ± 0.343789 | 0.026998 ± 0.014075 | 99.102603 ± 0.339511 | 19.719624 ± 2.926628 | 0.013127 ± 0.005848 | 0.002648 ± 0.001900 |
| age_stratum_BAA | foreign_flank | 99.282178 ± 1.078588 | 0.021040 ± 0.018358 | 98.767495 ± 2.148780 | 8.928571 ± 14.042718 | 0.011788 ± 0.013872 | 0.007675 ± 0.004906 |
| age_stratum_BAA | own_state | 99.261458 ± 1.136997 | 0.021729 ± 0.019443 | 98.732298 ± 2.285545 | 11.503623 ± 19.376196 | 0.012257 ± 0.014699 | 0.008353 ± 0.005573 |
| age_stratum_Neo | foreign_flank | 99.634675 ± 0.097473 | 0.010698 ± 0.002203 | 99.572047 ± 0.111281 | 21.195781 ± 3.926446 | 0.005645 ± 0.001244 | 0.001171 ± 0.000432 |
| age_stratum_Neo | own_state | 99.627830 ± 0.091939 | 0.010786 ± 0.002224 | 99.565331 ± 0.112632 | 21.073975 ± 2.579969 | 0.005734 ± 0.001232 | 0.001246 ± 0.000438 |
| age_stratum_Yam | foreign_flank | 99.848947 ± 0.226797 | 0.004631 ± 0.003686 | 99.915136 ± 0.109490 | 24.436090 ± 26.829434 (n=7) | 0.002170 ± 0.002525 | 0.002204 ± 0.001548 |
| age_stratum_Yam | own_state | 99.864351 ± 0.208192 | 0.004260 ± 0.003416 | 99.922836 ± 0.099576 | 33.068783 ± 39.810200 (n=7) | 0.001942 ± 0.002318 | 0.002078 ± 0.001545 |

Paired differences below are foreign flank minus own state within each parent, then mean ± sample SD. Accuracy/F1 differences use percentage points. Lower NLL/Brier/ECE differences favor foreign flank.

| Cohort | Δ accuracy pp | Δ NLL | Δ macro F1 pp | Δ boundary F1 pp | Δ Brier | Δ ECE |
|---|---:|---:|---:|---:|---:|---:|
| primary_present_day_182 | 0.011573 ± 0.069124 | -0.000106 ± 0.001109 | 0.012579 ± 0.053840 | 0.149225 ± 0.859173 | 0.000013 ± 0.000628 | -0.000279 ± 0.000859 |
| secondary_ancient_504 | 0.005565 ± 0.007210 | -0.000072 ± 0.000144 | 0.008394 ± 0.007567 | -0.037340 ± 1.041423 | -0.000054 ± 0.000107 | -0.000132 ± 0.000290 |
| all_queries_686 | 0.007159 ± 0.017878 | -0.000081 ± 0.000382 | 0.009651 ± 0.021544 | 0.089580 ± 0.662821 | -0.000036 ± 0.000209 | -0.000065 ± 0.000263 |
| age_stratum_Bronze_Age | 0.005234 ± 0.011859 | -0.000060 ± 0.000409 | 0.004979 ± 0.012201 | -0.151902 ± 1.471911 | 0.000002 ± 0.000227 | -0.000125 ± 0.000424 |
| age_stratum_BAA | 0.020720 ± 0.071472 | -0.000689 ± 0.001179 | 0.035198 ± 0.145169 | -2.575052 ± 7.623482 | -0.000469 ± 0.000849 | -0.000678 ± 0.001056 |
| age_stratum_Neo | 0.006846 ± 0.010715 | -0.000088 ± 0.000084 | 0.006716 ± 0.012492 | 0.121806 ± 2.076186 | -0.000089 ± 0.000070 | -0.000075 ± 0.000167 |
| age_stratum_Yam | -0.015404 ± 0.027222 | 0.000371 ± 0.000437 | -0.007701 ± 0.014696 | -8.632693 ± 25.695950 (n=7) | 0.000228 ± 0.000333 | 0.000126 ± 0.000278 |

All 96 prediction products were sealed before VAL truth; all reporting reads metric JSON only. No checkpoint, probability, query-truth payload or scientific algorithm was opened or changed by this aggregator.
