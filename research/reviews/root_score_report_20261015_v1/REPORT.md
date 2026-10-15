# Current detailed score extract

Frozen scoreboard: **2026-10-15T19:14:04.592719+00:00**. This report contains all17 selected European setups, all seven cohorts and all11 recorded aggregates. [Full metrics, uncertainty and source pins](metrics.json).

**Highest Present182 development accuracy:** H045 foreign_episode, **96.8270 ± 0.5870%**. Its paired gain over own_state is only0.009568percentagepoint, below the registered0.10pp utility criterion. H046 and H047 also failed their registered practical criteria. These rankings do not validate useful cross-tree exchange or establish novelty.

32 independent TRAIN and eight independent VAL simulations;2Mb;952 input haplotypes (476diploids),686query haplotypes and266proxies. Mean ± sampleSD uses equal-parent aggregation across eight VAL simulations (ddof1), with one optimizer seed/model. This measures simulation variation, not training-seed uncertainty. Native Relate post-size genealogy and common physical-span C4 support apply; our GNN also consumes observed SNPs. Native methods retain their disclosed inputs, objectives, schedules and selection. This is a development pipeline comparison; untouched outer TEST remains unfinished.

## Selected European setups: Present182

Percentage metrics use0–100 units. NLL uses raw probabilities; boundary metrics use1kb tolerance. — is unknown/undefined, never zero.

| Setup | Accuracy % ± SD | Balanced accuracy % ± SD | Macro F1 % ± SD | NLL ± SD | Boundary F1 % ± SD |
|---|---:|---:|---:|---:|---:|
| H045 foreign_episode | 96.8270 ± 0.5870 | 96.6953 ± 0.6366 | 96.6565 ± 0.6006 | 0.092980 ± 0.020423 | 18.1558 ± 1.3790 |
| H045 own_state | 96.8175 ± 0.6142 | 96.6733 ± 0.6613 | 96.6478 ± 0.6397 | 0.092976 ± 0.020230 | 18.0557 ± 1.7761 |
| H044 localized | 96.7919 ± 0.5967 | 96.6487 ± 0.6565 | 96.6298 ± 0.6214 | 0.094944 ± 0.020882 | 15.5374 ± 1.1777 |
| H046 foreign_flank | 96.7892 ± 0.6042 | 96.6361 ± 0.6678 | 96.6267 ± 0.6341 | 0.094685 ± 0.020530 | 17.4847 ± 1.7741 |
| H044 tree_pooled | 96.7866 ± 0.6161 | 96.6319 ± 0.6723 | 96.6307 ± 0.6378 | 0.094962 ± 0.021065 | 18.6750 ± 1.2687 |
| H047 original_risk | 96.7777 ± 0.5970 | 96.6295 ± 0.6565 | 96.6136 ± 0.6237 | 0.094808 ± 0.020849 | 18.2775 ± 1.7493 |
| H046 own_state | 96.7776 ± 0.6351 | 96.6294 ± 0.6829 | 96.6141 ± 0.6547 | 0.094792 ± 0.021067 | 17.3354 ± 1.6777 |
| H047 balanced_risk | 96.7576 ± 0.5287 | 96.6680 ± 0.5795 | 96.5750 ± 0.5481 | 0.096819 ± 0.021357 | 18.5574 ± 1.7800 |
| H042 exact local-clade flags | 96.5778 ± 0.5954 | 96.3904 ± 0.6304 | 96.4206 ± 0.5867 | 0.100656 ± 0.020069 | 18.0451 ± 1.8423 |
| H043 true_partner | 96.5182 ± 0.6201 | 96.3872 ± 0.6529 | 96.3658 ± 0.6024 | 0.101527 ± 0.019742 | 17.4301 ± 1.3301 |
| H043 own_state | 96.5029 ± 0.6006 | 96.3585 ± 0.6101 | 96.3410 ± 0.5752 | 0.101819 ± 0.019255 | 17.1055 ± 0.7889 |
| H040 tensor | 96.2674 ± 0.7493 | 96.1094 ± 0.6684 | 96.1525 ± 0.6781 | 0.109332 ± 0.021073 | 16.1986 ± 2.0270 |
| H040 event_mlp | 96.2007 ± 0.7048 | 96.1310 ± 0.6087 | 96.0644 ± 0.6290 | 0.110014 ± 0.020489 | 16.1312 ± 1.6911 |
| H038 aligned whole-tree PyG | 95.4891 ± 0.8338 | 95.1764 ± 0.8019 | 95.3410 ± 0.7329 | 0.133497 ± 0.026419 | 13.4195 ± 1.4686 |
| H038 zero-concordance control | 94.9252 ± 0.7917 | 94.0982 ± 1.0780 | 94.5950 ± 0.8236 | 0.147232 ± 0.023024 | 11.7904 ± 1.3304 |
| Current native ARGMix | 94.4461 ± 1.1491 | 94.3491 ± 1.3792 | 94.3809 ± 1.2151 | 0.219695 ± 0.023975 | 11.7372 ± 1.7328 |
| Native AncestralPaths derived C4 | 93.7830 ± 0.8385 | 93.7505 ± 0.9393 | 93.2918 ± 0.9790 | ∞ | 7.0281 ± 0.7779 |

| Setup | Boundary precision % ± SD | Boundary recall % ± SD | Brier ± SD | ECE ± SD | Dosage MSE ± SD | Dosage MAE ± SD |
|---|---:|---:|---:|---:|---:|---:|
| H045 foreign_episode | 15.2282 ± 1.5093 | 22.5890 ± 1.6757 | 0.047499 ± 0.008911 | 0.005594 ± 0.003098 | 0.023814 ± 0.004537 | 0.044235 ± 0.006572 |
| H045 own_state | 15.1997 ± 1.9787 | 22.3694 ± 1.8069 | 0.047462 ± 0.008910 | 0.005507 ± 0.002906 | 0.023784 ± 0.004549 | 0.044092 ± 0.006653 |
| H044 localized | 11.1788 ± 1.2041 | 25.8003 ± 1.9669 | 0.048353 ± 0.008942 | 0.005772 ± 0.002712 | 0.024247 ± 0.004535 | 0.044899 ± 0.006772 |
| H046 foreign_flank | 14.0982 ± 1.9278 | 23.2172 ± 1.9800 | 0.048275 ± 0.008912 | 0.005786 ± 0.002550 | 0.024194 ± 0.004533 | 0.044596 ± 0.006670 |
| H044 tree_pooled | 15.4763 ± 1.5766 | 23.7258 ± 1.7036 | 0.048438 ± 0.009157 | 0.005905 ± 0.002475 | 0.024278 ± 0.004640 | 0.044756 ± 0.006889 |
| H047 original_risk | 15.3047 ± 2.0751 | 22.8590 ± 1.6076 | 0.048271 ± 0.008964 | 0.005880 ± 0.002626 | 0.024199 ± 0.004575 | 0.044443 ± 0.006669 |
| H046 own_state | 13.9187 ± 1.9178 | 23.2015 ± 1.6936 | 0.048263 ± 0.009034 | 0.006065 ± 0.002511 | 0.024194 ± 0.004604 | 0.044658 ± 0.006761 |
| H047 balanced_risk | 15.6258 ± 2.0444 | 23.0180 ± 2.0603 | 0.048867 ± 0.008525 | 0.006078 ± 0.003096 | 0.024517 ± 0.004362 | 0.044292 ± 0.006251 |
| H042 exact local-clade flags | 15.2079 ± 2.0112 | 22.2952 ± 1.8578 | 0.051586 ± 0.009430 | 0.005352 ± 0.002373 | 0.025815 ± 0.004821 | 0.048461 ± 0.007077 |
| H043 true_partner | 14.5304 ± 1.3157 | 21.8518 ± 1.7550 | 0.052353 ± 0.009688 | 0.005803 ± 0.001879 | 0.026207 ± 0.004922 | 0.049220 ± 0.007125 |
| H043 own_state | 14.0782 ± 0.6934 | 21.8472 ± 1.5003 | 0.052597 ± 0.009457 | 0.005423 ± 0.002363 | 0.026327 ± 0.004802 | 0.049157 ± 0.006983 |
| H040 tensor | 13.3841 ± 1.8866 | 20.5928 ± 2.2946 | 0.056491 ± 0.010757 | 0.006276 ± 0.002345 | 0.028302 ± 0.005547 | 0.054001 ± 0.008337 |
| H040 event_mlp | 13.3733 ± 1.6927 | 20.4051 ± 1.8228 | 0.056967 ± 0.010707 | 0.005749 ± 0.002033 | 0.028568 ± 0.005502 | 0.054639 ± 0.008211 |
| H038 aligned whole-tree PyG | 10.7125 ± 1.2053 | 17.9867 ± 2.0061 | 0.067902 ± 0.012019 | 0.006785 ± 0.002859 | 0.034070 ± 0.006375 | 0.064029 ± 0.009222 |
| H038 zero-concordance control | 8.9251 ± 1.0426 | 17.3930 ± 1.9375 | 0.075842 ± 0.011726 | 0.007967 ± 0.003782 | 0.037983 ± 0.005983 | 0.069174 ± 0.008361 |
| Current native ARGMix | 9.1614 ± 1.4876 | 16.4073 ± 2.2223 | 0.092752 ± 0.015147 | 0.078235 ± 0.006887 | 0.046605 ± 0.007650 | 0.128048 ± 0.010330 |
| Native AncestralPaths derived C4 | 3.9474 ± 0.4790 | 32.3565 ± 2.0570 | 0.098592 ± 0.012700 | 0.010487 ± 0.002875 | 0.049766 ± 0.006758 | 0.085603 ± 0.009511 |

## Training and selection

| Setup | Run budget | Completed cumulative lineage updates | Selected pass / lineage updates |
|---|---|---:|---:|
| H045 foreign_episode | H0421920 +960updates; 30extra passes | 2880 | 30 / 2880 |
| H045 own_state | H0421920 +960updates; 30extra passes | 2880 | 30 / 2880 |
| H044 localized | H0421920 +960updates; 30extra passes | 2880 | 30 / 2880 |
| H046 foreign_flank | H0421920 +960updates; 30extra passes | 2880 | 30 / 2880 |
| H044 tree_pooled | H0421920 +960updates; 30extra passes | 2880 | 30 / 2880 |
| H047 original_risk | H0421920 +960updates; 30extra passes | 2880 | 30 / 2880 |
| H046 own_state | H0421920 +960updates; 30extra passes | 2880 | 30 / 2880 |
| H047 balanced_risk | H0421920 +960updates; 30extra passes | 2880 | 30 / 2880 |
| H042 exact local-clade flags | H038960 +960updates; 30extra passes | 1920 | 30 / 1920 |
| H043 true_partner | H0421920 +960updates; 30extra passes | 2880 | 30 / 2880 |
| H043 own_state | H0421920 +960updates; 30extra passes | 2880 | 30 / 2880 |
| H040 tensor | H038960 +960updates; 30extra passes | 1920 | 30 / 1920 |
| H040 event_mlp | H038960 +960updates; 30extra passes | 1920 | 30 / 1920 |
| H038 aligned whole-tree PyG | fresh30pass/960updates; selectedpass30 | 960 | 30 / 960 |
| H038 zero-concordance control | fresh30pass/960updates; selectedpass25 | 960 | 25 / 800 |
| Current native ARGMix | 11epochs; native earlystop,selectedepoch9; seed50 | — | — |
| Native AncestralPaths derived C4 | 5epochs,batch32,5,190,874rows; finalepoch5; seed101 | — | — |

Our models use seed101. H038 zero-concordance completed30passes/960updates but selected pass25/800updates. H040/H042 add960updates to H038; H043–H047 add960updates to H042, yielding2880selected lineage updates. Native ARGMix selected epoch9 of11 (seed50); native AP used five epochs/batch32/5,190,874sampled rows and final epoch5 (seed101). Update counts for these native baselines are unknown in the source.

AncestralPaths six-path probabilities were collapsed to C4 before argmax; native six-path SNP-focal results are distinct. Strict raw FP32 AP NLL is∞ because one VAL parent has zero true-class probability over800query-bp. No probability floor, posthoc renormalization or recalibration was applied. H044 metric artifacts were independently verified while its genuine post-completion CLI exit1 remains recorded.

## All seven cohort accuracies

All entries are mean ± sampleSD%, across eight VAL simulations. BronzeAge162 is only an approximate Post-Neolithic correspondence and is not asserted to be the identical paper cohort.

| Setup | Present182 | BronzeAge162 | BAA14 | Neo302 | Yam26 | Ancient504 | All686 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H045 foreign_episode | 96.8270 ± 0.5870 | 99.2047 ± 0.3199 | 99.1511 ± 1.2545 | 99.6063 ± 0.0850 | 99.9039 ± 0.1292 | 99.4799 ± 0.1347 | 98.7761 ± 0.2486 |
| H045 own_state | 96.8175 ± 0.6142 | 99.1658 ± 0.3983 | 99.2765 ± 1.1385 | 99.6424 ± 0.0859 | 99.8687 ± 0.1833 | 99.4907 ± 0.1501 | 98.7815 ± 0.2566 |
| H044 localized | 96.7919 ± 0.5967 | 99.1895 ± 0.3391 | 99.2328 ± 1.1873 | 99.6288 ± 0.0928 | 99.8685 ± 0.2092 | 99.4889 ± 0.1358 | 98.7734 ± 0.2475 |
| H046 foreign_flank | 96.7892 ± 0.6042 | 99.1902 ± 0.3478 | 99.2822 ± 1.0786 | 99.6347 ± 0.0975 | 99.8489 ± 0.2268 | 99.4931 ± 0.1417 | 98.7757 ± 0.2533 |
| H044 tree_pooled | 96.7866 ± 0.6161 | 99.1900 ± 0.3386 | 99.2662 ± 1.0684 | 99.6324 ± 0.0993 | 99.8593 ± 0.2264 | 99.4917 ± 0.1384 | 98.7740 ± 0.2550 |
| H047 original_risk | 96.7777 ± 0.5970 | 99.1860 ± 0.3514 | 99.2947 ± 1.0733 | 99.6369 ± 0.0956 | 99.8433 ± 0.2268 | 99.4931 ± 0.1426 | 98.7727 ± 0.2516 |
| H046 own_state | 96.7776 ± 0.6351 | 99.1850 ± 0.3438 | 99.2615 ± 1.1370 | 99.6278 ± 0.0919 | 99.8644 ± 0.2082 | 99.4875 ± 0.1368 | 98.7685 ± 0.2572 |
| H047 balanced_risk | 96.7576 ± 0.5287 | 99.1282 ± 0.3234 | 99.3747 ± 0.6939 | 99.5660 ± 0.0872 | 99.8167 ± 0.2111 | 99.4329 ± 0.1213 | 98.7231 ± 0.2210 |
| H042 exact local-clade flags | 96.5778 ± 0.5954 | 99.0748 ± 0.3242 | 99.0226 ± 1.0723 | 99.5291 ± 0.1771 | 99.8569 ± 0.1871 | 99.3859 ± 0.1560 | 98.6409 ± 0.2641 |
| H043 true_partner | 96.5182 ± 0.6201 | 98.9949 ± 0.3463 | 99.1731 ± 1.0093 | 99.5198 ± 0.1912 | 99.8251 ± 0.2007 | 99.3572 ± 0.1652 | 98.6040 ± 0.2688 |
| H043 own_state | 96.5029 ± 0.6006 | 99.0107 ± 0.3257 | 99.1048 ± 0.9583 | 99.5217 ± 0.1935 | 99.8547 ± 0.1785 | 99.3630 ± 0.1737 | 98.6042 ± 0.2691 |
| H040 tensor | 96.2674 ± 0.7493 | 98.8266 ± 0.4696 | 99.1439 ± 0.8901 | 99.5200 ± 0.1579 | 99.7215 ± 0.3045 | 99.2971 ± 0.2073 | 98.4933 ± 0.3390 |
| H040 event_mlp | 96.2007 ± 0.7048 | 98.8238 ± 0.4377 | 99.1147 ± 0.9216 | 99.5576 ± 0.1232 | 99.7569 ± 0.2389 | 99.3197 ± 0.1707 | 98.4922 ± 0.2948 |
| H038 aligned whole-tree PyG | 95.4891 ± 0.8338 | 98.1726 ± 0.8000 | 98.9020 ± 0.8854 | 99.4103 ± 0.1271 | 98.9598 ± 1.3143 | 98.9751 ± 0.3226 | 98.0503 ± 0.4312 |
| H038 zero-concordance control | 94.9252 ± 0.7917 | 98.4120 ± 0.5405 | 98.6137 ± 0.9470 | 99.1716 ± 0.2703 | 99.4215 ± 0.8073 | 98.9248 ± 0.2837 | 97.8637 ± 0.3727 |
| Current native ARGMix | 94.4461 ± 1.1491 | 97.7051 ± 0.7466 | 98.4370 ± 1.0186 | 99.2955 ± 0.2623 | 99.2218 ± 0.8966 | 98.7566 ± 0.3114 | 97.6130 ± 0.4975 |
| Native AncestralPaths derived C4 | 93.7830 ± 0.8385 | 94.7589 ± 1.1847 | 96.9352 ± 1.3195 | 97.5693 ± 0.2748 | 95.4303 ± 1.4096 | 96.5380 ± 0.3194 | 95.8071 ± 0.4473 |

The machine JSON preserves every metric for all119setup/cohort rows, including defined/undefined parent counts. Some Yam boundary metrics are defined for only seven simulations.

## Native competitors: Present182 ancestry classes

P/R/F1 use physical-span support and report mean ± sampleSD% over the same eight VAL simulations. [All seven native cohort/class tables](../../../research/reviews/native_common_cohort_diagnostics_20261015_v1/native_cohort_metrics_mean_SD.json) are retained in metrics.json.

| Native method | Class | Precision % ± SD | Recall % ± SD | F1 % ± SD |
|---|---|---:|---:|---:|
| ARGMix | Ana | 94.8454 ± 2.4102 | 94.0873 ± 1.5264 | 94.4383 ± 1.1392 |
| ARGMix | CHG | 92.1212 ± 2.4891 | 93.4994 ± 3.3211 | 92.7603 ± 2.0086 |
| ARGMix | WHG | 94.6941 ± 1.3465 | 93.4064 ± 2.4789 | 94.0370 ± 1.7696 |
| ARGMix | EHG | 96.1799 ± 1.1434 | 96.4034 ± 0.8998 | 96.2882 ± 0.8288 |
| AncestralPaths derived C4 | Ana | 94.3699 ± 1.0118 | 95.8304 ± 0.6764 | 95.0898 ± 0.4873 |
| AncestralPaths derived C4 | CHG | 93.9713 ± 1.2497 | 90.7080 ± 1.8070 | 92.3057 ± 1.3899 |
| AncestralPaths derived C4 | WHG | 87.8806 ± 2.2200 | 94.7423 ± 1.9590 | 91.1681 ± 1.7110 |
| AncestralPaths derived C4 | EHG | 95.5134 ± 1.2598 | 93.7213 ± 1.4371 | 94.6034 ± 1.1128 |

## Separate adapted ARGformer panel

European adapted pilot: two VAL simulations,280observed haplotypes,2Mb,tsinfer/tsdate trees;64present-day plus56ancient query haplotypes. Separate transductive encoder fit within each observed ARG: five MLM epochs plus one pure-reference contrastive epoch, without query ancestry. Scores condition on retained-center clipped-bin support and exclude uncovered tails. These scores are not directly comparable to Present182/All686 or the original American benchmark. SD is across two VAL simulations.

| Adapted setup | Present64 accuracy % ± SD | Ancient56 accuracy % ± SD | All120 accuracy % ± SD |
|---|---:|---:|---:|
| Adapted ARGformer top1 | 75.6818 ± 4.5195 | 75.4447 ± 3.5081 | 75.5711 ± 4.0476 |
| Adapted ARGformer top5 | 75.1358 ± 3.7500 | 74.7885 ± 4.0220 | 74.9737 ± 3.8769 |

Primary top1 and prespecified top5 use same-ARG pure-reference exhaustive retrieval and raw hard votes. Raw NLL is∞; no probability floor or fitted calibration. [Authoritative summary](../../../experiments/results/native_argformer_tsinfer_fit_v7_summary_v2/summary.json).

## GST molecular graph-set pilot

Metric: pooled per-atom sigmoid **average precision**, not ancestry accuracy. Actual scored condition: retained TRAIN1000/VALID1495,seed0,four fits;AdamW lr0.001/weight_decay0.0001,batch16,20epochs/1260updates per fit. One seed provides no training-seed SD.

| Native setup | VALID AP | Selected epoch | Parameters | Hidden width |
|---|---:|---:|---:|---:|
| gcn-ds | 0.63945129 | 17 | 134913 | 128 |
| gcn-st | 0.63823432 | 16 | 267777 | 128 |
| gsc-bc | 0.65378329 | 19 | 130622 | 84 |
| gsc-ca | 0.63791939 | 19 | 130798 | 76 |

Full TRAIN parsing retained10,439/10,500reactions; this is data readiness. Full-data multi-seed training is inactive/unfinished, our molecular adapter and native paired TEST comparison are unfinished, and TEST is withheld. Saved five-seed/20-fit arguments are planned suite settings, not the actual four seed0 fits above. Unordered reactant graph sets have no native chain order, so whole-H048 transfer is unvalidated.

## American C3, pending runs and empirical scope

These operational states are frozen at the scoreboard timestamp and may advance independently of the measured score tables.

- Original American C3 VAL000: full30Mb simulation, inferred observation and native10k-focal/path preparation completed, common R600/Q1000 and228477sites. FLARE conversion completed; native prediction/seal/common-truth scores remain unreported in this frozen snapshot. No C3 ancestry score exists for native ARGformer, FLARE or our GNN.
- Native ARGformer GPU5 MLM/count work shown in current status is **engineering TRAIN000**. The console had entered epoch3/5; five epochs/224775planned updates and final saved-Adam verification remain unfinished. TRAIN000 counts completed. These are not held-out ancestry results. VAL000 has a distinct source-ready native count route; it must be reported separately.
- Native comparison requires per-target vocabulary/MLM/reference-contrastive fitting; raw node-ID weights do not transfer between ARGs. The Table1 audit recovered reference-fitted PCA10 but not the exact clustering/label-mapping rule. Current-release top1 retrieval is a separate disclosed comparison; PCA10/KMeans3 is an inactive reconstruction proposal.
- Our C3 route uses20short2Mb TRAIN parents,30epochs/600updates; full30Mb VAL scoring is pending. The original five-VAL × six-checkpoint gate requires all30prediction seals before truth; no GNN VAL optimization or TEST access.
- H048 training completed30passes/960updates per arm with actual wait0; registered evaluation launched and accuracy is pending. It is excluded from the17measured rows.
- H049 is a registered three-arm32,446-parameter candidate with30passes/960updates per arm planned; first CPU preparation failed and repair is pending. No fit/accuracy evidence.
- H045 empirical probabilities, bp-weighted proportions and genotype PCA completed for24query haplotypes/12subjects. These are descriptive results without known ancestry truth, so no empirical ancestry accuracy is available.

## Focused README-generator findings

The inspected generator already reports all17setups, seven cohorts and all11aggregate metrics. Its current TRAIN000 narrative repair is acknowledged. Suggested focused follow-ups:

- **route_discovery**: Remaining VAL001-004 still uses the old packet route for every parent. Keep001 old route and discover002-004 in American_C3_remaining_VAL_namespace_memo_root_20261015_v1. Native count reporting points only to engineering TRAIN000; add the distinct VAL000 counts route. FLARE top table exposes VAL000 conversion but omits native/seal stage status. Source lines: 299, 304, 395, 402, 598, 599, 699.
- **explicit_scientific_role**: Engineering TRAIN000 narrative is now present in native_status and native_brief. Native MLM/count JSON objects should also carry explicit parent_id, split and scientific_role; counts_execution currently has none. Do not treat TRAIN000 MLM/count completion as held-out ancestry evidence. Source lines: 383, 402, 691, 692.
- **selected_training_lineage**: Render selected pass/update count and cumulative lineage from training_budget_lineage. H038 zero-concordance completed30passes/960updates but selected pass25/800updates; H043-H047 selected lineage2880 includes warmstarts. Source lines: 612, 616.
- **H049_progress_zero**: Replace hardcoded actual_optimizer_updates=0 with a verified receipt count or null/unknown before training starts; zero must not survive into a running fit. Source lines: 578.
- **native_class_metrics_and_notes**: All17x7x11 aggregate coverage is complete. Native class precision/recall/F1 is available in native_cohort_metrics_mean_SD.json; render or link a compact class table. Split shared native notes so ARGMix is not described as using AP six-path collapse. Source lines: 165, 623, 637.
- **dynamic_score_headline**: Derive headline score strings from frozen model rows when touching this section; H045/H047 headline values remain literal source strings. Source lines: 703.

No shared README/reporting scripts were changed. No reporting script, training, prediction, truth-array read, process probe, guard/accounting action or Git operation was performed for this extract. All17source/raw receipt pins matched; all119cohort rows match the normalized audit exactly.

## Frozen provenance

- Scoreboard: research/current_scoreboard.json; SHA256 5661fe8dbd0b8989d5d7858ef20f8469d38904b6b549186dd1ff679a2b751e2f.
- Normalized audit: research/reviews/current_score_reporting_audit_20261015_v1/all17_allcohort_metrics.actual.json; SHA256 c874376b2820239c61d557fa72cc1c2275868e504c0c6963736bbde402cfca61.
- Native diagnostics: research/reviews/native_common_cohort_diagnostics_20261015_v1/native_cohort_metrics_mean_SD.json; SHA256 d4556b358155505ca12ac58b8aca4e8d56904564fe0c24a7e35a7483f03a4d9e.
- README generator inspected: scripts/update_current_scoreboard.py; SHA256 be7d035a5e1c8d090bf7e3752a693da4cc16d7bf9ecdf340c78f4fd832601e4b.
- Full source receipt pins and remaining provenance: [metrics.json](metrics.json).
