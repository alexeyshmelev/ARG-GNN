# Completed European C4 cohort scores

The same original eight internal VAL parents, nominal2Mb,952input haplotypes and686queries. Present182 is primary. Native method inputs, objectives and selection/training budgets differ. These are descriptive development scores.

Accuracy is percent mean +/- sample SD in percentage points. SD uses ddof1 across eight independent simulation parents, one completed fit per method, not optimizer-seed SD, SE or a confidence interval.

| Cohort | ARGMix | AP derived C4 | H045 foreign | H047 original | H047 balanced |
|---|---:|---:|---:|---:|---:|
| Present182 | 94.4461 +/- 1.1491 | 93.7830 +/- 0.8385 | 96.8270 +/- 0.5870 | 96.7777 +/- 0.5970 | 96.7576 +/- 0.5287 |
| BronzeAge162 | 97.7051 +/- 0.7466 | 94.7589 +/- 1.1847 | 99.2047 +/- 0.3199 | 99.1860 +/- 0.3514 | 99.1282 +/- 0.3234 |
| BAA14 | 98.4370 +/- 1.0186 | 96.9352 +/- 1.3195 | 99.1511 +/- 1.2545 | 99.2947 +/- 1.0733 | 99.3747 +/- 0.6939 |
| Neo302 | 99.2955 +/- 0.2623 | 97.5693 +/- 0.2748 | 99.6063 +/- 0.0850 | 99.6369 +/- 0.0956 | 99.5660 +/- 0.0872 |
| Yam26 | 99.2218 +/- 0.8966 | 95.4303 +/- 1.4096 | 99.9039 +/- 0.1292 | 99.8433 +/- 0.2268 | 99.8167 +/- 0.2111 |
| Ancient504 | 98.7566 +/- 0.3114 | 96.5380 +/- 0.3194 | 99.4799 +/- 0.1347 | 99.4931 +/- 0.1426 | 99.4329 +/- 0.1213 |
| All686 | 97.6130 +/- 0.4975 | 95.8071 +/- 0.4473 | 98.7761 +/- 0.2486 | 98.7727 +/- 0.2516 | 98.7231 +/- 0.2210 |

Native ARGMix:11epochs, native early stopping. AncestralPaths:5epochs, native fixed schedule, recorded six-path probabilities collapsed to C4. H045foreign_episode and both H047 arms: H042warmstart +30passes/960updates per arm, seed101. H045 is the current selected development fit, not an independently confirmed best architecture.

## Other common recorded metrics

Rates below are percent with SD in percentage points. NLL, Brier and ECE retain original source units. n below8 identifies undefined-parent exclusions in the original aggregate. Native AP NLL is marked infinite when recorded infinite_nll_query_bp is positive in any parent; its sample SD is undefined. A null published aggregate without such an infinite-NLL witness stays unavailable.

| Cohort | Setup | NLL +/- SD | Macro F1 % +/- SD | Boundary F1 % +/- SD | Brier +/- SD | ECE +/- SD |
|---|---|---:|---:|---:|---:|---:|
| Present182 | ARGMix | 0.219695 +/- 0.023975 | 94.3809 +/- 1.2151 | 11.7372 +/- 1.7328 | 0.092752 +/- 0.015147 | 0.078235 +/- 0.006887 |
| Present182 | AP derived C4 | infinite; SD undefined (1/8 parents) | 93.2918 +/- 0.9790 | 7.0281 +/- 0.7779 | 0.098592 +/- 0.012700 | 0.010487 +/- 0.002875 |
| Present182 | H045 foreign | 0.092980 +/- 0.020423 | 96.6565 +/- 0.6006 | 18.1558 +/- 1.3790 | 0.047499 +/- 0.008911 | 0.005594 +/- 0.003098 |
| Present182 | H047 original | 0.094808 +/- 0.020849 | 96.6136 +/- 0.6237 | 18.2775 +/- 1.7493 | 0.048271 +/- 0.008964 | 0.005880 +/- 0.002626 |
| Present182 | H047 balanced | 0.096819 +/- 0.021357 | 96.5750 +/- 0.5481 | 18.5574 +/- 1.7800 | 0.048867 +/- 0.008525 | 0.006078 +/- 0.003096 |
| BronzeAge162 | ARGMix | 0.137894 +/- 0.018029 | 97.5701 +/- 0.7104 | 11.3857 +/- 2.6200 | 0.044121 +/- 0.010198 | 0.078360 +/- 0.006257 |
| BronzeAge162 | AP derived C4 | infinite; SD undefined (1/8 parents) | 94.1623 +/- 1.1755 | 3.0986 +/- 0.9384 | 0.082388 +/- 0.018556 | 0.007534 +/- 0.005079 |
| BronzeAge162 | H045 foreign | 0.026452 +/- 0.012930 | 99.1203 +/- 0.3202 | 20.7714 +/- 4.0220 | 0.012811 +/- 0.005395 | 0.002614 +/- 0.001679 |
| BronzeAge162 | H047 original | 0.027015 +/- 0.014085 | 99.1042 +/- 0.3453 | 20.8667 +/- 3.0520 | 0.013182 +/- 0.005900 | 0.002489 +/- 0.001713 |
| BronzeAge162 | H047 balanced | 0.026923 +/- 0.010640 | 99.0392 +/- 0.3130 | 20.0447 +/- 2.5542 | 0.013445 +/- 0.004809 | 0.002919 +/- 0.001277 |
| BAA14 | ARGMix | 0.134126 +/- 0.022226 | 97.3018 +/- 2.2022 | 0.0000 +/- 0.0000 | 0.039004 +/- 0.013836 | 0.088624 +/- 0.010676 |
| BAA14 | AP derived C4 | 0.101712 +/- 0.043356 | 93.7895 +/- 7.4457 | 1.0897 +/- 1.1980 | 0.048373 +/- 0.021433 | 0.008653 +/- 0.003687 |
| BAA14 | H045 foreign | 0.026361 +/- 0.026762 | 98.3460 +/- 2.5829 | 12.6786 +/- 18.9947 | 0.014188 +/- 0.017672 | 0.009820 +/- 0.008154 |
| BAA14 | H047 original | 0.020258 +/- 0.017187 | 98.7889 +/- 2.1329 | 9.8039 +/- 15.0793 | 0.011275 +/- 0.013061 | 0.007279 +/- 0.004510 |
| BAA14 | H047 balanced | 0.019795 +/- 0.014973 | 98.6400 +/- 1.7417 | 17.8906 +/- 22.6927 | 0.010744 +/- 0.010410 | 0.007543 +/- 0.004383 |
| Neo302 | ARGMix | 0.102450 +/- 0.005403 | 99.2418 +/- 0.2387 | 16.7763 +/- 3.9630 | 0.021148 +/- 0.003219 | 0.082591 +/- 0.002144 |
| Neo302 | AP derived C4 | infinite; SD undefined (1/8 parents) | 97.1480 +/- 0.4419 | 3.4446 +/- 0.7146 | 0.038234 +/- 0.004584 | 0.008276 +/- 0.001975 |
| Neo302 | H045 foreign | 0.011423 +/- 0.002659 | 99.5567 +/- 0.1048 | 23.4368 +/- 3.4480 | 0.006114 +/- 0.001455 | 0.001138 +/- 0.000580 |
| Neo302 | H047 original | 0.010674 +/- 0.002146 | 99.5742 +/- 0.1094 | 23.5846 +/- 3.3946 | 0.005629 +/- 0.001204 | 0.001162 +/- 0.000400 |
| Neo302 | H047 balanced | 0.012386 +/- 0.002046 | 99.5398 +/- 0.0838 | 23.5915 +/- 3.4118 | 0.006610 +/- 0.001254 | 0.001086 +/- 0.000463 |
| Yam26 | ARGMix | 0.099234 +/- 0.022864 | 99.6206 +/- 0.4101 | unavailable (n=7/8 defined) | 0.023563 +/- 0.012912 | 0.079989 +/- 0.011454 |
| Yam26 | AP derived C4 | infinite; SD undefined (2/8 parents) | 95.4225 +/- 1.4282 | unavailable (n=7/8 defined) | 0.071040 +/- 0.022790 | 0.011779 +/- 0.009506 |
| Yam26 | H045 foreign | 0.003531 +/- 0.002741 | 99.9407 +/- 0.0645 | 30.1587 +/- 40.9327 (n=7) | 0.001472 +/- 0.001773 | 0.001424 +/- 0.000810 |
| Yam26 | H047 original | 0.004814 +/- 0.003852 | 99.9122 +/- 0.1094 | 23.2456 +/- 26.5523 (n=7) | 0.002287 +/- 0.002632 | 0.002206 +/- 0.001528 |
| Yam26 | H047 balanced | 0.005930 +/- 0.004251 | 99.8929 +/- 0.1055 | 22.8852 +/- 26.5580 (n=7) | 0.002807 +/- 0.002799 | 0.002404 +/- 0.001411 |
| Ancient504 | ARGMix | 0.114557 +/- 0.008198 | 98.3958 +/- 0.4842 | 13.8259 +/- 2.5466 | 0.029153 +/- 0.004557 | 0.081095 +/- 0.002458 |
| Ancient504 | AP derived C4 | infinite; SD undefined (3/8 parents) | 95.3004 +/- 0.7410 | 2.9699 +/- 0.2703 | 0.054400 +/- 0.004876 | 0.004829 +/- 0.001563 |
| Ancient504 | H045 foreign | 0.016261 +/- 0.004808 | 99.3242 +/- 0.2277 | 21.9236 +/- 2.6705 | 0.008252 +/- 0.002199 | 0.001384 +/- 0.000734 |
| Ancient504 | H047 original | 0.015890 +/- 0.004889 | 99.3434 +/- 0.2304 | 22.1980 +/- 2.7222 | 0.008041 +/- 0.002225 | 0.001133 +/- 0.000651 |
| Ancient504 | H047 balanced | 0.016932 +/- 0.003598 | 99.2656 +/- 0.2057 | 21.5887 +/- 2.2840 | 0.008725 +/- 0.001832 | 0.001013 +/- 0.000422 |
| All686 | ARGMix | 0.142450 +/- 0.011556 | 97.1989 +/- 0.6197 | 12.5034 +/- 1.4764 | 0.046026 +/- 0.006914 | 0.080312 +/- 0.002686 |
| All686 | AP derived C4 | infinite; SD undefined (4/8 parents) | 94.8565 +/- 0.6681 | 4.6069 +/- 0.3380 | 0.066125 +/- 0.006794 | 0.004355 +/- 0.000956 |
| All686 | H045 foreign | 0.036615 +/- 0.008183 | 98.4993 +/- 0.3292 | 19.4440 +/- 1.5183 | 0.018664 +/- 0.003752 | 0.002101 +/- 0.001225 |
| All686 | H047 original | 0.036828 +/- 0.008563 | 98.5053 +/- 0.3240 | 19.6304 +/- 2.0264 | 0.018714 +/- 0.003814 | 0.002175 +/- 0.001060 |
| All686 | H047 balanced | 0.038126 +/- 0.007804 | 98.4504 +/- 0.2848 | 19.6316 +/- 1.8046 | 0.019375 +/- 0.003496 | 0.002190 +/- 0.000960 |

BronzeAge is not assumed identical to the paper's Post-Neolithic cohort. No significance, causal demographic mechanism, architecture attribution or original-paper reproduction claim is made. All common metrics and exact source hashes are in cohort_scores.json. Confusion figures use per-parent row normalization before the mean, never pooled-row normalization.

## Provenance

- research/reviews/native_common_cohort_diagnostics_20261015_v1/native_cohort_metrics_mean_SD.json; SHA256 d4556b358155505ca12ac58b8aca4e8d56904564fe0c24a7e35a7483f03a4d9e
- research/reviews/native_common_cohort_diagnostics_20261015_v1/per_parent_metrics.json; SHA256 f221906384c10e05803d06d0eaee8c2c40936bbb9c3063f410ad4d53501aebde
- research/reviews/native_common_cohort_diagnostics_20261015_v1/completed.json; SHA256 4dfd606507d50d7018795ba8d3108e95911d4fc2b33a0227adaf9e1c568e0e0a
- research/reviews/H045_completed_comparison_root_20261015_v1/summary.json; SHA256 d845b864d8ab0c9632e1b55f55b9db9fac172cd007457d6c37a06d778058f6c1
- experiments/results/H045_GPU5_episode_evaluate_seed101_v1/completed.json; SHA256 1f98815c0791120f512d7d472db39487a3839be96f7a7ff975a57326b3b1f14b
- research/reviews/H047_completed_comparison_root_20261015_v1/summary.json; SHA256 4a75df165b20e491b5d60b09ac0f7eba065562be3813cccaea2e0e8e2d33477d
