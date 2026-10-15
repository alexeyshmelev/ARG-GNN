# Current scientific scoreboard

Updated UTC: 2026-10-15T15:51:35.867152+00:00

[All measured cohort details](#detailed-cohort-metrics) · [Complete historical receipt inventory](all_recorded_metric_receipts.md) · [Fresh H048 novelty/cross-domain audit](../research/reviews/H048_novelty_crossdomain_fresh_20261015_v1/report.md)

## European C4 shared development

32 TRAIN and eight VAL simulations, 2 Mb, 952 input haplotypes and 686 query haplotypes. Scores use common physical-span support. SD is across eight VAL simulations, with one optimizer seed. This is development evidence. Native method input representations, objectives and training schedules are preserved and disclosed. Architecture-only attribution and original-paper reproduction are unproven.

| Setup | Present accuracy % ± SD | Present NLL | Present boundary F1 % ± SD | Training |
|---|---:|---:|---:|---|
| H045 foreign_episode | 96.8270 ± 0.5870 | 0.092980 ± 0.020423 | 18.1558 ± 1.3790 | H042 warmstart +30 passes /960 updates per arm, seed101 |
| H045 own_state | 96.8175 ± 0.6142 | 0.092976 ± 0.020230 | 18.0557 ± 1.7761 | H042 warmstart +30 passes /960 updates per arm, seed101 |
| H044 localized | 96.7919 ± 0.5967 | 0.094944 | 15.5374 ± 1.1777 | H042 warmstart +30 passes /960 updates per arm, seed101 |
| H044 tree_pooled | 96.7866 ± 0.6161 | 0.094962 | 18.6750 ± 1.2687 | H042 warmstart +30 passes /960 updates per arm, seed101 |
| H047 original_risk | 96.7777 ± 0.5970 | 0.094808 ± 0.020849 | 18.2775 ± 1.7493 | H042 warmstart +30 passes /960 updates per arm, seed101 |
| H047 balanced_risk | 96.7576 ± 0.5287 | 0.096819 ± 0.021357 | 18.5574 ± 1.7800 | H042 warmstart +30 passes /960 updates per arm, seed101 |
| H042 exact local-clade flags | 96.5778 ± 0.5954 | 0.100656 ± 0.020069 | 18.0451 ± 1.8423 | H038 warmstart +30 passes /960 updates, seed101 |
| H043 true_partner | 96.5182 ± 0.6201 | 0.101527 ± 0.019742 | 17.4301 ± 1.3301 | H042 warmstart +30 passes /960 updates per arm, seed101 |
| H043 own_state | 96.5029 ± 0.6006 | 0.101819 ± 0.019255 | 17.1055 ± 0.7889 | H042 warmstart +30 passes /960 updates per arm, seed101 |
| H040 tensor | 96.2674 ± 0.7493 | 0.109332 | — | H038 warmstart +30 passes /960 updates, seed101 |
| H040 event_mlp | 96.2007 ± 0.7048 | 0.110014 | — | H038 warmstart +30 passes /960 updates, seed101 |
| H038 aligned whole-tree PyG | 95.4891 ± 0.8338 | 0.133497 ± 0.026419 | 13.4195 ± 1.4686 | 30 passes / 960 updates, seed101 |
| H038 zero-concordance control | 94.9252 ± 0.7917 | 0.147232 ± 0.023024 | 11.7904 ± 1.3304 | 30 passes / 960 updates, seed101 |
| Current native ARGMix | 94.4461 ± 1.1491 | 0.219695 ± 0.023975 | 11.7372 ± 1.7328 | 11 epochs, native early stopping |
| Native AncestralPaths derived C4 | 93.7830 ± 0.8385 | ∞ | 7.0281 ± 0.7779 | 5 epochs, native fixed schedule |

## All measured cohort accuracies

| Setup | Present182 | BronzeAge162 | BAA14 | Neo302 | Yam26 | Ancient504 | All686 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H045 foreign_episode | 96.8270 ± 0.5870 | 99.2047 ± 0.3199 | 99.1511 ± 1.2545 | 99.6063 ± 0.0850 | 99.9039 ± 0.1292 | 99.4799 ± 0.1347 | 98.7761 ± 0.2486 |
| H045 own_state | 96.8175 ± 0.6142 | 99.1658 ± 0.3983 | 99.2765 ± 1.1385 | 99.6424 ± 0.0859 | 99.8687 ± 0.1833 | 99.4907 ± 0.1501 | 98.7815 ± 0.2566 |
| H044 localized | 96.7919 ± 0.5967 | 99.1895 ± 0.3391 | 99.2328 ± 1.1873 | 99.6288 ± 0.0928 | 99.8685 ± 0.2092 | 99.4889 ± 0.1358 | 98.7734 ± 0.2475 |
| H044 tree_pooled | 96.7866 ± 0.6161 | 99.1900 ± 0.3386 | 99.2662 ± 1.0684 | 99.6324 ± 0.0993 | 99.8593 ± 0.2264 | 99.4917 ± 0.1384 | 98.7740 ± 0.2550 |
| H047 original_risk | 96.7777 ± 0.5970 | 99.1860 ± 0.3514 | 99.2947 ± 1.0733 | 99.6369 ± 0.0956 | 99.8433 ± 0.2268 | 99.4931 ± 0.1426 | 98.7727 ± 0.2516 |
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

BronzeAge is not asserted identical to the paper's Post-Neolithic cohort. A dash means unavailable or undefined in the current aggregate, not zero. An n below8 identifies a metric defined for fewer simulations.

## Detailed cohort metrics

| Setup / cohort | Accuracy % ± SD | NLL ± SD | Macro F1 % ± SD | Boundary F1 % ± SD | Brier ± SD | ECE ± SD |
|---|---:|---:|---:|---:|---:|---:|
| H045 foreign_episode / Present182 | 96.8270 ± 0.5870 | 0.092980 ± 0.020423 | 96.6565 ± 0.6006 | 18.1558 ± 1.3790 | 0.047499 ± 0.008911 | 0.005594 ± 0.003098 |
| H045 foreign_episode / BronzeAge162 | 99.2047 ± 0.3199 | 0.026452 ± 0.012930 | 99.1203 ± 0.3202 | 20.7714 ± 4.0220 | 0.012811 ± 0.005395 | 0.002614 ± 0.001679 |
| H045 foreign_episode / BAA14 | 99.1511 ± 1.2545 | 0.026361 ± 0.026762 | 98.3460 ± 2.5829 | 12.6786 ± 18.9947 | 0.014188 ± 0.017672 | 0.009820 ± 0.008154 |
| H045 foreign_episode / Neo302 | 99.6063 ± 0.0850 | 0.011423 ± 0.002659 | 99.5567 ± 0.1048 | 23.4368 ± 3.4480 | 0.006114 ± 0.001455 | 0.001138 ± 0.000580 |
| H045 foreign_episode / Yam26 | 99.9039 ± 0.1292 | 0.003531 ± 0.002741 | 99.9407 ± 0.0645 | 30.1587 ± 40.9327 (n=7) | 0.001472 ± 0.001773 | 0.001424 ± 0.000810 |
| H045 foreign_episode / Ancient504 | 99.4799 ± 0.1347 | 0.016261 ± 0.004808 | 99.3242 ± 0.2277 | 21.9236 ± 2.6705 | 0.008252 ± 0.002199 | 0.001384 ± 0.000734 |
| H045 foreign_episode / All686 | 98.7761 ± 0.2486 | 0.036615 ± 0.008183 | 98.4993 ± 0.3292 | 19.4440 ± 1.5183 | 0.018664 ± 0.003752 | 0.002101 ± 0.001225 |
| H045 own_state / Present182 | 96.8175 ± 0.6142 | 0.092976 ± 0.020230 | 96.6478 ± 0.6397 | 18.0557 ± 1.7761 | 0.047462 ± 0.008910 | 0.005507 ± 0.002906 |
| H045 own_state / BronzeAge162 | 99.1658 ± 0.3983 | 0.026924 ± 0.014409 | 99.0858 ± 0.3837 | 20.9930 ± 3.8234 | 0.013275 ± 0.006512 | 0.002589 ± 0.001896 |
| H045 own_state / BAA14 | 99.2765 ± 1.1385 | 0.021315 ± 0.019422 | 98.7583 ± 2.2831 | 12.7206 ± 16.1811 | 0.011843 ± 0.014660 | 0.007931 ± 0.005831 |
| H045 own_state / Neo302 | 99.6424 ± 0.0859 | 0.010375 ± 0.002230 | 99.5728 ± 0.1061 | 23.2834 ± 4.0172 | 0.005493 ± 0.001233 | 0.001139 ± 0.000377 |
| H045 own_state / Yam26 | 99.8687 ± 0.1833 | 0.004646 ± 0.003603 | 99.9252 ± 0.0850 | 20.4762 ± 28.0495 (n=7) | 0.002110 ± 0.002403 | 0.002133 ± 0.001464 |
| H045 own_state / Ancient504 | 99.4907 ± 0.1501 | 0.015703 ± 0.004944 | 99.3404 ± 0.2470 | 22.0963 ± 3.0618 | 0.007996 ± 0.002354 | 0.001199 ± 0.000650 |
| H045 own_state / All686 | 98.7815 ± 0.2566 | 0.036204 ± 0.008384 | 98.5135 ± 0.3315 | 19.4555 ± 2.0663 | 0.018467 ± 0.003820 | 0.001965 ± 0.001146 |
| H044 localized / Present182 | 96.7919 ± 0.5967 | 0.094944 | — | 15.5374 ± 1.1777 | — | — |
| H044 localized / BronzeAge162 | 99.1895 ± 0.3391 | 0.027061 | — | — | — | — |
| H044 localized / BAA14 | 99.2328 ± 1.1873 | 0.022134 | — | — | — | — |
| H044 localized / Neo302 | 99.6288 ± 0.0928 | 0.010808 | — | — | — | — |
| H044 localized / Yam26 | 99.8685 ± 0.2092 | 0.004287 | — | — | — | — |
| H044 localized / Ancient504 | 99.4889 ± 0.1358 | 0.016011 | — | — | — | — |
| H044 localized / All686 | 98.7734 ± 0.2475 | 0.036952 | — | — | — | — |
| H044 tree_pooled / Present182 | 96.7866 ± 0.6161 | 0.094962 | — | 18.6750 ± 1.2687 | — | — |
| H044 tree_pooled / BronzeAge162 | 99.1900 ± 0.3386 | 0.026929 | — | — | — | — |
| H044 tree_pooled / BAA14 | 99.2662 ± 1.0684 | 0.022059 | — | — | — | — |
| H044 tree_pooled / Neo302 | 99.6324 ± 0.0993 | 0.010892 | — | — | — | — |
| H044 tree_pooled / Yam26 | 99.8593 ± 0.2264 | 0.004352 | — | — | — | — |
| H044 tree_pooled / Ancient504 | 99.4917 ± 0.1384 | 0.016019 | — | — | — | — |
| H044 tree_pooled / All686 | 98.7740 ± 0.2550 | 0.036963 | — | — | — | — |
| H047 original_risk / Present182 | 96.7777 ± 0.5970 | 0.094808 ± 0.020849 | 96.6136 ± 0.6237 | 18.2775 ± 1.7493 | 0.048271 ± 0.008964 | 0.005880 ± 0.002626 |
| H047 original_risk / BronzeAge162 | 99.1860 ± 0.3514 | 0.027015 ± 0.014085 | 99.1042 ± 0.3453 | 20.8667 ± 3.0520 | 0.013182 ± 0.005900 | 0.002489 ± 0.001713 |
| H047 original_risk / BAA14 | 99.2947 ± 1.0733 | 0.020258 ± 0.017187 | 98.7889 ± 2.1329 | 9.8039 ± 15.0793 | 0.011275 ± 0.013061 | 0.007279 ± 0.004510 |
| H047 original_risk / Neo302 | 99.6369 ± 0.0956 | 0.010674 ± 0.002146 | 99.5742 ± 0.1094 | 23.5846 ± 3.3946 | 0.005629 ± 0.001204 | 0.001162 ± 0.000400 |
| H047 original_risk / Yam26 | 99.8433 ± 0.2268 | 0.004814 ± 0.003852 | 99.9122 ± 0.1094 | 23.2456 ± 26.5523 (n=7) | 0.002287 ± 0.002632 | 0.002206 ± 0.001528 |
| H047 original_risk / Ancient504 | 99.4931 ± 0.1426 | 0.015890 ± 0.004889 | 99.3434 ± 0.2304 | 22.1980 ± 2.7222 | 0.008041 ± 0.002225 | 0.001133 ± 0.000651 |
| H047 original_risk / All686 | 98.7727 ± 0.2516 | 0.036828 ± 0.008563 | 98.5053 ± 0.3240 | 19.6304 ± 2.0264 | 0.018714 ± 0.003814 | 0.002175 ± 0.001060 |
| H047 balanced_risk / Present182 | 96.7576 ± 0.5287 | 0.096819 ± 0.021357 | 96.5750 ± 0.5481 | 18.5574 ± 1.7800 | 0.048867 ± 0.008525 | 0.006078 ± 0.003096 |
| H047 balanced_risk / BronzeAge162 | 99.1282 ± 0.3234 | 0.026923 ± 0.010640 | 99.0392 ± 0.3130 | 20.0447 ± 2.5542 | 0.013445 ± 0.004809 | 0.002919 ± 0.001277 |
| H047 balanced_risk / BAA14 | 99.3747 ± 0.6939 | 0.019795 ± 0.014973 | 98.6400 ± 1.7417 | 17.8906 ± 22.6927 | 0.010744 ± 0.010410 | 0.007543 ± 0.004383 |
| H047 balanced_risk / Neo302 | 99.5660 ± 0.0872 | 0.012386 ± 0.002046 | 99.5398 ± 0.0838 | 23.5915 ± 3.4118 | 0.006610 ± 0.001254 | 0.001086 ± 0.000463 |
| H047 balanced_risk / Yam26 | 99.8167 ± 0.2111 | 0.005930 ± 0.004251 | 99.8929 ± 0.1055 | 22.8852 ± 26.5580 (n=7) | 0.002807 ± 0.002799 | 0.002404 ± 0.001411 |
| H047 balanced_risk / Ancient504 | 99.4329 ± 0.1213 | 0.016932 ± 0.003598 | 99.2656 ± 0.2057 | 21.5887 ± 2.2840 | 0.008725 ± 0.001832 | 0.001013 ± 0.000422 |
| H047 balanced_risk / All686 | 98.7231 ± 0.2210 | 0.038126 ± 0.007804 | 98.4504 ± 0.2848 | 19.6316 ± 1.8046 | 0.019375 ± 0.003496 | 0.002190 ± 0.000960 |
| H042 exact local-clade flags / Present182 | 96.5778 ± 0.5954 | 0.100656 ± 0.020069 | 96.4206 ± 0.5867 | 18.0451 ± 1.8423 | 0.051586 ± 0.009430 | 0.005352 ± 0.002373 |
| H042 exact local-clade flags / BronzeAge162 | 99.0748 ± 0.3242 | 0.030127 ± 0.013364 | 98.9890 ± 0.3241 | 18.5716 ± 3.4095 | 0.014728 ± 0.005658 | 0.003044 ± 0.001611 |
| H042 exact local-clade flags / BAA14 | 99.0226 ± 1.0723 | 0.028516 ± 0.025591 | 98.3862 ± 1.8705 | 8.9583 ± 13.4795 | 0.014997 ± 0.014443 | 0.009864 ± 0.006341 |
| H042 exact local-clade flags / Neo302 | 99.5291 ± 0.1771 | 0.013332 ± 0.003853 | 99.5099 ± 0.1251 | 23.7177 ± 1.9930 | 0.007192 ± 0.002534 | 0.001504 ± 0.000871 |
| H042 exact local-clade flags / Yam26 | 99.8569 ± 0.1871 | 0.005583 ± 0.002992 | 99.9097 ± 0.0979 | 25.8503 ± 40.7653 (n=7) | 0.002324 ± 0.001884 | 0.003210 ± 0.002184 |
| H042 exact local-clade flags / Ancient504 | 99.3859 ± 0.1560 | 0.018753 ± 0.004803 | 99.2089 ± 0.2300 | 20.9611 ± 2.1043 | 0.009580 ± 0.002458 | 0.001312 ± 0.000600 |
| H042 exact local-clade flags / All686 | 98.6409 ± 0.2641 | 0.040482 ± 0.008360 | 98.3583 ± 0.3150 | 19.1084 ± 1.8937 | 0.020725 ± 0.004096 | 0.001963 ± 0.000752 |
| H043 true_partner / Present182 | 96.5182 ± 0.6201 | 0.101527 ± 0.019742 | — | 17.4301 ± 1.3301 | — | — |
| H043 true_partner / BronzeAge162 | 98.9949 ± 0.3463 | 0.030769 ± 0.011750 | — | 18.2673 ± 2.4465 | — | — |
| H043 true_partner / BAA14 | 99.1731 ± 1.0093 | 0.029701 ± 0.027286 | — | 8.6075 ± 13.3855 | — | — |
| H043 true_partner / Neo302 | 99.5198 ± 0.1912 | 0.013422 ± 0.003859 | — | 23.8549 ± 3.7104 | — | — |
| H043 true_partner / Yam26 | 99.8251 ± 0.2007 | 0.005712 ± 0.003086 | — | 26.8170 ± 40.5249 (n=7) | — | — |
| H043 true_partner / Ancient504 | 99.3572 ± 0.1652 | 0.019052 ± 0.004221 | — | 20.9989 ± 2.7105 | — | — |
| H043 true_partner / All686 | 98.6040 ± 0.2688 | 0.040933 ± 0.007877 | — | 18.6763 ± 1.5553 | — | — |
| H043 own_state / Present182 | 96.5029 ± 0.6006 | 0.101819 ± 0.019255 | — | 17.1055 ± 0.7889 | — | — |
| H043 own_state / BronzeAge162 | 99.0107 ± 0.3257 | 0.030553 ± 0.012229 | — | 18.1147 ± 2.2462 | — | — |
| H043 own_state / BAA14 | 99.1048 ± 0.9583 | 0.028291 ± 0.023303 | — | 9.6491 ± 15.0626 | — | — |
| H043 own_state / Neo302 | 99.5217 ± 0.1935 | 0.013512 ± 0.004139 | — | 23.5115 ± 2.9139 | — | — |
| H043 own_state / Yam26 | 99.8547 ± 0.1785 | 0.005701 ± 0.002890 | — | 25.2381 ± 40.9994 (n=7) | — | — |
| H043 own_state / Ancient504 | 99.3630 ± 0.1737 | 0.018997 ± 0.004465 | — | 20.7737 ± 2.3355 | — | — |
| H043 own_state / All686 | 98.6042 ± 0.2691 | 0.040970 ± 0.007892 | — | 18.3817 ± 1.1955 | — | — |
| H040 tensor / Present182 | 96.2674 ± 0.7493 | 0.109332 | — | — | — | — |
| H040 tensor / BronzeAge162 | 98.8266 ± 0.4696 | — | — | — | — | — |
| H040 tensor / BAA14 | 99.1439 ± 0.8901 | — | — | — | — | — |
| H040 tensor / Neo302 | 99.5200 ± 0.1579 | — | — | — | — | — |
| H040 tensor / Yam26 | 99.7215 ± 0.3045 | — | — | — | — | — |
| H040 tensor / Ancient504 | 99.2971 ± 0.2073 | — | — | — | — | — |
| H040 tensor / All686 | 98.4933 ± 0.3390 | — | — | — | — | — |
| H040 event_mlp / Present182 | 96.2007 ± 0.7048 | 0.110014 | — | — | — | — |
| H040 event_mlp / BronzeAge162 | 98.8238 ± 0.4377 | — | — | — | — | — |
| H040 event_mlp / BAA14 | 99.1147 ± 0.9216 | — | — | — | — | — |
| H040 event_mlp / Neo302 | 99.5576 ± 0.1232 | — | — | — | — | — |
| H040 event_mlp / Yam26 | 99.7569 ± 0.2389 | — | — | — | — | — |
| H040 event_mlp / Ancient504 | 99.3197 ± 0.1707 | — | — | — | — | — |
| H040 event_mlp / All686 | 98.4922 ± 0.2948 | — | — | — | — | — |
| H038 aligned whole-tree PyG / Present182 | 95.4891 ± 0.8338 | 0.133497 ± 0.026419 | 95.3410 ± 0.7329 | 13.4195 ± 1.4686 | 0.067902 ± 0.012019 | 0.006785 ± 0.002859 |
| H038 aligned whole-tree PyG / BronzeAge162 | 98.1726 ± 0.8000 | 0.056672 ± 0.027550 | 98.0673 ± 0.6973 | 13.4219 ± 3.9222 | 0.028040 ± 0.012190 | 0.005000 ± 0.005002 |
| H038 aligned whole-tree PyG / BAA14 | 98.9020 ± 0.8854 | 0.041318 ± 0.035436 | 97.5984 ± 2.6745 | 11.4469 ± 13.7319 | 0.018853 ± 0.016903 | 0.011479 ± 0.007373 |
| H038 aligned whole-tree PyG / Neo302 | 99.4103 ± 0.1271 | 0.017646 ± 0.004093 | 99.3102 ± 0.1346 | 18.3457 ± 3.6103 | 0.008982 ± 0.002010 | 0.001347 ± 0.000626 |
| H038 aligned whole-tree PyG / Yam26 | 98.9598 ± 1.3143 | 0.031237 ± 0.039239 | 99.4083 ± 0.7371 | — | 0.014574 ± 0.018678 | 0.008073 ± 0.005129 |
| H038 aligned whole-tree PyG / Ancient504 | 98.9751 ± 0.3226 | 0.031549 ± 0.010976 | 98.6906 ± 0.4364 | 15.5922 ± 2.8868 | 0.015670 ± 0.004836 | 0.002013 ± 0.001878 |
| H038 aligned whole-tree PyG / All686 | 98.0503 ± 0.4312 | 0.058596 ± 0.014197 | 97.6933 ± 0.4757 | 14.1991 ± 1.4048 | 0.029528 ± 0.006326 | 0.002720 ± 0.001292 |
| H038 zero-concordance control / Present182 | 94.9252 ± 0.7917 | 0.147232 ± 0.023024 | 94.5950 ± 0.8236 | 11.7904 ± 1.3304 | 0.075842 ± 0.011726 | 0.007967 ± 0.003782 |
| H038 zero-concordance control / BronzeAge162 | 98.4120 ± 0.5405 | 0.050818 ± 0.017204 | 98.2085 ± 0.5556 | 11.3571 ± 2.5173 | 0.024523 ± 0.007704 | 0.005103 ± 0.001702 |
| H038 zero-concordance control / BAA14 | 98.6137 ± 0.9470 | 0.042836 ± 0.026582 | 97.3308 ± 2.3398 | 11.6319 ± 13.6475 | 0.021767 ± 0.014609 | 0.008893 ± 0.005237 |
| H038 zero-concordance control / Neo302 | 99.1716 ± 0.2703 | 0.026157 ± 0.008162 | 99.1637 ± 0.2141 | 15.1690 ± 3.4723 | 0.012807 ± 0.004090 | 0.002698 ± 0.000514 |
| H038 zero-concordance control / Yam26 | 99.4215 ± 0.8073 | 0.020316 ± 0.024105 | 99.4852 ± 0.8104 | — | 0.009662 ± 0.013054 | 0.006778 ± 0.004410 |
| H038 zero-concordance control / Ancient504 | 98.9248 ± 0.2837 | 0.034246 ± 0.008445 | 98.5426 ± 0.4572 | 13.0851 ± 2.2646 | 0.016659 ± 0.004108 | 0.002850 ± 0.000872 |
| H038 zero-concordance control / All686 | 97.8637 ± 0.3727 | 0.064222 ± 0.010927 | 97.3936 ± 0.4620 | 12.2624 ± 1.3400 | 0.032361 ± 0.005505 | 0.003147 ± 0.001240 |
| Current native ARGMix / Present182 | 94.4461 ± 1.1491 | 0.219695 ± 0.023975 | 94.3809 ± 1.2151 | 11.7372 ± 1.7328 | 0.092752 ± 0.015147 | 0.078235 ± 0.006887 |
| Current native ARGMix / BronzeAge162 | 97.7051 ± 0.7466 | 0.137894 ± 0.018029 | 97.5701 ± 0.7104 | 11.3857 ± 2.6200 | 0.044121 ± 0.010198 | 0.078360 ± 0.006257 |
| Current native ARGMix / BAA14 | 98.4370 ± 1.0186 | 0.134126 ± 0.022226 | 97.3018 ± 2.2022 | 0.0000 ± 0.0000 | 0.039004 ± 0.013836 | 0.088624 ± 0.010676 |
| Current native ARGMix / Neo302 | 99.2955 ± 0.2623 | 0.102450 ± 0.005403 | 99.2418 ± 0.2387 | 16.7763 ± 3.9630 | 0.021148 ± 0.003219 | 0.082591 ± 0.002144 |
| Current native ARGMix / Yam26 | 99.2218 ± 0.8966 | 0.099234 ± 0.022864 | 99.6206 ± 0.4101 | — | 0.023563 ± 0.012912 | 0.079989 ± 0.011454 |
| Current native ARGMix / Ancient504 | 98.7566 ± 0.3114 | 0.114557 ± 0.008198 | 98.3958 ± 0.4842 | 13.8259 ± 2.5466 | 0.029153 ± 0.004557 | 0.081095 ± 0.002458 |
| Current native ARGMix / All686 | 97.6130 ± 0.4975 | 0.142450 ± 0.011556 | 97.1989 ± 0.6197 | 12.5034 ± 1.4764 | 0.046026 ± 0.006914 | 0.080312 ± 0.002686 |
| Native AncestralPaths derived C4 / Present182 | 93.7830 ± 0.8385 | ∞ | 93.2918 ± 0.9790 | 7.0281 ± 0.7779 | 0.098592 ± 0.012700 | 0.010487 ± 0.002875 |
| Native AncestralPaths derived C4 / BronzeAge162 | 94.7589 ± 1.1847 | — | 94.1623 ± 1.1755 | 3.0986 ± 0.9384 | 0.082388 ± 0.018556 | 0.007534 ± 0.005079 |
| Native AncestralPaths derived C4 / BAA14 | 96.9352 ± 1.3195 | 0.101712 ± 0.043356 | 93.7895 ± 7.4457 | 1.0897 ± 1.1980 | 0.048373 ± 0.021433 | 0.008653 ± 0.003687 |
| Native AncestralPaths derived C4 / Neo302 | 97.5693 ± 0.2748 | — | 97.1480 ± 0.4419 | 3.4446 ± 0.7146 | 0.038234 ± 0.004584 | 0.008276 ± 0.001975 |
| Native AncestralPaths derived C4 / Yam26 | 95.4303 ± 1.4096 | — | 95.4225 ± 1.4282 | — | 0.071040 ± 0.022790 | 0.011779 ± 0.009506 |
| Native AncestralPaths derived C4 / Ancient504 | 96.5380 ± 0.3194 | — | 95.3004 ± 0.7410 | 2.9699 ± 0.2703 | 0.054400 ± 0.004876 | 0.004829 ± 0.001563 |
| Native AncestralPaths derived C4 / All686 | 95.8071 ± 0.4473 | — | 94.8565 ± 0.6681 | 4.6069 ± 0.3380 | 0.066125 ± 0.006794 | 0.004355 ± 0.000956 |

## Live full-budget training

| Run | GPU | Verified process present | Passes | Actual updates per arm | Planned updates per arm | Validation accuracy |
|---|---:|---|---:|---|---:|---|
| H046 | 5 | True | 29/30 | {'foreign_flank': 928, 'own_state': 928} | 960 | pending |
| H047 | 4 | False | 30/30 | {'balanced_risk': 960, 'original_risk': 960} | 960 | 96.7777% original-risk, 96.7576% balanced-risk |
| H048 | 4 | True | 1/30 | {'graph_to_own_confidence_fixed_distance_chain': 32, 'graph_to_transition': 32} | 960 | pending |

Update counts below are verified receipt lower bounds, not live optimizer counters.

- H046: training; completed-pass update counts are a lower bound. AdamW lr=0.001, weight_decay=0.0001, seed=101, dtype=float32.
- H047: TRAIN and VAL completed. AdamW lr=0.001, weight_decay=0.0001, seed=101, dtype=float32.
- H048: training; completed-pass update counts are a lower bound. AdamW lr=0.001, weight_decay=0.0001, seed=101, dtype=float32.

## Registered next architecture experiment

GPU4 training; completed-pass update counts are a lower bound. 1/30 completed passes; {'graph_to_own_confidence_fixed_distance_chain': 32, 'graph_to_transition': 32} verified update lower bounds. No validation accuracy yet.

[Fresh independent novelty/cross-domain report](../research/reviews/H048_novelty_crossdomain_fresh_20261015_v1/report.md). Fresh independent H048 audit: reset transitions have RFMix precedent; a distinct graph-collection operator is unestablished. GST reaction-center sets have no native chain order, so whole-H048 transfer is not justified. PKU-MMD is only a possible future decoder study after native label-overlap semantics are resolved; no transfer result exists.

H048 retains 31,933 parameters per arm and a fixed 30-pass/960-update schedule. Its matched comparison places the same graph signal into transitions or confidence. The reset-chain components have prior art. No novelty or accuracy claim is established.

## American C3 benchmark

| Method | Current ancestry scores | Status |
|---|---|---|
| Native ARGformer | unavailable | V11 failed with actual root exit1; native optimizer updates unknown. V12 operational source reviewed; exact namespace cleanup completed, GPU binding/activation and launch pending. |
| Native FLARE | unavailable | Author build and TRAIN000 full-site conversion completed with actual wait0. Native inference: running on CPU; sealing: not admitted. Accuracy pending; TRAIN000 is development, not held-out. |
| Our C3 GNN | unavailable | TRAIN000 exact native focal/plan preparation: terminal; actual receipt review required. Shared 30 Mb VAL000 inference completed. Remaining TRAIN parents, checkpoint and full prespecified VAL gate pending |

The TRAIN000 conversion is development evidence. Independent VAL000 inference now has its own actual completion/wait receipts; its method predictions and common-truth scoring remain pending. Original 30 Mb VAL000 simulation: completed; actual wait0 and owned-group drain verified; inferred observation: completed; actual wait0 and owned-group drain verified; shared native-path/10k-focal preparation: running on CPU. R600/Q1000 identities stay common. No VAL ancestry score. Original-paper simulation assumptions and source versions remain explicitly documented.

## GST molecular graph-set pilot

Only the saved TRAIN 1000/VALID 1495, seed 0 CPU pilot is scored: four models, AdamW lr=0.001, weight_decay=0.0001, batch=16, 20 epochs/1260 updates per fit. Full TRAIN has 10439/10500 retained reactions; this count is data readiness, not a fit. The full-data multi-seed fit is inactive and unfinished. Our molecular adapter and native paired TEST comparison remain unfinished; TEST withheld. GST inputs are unordered reactant sets with no native chain order, so whole-H048 transfer is not justified. AP is pooled per-atom average precision. One seed supplies no training-seed SD.

| Native setup | VALID AP | Selected epoch |
|---|---:|---:|
| gcn-ds | 0.63945129 | 17 |
| gcn-st | 0.63823432 | 16 |
| gsc-bc | 0.65378329 | 19 |
| gsc-ca | 0.63791939 | 19 |

Full TRAIN parsing retains10,439/10,500 reactions. This is data readiness, not a new model score. [Fresh task-semantics/transfer assessment](../research/reviews/H048_novelty_crossdomain_fresh_20261015_v1/report.md).

## Earlier adapted ARGformer pilot

Two European VAL simulations,280observed haplotypes,64present-day/56ancient queries,2Mb,tsinfer/tsdate trees. Native per-ARG5epoch MLM plus1reference-only contrastive epoch. This differs from both the952input panel above and the original American benchmark. SD is across two VAL simulations. Raw hard-vote NLL is infinite. No probability floor or recalibration.

| Setup | Present64 accuracy % ± SD | Ancient56 accuracy % ± SD | All120 accuracy % ± SD |
|---|---:|---:|---:|
| Adapted ARGformer top1 | 75.6818 ± 4.5195 | 75.4447 ± 3.5081 | 75.5711 ± 4.0476 |
| Adapted ARGformer top5 | 75.1358 ± 3.7500 | 74.7885 ± 4.0220 | 74.9737 ± 3.8769 |

[Actual summary with all metrics](../experiments/results/native_argformer_tsinfer_fit_v7_summary_v2/summary.json).

## Real data and paper

H045 empirical probabilities, bp-weighted proportions and original-genotype PCA written. Actual successful wait and completion verified. Descriptive24query haplotypes/12subjects only. No known ancestry truth. No empirical ancestry accuracy exists. The AAAI draft is8pages and remains unfinished.

## Evidence and complete recorded inventory

- H045 foreign_episode: [research/reviews/H045_completed_comparison_root_20261015_v1/summary.json](../research/reviews/H045_completed_comparison_root_20261015_v1/summary.json)
- H045 own_state: [research/reviews/H045_completed_comparison_root_20261015_v1/summary.json](../research/reviews/H045_completed_comparison_root_20261015_v1/summary.json)
- H044 localized: [research/reviews/H044_completed_comparison_root_20261015_v1/summary.json](../research/reviews/H044_completed_comparison_root_20261015_v1/summary.json)
- H044 tree_pooled: [research/reviews/H044_completed_comparison_root_20261015_v1/summary.json](../research/reviews/H044_completed_comparison_root_20261015_v1/summary.json)
- H047 original_risk: [research/reviews/H047_completed_comparison_root_20261015_v1/summary.json](../research/reviews/H047_completed_comparison_root_20261015_v1/summary.json)
- H047 balanced_risk: [research/reviews/H047_completed_comparison_root_20261015_v1/summary.json](../research/reviews/H047_completed_comparison_root_20261015_v1/summary.json)
- H042 exact local-clade flags: [research/reviews/H042_completed_cohort_diagnostics_root_20261015_v1/summary.json](../research/reviews/H042_completed_cohort_diagnostics_root_20261015_v1/summary.json)
- H043 true_partner: [research/reviews/H043_completed_comparison_root_20261015_v1/summary.json](../research/reviews/H043_completed_comparison_root_20261015_v1/summary.json)
- H043 own_state: [research/reviews/H043_completed_comparison_root_20261015_v1/summary.json](../research/reviews/H043_completed_comparison_root_20261015_v1/summary.json)
- H040 tensor: [research/reviews/H040_completed_comparison_root_20261015_v1/summary.json](../research/reviews/H040_completed_comparison_root_20261015_v1/summary.json)
- H040 event_mlp: [research/reviews/H040_completed_comparison_root_20261015_v1/summary.json](../research/reviews/H040_completed_comparison_root_20261015_v1/summary.json)
- H038 aligned whole-tree PyG: [research/reviews/native_common_cohort_diagnostics_20261015_v1/same_support_comparison.json](../research/reviews/native_common_cohort_diagnostics_20261015_v1/same_support_comparison.json)
- H038 zero-concordance control: [research/reviews/native_common_cohort_diagnostics_20261015_v1/same_support_comparison.json](../research/reviews/native_common_cohort_diagnostics_20261015_v1/same_support_comparison.json)
- Current native ARGMix: [research/reviews/native_common_cohort_diagnostics_20261015_v1/native_cohort_metrics_mean_SD.json](../research/reviews/native_common_cohort_diagnostics_20261015_v1/native_cohort_metrics_mean_SD.json)
- Native AncestralPaths derived C4: [research/reviews/native_common_cohort_diagnostics_20261015_v1/native_cohort_metrics_mean_SD.json](../research/reviews/native_common_cohort_diagnostics_20261015_v1/native_cohort_metrics_mean_SD.json)

[All recorded metric receipts, including historical pilots](all_recorded_metric_receipts.md) · [Machine-readable current scoreboard](current_scoreboard.json)
