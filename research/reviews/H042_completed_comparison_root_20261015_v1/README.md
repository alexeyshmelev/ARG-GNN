# H042 completed local-clade control

All three compared GNN arms start from the same H038 checkpoint and receive 30 additional FP32 passes, 960 full-parent updates and six frozen checkpoint choices. Each has30,788parameters. All predictions precede VAL truth. One optimizer seed is used, and SD is across eight VAL simulations.

| Arm | Present182 accuracy, mean ± SD (%) | NLL |
| --- | ---: | ---: |
| Exact local-clade flags | 96.5778 ± 0.5954 | 0.100656 |
| tensor | 96.2674 ± 0.7493 | 0.109332 |
| event_mlp | 96.2007 ± 0.7048 | 0.110014 |

The simpler local-clade flag control performs better. H040's tensor layer therefore does not demonstrate a gain beyond local topology-change information. The prospective H043 experiment preserves these flags and tests actual neighboring-clade states against a matched partner-blind operator.

| Population | Flag accuracy, mean ± SD (%) |
| --- | ---: |
| age_stratum_BAA | 99.0226 ± 1.0723 |
| age_stratum_Bronze_Age | 99.0748 ± 0.3242 |
| age_stratum_Neo | 99.5291 ± 0.1771 |
| age_stratum_Yam | 99.8569 ± 0.1871 |
| all_queries_686 | 98.6409 ± 0.2641 |
| primary_present_day_182 | 96.5778 ± 0.5954 |
| secondary_ancient_504 | 99.3859 ± 0.1560 |

Current native ARGMix remains94.4461% on the same development endpoint, a2.1317pp pipeline difference. This comparison includes training/selection and SNP-input differences and does not reproduce the published panel. No independent test or top1 claim is established. Complete paired effects and original source refs are in [summary.json](summary.json).
