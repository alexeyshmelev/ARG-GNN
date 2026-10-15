# H042 completed population and boundary diagnostics

Derived only from the already completed selected-pass30 evaluation. One optimizer seed101. Mean and sample SD are across eight validation simulations. No new predictions, query-truth access or selection changes.

| Cohort | Accuracy, mean ± SD (%) | Boundary F1, mean ± SD (%) |
|---|---:|---:|
| Present (182) | 96.5778 ± 0.5954 | 18.0451 ± 1.8423 |
| Bronze Age (162) | 99.0748 ± 0.3242 | 18.5716 ± 3.4095 |
| BAA (14) | 99.0226 ± 1.0723 | 8.9583 ± 13.4795 |
| Neo (302) | 99.5291 ± 0.1771 | 23.7177 ± 1.9930 |
| Yam (26) | 99.8569 ± 0.1871 | 25.8503 ± 40.7653 |
| Ancient (504) | 99.3859 ± 0.1560 | 20.9611 ± 2.1043 |
| All (686) | 98.6409 ± 0.2641 | 19.1084 ± 1.8937 |

Present182 boundary precision is 15.2079 ± 2.0112% and recall is 22.2952 ± 1.8578%, with 1 kb tolerance. Boundary detection remains weak despite high span accuracy.

H042 has higher mean span accuracy than native AP and current native ARGMix in all seven reported cohorts. These are development-pipeline comparisons. Input features, training budgets and checkpoint selection differ. There is no original-paper superiority, independent-test confirmation or training-seed uncertainty claim.

Boundary metrics remain undefined where no relevant boundaries exist. Their mean and SD use only parents with a defined value, with metric-specific counts in the JSON. Present182 uses all eight parents. The JSON also records per-parent boundary values and pooled span confusion matrices. Pooled confusion is weighted by query-base pairs, while table values give equal weight to each simulation.

[Complete summary](summary.json) · [CSV](cohort_metrics.csv)
