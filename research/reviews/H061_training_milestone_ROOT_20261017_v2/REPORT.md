# H061 complete-fit training milestone

All six registered fits produced selections after 40 epochs and 800 updates each: 4,800 updates total. This snapshot precedes acceptance of the overall job wait and drain. TEST remains withheld.

| Model | Selected VAL MAE, cases | Selected VAL RMSE, cases | Seeds |
|---|---:|---:|---:|
| TreeLoom | 15.968411 ± 0.061488 | 25.520040 ± 0.261292 | 3 |
| TreeLoom-NoMem | 16.059604 ± 0.075709 | 25.929755 ± 0.270482 | 3 |

Sample SD is across initialization seeds, each at its own validation-selected epoch. Scores cover 78 validation weeks and all 20 counties, using 52 previous weeks and TRAIN-only scaling. These are selected validation forecasts, not TEST scores or ancestry accuracy. The registered TEST utility criterion is unchanged.

| Seed | Model | Selected epoch | VAL MAE | VAL RMSE |
|---|---|---:|---:|---:|
| 101 | full | 22 | 16.034358 | 25.807364 |
| 101 | no_memory | 12 | 16.113024 | 26.235912 |
| 202 | full | 37 | 15.958219 | 25.296648 |
| 202 | no_memory | 23 | 16.092824 | 25.830167 |
| 303 | full | 26 | 15.912656 | 25.456109 |
| 303 | no_memory | 38 | 15.972965 | 25.723185 |

The generic graph field, weekly clock and regression head are explicit domain adaptations. No ancestry weights were reused. All six complete histories and their metadata SHA references are preserved in the milestone JSON.
