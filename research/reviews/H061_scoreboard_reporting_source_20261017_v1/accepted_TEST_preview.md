
## H061 Chickenpox accepted TEST outcome

H061 TEST scoring completed and ROOT accepted after consumed outer 0, worker 0 and owned drain; registered memory utility gate failed.

Rolling one-week-ahead forecasts on all 79 TEST weeks (indices 443–521) and 20 counties: 1,580 forecasts per method, with 52 observed past weeks and a TRAIN-only scaler. Lower is better. Bold marks the best value within these five comparable methods for each metric.

| Method | TEST MAE (cases) ± seed SD | TEST RMSE (cases) ± seed SD |
|---|---:|---:|
| Full | 13.907918 ± 0.242561 | **24.268971 ± 0.094496** |
| NoMem | **13.648097 ± 0.050932** | 24.408159 ± 0.106380 |
| Last value | 17.496835 | 31.412134 |
| Seasonal lag 52 | 20.021519 | 33.840168 |
| Ridge α=1000, frozen prior pilot | 18.958667 | 27.298175 |

SD is sample SD in case-count units across initialization seeds 101, 202 and 303, conditional on one fixed county graph/time series. Fixed comparators have one prediction product each, so seed SD is unavailable.
The registered memory-utility gate failed: Full mean MAE ≤ 0.98 × NoMem: failed; upper 95% paired interval below zero: failed; mean RMSE no worse: passed. Full-minus-NoMem MAE +0.259820 cases, conditional 95% circular 13-week block interval [+0.105412, +0.427353] (10,000 draws). This interval describes one time series, not independent populations.
All 15 frozen products preceded scoring. Six 40-epoch fits completed 800 AdamW updates each (4,800 total), learning rate 0.001 and weight decay 0.0001; CPU, one thread, FP32. No new fit or update occurred during scoring. Legacy GRU is unavailable because its payload was not accepted. No broad superiority, ancestry-transfer or novelty claim follows.
Negative completion retained without tuning or replay. The selected VAL milestone below is historical evidence recorded before TEST release.

[Accepted TEST scores](../experiments/results/H061_postselection_score_tx_20261017_v1/scores.json) · [ROOT consumed-terminal acceptance](../research/reviews/H061_score_ROOT_acceptance_20261017_v1/score_acceptance.root_active_v1.json) · [Frozen protocol](../research/reviews/H061_Chickenpox_current_core_registration_20261017_v1/protocol.root_frozen_v1.json)


README/current-row summary:

Full TEST MAE 13.907918 ± 0.242561, RMSE **24.268971 ± 0.094496** cases; NoMem TEST MAE **13.648097 ± 0.050932**, RMSE 24.408159 ± 0.106380 cases. Sample SD in cases across three initialization seeds on one fixed series; 79 weeks × 20 counties. Full−NoMem MAE +0.259820 cases, conditional 95% 13-week block interval [+0.105412, +0.427353]. Registered memory utility failed
