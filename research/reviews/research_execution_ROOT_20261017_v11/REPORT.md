# Research and paper status

The paper and Figure 1 rewrite is published at commit `58d8b58`. The draft has nine pages. The figure shows matching ARG and marginal-tree identities, observed input vectors, two GNN passes, sample memory, a BiGRU and local ancestry inference (LAI) output. A fresh reader recovered the method from the image alone and found no drawing blocker after the input connection was corrected. The scientific goal remains active. The paper is not submission ready.

## Completed Chickenpox TEST result

All methods cover the same 79 TEST weeks and 20 counties, giving 1,580 forecasts per prediction product. All six fits completed 40 epochs and 800 updates each, or 4,800 updates in total. The table gives mean and sample SD across three initialization seeds in case-count units. Lower values are better.

| Method | MAE, cases | RMSE, cases |
|---|---:|---:|
| TreeLoom | 13.907918 ± 0.242561 | **24.268971 ± 0.094496** |
| TreeLoom-NoMem | **13.648097 ± 0.050932** | 24.408159 ± 0.106380 |
| Last value | 17.496835 | 31.412134 |
| Seasonal lag 52 | 20.021519 | 33.840168 |
| Ridge, frozen alpha 1000 | 18.958667 | 27.298175 |

Full failed the registered memory criterion. Its MAE was 1.904% worse than NoMem. The paired MAE difference was +0.259820 cases, with a conditional 95% interval of [+0.105412, +0.427353] using circular 13-week blocks. Its RMSE was 0.139188 cases lower. Shuffling the history worsened both candidates, supporting reliance on chronology only. These results concern one fixed graph and time series. They do not establish generalization to independent populations. The unavailable legacy GRU is excluded. The negative result is retained without retuning or replay.

## Running biological experiments

Native ARGformer has 193,000 of 227,990 saved updates verified across all 77 optimizer states. Training is incomplete and no ancestry score is available. The repaired H058 run was admitted and launched with all 1,920 scheduled updates. H060 American training and C3 prediction supervisors remain active, with no new accepted ancestry metric. FLARE's original VAL001 conversion remains active. Later stages have not been released.

## Efficiency and architecture

Nine small synthetic CPU fixtures passed for the Torch/PyG observation kernel. Integer counts agree exactly and FP32 summary features agree within the stated tolerance on those fixtures. This establishes neither production speedup nor compatibility with historical feature hashes. A fresh integration is being developed. The separate ordered-marker placement experiment passed independent source review and actual admission. Its complete 1,920-update training supervisor is launched. It has no verified update count or efficacy result yet. The proposed cavity exchange was rejected as established sum-product belief propagation and will not be presented as a new operator.

## Capacity

The selected accounting state at 03:16:45 UTC projects 978,580,882,913 bytes against the 980 GB operational ceiling and 1 TB human limit. The retention preoperation check returned exit 1 because its mutex was occupied. Actual retirement did not begin. No credit was accepted, no file was deleted, and no scientific success was inferred. The full exact-ALL Relate run remains unlaunched pending genuine capacity.

All project writes remain inside the dedicated server repository. Physical GPU 0 only. Existing external processes and all failed attempts are preserved.
