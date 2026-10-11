# GPU utilization follow-up: existing seven-mechanism run and cached seed303

The reviewed cache can be used for the prespecified seed303 without changing scientific sources. Root must accept the successful profile and supply fresh cache/memory, storage, registration and launcher authority. No additional cache implementation or review cycle is proposed.

The native ARGformer GPU3 probe/fit remains a separate root operation. GPU2 currently holds the exclusive seed202 run, so this draft targets GPU2 after that lease naturally releases unless root separately admits reviewed shared scheduling.

## Measured diagnosis

The completed seed101 run took 9.11 hours with peak CUDA allocation 4.29 GB. Its source/receipt counts imply 480 TRAIN bank loads shared across active models, 3,200 epoch-VAL loads, and 112 final selected-VAL loads: 3,792 total. VAL reloads dominate the number of bank requests.

The genuine one-tile profile passed full field/logit/VJP/clipped-AdamW/RNG parity for all seven models and exited0. Original loads averaged 2.023s, warm cache loads 0.993s, and cold cache initialization 2.977s. Warm load latency fell 50.9%. Full bank SHA verification remains on every hit. Peak profile CUDA allocation was 4.839GB, with0.542GB resident CPU/GPU cache for this single bank.

Applying that one TRAIN132-tree latency to all counted seed101 requests gives an illustrative 2.13h of loading (23.4% of total wall time). Warm-cache savings would be about 65min (11.9% of total), before cold costs/evictions. This is an extrapolation, not a measured full-run phase split or guaranteed speedup. Other banks/sizes and simultaneous CPU work differ.

The uncached runner has no training/loading timers. File-mtime wall segments provide a useful narrower observation:

| Snapshot | Seven-active epochs measured | Previous last VAL seal to next first VAL seal | First to last VAL seal | Last-seal epoch cadence |
|---|---:|---:|---:|---:|
| Seed101 | 27 | 11.97min | 6.90min | 18.88min |
| Seed202 | 8 | 11.25min | 7.07min | 18.32min |

The first segment includes TRAIN bank loads, all active optimizer steps, preceding bookkeeping and the first VAL parent's prediction/seal. The second includes remaining VAL loads, forwards, scoring and writes. Neither is pure GPU kernel time. The remainder of wall time cannot be labelled training compute from these logs. Existing CPU loading evidence separately measured0.934s SHA,0.729s archive decode and0.165s CPU conversion on this bank. This supports retaining the reviewed cache for the next seed, not removing hash guards.

Seed202 receipt snapshot is running: all seven models have 9 recorded epochs and 144 steps each. It is incomplete and its current VAL minima are provisional.

## Selected seed101 results

These are exact-support equal means over two development VAL parents, with no TEST parents. Each model selects its own checkpoint using present64 NLL. The five depth2 mechanisms stopped after28 epochs/448 steps under original patience8; depth1 variants ran30 epochs/480 steps.

| Mechanism | Selected epoch | Present64 accuracy | Present64 NLL | Present64 macro-F1 | Ancient56 accuracy | Ancient56 NLL |
|---|---:|---:|---:|---:|---:|---:|
| scalar_age_d2 | 20 | 84.80% | 0.4281 | 0.8332 | 88.36% | 0.3289 |
| age_propagation_only_d2 | 20 | 84.86% | 0.4279 | 0.8339 | 88.44% | 0.3268 |
| age_source_only_d2 | 20 | 84.85% | 0.4269 | 0.8341 | 88.27% | 0.3291 |
| age_gainmatched_d2 | 20 | 83.73% | 0.4531 | 0.8181 | 86.52% | 0.3775 |
| noncontext_age_d2 | 20 | 84.88% | 0.4262 | 0.8341 | 88.47% | 0.3263 |
| age_d1 | 28 | 84.70% | 0.4258 | 0.8331 | 88.41% | 0.3271 |
| age_d1_plus_pointwise | 26 | 84.74% | 0.4246 | 0.8319 | 88.33% | 0.3247 |

Gain-matched diagonal is worse in this seed. The other selected present64 accuracies span84.70–84.88%, and NLLs0.4246–0.4281. Depth1+pointwise has the lowest selected NLL; noncontext has the highest accuracy. This does not establish a reliable mechanism advantage from one optimizer seed and two reused development parents.

## Concrete root handoff

Drafts: seed303_profile_gate.draft.json, seed303_memory.draft.json, seed303_cache_admission.draft.json, seed303_launcher_admission.draft.json, seed303_ready_command.txt. All remain inactive. They preserve existing64GB CPU/32GB GPU tensor-cache caps and the reviewed49,152MiB total GPU/128GB total CPU headroom plan. These are reservations, not full32-bank measured peaks.

Root activation requires accepting the existing measured gate, fresh <=24h dates and memory/watchdog authority, fresh complete original storage snapshot, new52,321,048,576-byte output/tmp reservation, fresh first-read ledger pins, H020 seed303/cache repeat registration, and a free held GPU lease. The unchanged original runner still checks its own450GB snapshot arithmetic. The reviewed budgeted launcher supplies supplemental accounting without changing scientific arguments.

Use the provided exact original argument vector through arggnn.biological_bank_cache_v1. Keep seed202's active uncached run intact and do not rerun the completed seed101. Terminal bank_cache_execution_v1.json already reports request time, hits, misses, evictions and resident peaks, which will permit an actual cached-loop loading fraction without new infrastructure.

All work here was JSON/source/log/file-metadata inspection plus small draft/report writes in this dedicated folder. No arrays, models, GPU actions, live budget operations, Git, original sources, README or ledger were changed.
