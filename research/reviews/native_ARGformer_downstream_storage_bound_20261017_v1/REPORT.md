# Native ARGformer downstream storage bound, source only

Recommendation: prepare and independently review a new **60,000,000,000-byte aggregate-only failstop** for the unchanged full downstream route. Do not admit it at the supplied current projection. Retain 8/56/1 GiB group caps, two regular 3 GiB checkpoint files, all seven phases, complete output roots and every guard. This conditional route has unproved storage fit.64GiB is not a demonstrated scientific minimum. No source or policy is changed here.

Accepted projection 931,184,819,519B leaves 48,815,180,481B under 980 GB. Mandatory persisted tensor payloads require 49,979,548,096B, already 1,164,367,615B too large before any checkpoint, serialization, log, receipt or event. No cap fitting present headroom can finish unchanged science.

| Product | Rows | Tensor payload bytes |
|---|---:|---:|
| contrastive/train | 8,748,479 | 44,932,188,144 |
| contrastive/val | 972,757 | 4,996,079,952 |
| query_paths/train | 9,003 | 46,167,384 |
| query_paths/val | 997 | 5,112,616 |

Each reference row consumes 768 float32 +128 int64 IDs +128 int64 masks +1 int64 extant ID +1 int64 label = 5136 B. Queries omit labels and consume 5128 B. Exact dtypes/axes/dimension are required by src/arggnn/american_c3_native_concat_v1.py lines132-156. Immutable counts SHA502c111b... supply all 9,721,236 reference and 10,000 query rows. Four original outputs persist unchanged once.

The accepted I/O adaptation changes new checkpoint retention from -1 to 1 and makes complete combined tensor/sidecar writes RAM-only. All 107 save events remain:106 periodic 20ba plus final. Composer writes the next checkpoint before removing the old, so two regular files coexist, capped at6,442,450,944B total payload. Post-drain alias normalization requires one retained regular file, copies it to one separate temporary regular inode, then replaces the alias. This also peaks at two; final durable state retains final plus independent latest regular copy. The older108-file/all-retention hypothetical cap product347,892,350,976B does not apply to this disclosed keep1 route. Historical files/failures/charges remain.

Combined RAM tensors require 49,979,548,096B and full row-order arrays311,319,552B. Their50,290,867,648B combined disk duplication is avoided, not deleted or credited. Original normalization still allocates full normalized vectors. No RAM/GPU fit follows.

Reviewed new output+AUX/TMP/cache/log accepted-state cap is68,719,476,736B. Groups 8 GiB contrastive / 56 GiB embeddings / 1 GiB retrieval+finalization sum 65 GiB but remain intersected with 64 GiB. Other limits: 48 GiB per embedding file, 256 MiB retrieval file, 8 MiB scientific JSON, 3 GiB per checkpoint. These are failstops, not actual sizes. Direct author writes retain monitor/after-write checks; no instantaneous physical no-overshoot bound is established.

At 60 GB, exact tensors + both maximum checkpoint files + full 1 GiB retrieval/finalizer group total57,495,740,864B, leaving2,504,259,136B for embedding serialization and other contrastive output/AUX. Holding the entire 8 GiB contrastive group and 1 GiB retrieval group leaves only356,775,488B for all embedding overhead/AUX. These are allocation envelopes, not unknown-remainder upper bounds. Four separate 256 MiB CUDA cache settings permit 1 GiB in the four embedding phase roots alone. Neither 60 GB fit nor 64 GiB necessity/fit is proved. A new 60 GB intersection can honestly fail with partial files retained and science unchanged, like 64 GiB. The full new entry closure requires independent review.

After one genuine original numerical 20 GB Census retirement, actual retained bytes R and all later net accounting deltas D must satisfy strict inequalities including 512 KiB retirement event plus 512 KiB reserve event:

- 60GB: R + D < 8,814,131,905B.
- 64GiB: R + D < 94,655,169B.

A hypothetical D=2,000,000,000B yields 6,814,131,905 B and -1,905,344,831 B. This 2 GB is not an actual admission/delta. Pending FLARE admission and every operation/event must come from ROOT's later genuine state. Numerical retained bytes/credit stay UNKNOWN. Without that retirement, 60 GB+reserve event needs11,185,343,807B release and 64 GiB needs19,904,820,543B.

ROOT establishes no large completed-but-unclosed scientific key. AGENTS names a completed FLARE VAL001 seal retaining 40,388,608 B without ordinary close at that record. Even its entire allowance cannot bridge the native deficit. No close/retirement/completion inference occurs. Saved MLM outer 1 stays 1; artifact acceptance never establishes old-scope successful completion. Use accepted 5 epoch / 227990 update MLM only through separate inline contrastive initialization/qualification. MLM replay stays forbidden. Downstream 1 epoch / 2135 updates / 4096 batch + microbatch / FP32 / seed 36, all full rows, four batch 256 / head_norm embeddings and original top1 arithmetic/settings stay exact. Truth/scoring retains existing separate barriers.

Method: exact source and compact immutable metadata, SHA checks and integer arithmetic with .runtime/env/bin/python -B -s. No operational imports, guard/journal/filesystem inventories, resource/process checks, payload reads, optimizer calls, jobs, Git or accounting mutation. Writes only this report directory. Two authoring parse failures before remote execution are preserved in JSON. Exact source SHA refs and calculations are in bound.v1.json.
