# H048 throughput source audit

The observed long gap is consistent with waiting for the shared scheduler mutex. It is not explained by the per-update synchronization candidate below. Exact PID/FD evidence is in [BOUNDARY_FINDING.md](BOUNDARY_FINDING.md) and [boundary_evidence_actual_v1.json](boundary_evidence_actual_v1.json). Root independently owns mutex-owner diagnosis and interventions. This worker inspected only exact H048 PID metadata after explicit authorization, sources and existing JSON receipts, without science/checkpoint/truth/probability payloads or GPU profiling.

## Recorded throughput and its limits

Twelve genuine pass receipts record 384 updates per arm (768 arm updates). Passes 3–12 average 234.2048 seconds, sample SD 0.8070 seconds, range 232.5797–235.2831 seconds; each pass includes 32 parent visits and 64 independent arm updates. Dividing gives 3.6594 effective loop seconds per arm update, not a timed model-only step. Their timing is captured before saving the pass receipt, and excludes initial all-parent observation preparation, initialization and following scheduled checkpoint writes (producer lines 221–234, 258–274). It includes target reads/construction, input transfer, lightweight guards, both forward/backward/optimizer steps and first-step outer qualification where applicable. Pass 12 is not a scheduled checkpoint (passes 5/10/15/20/25/30).

The first actual shared parent has 1,785,014 nodes, 938 native holds, 13,327 sites, 19,997 scoring centers and 686 queries. The inherited step timer records transition 3.7973 seconds and own-confidence 3.1329 seconds; peak allocated CUDA memory is 14,491,702,784 and 14,496,125,952 bytes respectively, peak reserved 19,893,583,872 bytes. Those single qualified steps are not an ordinary-step benchmark: gradient hooks execute only there, and the timer excludes target construction/transfer and some outer qualification checks. GPU utilization/occupancy snapshots cannot partition these timings or establish progress.

## Exact steady-loop work

| Source | What repeats and why it matters |
|---|---|
| H048 producer 235–245; H044 producer 197–209 | Once per parent/pass, reread authenticated TRAIN truth, decode arrays, reconstruct full FP64 physical mass then copy to FP32, import metric helper source, transfer parent and target. Both arms share that parent/target residency. No target-cache change is proposed under the one-parent target cap. |
| H048 producer 92–127; marker readout 212–250 | All 32 grid-plus-marker bundles charge 9,803,647,386 bytes versus 171,798,691,840-byte grid cap; marker total 894,141,904 versus 2,147,483,648-byte marker cap. Thus all fit the LRUs. Successful steady cache hits use tensor mutation stamps, not repeated payload hashes; repeated flag/bin-feature rebuilding is not supported by these sizes and source. |
| Context factory 103; whole-tree trainer 107–115 | The per-update guard is lightweight lease/title/expiry/logical-job/RSS/CUDA-reservation validation. It does not replay storage journals. Six lightweight guard calls occur per ordinary parent visit: outer/cache plus pre/post each arm. Full storage checks occur at output saves. |
| Flag model 264–271; clade GNN 90–131 | Two complete upward/downward tree sweeps retain all nodes and all 952 haplotype contexts, followed by whole-parent genomic GRU history. Per-level Python loops express tree dependencies. Small parameter count does not describe this activation workload. |
| Whole-tree model 307–341; marker-grid model 239–255; CRF 101–135 | Site encoder has 256-site chunks and activation checkpointing; full-grid decode has 22 query blocks of at most32, each spanning the complete ~20,000-position chain. Balanced scan uses logarithmic tensor stages rather than a Python iteration per grid position. Checkpoint recomputation and chunk transfers are genuine work/memory tradeoffs, not shown to dominate. |

All 32 parents together contain 58,100,493 native nodes, 30,531 native holds, 426,936 sites and 639,908 scoring centers. The CPU bank receipt charges 6,915,631,908 bytes and full-grid CPU features 8,779,537,760 bytes. Reducing the tree/grid, detaching history, freezing the body or sharing learned arm states would change the registered science and is outside this proposal.

## One future numerical-path optimization

The inherited update in `train_continuous_clade_flank_v1.py:218–229` repeatedly obtains CUDA finite-check scalars in Python. Qualification proves 52 trainable parameter tensors. On the all-gradients-present successful path, gradient checks have 52 host boolean reads; parameter checks have52; AdamW moment checks have104, plus52 CPU step-counter checks. Installed pinned AdamW source 44–48,159–177 confirms default noncapturable/unfused step counters are CPU and two moments follow each CUDA parameter. Thus the three repeated finite gates nominally perform208 CUDA boolean reads per arm update, apart from the unchanged objective/loss, clip and logging scalar checks. This counts source-level host reads, not measured stall time. Later absent gradients would reduce the gradient contribution.

A future successor can retain each per-tensor finite predicate on device and obtain one conjunction per CUDA device at each of the same three gates: gradients before clipping/step, parameters after step, and optimizer state after that. Keep FP32 dtype checks and individual CPU state checks, `clip_grad_norm_(..., error_if_nonfinite=True)`, AdamW call/callback order, loss/objective, failure categories, `finally` gradient cleanup, first-step qualification and all guards unchanged. Nominal208 becomes3 finite-result CUDA reads on one device; no speedup claim follows without future approved real-model qualification.

The inert `grouped_finite_update_candidate_v1.py` and tiny synthetic CPU test source were written before root reprioritized mutex work. They are NOT imported, tested, activated, admitted or wired into any live/frozen source. CUDA behavior and real-model parity are unverified. Root explicitly deferred this candidate until the current H048 completion.

## Storage contention follow-up

[MEMO_SEMANTICS.md](MEMO_SEMANTICS.md) explains the current read-only memo, the root-mutation counterexample, and a consumer-only future FLARE route. The concrete current contention takes precedence over optimizing finite-check host reads. No scientific accuracy, model-only bottleneck share, speedup, completed-training state or attribution versus native GPU5 is claimed.
