# Benchmark campaign

The user released execution on 18 October 2026. Only physical GPU 0 is eligible.
The complete [campaign graph](../configs/benchmarks/campaign.json) lists planned
stages and dependencies. A stage marked pending is not a running job.
Only the exact jobs in the [execution policy](../configs/benchmarks/execution_policy.json)
enter the persistent execution queue.

| Workstream | Queue contents | Current status |
|---|---|---|
| Existing American comparison | Complete native ARGformer contrastive training, embeddings and retrieval, then the full five-parent comparison with TreeLoom and FLARE | Full contrastive training observed on GPU 0. Four embedding stages and native retrieval are armed as a finite dependent chain. An initial OrderedDict wrapper failure is preserved. This is exposed validation data. |
| Fresh American benchmark | 20 TRAIN, 5 VAL, 10 untouched TEST simulations, shared inference and support, TreeLoom, native ARGformer and FLARE | Simulation conventions and downstream integrations remain explicit pending stages. Native ARGformer fits separately on each evaluation ARG's observations. |
| AncestralPaths benchmark | Three TRAIN and five TEST sequences of 200 Mb, complete age roster, native six-path endpoint, TreeLoom and released AncestralPaths | Configurable generator, Relate preparation and streamed native features implemented. Sampling, complete TreeLoom preparation and bounded European execution remain unfinished. |
| ARGMix benchmark | Five TRAIN, one VAL and three IID TEST parents, all four admixed cohort reports, available author methods | Missing age/map/error/sampling settings and unavailable preprint implementation remain recorded. Five OOD parents require a separate perturbation recipe. |
| Additional evidence | GST architecture control, real-data inference and descriptive PCA | Pending integrations and dataset/protocol selection. Empirical accuracy is not invented. |

## First full GPU stage

The first attempt failed before its first update because our wrapper required an exact dict for the saved weights. The corrected attempt accepts the native OrderedDict without changing tensor values or author code.

The current job uses the unchanged ARGformer author contrastive trainer on the
existing 30 Mb american_c3_val_000 observations. The native corpus contains
8,748,479 training examples and 972,757 validation examples. Its complete budget
is one epoch, 2,135 updates, batch 4,096 and seed 36. The previously accepted MLM
checkpoint initializes the representation. Completion must be confirmed from
the actual saved Composer and optimizer counters.

This run uses released temperature 0.30 and the released uniform class-weight
fallback. The paper describes temperature 0.05 and inverse-frequency weighting.
The initializer also follows released masking behavior. These differences stay
explicit. This is an available-release baseline stage, not an exact paper-table
reproduction, untouched TEST result or completed ancestry comparison.

The worker directly owns the canonical GPU0 lease, sets the requested process
title before CUDA, and enforces 74 GiB allocator, 128 GiB RSS, 16 GiB output and
24-hour wall limits. It checks actual project storage before dispatch against
980 GB, preserving 20 GB below the human limit. It monitors only its own output
scope during training. There is no automatic failed-job replay.

The runtime queue is .runtime/benchmarking/queue.json, and controller output is
.runtime/benchmarking/controller.log. Scientific logs and final artifacts are
under experiments/results/benchmark_campaign_20261018/. These runtime files
remain on the server. Compact verified progress and results are committed to GitHub.

## Remaining release work

[Paper-specific differences](paper_benchmark_recipes.md) remain visible.
[European implementation gaps](benchmark_european_remaining.md) list concrete
preparation and memory requirements. Dependency barriers prevent a pending
model fit or incomplete set of predictions from becoming a score. All TEST
methods must finish the same frozen support before truth is released.

The selected TreeLoom architecture remains fixed. [Model history](model_history.md)
records prior findings so this campaign does not repeat answered development
hypotheses. Original files, partial runs and failed process waits are preserved.

## Reports produced by the campaign

Every shared simulation contributes one paired result per method. Report means and sample standard deviations across independent simulations, with per-source precision, recall, F1 and confusion matrices on identical targets. American ancestry classes are EUR, EAS and AFR. The ARGMix tables separately show Present, PostNeolithic, NeolithicFarmer and Yamnaya cohorts. The AncestralPaths tables preserve its six native path classes and separately show present-day, Bronze Age, BAA, Neo and Yam cohorts.

Report probability scores only when a method supplies native probabilities. Retrieval similarity is not an ancestry posterior. Empirical analyses provide biological descriptions and PCA, without invented ancestry accuracy.

## Active native continuation

The approved [native chain](../configs/benchmarks/native_argformer_val000_chain.json) waits for the existing full contrastive job. It then runs query TRAIN embeddings, query VAL embeddings, reference VAL embeddings, reference TRAIN embeddings and original top-1 retrieval in that order. Each stage uses all of its native rows. Actual training completion and process success are required before the next stage is constructed. A failed stage stops the chain.

The initial complete storage measurement was 481.915 GB. Current training and downstream output reservations total 92 GiB, below the 980 GB operational ceiling. Downstream full embedding outputs need roughly 50 GB. The chain reserves 76 GiB for these stages plus 4 MiB for its records, and preserves any separate CPU reservation. Only GPU0 is used.

[TreeLoom training readiness](treeloom_training_queue_readiness.md) lists existing observations and remaining integration. Planned stages are not reported as executed benchmarks.
