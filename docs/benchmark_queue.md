# Executable benchmark queue

The maintained workflow contains 325 concrete stages. The exact source-pinned
plan is [executable_workflow_20261018_v2.json](../configs/benchmarks/executable_workflow_20261018_v2.json).
This is separate from the earlier conceptual campaign graph. Execution requires
that exact plan reference in the canonical policy. The user has authorized execution.

[Current execution status](../research/benchmark_execution_current.json) and
[current matched results](../research/current_matched_benchmark_results.json)
are generated from actual worker records. Pending stages are not reported as
running or completed. No new common-data comparison has finished at release.

| Comparison | Shared data | Training and evaluation |
|---|---|---|
| TreeLoom, native ARGformer, native FLARE | 35 independent 30 Mb American simulations: 20 TRAIN, 5 validation, 10 TEST | TreeLoom: 30 passes, 600 complete-parent updates. ARGformer: native five-epoch MLM and one-epoch contrastive fit separately for every evaluation ARG, then all embeddings and native retrieval. All three methods use the same 10,000 focal targets per evaluation parent. |
| TreeLoom and released AncestralPaths | Three TRAIN and five TEST European sequences of 200 Mb | Native six-path task, 50,000 sampled TRAIN examples in total and 10,000 TEST examples per parent. TreeLoom: 30 passes, 90 complete-parent updates. Released native Dense6: five epochs. Cohorts reported separately. |
| TreeLoom and released ARGMix | Same European observed parents, plus one independent validation parent | Separate four-source TreeLoom fit. Released ARGMix model, original training settings, complete chunks and native validation selection. Both methods use the same observed-only interior-tree target registry, reflecting the release's exclusion of boundary trees. |
| Current TreeLoom application | Three existing 10 Mb Danish observed windows | Apply the new four-source checkpoint without further optimization. Produce interval predictions and descriptive ancestry summaries. No simulated accuracy is assigned to real data. |

The American and European CPU workstreams are interleaved. Up to eight CPU
stages can run subject to actual RAM and storage capacity. GPU stages use
**physical GPU 0 only**, with the requested process title. The existing native
ARGformer VAL000 training and embedding/retrieval chain keeps its current lease
and original limits. The new controller waits for that lease when necessary.

Six existing American TRAIN generation/inference pipelines completed and are
adopted only after their real successful process waits. Their data are reused,
not regenerated. The other 29 American parents and nine European parents are
prepared by the new workflow.

## Scientific interpretation

Each comparison uses the same observed parent, reference/query identities and
frozen scoring support across methods. Method-specific tensor, VCF and path
representations do not change the common observations. Native three-source,
four-source and six-path tasks remain separate. All registered TEST predictions
must finish before either European endpoint opens TEST truth. The American
scorer waits for all 45 method-parent predictions.

The available ARGMix release differs from the April preprint and its exact old
archive is unavailable. The released AncestralPaths trainer is Dense6, rather
than a recovered implementation of the paper's stated CNN. Native ARGformer
uses the released settings documented in [recipe notes](paper_benchmark_recipes.md).
These are disclosed comparisons on shared paper-derived simulations, not claims
that the original printed tables have been reproduced.

GST is a graph-set architecture reference, not a native LAI algorithm. A new
cross-domain GST benchmark, new OOD simulation protocols and empirical PCA are
not part of this 325-stage LAI queue. They must not be described as completed.

## Operation and recovery

The controller records dependencies, actual starts, direct waits, successful
outputs and failures under `.runtime/benchmarking/workflow_20261018_v2`. All new
scientific artifacts stay under
`experiments/results/benchmark_campaign_20261018_workflow/release_v2`.

The controller reserves active output caps against 980 GB using the initial
actual project measurement. It measures completed scopes, not the whole project
on each poll. Original files, failures and pinned source remain in place. Only
successful new ARGformer scratch can be retired after complete native retrieval
and actual process cleanup. The four ARGMix training/validation archives are
read directly without a duplicate merged training archive.

A failed stage blocks its dependents and leaves independent eligible work
running. No automatic scientific retry, cropping, reduced training budget or
hidden data fallback is permitted. An interrupted worker with unknown cleanup
blocks reuse until its actual state is resolved.

The README publication runs outside the scheduler loop. Routine Git updates are
throttled to 30 minutes, with failures, new scored comparisons and completion
published promptly. Reporting errors cannot terminate training. All Git runs
on anogena. The queue remains active after the controlling chat turn ends.

Results report mean and sample SD across independent simulation parents,
per-population or per-cohort metrics, and paired differences on identical
support. Probability metrics are reported only for native probabilities.
Retrieval similarity is not relabelled as an ancestry posterior.

The first controller exposed a European interpreter-symlink path error before those scientific stages started. Release v2 uses the same binaries by canonical names. Original failures and explicitly interrupted CPU attempts are retained. The existing native ARGformer run and its continuation were not stopped.
