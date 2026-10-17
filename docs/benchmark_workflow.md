# Benchmark workflow

Benchmark execution is paused by the user. Code preparation and small synthetic smoke tests are allowed. Training, simulation, prediction, real-data application and scoring resume only after a new explicit go-ahead.

## Entry points

From the dedicated server repository:

```bash
source scripts/project_env.sh
export PYTHONPATH=.:src:scripts
.runtime/env/bin/python -B -s -m arggnn.benchmarking methods
.runtime/env/bin/python -B -s -m arggnn.benchmarking paper-recipes
.runtime/env/bin/python -B -s -m arggnn.benchmarking paper-plan ancestralpaths_european
.runtime/env/bin/python -B -s -m arggnn.benchmarking status
```

These commands read small configuration and source metadata. They do not open scientific payloads, import Torch, inspect GPUs, reserve storage or launch jobs. A plan can be valid as a description while its preflight correctly reports missing artifacts and approval.

## Layout

| Location | Responsibility |
|---|---|
| `src/arggnn/benchmarking/contracts.py` | Protocol, provenance, common support and approval contracts |
| `methods.py`, `treeloom.py` | Actual method adapters and the selected model API |
| `paired.py`, `argformer_recipe.py`, `european_recipes.py` | Source-backed paper recipes and one shared observed parent per paired comparison |
| `targets.py` | Sampled-example TRAIN targets bound to the parent and sampling registry |
| `data.py` | Ordered observed-parent descriptors and CPU data-loading policy |
| `scheduler.py`, `idle_preflight.py` | Persistent single-GPU queue and cheap blocked-resource checks |
| `metrics.py` | Common ancestry metrics and paired-support validation |
| `configs/benchmarks/` | Method registry, separate biological protocols and execution hold |
| `research/reviews/` | Immutable numerical implementations, original guarded admissions and historical evidence |

Historical files stay in place because existing evidence and checkpoints bind their exact paths and hashes. The new interface consolidates their entry points without moving live dependencies or changing competitor algorithms.

## Source and environment prerequisites

All code, environments, caches and future outputs stay in the dedicated NFS2 repository. The existing project environment is `.runtime/env`; native methods retain their separately qualified environments. This refactor installs no packages and changes no author source. Runtime author checkouts are intentionally excluded from Git. Their repository URLs and exact revisions are recorded in `configs/benchmarks/methods.json`, and every registered stage verifies source hashes before resolution. An absent checkout is a visible prerequisite, never a reason to substitute an implementation.

## Scientific comparison

A protocol fixes independent simulation parents before sampling trees, windows or focal targets. Each method receives its native representation of the same observations, reference/query split and evaluation support. Different representations are expected. Altered observations, hidden target exclusions and missing predictions are rejected.

American three-source ancestry and European four-source ancestry are separate tasks. Native AncestralPaths six-path probabilities can be collapsed to four sources only through an explicit partition. That is a derived endpoint. It is not the native six-path task.

Record each method's training regime, author source revision, settings, checkpoint selection and any adaptation. Native ARGformer learns from observed data per ARG. TreeLoom amortizes supervised learning over training simulations. This is a pipeline comparison. It does not isolate architecture as the only cause of a difference.

[Paper recipes](paper_benchmark_recipes.md) are separate from the old reconstruction protocols. Native AncestralPaths six-path targets and sampled-example supervision have maintained interfaces. The historical exact-span loss remains confined to protocols that specify it. A native TRAIN sampler may use declared class quotas. Evaluation support must be fixed without query truth and before predictions.

A reconstructed common-data experiment is labelled as such. Missing publication parameters block an exact-paper reproduction claim. The current ARGMix release differs from its original preprint. Released AncestralPaths code differs from the paper's stated network. These differences remain visible.

Native hard calls are scored as supplied. They are not replaced with posterior argmax. Native posterior matrices must satisfy the simplex before any class collapse. Raw zero true-class probabilities give infinite NLL rather than an undocumented floor. Accuracy, class precision/recall/F1 and macro F1 use the declared support. Report equal-parent means and sample SD across independent simulations, separately for each class and separately from variation across optimizer seeds. Scoring requires target-support fingerprints frozen from the focal registries before prediction. Paired checks require every method and parent in the declared protocol split, preventing shared omissions from appearing valid. Empirical data have no simulated ancestry truth and receive descriptive biological analyses instead of invented accuracy.

## Efficient execution after approval

Only physical GPU 0 is eligible. The controller does not import Torch or hold a GPU context. It dispatches one admitted GPU job at a time, with bounded CPU concurrency for preparation. Existing scientific supervisors still enforce their physical leases, memory, storage, clocks, process title and owned cleanup.

Check cheap CPU-slot and RSS availability before expensive accounting capture. Keep authoritative fresh checks at admission. Data-loader workers return small immutable descriptors. They do not duplicate a complete parent tensor bank or create CUDA contexts.

TreeLoom's corrected implementation uses direct fast observed preparation, bounded GPU history windows and a compact five-second stage profiler. Its small FP32 calculation check passed. Full-size GPU fit and throughput remain unmeasured and need qualification after approval. A partial cache is useful only for fixed parents retained across passes. Never claim all parents fit a cache without measuring them.

## Release and recovery

The canonical `configs/benchmarks/execution_policy.json` is held. There is no command that approves it automatically.

The scientific enqueue boundary is deliberately closed in this source-only release. Native TreeLoom and competitor admissions have different schemas. A generic normalized packet cannot establish their provenance. Family-specific adapters must validate the real guarded product, scientific input, source/review bindings, resource limits and genuine preparation-wait lineage before they are released. New admissions themselves wait for user approval.

A future normalized packet is a ROOT-attested trust boundary. JSON metadata and matching hashes do not cryptographically prove an OS exit. An actual wait status must be an integer, so Boolean false must never be accepted as zero. The original guarded worker retains its runtime checks. The tested scheduler uses injected fake admission validators in smoke tests. An inert method specification or historical receipt cannot enter the production queue today.

The separately approved canonical policy must list the exact admitted job/input/receipt identities and a finite expiry. `run --execute` rechecks it before each process. Default `run` only plans. The CLI uses one fixed queue and logical controller lock. Existing physical lease ownership remains with the original guarded worker.

The scheduler records a process start before dispatch and uses the actual OS wait. File presence never proves success. Failed attempts are not retried automatically. An uncertain earlier start blocks its resource lane until ROOT resolves it from real process and wait evidence. Preserve all earlier artifacts and failures.

The old paused controllers and admissions are historical records. Reusing a recorded source does not reactivate an old execution authority. Fresh approval, source binding and current admissions are required.

## Smoke tests

Use the dedicated environment, with GPU visibility disabled and project-local temporary files:

```bash
source scripts/project_env.sh
export PYTHONPATH=.:src:scripts
export CUDA_VISIBLE_DEVICES=""
.runtime/env/bin/python -B -s -m unittest discover -s tests -p "test_benchmark_*.py"
```

Also run `test_paper_*.py` and `test_european_recipes.py` with the same environment. The latter verifies the configurable generator through an in-memory fake simulator and writer. The sampled-loss smoke test imports CPU Torch for a tiny analytic loss/gradient check. Planning imports no scientific runtime.

The suite uses tiny synthetic labels and metadata plus fake process runners. It does not perform a benchmark, train a model, read real genotype/truth payloads or allocate CUDA memory.

Last verification, 18 October 2026: 137 focused tests passed in 4.15 seconds.
The final shared-input integration adjustment then passed all 14 affected tests.
All three paper-plan commands and the status command passed without importing
Torch, NumPy, msprime or tskit. The tiny sampled-loss test used CPU Torch and
left CUDA uninitialized. Independent source review passed for code preparation.
No benchmark queue exists and execution approval remains false.
