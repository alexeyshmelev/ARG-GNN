# Readable commands

Run from /workspace-SR003.nfs2/ARG-GNN on anogena. These commands inspect, smoke-test or prepare inert descriptors; they do not launch the benchmark fit.

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .runtime/conda/envs/ancestralpaths_cpu_v1/bin/python3.11 -B -m arggnn.benchmarking.paper_ancestralpaths describe
```

The final CPU and scorer smoke receipts already exist. A separately authorized rerun must use a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .runtime/conda/envs/ancestralpaths_cpu_v1/bin/python3.11 -B -m arggnn.benchmarking.paper_ancestralpaths smoke --output research/benchmark_rebuild_20261024/ancestralpaths/smoke_cpu_new_attempt
```

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .runtime/conda/envs/ancestralpaths_cpu_v1/bin/python3.11 -B -m arggnn.benchmarking.paper_ancestralpaths scoring-smoke --output research/benchmark_rebuild_20261024/ancestralpaths/smoke_scoring_new_attempt
```

ROOT consumes `workflow.make_call` in the admitted campaign runtime. The factory binds its new scientific product directory to `<outer-task-output>/native`. Its fit stage takes the frozen plan and all three TRAIN products; predict takes the final fit receipt and one frozen TEST feature receipt; score takes all predeclared methods/five-parent probability receipts and the separately authenticated TEST target references. Task links are resolved by ROOT's dependency queue before the callable executes. The factory's source closure is pinned and contains no exec/eval/compile source-execution wrapper.
