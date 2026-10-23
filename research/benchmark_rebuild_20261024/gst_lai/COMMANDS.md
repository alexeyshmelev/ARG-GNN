# Readable commands

Use the project interpreter on anogena from /workspace-SR003.nfs2/ARG-GNN.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python3.11 -B -m arggnn.benchmarking.paper_gst_lai describe
```

The final synthetic receipt already exists. A separately authorized rerun uses a fresh output directory and a finite timeout:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 120 .runtime/env/bin/python3.11 -B -m arggnn.benchmarking.paper_gst_lai smoke --output research/benchmark_rebuild_20261024/gst_lai/smoke_cpu_new_attempt
```

ROOT schedules the ready DAG through `arggnn.benchmarking.paper_campaign`. The factories are ordinary readable Python modules. Their scientific products live under each task's new `native` child directory. Full fit/predict functions require ROOT's current execution and resource callbacks. The CLI exposes no unbounded scientific fit or prediction route.
