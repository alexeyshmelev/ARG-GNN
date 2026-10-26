# Preprocessing efficiency audit, 26 October 2026

The current pipeline is not fully efficient. Multiple workers are present, but several expensive preparation stages remain serial. These stages explain the observed idle GPUs during this restart. This audit inspected the live task configuration and its actual source dependencies and sampled task-owned CPU threads. It did not change, stop or relaunch any job.

## Observed execution

| Workstream | Existing parallelism | Remaining bottleneck |
|---|---|---|
| American TreeLoom | One background producer, one parent of lookahead, and an in-memory compact cache covering the twenty parents | The expensive complete-parent preparation runs in one producer thread. Torch CPU and interop threads are explicitly set to one. Tree enumeration, forest indexing and clade bookkeeping contain Python loops. The compact cache does not survive a process restart. |
| European TreeLoom prediction | Eight Torch CPU threads, visibly active during observed-bank encoding | VCF parsing is serial. Each parent is fully prepared and then predicted before preparation of the next parent begins. No overlap between preparation of the next parent and current prediction is implemented in this route. |
| GST-LAI-8 | Four persistent DataLoader workers, prefetch factor one, pinned memory and nonblocking transfers during the training stage | All three TRAIN observations are loaded serially before the DataLoader is constructed. The loader workers therefore cannot accelerate the initial VCF joins. |
| Native ARGformer | Eight persistent workers per configured loader, prefetch factor two | Training is actively using the GPU. Short GPU utilization dips are visible, but this sample does not establish their cause or the average training duty cycle. |

An initial 15.4-second sample measured approximately 1.00 CPU core for American TreeLoom, 1.00 for GST and 7.75 for European TreeLoom. GPUs 3 and 5 were at zero utilization in all five snapshots. GPU 4 alternated between 94 percent and zero. The saved follow-up sample provides independent task metadata and source hashes. These short samples are not campaign utilization averages.

The host had over 1.7 TB available RAM in the initial check. GPUs 3 and 5 held only about 505 MiB each. The observed wait is CPU preparation, not evidence of insufficient GPU memory. Wider preparation concurrency still needs bounded memory allocation to avoid duplicating expanded full-parent arrays.

## Concrete execution improvements

1. Replace per-sample Python VCF parsing with a compiled or vectorized parser. Preserve sample order, phasing, missing and multiallelic calls, original coordinates and the exact retained-site join. Share compact observed inputs where storage permits. Do not assume that all expanded banks fit within the 1 TB project limit.
2. Move independent tree feature extraction and forest bookkeeping into vectorized or compiled kernels, or a bounded process pool. Merely raising `torch.set_num_threads` does not parallelize Python loops. A Python thread pool also does not solve CPU-bound Python work constrained by the GIL.
3. Overlap independent observed-parent preparation with GPU execution. Deliver prepared parents in the existing order. Use compact shared arrays or bounded transfers rather than copying entire expanded banks between workers. The first parent after restart still requires an initial preparation phase.
4. Preserve the ordered merge of clade continuity across adjacent trees. Preserve all nodes, sites, focal identities, class order, loss weighting, recurrent state and optimizer boundaries. Independent observation work can run ahead without reordering training or exposing query truth.

These are open execution improvements, not changes already deployed. The running benchmark remains on its pinned sources and retains its model, data and training budget. A replacement needs a focused output-equivalence check and a complete-parent time/memory measurement before integration. No new simulation model or reduced data workload is necessary to address the identified serial preparation.

Evidence: [bounded live sample](sample.json), [readable sampling source](sample.py). The sample reads task metadata and owned processes, without a disk scan or model execution.
