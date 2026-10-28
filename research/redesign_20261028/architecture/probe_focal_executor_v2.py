"""Small CPU wall/RSS comparison on four actual registered observation packets.

A fresh process runs each version/mode. One warmup plus three measured label-
free forward/backward probes; no optimizer, no ancestry loss, and no CUDA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time
import torch
from . import window_model_frozen_v1 as old
from . import window_model_focal_executor_v2 as new
from .test_focal_executor_v2 import four_real_packets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=('v1', 'v2'), required=True)
    parser.add_argument('--mode', choices=('structure', 'leaf_only', 'no_cross'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(711)
    root = Path(__file__).resolve().parent
    start = time.perf_counter()
    arrays, paths = four_real_packets()
    read_seconds = time.perf_counter() - start
    start = time.perf_counter()
    source = old.prepare_observations(arrays)
    source_prepare_seconds = time.perf_counter() - start
    start = time.perf_counter()
    batch = source if args.version == 'v1' else new.prepare_executor(source)
    plan_prepare_seconds = time.perf_counter() - start
    module = old if args.version == 'v1' else new
    model = module.WindowTreeLoom(width=32)
    timings = []
    for index in range(4):
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        output = model(batch, mode=args.mode)
        forward_seconds = time.perf_counter() - start
        start = time.perf_counter()
        output.square().mean().backward()
        backward_seconds = time.perf_counter() - start
        assert torch.isfinite(output).all()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        if index:
            timings.append(dict(forward_seconds=forward_seconds, backward_seconds=backward_seconds,
                                total_seconds=forward_seconds + backward_seconds))
    report = dict(schema='focal_executor_four_real_packets_CPU_probe_v1', version=args.version, mode=args.mode,
                  source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  CPU_threads=2, width=32, read_seconds=read_seconds,
                  source_prepare_seconds=source_prepare_seconds, plan_prepare_seconds=plan_prepare_seconds,
                  source_trees=len(source.leaves), focal_trees=len(torch.unique(source.focal_tree)),
                  source_nodes=len(source.parent), samples_per_tree=source.leaves.shape[1],
                  site_occurrences=len(source.site_tree), focal_rows=len(source.focal_tree),
                  packet_paths=[str(path.relative_to(root.parent)) for path in paths],
                  packet_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
                  repetitions=3, warmups=1, timings=timings,
                  median_forward_seconds=statistics.median(row['forward_seconds'] for row in timings),
                  median_backward_seconds=statistics.median(row['backward_seconds'] for row in timings),
                  median_forward_backward_seconds=statistics.median(row['total_seconds'] for row in timings),
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  optimizer_updates=0, labels_used=False, ancestry_fitting=False,
                  CUDA_initialized=torch.cuda.is_initialized(), performance_claim='small CPU measurement, not GPU or broad throughput evidence')
    assert not report['CUDA_initialized']
    (root / f'FOCAL_EXECUTOR_PROBE_{args.version}_{args.mode}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
