"""One bounded, read-only sample of task-owned CPU threads and GPU activity."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = ROOT / 'research/preprocessing_efficiency_20261026'
state = json.loads((ROOT / '.runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json').read_text())
plan = json.loads((ROOT / 'configs/benchmarks/paper_campaign_resume_20261026_v1.json').read_text())
tasks = {t['id']: t for t in plan['tasks']}
rows = {key: row for key, row in state['tasks'].items() if row['status'] == 'running'}
clock = os.sysconf('SC_CLK_TCK')

def capture(pid):
    todo, seen, ticks = [pid], set(), {}
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            for thread in (Path('/proc') / str(current) / 'task').iterdir():
                fields = (thread / 'stat').read_text().rsplit(')', 1)[1].split()
                ticks[int(thread.name)] = int(fields[11]) + int(fields[12])
                todo.extend(int(c) for c in (thread / 'children').read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            pass
    return ticks

before = {key: capture(int(row['pid'])) for key, row in rows.items()}
start = time.monotonic()
gpus = []
for _ in range(5):
    gpus.append(subprocess.check_output([
        'nvidia-smi', '--id=3,4,5', '--query-gpu=index,utilization.gpu,memory.used',
        '--format=csv,noheader'], text=True).strip())
    time.sleep(3)
elapsed = time.monotonic() - start
result = {}
for key, row in rows.items():
    after = capture(int(row['pid']))
    usage = [(value - before[key][tid]) / clock / elapsed
             for tid, value in after.items() if tid in before[key]]
    progress = ROOT / tasks[key]['output_directory'] / 'resource_progress.json'
    result[key] = dict(pid=row['pid'], physical_gpu=row['physical_gpu'],
        mean_cpu_cores_used=round(sum(usage), 3), threads_observed=len(after),
        threads_above_0_1_core=sum(value > .1 for value in usage),
        largest_thread_core_fractions=sorted(usage, reverse=True)[:8],
        last_stage_receipt=json.loads(progress.read_text()) if progress.exists() else None)
source_paths = [
    'research/repairs/TreeLoom_lazy_compact_prefetch_20261024_v1/training.py',
    'research/repairs/TreeLoom_lazy_compact_prefetch_20261024_v1/pool.py',
    'research/reviews/TreeLoom_preparation_reuse_source_20261017_v1/optimized_preparation_v1.py',
    'research/reviews/H060_isolated_hold_compatibility_source_20261017_v1/episode_preparation_v1.py',
    'src/arggnn/benchmarking/european_treeloom.py',
    'src/arggnn/benchmarking/european_training.py',
    'src/arggnn/benchmarking/paper_gst_lai_runtime_v2/dataset.py',
    'research/repairs/benchmark_recovery_20261025/gst/workflow.py',
    'research/repairs/benchmark_recovery_20261025/european/ap_prediction_chunked_gru.py',
    'experiments/results/paper_american_20261024/argformer_american_validation_001/mlm/native/resume_20261026_batch119500/resolved_config.yaml']
record = dict(verified_moscow=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
    sample_seconds=elapsed, controller_counts=dict(Counter(r['status'] for r in state['tasks'].values())),
    workers=result, gpu_samples=gpus, short_sample_not_campaign_average=True,
    host_memory=[line for line in Path('/proc/meminfo').read_text().splitlines()
                 if line.startswith(('MemTotal:', 'MemAvailable:', 'SwapFree:'))],
    source_refs=[dict(path=p, sha256=hashlib.sha256((ROOT / p).read_bytes()).hexdigest())
                 for p in source_paths], live_jobs_modified=False)
(OUT / 'sample.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: v['mean_cpu_cores_used'] for k, v in result.items()}))
print(json.dumps(gpus))
