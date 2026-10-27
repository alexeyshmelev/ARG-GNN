"""Summarize completed diagnostic stages without attributing shared GPU duty."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')


def file_ref(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    root = (ROOT / args.profile).resolve()
    if not root.is_relative_to(ROOT / 'research/efficiency_20261027/root/runs'):
        raise ValueError('Owned profile directory required')
    terminal = json.loads((root / 'actual_wait.json').read_text())
    telemetry = json.loads((root / 'telemetry_completion_v2.json').read_text())
    if not terminal.get('actual_wait') or not terminal.get('owned_group_absent'):
        raise ValueError('Exact process completion required')
    if telemetry['terminal'] not in ('profile_terminal_receipt', 'profile_terminal_status'):
        raise ValueError('Incomplete telemetry sampling')
    request = json.loads((root / 'request.json').read_text())
    gpu = request['physical_gpu']
    with (root / 'device_telemetry_v2.csv').open() as handle:
        samples = [dict(time=float(row['observed_unix']), utilization=float(row['utilization_percent']),
                        power=float(row['power_W'])) for row in csv.DictReader(handle) if int(row['gpu']) == gpu]
    records = [json.loads(line) for line in (root / 'data/stage_clock.jsonl').read_text().splitlines()]
    process_rows = [json.loads(line) for line in (root / 'process_telemetry_v2.jsonl').read_text().splitlines()]
    pid = str(json.loads((root / 'launch.json').read_text())['pid'])
    peers = sorted({row[1] for sample in process_rows for row in sample['rows']
                    if row[0] == request['gpu_uuid'] and row[1] != pid})
    stages = []
    for stage in records:
        selected = [row for row in samples if stage['wall_start_unix'] <= row['time'] < stage['wall_end_unix']]
        values = [row['utilization'] for row in selected]
        stages.append(dict(**stage, sampled_seconds=len(selected),
            mean_shared_device_utilization=None if not values else statistics.mean(values),
            median_shared_device_utilization=None if not values else statistics.median(values),
            maximum_shared_device_utilization=None if not values else max(values),
            fraction_samples_at_least80=None if not values else sum(v >= 80 for v in values) / len(values)))
    result = dict(schema='TreeLoom_shared_device_stage_summary_v1', physical_gpu=gpu,
        profile_pid=int(pid), same_device_peer_pids=peers, stages=stages,
        device_sample_count=len(samples), sampler=telemetry, actual_returncode=terminal['actual_returncode'],
        caveats=['Utilization is the device-wide NVIDIA counter, not per-process SM occupancy or useful-work attribution.',
                 'One-second samples only approximate phase coverage. Short stages can have zero samples.',
                 'No observed utilization value alone establishes predictive quality or throughput.'],
        source_refs=[file_ref(root / name) for name in ('actual_wait.json', 'telemetry_completion_v2.json',
            'device_telemetry_v2.csv', 'process_telemetry_v2.jsonl', 'data/stage_clock.jsonl')])
    with (root / 'telemetry_stage_summary.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status='completed', profile=args.profile, stages=len(stages), peers=peers)))


if __name__ == '__main__':
    main()
