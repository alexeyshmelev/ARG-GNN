"""Read-only device telemetry with explicit startup and query failure receipts.

GPU utilization is a shared-device counter. Process rows record memory only.
No existing telemetry is overwritten and no processes are signaled.
"""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
UUIDS = {
    'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786',
    'GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71',
    'GPU-3db85505-d061-e45d-d154-e7e8c1cad497',
}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    directory = (ROOT / args.profile).resolve()
    if not directory.is_relative_to(ROOT / 'research/efficiency_20261027/root/runs'):
        raise ValueError('Owned efficiency directory required')
    if not (directory / 'request.json').is_file():
        raise ValueError('Existing explicit profile request required')
    counters = dict(device_samples=0, process_samples=0, startup_waits=0, errors=0)
    started = time.time()
    terminal = 'maximum_duration'
    with (directory / 'device_telemetry_v2.csv').open('x', newline='') as stream, (directory / 'process_telemetry_v2.jsonl').open('x') as processes, (directory / 'telemetry_errors_v2.jsonl').open('x') as errors:
        writer = csv.writer(stream)
        writer.writerow(['observed_unix', 'gpu', 'uuid', 'utilization_percent', 'used_MiB', 'power_W'])
        stream.flush()
        for index in range(14400):
            tick = time.monotonic()
            if (directory / 'actual_wait.json').exists() or (directory / 'failure.json').exists():
                terminal = 'profile_terminal_receipt'
                break
            try:
                status = json.loads((directory / 'status.json').read_text())
            except FileNotFoundError:
                counters['startup_waits'] += 1
                time.sleep(1)
                continue
            if status.get('status') in ('finished', 'failed'):
                terminal = 'profile_terminal_status'
                break
            try:
                raw = subprocess.check_output([
                    'nvidia-smi', '--id=3,4,5',
                    '--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw',
                    '--format=csv,noheader,nounits'], text=True, timeout=15, stderr=subprocess.PIPE)
                now = time.time()
                for row in csv.reader(raw.splitlines()):
                    fields = [field.strip() for field in row]
                    if len(fields) != 5 or fields[1] not in UUIDS:
                        raise ValueError('Unexpected GPU telemetry row')
                    writer.writerow([now] + fields)
                stream.flush()
                counters['device_samples'] += 1
                if index % 5 == 0:
                    raw = subprocess.check_output([
                        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                        '--format=csv,noheader,nounits'], text=True, timeout=15, stderr=subprocess.PIPE)
                    rows = [[field.strip() for field in row] for row in csv.reader(raw.splitlines())]
                    rows = [row for row in rows if len(row) == 3 and row[0] in UUIDS]
                    processes.write(json.dumps(dict(observed_unix=time.time(), rows=rows)) + '\n')
                    processes.flush()
                    counters['process_samples'] += 1
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                row = dict(observed_unix=time.time(), error_type=type(exc).__name__, error=str(exc))
                errors.write(json.dumps(row) + '\n')
                errors.flush()
                counters['errors'] += 1
                print(json.dumps(row), flush=True)
                if counters['errors'] >= 10:
                    terminal = 'query_error_limit'
                    break
            time.sleep(max(0., 1. - (time.monotonic() - tick)))
    result = dict(status='completed' if counters['device_samples'] else 'no_samples', terminal=terminal,
                  counters=counters, started_unix=started, finished_unix=time.time(),
                  scope='Shared-device utilization. Per-process GPU memory does not attribute utilization.')
    with (directory / 'telemetry_completion_v2.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
