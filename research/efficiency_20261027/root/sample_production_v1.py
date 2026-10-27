"""Observe the admitted TreeLoom successor without controlling any processes."""
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
STATE = ROOT / '.runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json'
OUT = ROOT / 'research/efficiency_20261027/root/production_attempt7'
NATIVE = ROOT / 'experiments/results/paper_american_20261024/treeloom_fit/native/exact_preprocessing_from_update268_explicit512_attempt7'
TASK = 'paper_american_treeloom_fit'
PID = 1166415
START = '574375145'
GPU_UUID = 'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786'


def identity_matches():
    try:
        fields = Path('/proc/%d/stat' % PID).read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == START and fields[0] != 'Z'


def main():
    OUT.mkdir(exist_ok=False)
    state = json.loads(STATE.read_text())
    worker = state['tasks'][TASK]
    if worker.get('pid') != PID or worker.get('attempt') != 7 or worker.get('physical_gpu') != 3 or not identity_matches():
        raise RuntimeError('Expected live owned attempt7 is required')
    request = dict(task=TASK, pid=PID, process_start_ticks=START, attempt=7,
                   interval_seconds=2, maximum_seconds=2700, stop_at_update=270,
                   scope='Device-wide utilization plus owned task progress. No process control.',
                   started_unix=time.time())
    (OUT / 'request.json').write_text(json.dumps(request, indent=2)+'\n')
    utilization = []
    errors = []
    last_update = 268
    reason = 'maximum_duration'
    tick0 = time.monotonic()
    with (OUT / 'samples.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['unix','GPU_utilization_percent','GPU_used_MiB','power_W','completed_updates','stage'])
        while time.monotonic()-tick0 < 2700:
            tick = time.monotonic()
            if not identity_matches():
                reason = 'expected_process_no_longer_live'
                break
            checkpoint = NATIVE / 'latest_checkpoint.json'
            if checkpoint.exists():
                last_update = json.loads(checkpoint.read_text())['complete_parent_updates']
            stage = 'preparation'
            progress = NATIVE / 'training_live.json'
            if progress.exists():
                live = json.loads(progress.read_text())
                stage = live.get('latest', {}).get('last_mark', 'training')
            try:
                raw = subprocess.check_output(['nvidia-smi','--id=3',
                    '--query-gpu=uuid,utilization.gpu,memory.used,power.draw',
                    '--format=csv,noheader,nounits'], text=True, timeout=10)
                fields = [v.strip() for v in next(csv.reader(raw.splitlines()))]
                if fields[0] != GPU_UUID:
                    raise ValueError('GPU identity changed')
                value = float(fields[1])
                utilization.append(value)
                writer.writerow([time.time(),value,fields[2],fields[3],last_update,stage])
                stream.flush()
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                errors.append(dict(unix=time.time(),type=type(exc).__name__,message=str(exc)))
                if len(errors) >= 5:
                    reason = 'telemetry_error_limit'
                    break
            if last_update >= 270:
                reason = 'two_new_optimizer_updates_completed'
                break
            time.sleep(max(0,2-(time.monotonic()-tick)))
    result = dict(request=request,finished_unix=time.time(),terminal_reason=reason,
                  last_completed_update=last_update,samples=len(utilization),errors=errors,
                  mean_GPU_utilization=statistics.mean(utilization) if utilization else None,
                  fraction_samples_at_least80=sum(v>=80 for v in utilization)/len(utilization) if utilization else None,
                  limitations='Begins after worker startup. Device counters do not independently attribute activity to this process. Cache rebuild is included where observed.')
    (OUT / 'completed.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
