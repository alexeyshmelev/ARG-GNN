"""Read-only bounded production observation with explicit shared-GPU detection.

Device utilization is not a measure of FLOP efficiency or model accuracy.
Exclusive samples require the expected TreeLoom PID to be the only reported
compute client. Attribution remains device-counter inference, not kernel tracing.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
STATE = ROOT / '.runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json'
TASK = 'paper_american_treeloom_fit'
UUIDS = {3:'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786',
         4:'GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71',
         5:'GPU-3db85505-d061-e45d-d154-e7e8c1cad497'}


def contained(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Project-contained path required')
    return path


def identity(pid):
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def small_json(path):
    if path.stat().st_size > 2**20:
        raise ValueError('Bounded production metadata required')
    return json.loads(path.read_text())


def check_worker(args):
    state = small_json(STATE)
    row = state['tasks'][TASK]
    if time.time()-state['updated_unix'] > 120:
        raise ValueError('Campaign state is stale')
    if (row.get('pid'), row.get('process_start_ticks'), row.get('attempt'), row.get('physical_gpu')) != (
            args.pid, args.start_ticks, args.attempt, args.gpu):
        raise ValueError('Expected own task identity does not match campaign state')
    if identity(args.pid) != args.start_ticks:
        raise ValueError('Expected process identity is no longer live')


def device_sample(gpu):
    raw = subprocess.check_output(['nvidia-smi', '--id='+str(gpu),
        '--query-gpu=uuid,utilization.gpu,memory.used,power.draw',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    fields = [v.strip() for v in next(csv.reader(raw.splitlines()))]
    if fields[0] != UUIDS[gpu]:
        raise ValueError('Physical GPU UUID changed')
    processes = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
        text=True, timeout=10)
    pids = []
    for row in csv.reader(processes.splitlines()):
        if len(row) == 2 and row[1].strip() == UUIDS[gpu]:
            pids.append(int(row[0].strip()))
    return dict(utilization_percent=float(fields[1]), memory_MiB=float(fields[2]),
                power_W=float(fields[3]), compute_pids=sorted(set(pids)))


def summarize(values):
    return dict(samples=len(values), mean_percent=statistics.mean(values) if values else None,
        median_percent=statistics.median(values) if values else None,
        fraction_at_least80=sum(v >= 80 for v in values)/len(values) if values else None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--native', required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--start-ticks', required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--gpu', type=int, choices=(3,4,5), required=True)
    parser.add_argument('--seconds', type=int, default=10800)
    parser.add_argument('--interval', type=float, default=2)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.seconds <= 28800 and 1 <= args.interval <= 30):
        raise ValueError('Bounded duration and sampling interval required')
    output, native = contained(args.output), contained(args.native)
    if output.exists() or not native.is_dir():
        raise ValueError('Fresh output and existing native attempt directory required')
    check_worker(args)
    first = device_sample(args.gpu)
    if args.check_only:
        print(json.dumps(dict(status='read_only_identity_and_device_probe_passed',sample=first)))
        return
    output.mkdir()
    started = time.time()
    (output/'request.json').write_text(json.dumps(dict(arguments=vars(args),started_unix=started,
        controls_processes=False,scope=__doc__),indent=2)+'\n')
    values, exclusive, errors, intervals = [], [], [], []
    last_update = None
    reason = 'maximum_duration'
    begin = time.monotonic()
    with (output/'samples.csv').open('x',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['unix','utilization_percent','memory_MiB','power_W','compute_pids',
                         'expected_PID_is_only_compute_client','completed_updates','last_mark','probe_seconds'])
        while time.monotonic()-begin < args.seconds:
            tick = time.monotonic()
            if identity(args.pid) != args.start_ticks:
                reason='expected_process_ended'
                break
            try:
                check_worker(args)
                sample=device_sample(args.gpu)
                mark='preparation_or_no_live_stage'
                if (native/'training_live.json').exists():
                    live=small_json(native/'training_live.json')
                    mark=live.get('latest',{}).get('last_mark',mark)
                if (native/'latest_checkpoint.json').exists():
                    update=small_json(native/'latest_checkpoint.json')['complete_parent_updates']
                    if update != last_update:
                        intervals.append(dict(observed_unix=time.time(),completed_updates=update))
                        last_update=update
                only=sample['compute_pids']==[args.pid]
                value=sample['utilization_percent']
                values.append(value)
                if only:exclusive.append(value)
                writer.writerow([time.time(),value,sample['memory_MiB'],sample['power_W'],
                    json.dumps(sample['compute_pids']),only,last_update,mark,time.monotonic()-tick])
                stream.flush()
            except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
                errors.append(dict(unix=time.time(),kind=type(error).__name__,message=str(error)))
                if len(errors)>=5:
                    reason='telemetry_error_limit'
                    break
            if stream.tell() > 8*2**20:
                reason='output_bound'
                break
            time.sleep(max(0,args.interval-(time.monotonic()-tick)))
    result=dict(started_unix=started,finished_unix=time.time(),terminal_reason=reason,
        all_device_samples=summarize(values),expected_PID_only_samples=summarize(exclusive),
        checkpoint_observations=intervals,errors=errors,
        limitations='Sampling begins after process launch. Compute-client snapshots can miss short-lived peers. Exclusive device counters are not kernel occupancy or FLOP efficiency. CPU preparation is included. Stage marks may lag actual execution.')
    (output/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
