"""Read bounded campaign/task metadata and report only measured progress."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import json
import subprocess
import re
from arggnn.benchmarking.paper_campaign import runtime as r

out=r.path('research/operations_resume_20261026')
launch=r.read(r.ref(out/'launch.json'))
state_path=r.path('.runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json')
state=json.loads(state_path.read_text())
plan=r.read(launch['plan_ref'])
rows={}
for task in plan['tasks']:
    row=state['tasks'][task['id']]
    if row['status'] not in ('running','failed','blocked'):
        continue
    item={k:row[k] for k in ('status','pid','physical_gpu','started_unix','error','log_path','observed_pss_bytes') if k in row}
    progress=r.path(task['output_directory'])/'resource_progress.json'
    if progress.is_file():
        item['resource_progress']=json.loads(progress.read_text())
    proc=Path('/proc')/str(row.get('pid',-1))
    if proc.exists():
        item['process_title']=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
        item['cpu_ticks']=sum(int(x) for x in (proc/'stat').read_text().rsplit(')',1)[1].split()[11:13])
    rows[task['id']]=item
arg=r.path('experiments/results/paper_american_20261024/argformer_american_validation_001/mlm/native/resume_20261026_batch119500/loaded_resume_state.json')
tree=r.path('experiments/results/paper_american_20261024/treeloom_fit/native/resume_20261026_update238/training_live.json')
record=dict(verified_moscow=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
            controller_pid=state['controller_pid'],state_updated_unix=state.get('updated_unix'),
            counts=dict(Counter(x['status'] for x in state['tasks'].values())),
            task_rows=rows, project_reserved_bytes=state.get('project_reserved_bytes'),
            waiting_reasons=dict(Counter(x.get('waiting_reason') for x in state['tasks'].values() if x['status']=='pending')),
            completed_benchmark_comparison_claimed=False,
            gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
arg_task='paper_american_argformer_american_validation_001_mlm'
if arg_task in rows and rows[arg_task].get('log_path'):
    log=r.path(rows[arg_task]['log_path'])
    with log.open('rb') as handle:
        handle.seek(max(0, log.stat().st_size-65536))
        tail=handle.read().decode(errors='replace')
    metrics={}
    for key in ('time/batch','loss/train/total','throughput/batches_per_sec'):
        values=re.findall(r'Train '+re.escape(key)+r': ([0-9.]+)',tail)
        if values:
            metrics[key]=float(values[-1])
    record['argformer_latest_logged_training_metrics']=metrics
if arg.exists():record['argformer_loaded_resume_state']=json.loads(arg.read_text())
if tree.exists():record['treeloom_training_live']=json.loads(tree.read_text())
r.save(out/'latest_status.json',record)
print(json.dumps(record,indent=2))
