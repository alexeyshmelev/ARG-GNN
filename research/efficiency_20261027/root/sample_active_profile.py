"""Read-only GPU telemetry until an owned diagnostic profile finishes."""
import argparse,csv,json,subprocess,time
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
p=argparse.ArgumentParser(__doc__)
p.add_argument('--profile',required=True)
args=p.parse_args()
directory=(ROOT/args.profile).resolve()
assert directory.is_relative_to(ROOT/'research/efficiency_20261027/root/runs')
assert (directory/'request.json').is_file()
with (directory/'device_telemetry.csv').open('x',newline='') as stream, (directory/'process_telemetry.jsonl').open('x') as processes:
 writer=csv.writer(stream)
 writer.writerow(['observed_unix','gpu','uuid','utilization_percent','used_MiB','power_W'])
 for index in range(14400):
  tick=time.monotonic()
  status=json.loads((directory/'status.json').read_text())
  if status.get('status') in ('finished','failed') or (directory/'actual_wait.json').exists() or (directory/'failure.json').exists():break
  raw=subprocess.check_output(['nvidia-smi','--id=3,4,5','--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=15)
  now=time.time()
  for row in csv.reader(raw.splitlines()):writer.writerow([now]+[x.strip() for x in row])
  stream.flush()
  if index%5==0:
   raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=15)
   processes.write(json.dumps({'observed_unix':now,'rows':[row for row in csv.reader(raw.splitlines()) if row[0].strip() in {'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786','GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71','GPU-3db85505-d061-e45d-d154-e7e8c1cad497'}]})+'\n')
   processes.flush()
  time.sleep(max(0.,1.-(time.monotonic()-tick)))
print(json.dumps({'status':'telemetry_finished','profile':str(directory.relative_to(ROOT))}))
