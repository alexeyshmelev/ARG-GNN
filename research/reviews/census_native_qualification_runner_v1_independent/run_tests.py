"""Replay immutable source suite or independent handmade guards; owned outputs only."""
from pathlib import Path
import subprocess,os,json,time,hashlib
from datetime import datetime,timezone
import argparse
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OWN=ROOT/'research/reviews/census_native_qualification_runner_v1_independent'
parser=argparse.ArgumentParser();parser.add_argument('--suite',choices=('source','independent'),required=True);args=parser.parse_args()
attempt=1
while (OWN/f'{args.suite}_pytest_attempt_{attempt:02d}.json').exists():attempt+=1
before=json.loads((OWN/'source_hashes_before.json').read_text())
def hashes():return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in before}
command=[str(ROOT/'.runtime/env/bin/python'),'-m','pytest','tests/test_census_native_qualification_runner_v1.py' if args.suite=='source' else str(OWN/'test_independent.py'),'-q','-p','no:cacheprovider','--basetemp='+str(OWN/f'{args.suite}_pytest_tmp_{attempt:02d}')]
record={'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_before':hashes(),'command':command,'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TMPDIR')}}
start=time.monotonic();result=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
record.update(returncode=result.returncode,wall_seconds=time.monotonic()-start,source_after=hashes())
text=OWN/f'{args.suite}_pytest_attempt_{attempt:02d}.txt';text.write_text(result.stdout);record['output_sha256']=hashlib.sha256(text.read_bytes()).hexdigest()
(OWN/f'{args.suite}_pytest_attempt_{attempt:02d}.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(result.stdout);print(json.dumps({'attempt':attempt,'returncode':result.returncode,'wall_seconds':record['wall_seconds']}))
raise SystemExit(result.returncode)
