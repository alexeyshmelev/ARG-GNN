from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
root=Path('/workspace-SR003.nfs2/ARG-GNN');owned=root/'research/reviews/census_native_qualification_runner_v1_author'
attempt=1
while (owned/f'pytest_attempt_{attempt:02d}.json').exists():attempt+=1
source=['src/arggnn/census_native_qualification_runner_v1.py','configs/census_native_qualification_runner_v1.json','tests/test_census_native_qualification_runner_v1.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record={'started_at_utc':datetime.now(timezone.utc).isoformat(),'before':{p:sha(root/p) for p in source},'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','TMPDIR','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
command=[str(root/'.runtime/env/bin/python'),'-m','pytest','tests/test_census_native_qualification_runner_v1.py','-q','-p','no:cacheprovider','--basetemp='+str(owned/f'pytest_tmp_{attempt:02d}')]
start=time.perf_counter();p=subprocess.run(command,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);record.update(command=command,seconds=time.perf_counter()-start,exit_code=p.returncode,after={q:sha(root/q) for q in source})
out=owned/f'pytest_attempt_{attempt:02d}.txt';out.write_text(p.stdout);record['output_sha256']=sha(out);(owned/f'pytest_attempt_{attempt:02d}.json').write_text(json.dumps(record,indent=2)+'\n');print(p.stdout);print(json.dumps({'seconds':record['seconds'],'exit_code':p.returncode}))
