import hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
root=Path('/workspace-SR003.nfs2/ARG-GNN');owned=root/'research/reviews/census_biological_bank_v1_independent'
paths=['src/arggnn/census_biological_bank_v1.py','tests/test_census_biological_bank_v1.py','configs/census_biological_bank_v1.json','docs/census_biological_bank_v1.md','research/reviews/census_biological_bank_v1_author/receipt.json','research/protocols/census_biological_native_join_v1.json']
config=json.loads((root/paths[2]).read_text());paths+=list(config['source_sha256']);paths=list(dict.fromkeys(paths))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
attempt=1
while (owned/f'pytest_attempt_{attempt:02d}.json').exists():attempt+=1
record={'started_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Source plus independent handmade CPU fixtures and rerun sealed author handmade suite only. No actual observed/model/target array access, simulation, native reconstruction, training, GPU, install or Git.','source_sha256_before':{p:sha(root/p) for p in paths},'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','TMPDIR','OMP_NUM_THREADS','MKL_NUM_THREADS']}}
command=[str(root/'.runtime/env/bin/python'),'-m','pytest','tests/test_census_biological_bank_v1.py',str(owned/'test_independent_join.py'),'-q','-p','no:cacheprovider','--basetemp='+str(owned/f'pytest_tmp_{attempt:02d}')]
record['command']=command;start=time.perf_counter();process=subprocess.run(command,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);record['seconds']=time.perf_counter()-start;record['exit_code']=process.returncode
output=owned/f'pytest_attempt_{attempt:02d}.txt';output.write_text(process.stdout)
record['output_sha256']=sha(output);record['source_sha256_after']={p:sha(root/p) for p in paths};record['all_pinned_source_unchanged']=record['source_sha256_before']==record['source_sha256_after']
(owned/f'pytest_attempt_{attempt:02d}.json').write_text(json.dumps(record,indent=2)+'\n')
print(process.stdout);print(json.dumps({k:record[k] for k in ['seconds','exit_code','all_pinned_source_unchanged']},indent=2))
