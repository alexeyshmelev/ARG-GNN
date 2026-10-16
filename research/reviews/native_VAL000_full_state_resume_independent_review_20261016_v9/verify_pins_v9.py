from pathlib import Path
import ast,hashlib,json,os,stat
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SRC=ROOT/'research/reviews/native_VAL000_full_state_resume_source_20261016_v9'
OUT=ROOT/'research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v9'
TOP={'source_handoff.v9.json':'afbfcd9e947e9a31c72c91be014ee67808c53ea4ff3a744646134ac9b6ce2233','source_manifest.v1.json':'ac747115d24678218ddd3ed084d0a41375d03cd6e1913298155f5f1bcdd8fd7d','supplemental_real_public_helpers_inert_v1.py':'626b8197a21cfbbd1a269a2678a653f71e57d02479705186dadc417b8b683aad','supplemental_real_public_helpers_inert.v1.json':'1dffb9ebad8ee4de47707dd3adb895aab5a0e7ccc238c79d825e79a6adf2675e'}
refs={};where={}
def walk(v,origin):
 if isinstance(v,dict):
  if set(v)=={'path','sha256'} and isinstance(v['path'],str) and isinstance(v['sha256'],str):
   key=v['path'];assert key not in refs or refs[key]==v
   refs[key]=v;where.setdefault(key,[]).append(origin)
  else:
   for k,value in v.items():walk(value,origin+'/'+k)
 elif isinstance(v,list):
  for i,value in enumerate(v):walk(value,origin+'/'+str(i))
for name,sha in TOP.items():
 p=SRC/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,name
 refs[str(p.relative_to(ROOT))]={'path':str(p.relative_to(ROOT)),'sha256':sha}
 if p.suffix=='.json':walk(json.loads(p.read_text()),name)
for name in ('ROOT_fresh_entry_API.INACTIVE.v9.json',):walk(json.loads((SRC/name).read_text()),name)
results=[]
for path,r in sorted(refs.items()):
 p=ROOT/path
 assert p.resolve()==p and ROOT in p.parents and '..' not in p.parts,path
 assert p.suffix not in ('.pt','.pth','.bin','.safetensors'),('NO_TENSOR_ACCESS',path)
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size<=32*2**20,path
 actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==r['sha256'],(path,actual,r['sha256'])
 results.append(dict(r,bytes=s.st_size))
asts=[]
for p in sorted(SRC.glob('*.py')):
 ast.parse(p.read_text(),filename=str(p));asts.append(str(p.relative_to(ROOT)))
value=dict(schema='native_v9_independent_pin_verification_v1',status='PASS',distinct_declared_refs=len(results),verified_refs=results,new_source_AST_parsed=asts,checkpoint_payload_or_actual_resource_access=False)
(OUT/'hash_verification.v9.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status='PASS',distinct_refs=len(results),new_source_AST_count=len(asts))))
