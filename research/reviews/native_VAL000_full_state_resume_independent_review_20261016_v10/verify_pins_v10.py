from pathlib import Path
import ast,hashlib,json,stat
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');SRC=ROOT/'research/reviews/native_VAL000_full_state_resume_source_20261016_v10'
OUT=ROOT/'research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v10'
TOP={'source_handoff.v10.json':'f4631c2d5db822ded0070c8db7ecf99620509fec562a8c8e5449756298dea423',
'source_manifest.v1.json':'9ab34741ee948a2ddc9fe986abb0bd72533ead0a788c79270fe3064307193719'}
refs={}
def walk(v):
 if isinstance(v,dict):
  if set(v)=={'path','sha256'}:
   assert v['path'] not in refs or refs[v['path']]==v
   refs[v['path']]=v
  else:
   for value in v.values():walk(value)
 elif isinstance(v,list):
  for value in v:walk(value)
for n,sha in TOP.items():
 p=SRC/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 refs[str(p.relative_to(ROOT))]=dict(path=str(p.relative_to(ROOT)),sha256=sha)
 walk(json.loads(p.read_text()))
walk(json.loads((SRC/'ROOT_controller_failure_API.INACTIVE.v10.json').read_text()))
# Follow only source and exact compact snapshot pins declared by the sealed audit.
audit=json.loads((ROOT/'research/reviews/native_v9_controller_entry_failure_audit_20261016_v1/audit.v1.json').read_text())
for r in audit['source_refs']:refs[r['path']]=r
r=audit['registry_observation']['immutable_exact_snapshot_ref'];refs[r['path']]=r
verified=[]
for path,r in sorted(refs.items()):
 p=ROOT/path;assert ROOT in p.parents and p.resolve()==p and '..' not in p.parts
 assert p.suffix not in ('.pt','.pth','.bin','.safetensors'),('NO_PAYLOAD',path)
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size<=32*2**20
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],path
 verified.append(dict(r,bytes=s.st_size))
asts=[]
for p in SRC.glob('*.py'):ast.parse(p.read_text());asts.append(p.name)
v=dict(schema='native_v10_independent_declared_pin_verification_v1',status='PASS',distinct_refs=len(verified),
 inherited_source_authentication_ref_count=len(json.loads((SRC/'source_manifest.v1.json').read_text())['source_authentication_refs']),
 verified_refs=verified,AST_parsed=sorted(asts),payload_or_actual_resource_access=False)
(OUT/'hash_verification.v10.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status='PASS',distinct_refs=len(verified),ASTs=len(asts))))
