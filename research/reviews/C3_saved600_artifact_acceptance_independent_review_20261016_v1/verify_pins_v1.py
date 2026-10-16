from pathlib import Path
import ast,hashlib,json,stat
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');SRC=ROOT/'research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1'
OUT=ROOT/'research/reviews/C3_saved600_artifact_acceptance_independent_review_20261016_v1'
TOP={'source_handoff.v1.json':'d57045ec59223ebbea48efcff9f67550717be84090efa4560855d110b31ee010','source_manifest.v1.json':'a76e6eb58fef091dd80ca8ba918c4873b1a31b7f28f28b2e6e84104ec910af35','acceptance_contract_v1.py':'3b9a60267873f4c84fe1ca03010088df510547c92c1c61cad20dcd4d02f6f1c2'}
refs={}
def walk(v):
 if isinstance(v,dict):
  if set(v)=={'path','sha256'}:
   assert v['path'] not in refs or refs[v['path']]==v
   refs[v['path']]=v
  else:
   for x in v.values():walk(x)
 elif isinstance(v,list):
  for x in v:walk(x)
for n,sha in TOP.items():
 p=SRC/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 refs[str(p.relative_to(ROOT))]=dict(path=str(p.relative_to(ROOT)),sha256=sha)
 if p.suffix=='.json':walk(json.loads(p.read_text()))
m=json.loads((SRC/'source_manifest.v1.json').read_text())
for path,sha in m['source_sha256'].items():
 r=dict(path=path,sha256=sha);assert path not in refs or refs[path]==r;refs[path]=r
for p in SRC.iterdir():
 if p.is_file() and p.name!='source_handoff.v1.json':
  path=str(p.relative_to(ROOT));sha=hashlib.sha256(p.read_bytes()).hexdigest()
  assert path not in refs or refs[path]['sha256']==sha
  refs[path]=dict(path=path,sha256=sha)
verified=[]
for path,r in sorted(refs.items()):
 p=ROOT/path;assert ROOT in p.parents and '..' not in p.parts and p.resolve()==p
 assert p.suffix not in ('.pt','.pth','.npz','.bin','.safetensors'),('PAYLOAD_REF_NEVER_FOLLOWED',path)
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size<=32*2**20
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],path
 verified.append(dict(r,bytes=s.st_size))
asts=[]
for p in SRC.glob('*.py'):ast.parse(p.read_text());asts.append(p.name)
v=dict(schema='C3_saved600_independent_declared_source_pins_v1',status='PASS',distinct_refs=len(verified),verified_refs=verified,AST_parsed=asts,
 scientific_checkpoint_truth_prediction_payloads_accessed=False,actual_guard_probe_accounting_or_execution=False)
(OUT/'hash_verification.v1.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status='PASS',refs=len(verified),ASTs=len(asts))))
