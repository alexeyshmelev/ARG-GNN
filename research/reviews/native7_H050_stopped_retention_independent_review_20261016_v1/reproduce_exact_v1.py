"""Exact sealed source authentication and independent reproduction; synthetic fixtures only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,sys,tempfile,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');CAND='research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1';REVIEW='research/reviews/native7_H050_stopped_retention_independent_review_20261016_v1'
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def save(name,value):
 with (ROOT/REVIEW/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def run():
 expected={'source_handoff.v1.json':'387da53626ceed1b3d462b28dd8ef4b19fa42a74c6f600418f019c8498daeada','source_manifest.v1.json':'cb6c3af2858d007d74edb08742f8633ddef5f2a9cb4c6cd5b3682205fee32d88','independent_review_request.v1.json':'3b7fa8bc75902395b72987243ea205b20fd518cd71d588be76f308d20b33983c'}
 refs={}
 def walk(x):
  if isinstance(x,dict):
   if set(('path','sha256'))<=x.keys() and isinstance(x['path'],str):refs[(x['path'],x['sha256'])]=dict(path=x['path'],sha256=x['sha256'])
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 for n,sha in expected.items():
  assert ref(CAND+'/'+n)['sha256']==sha
  walk(json.loads((ROOT/CAND/n).read_text()))
 checked=[];excluded=[]
 for record in refs.values():
  p=ROOT/record['path']
  if p.suffix=='.pt':excluded.append(record);continue
  assert p.suffix in ('.py','.json','.md','.yaml') and p.resolve()==p and '..' not in p.parts and not p.is_symlink() and p.stat().st_size<=32*2**20,record
  assert ref(record['path'])==record,record
  checked.append(record)
 save('hash_verification.v1.json',dict(status='PASS',checked_ref_count=len(checked),checked_refs=checked,excluded_checkpoint_refs=excluded,checkpoint_payloads_opened=False))
 # Temporary fixture roots are relocated into reviewer-owned directory only.
 old_temp=tempfile.TemporaryDirectory
 def review_temp(*a,**kw):kw['dir']=ROOT/REVIEW;return old_temp(*a,**kw)
 tempfile.TemporaryDirectory=review_temp
 try:
  f=load(CAND+'/inert_fixtures_v1.py','_independent_retention_exact_inert');a=f.run()
 finally:tempfile.TemporaryDirectory=old_temp
 assert a['check_count']==49 and a['status']=='PASS_SOURCE_ONLY';save('author_inert_reproduction.v1.json',a)
 f=load(CAND+'/selector_factory_inert_v1.py','_independent_retention_exact_selector');b=f.run()
 assert b['check_count']==12 and b['status']=='PASS_SOURCE_ONLY';save('author_selector_factory_reproduction.v1.json',b)
 save('reproduction_summary.v1.json',dict(status='PASS_SOURCE_ONLY_SOFTWARE_FIXTURES',sealed_ref_count=len(checked),inert_checks=49,real_selector_factory_memo_checks=12,role_counts={k:v for k,v in a['full_role_models'].items() if k.endswith('count') or k.endswith('fits64')},actual_guards_probes_measurement_accounting_revocation_activation_admission_jobs_payload_or_Git=False,all_writes_owned_review_directory_only=True,finished_utc=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':
 try:run()
 except BaseException as e:save('reproduce.FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
