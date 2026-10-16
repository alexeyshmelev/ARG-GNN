from pathlib import Path
import json,hashlib,sys,traceback,os,importlib.util
from datetime import datetime,timezone
sys.excepthook=lambda t,v,b:traceback.print_exception(t,v,b,file=sys.stdout)
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/H056_chr21_native_actual_root_20261016_v1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
def ref(p):
 p=Path(p)
 return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(n):return json.loads((HERE/n).read_text())
def write(n,v):
 p=HERE/n
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 return ref(p)
def load(r,name):
 p=ROOT/r['path']
 assert ref(p)==r
 sp=importlib.util.spec_from_file_location(name,p)
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m
 exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)
 return m
assert '--execute' in sys.argv
cfg=read('config.json');fixed=cfg['fixed_refs'];runtime=read('runtime.json')
assert datetime.now(timezone.utc)<datetime.fromisoformat(runtime['expires_at_utc'])
assert not (ROOT/runtime['output_directory']).exists()
assert not (ROOT/runtime['auxiliary_directory']).exists()
assert not (HERE/'reserve_receipt.actual_v1.json').exists()
assert not (HERE/'input.json').exists()
manifest=read('source_manifest.json');b=manifest['original_bindings']
request=read('reservation_request.root_v1.json')
assert request['reserved_bytes']==cfg['caps']['output_cap_bytes']==87031808
assert len(request['consumer_bindings'])==37
for rr in request['consumer_bindings']:assert ref(ROOT/rr['path'])==rr
renewal=load(fixed['renewal'],'_H056_native_admit_renewal')
renewal.install_renewal_v1(fixed['renewal_certificate'])
guard=load(fixed['guard'],'_H056_native_admit_guard')
guard.configure_v2(fixed['repair_authority'],policy_ref=fixed['policy'])
kw={'expected_source_sha256':b['implementation']['sha256'],'expected_launcher_sha256':b['launcher']['sha256']}
view=guard.inspect_v1(**kw)
assert view['bindings']==b
auth=guard.operation_draft_v1('reserve',request,expected_before=guard._pins(view),bindings=b)
auth.update(active=True,expires_at_utc=runtime['expires_at_utc'])
ar=write('reserve_authority.root_active_v1.json',auth)
receipt=guard.apply_reserve_v1(ar,**kw)
rr=write('reserve_receipt.actual_v1.json',receipt)
components={'consumer_adapter':fixed['guard'],'implementation':b['implementation'],'launcher':b['launcher']}
storage={'kind':'project_budget_v9_storage_record','active':True,'consumer_uses_v9_guard':True,'accounting_mode':'prospective_980gb_v1','job_id':runtime['job_id'],'output_directory':runtime['output_directory'],'expires_at_utc':runtime['expires_at_utc'],'reserved_new_bytes':request['reserved_bytes'],'reservation':receipt['value'],'initial_ledger_pins':receipt['initial_ledger_pins'],'completed_overlay_policy_ref':fixed['policy'],'budget_components':components}
sr=write('storage.json',storage)
inp=read('input.inactive.json')
inp.update(status='active',execution_authorized=True,storage_ref=sr)
ir=write('input.json',inp)
result={'schema':'H056_native_actual_admission_v1','direct_source_bindings':len(request['consumer_bindings']),'input_ref':ir,'reserve_receipt_ref':rr,'projected_bytes':receipt['projected_bytes'],'scientific_job_spawned':False,'original_expiry_preserved':True}
write('actual_admission_completed.root_v1.json',result)
print(json.dumps(result),flush=True)
