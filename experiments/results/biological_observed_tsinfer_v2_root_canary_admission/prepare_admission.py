from pathlib import Path
import json,hashlib,datetime,importlib.util,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'experiments/results/biological_observed_tsinfer_v2_root_canary_admission'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def write(name,value):
 p=OUT/name
 assert not p.exists(), str(p)
 p.write_text(json.dumps(value,indent=2)+'\n')
source=ROOT/'scripts/infer_biological_paths_tsinfer_observed_v2.py'
config=ROOT/'configs/biological_paths_tsinfer_observed_v2.json'
review_path=ROOT/'research/reviews/biological_paths_tsinfer_observed_v2_independent/revision_v3/review_receipt.json'
assert sha(source)=='9904dbbed05d38d2e1c3e03c7bc07f84376df309fff8fe4852ddeec229aac7b8'
assert sha(config)=='450a502066b74231b81f364ce47f4f927c1c28bf22be20f6d7a9abf945bd0953'
assert sha(review_path)=='95ca4e07aa9872fde1b3e8cd91903eeabc145dc9ec562a6baff656887e07bb67'
review=json.loads(review_path.read_text())
assert review['status']=='PASS'
assert not review['blocking_findings']
review_sources=review['reviewed_source_sha256']
assert review_sources['scripts/infer_biological_paths_tsinfer_observed_v2.py']==sha(source)
assert review_sources['configs/biological_paths_tsinfer_observed_v2.json']==sha(config)
spec=importlib.util.spec_from_file_location('root_observed_canary_v2_admission',source)
wrapper=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=wrapper
spec.loader.exec_module(wrapper)
base=wrapper.load_producer()
c=json.loads(config.read_text())
bio,lock=wrapper.validate_config(base,c)
assert not (ROOT/c['output_root']).exists()
runtime=json.loads(base.checked_ref(c['runtime']).read_text())
assert Path(sys.prefix).resolve()==(ROOT/bio['runtime_prefix']).resolve()
assert sha(Path(sys.executable))==runtime['python_executable_sha256']
inventory=base.runtime_inventory(bio['runtime_prefix'])
assert inventory==runtime['files_sha256'], 'frozen isolated runtime inventory changed'
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
write('runtime_preflight.json',{'kind':'root_frozen_isolated_runtime_preflight_v2','utc':now,'runtime':c['runtime'],'inventory_files':len(inventory),'complete_non_pyc_inventory_equal':True,'python_executable_sha256':sha(Path(sys.executable)),'isolated_prefix':str(Path(sys.prefix)),'observed_arrays_or_native_inference_executed':False})
templates=ROOT/'research/reviews/biological_paths_tsinfer_observed_v2_author/revision_v3'
storage=json.loads((templates/'canary_storage_receipt_template.json').read_text())
storage.update(template_inactive_not_authorization=False,ledger_sha256_at_admission=sha(ROOT/'.runtime/gpup/storage_reservations.json'),root_utc=now,authorizer='root_agent')
context=wrapper.storage_context(base,c,bio,storage,'canary')
account=context.check()
assert account['projected_bytes']==438354003968
assert account['projected_bytes']<450000000000
write('storage_admission.json',storage)
write('storage_preflight.json',{'kind':'root_observed_canary1000_storage_preflight','utc':now,'account':account,'storage_admission':ref(OUT/'storage_admission.json'),'complete_global_storage_measurement':False,'observed_arrays_executed':False})
a=json.loads((templates/'canary_root_admission_template.json').read_text())
a.update(authorizer='root_agent',template_inactive_not_authorization=False,root_utc=now,experiment_registration=ref(OUT/'experiment_registration.json'),storage=ref(OUT/'storage_admission.json'),independent_source_review=ref(review_path),independent_source_review_status=review['status'],independent_review_verdict='PASS',runtime_preflight=ref(OUT/'runtime_preflight.json'),root_supervisor=ref(OUT/'supervisor.py'),remaining_automatically_admitted=False)
write('admission.json',a)
print(json.dumps({'kind':'root_TRAIN000_only_canary1000_active_admission','admission':ref(OUT/'admission.json'),'runtime_files':len(inventory),'projected_storage_bytes':account['projected_bytes'],'native_execution_started':False}))
