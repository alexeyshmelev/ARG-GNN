"""Independent stdlib admission ordering, attribution and seal checks, no live lease/budget."""
import ast,hashlib,json,os,sys
from pathlib import Path
from arggnn import census_tsinfer_fit_v1 as fit,census_tsinfer_device_profile_v1 as profile
ROOT=fit.ROOT;DIR=ROOT/'research/reviews/census_tsinfer_consumers_v1_independent'
checks=[]
def check(n,v=True):
 if not v:raise AssertionError(n)
 checks.append(n)
def reject(n,f):
 try:f()
 except (ValueError,KeyError,TypeError):check(n);return
 raise AssertionError(n)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pins={fit.SOURCE:'fd55aaa6b8f88983514ff6888ad2efbadca4ee8b2e3acabd09fbd4a361b5acf3',
 fit.CONFIG:'7c2436d5479e68beb1a8f3edb04ca0ee729a23be35ca2d7893827c0db68e3b00',
 'src/arggnn/census_tsinfer_device_profile_v1.py':'c596f8c11137d29f6e9ec25a1b8a2c9090f85e41d072395730f7dd1fd3d9d6d2',
 'configs/census_tsinfer_device_profile_v1.json':'244afae3271955104186c055653fe25954a1c4791e7a6989e0ccc88a32e777b6',
 'research/reviews/census_tsinfer_fit_v1_author/author_receipt.json':'bd0e4866e9de15f564a5c35574af223cf808ea1d98e5d9d661175a1c32f8b843',
 'research/reviews/census_tsinfer_device_profile_v1_author/author_receipt.json':'29faa83f2a06c7b7fd64f7e86cad65f57147b25a6a1f8caebe09e1cfae7afd5d'}
for p,h in pins.items():check('final_author_pin_'+p,sha(ROOT/p)==h)
for p in [p for p in pins if p.endswith('author_receipt.json')]:
 receipt=json.loads((ROOT/p).read_text())
 for name,h in receipt['artifact_sha256'].items():check('author_inventory_'+name,sha(ROOT/name)==h)
check('imports_remain_standard_library_only',not any(n in sys.modules for n in ('torch','numpy','tskit')))
for path,func in [(fit.SOURCE,'admit_fit_v1'),('src/arggnn/census_tsinfer_device_profile_v1.py','admit_profile')]:
 tree=ast.parse((ROOT/path).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==func)
 calls=[]
 for statement in node.body:
  for n in ast.walk(statement):
   if isinstance(n,ast.Call):
    f=n.func
    name=f.id if isinstance(f,ast.Name) else f.attr if isinstance(f,ast.Attribute) else ''
    if name: calls.append((n.lineno,name))
 calls.sort();names=[n for l,n in calls]
 check('held_logical_job_before_metadata_gate_'+func,names.index('require_logical_job_v1')<names.index('gate_from_reader_v1'))
 check('fresh_storage_before_opaque_closure_'+func,names.index('storage_v1')<names.index('verify_observation_closure_v1'))
 check('fresh_memory_before_opaque_closure_'+func,names.index('memory_v1')<names.index('verify_observation_closure_v1'))
 check('opaque_closure_before_scientific_runtime_'+func,names.index('verify_observation_closure_v1')<names.index('runtime_v1'))
 gate=next(n for n in ast.parse((ROOT/fit.SOURCE).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='gate_from_reader_v1')
 literals={n.value for n in ast.walk(gate) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
 check('metadata_gate_does_not_read_opaque_closure_'+func,not {'observed_artifact_sha256','derived_artifact_sha256'}&literals)
new=ast.parse((ROOT/fit.SOURCE).read_text())
resources=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='_check_resources_v1')
check('every_fit_resource_boundary_checks_logical_job',any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require_logical_job_v1' for n in ast.walk(resources)))
for mod in (fit,profile):
 os.environ['ARGGNN_JOB_NAME']='handmade-exclusive';os.environ.pop('ARGGNN_LOGICAL_JOB_NAME',None)
 mod.require_logical_job_v1({'launcher_metadata':{'job':'handmade-exclusive'}},{'job_id':'handmade-exclusive'})
 check('exclusive_worker_attribution_'+mod.__name__)
 os.environ['ARGGNN_JOB_NAME']='handmade-pool';os.environ['ARGGNN_LOGICAL_JOB_NAME']='handmade-logical'
 lease={'launcher_metadata':{'job':'handmade-pool'}};admit={'job_id':'handmade-logical'}
 mod.require_logical_job_v1(lease,admit);check('shared_pool_logical_attribution_'+mod.__name__)
 reject('wrong_admission_logical_job_'+mod.__name__,lambda:mod.require_logical_job_v1(lease,{'job_id':'another'}))
 reject('wrong_lease_pool_job_'+mod.__name__,lambda:mod.require_logical_job_v1({'launcher_metadata':{'job':'different-pool'}},admit))
 os.environ.pop('ARGGNN_LOGICAL_JOB_NAME',None)
 reject('missing_shared_pool_logical_job_'+mod.__name__,lambda:mod.require_logical_job_v1(lease,admit))
check('pure_attribution_guard_does_not_import_science',not any(n in sys.modules for n in ('torch','numpy','tskit')))
for p,h in pins.items():check('final_author_pin_preserved_'+p,sha(ROOT/p)==h)
r={'schema':'census_tsinfer_consumer_independent_admission_checks_v1','status':'passed','count':len(checks),'checks':checks,
 'artifact_sha256':pins,'test_source_sha256':sha(Path(__file__)),'actual_arrays_GPU_live_leases_budgets_or_Git':False}
(DIR/'admission_order_checks.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'status':'passed','checks':len(checks)},indent=2))
