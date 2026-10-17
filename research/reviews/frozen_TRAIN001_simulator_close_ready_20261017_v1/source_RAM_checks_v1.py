"""Actual admitted TRAIN001 metadata plus explicitly synthetic future success in RAM."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,contextlib,hashlib,io,json,types
D=Path('research/reviews/frozen_TRAIN001_simulator_close_ready_20261017_v1')
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
checks=[];negative=[]
def check(ok,label):
    if not ok:raise AssertionError(label)
    checks.append(label)
def reject(fn,label):
    try:fn()
    except(ValueError,KeyError,TypeError,AssertionError):negative.append(label);return
    raise AssertionError('Accepted invalid '+label)
source=(D/'close_ROOT_v1.py').read_text()
env={'__name__':'RAM_ONLY_CLOSE'}
tree=ast.parse(source);tree.body=[n for n in tree.body if not isinstance(n,ast.If)]
exec(compile(tree,'RAM_CLOSE_SOURCE','exec',dont_inherit=True,optimize=0),env)
with contextlib.redirect_stdout(io.StringIO())as out:rc=env['main']([])
check(rc==0 and json.loads(out.getvalue())['status']=='SOURCE_ONLY_INACTIVE','Actual default entry inert')
base=json.loads((D/'pins.v1.json').read_text())
manifest=json.loads((D/'source_manifest.v1.json').read_text())
cache={}
def actual_read(r):
    b=(ROOT/r['path']).read_bytes();check(hashlib.sha256(b).hexdigest()==r['sha256'],'Actual compact metadata/source ref '+r['path']);cache[r['path']]=b;return json.loads(b)
for r in manifest['source_refs']:
    b=(ROOT/r['path']).read_bytes();check(hashlib.sha256(b).hexdigest()==r['sha256'],'Exact source closure '+r['path']);cache[r['path']]=b
input_ref=dict(path='research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v5/ROOT_actual_TRAIN001_root02_oct17_v5/input.json',sha256='91743348259ade65ccf83017ece27a01ba61df0147c8f5ee192e9f05b289c491')
inp=actual_read(input_ref);storage=actual_read(inp['refs']['storage']);q=actual_read(inp['root_stage_request_ref']);cfg=actual_read(inp['refs']['config']);caps=actual_read(inp['caps_ref'])
own=storage['reservation']
check(own['reserved_bytes']==7000000000 and len(own['consumer_bindings'])==61,'Actual original7GB and61 admitted bindings preserved')
for r in own['consumer_bindings']:
    b=(ROOT/r['path']).read_bytes();check(hashlib.sha256(b).hexdigest()==r['sha256'],'Actual admitted binding '+r['path']);cache[r['path']]=b
docs={r['path']:v for r,v in [(input_ref,inp),(inp['refs']['storage'],storage),(inp['root_stage_request_ref'],q),(inp['refs']['config'],cfg),(inp['caps_ref'],caps)]}
def ram(path,value):
    b=json.dumps(value,sort_keys=True,separators=(',',':')).encode();r=dict(path=path,sha256=hashlib.sha256(b).hexdigest());cache[path]=b;docs[path]=copy.deepcopy(value);return r
completed=ram(inp['output_directory']+'/completed.json',dict(status='completed_C3_simulation_reconstruction',parent_id='american_c3_train_001',sample_unit='haploid',class_order=['EUR','EAS','AFR'],sites=200000,actual_tables_bytes=1000000,observation_ref=ram('RAM_ONLY/observed.json',dict(schema='american_c3_observed_v1',parent_id='american_c3_train_001',sequence_length_bp=30000000,counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),query_ancestry_truth_in_inputs=False,migrations_and_internal_population_labels_removed=True))))
launch=ram('RAM_ONLY/launch.json',dict(pid=6000,input_ref=input_ref,ownership=dict(signals_use_pidfd_only=True,no_numeric_group_signals=True,owned_identities=[dict(pid=6000,start=101)])))
wait=ram('RAM_ONLY/worker_wait.json',dict(schema='American_C3_full_original_CPU_actual_wait_v1',parent_id='american_c3_train_001',phase='simulate_train_001',input_ref=input_ref,pid=6000,completed_ref=completed,returncode=0,actual_OS_wait_status=0,actual_direct_worker_wait=True,actual_direct_launcher_wait=True,all_owned_processes_and_groups_drained=True,cleanup_errors=[],operational_failure=None,adopted_waits=[],group_observations=[dict(pgid=6000,absent=True)],known_owned_pids=[6000],known_owned_process_groups=[6000],supervisor_pid=5000,supervisor_process_group=5000))
supervisor=ram('RAM_ONLY/supervisor_wait.json',dict(pid=5000,actual_direct_wait=True,actual_returncode=0))
outer=ram('RAM_ONLY/outer.json',dict(schema='ROOT_frozen_TRAIN001_simulator_actual_outer_terminal_v1',root_authorizer='root_agent',actual_terminal_consumed=True,actual_exit_code=0,actual_tool_session_id=123,actual_worker_wait_ref=wait,actual_supervisor_wait_ref=supervisor,completed_ref=completed,input_ref=input_ref,all_owned_processes_and_groups_drained=True,new_ancestry_score=False))
digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
p=dict(base,input=input_ref,launch=launch,wait=wait,completed=completed,outer=outer,supervisor_wait=supervisor,storage=inp['refs']['storage'],original_request=inp['root_stage_request_ref'],config=inp['refs']['config'],reservation_key=own['key'],reservation_sha256=digest(own))
def read_ram(r):return copy.deepcopy(docs[r['path']])
env['read']=read_ram;env['raw']=lambda r:cache[r['path']]
joined=env['metadata_join'](p)
check(joined[5]==own and joined[6]==[5000,6000]and joined[7]==[5000,6000],'Complete actual-input/future-RAM-success metadata join')
def mutate_reject(label,path,key,value):
    old=copy.deepcopy(docs[path]);docs[path][key]=value
    reject(lambda:env['metadata_join'](p),label);docs[path]=old
mutate_reject('Worker nonzero',wait['path'],'returncode',1)
mutate_reject('Worker raw wait nonzero',wait['path'],'actual_OS_wait_status',15)
mutate_reject('Unknown owned drain',wait['path'],'all_owned_processes_and_groups_drained',False)
mutate_reject('Unconsumed ROOT outer',outer['path'],'actual_terminal_consumed',False)
mutate_reject('ROOT outer nonzero',outer['path'],'actual_exit_code',1)
mutate_reject('Unknown ROOT session',outer['path'],'actual_tool_session_id',None)
mutate_reject('Wrong completed parent',completed['path'],'parent_id','american_c3_train_000')
mutate_reject('Unsuccessful completion',completed['path'],'status','FAILED')
mutate_reject('Supervisor wait mismatch',supervisor['path'],'pid',5001)
mutate_reject('Supervisor nonzero',supervisor['path'],'actual_returncode',1)
gsource=(ROOT/base['guard']['path']).read_text();gtree=ast.parse(gsource)
gnames={'_proof','_pointer','_identity','_completed_identity','consumer_bindings_v1'}
g={}
for n in gtree.body:
    if isinstance(n,ast.FunctionDef)and n.name in gnames:exec(compile(ast.Module(body=[n],type_ignores=[]),'RAM_ORIGINAL_'+n.name,'exec'),g)
class Fixed:
    @staticmethod
    def now(tz):return datetime(2100,1,1,tzinfo=timezone.utc)
    fromisoformat=staticmethod(datetime.fromisoformat)
g.update(dict(ROOT=ROOT,Path=Path,datetime=Fixed,timezone=timezone,copy=copy,require=env['need'],_json=read_ram,_completed_json=read_ram,budget=types.SimpleNamespace(digest=digest,v4=types.SimpleNamespace(encoded=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),integer=lambda v,l,m=0:env['need'](type(v)is int and v>=m,'Exact integer'),canonical=lambda r,s:Path(s)if Path(s).is_absolute()else r/s))))
rows=[dict(pid=n,absent=True,method='targeted os.kill(pid,0) ESRCH',checked_utc='2099-12-31T00:00:00+00:00')for n in joined[6]]
obs=ram('RAM_ONLY/targeted_observation.json',dict(schema='concurrent_completed_scope_targeted_root_observation_v1',root_authorizer='root_agent',key=own['key'],reservation_sha256=digest(own),storage_ref=p['storage'],launch_ref=launch,direct_wait_ref=wait,completed_ref=completed,launcher_pid=6000,all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=joined[6],observations=rows))
g['_absent']=lambda pid:next(copy.deepcopy(r)for r in rows if r['pid']==pid)
request=env['request_value'](p,own,joined[3],obs)
for observe in(False,True):check(g['_proof'](request,own,observe=observe)['completion_verified']is True,'Original exact ordinary proof accepts RAM success observe='+str(observe))
bad=copy.deepcopy(request);bad['reservation_sha256']='0'*64;reject(lambda:g['_proof'](bad,own),'Original proof rejects changed reservation digest')
saved=copy.deepcopy(docs[obs['path']]);docs[obs['path']]['observations'][0]['absent']=False
reject(lambda:g['_proof'](request,own),'Original proof rejects nonabsent target');docs[obs['path']]=saved
# Real current47-role metadata snapshot, no live state or accounting read.
roles=json.loads((ROOT/'research/reviews/FLARE_VAL004_native_ROOT_close_Census_root01_oct17/ordinary_close_authority.json').read_text())['bindings']
g['_raw']=lambda r:cache[r['path']]
union=g['consumer_bindings_v1'](own['consumer_bindings'],list(roles.values()))
check(len(roles)==47 and union==own['consumer_bindings']and len(union)==61,'All actual47role refs already in actual61binding reservation')
# Full original scope validator with explicit RAM file boundary, real canonical source function.
vsource=(ROOT/'src/arggnn/project_budget_v4.py').read_text();vtree=ast.parse(vsource)
v=dict(Path=Path,json=json,LEGACY='.runtime/storage/ledger.json',LOCK='.runtime/locks/scheduler.lock')
names={'encoded','integer','bound','paths_overlap','accounting_paths','canonical','scope_spec'}
for n in vtree.body:
    if isinstance(n,ast.FunctionDef)and n.name in names:exec(compile(ast.Module(body=[n],type_ignores=[]),'RAM_ORIGINAL_V4_'+n.name,'exec'),v)
v['file_bytes']=lambda root,path:(cache[str(path)],hashlib.sha256(cache[str(path)]).hexdigest());v['parse']=json.loads
reservation={k:value for k,value in own.items()if k!='root_admission'}
check(len(v['scope_spec'](ROOT,reservation))==2,'Original full scope_spec accepts exact actual original primary+AUX reservation')
bad=copy.deepcopy(reservation);bad['consumer_bindings'].append(dict(path='RAM_ONLY/lease.lock',sha256='0'*64));reject(lambda:v['scope_spec'](ROOT,bad),'Original scope rejects raw lock binding')
# Preserve exact apply body, measurement and append call order.
apply=next(n for n in gtree.body if isinstance(n,ast.FunctionDef)and n.name=='apply_close_completed_v1')
calls=[n.func.attr if isinstance(n.func,ast.Attribute)else n.func.id for n in ast.walk(apply)if isinstance(n,ast.Call)and isinstance(n.func,(ast.Name,ast.Attribute))]
check('measure_scopes'in calls and '_proof'in calls and '_append'in calls,'Original apply retains full measurement, proof, postabsence and append')
result=dict(schema='frozen_TRAIN001_ordinary_close_author_RAM_checks_v1',passed=True,positive=len(checks),negative=len(negative),checks=checks,negative_cases=negative,actual_input_ref=input_ref,actual_original_charge_bytes=7000000000,actual_original_consumer_bindings=61,actual_47roles_preserved_in_same61union=True,future_launch_wait_completion_outer_and_target_observation_are_RAM_only=True,no_actual_success_or_new_score_or_credit_claimed=True,exact_original_proof_AST_exercised_both_observe_modes=True,full_original_scope_spec_AST_exercised=True,actual_guard_imports_calls=0,process_group_absence_or_measurement_operations=0,accounting_or_Git_or_payload_operations=0,independent_review_claimed=False)
print(json.dumps(result,sort_keys=True,indent=2))
