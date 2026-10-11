"""Independent source delta/memory guards plus contained author-controller replay."""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,json,os,sys,types
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/budgeted_shared_pool_v3_independent'
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
from arggnn import budgeted_shared_pool_v3 as m,budgeted_shared_pool_v2 as v2,gpu_pool_v1 as g
checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def check(n,v):
 if not v:raise AssertionError(n)
 checks.append(n)
def reject(n,call):
 try:call()
 except (ValueError,RuntimeError):checks.append(n);return
 raise AssertionError('unexpected acceptance '+n)
def write(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
pins={
'src/arggnn/budgeted_shared_pool_v3.py':'2dd01da79462e0c469a1671bd509f0f0ce0b7ac700e63dcd4922db9fc9e9ad13',
'configs/budgeted_shared_pool_v3.json':'d0e3dba441ecad299ff61b53a262e38b767e1d78118c7d351c0f96888b077c28',
'research/reviews/budgeted_shared_pool_v3_author/author_receipt.json':'bd71bebb914a32e08470da0544c234e0c4e9ae037aec51843a4fd382a5a311c2',
'src/arggnn/budgeted_shared_pool_v2.py':'16812adfc6a672ad0be157ff20b089597c9f0134ede24a14e5d7522bc4f32f72',
'configs/budgeted_shared_pool_v2.json':'42769ba1fc4e654c5aeba6370b74708549f05ff99a73f09022b208821a330327',
'src/arggnn/census_biological_fit_v2.py':'c21884ad90a2c940b49968353545edc52d345c0c7bc10fd87c492c54d4b38c34',
'configs/census_biological_fit_v2.json':'a512816bbab61e80052e7466613d2de4c861b77ef61d0d35cd50fbf81905c621',
'src/arggnn/census_biological_device_profile_v1.py':'f8915b311b0c1fbb219ff7265a37c46b037e1c49a95d2be85bd754623fd3d69f',
'configs/census_biological_device_profile_v1.json':'36c782b162881391822a528f7aef039a2c61fa4d232bfb5591da7bf9fe87f8d0'}
for p,h in pins.items():check('requested_pin_'+p,sha(ROOT/p)==h)
author=json.loads((ROOT/'research/reviews/budgeted_shared_pool_v3_author/author_receipt.json').read_text())
for r in author['artifacts']:check('sealed_author_artifact_'+r['path'],ref(ROOT/r['path'])==r)
s3=m.SOURCE.read_text();s2=v2.SOURCE.read_text();normalized=s3.replace('budgeted_shared_pool_v3','budgeted_shared_pool_v2')
old_line=next(x for x in s2.splitlines(True) if x.startswith('ADDITIONAL_WORKERS = '));new_line=next(x for x in normalized.splitlines(True) if x.startswith('ADDITIONAL_WORKERS = '))
check('complete_source_bytes_exact_version_bindings_and_one_worker_contract',normalized.replace(new_line,old_line)==s2)
def clean(source):
 tree=ast.parse(source)
 tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='ADDITIONAL_WORKERS' for x in n.targets))]
 return ast.dump(tree,include_attributes=False)
check('complete_normalized_AST_unchanged',clean(s3.replace('budgeted_shared_pool_v3','budgeted_shared_pool_v2'))==clean(s2))
newworker='arggnn.census_biological_fit_v2';expected=copy.deepcopy(v2.ADDITIONAL_WORKERS);expected[newworker]=copy.deepcopy(m.ADDITIONAL_WORKERS[newworker])
check('old_worker_contracts_preserved_and_only_fit2_added',m.ADDITIONAL_WORKERS==expected and set(m.ADDITIONAL_WORKERS)-set(v2.ADDITIONAL_WORKERS)=={newworker})
c3=json.loads(m.CONFIG.read_text());c2=json.loads(v2.CONFIG.read_text());wanted=copy.deepcopy(c2);wanted['kind']='budgeted_shared_pool_v3_policy';wanted['allowed_modules']=sorted(c2['allowed_modules']+[newworker]);wanted['additional_worker_contracts'][newworker]=m.ADDITIONAL_WORKERS[newworker]
check('entire_config_exact_only_kind_allowlist_fit2_contract',c3==wanted and c3==m.FIXED_POLICY)
check('fit2_contract_exact_sealed_source_config',m.ADDITIONAL_WORKERS[newworker]['source']==ref(ROOT/'src/arggnn/census_biological_fit_v2.py') and m.ADDITIONAL_WORKERS[newworker]['config']==ref(ROOT/'configs/census_biological_fit_v2.json') and m.ADDITIONAL_WORKERS[newworker]['shared_launch_allowed'] is True)
check('profile1_eligible_unchanged',m.PROFILE_MODULE in c3['allowed_modules'] and m.PROFILE_MODULE==v2.PROFILE_MODULE)
check('native5_shared_route_blocked_and_native6_absent',m.ADDITIONAL_WORKERS['arggnn.native_argformer_tsinfer_fit_v5']['shared_launch_allowed'] is False and 'arggnn.native_argformer_tsinfer_fit_v6' not in c3['allowed_modules'])
check('original_controller_cap_and_eligible_devices',g.ALLOWED_GPUS==(2,3,4,5) and c3['eligible_physical_gpus']==[2,3,4,5] and c3['maximum_physical_gpus']==2)
check('v4_950GB_schedule_1TB_hard_unchanged',c3['scheduling_ceiling_bytes']==950000000000 and c3['hard_limit_bytes']==1000000000000 and c3['budget_source']==c2['budget_source'] and c3['budget_config']==c2['budget_config'])
for k in ('legacy_pool_source','budget_source','budget_config','human_authorization','cache_source','cache_config'):
 r=c3[k];check('pinned_resource_authority_'+k,ref(ROOT/r['path'])==r)
for name in ('start','_process_scan','_memory','submit','_launch_queue','_drain_logs','_signal_own','tick','shutdown','_release_handles','close','dispatch','run'):
 check('original_controller_method_inherited_'+name,getattr(m.BudgetedController,name) is getattr(g.Controller,name))
# Independent arithmetic checks on the actual inherited scheduler memory method.
rows=[{'index':i,'uuid':'GPU-00000000-0000-0000-0000-'+('%012d'%i),'used_mib':0,'total_mib':100,'utilization':0,'compute':[]} for i in g.ALLOWED_GPUS]
obj=types.SimpleNamespace(gpu=2,metadata={'uuid':rows[0]['uuid']},device={'used_mib':4,'total_mib':100},headroom=10,jobs={'a':{'state':'launching','requested':{'peak_mib':40}},'b':{'state':'draining','requested':{'peak_mib':35}}},pid_jobs={},ambiguous=[],inventory=lambda:copy.deepcopy(rows))
view=g.Controller._memory(obj,11)
check('pending_CUDA_and_draining_peaks_retained',view['active_peak_mib']==75 and view['projected_mib']==100 and view['blocked_by']==[])
check('one_MiB_over_safe_capacity_blocks',g.Controller._memory(obj,12)['blocked_by']==['memory_headroom'])
obj.ambiguous=[999999];check('unknown_lease_descendant_blocks_low_usage','unattributed_descendants_or_lease_holders' in g.Controller._memory(obj)['blocked_by']);obj.ambiguous=[]
rows[0]['compute']=[{'pid':999999,'name':'foreign','used_mib':1}];rows[0]['used_mib']=1
check('foreign_GPU_worker_blocks_even_low_usage','foreign_compute_pids' in g.Controller._memory(obj)['blocked_by'])
obj.pid_jobs={111:'a'};rows[0].update(compute=[{'pid':111,'name':'owned','used_mib':41}],used_mib=41)
check('observed_worker_peak_overrun_blocks','worker_peak_overrun' in g.Controller._memory(obj)['blocked_by'])
rows[0].update(compute=[],used_mib=20);obj.pid_jobs={}
check('unattributed_current_memory_remains_charged','memory_headroom' in g.Controller._memory(obj)['blocked_by'])
rows[0].update(compute=[],used_mib=0);uuid=rows[0]['uuid'];rows[0]['uuid']='GPU-00000000-0000-0000-0000-000000009999'
reject('physical_UUID_substitution_rejected',lambda:g.Controller._memory(obj));rows[0]['uuid']=uuid
# Worker eligibility is transport only. Their own exact root gates remain in source.
fit_source=(ROOT/'src/arggnn/census_biological_fit_v2.py').read_text();profile_source=(ROOT/'src/arggnn/census_biological_device_profile_v1.py').read_text()
check('fit2_literal_admission_CLI_and_protocol_gate_preserved',all(x in fit_source for x in ("parser.add_argument('--admission',required=True)","parser.add_argument('--expected-admission-sha256',required=True)",'Fresh independent exact fit source/config review required','Bound full-tree backward profile and resource evidence required')))
check('profile_requires_separate_root_admission',all(x in profile_source for x in ('Exact root target-free CUDA profile admission required','--expected-admission-sha256')))
# Replay the sealed controller tests using exclusively this new independent fake root.
spec=importlib.util.spec_from_file_location('_independent_controller_replay',ROOT/'research/reviews/budgeted_shared_pool_v3_author/source_only_checks.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
fixture.REVIEW=OUT;fixture.F=OUT/'f';result=fixture.main()
check('contained_controller_replay_95_assertions_PASS',result['status']=='PASS' and result['check_count']==95 and len(set(result['checks']))==76)
check('replay_no_actual_workers_GPU_live_ledgers_or_arrays',result['details']['actual_worker_processes']==0 and result['actual_live_lock_or_gpu_execution_performed'] is False and result['actual_live_ledgers_or_device_queries_performed'] is False and result['actual_model_arrays_read'] is False)
check('replay_fake_resources_closed_removed',not fixture.F.exists())
check('no_scientific_libraries_imported','torch' not in sys.modules and 'numpy' not in sys.modules)
write('controller_replay.json',result)
for p,h in pins.items():check('final_pin_'+p,sha(ROOT/p)==h)
for r in author['artifacts']:check('final_author_artifact_'+r['path'],ref(ROOT/r['path'])==r)
report={'kind':'budgeted_shared_pool_v3_independent_source_checks','status':'PASS','independent_check_count':len(checks),'checks':checks,'controller_replay_assertions':result['check_count'],'controller_replay_distinct_names':len(set(result['checks'])),'controller_replay':ref(OUT/'controller_replay.json'),'pinned_inputs':pins,'whole_byte_and_config_delta_exact':True,'scheduler_changed':False,'fit2_and_profile_eligible_transport_only':True,'concrete_operational_plan_reviewed':False,'actual_GPU_workers_devices_live_budgets_or_scientific_imports':False,'actual_fake_scheduler_lock_and_socket_operations_only':True,'issues':[]}
write('review_checks.json',report)
print(json.dumps({'status':'PASS','independent_check_count':len(checks),'controller_replay_assertions':result['check_count'],'issues':[]},indent=2))
