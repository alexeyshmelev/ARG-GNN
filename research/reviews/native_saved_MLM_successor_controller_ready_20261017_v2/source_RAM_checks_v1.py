"""Author RAM-only checks for concrete empty-owner source. No operational imports."""
from pathlib import Path
import argparse,ast,contextlib,copy,hashlib,io,json,types
D=Path('research/reviews/native_saved_MLM_successor_controller_ready_20261017_v2')
C=Path('research/reviews/native_saved_MLM_current_Census64_integration_source_20261017_v2')
checks=[];negative=[]
def need(ok,name):
    if not ok:raise AssertionError(name)
    checks.append(name)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=(D/'empty_gpu0_controller_v1.py').read_text();tree=ast.parse(source)
env={'__name__':'RAM_SOURCE_ONLY'}
nodes=[n for n in tree.body if not(isinstance(n,ast.If)and any(isinstance(x,ast.Name)and x.id=='__name__'for x in ast.walk(n)))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'RAM_EMPTY_OWNER','exec',dont_inherit=True,optimize=0),env)
# Only stdlib and definitions were executed. No load/read/guard/probe/main runtime branch.
with contextlib.redirect_stdout(io.StringIO())as out:r=env['main']([])
need(r==0 and json.loads(out.getvalue())==dict(status='SOURCE_ONLY_INACTIVE',operations=0),'Actual default controller main inert')
parenttree=ast.parse((D/'owner_parent_v1.py').read_text());penv={'__name__':'RAM_PARENT','c':types.SimpleNamespace(HERE=str(D),need=env['need'])}
pnodes=[n for n in parenttree.body if not isinstance(n,ast.ImportFrom)and not isinstance(n,ast.If)]
exec(compile(ast.Module(body=pnodes,type_ignores=[]),'RAM_EMPTY_PARENT','exec',dont_inherit=True,optimize=0),penv)
with contextlib.redirect_stdout(io.StringIO())as out:r=penv['main']([])
need(r==0 and json.loads(out.getvalue())==dict(status='SOURCE_ONLY_INACTIVE',operations=0),'Actual default parent main inert')
manifest=json.loads((D/'source_manifest.v1.json').read_bytes())
for p,s in manifest['source_sha256'].items():need(digest(Path(p))==s,'Source closure '+p)
need(not any(p.startswith('.runtime/storage/')or Path(p).suffix in('.pt','.pth','.npy','.npz','.pkl','.trees')for p in manifest['source_sha256']),'No runtime journal or scientific array closure path')
for p in D.glob('*.py'):compile(p.read_bytes(),str(p),'exec',dont_inherit=True,optimize=0);need(True,'Compile '+p.name)
oldtree=ast.parse((C/'direct_gpu0_v1.py').read_text());newtree=ast.parse((D/'native_direct_gpu0.NEW_ENDPOINT.source.py').read_text())
def remove_endpoint(t):
    t=copy.deepcopy(t);t.body=[n for n in t.body if not isinstance(n,ast.Assign)or not any(isinstance(x,ast.Name)and x.id=='ENDPOINT_SOURCE_REF'for x in n.targets)];return t
need(ast.dump(remove_endpoint(oldtree),include_attributes=False)==ast.dump(remove_endpoint(newtree),include_attributes=False),'Native whole AST unchanged except exact ENDPOINT_SOURCE_REF assignment')
need(not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in('Popen','fork','kill','killpg')for n in ast.walk(tree)),'Owner source has no spawn or numeric signaling route')
need(not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in('kill','killpg')for n in ast.walk(parenttree)),'Parent has no numeric signal fallback')
need(any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='signal_identity'for n in ast.walk(parenttree)),'Parent retains original pidfd identity helper')
now=env['time'].time()
def fake_ref(label):return dict(path='RAM_ONLY/'+label+'.json',sha256=hashlib.sha256(label.encode()).hexdigest())
records={}
def put(label,value):
    r=fake_ref(label);records[r['path']]=copy.deepcopy(value);return r
source_ref=dict(path=env['SOURCE'],sha256='a'*64)
mf=put('manifest',dict(source_sha256={env['SOURCE']:'a'*64,env['C64_CONTRACT']:'b'*64}))
review=put('review',dict(independent=True,verdict='SOURCE_PASS',blockers=[],candidate_manifest_ref=mf))
rev=put('revocation',dict(RAM_ONLY=True));sig=put('signal',dict(RAM_ONLY=True));h058=put('H058',dict(RAM_ONLY=True))
cert=put('known_absence',dict(root_authorizer='root_agent',all_listed_known_roots_absent=True,historical_drain_proven=False,includes_old_controller=True,known_identities=[env['OLD_ID']],observed_at_utc='2026-10-17T00:00:00+00:00'))
base=dict(status='active',root_authorizer='root_agent',execution_authorized=True,template_only=False,
 created_at_utc='2026-10-01T00:00:00+00:00',expires_at_utc='2026-11-01T00:00:00+00:00')
handover=put('handover',dict(base,schema='ROOT_empty_GPU0_successor_handover_v1',old_controller_capability_ref=env['OLD_CAP'],old_controller_identity=env['OLD_ID'],future_old_controller_and_C3_routes_revoked=True,all_current_borrower_routes_revoked=True,historical_old_drain_proven=False,old_files_charges_failures_unknowns_preserved=True,saved_MLM_original_outer_exit_code=1,saved_MLM_full_fit_success_claimed=False,old_controller_revocation_ref=rev,old_controller_signal_ref=sig,H058_actual_terminal_ref=h058,known_root_absence_ref=cert))
renewal=put('renewal',dict(renewed_expires_at_utc='2026-11-02T00:00:00+00:00'))
cfg=put('config',dict(schema='ROOT_empty_GPU0_owner_config_v1',output_cap_bytes=32*2**20,physical_gpu=0,jobs={},submit_authorized=False,headroom_mib=4096,max_cpu_rss_bytes=2*2**30))
runtime=put('runtime',dict(schema='ROOT_empty_GPU0_owner_runtime_v1',python='.runtime/env/bin/python',source_ref=source_ref,CUDA_VISIBLE_DEVICES=''))
selector=fake_ref('actual_Census_selector')
storage=put('storage',dict(job_id='RAM-empty-owner',reserved_new_bytes=32*2**20,reservation=dict(auxiliary_scopes=[],consumer_bindings=[source_ref,mf,cfg,runtime]),output_directory='RAM_ONLY/output',storage_policy_selector_ref=selector))
plan=dict(base,schema='ROOT_empty_GPU0_owner_plan_v1',jobs={},submit_authorized=False,physical_gpu=0,gpu_uuid=env['UUID'],maximum_distinct_project_GPUs=1,maximum_concurrent_jobs=0,protected_envelope_bytes=0,headroom_mib=4096,controller_rss_cap_bytes=2*2**30,external_job_sharing_authorized=True,preserve_external_processes=True,borrowed_workers_use_separate_logical_gate=True,preserve_owner_until_explicit_downstream_drain=True,pool='ram-empty',expires_unix=now+1000,drain_expires_unix=now+1035,source_manifest_ref=mf,source_refs={env['SOURCE']:source_ref},source_review_ref=review,handover_ref=handover,controller_job_id='RAM-empty-owner',output_directory='RAM_ONLY/output',storage_policy_selector_ref=selector,config_ref=cfg,runtime_ref=runtime,marker_repair_renewal_ref=renewal,base_pool_ref=env['BASE_POOL'])
planref=put('plan',plan);capref=put('capability',dict(plan,schema='ROOT_empty_GPU0_owner_capability_v1',plan_ref=planref,controller_storage_ref=storage))
original=copy.deepcopy(records)
env['raw']=lambda r:b'RAM_ONLY_SOURCE_BOUNDARY'
env['read']=lambda r:copy.deepcopy(records[r['path']])
env['ref']=lambda path:mf if path==env['MANIFEST']else source_ref if path==env['SOURCE']else dict(path=path,sha256='b'*64)
need(env['validate_cap_v1'](capref)[0]['jobs']=={},'Complete actual empty capability validator positive in RAM')
def reject(name,mutate):
    records.clear();records.update(copy.deepcopy(original));mutate(records)
    try:env['validate_cap_v1'](capref)
    except(ValueError,KeyError,TypeError):negative.append(name)
    else:raise AssertionError('Missing rejection '+name)
def capchange(key,value):
    def apply(d):d[capref['path']][key]=value;d[planref['path']][key]=value
    return apply
reject('nonempty declared jobs',capchange('jobs',{'fake':{}}))
reject('submit enabled',capchange('submit_authorized',True))
reject('physicalGPU1',capchange('physical_gpu',1))
reject('nonzero declared compute slots',capchange('maximum_concurrent_jobs',1))
reject('old C3 protected26GiB envelope',capchange('protected_envelope_bytes',26*2**30))
reject('expired owner clock',capchange('expires_unix',now-1))
reject('insufficient original drain reserve',capchange('drain_expires_unix',now+1001))
reject('inactive owner',capchange('status','inactive'))
reject('template owner',capchange('template_only',True))
reject('failed complete source review',lambda d:d[review['path']].update(verdict='SOURCE_FAIL'))
reject('old source review does not bind new manifest',lambda d:d[review['path']].update(candidate_manifest_ref=fake_ref('old_manifest')))
reject('invented historical full drain',lambda d:d[handover['path']].update(historical_old_drain_proven=True))
reject('oldMLM outer0 relabel',lambda d:d[handover['path']].update(saved_MLM_original_outer_exit_code=0))
reject('oldMLM full-fit success relabel',lambda d:d[handover['path']].update(saved_MLM_full_fit_success_claimed=True))
reject('missing old controller revocation',lambda d:d[handover['path']].update(future_old_controller_and_C3_routes_revoked=False))
reject('missing old borrower route revocation',lambda d:d[handover['path']].update(all_current_borrower_routes_revoked=False))
reject('known absence certificate omits old owner',lambda d:d[cert['path']].update(known_identities=[]))
reject('wrong controller scope size',lambda d:d[storage['path']].update(reserved_new_bytes=64*2**20))
reject('missing actual producer direct binding',lambda d:d[storage['path']]['reservation'].update(consumer_bindings=[mf,cfg,runtime]))
reject('crossed policy selector',lambda d:d[storage['path']].update(storage_policy_selector_ref=fake_ref('wrong_selector')))
reject('configuration has hidden job',lambda d:d[cfg['path']].update(jobs={'hidden':{}}))
reject('runtime makes CUDA visible',lambda d:d[runtime['path']].update(CUDA_VISIBLE_DEVICES='0'))
reject('marker renewal expires before drain',lambda d:d[renewal['path']].update(renewed_expires_at_utc='2026-10-01T00:00:00+00:00'))
records.clear();records.update(copy.deepcopy(original))
# Pure dispatch preserves original wire shape. No file/socket boundary is crossed.
status=lambda:dict(jobs=[],physical_gpu=0)
answer,shutdown=env['dispatch_v1'](dict(v=1,op='status'),capref,plan,status,lambda p:None)
need(answer==status()and shutdown is False,'Exact status protocol positive')
release=dict(base,schema='ROOT_empty_GPU0_owner_downstream_release_v1',controller_capability_ref=capref,all_authorized_borrowers_actually_drained=True,all_future_borrower_routes_revoked=True,genuine_wait_and_outer_refs=[fake_ref('genuine_wait')])
dispatchcap=dict(plan,ROOT_downstream_release_path='RAM_ONLY/release.json')
answer,shutdown=env['dispatch_v1'](dict(v=1,op='shutdown',terminate=False),capref,dispatchcap,status,lambda p:release)
need(shutdown is True,'Exact shutdown requires separately authenticated ROOT release')
for value in [dict(v=1,op='submit',job='x'),dict(v=1,op='worker'),dict(v=1,op='cancel'),dict(v=1,op='shutdown',terminate=True),dict(v=True,op='status'),dict(v=1,op='status',extra=True),[]]:
    try:env['dispatch_v1'](value,capref,dispatchcap,status,lambda p:release)
    except(ValueError,KeyError,TypeError):negative.append('Wire rejected '+repr(value))
    else:raise AssertionError(value)
badrelease=dict(release,all_authorized_borrowers_actually_drained=False)
try:env['dispatch_v1'](dict(v=1,op='shutdown',terminate=False),capref,dispatchcap,status,lambda p:badrelease)
except ValueError:negative.append('Shutdown before actual borrower drain rejected')
else:raise AssertionError('Premature release')
# Actual original endpoint function, RAM base encoder only, must agree exactly.
oldsrc=Path('research/reviews/C3_native_worker_module_route_fix_source_20261016_v3/gpu0_declared_pool_v1.py').read_text()
oldnode=next(n for n in ast.parse(oldsrc).body if isinstance(n,ast.FunctionDef)and n.name=='socket_name')
oldenv=dict(hashlib=hashlib,base=types.SimpleNamespace(encoded=env['encoded']))
exec(compile(ast.Module(body=[oldnode],type_ignores=[]),'RAM_ORIGINAL_ENDPOINT','exec'),oldenv)
need(env['socket_name'](dict(pool='ram-empty',plan_ref=planref))==oldenv['socket_name'](dict(pool='ram-empty',plan_ref=planref)),'Original endpoint address and plan token preserved')
binding=json.loads((D/'binding_union.source_only.v1.json').read_text())
need(binding['returned_role_count']==47 and binding['unique_baseline_refs']==40 and binding['all_shapes_within64'],'Complete real baseline47roles/40refs union fits unchanged64 ceiling')
result=dict(schema='ROOT_empty_GPU0_owner_author_RAM_checks_v1',passed=True,positive_or_static_checks=len(checks),negative_checks=len(negative),checks=checks,negative_cases=negative,source_only=True,independent_review_claimed=False,actual_guard_imports_calls=0,process_socket_lock_inventory_probes=0,active_controls_written=0,accounting_or_runtime_operations=0,scientific_payload_reads_or_hashes=0,limits=['RAM boundaries do not establish actual resource fit, lease release or ROOT authorization.','Native endpoint patch must be adopted into a fresh independently reviewed complete namespace before execution.'])
print(json.dumps(result,sort_keys=True,indent=2))
