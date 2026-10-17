"""Focused v8 AST/stdlib/RAM checks. Never invokes an actual guard/process/GPU/numerical operation."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,collections,copy,hashlib,importlib,json,os,signal,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
D='research/reviews/H058_GPU0_full_training_successor_source_20261017_v8'
B='research/reviews/H058_GPU0_full_training_successor_source_20261017_v7'
sys.path.insert(0,str(ROOT))
rows=[];negatives=0

def ck(name,condition=True):
    if not condition:raise AssertionError(name)
    rows.append(dict(name=name,status='PASS'))
def reject(name,operation,kind=Exception):
    global negatives
    try:operation()
    except kind:negatives+=1;ck(name)
    else:raise AssertionError(name)
def rf(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def read(p):return json.loads((ROOT/p).read_bytes())
def parse(p):return ast.parse((ROOT/p).read_text())
def fn(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
def emit(nodes,ns):
    m=ast.Module(body=copy.deepcopy(nodes),type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,'<H058_v8_source_RAM_only>','exec'),ns)
def normalize(s):return s.replace(D,B).replace(D.replace('/','.'),B.replace('/','.')).replace('H058_GPU0_v8_','H058_GPU0_v7_').replace('H058-GPU0-v8-','H058-GPU0-v7-')
for p in (ROOT/D).iterdir():
    if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
ck('all_python_sources_compile_without_importing_science')
package=D.replace('/','.')
c=importlib.import_module(package+'.common_v1');op=importlib.import_module(package+'.operational_v1')
d=op.driver_role_v1();p=op.producer_role_v1();ct=importlib.import_module(package+'.context_factory_v1');direct=importlib.import_module(package+'.direct_gpu0_v1')
ck('actual_driver_and_producer_are_distinct_namespaces',d is not p and d.__dict__ is not p.__dict__)
ck('both_roles_authenticate_same_actual_operational_file',d.SOURCE==p.SOURCE==c.DRIVER==D+'/operational_v1.py' and Path(d.__file__)==Path(p.__file__)==ROOT/c.DRIVER)
ck('actual_common_driver_context_direct_bind_v8',d.HERE==c.HERE==ct.HERE==direct.HERE==D and d.MANIFEST==c.MANIFEST and p.MANIFEST==D+'/producer_manifest.v1.json')
ck('actual_roles_equal_human_readable_audit_mirrors',op._DRIVER_SOURCE==(ROOT/D/'driver_role.source.txt').read_text() and op._PRODUCER_SOURCE==(ROOT/D/'producer_role.source.txt').read_text())
helper_ref=rf('research/reviews/H058_lean_math_guard_design_source_20261017_v1/lean_math_guard_v1.py')
ck('reviewed_155line_helper_is_embedded_byte_exact',hashlib.sha256(op._LEAN_MATH_SOURCE.encode()).hexdigest()==helper_ref['sha256']=='fa5bddb450d2e531859f8d0c62bc6a1e49f60e82359baf97adede3e5cdb86966')
review=read('research/reviews/H058_lean_math_guard_design_independent_20261017_v1/review.v1.json')
ck('helper_independent_SOURCE_PASS_is_prototype_scope_only',review['verdict']=='SOURCE_PASS' and review['blockers']==[])
for name in ('__init__.py','direct_gpu0_v1.py','observations_gpu_v1.py'):
    ck('unchanged_whole_AST_except_v8_namespace:'+name,dump(ast.parse(normalize((ROOT/D/name).read_text())))==dump(parse(B+'/'+name)))
oldp=parse(B+'/producer_v1.py');newp=ast.parse(op._PRODUCER_SOURCE)
# Recover exact original producer AST by reversing only shared identity and reviewed train_one callback wrapper.
restored=copy.deepcopy(newp)
for n in restored.body:
    if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SOURCE'for t in n.targets):n.value=ast.parse('HERE+"/producer_v1.py"',mode='eval').body
step=fn(restored,'scheduled_step_v1');block=next(n for n in step.body if isinstance(n,ast.Try))
start=next(i for i,n in enumerate(block.body)if isinstance(n,ast.FunctionDef)and n.name=='actual_step')
block.body[start:start+2]=copy.deepcopy(next(n for n in fn(oldp,'scheduled_step_v1').body if isinstance(n,ast.Try)).body[1:2])
ck('whole_scientific_producer_AST_recovers_original_after_exact_reviewed_guard_wrapper',normalize(dump(restored))==dump(oldp))
oldct=parse(B+'/context_factory_v1.py');newct=parse(D+'/context_factory_v1.py')
ck('original_context_full_guard_AST_is_unchanged',dump(fn(ast.Module(body=fn(newct,'build_context_v1').body,type_ignores=[]),'guard'))==dump(fn(ast.Module(body=fn(oldct,'build_context_v1').body,type_ignores=[]),'guard')))
newbody=fn(newct,'build_context_v1').body
full=fn(ast.Module(body=newbody,type_ignores=[]),'guard');math=copy.deepcopy(fn(ast.Module(body=newbody,type_ignores=[]),'math_guard'))
expected=copy.deepcopy(full);expected.name='math_guard';expected.body[-1]=ast.Expr(value=copy.deepcopy(expected.body[-1].value.values[0])) if isinstance(expected.body[-1].value,ast.Tuple)else expected.body[-1]
# Source separates the census and final full budget check with a semicolon.
ck('math_guard_omits_only_final_recursive_budget_check',dump(ast.Module(body=math.body,type_ignores=[]))==dump(ast.Module(body=full.body[:-1],type_ignores=[])))
ck('both_scheduled_cache_validations_remain',sum(isinstance(n,ast.Call)and ast.unparse(n.func)=='observations.validate_scheduled_cache_v1'for n in ast.walk(fn(newp,'scheduled_step_v1')))==2)
ck('original_science_train_one_arguments_and_count_callback_retained','h45.train_one(value,optimizer,bank,mass,payload,bound_guard,\n                qualify=first,after_optimizer_step=counted)'in op._PRODUCER_SOURCE)
newd=ast.parse(op._DRIVER_SOURCE);oldd=parse(B+'/root_driver_v1.py')
ck('RSS_identity_monitor_remains_exact',dump(fn(newd,'rss_identity_v1'))==dump(fn(oldd,'rss_identity_v1')))
ck('supervisor_capacity_and_attachment_calls_and_2s_poll_unchanged',all(x in op._DRIVER_SOURCE for x in ('direct.capacity_v1(q,direct.base_v1(),worker_pid=proc.pid)','direct.check_attachment_v1(q,ticket,physical_fd,logical_fd)','capacity_samples+=1;next_capacity=time.monotonic()+2')))
ck('pre_Torch_cache_JIT_controls_in_child_environment',all(x in op._DRIVER_SOURCE for x in ('CUDA_CACHE_DISABLE="1"','USE_PYTORCH_KERNEL_CACHE="0"','PYTORCH_JIT="0"','TORCH_COMPILE_DISABLE="1"','TORCHDYNAMO_DISABLE="1"')))
command=next(n for n in ast.walk(fn(newd,'execute_v1'))if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='command'for t in n.targets))
ns=dict(vars(d),input_ref=dict(path='RAM/input.json',sha256='0'*64),gate_read=17);emit([command],ns)
ck('actual_emitted_child_argv_routes_to_shared_v8_module',ns['command'][3]==package+'.operational_v1')
ret=next(n for n in reversed(fn(newd,'admit_v1').body)if isinstance(n,ast.Return));ns.update(admitted_ref=dict(path='RAM/admitted.json',sha256='1'*64))
ans=eval(compile(ast.Expression(ret.value),'<admission_return_only>','eval'),ns)
ck('actual_admission_exec_argv_routes_to_shared_v8_module',ans['one_shot_exec_argv'][4]==package+'.operational_v1')
# Original ancestry algorithm, fake process table only.
ownedtree=parse(c.OWNED['path']);backend_ns=dict(os=SimpleNamespace(getpid=lambda:1),_stat=None,_children=None)
emit([fn(ownedtree,'_same'),next(n for n in ownedtree.body if isinstance(n,ast.ClassDef)and n.name=='OwnedTree')],backend_ns)
def process(pid,start,ppid=100,group=100,state='R'):return dict(pid=pid,start=start,ppid=ppid,group=group,session=100,state=state)
def fixture():
    table={100:process(100,10,ppid=1)};children={1:[100],100:[]}
    backend=SimpleNamespace(OwnedTree=backend_ns['OwnedTree'],_same=backend_ns['_same'])
    tree=d.OwnedLifecycleV8(backend,100,owner=1,read_stat=lambda pid:table.get(pid),read_children=lambda pid:children.get(pid,[]))
    return tree,table,children
tr,table,ch=fixture()
for k in range(20000):
    table.pop(1000+k-1,None);table[1000+k]=process(1000+k,100+k);ch[100]=[1000+k];tr.refresh()
ck('20000_sequential_helpers_exceed_64_and_4096_without_historical_pending_growth',tr.discovered.total==20001 and len(tr.known)==2 and tr.retired.total==19999)
ck('retirement_evidence_stays_compact_with_exact_counts_and_chained_digest',len(json.dumps(tr.evidence()))<20000 and len(tr.retired.first)==len(tr.retired.last)==16 and len(tr.retired.digest)==32)
tr,table,ch=fixture();table[200]=process(200,20);ch[100]=[200];tr.refresh();table[200]=process(200,21,ppid=999);ch[100]=[];tr.refresh()
ck('unrelated_reused_numeric_PID_excluded_and_old_identity_retired',200 not in tr.known and tr.retirement_reasons['replaced']==1)
table[200]=process(200,22);ch[100]=[200];tr.refresh()
ck('genuinely_new_owned_identity_can_reuse_numeric_PID_via_original_ancestry',tr.known[200]['start']==22 and tr.discovered.total==3)
table[200]['state']='Z';tr.refresh();ck('zombie_exact_identity_stays_pending_until_actual_reap_or_absence',200 in tr.known and not tr.identities_absent())
table.pop(200);ch[100]=[];tr.refresh();ck('reaped_absent_zombie_then_retires',200 not in tr.known)
limits=dict(d.OWNED_LIMITS_V8)
for key,value in [('pending',2),('discoveries',2),('groups',1)]:
    tr,table,ch=fixture();d.OWNED_LIMITS_V8[key]=value
    table.update({200:process(200,20,group=200 if key=='groups'else 100),201:process(201,21)});ch[100]=[200,201]
    reject('finite_'+key+'_overflow_fails_closed',tr.refresh,ValueError)
    ck('overflow_'+key+'_never_claims_complete_ownership',tr.complete is False and len(tr.known)<=d.OWNED_LIMITS_V8['pending'])
    table.clear();ch.clear();tr.refresh();ck('overflow_'+key+'_cannot_later_claim_absence_success',not tr.identities_absent())
    d.OWNED_LIMITS_V8.clear();d.OWNED_LIMITS_V8.update(limits)
tr,table,ch=fixture()
def inaccessible(pid):raise PermissionError('RAM metadata denied')
tr.original.read_stat=inaccessible;reject('refresh_error_preserves_pending_exact_identity',tr.refresh,PermissionError)
ck('refresh_error_marks_incomplete_without_discarding_known_signal_targets',tr.complete is False and list(tr.known)==[100])
# Execute the actual cleanup while body in a fake OS/time namespace. No process is created/signalled/reaped.
execute=fn(newd,'execute_v1');finally_block=next(n for n in execute.body if isinstance(n,ast.Try)).finalbody
cleanup=next(n for n in next(n for n in finally_block if isinstance(n,ast.If)and ast.unparse(n.test)=='proc is not None').body if isinstance(n,ast.While))
clock=[0.];calls=[];waits=collections.deque([(100,15)]+[(i,0 if i!=250 else 256)for i in range(200,300)])
def mono():clock[0]+=.001;return clock[0]
def waitpid(pid,flags):
    if waits:return waits.popleft()
    raise ChildProcessError()
failed_tree=SimpleNamespace(known={100:process(100,10,ppid=1)},complete=False,refresh=lambda:(_ for _ in ()).throw(ValueError('RAM refresh failure')))
errors=[];ns=dict(time=SimpleNamespace(monotonic=mono,sleep=lambda value:clock.__setitem__(0,clock[0]+value)),cleanup_deadline=1.,cleanup_started=0.,raw_status=None,proc=SimpleNamespace(pid=100,returncode=None),os=SimpleNamespace(waitpid=waitpid,WNOHANG=1,waitstatus_to_exitcode=os.waitstatus_to_exitcode),require=c.require,tree=failed_tree,owned=SimpleNamespace(signal_identity=lambda row,sig:calls.append((row['pid'],sig))),signal=signal,note_cleanup=lambda exc:errors.append(str(exc)),adopted=d._HistoryV8(),OWNED_LIMITS_V8=dict(d.OWNED_LIMITS_V8,adopted_waits=2),failure=None,retain_pipe=lambda:None,pipe_eof=False)
emit([cleanup],ns)
ck('actual_cleanup_keeps_signalling_known_PIDFD_targets_after_refresh_failure',bool(calls) and all(pid==100 for pid,_ in calls))
ck('actual_cleanup_preserves_genuine_direct_wait15_minus15',ns['raw_status']==15 and ns['proc'].returncode==-15)
ck('adopted_wait_summary_limit_does_not_stop_genuine_reaping',ns['adopted'].total==100 and not waits and ns['adopted'].nonzero==1)
ck('adopted_nonzero_status_remains_in_digest_summary_and_prevents_success',ns['failure'] is not None and ns['adopted'].evidence()['nonzero_returncodes']==1)
ck('cleanup_mock_stays_inside_original_aggregate_deadline',clock[0]<1.1)
# Targeted group absence proof is bounded, covers all retained groups and never signals a group.
ns=dict(vars(d));tick=[0.];probes=[]
ns['time']=SimpleNamespace(monotonic=lambda:tick[0]);ns['os']=SimpleNamespace(killpg=lambda group,sig:(probes.append((group,sig)),(_ for _ in ()).throw(ProcessLookupError()))[-1])
emit([fn(newd,'observe_groups_v8')],ns);scan=ns['observe_groups_v8'](SimpleNamespace(groups={100,200,300}),1.)
ck('every_retained_group_has_actual_targeted_ESRCH_observation',scan['all_owned_groups_observed_absent'] and scan['absent_group_count']==3 and probes==[(100,0),(200,0),(300,0)])
tick[0]=1.;probes.clear();scan=ns['observe_groups_v8'](SimpleNamespace(groups={100}),1.)
ck('expired_cleanup_never_starts_group_probe_or_invents_absence',not probes and not scan['all_owned_groups_observed_absent'])
tick[0]=0.;probes.clear()
def late_probe(group,sig):
    probes.append((group,sig));tick[0]=2.;raise ProcessLookupError()
ns['os']=SimpleNamespace(killpg=late_probe)
scan=ns['observe_groups_v8'](SimpleNamespace(groups={100}),1.)
ck('final_group_probe_finishing_after_deadline_cannot_claim_complete_drain',probes==[(100,0)] and scan['absent_group_count']==1 and not scan['scan_finished_within_deadline'] and not scan['all_owned_groups_observed_absent'])
# Exact actual admission operand union, immutable old guard-role metadata + one current Census capsule; no state query.
binding=read('research/reviews/H059_outer000_Relate64_execution_source_20261017_v1/source_bindings.v1.json');snapshot=binding['guard_role_snapshot_ref'];ck('immutable_guard_snapshot_hash',rf(snapshot['path'])==snapshot)
guards=copy.deepcopy(read(snapshot['path'])['bindings']);guards['Census_observation_explicit_selector_capsule']=c.SELECTOR_REF
virtual={}
def vr(path):
    if (ROOT/path).is_file():return rf(path)
    virtual[path]=(path+' RAM NOT AUTHORITY').encode();return dict(path=path,sha256=hashlib.sha256(virtual[path]).hexdigest())
q={k:vr('RAM_NOT_AUTHORITY/'+k+'.json')for k in ('config_ref','science_handoff_ref','source_review_ref','direct_attachment_authority_ref','native_resource_disposition_ref')};q['phase']='train'
ns=dict(vars(d),q=q,cfg=dict(source_manifest_ref=rf(D+'/producer_manifest.v1.json')),producer=rf(c.DRIVER),request_ref=vr('RAM_NOT_AUTHORITY/request.json'),runtime=vr('RAM_NOT_AUTHORITY/runtime.json'),closure=vr('RAM_NOT_AUTHORITY/closure.json'),ref=vr)
admit=fn(newd,'admit_v1');nodes=[n for n in admit.body if ((isinstance(n,ast.Assign)and any(isinstance(k,ast.Name)and k.id=='needed'for k in n.targets))or(isinstance(n,ast.AugAssign)and isinstance(n.target,ast.Name)and n.target.id=='needed')or(isinstance(n,ast.If)and ast.unparse(n.test)=="q['phase'] == 'train'"))];emit(nodes,ns)
consumer=parse(c.GUARD['path']);unions=dict(copy=copy,require=c.require,_raw=lambda r:virtual[r['path']]if r['path']in virtual else c.raw(r));emit([fn(consumer,'consumer_bindings_v1')],unions)
joined=unions['consumer_bindings_v1'](list(guards.values()),ns['needed'])
ck('actual_Census_original_64_ceiling_and_actual_needed_operands_fit_exactly',len(guards)==47 and len(joined)==64)
reject('one_unconsolidated_producer_role_would_reproduce_65ref_blocker',lambda:unions['consumer_bindings_v1'](joined,[vr('RAM_NOT_AUTHORITY/separate_producer.py')]),ValueError)
# Fresh27h clock/source join, exact current directive and original historical fields: metadata only.
now=datetime.now(timezone.utc);q=read(D+'/ROOT_request.INACTIVE.template.v1.json');q.update(status='active',execution_authorized=True,created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(hours=27)).isoformat(),adapter_manifest_ref=rf(c.MANIFEST),historical_ROOT_clocks_v8=c.historical_clocks_v8(),fit_clock_compatibility_v5=c.fit_clock_acceptance_v5())
cfg=read(D+'/scientific_config.INACTIVE.template.v1.json');cfg['producer_ref']=rf(c.DRIVER);cfg['full_training_directive_ref']=vr('RAM_NOT_AUTHORITY/current_directive.json')
original=dict(root_authorizer='root_agent',caps=c.CAPS,config_ref=vr('RAM_NOT_AUTHORITY/oldcfg.json'),created_at_utc='2026-10-17T07:19:39.628628+00:00',expires_at_utc='2026-10-18T10:05:49.430020+00:00')
oldcfg=dict(full_training_directive_ref=c.CLOCK_ROOT_DIRECTIVE_V4,scientific_protocol=cfg['scientific_protocol'])
admitted=dict(root_authorizer='root_agent',root_stage_request_ref=c.CLOCK_ROOT_REQUEST_V4,refs=dict(config=original['config_ref']),**{k:original[k]for k in ('created_at_utc','expires_at_utc')})
old_directive=dict(root_authorizer='root_agent',scientific_protocol=cfg['scientific_protocol'],full30pass_training_authorized=True,extra_probe_updates_authorized=False,**{k:original[k]for k in ('created_at_utc','expires_at_utc')})
current=dict(old_directive,status='active',created_at_utc=q['created_at_utc'],expires_at_utc=q['expires_at_utc'],device='cuda',physical_gpu=0,GPU_entry_ref=rf(c.DRIVER),new_finite_v8_interval_authorized=True,historical_ROOT_clocks_v8=c.historical_clocks_v8(),historical_files_scopes_charges_clocks_failures_and_unknowns_preserved=True)
docs={c.CLOCK_ROOT_REQUEST_V4['path']:original,c.CLOCK_ROOT_INPUT_V4['path']:admitted,c.CLOCK_ROOT_DIRECTIVE_V4['path']:old_directive,original['config_ref']['path']:oldcfg,cfg['full_training_directive_ref']['path']:current}
ns=dict(vars(c));ns['read']=lambda r:docs[r['path']]if r['path']in docs else c.read(r);emit([fn(parse(D+'/common_v1.py'),'validate_root_fit_clock_v5')],ns)
ck('actual_clock_join_accepts_new27h_interval_and_preserves_original_clock_fields',ns['validate_root_fit_clock_v5'](q,cfg)==current)
bad=copy.deepcopy(q);bad['expires_at_utc']=(now+timedelta(hours=27,seconds=1)).isoformat();reject('fresh_interval_cannot_silently_change_duration',lambda:ns['validate_root_fit_clock_v5'](bad,cfg),ValueError)
bad=copy.deepcopy(q);bad['historical_ROOT_clocks_v8']['v7']['expires_at_utc']=bad['expires_at_utc'];reject('new_v8_clock_cannot_rewrite_v7_historical_clock',lambda:ns['validate_root_fit_clock_v5'](bad,cfg),ValueError)
current['extra_probe_updates_authorized']=True;reject('no_extra_probe_updates_on_fresh_clock_authority',lambda:ns['validate_root_fit_clock_v5'](q,cfg),ValueError);current['extra_probe_updates_authorized']=False
request_tree=fn(parse(D+'/common_v1.py'),'validate_request_v1');fresh=next(n for n in request_tree.body if isinstance(n,ast.If)and ast.unparse(n.test)=='not admitted_input')
ns=dict(vars(c),q=dict(expires_at_utc=(now+timedelta(hours=23)).isoformat()),admitted_input=False);reject('fresh_admission_still_demands_full24h_plus_margin',lambda:emit([fresh],ns),ValueError);ns['admitted_input']=True;emit([fresh],ns);ck('already_admitted_run_does_not_reapply_fresh24h_test')
# Prior v7 join uses only exact permitted wait/minimal/log plus future ROOT records in RAM.
prior=read(D+'/prior_CPU_disposition.INACTIVE.template.v1.json');prior.update(prior_GPU_v7_actual_wait_ref=c.PRIOR_V7_ACTUAL_WAIT_REF,prior_GPU_v7_actual_wait_minimal_ref=c.PRIOR_V7_MINIMAL_WAIT_REF,prior_GPU_v7_log_ref=c.PRIOR_V7_LOG_REF,prior_GPU_v7_input_ref=c.PRIOR_V7_INPUT_REF,prior_GPU_v7_files_scopes_full_charges_clocks_sources_failures_and_unknowns_preserved=True,prior_GPU_v7_future_launch_resume_write_routes_permanently_revoked=True,prior_GPU_v7_actual_optimizer_steps={'same_marker_aligned':None,'same_marker_alignment_null':None},prior_GPU_v7_final_optimizer_totals_status='UNKNOWN_no_new_update_verified',prior_GPU_v7_verified_optimizer_steps=0,historical_ROOT_clocks_v8=c.historical_clocks_v8(),actual_prior_clocks_match_immutable_source_lineage=True,new_full_1920_update_unfinished_cell_registration=True,new_gpu_request_job_id=q['job_id'],no_CPU_or_GPU_optimizer_state_transfer=True,new_GPU_budget=1920,new_GPU_budget_per_arm=960,prior_GPU_v7_actual_outer_terminal_ref=vr('RAM_NOT_AUTHORITY/v7_outer.json'),fresh_targeted_v7_absence_ref=vr('RAM_NOT_AUTHORITY/v7_absence.json'))
outer=dict(schema='ROOT_H058_GPU0_v7_actual_outer_terminal_v1',root_authorizer='root_agent',actual_terminal_consumed=True,actual_outer_exit_code=241,actual_direct_worker_returncode=-15,actual_wait_ref=c.PRIOR_V7_ACTUAL_WAIT_REF,actual_wait_minimal_ref=c.PRIOR_V7_MINIMAL_WAIT_REF,log_ref=c.PRIOR_V7_LOG_REF,completed_ref=None,historical_roster_truncation_preserved=True,successful_outer_terminal_claimed=False)
absence=dict(root_authorizer='root_agent',actual_outer_terminal_ref=prior['prior_GPU_v7_actual_outer_terminal_ref'],actual_wait_ref=c.PRIOR_V7_ACTUAL_WAIT_REF,actual_supervisor_terminal_consumed=True,current_known_v7_identities_groups_and_aux_writers_absent=True,historical_roster_was_truncated=True,historical_complete_roster_claimed=False,remaining_historical_ownership_uncertainty_ROOT_accepted_for_fresh_scope=True,all_v7_launch_resume_write_routes_permanently_revoked=True,observed_at_utc=now.isoformat())
ns=dict(vars(c));ns['read']=lambda r:outer if r==prior['prior_GPU_v7_actual_outer_terminal_ref']else absence if r==prior['fresh_targeted_v7_absence_ref']else c.read(r);emit([fn(parse(D+'/common_v1.py'),'validate_prior_v7_terminal_v8')],ns)
ck('actual_prior_v7_predicate_joins_exact_allowed_packet_to_future_ROOT_RAM_terminal',ns['validate_prior_v7_terminal_v8'](prior,q)==prior)
for key in ('actual_terminal_consumed','historical_roster_truncation_preserved'):
    outer[key]=False;reject('prior_v7_requires_'+key,lambda:ns['validate_prior_v7_terminal_v8'](prior,q),ValueError);outer[key]=True
absence['historical_complete_roster_claimed']=True;reject('cannot_invent_complete_v7_historical_roster',lambda:ns['validate_prior_v7_terminal_v8'](prior,q),ValueError);absence['historical_complete_roster_claimed']=False
bad=copy.deepcopy(prior);bad['prior_GPU_v7_actual_optimizer_steps']={a:0 for a in bad['prior_GPU_v7_actual_optimizer_steps']};reject('zero_verified_never_becomes_zero_actual_updates',lambda:ns['validate_prior_v7_terminal_v8'](bad,q),ValueError)
absence['observed_at_utc']=(now-timedelta(seconds=301)).isoformat();reject('new_admission_demands_fresh_v7_absence',lambda:ns['validate_prior_v7_terminal_v8'](prior,q),ValueError);ns['validate_prior_v7_terminal_v8'](prior,q,fresh=False);ck('historical_absence_does_not_expire_already_admitted_input')
manifest=read(D+'/source_manifest.v1.json');oldmanifest=read(B+'/source_manifest.v1.json')
ck('all_original355_closure_refs_preserved_exactly',all(r in manifest['refs']for r in oldmanifest['refs']))
for rr in manifest['refs']:
    if rr['path'].startswith(D+'/'):ck('new_source_pin:'+Path(rr['path']).name,rf(rr['path'])==rr)
ck('source_hash_map_agrees_with_declared_refs',manifest['source_sha256']=={r['path']:r['sha256']for r in manifest['refs']})
for path,h in read(D+'/sealed_v7_baseline.v8.json').items():ck('sealed_v7_unchanged:'+Path(path).name,rf(path)['sha256']==h)
ck('no_Torch_PyG_Numpy_imported_by_inert_actual_role_entry',not any(n=='torch'or n.startswith(('torch.','torch_geometric','numpy'))for n in sys.modules))
result=dict(schema='H058_GPU0_v8_focused_source_AST_RAM_checks_v1',status='PASS',source_only=True,independent=False,assertions=len(rows),negative_fixtures=negatives,sequential_helper_fixture_discoveries=20001,current_pending_after_churn=2,guard_roles=47,actual_admission_union=64,source_ref=rf(D+'/operational_v1.py'),source_manifest_ref=rf(D+'/source_manifest.v1.json'),checks=rows,no_actual_guard_state_process_GPU_payload_checkpoint_truth_or_numerical_execution=True)
if '--emit-result' in sys.argv:
    with (ROOT/D/'checks.result.v8.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k!='checks'},sort_keys=True))
