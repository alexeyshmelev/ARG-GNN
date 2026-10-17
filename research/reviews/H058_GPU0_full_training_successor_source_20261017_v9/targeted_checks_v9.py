"""Focused H058 v9 changed-group and cleanup regression cases; RAM only, no churn campaign."""
from pathlib import Path
from types import SimpleNamespace
import ast,collections,copy,hashlib,importlib,json,os,signal,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D='research/reviews/H058_GPU0_full_training_successor_source_20261017_v9';sys.path.insert(0,str(ROOT));c=importlib.import_module(D.replace('/','.')+'.common_v1');op=importlib.import_module(D.replace('/','.')+'.operational_v1');d=op.driver_role_v1()
checks=[];negative=0

def ck(n,v=True):
 if not v:raise AssertionError(n)
 checks.append(dict(name=n,status='PASS'))
def reject(n,f):
 global negative
 try:f()
 except ValueError:negative+=1;ck(n)
 else:raise AssertionError(n)
def emit(nodes,scope):exec(compile(ast.Module(body=copy.deepcopy(nodes),type_ignores=[]),'<H058_v9_material_group_RAM>','exec'),scope)
def fn(t,n):return next(k for k in t.body if isinstance(k,ast.FunctionDef)and k.name==n)
def rf(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def row(pid,start,ppid=1000,group=1000):return dict(pid=pid,start=start,ppid=ppid,group=group,session=1000,state='R')
ot=ast.parse((ROOT/c.OWNED['path']).read_text());bs=dict(_stat=None,_children=None,os=SimpleNamespace(getpid=lambda:1,close=lambda fd:None));emit([n for n in ot.body if(isinstance(n,ast.FunctionDef)and n.name in('_same','signal_identity'))or(isinstance(n,ast.ClassDef)and n.name=='OwnedTree')],bs)
def fixture():
 table={1000:row(1000,1,ppid=1),2000:row(2000,2)};children={1:[1000],1000:[2000]};backend=SimpleNamespace(OwnedTree=bs['OwnedTree'],_same=bs['_same']);tree=d.OwnedLifecycleV8(backend,1000,owner=1,read_stat=lambda pid:table.get(pid),read_children=lambda pid:children.get(pid,[]));tree.refresh();return tree,table,children
tr,table,ch=fixture();before=tr.discovered.total;table[2000]=row(2000,2,group=2000);tr.refresh();ck('valid_owned_child_same_PID_start_changes_PGID_and_retains_both',tr.groups=={1000,2000}and tr.known[2000]['group']==2000 and tr.discovered.total==before)
limits=dict(d.OWNED_LIMITS_V8);d.OWNED_LIMITS_V8['groups']=2;snapshot=(copy.deepcopy(tr.known),set(tr.groups),tr.discovered.total,tr.retired.total);table[2000]=row(2000,2,group=3000);reject('new_group_over_limit_rejected_before_row_set_or_history_change',tr.refresh);ck('overflow_is_permanently_incomplete_and_preserves_previous_bounded_state',not tr.complete and snapshot==(dict(tr.known),set(tr.groups),tr.discovered.total,tr.retired.total))
opened=[];sent=[];closed=[]
ck('actual_original_pidfd_signal_still_stops_known_identity_after_group_overflow',bs['signal_identity'](tr.known[2000],15,read_stat=tr.cleanup_read_stat_v9,open_fd=lambda pid,flags:(opened.append(pid)or 91),send_fd=lambda fd,sig,*args:sent.append((fd,sig)),close_fd=lambda fd:closed.append(fd))and opened==[2000]and sent==[(91,15)]and closed==[91]and tr.complete is False)
d.OWNED_LIMITS_V8.clear();d.OWNED_LIMITS_V8.update(limits)
tr,table,ch=fixture();sequence=collections.deque([row(2000,2,group=2000),None]);base=tr._original_read_stat;tr._original_read_stat=lambda pid:sequence.popleft()if pid==2000 and sequence else None if pid==2000 else base(pid);tr.refresh();ck('group_observed_in_retire_stat_is_retained_if_identity_disappears_before_refresh_set',2000 not in tr.known and tr.groups=={1000,2000})
tr,table,ch=fixture();tr.refresh=lambda:tr;table[2000]=row(2000,2,group=2000);ck('live_stat_observation_also_retains_new_group',2000 in tr.live()and tr.groups=={1000,2000})
tr,table,ch=fixture();table[2000]=row(2000,2,group=2000);backend=SimpleNamespace(_stat=tr.read_stat_v9,_same=bs['_same']);ns=dict(vars(d),Path=lambda p:SimpleNamespace(read_text=lambda:'0 7'),os=SimpleNamespace(sysconf=lambda key:4096));emit([fn(ast.parse(op._DRIVER_SOURCE),'rss_identity_v1')],ns);ck('actual_RSS_monitor_keeps_exact_PID_start_and_records_observed_group',ns['rss_identity_v1'](backend,tr.known[2000])==7*4096 and tr.groups=={1000,2000})
tr,table,ch=fixture();table[2000]=row(2000,3,ppid=999,group=3000);ch[1000]=[];tr.refresh();ck('unrelated_reused_PID_new_group_is_never_owned_or_retained',2000 not in tr.known and tr.groups=={1000})
tr,table,ch=fixture();table[2000]=row(2000,2,group=2000);tr.refresh();probes=[]
def present_new(group,sig):
 probes.append((group,sig))
 if group==1000:raise ProcessLookupError()
ns=dict(vars(d),os=SimpleNamespace(killpg=present_new),time=SimpleNamespace(monotonic=lambda:0.));emit([fn(ast.parse(op._DRIVER_SOURCE),'observe_groups_v8')],ns);scan=ns['observe_groups_v8'](tr,1.);ck('actual_final_group_scan_probes_both_groups_and_live_new_group_blocks_drain',probes==[(1000,0),(2000,0)]and not scan['all_owned_groups_observed_absent'])
# Execute exact cleanup while body after strict group overflow, with genuine-status RAM wait/fd doubles.
tr,table,ch=fixture();d.OWNED_LIMITS_V8['groups']=1;table[2000]=row(2000,2,group=2000);reject('cleanup_fixture_starts_from_actual_strict_observation_overflow',tr.refresh)
clock=[0.];signals=[];waited=[];adopted_ready=[False]
def mono():clock[0]+=.001;return clock[0]
def waitpid(pid,flags):
 if pid==1000 and 1000 in table:table.pop(1000);ch[1]=[2000];waited.append((1000,0));return 1000,0
 if pid==-1 and adopted_ready[0]:adopted_ready[0]=False;ch[1]=[];waited.append((2000,0));return 2000,0
 if pid==-1:raise ChildProcessError()
 return 0,0
def send(fd,sig,*args):signals.append((fd,sig));table.pop(2000,None);adopted_ready[0]=True
bs['pidfd_open_v1']=lambda pid,flags:91;bs['pidfd_send_signal_v1']=send
owned=SimpleNamespace(signal_identity=bs['signal_identity'],_stat=lambda pid:table.get(pid),_children=lambda pid:ch.get(pid,[]));dt=ast.parse(op._DRIVER_SOURCE);execute=fn(dt,'execute_v1');finally_body=next(n for n in execute.body if isinstance(n,ast.Try)).finalbody;cleanup=next(n for n in next(n for n in finally_body if isinstance(n,ast.If)and ast.unparse(n.test)=='proc is not None').body if isinstance(n,ast.While));errors=[]
ns=dict(time=SimpleNamespace(monotonic=mono,sleep=lambda delay:clock.__setitem__(0,clock[0]+delay)),cleanup_deadline=1.,cleanup_started=0.,raw_status=None,proc=SimpleNamespace(pid=1000,returncode=None),os=SimpleNamespace(waitpid=waitpid,WNOHANG=1,waitstatus_to_exitcode=os.waitstatus_to_exitcode,getpid=lambda:1),require=c.require,tree=tr,owned=owned,signal=signal,note_cleanup=lambda exc:errors.append(str(exc)),adopted=d._HistoryV8(),OWNED_LIMITS_V8=d.OWNED_LIMITS_V8,failure=None,retain_pipe=lambda:None,pipe_eof=True,group_scan=None,observations=[],identity_absence=None);emit([cleanup],ns)
ck('actual_cleanup_loop_signals_child_through_cleanup_only_stat_path_after_overflow',bool(signals)and all(fd==91 for fd,sig in signals))
ck('actual_cleanup_loop_preserves_direct_and_adopted_real_statuses',ns['raw_status']==0 and ns['adopted'].total==1 and waited==[(1000,0),(2000,0)])
ck('overflow_cannot_become_success_even_after_all_actual_waits',tr.complete is False and ns['identity_absence']is False and ns['group_scan']is None)
d.OWNED_LIMITS_V8.clear();d.OWNED_LIMITS_V8.update(limits)
ck('H058_external_RSS_and_pidfd_paths_use_correct_strict_cleanup_APIs','rss_owned=types.SimpleNamespace(_stat=tree.read_stat_v9,_same=owned._same)'in op._DRIVER_SOURCE and 'sum(rss_identity_v1(rss_owned,r)'in op._DRIVER_SOURCE and 'read_stat=tree.cleanup_read_stat_v9 if tree is not None else owned._stat'in op._DRIVER_SOURCE)
contract=json.loads((ROOT/D/'LIFECYCLE_REUSE_CONTRACT.v9.json').read_bytes());tree=ast.parse(op._DRIVER_SOURCE)
ck('shared_C3_helper_component_exact_source_slices_unchanged',all(hashlib.sha256(ast.get_source_segment(op._DRIVER_SOURCE,next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)).encode()).hexdigest()==digest for name,digest in contract['source_slice_sha256'].items()))
ck('reviewed_Census_bridge_source_is_embedded_exactly',op._CENSUS_MEMO_BRIDGE_SOURCE==(ROOT/'research/reviews/FLARE_original_VAL002004_conversion_execution_source_20261017_v1/root_memo_bridge_v1.py').read_text())
ck('reviewed_math_helper_bytes_and_all_original_science_remain_untouched',hashlib.sha256(op._LEAN_MATH_SOURCE.encode()).hexdigest()=='fa5bddb450d2e531859f8d0c62bc6a1e49f60e82359baf97adede3e5cdb86966')
ck('original_capacity_attachment_and_2s_sampling_remain_exact',all(s in op._DRIVER_SOURCE for s in ('direct.capacity_v1(q,direct.base_v1(),worker_pid=proc.pid)','direct.check_attachment_v1(q,ticket,physical_fd,logical_fd)','capacity_samples+=1;next_capacity=time.monotonic()+2')))
for name in ('context_factory_v1.py','direct_gpu0_v1.py','observations_gpu_v1.py','producer_role.source.txt'):
 current=(ROOT/D/name).read_text().replace(D,D[:-1]+'8').replace(D.replace('/','.'),(D[:-1]+'8').replace('/','.')).replace('_H058_v9_','_H058_v8_')
 ck('entire_'+name+'_is_v8_namespace_only',ast.dump(ast.parse(current),include_attributes=False)==ast.dump(ast.parse((ROOT/(D[:-1]+'8')/name).read_text()),include_attributes=False))
ck('no_Torch_PyG_Numpy_imports',not any(n=='torch'or n.startswith(('torch.','torch_geometric','numpy'))for n in sys.modules))
result=dict(schema='H058_v9_material_group_cleanup_source_checks_v1',source_only=True,independent=False,status='PASS',assertions=len(checks),negative_fixtures=negative,checks=checks,source_ref=rf(c.DRIVER),source_manifest_ref=rf(c.MANIFEST),no_actual_process_pidfd_group_RSS_guard_GPU_payload_or_numerical_calls=True)
if '--emit-result'in sys.argv:
 with(ROOT/D/'targeted_checks.result.v9.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k!='checks'},sort_keys=True))
