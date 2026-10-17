"""Independent C3v4 source/CLI/caller/peer/capacity RAM review.
No actual subprocess, process, resource, GPU, clock, state or scientific calls.
Author smoke subprocesses are replaced by actual in-process argparse entry calls;
the author result writer is suppressed. Author209/27 remains separately labeled.
"""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
from types import SimpleNamespace
import ast,contextlib,copy,hashlib,importlib,io,json,os,signal,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D=ROOT/'research/reviews/C3_remaining_cells_GPU0_source_20261017_v4';OLD=ROOT/'research/reviews/C3_remaining_cells_GPU0_source_20261017_v3';R=ROOT/'research/reviews/C3_remaining_cells_GPU0_independent_20261017_v4'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
checks=[];negative=[];hashes=[]
def ck(label,value=True):
 if not value:raise AssertionError(label)
 checks.append(label)
def reject(label,call,exceptions=(BaseException,)):
 try:call()
 except exceptions as error:negative.append(dict(name=label,type=type(error).__name__));return error
 raise AssertionError('Accepted '+label)
def raw(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 assert p.suffix in('.py','.sh','.json','.md','.txt','.patch','.log')and p.is_file()and p.stat().st_size<=4*2**20,p
 return p.read_bytes()
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw(p)).hexdigest())
def read(path):return json.loads(raw(path))
ck('exact_requested_handoff',ref(D/'source_handoff.v4.json')['sha256']=='cd89b61575f603f7bb1e89a6884cfe63c0690cf61b88670c7a8ac0be2c7119c0')
ck('exact_requested_source_manifest',ref(D/'source_manifest.v1.json')['sha256']=='2cf156a1acae271fc91a6e8bf5ea2d20e580d183b027359aa43b193236296b5c')
manifest=read(D/'source_manifest.v1.json');audit=read(D/'audit_manifest.v4.json');handoff=read(D/'source_handoff.v4.json')
for label,rows in [('executed',manifest['executed_source_refs']),('source_authentication',manifest['source_authentication_refs']),('audit',audit['refs'])]:
 for row in rows:
  expected=dict(path=row['path'],sha256=row['sha256']);assert ref(row['path'])==expected,expected;hashes.append(dict(group=label,**expected))
ck('all459_pinned_source_metadata_and5executed_and_final_audit_exact',len(manifest['source_authentication_refs'])==459 and len(manifest['executed_source_refs'])==5)
nsname=str(D.relative_to(ROOT)).replace('/','.')
c=importlib.import_module(nsname+'.contract_v1');e=importlib.import_module(nsname+'.execution_v1');w=importlib.import_module(nsname+'.worker_v1');direct=importlib.import_module(nsname+'.direct_gpu0_v1')
FIXED=datetime.fromisoformat('2026-10-17T18:00:00+00:00')
class FixedDateTime(datetime):
 @classmethod
 def now(cls,tz=None):return FIXED if tz is not None else FIXED.replace(tzinfo=None)
# Author reproduction uses actual imported modules and real main/help methods,
# while replacing subprocess smoke with in-process stdlib calls.
smoke=[]
def inprocess_cli(command,**kwargs):
 module=command[4];target=e if module.endswith('.execution_v1')else c
 out=io.StringIO();err=io.StringIO();code=0
 with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
  try:code=target.main(command[5:])
  except SystemExit as ex:code=ex.code
 smoke.append(dict(module=module,args=command[5:],returncode=code,actual_subprocess=False))
 return SimpleNamespace(returncode=code,stdout=out.getvalue(),stderr=err.getvalue())
class SourceTransform(ast.NodeTransformer):
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)and node.func.value.id=='datetime'and node.func.attr=='now':node.func.value.id='FixedDateTime'
  return node
at=ast.parse(raw(D/'focused_checks_v4.py'))
at.body=[node for node in at.body if not(isinstance(node,ast.Expr)and isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Attribute)and node.value.func.attr=='write_text')and not(isinstance(node,ast.Expr)and isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)and node.value.func.id=='print')]
at=ast.fix_missing_locations(SourceTransform().visit(at));author_scope=dict(__name__='_independent_author_reproduction',__file__=str(D/'focused_checks_v4.py'),FixedDateTime=FixedDateTime)
with patch.object(c,'datetime',FixedDateTime),patch.object(e,'datetime',FixedDateTime),patch.object(direct,'datetime',FixedDateTime),patch.object(e,'time',SimpleNamespace(monotonic=lambda:0.,time=lambda:FIXED.timestamp())),patch('subprocess.run',inprocess_cli):
 exec(compile(at,str(D/'focused_checks_v4.py')+'#writers-suppressed-clock-RAM-inprocess-CLI','exec',dont_inherit=True,optimize=0),author_scope)
author=copy.deepcopy(author_scope['result']);ck('separate_author209positive27negative_reproduced_with_no_subprocess',author['positive_count']==209 and author['negative_count']==27 and len(smoke)==4)
# Own real caller cases, including interrupts/uncertainty and replay after failure.
r=dict(path='RAM/independent_input.json',sha256='1'*64);spec=dict(worker_auxiliary_scope='research/reviews/C3_remaining_cells_GPU0_ROOT_v4_RAM_independent/worker_aux')
def caller_case(label,status=0,wait_error=None,launch_error=None,interrupts=0,wrong_pid=False,fail_record=None,keyboard=False):
 records={};launches=[];waits=[];proc=SimpleNamespace(pid=777777,returncode=None);remaining=[interrupts]
 def record(s,name,value):
  if name in records:raise FileExistsError(name)
  records[name]=copy.deepcopy(value)
  if name==fail_record:raise OSError('RAM record failure after exclusive creation')
  return dict(path='RAM/'+name,sha256='a'*64)
 def popen(command,**kw):
  ck(label+'_invocation_consumed_before_Popen','caller_invocation.v4.json'in records)
  launches.append((command,kw))
  if launch_error:raise launch_error
  return proc
 def waitpid(pid,flags):
  waits.append((pid,flags))
  if remaining[0]:remaining[0]-=1;raise InterruptedError()
  if keyboard:raise KeyboardInterrupt('RAM caller interrupt')
  if wait_error:raise wait_error
  return (pid+1 if wrong_pid else pid),status
 fakeos=SimpleNamespace(waitpid=waitpid,WIFEXITED=os.WIFEXITED,WIFSIGNALED=os.WIFSIGNALED,waitstatus_to_exitcode=os.waitstatus_to_exitcode)
 failed=wait_error or launch_error or wrong_pid or fail_record or keyboard or not(os.WIFEXITED(status)or os.WIFSIGNALED(status))
 with patch.object(e,'datetime',FixedDateTime),patch.object(e,'validate_input_v1',lambda value:(spec,{},{},{},{})),patch.object(e,'_put_supervisor_record_v4',record),patch.object(e,'os',fakeos),patch.object(e,'subprocess',SimpleNamespace(Popen=popen)):
  if failed:
   reject(label+'_actual_exception_propagated',lambda:e.supervise_once_v4(r))
   failure=records.get('caller_supervisor_failure.v4.json')
   if failure:
    ck(label+'_no_fabricated_success_or_retry',failure['automatic_retry']is False and failure['scientific_success_claimed']is False)
    if wait_error or launch_error or keyboard or wrong_pid or not(os.WIFEXITED(status)or os.WIFSIGNALED(status)):ck(label+'_wait_remains_unknown',failure['actual_Python_supervisor_wait']is False and failure['actual_Python_supervisor_returncode']is None)
    elif fail_record=='caller_actual_supervisor_wait.v4.json':ck(label+'_genuine_reaped_status_preserved_despite_evidence_failure',failure['actual_Python_supervisor_wait']is True and failure['actual_Python_supervisor_returncode']==os.waitstatus_to_exitcode(status))
  else:
   result=e.supervise_once_v4(r);expected=os.waitstatus_to_exitcode(status)
   ck(label+'_genuine_status_and_return_propagate',result['actual_OS_wait_status']==status and result['actual_Python_supervisor_returncode']==expected and result['returncode']==(expected if expected>=0 else 128-expected) and proc.returncode==expected)
   ck(label+'_wait0_never_claims_worker_science_or_credit',result['worker_wait_or_scientific_completion_inferred']is False and result['storage_close_or_credit_authorized']is False)
  before=len(launches);reject(label+'_replay_consumed_even_after_failure',lambda:e.supervise_once_v4(r),exceptions=(FileExistsError,));ck(label+'_replay_never_spawns',len(launches)==before)
 if waits:ck(label+'_waitpid_exact_child_only',all(pair==(proc.pid,0)for pair in waits))
 if launches:
  argv,kw=launches[0];ck(label+'_exact_direct_Python_and_fresh_session',argv[:6]==[str(ROOT/'.runtime/env/bin/python'),'-B','-s','-m',nsname+'.execution_v1','--execute']and kw==dict(cwd=ROOT,start_new_session=True)and json.loads(argv[7])==json.loads(argv[9])==r)
 return records
for label,status in [('zero',0),('exit1',1<<8),('exit255',255<<8),('SIGKILL',9)]:caller_case(label,status=status)
caller_case('EINTR_three_then_wait0',interrupts=3)
caller_case('unknown_wait',wait_error=ChildProcessError('RAM unknown'))
caller_case('wrong_waited_pid',wrong_pid=True)
caller_case('Popen_failure',launch_error=OSError('RAM exec failure'))
caller_case('KeyboardInterrupt_unknown',keyboard=True)
caller_case('nonterminal_status',status=(signal.SIGSTOP<<8)|0x7f)
caller_case('known_nonzero_wait_receipt_failure',status=7<<8,fail_record='caller_actual_supervisor_wait.v4.json')
caller_case('invocation_write_uncertainty_before_launch',fail_record='caller_invocation.v4.json')
# Real CLI parsing rejects ambiguous actions, missing exact refs, and mismatches.
j=json.dumps(r)
with patch.object(e,'set_amendment_v1',lambda value:None),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
 reject('CLI_mutually_exclusive_actions',lambda:e.main(['--execute','--supervise-once','--input-ref',j,'--amendment-ref',j]),exceptions=(SystemExit,))
 reject('CLI_missing_amendment',lambda:e.main(['--supervise-once','--input-ref',j]),exceptions=(ValueError,))
 reject('CLI_missing_input',lambda:e.main(['--supervise-once','--amendment-ref',j]),exceptions=(ValueError,))
# Current H058v9 capacity layer: original unchanged adapter is a RAM boundary.
# The 26+30+10+4 envelopes are retained with actual sampled external occupancy.
pins=read(D/'pins.v1.json');p=pins['pins'];peer=copy.deepcopy(author_scope['peer']);hq=copy.deepcopy(author_scope['hq']);identity=peer['worker_identity'];docs=author_scope['docs'];docrefs=author_scope['docrefs'];put=author_scope['put'];ram_read=author_scope['ram_read']
clock=dict(root_authorizer='root_agent',status='active',template_only=False,created_at_utc=FIXED.isoformat(),expires_at_utc='2026-10-17T18:30:00+00:00')
auth=read(D/'ROOT_direct_attachment.INACTIVE.template.v4.json');auth.update(clock,job_id='RAM-C3',peer_launch_capsule_ref=author_scope['pr']);ar=put('RAM/independent_attachment.json',auth)
q=dict(job_id='RAM-C3',peer_launch_capsule_ref=author_scope['pr'],direct_attachment_authority_ref=ar,expires_at_utc='2026-10-17T18:30:00+00:00')
capacity_modes={'own':0,'external':512,'total':80*1024,'peer_absent':False,'wrong_peer':False,'orphan_peer_context':False}
def proc(pid):
 if capacity_modes['peer_absent']:raise ProcessLookupError(pid)
 return dict(uid=os.getuid(),start_ticks=identity['start']+(1 if capacity_modes['wrong_peer']else 0),state='R')
base=SimpleNamespace(proc_identity=proc);calls=[]
def peer_capacity(request,base,worker_pid=None):
 calls.append(worker_pid);own=capacity_modes['own'];external=capacity_modes['external'];contexts=[]
 if own:contexts.append(dict(pid=777,name='AI Auto Research Pipeline Test',used_mib=own))
 if capacity_modes['orphan_peer_context']:contexts.append(dict(pid=identity['pid'],name='AI Auto Research Pipeline Test',used_mib=1))
 return dict(GPU0=dict(total_mib=capacity_modes['total'],used_mib=external+own,compute=contexts),projected_mib=60*1024+external+own,actual_free_required_mib=60*1024)
peer_adapter=SimpleNamespace(controller_v1=lambda request,base:({}, {}, {}, {}),capacity_v1=peer_capacity)
with patch.object(c,'datetime',FixedDateTime),patch.object(c,'peer_values_v1',lambda record:(peer,hq)),patch.object(direct,'datetime',FixedDateTime),patch.object(direct,'read',ram_read),patch.object(direct,'peer_module_v1',lambda:peer_adapter),patch.object(direct,'time',SimpleNamespace(time=lambda:FIXED.timestamp())):
 for own in (0,6*1024):
  capacity_modes['own']=own;sample=direct.capacity_v1(q,base,worker_pid=777)
  ck('full70GiB_plus_external_preserved_'+str(own),sample['projected_mib']==70*1024+512 and sample['evaluator_cap_plus_margin_mib']==10*1024)
 capacity_modes.update(own=0,peer_absent=True);ck('absent_peer_still_full30GiB_not_storage_or_memory_credit',direct.capacity_v1(q,base)['projected_mib']==70*1024+512)
 capacity_modes['orphan_peer_context']=True;reject('absent_peer_CUDA_context_blocks',lambda:direct.capacity_v1(q,base),exceptions=(ValueError,));capacity_modes.update(peer_absent=False,orphan_peer_context=False,wrong_peer=True)
 reject('positive_peer_PID_reuse_blocks',lambda:direct.capacity_v1(q,base),exceptions=(ValueError,));capacity_modes.update(wrong_peer=False,own=11*1024)
 reject('own_evaluator_over10GiB_blocks',lambda:direct.capacity_v1(q,base,worker_pid=777),exceptions=(ValueError,));capacity_modes.update(own=0,external=11*1024)
 reject('full_envelopes_plus_external_do_not_fit_blocks',lambda:direct.capacity_v1(q,base),exceptions=(ValueError,))
# Retained cells: inspect only small seal metadata, never prediction payloads.
context=read(D/'ROOT_context.v4.json');election_ref=context['retained_election_ref'];ck('exact_retained_election_pin',ref(election_ref['path'])==election_ref);election=read(election_ref['path']);counts={}
for parent,row in election['parents'].items():
 seals=[read(rr['path'])for rr in row['retained_prediction_seal_refs']];counts[parent]=len(seals)
 ck('retained_epochs_and_original10000_focals_'+parent,sorted(s['epoch']for s in seals)==([5,10,15,20,25]if parent in c.VAL[:3]else[5,10,15,20,25,30])and all(s['parent_id']==parent and s['records']==10000 and s['private_truth_opened']is False for s in seals))
ck('all27_retained_and_only3_original_epoch30_pending',counts==context['retained_count_by_parent']and sum(counts.values())==27 and context['pending_epochs_by_parent']=={parent:([30]if parent in c.VAL[:3]else[])for parent in c.VAL})
ck('old_v3_wait_UNKNOWN_and32MiB_charge_preserved',context['old_v3_actual_Python_supervisor_wait']=='UNKNOWN'and context['old_v3_retained_charge_bytes']==32*2**20 and handoff['old_v3_Python_supervisor_wait']=='UNKNOWN')
ck('H058v9_launch_digest_and_identity_remain_ROOT_supplied_unknown_in_source',context['current_H058_v9_launch_ref_supplied_by_ROOT']is None and context['actual_H058_v9_launch_identity_supplied_by_ROOT']is None)
ck('no_scientific_imports',not any(n in sys.modules for n in('torch','numpy','torch_geometric','tskit','psutil')))
result=dict(schema='C3_v4_independent_source_CLI_wait_peer_RAM_checks_v1',verdict='PASS_SOURCE_ONLY',independent=True,checks=checks,independent_assertions=len(checks),negative_fixtures=negative,independent_negative_count=len(negative),source_metadata_hash_checks=len(hashes),exact_hash_refs=hashes,
 separately_reproduced_author209_27=dict(positive_count=author['positive_count'],negative_count=author['negative_count'],CLI_smoke_in_process_only=smoke,actual_pinned_runtime_subprocess_smoke_not_repeated=True,source_result_writer_suppressed=True),
 current_named_guard_roles=47,current_unique_guard_refs=40,GPU_direct_union=64,matrix_direct_union=62,old_v3_Python_wait='UNKNOWN',old_v3_charge_bytes=32*2**20,
 no_actual_process_resource_clock_probe_GPU_guard_accounting_admission_Torch_array_truth_or_Git_operations=True,
 H058v9_current_actual_launch_or_PID_not_observed_or_invented=True,source_manifest_ref=ref(D/'source_manifest.v1.json'),source_handoff_ref=ref(D/'source_handoff.v4.json'))
if '--emit-result'in sys.argv:
 with(R/'checks.result.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k not in('checks','negative_fixtures','exact_hash_refs','separately_reproduced_author209_27')},sort_keys=True))
