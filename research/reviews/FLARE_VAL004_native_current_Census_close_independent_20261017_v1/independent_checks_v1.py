"""Focused VAL004 source/immutable-metadata review. No runtime imports/probes.
Only exact metadata_join/request_value ASTs execute with preauthenticated RAM
bytes. Complete ordinary close/proof behavior reuses the hash-bound VAL003 pass.
"""
from pathlib import Path
from datetime import datetime
import ast,copy,hashlib,json,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/FLARE_VAL004_native_current_Census_close_independent_20261017_v1'
D='research/reviews/FLARE_VAL004_native_current_Census_close_source_20261017_v1/'
H={'path':D+'source_handoff.v1.json','sha256':'27f8db31fed8a5d634dae7b4f36af37eba7fd41dac2bf39961990821da9aa034'}
M={'path':D+'source_manifest.v1.json','sha256':'19f9a57b0c247e7ccbaa7a676a9d20a3410f448a1061303271c53f2d5860a418'}
verified={};checks=[];negative=[]
def need(ok,message):
 if not ok:raise ValueError(message)
def check(ok,message):
 if not ok:raise AssertionError(message)
 checks.append(message)
def authentic(record):
 path=record['path'];need(type(record)is dict and set(record)=={'path','sha256'}and not Path(path).is_absolute()and'..'not in Path(path).parts,'Exact source/metadata ref')
 data=(ROOT/path).read_bytes();need(len(data)<=2**20 and hashlib.sha256(data).hexdigest()==record['sha256'],'Authentic bounded metadata: '+path)
 verified[path]=record['sha256'];return data
h=json.loads(authentic(H));manifest=json.loads(authentic(M));p=json.loads(authentic(h['pins_ref']))
for record in manifest['source_refs']:authentic(record)
check(len(manifest['source_refs'])==31,'All31 selected source/immutable-metadata manifest refs authenticate')
for field in ('audit_manifest_ref','source_audit_ref','authorization_template_ref','revocation_template_ref','checks_source_ref','checks_result_ref','recipe_ref'):authentic(h[field])
passed=json.loads(authentic(h['predecessor_passed_source_review_ref']))
check(passed['independent']is True and passed['verdict']=='SOURCE_PASS'and passed['blockers']==[]and passed['whole_selected_close_source_review_complete']is True and passed['original_ordinary_proof_and_apply_source_review_complete']is True,'Authentic complete VAL003 ordinary-close/proof predecessor pass')
old=ast.parse(authentic(passed['source_ref']));tree=ast.parse(authentic(h['source_ref']))
for field in ('checks_source_ref','checks_result_ref','ROOT_remaining_requirements_ref'):authentic(passed[field])
class Namespace(ast.NodeTransformer):
 def visit_Constant(self,node):
  if type(node.value)is str:return ast.copy_location(ast.Constant(value=node.value.replace('VAL004','VAL003').replace('val_004','val_003')),node)
  return node
normal=Namespace().visit(copy.deepcopy(tree))
exclude=lambda module:ast.Module(body=[n for n in module.body if not(isinstance(n,ast.FunctionDef)and n.name=='metadata_join')],type_ignores=[])
check(ast.dump(exclude(old),include_attributes=False)==ast.dump(exclude(normal),include_attributes=False),'Entire module outside metadata_join unchanged after independently fixed namespace substitutions only')
check(p['guard']['sha256']==json.loads(authentic(passed['checks_result_ref']))['verified_refs'][p['guard']['path']],'Original ordinary proof/apply current-Census source is identical to authenticated passed predecessor')
# All runtime reads here resolve only to preauthenticated source/metadata bytes.
blobs={}
for field,record in p.items():
 if type(record)is dict and set(record)=={'path','sha256'}:blobs[record['path']]=authentic(record)
def ram_raw(record):
 need(verified.get(record['path'])==record['sha256'],'Only preauthenticated immutable metadata in RAM')
 return blobs[record['path']]
def ram_read(record):return json.loads(ram_raw(record))
def ram_local(value):
 path=Path(value);path=path if path.is_absolute()else ROOT/path
 need(path.is_relative_to(ROOT)and'..'not in path.parts,'Contained source-declared path in RAM')
 return path
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ('metadata_join','request_value')]
check(len(selected)==2,'Only exact source metadata_join and pure request_value ASTs selected')
ns=dict(need=need,read=ram_read,raw=ram_raw,local=ram_local,ROOT=ROOT,datetime=datetime,json=json)
exec(compile(ast.Module(body=selected,type_ignores=[]),h['source_ref']['path'],'exec',dont_inherit=True,optimize=0),ns)
joined=ns['metadata_join'](p)
inp,wait,done,launch,storage,own,pids,groups=joined
check(pids==groups==[175310,176516,178698,181613],'Exact genuine recorded supervisor/worker/child PID and group roster')
check(own['reserved_bytes']==2003235369 and len(own['consumer_bindings'])==63 and hashlib.sha256(json.dumps(own,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==p['reservation_sha256']=='dd677f0fbad49e800cef8c86c9759f576c6a847bc4d6f1a5fe3ed9477ad67e33','Whole original reservation,63bindings and original budget-encoding SHA preserved')
spawn=ram_read(p['direct_spawn']);direct=ram_read(p['direct_supervisor_wait']);outer=ram_read(p['outer'])
post=spawn['actual_post_Popen_stat_observation']
check(spawn['actual_Popen']is True and 'proc_stat'not in spawn and post['observed_after_Popen']is True and post['pid']==175310 and post['start_jiffies']==492310847,'Genuine Popen PID/start observed after Popen, with no invented proc_stat schema')
check(post['pgrp']==post['session']==175309 and wait['supervisor_process_group']==175310 and 175309 not in pids and 175309 not in groups,'Pre-setsid175309 pgrp/session remain separate from final175310 group and are excluded from all close targets')
check(wait['pid']==176516 and launch['ownership']['owned_identities']==[dict(pid=176516,start=492327635)]and wait['returncode']==wait['actual_OS_wait_status']==0 and outer['actual_tool_session_id']==40429 and outer['actual_exit_code']==0,'Actual worker identity/raw OSwait0/drain and consumed outer40429exit0 retained')
admission_outer=ram_read(p['admission_outer']);issued=ram_read(p['issued_once'])
check(admission_outer['actual_tool_session_id']==14772 and admission_outer['actual_exit_code']==0 and admission_outer['actual_terminal_consumed']is True and issued['actual_once_execution_argv']==spawn['argv'] and datetime.fromisoformat(admission_outer['utc'])<=datetime.fromisoformat(issued['utc'])<=datetime.fromisoformat(spawn['utc'])<=datetime.fromisoformat(direct['utc'])<=datetime.fromisoformat(outer['utc']),'Genuine admission outer14772, issued-once argv and actual spawn/directwait/outer chronology retained')
check(p['original_caps']==ram_read(p['original_request'])['caps']and p['original_caps']['reservation_bytes']==2003235369 and p['original_caps']['output_bytes']==1994846761,'Exact original caps retained')
check(ram_read(p['count'])['num_sites']==229643 and ram_read(p['conversion'])['native_expected_markers']==26532 and ram_read(p['conversion'])['reference_haplotypes']==600 and ram_read(p['conversion'])['query_haplotypes']==1000 and ram_read(p['conversion'])['common_focal_records']==10000,'Full input/native-marker/reference/query/focal metadata retained without payload reads')
# New metadata joins are exercised independently; unchanged close controls reuse the predecessor pass.
cases=[
 ('direct_spawn',lambda v:v.update(actual_Popen=False),'nonactual Popen'),
 ('direct_spawn',lambda v:v.update(proc_stat='invented'),'invented proc_stat wrapper'),
 ('direct_spawn',lambda v:v.update(start_new_session=True),'changed Popen session mode'),
 ('direct_spawn',lambda v:v['argv'].__setitem__(0,'not-setsid'),'wrong execution argv'),
 ('direct_spawn',lambda v:v['actual_post_Popen_stat_observation'].update(observed_after_Popen=False),'unconfirmed post-Popen observation'),
 ('direct_spawn',lambda v:v['actual_post_Popen_stat_observation'].update(start_jiffies=492310848),'wrong observed start'),
 ('direct_spawn',lambda v:v['actual_post_Popen_stat_observation'].update(pgrp=175310),'pre-setsid pgrp relabelled as final'),
 ('direct_spawn',lambda v:v['actual_post_Popen_stat_observation'].update(session=175310),'pre-setsid session relabelled as final'),
 ('wait',lambda v:v.update(supervisor_process_group=175309),'pre-setsid group used as final supervisor group'),
 ('wait',lambda v:v['known_owned_pids'].append(175309),'pre-setsid parent added to PID targets'),
 ('wait',lambda v:(v['known_owned_process_groups'].append(175309),v['group_observations'].append(dict(pgid=175309,absent=True))),'pre-setsid parent added to group targets'),
 ('outer',lambda v:v.update(schema='old_VAL003_shape'),'wrong outer schema'),
 ('outer',lambda v:v.update(actual_tool_session_id=40430),'wrong outer session'),
 ('outer',lambda v:v.update(actual_direct_worker_returncode=1),'nonzero recorded direct worker'),
 ('outer',lambda v:v.update(actual_direct_wait_ref={}),'wrong supervisor wait ref'),
 ('outer',lambda v:v.update(actual_worker_wait_ref={}),'wrong worker wait ref'),
 ('admission_invocation',lambda v:v.update(ROOT_all_mutations_serialized=False),'unserialized admission'),
 ('admission_invocation',lambda v:v.update(automatic_retry=True),'retry admitted in invocation'),
 ('admission_direct_wait',lambda v:v.update(actual_returncode=1),'nonzero admission direct wait'),
 ('admission_direct_wait',lambda v:v.update(pid=169567),'wrong admission direct PID'),
 ('admission_outer',lambda v:v.update(actual_tool_session_id=14773),'wrong consumed admission session'),
 ('admission_outer',lambda v:v.update(actual_terminal_consumed=False),'unconsumed admission outer'),
 ('issued_once',lambda v:v.update(actual_once_execution_argv=[]),'wrong issued-once execution argv'),
 ('issued_once',lambda v:v.update(utc='2026-10-17T14:00:00+00:00'),'issued-once before consumed admission'),
 ('native_wait',lambda v:v.update(returncode=1),'nonzero native child wait'),
 ('count',lambda v:v.update(num_sites=229642),'changed input support'),
 ('conversion',lambda v:v.update(common_focal_records=9999),'reduced focals'),
 ('conversion',lambda v:v.update(native_expected_markers=26531),'changed native marker support'),
 ('completed',lambda v:v['native_options'].update(seed=1),'changed original author seed'),
 ('storage',lambda v:v['reservation'].update(reserved_bytes=2003235368),'changed original charge'),
 ('wait',lambda v:v.update(all_owned_processes_and_groups_drained=False),'incomplete original drain'),
]
for field,mutate,label in cases:
 changed=ram_read(p[field]);mutate(changed)
 ns['read']=lambda record,field=field,changed=changed:copy.deepcopy(changed)if record==p[field]else ram_read(record)
 try:ns['metadata_join'](p)
 except (ValueError,KeyError,TypeError):negative.append(label)
 else:raise AssertionError('Accepted new-join corruption: '+label)
ns['read']=ram_read
request=ns['request_value'](p,own,launch,{'path':'RAM_ONLY_observation.json','sha256':'0'*64})
check(request['launch_pid_locator']==request['wait_pid_locator']==['pid']and request['launch_identity']==dict(pid=176516,input_ref=p['input'])and request['wait_identity']['actual_OS_wait_status']==0 and request['wait_identity']['all_owned_processes_and_groups_drained']is True,'Unchanged original ordinary-proof request locators/identities join exact native metadata in RAM')
check(request['reservation_sha256']==p['reservation_sha256']and request['completed_identity']==dict(status='completed_C3_native_FLARE',parent_id='american_c3_val_004'),'Original ordinary-close request remains exact VAL004 native')
auth=json.loads(authentic(h['authorization_template_ref']));rev=json.loads(authentic(h['revocation_template_ref']))
check(auth['active']is False and auth['once_authorized']is False and auth['ROOT_has_serialized_all_mutations_and_resolved_prior_inflight_actions']is False and auth['created_at_utc']is auth['expires_at_utc']is None,'Inactive authority supplies neither clocks, serialization proof nor actual execution authority')
check(rev['active']is False and rev['revoked_at_utc']is None,'Actual nonfuture permanent revocation remains ROOT-owned')
report=dict(schema='FLARE_VAL004_native_focused_independent_source_RAM_checks_v1',passed=True,
 positive_count=len(checks),negative_count=len(negative),checks=checks,negative_cases=negative,
 source_handoff_ref=H,source_manifest_ref=M,source_ref=h['source_ref'],predecessor_passed_review_ref=h['predecessor_passed_source_review_ref'],
 exact_selected_metadata_join_executed_in_RAM=True,whole_non_metadata_module_AST_equal=True,original_proof_apply_behavior_reused_from_authenticated_predecessor=True,
 guard_modules_imported=0,actual_state_process_resource_lock_measurement_accounting_GPU_or_Git_operations=0,actual_clock_reads=0,scientific_payload_reads=0,
 verified_refs=verified)
(HERE/'checks.result.v1.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ('verified_refs','checks','negative_cases')},indent=2,sort_keys=True))
