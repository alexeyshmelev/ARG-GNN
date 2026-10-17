"""Focused VAL003 source/immutable-metadata review. No runtime imports/probes.
Only exact metadata_join/request_value ASTs execute with preauthenticated RAM
bytes. Complete ordinary close/proof behavior reuses the hash-bound VAL002 pass.
"""
from pathlib import Path
from datetime import datetime
import ast,copy,hashlib,json,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/FLARE_VAL003_native_current_Census_close_independent_20261017_v1'
D='research/reviews/FLARE_VAL003_native_current_Census_close_source_20261017_v1/'
H={'path':D+'source_handoff.v1.json','sha256':'439571c95090a54106c12759e33ee7c890c45cf2fcf1dad39adeff64ab904cc4'}
M={'path':D+'source_manifest.v1.json','sha256':'3925cecacea0285ef0e6578cc5999b6528dc6e5a02f5d6f96c8180e0102a50a2'}
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
check(len(manifest['source_refs'])==29,'All29 selected source/immutable-metadata manifest refs authenticate')
for field in ('audit_manifest_ref','source_audit_ref','authorization_template_ref','revocation_template_ref','checks_source_ref','checks_result_ref','recipe_ref'):authentic(h[field])
passed=json.loads(authentic(h['predecessor_passed_source_review_ref']))
check(passed['independent']is True and passed['verdict']=='SOURCE_PASS'and passed['blockers']==[]and passed['whole_selected_close_source_review_complete']is True and passed['original_ordinary_proof_and_apply_source_review_complete']is True,'Authentic complete VAL002 ordinary-close/proof predecessor pass')
old=ast.parse(authentic(passed['source_ref']));tree=ast.parse(authentic(h['source_ref']))
for field in ('checks_source_ref','checks_result_ref','ROOT_remaining_requirements_ref'):authentic(passed[field])
class Namespace(ast.NodeTransformer):
 def visit_Constant(self,node):
  if type(node.value)is str:return ast.copy_location(ast.Constant(value=node.value.replace('VAL003','VAL002').replace('val_003','val_002')),node)
  return node
normal=Namespace().visit(copy.deepcopy(tree))
exclude=lambda module:ast.Module(body=[n for n in module.body if not(isinstance(n,ast.FunctionDef)and n.name=='metadata_join')],type_ignores=[])
check(ast.dump(exclude(old),include_attributes=False)==ast.dump(exclude(normal),include_attributes=False),'Entire module outside metadata_join unchanged after independently fixed namespace substitutions only')
check(p['guard']['sha256']==passed['integrity']['verified_refs'][p['guard']['path']],'Original ordinary proof/apply current-Census source is identical to authenticated passed predecessor')
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
check(pids==groups==[134837,137046,146904,156629],'Exact genuine recorded supervisor/worker/child PID and group roster')
check(own['reserved_bytes']==2003398262 and len(own['consumer_bindings'])==63 and hashlib.sha256(json.dumps(own,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==p['reservation_sha256']=='79888af356b63a8ebe897154eff1c4c7acfe25b0f9b09884eccc4ac52565038a','Whole original reservation,63bindings and original budget-encoding SHA preserved')
spawn=ram_read(p['direct_spawn']);later=ram_read(p['later_supervisor_observation']);direct=ram_read(p['direct_supervisor_wait']);outer=ram_read(p['outer'])
row=next(r for r in later['rows']if r.get('pid')==134837)
check(spawn['actual_Popen']is True and 'proc_stat'not in spawn and 'issued_once'not in p and 'admission_outer'not in p,'Genuine direct spawn and missing earlier/wrapper fields are consumed without fabrication')
check(row['expected_start']==row['observed_start']==492006344 and datetime.fromisoformat(spawn['utc'])<=datetime.fromisoformat(later['utc'])<=datetime.fromisoformat(direct['utc'])<=datetime.fromisoformat(outer['utc']),'Later supervisor identity remains explicitly later, bounded by genuine spawn/directwait/outer timestamps')
check(wait['pid']==137046 and launch['ownership']['owned_identities']==[dict(pid=137046,start=492019475)]and wait['returncode']==wait['actual_OS_wait_status']==0 and outer['actual_tool_session_id']==68484 and outer['actual_exit_code']==0,'Actual worker identity/raw OSwait0/drain and consumed outer68484exit0 retained')
check(p['original_caps']==ram_read(p['original_request'])['caps']and p['original_caps']['reservation_bytes']==2003398262 and p['original_caps']['output_bytes']==1995009654,'Exact original caps retained')
check(ram_read(p['count'])['num_sites']==227361 and ram_read(p['conversion'])['native_expected_markers']==26537 and ram_read(p['conversion'])['reference_haplotypes']==600 and ram_read(p['conversion'])['query_haplotypes']==1000 and ram_read(p['conversion'])['common_focal_records']==10000,'Full input/native-marker/reference/query/focal metadata retained without payload reads')
# New metadata joins are exercised independently; unchanged close controls reuse the predecessor pass.
cases=[
 ('direct_spawn',lambda v:v.update(actual_Popen=False),'nonactual Popen'),
 ('direct_spawn',lambda v:v.update(proc_stat='invented'),'invented earlier stat'),
 ('direct_spawn',lambda v:v.update(start_new_session=True),'changed direct Popen session mode'),
 ('direct_spawn',lambda v:v['argv'].__setitem__(0,'not-setsid'),'wrong execution argv'),
 ('later_supervisor_observation',lambda v:next(r for r in v['rows']if r.get('pid')==134837).update(observed_start=492006345),'changed later start'),
 ('later_supervisor_observation',lambda v:next(r for r in v['rows']if r.get('pid')==134837).update(same_identity=False),'unconfirmed later identity'),
 ('later_supervisor_observation',lambda v:v.update(utc='2026-10-17T14:00:00+00:00'),'later observation predates spawn'),
 ('later_supervisor_observation',lambda v:v.update(utc='2026-10-17T15:00:00+00:00'),'later observation postdates direct wait'),
 ('outer',lambda v:v.update(schema='old_VAL002_shape'),'wrong genuine outer schema'),
 ('outer',lambda v:v.update(actual_tool_session_id=68485),'wrong outer session'),
 ('outer',lambda v:v.update(actual_direct_wait_ref={}),'wrong supervisor wait ref'),
 ('outer',lambda v:v.update(actual_worker_wait_ref={}),'wrong worker wait ref'),
 ('admission_invocation',lambda v:v.update(all_ROOT_accounting_mutations_serialized=False),'unserialized admission'),
 ('admission_direct_wait',lambda v:v.update(actual_returncode=1),'nonzero admission direct wait'),
 ('admission_receipt',lambda v:v.update(execution_argv=[]),'mismatched admitted execution argv'),
 ('native_wait',lambda v:v.update(returncode=1),'nonzero native child wait'),
 ('count',lambda v:v.update(num_sites=227360),'changed input site support'),
 ('conversion',lambda v:v.update(common_focal_records=9999),'reduced original focals'),
 ('conversion',lambda v:v.update(query_haplotypes=999),'reduced query support'),
 ('conversion',lambda v:v.update(native_expected_markers=26536),'changed native marker support'),
 ('completed',lambda v:v['native_options'].update(seed=1),'changed author native seed'),
 ('storage',lambda v:v['reservation'].update(reserved_bytes=2003398261),'changed original charge'),
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
check(request['launch_pid_locator']==request['wait_pid_locator']==['pid']and request['launch_identity']==dict(pid=137046,input_ref=p['input'])and request['wait_identity']['actual_OS_wait_status']==0 and request['wait_identity']['all_owned_processes_and_groups_drained']is True,'Unchanged original ordinary-proof request locators/identities join exact native metadata in RAM')
check(request['reservation_sha256']==p['reservation_sha256']and request['completed_identity']==dict(status='completed_C3_native_FLARE',parent_id='american_c3_val_003'),'Original ordinary-close request remains exact VAL003 native')
auth=json.loads(authentic(h['authorization_template_ref']));rev=json.loads(authentic(h['revocation_template_ref']))
check(auth['active']is False and auth['once_authorized']is False and auth['ROOT_has_serialized_all_mutations_and_resolved_prior_inflight_actions']is False and auth['created_at_utc']is auth['expires_at_utc']is None,'Inactive authority supplies neither clocks, serialization proof nor actual execution authority')
check(rev['active']is False and rev['revoked_at_utc']is None,'Actual nonfuture permanent revocation remains ROOT-owned')
report=dict(schema='FLARE_VAL003_native_focused_independent_source_RAM_checks_v1',passed=True,
 positive_count=len(checks),negative_count=len(negative),checks=checks,negative_cases=negative,
 source_handoff_ref=H,source_manifest_ref=M,source_ref=h['source_ref'],predecessor_passed_review_ref=h['predecessor_passed_source_review_ref'],
 exact_selected_metadata_join_executed_in_RAM=True,whole_non_metadata_module_AST_equal=True,original_proof_apply_behavior_reused_from_authenticated_predecessor=True,
 guard_modules_imported=0,actual_state_process_resource_lock_measurement_accounting_GPU_or_Git_operations=0,actual_clock_reads=0,scientific_payload_reads=0,
 verified_refs=verified)
(HERE/'checks.result.v1.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ('verified_refs','checks','negative_cases')},indent=2,sort_keys=True))
