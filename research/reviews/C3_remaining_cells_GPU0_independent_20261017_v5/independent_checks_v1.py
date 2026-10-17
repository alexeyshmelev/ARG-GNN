"""Narrow independent genuine H058v9 metadata + pinned producer/schema review.
No live process, resource, clock, guard/accounting, launch or scientific calls.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from unittest.mock import patch
import ast,collections,contextlib,copy,hashlib,importlib,io,json,os,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D=ROOT/'research/reviews/C3_remaining_cells_GPU0_source_20261017_v5';OLD=D.parent/'C3_remaining_cells_GPU0_source_20261017_v4';R=ROOT/'research/reviews/C3_remaining_cells_GPU0_independent_20261017_v5'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
checks=[];negatives=[];refs={}
def ck(name,value=True):
 if not value:raise AssertionError(name)
 checks.append(name)
def bounded_bytes(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 assert p.suffix in('.py','.json','.md','.txt','.patch')and p.is_file()and p.stat().st_size<=2**20,p
 raw=p.read_bytes();refs[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest();return raw
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(bounded_bytes(p)).hexdigest())
def bounded(record):
 raw=bounded_bytes(record['path']);assert hashlib.sha256(raw).hexdigest()==record['sha256'];return json.loads(raw)
def reject(name,call,kind=(ValueError,KeyError,TypeError,AssertionError)):
 try:call()
 except kind as error:negatives.append(dict(name=name,type=type(error).__name__));return error
 raise AssertionError('Accepted '+name)
ck('exact_requested_v5_manifest',ref(D/'source_manifest.v1.json')['sha256']=='24fc06c2165003792f24b2e9393840bd0badc68bb2e1af8594b15145cc447a2b')
ck('exact_requested_v5_handoff',ref(D/'source_handoff.v5.json')['sha256']=='ab0b969679f967c737ac78376dc0d25e1c933e44704e962108838c4c0927e3eb')
manifest=json.loads(bounded_bytes(D/'source_manifest.v1.json'));oldmanifest=json.loads(bounded_bytes(OLD/'source_manifest.v1.json'));pins=json.loads(bounded_bytes(D/'pins.v1.json'));p=pins['pins']
oldset={json.dumps(r,sort_keys=True)for r in oldmanifest['source_authentication_refs']};newset={json.dumps(r,sort_keys=True)for r in manifest['source_authentication_refs']}
ck('all459_inherited_authentication_pins_preserved_without_broad_rerun',oldset<=newset and len(oldset)==459 and len(newset)==464)
newrefs=[r for r in manifest['source_authentication_refs']if json.dumps(r,sort_keys=True)not in oldset]
for record in manifest['executed_source_refs']+newrefs:ck('exact_current_or_new_source_'+record['path'],ref(record['path'])==record)
audit=json.loads(bounded_bytes(D/'audit_manifest.v5.json'))
for record in audit['refs']:assert ref(record['path'])==dict(path=record['path'],sha256=record['sha256'])
ck('v5_final_audit_exact')
launch=bounded(p['H058_LAUNCH']);inp=bounded(p['H058_INPUT']);request=bounded(p['H058_REQUEST']);attachment=bounded(p['H058_ATTACHMENT'])
peerref=ref('research/reviews/C3_remaining_GPU0_v4_ROOT_preparation_root01_oct17/H058_v9_peer.root_active_v1.json');ck('genuine_ROOT_historical_peer_exact',peerref['sha256']=='b995787bc183f318bdd6dbbb3e25d5872078227fa33e784efe8d906da24ab7b7')
peer=bounded(peerref);qualification=bounded(peer['peer_source_qualification_ref'])
ck('genuine_input_request_launch_attachment_chain',launch['input_ref']==p['H058_INPUT']and inp['root_stage_request_ref']==p['H058_REQUEST']and launch['attachment_ref']==p['H058_ATTACHMENT']and attachment['job_id']==inp['job_id']==request['job_id']and attachment['direct_attachment_authority_ref']==request['direct_attachment_authority_ref'])
ck('current_summary_is_real_recorded_schema_no_legacy_key','current_owned_identities'in launch['ownership']and'owned_identities'not in launch['ownership'])
identity=next(row for row in launch['ownership']['current_owned_identities']if row['pid']==launch['pid'])
ck('full_recorded_identity_not_invented',identity==dict(pid=46648,start=491287607,group=46648,state='R')and peer['worker_identity']==qualification['worker_identity']==identity and launch['parent_pid']==39747)
# Freeze only the observation time at a point inside the genuine immutable peer's
# interval. This does not create/renew clocks or assert current execution authority.
FIXED=max(datetime.fromisoformat(peer['created_at_utc']),datetime.fromisoformat(qualification['created_at_utc']))+timedelta(microseconds=1)
assert FIXED<datetime.fromisoformat(peer['expires_at_utc'])and FIXED<datetime.fromisoformat(qualification['expires_at_utc'])
class FixedDateTime(datetime):
 @classmethod
 def now(cls,tz=None):return FIXED if tz is not None else FIXED.replace(tzinfo=None)
ns=str(D.relative_to(ROOT)).replace('/','.');oldns=str(OLD.relative_to(ROOT)).replace('/','.')
c=importlib.import_module(ns+'.contract_v1');oldc=importlib.import_module(oldns+'.contract_v1');e=importlib.import_module(ns+'.execution_v1')
with patch.object(c,'datetime',FixedDateTime),patch.object(oldc,'datetime',FixedDateTime):
 error=reject('v4_real_missing_owned_identities_failure_reproduced',lambda:oldc.peer_values_v1(peerref),kind=(KeyError,));ck('v4_exact_KeyError_matches_ROOT_failure',error.args==('owned_identities',))
 actual=c.peer_values_v1(peerref);ck('actual_v5_peer_join_accepts_genuine_unchanged_metadata',actual==(peer,request))
 # RAM negatives override only a reader result after separate genuine SHA checks.
 # Positive fixtures above are the actual documents, never handwritten ownership.
 realread=c.read
 def negative_doc(name,record,mutate):
  bad=copy.deepcopy(bounded(record));mutate(bad)
  with patch.object(c,'read',lambda r:bad if r==record else realread(r)):reject(name,lambda:c.peer_values_v1(peerref))
 negative_doc('missing_current_roster_no_legacy_fallback',p['H058_LAUNCH'],lambda v:v['ownership'].pop('current_owned_identities'))
 negative_doc('legacy_key_injected_rejected',p['H058_LAUNCH'],lambda v:v['ownership'].update(owned_identities=[]))
 negative_doc('incomplete_ownership_rejected',p['H058_LAUNCH'],lambda v:v['ownership'].update(complete_owned_roster_known=False))
 negative_doc('truncated_ownership_rejected',p['H058_LAUNCH'],lambda v:v['ownership'].update(owned_roster_truncated=True))
 negative_doc('non_summary_schema_rejected',p['H058_LAUNCH'],lambda v:v['ownership'].update(historical_roster_summarized=False))
 negative_doc('wrong_launch_attachment_join_rejected',p['H058_LAUNCH'],lambda v:v.update(attachment_ref=dict(path='RAM/wrong',sha256='0'*64)))
 negative_doc('duplicate_recorded_worker_row_rejected',p['H058_LAUNCH'],lambda v:v['ownership']['current_owned_identities'].append(copy.deepcopy(identity)))
 for name,key,value in [('wrong_attachment_schema','schema','legacy'),('physical_borrow_denied','borrowed_physical_inode',False),('missing_owned_logical_lock','H058_logical_lock_owned_by_supervisor',False),('false_physical_owner','physical_exclusive_lock_owner','H058'),('wrong_attachment_job','job_id','old'),('wrong_GPU','physical_gpu',1),('wrong_GPU_UUID','gpu_uuid','GPU-other'),('wrong_authority','direct_attachment_authority_ref',dict(path='RAM/wrong',sha256='0'*64))]:
  negative_doc(name,p['H058_ATTACHMENT'],lambda v,key=key,value=value:v.update({key:value}))
 negative_doc('future_invented_launch_digest_rejected',peerref,lambda v:v.update(H058_launch_ref=dict(p['H058_LAUNCH'],sha256='0'*64)))
 negative_doc('partial_worker_identity_rejected',peerref,lambda v:v.update(worker_identity={k:x for k,x in identity.items()if k!='state'}))
# Independently extract exact current summary producer from actual pinned module.
# Its constructor/refresh/proc IO are never called; evidence/history are pure RAM.
ck('exact_actual_H058_operational_source',ref(p['H058_DRIVER']['path'])==p['H058_DRIVER'])
outer=ast.parse(bounded_bytes(p['H058_DRIVER']['path']));driver_source=ast.literal_eval(next(n.value for n in outer.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_DRIVER_SOURCE'for t in n.targets)))
ck('embedded_driver_source_matches_human_mirror',driver_source==bounded_bytes('research/reviews/H058_GPU0_full_training_successor_source_20261017_v9/driver_role.source.txt').decode())
tree=ast.parse(driver_source);history=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='_HistoryV8');owned=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='OwnedLifecycleV8');evidence=next(n for n in owned.body if isinstance(n,ast.FunctionDef)and n.name=='evidence')
limitnode=next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='OWNED_LIMITS_V8'for t in n.targets))
limits=eval(compile(ast.Expression(body=limitnode),'<pinned_literal_ownership_limits>','eval'),{'__builtins__':{},'dict':dict})
ck('actual_limits_bind_pinned_producer_policy',limits==launch['ownership']['limits'])
env=dict(collections=collections,hashlib=hashlib,json=json,OWNED_LIMITS_V8=limits);exec(compile(ast.Module(body=[history,evidence],type_ignores=[]),'<exact_pinned_current_v9_evidence_RAM>','exec',dont_inherit=True,optimize=0),env)
discovered=env['_HistoryV8']();discovered.add({key:identity[key]for key in('pid','start','group')})
ramtree=SimpleNamespace(groups={identity['group']},complete=True,known={identity['pid']:copy.deepcopy(identity)},discovered=discovered,retired=env['_HistoryV8'](),reused=env['_HistoryV8'](),retirement_reasons=dict(absent=0,replaced=0))
ck('entire_genuine_ownership_reproduced_by_actual_pinned_summary_and_history',env['evidence'](ramtree)==launch['ownership'])
# Minimal delta: current peer reader only; original caller/science/lifecycle and
# direct role function remain exact after explicit namespace normalization.
def normalized(text):
 for a,b in [('C3_remaining_cells_GPU0_source_20261017_v5','C3_remaining_cells_GPU0_source_20261017_v4'),('_v5','_v4'),('v5_','v4_'),('-v5-','-v4-'),('C3_v5_','C3_v4_'),('.v5.json','.v4.json'),('fresh v5','fresh v4'),('Fresh v5','Fresh v4')]:text=text.replace(a,b)
 return text
for name in('execution_v1.py','direct_gpu0_v1.py','worker_v1.py','__init__.py'):
 ck('inherited_AST_unchanged_'+name,ast.dump(ast.parse(normalized(bounded_bytes(D/name).decode())),include_attributes=False)==ast.dump(ast.parse(bounded_bytes(OLD/name)),include_attributes=False))
oldmap={n.name:n for n in ast.parse(bounded_bytes(OLD/'contract_v1.py')).body if isinstance(n,ast.FunctionDef)};newmap={n.name:n for n in ast.parse(normalized(bounded_bytes(D/'contract_v1.py').decode())).body if isinstance(n,ast.FunctionDef)}
ck('contract_only_peer_join_changes_beyond_fresh_namespaces',set(oldmap)==set(newmap)and all(ast.dump(oldmap[name],include_attributes=False)==ast.dump(newmap[name],include_attributes=False)for name in oldmap if name!='peer_values_v1'))
ck('exact_original_direct_role_function',ast.dump(oldmap['direct_refs_v1'],include_attributes=False)==ast.dump(newmap['direct_refs_v1'],include_attributes=False))
# Reproduce author's narrow34/5 only, with its result writer suppressed and its
# one help subprocess replaced by actual in-process argparse; no broad209 rerun.
smokes=[]
def inprocess_help(command,**kwargs):
 out=io.StringIO();err=io.StringIO();code=0
 with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
  try:code=e.main(command[5:])
  except SystemExit as ex:code=ex.code
 smokes.append(command[5:]);return SimpleNamespace(returncode=code,stdout=out.getvalue(),stderr=err.getvalue())
author_tree=ast.parse(bounded_bytes(D/'narrow_checks_v5.py'));author_tree.body=[n for n in author_tree.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and((isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='write_text')or(isinstance(n.value.func,ast.Name)and n.value.func.id=='print')))]
author_scope=dict(__name__='_narrow_author_reproduction',__file__=str(D/'narrow_checks_v5.py'))
with patch.object(c,'datetime',FixedDateTime),patch.object(oldc,'datetime',FixedDateTime),patch.object(e,'datetime',FixedDateTime),patch('subprocess.run',inprocess_help):exec(compile(author_tree,str(D/'narrow_checks_v5.py')+'#writer-suppressed-help-inprocess','exec',dont_inherit=True,optimize=0),author_scope)
author=author_scope['result'];ck('separate_author34positive5negative_reproduced',author['positive_count']==34 and author['negative_count']==5 and smokes==[['--help']])
context=json.loads(bounded_bytes(D/'ROOT_context.v5.json'));ck('historical_v3_wait_and_charge_preserved',context['old_v3_actual_Python_supervisor_wait']=='UNKNOWN'and context['old_v3_retained_charge_bytes']==32*2**20)
ck('retained27_pending3_original_cells_preserved',sum(context['retained_count_by_parent'].values())==27 and context['pending_epochs_by_parent']=={parent:([30]if parent in c.VAL[:3]else[])for parent in c.VAL})
ck('no_scientific_imports',not any(name in sys.modules for name in('torch','numpy','torch_geometric','tskit','psutil')))
result=dict(schema='C3_v5_independent_genuine_peer_metadata_schema_checks_v1',verdict='PASS_SOURCE_ONLY',independent=True,independent_assertions=len(checks),checks=checks,independent_negative_count=len(negatives),negative_cases=negatives,
 author_narrow_checks_separately_reproduced=dict(positive_count=34,negative_count=5,help_inprocess_only=True,result_writer_suppressed=True,broad209campaign_not_repeated=True),
 genuine_metadata_refs={key:p[key]for key in('H058_INPUT','H058_REQUEST','H058_LAUNCH','H058_ATTACHMENT','H058_DRIVER')},genuine_historical_ROOT_peer_ref=peerref,
 entire_actual_launch_ownership_reproduced_from_pinned_producer=True,v4_actual_KeyError_reproduced=True,positive_legacy_fixture_or_fallback=False,
 inherited_v4_full_source_review_ref=ref('research/reviews/C3_remaining_cells_GPU0_independent_20261017_v4/review.v1.json'),
 exact_bounded_source_metadata_refs=[dict(path=key,sha256=value)for key,value in sorted(refs.items())],
 no_actual_clock_process_resource_probes_guard_accounting_admission_authority_Torch_arrays_truth_or_Git=True,
 source_manifest_ref=ref(D/'source_manifest.v1.json'),source_handoff_ref=ref(D/'source_handoff.v5.json'))
if '--emit-result'in sys.argv:
 with(R/'checks.result.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k not in('checks','negative_cases','exact_bounded_source_metadata_refs')},sort_keys=True))
