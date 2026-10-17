"""Bounded independent timing repair and complete metadata-entry review; source/JSON/RAM only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import ast,copy,contextlib,hashlib,io,json
R=Path('/workspace-SR003.nfs2/ARG-GNN');D=R/'research/reviews/H059_saved_outer000_acceptance_and_continuation_source_20261017_v2';P=R/'research/reviews/H059_saved_outer000_acceptance_and_continuation_source_20261017_v1';O=R/'research/reviews/H059_saved_outer000_acceptance_and_continuation_independent_20261017_v2'
checks=[];negative=[]
def check(ok,label):assert ok,label;checks.append(label)
def reject(fn,label):
 try:fn()
 except(ValueError,TypeError,KeyError,AssertionError,PermissionError,OSError):negative.append(label);return
 raise AssertionError('Invalid accepted: '+label)
def reference(p):return dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
expected={'source_handoff.v1.json':'06d715008a8316d1c73843b4acc5e6635f1fe55950d2f5f9294f2a929fe9ab39','source_manifest.v1.json':'47f5f7985e3c4ac596e4340b40e5cee389c536d268f93f028b0c49f0a7438ae1'}
for n,h in expected.items():check(reference(D/n)['sha256']==h,'Exact candidate '+n)
m=json.loads((D/'source_manifest.v1.json').read_text());oldm=json.loads((P/'source_manifest.v1.json').read_text());handoff=json.loads((D/'source_handoff.v1.json').read_text())
check(all(r in m['refs']for r in oldm['refs']),'All369 original closure refs retained')
check(m['expected_original_metadata_and_assets']==oldm['expected_original_metadata_and_assets'],'Original assets carried unchanged without payload reads')
for r in m['refs']:
 p=R/r['path'];check(p.suffix in('.py','.json','.md','.sh','.patch','.diff')and p.stat().st_size<=32*2**20,'Source/metadata closure bounded')
 check(reference(p)==r,'Exact401 closure ref '+r['path'])
audit=json.loads((D/'audit_manifest.v2.json').read_text())
for row in audit['files']:
 p=D/row['path'];raw=p.read_bytes();check(len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],'Exact current artifact '+row['path'])
# All unchanged science/hashing/absence implementations are compared, not rerun on payloads.
def nodes(text):return {n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(n):return ast.dump(n,include_attributes=False)
normalize=lambda text:text.replace(str(D.relative_to(R)),str(P.relative_to(R)))
for i in range(1,8):
 n='prepare_outer%03d_v1.py'%i;check(normalize((D/n).read_text())==(P/n).read_text(),'Complete unchanged generation source '+str(i))
check(normalize((D/'prediction_stage_v1.py').read_text())==(P/'prediction_stage_v1.py').read_text(),'Entire five-method40plus8 prediction source unchanged')
oldnodes=nodes((P/'saved_outer000_v1.py').read_text());newnodes=nodes((D/'saved_outer000_v1.py').read_text())
for n in('metadata_v1','native_metadata_v1','known_absence_v1','validate_absence_values_v1','stream_observed_v1','validate_saved_v1'):
 check(dump(oldnodes[n])==dump(newnodes[n]),'Exact original proof/science '+n)
oldriver=nodes((P/'validation_driver_v1.py').read_text());newdriver=nodes((D/'validation_driver_v1.py').read_text())
for n in('boundary','sample_scope','rss_identity_v1','verify_generation_roles_v1'):
 check(dump(oldriver[n])==dump(newdriver[n]),'Original cap/identity function '+n)
# Inspected author fixtures and eight complete actual request/admit/input joins only.
# Cut off its result writer so this review never writes to the candidate directory.
author=(D/'source_check_v2.py').read_text();prefix=author[:author.index('summary=dict(')]
env={'__name__':'independent_inspected_H059_v2_RAM_entry_harness'}
with contextlib.redirect_stdout(io.StringIO()):exec(compile(prefix,'<inspected eight-entry RAM harness only>','exec'),env)
check(len(env['reserve_calls'])==8,'Eight complete actual metadata entry joins')
check([x[3]for x in env['answers']]==[62]*7+[58],'All source-derived role unions remain62/58')
saved=env['saved'];read=env['read'];ref=env['ref'];docs=env['docs'];answers=env['answers'];vns,qref,answer,_=answers[-1]
spec,q,cfg=vns['input_spec'](answer['input_ref']);proof=copy.deepcopy(spec['admission_preflight_v2']);qualification=read(q['original_readonly_scope_qualification_ref']);revocation=read(q['old_writer_revocation_ref'])
check(datetime.fromisoformat(proof['checked_at_utc'])<vns['datetime'].now(timezone.utc)-timedelta(seconds=899),'Actual reserve advances900s without renewing preflight')
future=datetime.fromisoformat(proof['checked_at_utc'])+timedelta(hours=2)
class Later:
 fromisoformat=staticmethod(datetime.fromisoformat)
 @staticmethod
 def now(tz):return future
saved['datetime']=Later;vns['datetime']=Later
check(vns['input_spec'](answer['input_ref'])[0]['admission_preflight_v2']==proof,'Complete input entry succeeds two hours later on same immutable preflight')
check(saved['acceptance_values_v1'](env['accept'],read,ref)['original_history']==saved['HISTORY'],'Complete saved acceptance succeeds later without refreshing prior qualification')
before=(len(env['reserve_calls']),len(env['puts']))
reject(lambda:vns['admit'](qref),'Fresh admission with now-stale qualification rejects at entry')
check(before==(len(env['reserve_calls']),len(env['puts'])),'Stale admission rejected before any RAM publication or reserve')
def badproof(change,label):
 p=copy.deepcopy(proof);p.update(change)
 reject(lambda:saved['admission_preflight_values_v2'](p,q,qref,qualification,revocation,ref(vns['SOURCE']),now=future),label)
badproof({'qualification_max_age_seconds':True},'Boolean max age rejects')
for field,value in[
 ('checked_at_utc',(datetime.fromisoformat(qualification['observed_at_utc'])+timedelta(seconds=121)).isoformat()),
 ('checked_at_utc',(datetime.fromisoformat(q['created_at_utc'])-timedelta(seconds=1)).isoformat()),
 ('checked_at_utc',(future+timedelta(seconds=1)).isoformat()),
 ('checked_at_utc',future.replace(tzinfo=None).isoformat()),
 ('request_ref',env['wait']),('driver_source_ref',env['wait']),('qualification_ref',env['wait']),
 ('old_writer_revocation_ref',env['wait']),('qualification_observed_at_utc',future.isoformat()),
 ('revoked_at_utc',future.isoformat()),('initial_admission_preflight_only',False),('status','FAIL')]:
 badproof({field:value},'Typed immutable preflight rejects '+field+' '+str(value)[:30])
p=copy.deepcopy(proof);p['qualification_max_age_seconds']=120.0
reject(lambda:saved['admission_preflight_values_v2'](p,q,qref,qualification,revocation,ref(vns['SOURCE']),now=future),'Float max age rejects')
p=copy.deepcopy(proof);p['unexpected']=None
reject(lambda:saved['admission_preflight_values_v2'](p,q,qref,qualification,revocation,ref(vns['SOURCE']),now=future),'Extra evidence key rejects')
p=copy.deepcopy(proof);p['checked_at_utc']=(datetime.fromisoformat(qualification['observed_at_utc'])+timedelta(seconds=120)).isoformat()
check(saved['admission_preflight_values_v2'](p,q,qref,qualification,revocation,ref(vns['SOURCE']),now=future)==p,'Exact inclusive120s admission relation accepts')
# Independent mutations pass through actual complete input/request/acceptance functions.
def mutate(r,field,value,fn,label):
 old=copy.deepcopy(docs[r['path']]);docs[r['path']][field]=value
 try:reject(fn,label)
 finally:docs[r['path']]=old
mutate(answer['input_ref'],'admission_preflight_v2',dict(proof,driver_source_ref=env['wait']),lambda:vns['input_spec'](answer['input_ref']),'Actual input rejects mismatched preflight driver')
for r,field,value in[(env['result'],'admission_preflight_v2',dict(proof,status='FAIL')),(env['result'],'original_readonly_scope_qualification_ref',env['wait']),(env['wait'],'complete_owned_roster_known',False),(env['terminal'],'actual_tool_terminal_consumed',False)]:
 mutate(r,field,value,lambda:saved['acceptance_values_v1'](env['accept'],read,ref),'Actual saved acceptance rejects '+field)
for index,(ns,r,a,union)in enumerate(answers[:7],1):
 check(read(r)['parent_id']=='ap_expanded_outer_%03d'%index,'Exact original generation parent '+str(index))
 mutate(r,'parent_id','ap_expanded_outer_000',lambda ns=ns,r=r:ns['request'](r),'Generation rejects old outer000 substitution '+str(index))
for field,value in[('original_charge_retained',False),('complete_original_output_and_packet_measured',False),('old_source_guard_caps_and_clock_unchanged',False),('historical_monitoring_gap_recovered',True)]:
 mutate(q['original_readonly_scope_qualification_ref'],field,value,lambda:vns['request'](qref),'Full validator request rejects '+field)
# Actual worker composition with RAM lower-layer APIs only; no saved payload validator, process or package import executes.
exec(compile(ast.Module(body=[copy.deepcopy(newdriver['boundary'])],type_ignores=[]),'<restore original boundary after author negative fixture>','exec'),vns)
clock=[100.];calls=[];g=vns['guard'](q['policy_selector_ref'],True)
g.validate_storage_v9=lambda *a,**k:calls.append('full_initial_storage')
def accounting(*a,**k):calls.append('full_write_accounting');clock[0]+=50.;return {}
g.budget_check_v9=accounting;g.budget.v4.measure_scopes=lambda *a,**k:dict(conservative_scope_bytes=0)
vns['time']=SimpleNamespace(monotonic=lambda:clock[0])
vns['sample_scope']=lambda s:calls.append('cheap_scope')
vns['resource']=SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda who:SimpleNamespace(ru_maxrss=1024))
vns['os']=SimpleNamespace(environ={'H059_VALIDATOR_SETUP_DEADLINE_MONOTONIC':'7200','H059_VALIDATOR_WORK_DEADLINE_MONOTONIC':'10500','CUDA_VISIBLE_DEVICES':''},
 read=lambda fd,n:b'G',close=lambda fd:calls.append('closed_gate'),getpid=lambda:77,getpgrp=lambda:77,getsid=lambda pid:77,
 sched_getaffinity=lambda pid:set(range(8)),sched_setaffinity=lambda pid,cpus:calls.append('RAM_affinity'))
vns['sys']=SimpleNamespace(version_info=(3,11),dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=True))
def validation(readfn,reffn,callback,*,wall_seconds):
 check(wall_seconds==600,'Worker passes full original600s validation budget')
 before=calls.count('full_write_accounting')
 for i in range(12):callback();clock[0]+=10.
 check(calls.count('full_write_accounting')==before,'Read chunks do no recursive full accounting')
 return dict(schema='RAM_VALIDATED_NOT_ACTUAL',original_history=saved['HISTORY'])
saved_mock=SimpleNamespace(**dict(saved,validate_saved_v1=validation))
class RAMImports:
 def import_module(self,name):
  if name==vns['HERE'].replace('/','.')+'.saved_outer000_v1':return saved_mock
  check(name in cfg['runtime'],'Only qualified runtime package name requested')
  return SimpleNamespace(__version__=cfg['runtime'][name])
vns['importlib']=RAMImports()
check(vns['worker'](answer['input_ref'],7)==0,'Complete actual worker metadata-to-validation/write route with RAM APIs')
check(calls.count('full_initial_storage')==1 and calls.count('full_write_accounting')==3 and calls.count('cheap_scope')==12,'Worker preserves full setup and both final writes, cheap scope on every chunk')
# Slow accounting cannot publish after a passed pre-guard timestamp.
vns['boundary']=lambda *a,**kw:clock.__setitem__(0,999.)
clock[0]=100.;reject(lambda:vns['write_boundary_v2'](g,spec,deadline=500.),'Slow full accounting overrun rejects before caller put')
clock[0]=100.;vns['resource']=SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda who:SimpleNamespace(ru_maxrss=4*2**30//1024+1))
reject(lambda:vns['readonly_boundary_v2'](spec,500.),'Read-only RSS cap rejects')
clock[0]=500.;reject(lambda:vns['readonly_boundary_v2'](spec,500.),'Read-only monotonic deadline rejects')
check(saved['HISTORY']['original_outer_exit_code']==1 and saved['HISTORY']['original_worker_OS_wait']==saved['HISTORY']['original_owned_descendant_drain']=='UNKNOWN','Original34504outer1/wait/drain unknowns remain immutable')
for n,h in expected.items():check(reference(D/n)['sha256']==h,'Candidate seal unchanged '+n)
result=dict(schema='H059_v2_independent_bounded_timing_entry_review_v1',independent=True,status='PASS_SOURCE_RAM_ONLY',checks=len(checks),negative_cases=len(negative),cases=checks,negatives=negative,inspected_author_focused_assertions=len(env['checks']),inspected_author_negative_cases=len(env['negative']),complete_actual_RAM_request_admit_input_routes=8,complete_actual_RAM_validator_worker_route=1,source_derived_role_unions=[x[3]for x in answers],
 actual_guard_state_resource_process_payload_truth_numerical_launch_or_source_edits=False,
 source_manifest_ref=reference(D/'source_manifest.v1.json'),source_handoff_ref=reference(D/'source_handoff.v1.json'))
(O/'independent_checks.result.v2.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in('cases','negatives')},sort_keys=True))
