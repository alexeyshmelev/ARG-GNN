"""Independent fake-only v9 boundary and tiny owned filesystem tests."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from unittest.mock import patch
import copy,hashlib,importlib,json,os,sys,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/native_VAL000_full_state_resume_source_20261016_v9'
OUT=ROOT/'research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v9'
PKG=HERE.replace('/','.')
c=importlib.import_module(PKG+'.amendment_contract_v1')
f=importlib.import_module(PKG+'.fresh_entry_v9')
a=importlib.import_module(PKG+'.alias_namespace_v7')
native=importlib.import_module('research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2.native_child_v1')
checks=[]
def rejected(name,fn,exc=(ValueError,RuntimeError,OSError,AssertionError)):
 try:fn()
 except exc as e:checks.append(dict(name=name,status='PASS_REJECTED',exception=type(e).__name__));return
 raise AssertionError('Unexpected acceptance '+name)
now=datetime(2026,10,16,18,tzinfo=timezone.utc)
def gone(value):raise ProcessLookupError()
known=dict(pid=900001,start_ticks=100)
unknown=dict(pid=900002,start_ticks=None)
def probe(row):return f.probe_PID_v9(row,kill0=gone,now=lambda:now)
registry=dict(identities=[known,unknown],process_groups=[900003])
value=f.collect_v9(registry,stage='worker',context=dict(SYNTHETIC_ONLY=True),tick=lambda:None,
 pid_probe=probe,group_probe=lambda pg:f.probe_group_v9(pg,killpg0=gone,now=lambda:now),
 wall_now=lambda:now,monotonic=lambda:0.)
assert f.require_fresh_v9(value,now=now+timedelta(seconds=120)) is True
rejected('120_second_boundary_plus_epsilon',lambda:f.require_fresh_v9(value,now=now+timedelta(seconds=120.000001)))
rejected('future_timestamp',lambda:f.require_fresh_v9(value,now=now-timedelta(microseconds=1)))
omitted=copy.deepcopy(value);omitted['PID_observations'].pop()
rejected('omitted_row',lambda:f.require_fresh_v9(omitted,now=now))
rejected('old_ROOT_snapshot_schema',lambda:f.require_fresh_v9(c.read(c.V8_KNOWN_BUNDLE_REF),now=now))
def denied(value):raise PermissionError('SYNTHETIC_OPAQUE')
rejected('PID_EPERM',lambda:f.probe_PID_v9(known,kill0=denied))
rejected('group_EPERM',lambda:f.probe_group_v9(900003,killpg0=denied))
rejected('unknown_start_numeric_PID_exists',lambda:f.probe_PID_v9(unknown,kill0=lambda p:None,read_stat=lambda p:101))
rejected('same_PID_start_alive',lambda:f.probe_PID_v9(known,kill0=lambda p:None,read_stat=lambda p:100))
rejected('proc_stat_race_ESRCH',lambda:f.probe_PID_v9(known,kill0=lambda p:None,read_stat=gone))
replacement=f.probe_PID_v9(known,kill0=lambda p:None,read_stat=lambda p:101,now=lambda:now)
assert replacement['known_identity_absent'] is True and replacement['numeric_PID_absent'] is False and replacement['observed_different_start_ticks']==101
checks.append(dict(name='known_PID_reuse_honest_old_identity_absence',status='PASS'))
clock=[0.]
def timeout_probe(row):clock[0]=90.001;return probe(row)
try:f.collect_v9(registry,stage='worker',context={},tick=lambda:None,pid_probe=timeout_probe,
 group_probe=lambda g:None,wall_now=lambda:now,monotonic=lambda:clock[0])
except f.EntryProbeFailure as e:
 assert e.value['status']=='ENTRY_PROBES_FAILED' and len(e.value['PID_observations'])==1 and not e.value['all_known_observations_complete']
 checks.append(dict(name='90_second_collection_bound_retains_honest_partial_failure',status='PASS'))
else:raise AssertionError('90-second cap relaxed')
# Actual imported exact native nested code object, with explicit fake closures only.
nested=next(v for v in native.run_v1.__code__.co_consts if isinstance(v,types.CodeType) and v.co_name=='fast_guard')
def cell(v):
 def inner():return v
 return inner.__closure__[0]
def fast(proxy,lease):
 vals=dict(guard=proxy,lease=lease,require=c.require,spec={})
 return types.FunctionType(nested,dict(c=types.SimpleNamespace(active=lambda spec:None)),closure=tuple(cell(vals[n]) for n in nested.co_freevars))
lease=dict(SYNTHETIC=True)
class Guard:
 def __init__(self):self.calls=0
 def apply_and_validate_title(self):pass
 def validate_lease(self):
  self.calls+=1
  return lease if self.calls==1 else dict(SYNTHETIC=False)
events=[]
proxy=f.ExactNativeEntryGuardV9(Guard(),native.run_v1,capture=lambda:events.append('capture') or dict(path='SYNTHETIC',sha256='0'*64),verify=lambda r:events.append('verify'))
rejected('post_receipt_changed_lease',lambda:fast(proxy,lease)(f.WORKER_EVENT))
assert events==['capture','verify']
# File+directory fsync error must block and must never install an accepted entry ref.
class FakePath:
 def __init__(self,s):self.s=s
 def __truediv__(self,s):return FakePath(self.s+'/'+s)
 def is_dir(self):return True
 def exists(self):return False
spec=dict(worker_auxiliary_scope='SYNTHETIC_OWN_AUX')
entryrefs={};saved=[];printed=[]
marker=dict(path='SYNTHETIC',sha256='0'*64)
fc=types.SimpleNamespace(require=c.require,active_amendment_v1=lambda:None,entry_known_registry_v9=lambda:registry,
 local=lambda path:FakePath(path),charged_census_v1=lambda *a,**kw:None,current_amendment_ref_v1=lambda:marker,
 MANIFEST='SYNTHETIC',ref=lambda path:marker,V8_KNOWN_BUNDLE_REF=marker,V8_TERMINAL_REF=marker,
 save=lambda *args:saved.append(args) or dict(path='SYNTHETIC_OWN_AUX/'+f.ENTRY_NAMES['worker'],sha256='1'*64))
def failfsync(fd):raise OSError('SYNTHETIC_DIR_FSYNC_FAILURE')
fakeos=types.SimpleNamespace(getpid=lambda:900004,open=lambda *a:88,fsync=failfsync,close=lambda fd:None,
 O_RDONLY=0,O_DIRECTORY=0,O_NOFOLLOW=0)
ns=dict(f.capture_entry_v9.__globals__,c=fc,os=fakeos,ENTRY_REFS=entryrefs,
 collect_v9=lambda *args,**kw:copy.deepcopy(value),print=lambda *args,**kw:printed.append(args))
capture=types.FunctionType(f.capture_entry_v9.__code__,ns)
g=types.SimpleNamespace(budget_check_v9=lambda *a,**kw:None)
rejected('directory_fsync_failure_no_entry_authority',lambda:capture(spec,{},g,{},stage='worker',tick=lambda:None))
assert len(saved)==1 and not entryrefs and printed and 'actual_entry_receipt_ref' in printed[0][0]
# Real tiny owned checkpoint bytes, synthesized exact immutable historical receipts,
# actual new consume function and actual alias receipt verifier. No original files.
work=OUT/'tiny_alias_independent';work.mkdir()
check=work/'checkpoints';check.mkdir();aux=work/'aux';aux.mkdir()
target=check/'ep0-ba6500-rank0.pt';target.write_bytes(b'SYNTHETIC_INDEPENDENT_6500')
original=target.read_bytes();identity=a.identity(target.lstat())
def ref(path):
 p=Path(path);return dict(path=str(p.relative_to(work)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(name,v):
 p=aux/name;p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');return ref(p)
def read(r):
 p=work/r['path'];data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==r['sha256'];return json.loads(data)
targetref=ref(target)
v8=dict(path='SYNTHETIC_V8_AMENDMENT',sha256='1'*64);manifest=dict(path='SYNTHETIC_V8_MANIFEST',sha256='2'*64)
before=dict(schema='native_VAL000_durable_ROOT_Composer_alias_before_unlink_v7',
 context=dict(actual_execution_amendment_ref=v8,adapter_manifest_ref=manifest,
 historical_v3_descendant_coverage='UNKNOWN_TRUNCATED64',historical_v3_unknown_descendants_absent=None),
 target_ref=targetref,target_identity=identity,alias='checkpoints/latest-rank0.pt',target_text=target.name,
 actual_complete_owned_writer_drain_verified=True,regular_checkpoint_payload_read_or_copied=False,
 regular_checkpoint_metadata_before=a.regular_snapshot(check))
beforeref=save('before.json',before)
after=dict(schema='native_VAL000_exact_Composer_alias_removed_v7',before_unlink_metadata_ref=beforeref,
 alias=before['alias'],alias_absent=True,all_regular_checkpoint_metadata_unchanged=True,
 regular_checkpoint_payload_read_copied_or_deleted=False,strict_original_boundary_after_unlink=dict(
 original_unchanged_budget_check_boundary_True_passed=True,original_whole_scope_no_follow_charge_caps_passed=True))
afterref=save('after.json',after);strict=[]
ns=dict(c.normalize_original_alias_v7.__globals__,ROOT=work,ORIGINAL_STATE=True,RESUME_STATE=dict(checkpoint_identity=identity),
 V8_AMENDMENT_REF=v8,V8_MANIFEST_REF=manifest,V8_ALIAS_BEFORE_REF=beforeref,V8_ALIAS_AFTER_REF=afterref,
 V8_TERMINAL_REF=marker,RESUME_CHECKPOINT_REF=targetref,read=read,local=lambda p:work/p,
 alias_module_v7=lambda:a,validate_v8_failure_lineage_v9=lambda:True,
 alias_strict_boundary_v7=lambda *args:strict.append('actual_faked_strict_boundary_endpoint') or dict(SYNTHETIC=True))
consume=types.FunctionType(c.normalize_original_alias_v7.__code__,ns)
result=consume({},None,None,None)
assert result['unlink_replayed'] is False and len(strict)==1 and target.read_bytes()==original
checks.append(dict(name='tiny_original_alias_absent_consumed_without_unlink_with_strict_endpoint',status='PASS'))
os.symlink(target.name,check/'latest-rank0.pt')
rejected('tiny_reappeared_alias',lambda:consume({},None,None,None))
assert (check/'latest-rank0.pt').is_symlink() and target.read_bytes()==original
# Deliberate tiny-fixture cleanup only permits another independent failure case.
(check/'latest-rank0.pt').unlink()
wrong=types.FunctionType(c.normalize_original_alias_v7.__code__,dict(ns,V8_ALIAS_BEFORE_REF=dict(beforeref,sha256='0'*64)))
rejected('tiny_wrong_historical_digest',lambda:wrong({},None,None,None))
(check/'unexpected.pt').write_bytes(b'SYNTHETIC_EXTRA')
rejected('tiny_unexpected_original_regular_roster',lambda:consume({},None,None,None))
assert target.read_bytes()==original
checks.append(dict(name='tiny_original_bytes_preserved_all_cases',status='PASS'))
assert not any(n in sys.modules for n in ('torch','numpy','composer','torch_geometric','tskit','tsinfer'))
result=dict(schema='native_v9_independent_fake_boundary_tests_v1',status='PASS_SOFTWARE_ONLY',checks=checks,
 actual_pinned_interpreter=sys.executable,actual_native_nested_code_object=native.run_v1.__code__.co_filename,
 actual_process_GPU_capacity_accounting_guard_checkpoint_or_scientific_execution=False,
 owned_tiny_fixture_only=True,source_files_modified=False)
(OUT/'independent_boundary_tests.v9.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status='PASS',checks=len(checks))))
