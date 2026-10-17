"""Focused independent parser/ownership/copy/lifecycle checks. No live adoption or operational modules."""
from pathlib import Path
from types import ModuleType,FunctionType,CodeType,SimpleNamespace
from contextlib import contextmanager
import ast,copy,hashlib,json,marshal,os,stat,sys,threading,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D='research/reviews/consumer_strict_JSON_parse_memo_source_20261017_v1/'
OUT=ROOT/'research/reviews/consumer_strict_JSON_parse_memo_independent_20261017_v1'
checks=[];negative=[];verified={}
def need(ok,message):
 if not ok:raise ValueError(message)
def passed(name,ok=True):need(ok,name);checks.append(name)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def packed(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
def bounded(path,expected=None):
 p=ROOT/path;s=p.lstat();need(p.resolve()==p and stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=2**21,'Bounded immutable source/metadata')
 raw=p.read_bytes();a=p.lstat();key=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
 need(key(s)==key(a)and len(raw)==s.st_size,'Stable source bytes');digest=sha(raw);need(expected is None or digest==expected,'Exact pinned source '+path);verified[path]=digest;return raw
def inert_module(path,name,raw,extra=None):
 m=ModuleType(name);m.__file__=str(ROOT/path);m.__dict__.update(dict(Path=Path,CodeType=CodeType,FunctionType=FunctionType,contextmanager=contextmanager,ast=ast,copy=copy,hashlib=hashlib,json=json,marshal=marshal,os=os,stat=stat,sys=sys,threading=threading));m.__dict__.update(extra or {})
 nodes=[n for n in ast.parse(raw).body if not isinstance(n,(ast.Import,ast.ImportFrom,ast.If))]
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),m.__file__,'exec',dont_inherit=True,optimize=0),m.__dict__);return m
def bare(path,name,raw,names,extra):
 m=ModuleType(name);m.__file__=str(ROOT/path);m.__dict__.update(extra);root=compile(raw,m.__file__,'exec',dont_inherit=True,optimize=0)
 for name in names:
  code=next(c for c in root.co_consts if isinstance(c,CodeType)and c.co_name==name);node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)and n.name==name)
  defaults=tuple(ast.literal_eval(x)for x in node.args.defaults)or None;m.__dict__[name]=FunctionType(code,m.__dict__,name,defaults)
 return m
class RAMLock:
 def __enter__(self):return self
 def __exit__(self,*args):return False

def main():
 handoff=json.loads(bounded(D+'HANDOFF.v1.json','936abdbda30715112c097574955fc92923d4737aefc645dad0ef2893abf97b56'));manifest=json.loads(bounded(D+'MANIFEST.v1.json','9ee30c5c3c33cb6864ecb53763ceceaee841d6609ba75b49f7de9b6b19e38d0d'))
 for r in manifest['files']:bounded(r['path'],r['sha256'])
 passed('all8source_manifest_members_authenticated',len(manifest['files'])==8)
 raw=bounded(D+'parse_memo_v1.py','312c9318912097d2b4c08e146bef41b1d055d089e1765707603a7c90da3a9d6c');p=inert_module(D+'parse_memo_v1.py','_independent_RAM_parse_component',raw)
 v4raw=bounded(p.V4_REF['path'],p.V4_REF['sha256']);memo_raw=bounded(p.MEMO_REF['path'],p.MEMO_REF['sha256']);nsraw=bounded(p.NS_REF['path'],p.NS_REF['sha256'])
 v9path='scripts/project_budget_v9.py';v9raw=bounded(v9path,'483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e')
 v4=bare(p.V4_REF['path'],'_independent_RAM_v4_pure_frames',v4raw,['bound','parse','reference','scope_spec','canonical','file_bytes','mutex'],dict(Path=Path,json=json,MAX_JSON=32*2**20))
 budget=bare(v9path,'_independent_RAM_budget_frame',v9raw,['successor_paths'],dict(v4=v4))
 adapter=inert_module(p.MEMO_REF['path'],'_independent_RAM_original_adapter',memo_raw,dict(threading=SimpleNamespace(local=SimpleNamespace,Lock=RAMLock)))
 ns=inert_module(p.NS_REF['path'],'_independent_RAM_namespace',nsraw);ns._load_pinned_memo_v1=lambda:adapter
 handle=ns.install_consumer_v1(v4,budget,consumer_role='read_only_consumer',expected_source_path=ROOT/p.V4_REF['path'],expected_source_sha256=p.V4_REF['sha256'])
 passed('actual_original_namespace_install_and_handle_frames_on_RAM_modules',handle.receipt['namespace_extension_ref']==p.NS_REF and handle.receipt['original_memo2_ref']==p.MEMO_REF and handle.receipt['installed']is True)
 guardpath='research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py';guardref=dict(path=guardpath,sha256='72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb');guardraw=bounded(guardpath,guardref['sha256'])
 policy=dict(path='research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json',sha256='5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5');selector=dict(path='research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json',sha256='fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4')
 guard=bare(guardpath,'_independent_RAM_Census_pure_readers',guardraw,['require','_raw','_json','_completed_json'],dict(ROOT=ROOT,budget=budget,MAX_JSON=65536,MAX_COMPLETED_JSON=2**20,_POLICY_REF=policy,_SELECTOR_REF=selector,_REPAIR_AUTHORITY_REF=None))
 histpath='research/reviews/Census_observation_two_scope_retention_source_20261017_v1/source_manifest.v1.json';histbytes=bounded(histpath);histref=dict(path=histpath,sha256=sha(histbytes));historical=v4.parse(histbytes);eligible=historical['targets']['census_tsinfer_observation_v1/census-tsinfer-inspect-v1']['storage_ref'];genuine=bounded(eligible['path'],eligible['sha256']);baseline=v4.parse(genuine)
 basekwargs=dict(guard_source_entries=[(guard,guardref)],frozen_history_manifest_refs=[histref],eligible_historical_refs=[eligible],current_control_refs=[policy,selector])
 memo=p.StrictJSONParseMemo(handle,v4,budget,**basekwargs)
 passed('complete_component_constructor_authenticates_original_pure_frames_genuine_history',memo.original_parse is v4.parse and memo.original_bound is v4.bound)
 # Every operand of the actual old lock/IO/snapshot operation below is a RAM provider.
 source_bytes=p._bytes;ram_bytes={};statebox={};events=[];provider={}
 def bytes_provider(r):
  if r['path']in ram_bytes:
   p._record(r);data=ram_bytes[r['path']];need(sha(data)==r['sha256'],'RAM immutable ref SHA');return data
  return source_bytes(r)
 p._bytes=bytes_provider
 def fresh_state(ref,data):
  key=handle._key(ROOT,ref['path'])[0]
  return dict(raw={key:(ROOT,ref['path'],('RAM_identity',),(data,ref['sha256']))},canonical={},entries=0,charge=0,raw_hits=0,raw_misses=0,raw_bytes_missed=0,canonical_hits=0,canonical_misses=0,capacity_bypasses=0,capacity_bypass_bytes=0)
 def original_reader(root,path):need(root==ROOT and path==provider['ref']['path'],'Exact uncached RAM path');events.append('final_uncached_read');return provider['raw'],provider['sha']
 @contextmanager
 def RAM_old_mutex(root):
  need(root==ROOT and handle._state()is None,'Single original RAM transaction');events.append('old_mutex_enter');handle.tls.state=statebox['state']
  try:
   yield
   need(v4.bound is currentmemo[0].original_bound and all(getattr(mod,name)is original for mod,name,original,_ in currentmemo[0].guard_readers),'Restored before original final snapshot frontier');events.append('old_final_snapshot_frontier')
  finally:handle.tls.state=None;events.append('old_mutex_exit')
 handle.original['file_bytes']=original_reader;currentmemo=[memo]
 def install(m,ref,data):
  currentmemo[0]=m;statebox['state']=fresh_state(ref,data);provider.update(ref=ref,raw=data,sha=ref['sha256']);m.expected_mutex=RAM_old_mutex;v4.mutex=RAM_old_mutex;m.install()
 def restore(m):m.uninstall();v4.mutex=handle.mutex_wrapper
 install(memo,eligible,genuine)
 with v4.mutex(ROOT):
  a=guard._json(eligible);a['reservation']['consumer_bindings'][0]['path']='OWN_COPY_MUTATION'
  b=guard._json(eligible);c=v4.bound(ROOT,eligible,True)
  passed('genuine_guard_and_bound_miss_hit_values_and_mutation_isolation',b==baseline and c==baseline and a is not b and b is not c and memo.active['counts']['parse_misses']==1 and memo.active['counts']['parse_hits']==2)
 passed('final_uncached_bytes_then_restored_old_snapshot_frontier',events==['old_mutex_enter','final_uncached_read','old_final_snapshot_frontier','old_mutex_exit']and memo.active is None)
 with v4.mutex(ROOT):guard._json(eligible);passed('next_original_operation_is_cold',memo.active['counts']['parse_misses']==1 and memo.active['counts']['parse_hits']==0)
 def rejection(name,fn,types=(ValueError,RuntimeError)):
  try:fn()
  except types:negative.append(name);return
  raise AssertionError('Unexpected acceptance '+name)
 with v4.mutex(ROOT):
  state=handle._state();entry=next(iter(state['raw'].values()));key=next(iter(state['raw']))
  state['raw'][key]=(entry[0],entry[1],entry[2],(bytes(bytearray(genuine)),entry[3][1]))
  rejection('equal_bytes_different_object_owner_rejected',lambda:memo.parse_record(eligible,genuine))
 statebox['state']=fresh_state(eligible,genuine)
 with v4.mutex(ROOT):
  memo.controls.add(eligible['path']);value=guard._json(eligible);passed('explicit_control_overlap_never_cached',value==baseline and not memo.active['parsed']and memo.active['counts']['uncached_parses']==1);memo.controls.remove(eligible['path'])
 for limit,label in [('raw_bytes','raw_cap_bypass'),('parsed_bytes','parsed_cap_bypass')]:
  with v4.mutex(ROOT):
   memo.active[limit]=p.MAX_RAW_BYTES if limit=='raw_bytes'else p.MAX_PARSED_BYTES
   passed(label,guard._json(eligible)==baseline and not memo.active['parsed']and memo.active['counts']['capacity_bypasses']==1)
 try:
  with v4.mutex(ROOT):guard._json(eligible);provider['raw']=genuine+b' '
 except ValueError:negative.append('final_uncached_bytes_drift_fails')
 else:raise AssertionError('Final drift accepted')
 provider['raw']=genuine;passed('failure_destroys_state_and_restores_seams',memo.active is None and v4.bound is memo.original_bound and all(getattr(mod,name)is original for mod,name,original,_ in memo.guard_readers))
 restore(memo)
 # Independent strict parser/value checks; no genuine source parser or JSON library changes.
 for label,data in [('duplicate',b'{"x":1,"x":2}'),('NaN',b'{"x":NaN}'),('Infinity',b'{"x":Infinity}')]:rejection('original_strict_reject_'+label,lambda data=data:v4.parse(data))
 samples=[b'null',b'false',b'123',b'[-0.0,1e999,"\\ud800",{},[]]']
 passed('original_parser_values_mutable_JSON_types_and_existing_float_behavior',all(type(v4.parse(data))in(type(None),bool,int,list)for data in samples))
 # Legitimate API-format provenance in RAM for synthetic valid deep JSON; no real record/source is changed.
 depth=600;deep=('['*depth+'0'+']'*depth).encode();deepref=dict(path='research/reviews/frozen_nested_value.json',sha256=sha(deep));frozen=packed(dict(targets={},runtime_authenticated_historical_source_and_metadata_refs=[deepref]));frozenref=dict(path='research/reviews/frozen_provenance_manifest.json',sha256=sha(frozen));ram_bytes[frozenref['path']]=frozen
 original=v4.parse(deep);passed('original_pure_parser_accepts_valid_deep_JSON',isinstance(original,list)and len(deep)==1201)
 deepmemo=p.StrictJSONParseMemo(handle,v4,budget,**dict(basekwargs,frozen_history_manifest_refs=[frozenref],eligible_historical_refs=[deepref]));passed('deep_value_constructor_accepts_exact_frozen_eligibility_and_original_frames',deepmemo.eligible=={deepref['path']:deepref['sha256']})
 install(deepmemo,deepref,deep);regression={}
 try:
  with v4.mutex(ROOT):
   entry=next(iter(handle._state()['raw'].values()));passed('deep_reproduction_uses_exact_current_old_raw_object_path_and_SHA',entry[3][0]is deep and entry[3][1]==deepref['sha256']and deepmemo._owned(deepref,deep)==deepref['path'])
   guard._json(deepref)
 except RecursionError as error:regression=dict(confirmed=True,original_parser='success',component='RecursionError',message=str(error),depth=depth,raw_bytes=len(deep),recursion_limit=sys.getrecursionlimit(),path=deepref['path'],sha256=deepref['sha256'],full_component_constructor_passed=True,exact_original_old_namespace_handle=True,exact_original_guard_reader_frame=True,exact_current_raw_path_SHA_object_owner=True,source_candidate_changed=False)
 else:raise AssertionError('Expected copy recursion regression did not reproduce')
 passed('copy_failure_cleans_parse_state_and_restores_originals',deepmemo.active is None and v4.bound is deepmemo.original_bound and all(getattr(mod,name)is original for mod,name,original,_ in deepmemo.guard_readers));restore(deepmemo)
 passed('caps_exact_and_global_parse_never_patched',p.MAX_ENTRIES==256 and p.MAX_RAW_BYTES==4194304 and p.MAX_PARSED_BYTES==33554432 and v4.parse is memo.original_parse)
 passed('default_guardless_noadoption_inert_source',b'"status":"INACTIVE_SOURCE_ONLY"'in raw and b'"actual_consumer_adoption":False'in raw)
 need(sha(bounded(D+'parse_memo_v1.py'))=='312c9318912097d2b4c08e146bef41b1d055d089e1765707603a7c90da3a9d6c','Candidate preserved')
 return dict(schema='consumer_strict_JSON_parse_memo_independent_focused_checks_v1',status='SOURCE_BLOCKED_COPY_RECURSION_REGRESSION',positive_count=len(checks),negative_count=len(negative),checks=checks,negative_cases=negative,confirmed_regression=regression,verified_selected_refs=verified,author_campaign_repeated=False,original_parser_executed=True,actual_operational_modules_imported=False,original_namespace_install_on_RAM_modules_only=True,original_lock_read_snapshot_replaced_with_RAM_providers=True,deep_frozen_provenance_and_valid_JSON_are_RAM_fixtures=True,genuine_historical_metadata_baseline=True,actual_state_lock_process_GPU_accounting_or_Git_operations=False,actual_clock_read=False,actual_live_adoption=False,whole_route_control_exclusion_completeness_asserted=False,runtime_speed_or_fit_claimed=False)
try:result=main()
except BaseException:
 diagnostic=traceback.format_exc();(OUT/'first_diagnostic.v1.txt').write_text(diagnostic);print(diagnostic,flush=True);raise
with (OUT/'checks.result.v1.json').open('xb')as stream:stream.write(packed(result))
print(json.dumps({k:result[k]for k in ('status','positive_count','negative_count','confirmed_regression','actual_live_adoption')},sort_keys=True))
