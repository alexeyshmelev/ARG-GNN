"""Independent v2 delta and targeted exact-source RAM checks. No actual adoption or operational providers."""
from pathlib import Path
from types import ModuleType,FunctionType,CodeType,SimpleNamespace
from contextlib import contextmanager
import ast,copy,hashlib,json,sys,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D='research/reviews/consumer_strict_JSON_parse_memo_source_20261017_v2/';OUT=ROOT/'research/reviews/consumer_strict_JSON_parse_memo_independent_20261017_v2'
checks=[];negatives=[];verified={}
def need(ok,text):
 if not ok:raise ValueError(text)
def check(text,ok=True):need(ok,text);checks.append(text)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def pack(v):return (json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
def bounded(path,expected=None):
 import stat
 p=ROOT/path;s=p.lstat();need(p.resolve()==p and stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=2**21,'Bounded singly-linked source/metadata');data=p.read_bytes();a=p.lstat();identity=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
 need(identity(s)==identity(a)and len(data)==s.st_size,'Stable source/metadata');digest=sha(data);need(expected is None or digest==expected,'Pinned '+path);verified[path]=digest;return data
def full(path,name,raw):
 m=ModuleType(name);m.__file__=str(ROOT/path);exec(compile(raw,m.__file__,'exec',dont_inherit=True,optimize=0),m.__dict__);return m

def bare(path,name,raw,names,constants):
 m=ModuleType(name);m.__file__=str(ROOT/path);m.__dict__.update(constants);top=compile(raw,m.__file__,'exec',dont_inherit=True,optimize=0)
 for n in names:
  node=next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef)and x.name==n);code=next(x for x in top.co_consts if isinstance(x,CodeType)and x.co_name==n);defaults=tuple(ast.literal_eval(x)for x in node.args.defaults)or None;setattr(m,n,FunctionType(code,m.__dict__,n,defaults))
 return m
class RAMLock:
 def __enter__(self):return self
 def __exit__(self,*args):return False

def leaf(value):
 ids=[]
 while type(value)is list:ids.append(id(value));need(len(value)==1,'Single nested list');value=value[0]
 return value,ids

def mutate_leaf(value):
 while type(value[0])is list:value=value[0]
 value[0]=99

def deep_call(n,fn):return deep_call(n-1,fn)if n else fn()

def main():
 h=json.loads(bounded(D+'HANDOFF.v2.json','eef069c8c5354b2a2b260c9eef00568c0ce789e63bd1c5f84f03de264bab95b0'));m=json.loads(bounded(D+'MANIFEST.v2.json','c7bb705cd07ae7be61c0cce58a546c90418db3a4c089d1bccda856b3b7127153'))
 for ref in m['files']:bounded(ref['path'],ref['sha256'])
 check('all9v2manifest_members_authenticated',len(m['files'])==9)
 v2path=D+'parse_memo_v2.py';v2raw=bounded(v2path,'62cc982b18ce8819998270ea9e583d60e76f1bebbd9e943bab6f7d74eaba42c6');v1path='research/reviews/consumer_strict_JSON_parse_memo_source_20261017_v1/parse_memo_v1.py';v1raw=bounded(v1path,'312c9318912097d2b4c08e146bef41b1d055d089e1765707603a7c90da3a9d6c')
 oldreview=json.loads(bounded(h['blocked_v1_review_ref']['path'],h['blocked_v1_review_ref']['sha256']));check('v1_BLOCKED_review_preserved',oldreview['verdict']=='BLOCKED'and h['blocked_v1_review_ref']['sha256']=='c423abd71b291e29d40fa10402c846eb23cfca55337d1db945391ba97a3ec845')
 t1,t2=ast.parse(v1raw),ast.parse(v2raw);c1=next(x for x in t1.body if isinstance(x,ast.ClassDef));c2=next(x for x in t2.body if isinstance(x,ast.ClassDef));methods1={x.name:x for x in c1.body if isinstance(x,ast.FunctionDef)};methods2={x.name:x for x in c2.body if isinstance(x,ast.FunctionDef)}
 changed={n for n in methods1 if ast.dump(methods1[n],include_attributes=False)!=ast.dump(methods2[n],include_attributes=False)}
 check('exact_class_delta_is_fallback_final_and_transaction_state_only',changed=={'parse_record','_final','_mutex_wrapper'}and set(methods2)-set(methods1)=={'_mark_copy_uncacheable'})
 module1=full(v1path,'_independent_v1_delta_constants',v1raw);module2=full(v2path,'_independent_v2_delta_constants',v2raw)
 check('constructor_ownership_controls_parser_derivation_caps_unchanged',all(ast.dump(methods1[n],include_attributes=False)==ast.dump(methods2[n],include_attributes=False)for n in ('__init__','_owned','_bound','install','uninstall'))and all(getattr(module1,k)==getattr(module2,k)for k in ('V4_REF','MEMO_REF','NS_REF','FORBIDDEN_WORDS','MAX_ENTRIES','MAX_RAW_BYTES','MAX_PARSED_BYTES')))
 v4raw=bounded(module2.V4_REF['path'],module2.V4_REF['sha256']);adapterraw=bounded(module2.MEMO_REF['path'],module2.MEMO_REF['sha256']);nsraw=bounded(module2.NS_REF['path'],module2.NS_REF['sha256']);v9path='scripts/project_budget_v9.py';v9raw=bounded(v9path,'483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e')
 guardpath='research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py';guardraw=bounded(guardpath,'72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb');guardref=dict(path=guardpath,sha256=sha(guardraw))
 policy=dict(path='research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json',sha256='5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5');selector=dict(path='research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json',sha256='fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4')
 histpath='research/reviews/Census_observation_two_scope_retention_source_20261017_v1/source_manifest.v1.json';histbytes=bounded(histpath);histref=dict(path=histpath,sha256=sha(histbytes));history=json.loads(histbytes);realref=history['targets']['census_tsinfer_observation_v1/census-tsinfer-inspect-v1']['storage_ref'];realbytes=bounded(realref['path'],realref['sha256'])
 serial=[0]
 def setup(version=2,depth=None,data=None,genuine=False):
  serial[0]+=1;tag=str(serial[0]);path,raw=(v1path,v1raw)if version==1 else(v2path,v2raw);p=full(path,'_independent_RAM_component_'+tag,raw)
  v4=bare(p.V4_REF['path'],'_independent_RAM_v4_'+tag,v4raw,['bound','parse','reference','scope_spec','canonical','file_bytes','mutex'],dict(Path=Path,json=json,MAX_JSON=32*2**20));budget=bare(v9path,'_independent_RAM_budget_'+tag,v9raw,['successor_paths'],dict(v4=v4))
  adapter=full(p.MEMO_REF['path'],'_independent_RAM_full_original_adapter_'+tag,adapterraw);adapter.threading=SimpleNamespace(local=SimpleNamespace,Lock=RAMLock)
  ns=full(p.NS_REF['path'],'_independent_RAM_full_namespace_'+tag,nsraw);ns._load_pinned_memo_v1=lambda:adapter
  handle=ns.install_consumer_v1(v4,budget,consumer_role='read_only_consumer',expected_source_path=ROOT/p.V4_REF['path'],expected_source_sha256=p.V4_REF['sha256'])
  guard=bare(guardpath,'_independent_RAM_guard_readers_'+tag,guardraw,['require','_raw','_json','_completed_json'],dict(ROOT=ROOT,budget=budget,MAX_JSON=65536,MAX_COMPLETED_JSON=2**20,_POLICY_REF=policy,_SELECTOR_REF=selector,_REPAIR_AUTHORITY_REF=None))
  payload=realbytes if genuine else data if data is not None else ('['*depth+'0'+']'*depth).encode();target=realref if genuine else dict(path='research/reviews/frozen_copy_value.json',sha256=sha(payload))
  provenance=histref
  if not genuine:
   frozen=pack(dict(targets={},runtime_authenticated_historical_source_and_metadata_refs=[target]));provenance=dict(path='research/reviews/frozen_history_manifest.json',sha256=sha(frozen));original_bytes=p._bytes
   def ram_bytes(ref):
    if ref['path']==provenance['path']:p._record(ref);need(ref['sha256']==sha(frozen),'RAM frozen SHA');return frozen
    return original_bytes(ref)
   p._bytes=ram_bytes
  kwargs=dict(guard_source_entries=[(guard,guardref)],frozen_history_manifest_refs=[provenance],eligible_historical_refs=[target],current_control_refs=[policy,selector]);memo=p.StrictJSONParseMemo(handle,v4,budget,**kwargs)
  item=handle._key(ROOT,target['path'])[0];state=dict(raw={item:(ROOT,target['path'],('RAM_identity',),(payload,target['sha256']))},canonical={},entries=0,charge=0,raw_hits=0,raw_misses=0,raw_bytes_missed=0,canonical_hits=0,canonical_misses=0,capacity_bypasses=0,capacity_bypass_bytes=0);events=[];provider=dict(raw=payload,sha=target['sha256'])
  def reader(root,path):need(root==ROOT and path==target['path'],'Exact final RAM path');events.append('final_uncached_bytes');return provider['raw'],provider['sha']
  handle.original['file_bytes']=reader
  @contextmanager
  def ram_mutex(root):
   need(root==ROOT and handle._state()is None,'Original RAM lock state');handle.tls.state=state;events.append('old_mutex_enter')
   try:
    yield
    need(v4.bound is memo.original_bound and all(getattr(mod,n)is old for mod,n,old,_ in memo.guard_readers),'Exact restoration frontier');events.append('old_final_snapshot_frontier')
   finally:handle.tls.state=None;events.append('old_mutex_exit')
  memo.expected_mutex=ram_mutex;v4.mutex=ram_mutex;memo.install()
  return SimpleNamespace(p=p,v4=v4,budget=budget,guard=guard,handle=handle,memo=memo,target=target,data=payload,events=events,provider=provider,state=state,kwargs=kwargs)
 old=setup(1,600);check('original_strict_parser_accepts_same_depth600_fixture',leaf(old.memo.original_parse(old.data))[0]==0 and len(old.data)==1201 and sha(old.data)=='250f8f81de72fe62fb3c3202430524b7a6938c02969b8be425e8b265d6899eea')
 try:
  with old.v4.mutex(ROOT):old.guard._json(old.target)
 except RecursionError:check('preserved_v1_ownership_path_reproduces_copy_failure')
 else:raise AssertionError('V1 regression disappeared')
 old.memo.uninstall()
 fixed=setup(2,600)
 with fixed.v4.mutex(ROOT):
  check('v2_full_constructor_source_frames_and_owned_raw_path',fixed.memo._owned(fixed.target,fixed.data)==fixed.target['path']and fixed.handle.receipt['original_memo2_ref']==fixed.p.MEMO_REF)
  a=fixed.guard._json(fixed.target);b=fixed.v4.bound(ROOT,fixed.target,True);c=fixed.guard._completed_json(fixed.target)
  check('initial_fallback_and_all_repeated_reader_bound_returns_are_fresh',leaf(a)[0]==leaf(b)[0]==leaf(c)[0]==0 and not(set(leaf(a)[1])&set(leaf(b)[1]))and not(set(leaf(b)[1])&set(leaf(c)[1])))
  mutate_leaf(a);check('fallback_return_mutation_cannot_affect_later_original_parse',leaf(fixed.guard._json(fixed.target))[0]==0 and leaf(b)[0]==0)
  state=fixed.memo.active;marker=state['uncacheable'][fixed.target['path']]
  check('initial_fallback_evicts_parsed_object_releases_only_parsed_charge',not state['parsed']and state['parsed_bytes']==0 and state['raw_bytes']==1201 and set(marker)=={'raw','sha256'}and marker['raw']is fixed.data and state['counts']['deepcopy_recursion_fallbacks']==1 and state['counts']['uncached_parses']==3)
 check('fallback_final_original_bytes_then_restored_old_snapshot',fixed.events[-3:]==['final_uncached_bytes','old_final_snapshot_frontier','old_mutex_exit']and state['parsed']=={}and state['uncacheable']=={}and fixed.memo.active is None)
 with fixed.v4.mutex(ROOT):fixed.guard._json(fixed.target);check('new_original_transaction_has_cold_uncacheable_map',fixed.memo.active['counts']['deepcopy_recursion_fallbacks']==1)
 try:
  with fixed.v4.mutex(ROOT):fixed.guard._json(fixed.target);state=fixed.memo.active;fixed.provider['raw']=fixed.data+b' '
 except ValueError:negatives.append('uncacheable_final_bytes_drift_rejects')
 else:raise AssertionError('Final bytes drift accepted')
 check('drift_restores_and_clears_both_maps',fixed.memo.active is None and state['parsed']=={}and state['uncacheable']=={}and fixed.v4.bound is fixed.memo.original_bound);fixed.provider['raw']=fixed.data;fixed.memo.uninstall()
 hit=setup(2,300);check('original_parser_accepts_later_deeper_caller',leaf(deep_call(450,lambda:hit.memo.original_parse(hit.data)))[0]==0)
 with hit.v4.mutex(ROOT):
  first=hit.guard._json(hit.target);canonical=hit.memo.active['parsed'][hit.target['path']]['parsed'];check('shallower_first_read_is_actually_cached',hit.memo.active['counts']['deepcopy_recursion_fallbacks']==0)
  second=deep_call(450,lambda:hit.guard._json(hit.target));third=hit.guard._json(hit.target)
  check('hit_copy_recursion_evicts_and_freshly_parses',not hit.memo.active['parsed']and set(hit.memo.active['uncacheable'][hit.target['path']])=={'raw','sha256'}and hit.memo.active['parsed_bytes']==0 and hit.memo.active['raw_bytes']==601 and hit.memo.active['counts']['deepcopy_recursion_fallbacks']==1)
  check('hit_fallback_returns_never_alias_old_canonical_or_each_other',all(not(set(leaf(x)[1])&set(leaf(y)[1]))for x,y in ((first,second),(second,third),(second,canonical))))
  mutate_leaf(second);check('hit_fallback_mutation_does_not_escape',leaf(first)[0]==leaf(third)[0]==leaf(canonical)[0]==0)
 check('hit_fallback_keeps_final_uncached_read',hit.events[-3:]==['final_uncached_bytes','old_final_snapshot_frontier','old_mutex_exit']);hit.memo.uninstall()
 genuine=setup(genuine=True);baseline=genuine.memo.original_parse(genuine.data)
 with genuine.v4.mutex(ROOT):
  a=genuine.guard._json(genuine.target);b=genuine.v4.bound(ROOT,genuine.target,True);check('genuine_historical_values_and_ordinary_hit_behavior_preserved',a==baseline and b==baseline and a is not b and genuine.memo.active['counts']['parse_hits']==1)
  genuine.memo.controls.add(genuine.target['path']);check('current_control_path_stays_uncached',genuine.guard._json(genuine.target)==baseline and genuine.memo.active['counts']['uncached_parses']==1);genuine.memo.controls.remove(genuine.target['path'])
 genuine.memo.uninstall()
 for name,data in [('duplicate',b'{"a":1,"a":2}'),('NaN',b'{"a":NaN}'),('Infinity',b'{"a":Infinity}'),('syntax',b'{')]:
  bad=setup(data=data)
  try:bad.memo.original_parse(data)
  except ValueError as e:original_type=type(e);original_message=str(e)
  else:raise AssertionError('Original strict parser accepted '+name)
  try:
   with bad.v4.mutex(ROOT):bad.guard._json(bad.target)
  except ValueError as e:need(type(e)is original_type and str(e)==original_message,'Exact original parse error');negatives.append('unchanged_original_'+name+'_error')
  else:raise AssertionError('Component swallowed original parse error')
  need(bad.memo.active is None,'Strict error cleanup');bad.memo.uninstall()
 for mode in ('miss','hit'):
  other=setup(2,3);saved=other.p.copy
  try:
   with other.v4.mutex(ROOT):
    if mode=='hit':other.guard._json(other.target)
    def memory_error(value):raise MemoryError('INDEPENDENT_RAM_MEMORYERROR')
    other.p.copy=SimpleNamespace(deepcopy=memory_error);other.guard._json(other.target)
  except MemoryError:negatives.append('unchanged_'+mode+'_MemoryError')
  else:raise AssertionError('MemoryError swallowed')
  finally:other.p.copy=saved
  need(other.memo.active is None and other.v4.bound is other.memo.original_bound,'Other error restoration');other.memo.uninstall()
 check('only_copy_RecursionError_is_caught_original_parser_errors_untouched',all(type(n.type)is ast.Name and n.type.id=='RecursionError'for n in ast.walk(methods2['parse_record'])if isinstance(n,ast.ExceptHandler)))
 check('raw_entry_and_total_path_caps_remain_bounded',module2.MAX_ENTRIES==256 and module2.MAX_RAW_BYTES==4194304 and module2.MAX_PARSED_BYTES==33554432 and 'len(cache)>=MAX_ENTRIES'in v2raw.decode()and '0<len(self.eligible)<=MAX_ENTRIES'in v2raw.decode())
 check('global_parser_unpatched_and_default_inert',genuine.v4.parse is genuine.memo.original_parse and '"status":"INACTIVE_SOURCE_ONLY"'in v2raw.decode()and '"actual_consumer_adoption":False'in v2raw.decode())
 need(sha(bounded(v1path))=='312c9318912097d2b4c08e146bef41b1d055d089e1765707603a7c90da3a9d6c','V1 remains immutable');need(sha(bounded(v2path))=='62cc982b18ce8819998270ea9e583d60e76f1bebbd9e943bab6f7d74eaba42c6','V2 remains immutable')
 return dict(schema='strict_JSON_parse_memo_v2_independent_targeted_checks_v1',status='PASS',positive_count=len(checks),negative_count=len(negatives),checks=checks,negative_cases=negatives,verified_selected_refs=verified,changed_original_methods=sorted(changed),original_accept_v1_fail_v2_pass=True,full_constructor_and_original_source_frames=True,deepcopy_initial_and_hit_fallback=True,raw_SHA_only_markers_and_accounting_verified=True,genuine_historical_metadata_baseline=True,original_scheduler_mutex_IO_snapshot_are_RAM=True,actual_runtime_guard_imports=False,actual_live_state_resource_process_GPU_job_lease_install_or_Git_operations=False,actual_live_adoption=False,actual_clock_read=False,whole_route_control_completeness_or_performance_claimed=False,interpreter=sys.version.split()[0])
try:result=main()
except BaseException:
 diagnostic=traceback.format_exc();(OUT/'first_diagnostic.v2.txt').write_text(diagnostic);print(diagnostic,flush=True);raise
with(OUT/'checks.result.v2.json').open('xb')as f:f.write(pack(result))
print(json.dumps({k:result[k]for k in ('status','positive_count','negative_count','original_accept_v1_fail_v2_pass','actual_live_adoption','interpreter')},sort_keys=True))
