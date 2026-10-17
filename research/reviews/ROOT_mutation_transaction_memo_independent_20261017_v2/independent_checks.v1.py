"""Independent memo-v2 source/actual-predicate AST and memory-only fixtures."""
from pathlib import Path,PurePosixPath
from types import ModuleType,SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,json,sys
R=Path('/workspace-SR003.nfs2/ARG-GNN');S=R/'research/reviews/ROOT_mutation_transaction_memo_source_20261017_v2';O=R/'research/reviews/ROOT_mutation_transaction_memo_independent_20261017_v2'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def ref(p):
 p=Path(p);p=p if p.is_absolute()else R/p
 return dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def fn(t,n):return next(x for x in t.body if isinstance(x,ast.FunctionDef)and x.name==n)
def iso(nodes,ns):exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),'<independent-memory-AST>','exec'),ns);return ns
def rejects(action):
 try:action()
 except (ValueError,KeyError,TypeError):return True
 return False
m=json.loads((S/'source_manifest.v1.json').read_text());a=json.loads((S/'audit_manifest.v1.json').read_text())
for ps in (m['source_sha256'],a['files_sha256']):
 for p,h in ps.items():need(ref(p)['sha256']==h,'SHA '+p)
source=(S/'root_mutation_memo_v1.py').read_bytes();tree=ast.parse(source)
need(all(isinstance(n,(ast.Expr,ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef,ast.ClassDef,ast.If))for n in tree.body),'top-levelshape')
for n in tree.body:
 if isinstance(n,ast.Assign):need(not any(isinstance(x,ast.Call)and ast.unparse(x)!="Path('/workspace-SR003.nfs2/ARG-GNN')"and not(isinstance(x.func,ast.Name)and x.func.id=='dict')for x in ast.walk(n)),'import-timeIO')
ns={'__name__':'independent_memo_definitions'};exec(compile(tree,str(S/'root_mutation_memo_v1.py'),'exec'),ns)
old=ast.parse((R/'research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1/root_mutation_memo_v1.py').read_text());new=copy.deepcopy(tree)
for t in (old,new):t.body=[n for n in t.body if not(isinstance(n,ast.ClassDef)and n.name=='_Transaction')]
for n in new.body:
 if isinstance(n,ast.Expr)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str):n.value.value=old.body[0].value.value
 if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='HERE'for x in n.targets):n.value.value='research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1'
for n in ast.walk(new):
 if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='_Transaction':n.keywords=[x for x in n.keywords if x.arg!='uncached_paths']
need(ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False),'undocumentedoutsideTransactiondelta')
class MP(PurePosixPath):
 def exists(self):return False
root=MP('/memory/source_review_never_materialized');locked=[True];reads={};canon=[0];badpath=[False]
memory={'src/code.py':b'pass\n','data/history/config.json':b'{"value":[1]}','data/history/expiry.json':b'{"expires_at_utc":"2026-10-17T00:00:10+00:00"}','data/raw.json':b'{"raw":1}'}
current=['research/current_authority.json','research/revocation.json','research/observation.json','research/proof.json','research/instruction.json','research/review.json','research/manifest.json']
excluded=['.runtime/state.json','research/prefix.json','research/journal.json','research/fingerprint.json','research/inventory.json','data/payload.npz','data/model.pt']
for p in current+excluded:memory[p]=b'{"fresh":1}'
def relative(p):return str(MP(p).relative_to(root))if MP(p).is_absolute()else str(MP(p))
def fb(r,p):
 need(locked[0] and r==root,'outsidefixturelock');p=relative(p);reads[p]=reads.get(p,0)+1;v=memory[p];return v,hashlib.sha256(v).hexdigest()
def canonical(r,p):
 need(locked[0] and r==root,'outsidefixturelock');canon[0]+=1;p=MP(p);p=p if p.is_absolute()else r/p;return p/'drift'if badpath[0]else p
v4=ModuleType('independent_memory_v4');v4.__dict__.update(Path=MP,json=json,hashlib=hashlib,file_bytes=fb,canonical=canonical)
v4src=ast.parse((R/ns['PINS']['v4']['path']).read_text());iso([fn(v4src,n)for n in ('parse','bound','read_json','reference','integer','paths_overlap','scope_spec')],v4.__dict__)
v4.accounting_paths=lambda r:[];v4.mutex=lambda r:None
budget=ModuleType('independent_memory_budget');budget.__dict__.update(v4=v4,JOURNAL_REL='.runtime/journal')
budgetsrc=ast.parse((R/ns['PINS']['budget']['path']).read_text());iso([fn(budgetsrc,'successor_paths')],budget.__dict__)
original={k:getattr(v4,k)for k in ('bound','reference','scope_spec','canonical','file_bytes','mutex')};originalsucc=budget.successor_paths
refs=lambda p:dict(path=p,sha256=hashlib.sha256(memory[p]).hexdigest())
# Original bound syntax and strict parse on every cache call, including hits.
tx=ns['_Transaction'](root,v4,budget,{},uncached_paths=current);tx.install();cfg='data/history/config.json';cr=refs(cfg)
for _ in range(10):
 value=v4.bound(root,cr,True);value['value'].append(99)
 need(v4.bound(root,cr,True)=={'value':[1]},'parsedobjectcached')
need(reads[cfg]==1 and tx.counts['json_hits']==19,'dynamicJSONhit')
for bad in ({'path':cfg,'sha256':cr['sha256'],'extra':1},{'path':cfg,'sha256':'A'*64},{'path':cfg,'sha256':'short'}):
 before=reads[cfg];need(rejects(lambda:v4.bound(root,bad,True))and reads[cfg]==before,'originalsyntaxgatebeforehit')
need(rejects(lambda:v4.bound(root,dict(path=cfg,sha256='f'*64),True))and reads[cfg]==1,'conflictingSHAhit')
# SHA-bound raw bytes follow original false-as_json semantics; later strict parse still runs.
for p,raw in [('research/duplicate.json',b'{"x":1,"x":2}'),('research/nonfinite.json',b'{"x":NaN}'),('research/invalid.json',b'{')]:
 memory[p]=raw
 for _ in range(2):need(rejects(lambda:v4.bound(root,refs(p),True)),'strictoriginalparse')
 need(p not in tx.sources and reads[p]==2,'invalidparseenrolled')
# Raw bound retains original behavior; later requested strict parsing is never bypassed.
p='research/raw_invalid.json';memory[p]=b'{';rr=refs(p)
need(v4.bound(root,rr,False)==b'{' and p in tx.sources,'originalrawboundsemantics')
for _ in range(2):need(rejects(lambda:v4.bound(root,rr,True)),'laterrequestedstrictparse')
need(reads[p]==1,'rawthenJSONhitfixture')
# Current-operation evidence and exclusion classes never byte-cache even when SHA-bound.
for p in current+excluded:
 for _ in range(3):v4.bound(root,refs(p),True)
 need(reads[p]==3 and p not in tx.sources,'excludedcached '+p)
for p in current:
 memory[p]=b'{"fresh":2}';need(v4.bound(root,refs(p),True)=={'fresh':2} and reads[p]==4,'currentfreshread')
# Raw JSON read_json/reference/file_bytes stay uncached after actual expectedSHA caching.
before=reads[cfg]
for _ in range(3):v4.file_bytes(root,cfg);v4.read_json(root,cfg);v4.reference(root,cfg)
need(reads[cfg]==before+9,'rawJSONcached')
# Exact existing _cpu_reader bypass now uses dynamic bound source bytes, no module exec.
terminal='src/arggnn/project_storage_terminal_scopes_v1.py';cpupath='src/arggnn/project_storage_cpu_closed_scopes_v1.py';prevpath='src/arggnn/project_storage_closed_scopes_v1.py'
for p in (cpupath,prevpath):memory[p]=(R/p).read_bytes()
sentinel={'old_frozen_namespace':True};scope=dict(budget=budget,ROOT=root,CPU_SOURCE=cpupath,CPU_SHA256=hashlib.sha256(memory[cpupath]).hexdigest(),previous=SimpleNamespace(SOURCE=prevpath),PREVIOUS_SHA256=hashlib.sha256(memory[prevpath]).hexdigest(),_CPU=sentinel)
cpu=iso([fn(ast.parse((R/terminal).read_text()),'_cpu_reader')],scope)['_cpu_reader']
for _ in range(10):need(cpu()is sentinel,'cpu_existingnamespace')
need(reads[cpupath]==reads[prevpath]==1,'cpu_reader_bypass_notcovered')
# Per-call original current UTC predicate runs after reparsing an actual JSON hit.
clock=[datetime(2026,10,17,tzinfo=timezone.utc)]
class Clock:
 fromisoformat=staticmethod(datetime.fromisoformat)
 @staticmethod
 def now(tz):return clock[0]
guardsrc=ast.parse((R/ns['PINS']['guard']['path']).read_text());expiry=iso([fn(guardsrc,'_expire')],dict(datetime=Clock,timezone=timezone,require=need))['_expire'];er=refs('data/history/expiry.json')
expiry(v4.bound(root,er,True));clock[0]+=timedelta(seconds=11)
need(rejects(lambda:expiry(v4.bound(root,er,True)))and reads[er['path']]==1,'timepredicatecached')
# Original primitives stay original; original final checks see every unique cached ref/path.
need(all(getattr(v4,k)is original[k]for k in ('canonical','file_bytes','mutex')),'originalprimitivepatched')
unique=list(tx.sources);before=dict(reads);names=list(tx.names);tx.finish()
need(not tx.enabled and all(reads[p]==before[p]+1 for p in unique),'uniqueuncachedfinalbytes')
need(tx.counts['revalidated_sources']==len(unique)and tx.counts['revalidated_names']==len(names),'uniquevalidationcounts')
before=reads[cfg];v4.bound(root,cr,True);need(reads[cfg]==before+1,'postfinishcachehit');tx.close()
need(not tx.sources and not tx.names and not tx.uncached and not tx.bytes,'destroy')
need(all(getattr(v4,k)is original[k]for k in original)and budget.successor_paths is originalsucc,'restore')
# Drift paths fail original final validation and close restores; zero crossgeneration hits.
drift_results=[]
for case in ('content','canonical','operation_error','patch_replaced'):
 saved=memory[cfg];badpath[0]=False;tx=ns['_Transaction'](root,v4,budget,{});tx.install();v4.bound(root,cr,True);v4.reference(root,cfg)
 if case=='content':memory[cfg]=b'{"value":[2]}'
 if case=='canonical':badpath[0]=True
 if case=='patch_replaced':v4.bound=lambda *a:None
 if case in ('content','canonical'):need(rejects(tx.finish),'driftnotblocked')
 if case=='patch_replaced':need(rejects(tx.close),'unexpectedreplacementnotnoticed')
 else:tx.close()
 memory[cfg]=saved;badpath[0]=False
 need(all(getattr(v4,k)is original[k]for k in original)and not tx.sources and not tx.names,'exceptionrestoration')
 drift_results.append(case)
for badkw in (dict(max_entries=0),dict(max_entries=2049),dict(max_entries=True),dict(max_bytes=0),dict(max_bytes=64*2**20+1)):
 need(rejects(lambda:ns['_Transaction'](root,v4,budget,{},**badkw)),'badlimits')
for _ in range(2):
 before=reads[cfg];tx=ns['_Transaction'](root,v4,budget,{},max_entries=1,max_bytes=64);tx.install();v4.bound(root,cr,True);v4.bound(root,er,True)
 need(len(tx.sources)<=1 and tx.bytes<=64 and reads[cfg]==before+1,'cachebound_or_crossgeneration');tx.finish();tx.close()
# Public reserve critical AST is unchanged beyond mutex moved to adapter; no actual guard call.
reserve=fn(guardsrc,'apply_reserve_v1');derived=ns['critical_reserve'](reserve,dict(__name__='independent_reserve_AST'),lambda *a:None)
need(derived.__code__.co_filename.endswith('#exact-reserve-critical-body'),'criticalidentity')
run=fn(next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='RootMutationMemo'),'_run')
withnode=next(n for n in run.body if isinstance(n,ast.Try)).body[0];need(isinstance(withnode,ast.With)and ast.unparse(withnode.items[0].context_expr)=='original_mutex(ROOT)','originalmutex')
constructor=next(n for n in ast.walk(run)if isinstance(n,ast.Call)and ast.unparse(n.func)=='_Transaction')
need(any(k.arg=='uncached_paths' and 'authority_ref' in ast.unparse(k.value)for k in constructor.keywords),'authorityexclusionmissing')
# Source function identity matcher fails changed callable/global provenance on pure synthetic module.
code=b'def target(x):\n return x+1\n';module=ModuleType('synthetic_identity_fixture');module.__file__=str(R/'research/synthetic_identity_fixture.py');exec(compile(code,module.__file__,'exec'),module.__dict__)
savedreader=ns['source_bytes'];ns['source_bytes']=lambda _:code
record=dict(path='research/synthetic_identity_fixture.py',sha256=hashlib.sha256(code).hexdigest());matcher=ns['functions_from_source'];matcher(module,record,('target',));module.target=lambda x:x
need(rejects(lambda:matcher(module,record,('target',))),'changedcallableaccepted');ns['source_bytes']=savedreader
need(not any(n.startswith(('arggnn','scripts.project_budget','_Relate64','_H058'))for n in sys.modules),'actualguardimport')
result=dict(schema='ROOT_transaction_memo_v2_independent_checks_v1',source_ref=ref(S/'root_mutation_memo_v1.py'),source_manifest_ref=ref(S/'source_manifest.v1.json'),source_handoff_ref=ref(S/'source_handoff.v1.json'),audit_manifest_ref=ref(S/'audit_manifest.v1.json'),manifest_refs_authenticated=len(m['source_sha256']),audit_refs_authenticated=len(a['files_sha256']),outside_Transaction_exact_v1_except_identity_and_authority_exclusion=True,actual_original_bound_and_strict_parse_each_hit=True,dynamic_JSON_new_object_every_call=True,syntax_and_SHA_conflicts_reject_before_read=True,invalid_duplicate_nonfinite_JSON_as_json_true_misses_never_enrol=True,raw_bound_then_requested_parse_original_semantics=True,explicit_current_operation_exclusions_tested=current,raw_JSON_and_runtime_prefix_fingerprint_inventory_payload_uncached=True,exact_cpu_reader_AST_dynamic_code_reads_once_then_final=True,actual_original_expire_rechecked_on_JSON_bytehit=True,unique_cached_refs_final_original_reads=len(unique),unique_canonical_final_original_checks=len(names),three_original_primitives_unchanged=True,exceptional_cases_restore=drift_results,caches_bounded_and_cold=True,public_reserve_current_authority_excluded=True,actualguard_state_mutex_resources_process_payload_science_Git=False,real_runtime_gain_measured=False)
(O/'independent_checks.v1.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(source_ref=result['source_ref'],manifest_refs=19,audit_refs=8,independent_fixtures='PASS',unique_cached_refs=len(unique),explicit_current_exclusions=len(current)),sort_keys=True))
