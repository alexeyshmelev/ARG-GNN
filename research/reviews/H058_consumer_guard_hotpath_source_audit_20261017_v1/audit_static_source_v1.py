"""Static source/JSON audit only. No production guard/module bodies are executed."""
from pathlib import Path
import ast,hashlib,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');HERE='research/reviews/H058_consumer_guard_hotpath_source_audit_20261017_v1';P=ROOT/HERE
CHECKS=[]
def need(v,n):
 if not v:raise AssertionError(n)
 CHECKS.append(dict(name=n,status='PASS'))
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def text(p):return (ROOT/p).read_text()
def fns(p):return {n.name:n for n in ast.parse(text(p)).body if isinstance(n,ast.FunctionDef)}
def calls(n,name):return [c for c in ast.walk(n)if isinstance(c,ast.Call)and ast.unparse(c.func)==name]
def literal(p,name):
 tree=ast.parse(text(p));n=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets));return ast.literal_eval(n.value)
pins=json.loads((P/'input_pins.v1.json').read_bytes())['refs']
for r in pins:need(ref(r['path'])==r,'Exact sealed source/recorded metadata '+r['path'])
base='research/reviews/H058_GPU0_full_training_successor_source_20261017_v5'
common=fns(base+'/common_v1.py');context=fns(base+'/context_factory_v1.py');driver=fns(base+'/root_driver_v1.py')
need('consumer=True'in ast.unparse(common['guard_v1'])and '_SELECTOR_API.guard(consumer)'in ast.unparse(common['guard_v1']),'H058 already uses exact cached selector/consumer route')
need('RootMutationMemo'in ast.unparse(driver['mutation_memo_v1'])and len(calls(driver['admit_v1'],'memo.read_state_v1'))==len(calls(driver['admit_v1'],'memo.reserve_v1'))==1,'ROOT memo already used for separate admission state/reserve transactions')
need(len(calls(fns('src/arggnn/train_ordered_clade_episode_v1.py')['train_one'],'guard'))==2,'Original train_one retains two guards per optimizer update')
guard=next(n for n in ast.walk(context['build_context_v1'])if isinstance(n,ast.FunctionDef)and n.name=='guard')
need(all(calls(guard,k)for k in('active','require_work_window_v2','resources','charged_census_v1','g.budget_check_v9')),'Step guard retains authority/resources/census/full budget checks')
need(len(calls(guard,'g.budget_check_v9'))==1 and ast.unparse(calls(guard,'g.budget_check_v9')[0])=='g.budget_check_v9(storage, boundary=False)','Exactly one full boundary=False accounting transaction per step guard')
need(len(calls(common['charged_census_v1'],'measured'))==3,'Charged census retains output/whole-packet/AUX walks')
need('fit_source_authenticator_v5'in ast.unparse(common['root_resources_view_v5'])and 'functions_from_source'not in ast.unparse(guard),'Fit function source authentication is context construction, not per-step compilation')
selector=fns('research/reviews/European_outer8_current_policy_successor_source_20261016_v1/prepare_outer000_v1.py')['guard']
need('install_consumer_v1'in ast.unparse(selector)and "hasattr(g, '_outer_memo')"in ast.unparse(selector),'Existing namespace/memo2 installation is guarded against duplicate adoption')
binding=json.loads((P/'immutable_parse_binding.INACTIVE.v1.json').read_bytes());outer=binding['outer_readers'];need(len(outer)==7,'Exact seven frozen outer guard layers')
policy_json=[]
for row in outer:
 fs=fns(row['source_ref']['path']);state=fs['_state_locked']
 need(len(calls(state,'_BASE._state_locked'))==1 and len(calls(state,'_policy'))==1,'Outer layer retains policy plus recursive state replay '+row['source_ref']['path'])
 policy_json.append(len(calls(fs['_policy'],'_json')))
 for name in row['parse_readers']:
  need('budget.v4.parse(raw)'in ast.unparse(fs[name])and '_raw(record)'in ast.unparse(fs[name]),'Exact original bound bytes→length→parse reader '+row['source_ref']['path']+'/'+name)
need(policy_json==[5,4,3,2,2,1,1],'Source policy-reader call counts preserved')
need(len(calls(fns(outer[0]['source_ref']['path'])['check_storage_locked_v1'],'_policy'))==1,'Top check repeats policy before its recursive state call')
low=text('research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py')
need("_CACHE[1]==before: return copy.deepcopy(_CACHE[2])"in low and "before = cpu['_fingerprint'](watched)"in low,'Warm marker-repaired ancestor cache retains fresh dependency fingerprint; not unconditional full base replay')
memo=text('research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py')
need('self._validate_snapshot(state)'in memo and 'state["raw"].clear();state["canonical"].clear()'in memo and 'self.tls.state=None'in memo,'Existing consumer memo snapshot validation and per-operation destruction remain')
need('parse('not in ast.unparse(next(n for n in ast.parse(memo).body if isinstance(n,ast.ClassDef)and n.name=='TransactionMemoAdapter')),'Existing memo2 caches raw bytes/namespace, not parsed JSON')
namespace=text('research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py')
need('scope_code,successor_code'in namespace and 'len(original_callers)+2'in namespace,'Existing namespace extension already includes both exact comprehensions')
startup=text('research/reviews/guard_latency_minimal_reuse_source_20261017_v1/startup_reuse_v1.py')
close=text('research/reviews/ROOT_completed_close_code_memo_source_20261017_v1/root_completed_close_code_memo_v1.py')
need('validate_storage_with_growth_v1'in startup and 'def close_completed_v1'in close,'Earlier candidates are startup-only and completed-close-only, not step guards')
old=json.loads((ROOT/'research/reviews/resource_guard_hotpath_source_audit_v1/root_readonly_live_inspection_v1.json').read_bytes())
need(old['elapsed_wall_seconds_including_lock_wait']==13.140873406082392 and old['CPU_seconds']==6.634369627 and old['production_speedup_claim']is False,'Recorded old v3/memo1 timing retained with no H058 speed extrapolation')
candidate=P/'immutable_parse_reuse.INACTIVE.v1.py';code=candidate.read_text();tree=ast.parse(code);compile(code,str(candidate),'exec')
need(all(not isinstance(n,ast.Expr)or isinstance(n.value,(ast.Constant,))for n in tree.body if not isinstance(n,ast.If)),'Candidate import has no operation/guard/installation call')
need(len(literal(HERE+'/immutable_parse_reuse.INACTIVE.v1.py','ALLOW_REFS'))==23,'Candidate finite exact23 immutable refs only')
need('entry[3][0]is not raw'in code and "entry[3][1]!=record['sha256']"in code and "state['raw'].get"in code,'Candidate hits require exact original memo raw object plus digest coverage')
need("return copy.deepcopy(cache['values'][identity])"in code and "cache['values'][identity]=copy.deepcopy(result)"in code,'Candidate avoids shared mutable parsed-result aliases')
need('self.memo._room(state,charge)'in code and 'MAX_PARSE_BYTES=16*2**20'in code and 'MAX_PARSE_ENTRIES=128'in code,'Candidate retains original memo total room plus fixed parsed cache bounds')
need("cache=state.pop(CACHE_FIELD);cache['values'].clear();cache['bytes']=0"in code and 'with self.old_mutex(root)'in code,'Candidate destroys only parsed cache before original snapshot validation/unlock, including exceptions')
need(not any(k=='torch'or k.startswith(('arggnn','numpy'))for k in sys.modules),'No scientific or production project modules imported')
for r in pins:need(ref(r['path'])==r,'Final original source/recorded metadata unchanged '+r['path'])
result=dict(schema='H058_consumer_guard_hotpath_static_audit_checks_v1',status='PASS_STATIC_SOURCE_ONLY',source_only=True,active=False,execution_authorized=False,check_count=len(CHECKS),checks=CHECKS,source_pins_ref=ref(HERE+'/input_pins.v1.json'),candidate_ref=ref(HERE+'/immutable_parse_reuse.INACTIVE.v1.py'),candidate_status='UNQUALIFIED_INACTIVE_DRAFT_NOT_FOR_INSTALLATION',candidate_guard_lifecycle_or_RAM_semantics_not_executed=True,production_guard_or_candidate_constructor_installation_executed=False,source_or_file_timing_probe=False,performance_measured=False)
with(P/'source_checks.result.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],check_count=len(CHECKS),candidate_status=result['candidate_status']),sort_keys=True),flush=True)
