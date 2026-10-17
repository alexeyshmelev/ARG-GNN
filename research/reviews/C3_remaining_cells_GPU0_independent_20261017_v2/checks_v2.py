"""Independent pinned-source and focused RAM AST checks. No real guard/process calls."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace,ModuleType,FunctionType,CodeType
import ast,collections,copy,hashlib,json,marshal
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
D=ROOT/'research/reviews/C3_remaining_cells_GPU0_source_20261017_v2'
OLD=ROOT/'research/reviews/C3_remaining_cells_GPU0_source_20261017_v1'
OUT=ROOT/'research/reviews/C3_remaining_cells_GPU0_independent_20261017_v2'
checks=[];rejects=[];authenticated={}
def need(ok,label):
    if not ok:raise AssertionError(label)
    checks.append(label)
def reference(p):
    p=Path(p);p=p if p.is_absolute()else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def auth(r):
    need(Path(r['path']).suffix in ('.py','.json','.md','.txt','.patch'),'source metadata allowlist '+r['path'])
    need(reference(r['path'])==r,'exact source pin '+r['path']);authenticated[r['path']]=r
    return (ROOT/r['path']).read_bytes()
def doc(p):return json.loads(Path(p).read_text())
def nodes(p):return {n.name:n for n in ast.parse(Path(p).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def execute_defs(ns,source,names,label):
    exec(compile(ast.Module(body=[source[n]for n in names],type_ignores=[]),label,'exec'),ns)
def require(ok,msg):
    if not ok:raise ValueError(msg)
expected={'source_handoff.v2.json':'d510f3d80b1a207a4d2019176808177296277c4f30fb3987de7536cda44d9707','source_manifest.v1.json':'7a7ea02de0b2f51a9eaacba02dee5cdcb196f364df4aeae3f1f034af08895fbd','audit_manifest.v2.json':'185b82d6c7a0f19a60bd57c0d3cb75139413e10aef697548e55e8e8c007a73ee'}
for name,digest in expected.items():auth(dict(path=str((D/name).relative_to(ROOT)),sha256=digest))
m=doc(D/'source_manifest.v1.json')
for r in m['executed_source_refs']+m['source_authentication_refs']+doc(D/'audit_manifest.v2.json')['refs']+doc(D/'preserved_v1.v2.json')['upstream_refs']:auth(r)
need(len(m['source_authentication_refs'])==449,'all449 source authentication entries')
oldreview=ROOT/'research/reviews/C3_remaining_cells_GPU0_independent_20261017_v1/review.v1.json'
need(doc(oldreview)['verdict']=='SOURCE_BLOCKED','original independent v1 failure preserved')
new=nodes(D/'execution_v1.py');old=nodes(OLD/'execution_v1.py');contract=nodes(D/'contract_v1.py')
for r in m['executed_source_refs']:
    compile((ROOT/r['path']).read_bytes(),str(ROOT/r['path']),'exec');need(True,'compile inert source '+r['path'])
for name in ('worker_v1.py','direct_gpu0_v1.py'):
    need((D/name).read_text().replace(D.name,OLD.name)==(OLD/name).read_text(),'whole science and direct source unchanged '+name)
for name in ('main','worker_v1','rss_identity_v1','_RAMLog','_subreaper','_children'):
    normalized=ast.parse(ast.unparse(old[name]).replace(OLD.name,D.name)).body[0]
    need(ast.dump(new[name])==ast.dump(normalized),'unchanged entry/gate/RSS '+name)
for name,node in nodes(OLD/'contract_v1.py').items():
    if name=='new_stage_values_v1':continue
    normalized=ast.parse(ast.unparse(node).replace(OLD.name,D.name)).body[0]
    need(ast.dump(contract[name])==ast.dump(normalized),'unchanged old election/admission contract '+name)
keep={'charged_census_v1','g.budget_check_v9','g.validate_storage_v9','direct.capacity_v1','direct.check_attachment_v1','direct.attach_v1'}
def calls(node):return [ast.unparse(n)for n in ast.walk(node)if isinstance(n,ast.Call)and ast.unparse(n.func)in keep]
need(calls(new['execute_v1'])==calls(old['execute_v1']),'original scope storage capacity sampling preserved')
helpers=('_HistoryV8','_PendingV8','OwnedLifecycleV8','observe_groups_v8')
shared=nodes(ROOT/'research/reviews/H058_GPU0_full_training_successor_source_20261017_v8/driver_role.source.txt')
for n in helpers:need(ast.dump(new[n])==ast.dump(shared[n]),'exact shared H058v8 helper AST '+n)
backend_nodes=nodes(ROOT/'research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py')
limits=dict(pending=1024,discoveries=2**28,groups=65536,adopted_waits=2**28,history_rows=16)
def fixture(group_cap=65536):
    rows={1000:dict(pid=1000,ppid=100,group=1000,session=1000,state='S',start=1)};children={100:[1000],1000:[]};probes=[]
    def stat(p):return copy.deepcopy(rows.get(p))
    ns=dict(_stat=stat,_children=lambda p:children.get(p,()),os=SimpleNamespace(getpid=lambda:100))
    execute_defs(ns,backend_nodes,('_same','OwnedTree'),'<actual-original-backend-RAM>');backend=SimpleNamespace(**ns)
    def probe(group,sig):
        probes.append(group)
        if group!=2000:raise ProcessLookupError()
    h=dict(collections=collections,hashlib=hashlib,json=json,OWNED_LIMITS_V8=dict(limits,groups=group_cap),time=SimpleNamespace(monotonic=lambda:0),os=SimpleNamespace(killpg=probe))
    execute_defs(h,new,helpers,'<actual-C3v2-helper-RAM>')
    tree=h['OwnedLifecycleV8'](backend,1000)
    return rows,children,probes,h,tree
rows,children,probes,h,tree=fixture()
for i in range(8192):
    pid=3000+i%5;rows[pid]=dict(pid=pid,ppid=1000,group=1000,session=1000,state='S',start=10+i);children[1000]=[pid];tree.refresh()
    del rows[pid];children[1000]=[];tree.refresh()
need(tree.complete and len(tree.known)==1 and tree.discovered.total==8193,'v1 cumulative4096 blocker repaired for8192 absent helpers')
need(len(json.dumps(tree.evidence()))<16384,'bounded digest history remains small')
# Material blocker: same identity changes pgid, including when the group cap is full.
reproductions=[]
for cap in (65536,1):
    rows,children,probes,h,tree=fixture(cap)
    rows[2000]=dict(pid=2000,ppid=1000,group=1000,session=1000,state='S',start=2);children[1000]=[2000];tree.refresh()
    rows[2000]['group']=2000;tree.refresh()
    need(tree.known[2000]['group']==2000 and tree.complete and tree.groups=={1000},'REPRO current known group omitted and complete remains true cap'+str(cap))
    before=tree.evidence();rows.clear();children.clear();need(tree.identities_absent(),'actual current identities retired after RAM absence')
    scan=h['observe_groups_v8'](tree,30)
    need(scan['all_owned_groups_observed_absent'] and probes==[1000],'REPRO final group scan omits observed positive group2000 cap'+str(cap))
    own=tree.evidence()
    wait=dict(schema='C3_remaining_cells_actual_wait_v2',owned_roster_policy_ref=reference(D/'execution_v1.py'),owned_roster_policy=limits,ownership=own,owned_identity_absence=True,group_observations_summarized=True,group_absence_summary=scan,adopted_waits=h['_HistoryV8']().evidence(),group_observations=scan['history']['first']+scan['history']['last'])
    # The actual policy must match for the normal-cap reproduction; cap1 is only a bound-bypass fixture.
    if cap==65536:
        cn=dict(require=require,ref=reference,DRIVER=str((D/'execution_v1.py').relative_to(ROOT)),OWNED_LIMITS_V8=limits)
        execute_defs(cn,contract,('owned_drain_values_v2',),'<actual-C3v2-contract-RAM>')
        need(cn['owned_drain_values_v2'](wait)==wait,'REPRO actual new-cell contract accepts incomplete retained-group proof')
    reproductions.append(dict(group_cap=cap,known_row_before_retirement=before['current_owned_identities'],retained_groups=sorted(tree.groups),complete=tree.complete,actual_RAM_group_probes=probes,unprobed_group2000_exists_in_RAM=True,group_scan=scan))
# Check C3's actual read-only memo source join without importing or running a guard.
memo_path=ROOT/'research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py'
memo_nodes=nodes(memo_path);memo_tree=ast.parse(memo_path.read_text());memo_ns=dict(Path=Path,hashlib=hashlib,CodeType=CodeType,ROOT=ROOT,marshal=marshal,SOURCE=str(memo_path.relative_to(ROOT)))
for n in memo_tree.body:
    if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in ('MEMO_REF','V4_REF','V9_REF')for t in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),'<memo-pins>','exec'),memo_ns)
execute_defs(memo_ns,memo_nodes,('_source_v1','_original_parent_v1','_child_v1','install_consumer_v1'),'<actual-C3-memo-contract-RAM>')
def code_module(path,name,names):
    p=ROOT/path;mod=ModuleType(name);mod.__file__=str(p)
    compiled=compile(p.read_bytes(),str(p),'exec',dont_inherit=True,optimize=0)
    for n in names:
        code=next(c for c in compiled.co_consts if isinstance(c,CodeType)and c.co_name==n)
        setattr(mod,n,FunctionType(code,mod.__dict__,n))
    return mod
v4=code_module('src/arggnn/project_budget_v4.py','RAM_exact_original_v4',('scope_spec',))
v9=code_module('scripts/project_budget_v9.py','RAM_exact_original_v9',('successor_paths',));v9.v4=v4
class Handle:
    def __init__(self,*a,**k):self.namespace_callers=frozenset((object(),object()));self.receipt={}
    def install(self):return self
memo_ns['_load_pinned_memo_v1']=lambda:SimpleNamespace(TransactionMemoAdapter=Handle)
handle=memo_ns['install_consumer_v1'](v4,v9,consumer_role='read_only_consumer',expected_source_path=ROOT/memo_ns['V4_REF']['path'],expected_source_sha256=memo_ns['V4_REF']['sha256'])
need(len(handle.namespace_callers)==4,'C3 original namespace memo accepts actual pinned v4/v9 code-object join')
need(any(isinstance(n,ast.ImportFrom)and n.module=='scripts'and any(a.name=='project_budget_v9'and a.asname=='budget'for a in n.names)for n in ast.parse((ROOT/'research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py').read_text()).body),'actual Census imports exact original scripts.project_budget_v9')
result=dict(schema='C3_remaining_cells_independent_source_checks_v2',utc=datetime.now(timezone.utc).isoformat(),verdict='SOURCE_BLOCKED',checks=checks,positive_checks=len(checks),negative_cases=rejects,unique_authenticated_source_metadata_refs=len(authenticated),source_authentication_entries=449,churn_helpers=8192,group_transition_reproductions=reproductions,C3_original_v4_v9_memo_code_object_join_pass=True,actual_guard_state_resource_process_GPU_scope_or_numerical_calls=False,scientific_payload_reads=False,author_checkers_executed=False)
(OUT/'checks.result.v2.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ('checks','group_transition_reproductions')},sort_keys=True))
