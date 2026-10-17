"""Full original reservation schema/binding boundary in RAM only."""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
import ast,copy,json,hashlib
BASE=Path('research/reviews/frozen_benchmark_training_ready_20261017_v1')
V5=BASE/'train001_simulate_v5'
ROOT=PurePosixPath('/workspace-SR003.nfs2/ARG-GNN')
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def need(ok,msg):
    if not ok:raise ValueError(msg)
# Reconstruct exact old literal pins without importing their module.
old=Path('research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1/root_contract_v1.py')
ot=ast.parse(old.read_text());lit={}
for n in ot.body:
    if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in ['PIDFD','OWNED','NAMESPACE_REF','V4_REF']for t in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),'#literal-pins-only','exec'),lit)
future={}
def ph(name):
    r=dict(path=str(V5/'RAM_FUTURE')+'/'+name+'.json',sha256=hashlib.sha256(('RAM_ONLY:'+name).encode()).hexdigest());future[r['path']]=r;return r
c=SimpleNamespace(ref=ref,**{n:lit[n]for n in ['PIDFD','OWNED','NAMESPACE_REF','V4_REF']},
    POLICY_REF=ref('research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json'))
q=dict(caps_ref=ref(V5/'caps.proposed.inactive.v5.json'),runtime_ref=ref('research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json'),
    root_directive_ref=ph('directive'),actual_capacity_ref=ph('capacity'),source_review_ref=ph('review'))
src=V5/'worker_entry_v1.py';tree=ast.parse(src.read_text())
scope=dict(c=c,SOURCE=str(src),MANIFEST=str(V5/'source_manifest.v5.json'),SCIENCE=str(V5/'scientific_condition.v5.json'),OLD=str(old.parent),
 SELECTOR=ref('research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json'),
 CHILD=ref('research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='direct_roles')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'#exact-v5-direct-roles','exec'),scope)
direct=scope['direct_roles'](ph('request'),q,ph('config'),ph('closure'),q['runtime_ref'])
capture=Path('research/reviews/Census_observation_ROOT_activation_20261017_v1/actual_parent_capture.root_v1.json')
need(ref(capture)['sha256']=='58ee37959e97f2cb300a03ff7167fda10d7dbdd3a36a1dd947476f7e4cef0004','Exact immutable47role lineage')
roles=list(json.loads(capture.read_text())['all_actual_parent_bindings'].values())+[scope['SELECTOR']]
bindings=[]
for r in roles+direct:
    if r not in bindings:bindings.append(r)
need(len(roles)==47 and len(bindings)==61 and all(r in bindings for r in roles+direct),'Complete actual role union')
known={r['path']:r['sha256']for r in bindings}
# Pure paths deliberately avoid filesystem resolve/stat. All original helper
# predicates are retained and their actual IO dependency is an explicit RAM map.
class RAMPath(PurePosixPath):
    def resolve(self):return self
budget=Path('src/arggnn/project_budget_v4.py');bt=ast.parse(budget.read_text())
names={'integer','canonical','paths_overlap','accounting_paths','bound','scope_spec'}
nodes=[n for n in bt.body if isinstance(n,ast.FunctionDef)and n.name in names]
g=dict(Path=RAMPath,LEGACY='.runtime/gpup/storage_reservations.json',LOCK='.runtime/locks/scheduler.lock',
    file_bytes=lambda root,path:(b'EXPLICIT_RAM_BYTES',known[str(path)]),parse=json.loads)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(budget)+'#exact-full-scope-spec-RAM','exec'),g)
root=RAMPath(str(ROOT))
valid=dict(key='H041/TreeLoom-frozen-full30Mb-TRAIN001-simulate-root02',
 output=str(ROOT/'experiments/results/frozen_TreeLoom_full30Mb_TRAIN001_root02'),
 auxiliary_scopes=[str(ROOT/V5/'ROOT_actual_TRAIN001_root02'/'worker_aux')],
 reserved_bytes=7000000000,consumer_bindings=copy.deepcopy(bindings))
tests=[]
def test(name,value,want):
    accepted=True
    try:g['scope_spec'](root,value)
    except (ValueError,KeyError):accepted=False
    need(accepted==want,name);tests.append(dict(name=name,accepted=accepted,expected=want))
test('complete_original61binding_reservation_accepts',valid,True)
bad=copy.deepcopy(valid);bad['consumer_bindings'] += [dict(path=str(old.parent/'cpu_slot_0.lock'),sha256=hashlib.sha256(b'').hexdigest()),dict(path=str(old.parent/'cpu_slot_1.lock'),sha256=hashlib.sha256(b'').hexdigest())]
test('preserved_v4_full63binding_shape_rejects_lock_source_nodes',bad,False)
for name,change in [
 ('reject_npz_payload_binding',lambda v:v['consumer_bindings'].append(dict(path='RAM/not_source.npz',sha256='0'*64))),
 ('reject_noncanonical_scope',lambda v:v.update(output='experiments/results/relative')),
 ('reject_overlapping_output_AUX',lambda v:v.update(auxiliary_scopes=[v['output']+'/child'])),
 ('reject_accounting_overlap',lambda v:v.update(output=str(ROOT/'.runtime/storage/new'))),
 ('reject_zero_charge',lambda v:v.update(reserved_bytes=0)),
 ('reject_bool_charge',lambda v:v.update(reserved_bytes=True)),
 ('reject_unknown_field',lambda v:v.update(extra=True)),
 ('reject_wrong_SHA_shape',lambda v:v['consumer_bindings'][0].update(sha256='F'*64)),
 ('reject_drifted_bound_SHA',lambda v:v['consumer_bindings'][0].update(sha256='0'*64)),
 ('reject_65_consumer_refs',lambda v:v.update(consumer_bindings=[*v['consumer_bindings'],*[dict(path='RAM/x%d.json'%i,sha256='0'*64)for i in range(4)]]))]:
    bad=copy.deepcopy(valid);change(bad);test(name,bad,False)
condition=json.loads((V5/'scientific_condition.v5.json').read_text())
need(condition['synchronization_pool']['slot_paths']==[str(old.parent/'cpu_slot_0.lock'),str(old.parent/'cpu_slot_1.lock')],'Shared pool contract preserved')
need(not any(Path(r['path']).suffix=='.lock'for r in bindings),'No synchronization nodes falsely treated as source metadata')
need('snippet=snippet.replace' in src.read_text()and 'Original shared slot is not genuinely held'in src.read_text(),'Original shared slot/probe path retained')
result=dict(schema='frozen_TRAIN001_v5_original_scope_spec_RAM_v1',source_ref=ref(src),manifest_ref=ref(V5/'source_manifest.v5.json'),original_budget_source_ref=ref(budget),
 current_lineage_roles=47,current_unique_lineage_refs=len({r['path']for r in roles}),direct_stage_refs=len(direct),complete_unique_consumer_refs=len(bindings),
 original_scope_spec_and_helper_functions_AST_unmodified=True,stored=False,tests=tests,
 synchronization_pool_contract_preserved=True,guard_or_production_module_imported=False,
 pure_RAM_paths_and_explicit_file_bytes_stub=True,actual_fresh_filesystem_canonicality_or_byte_authentication_claimed=False,
 no_actual_state_probe_admission_or_execution=True)
out=V5/'scope_spec_RAM_checks.v5.json';out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'receipt_ref':ref(out),'cases':len(tests),'complete_consumer_refs':len(bindings),'all47_roles_preserved':True},indent=2))
