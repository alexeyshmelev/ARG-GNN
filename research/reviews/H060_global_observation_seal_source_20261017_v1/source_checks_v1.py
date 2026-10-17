"""Tiny handwritten metadata/AST fixtures only; no project/scientific imports or payloads."""
from pathlib import Path
from types import SimpleNamespace,ModuleType,FunctionType,CodeType
import ast,copy,hashlib,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D=ROOT/'research/reviews/H060_global_observation_seal_source_20261017_v1'
checks=[];negative=[]
def need(ok,label):
    if not ok:raise AssertionError(label)
    checks.append(label)
def reject(fn,label):
    try:fn()
    except(ValueError,KeyError,TypeError,AssertionError):negative.append(label);return
    raise AssertionError('Invalid accepted: '+label)
def ref(path):
    p=Path(path);p=p if p.is_absolute()else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
source=(D/'global_observation_protocol_v1.py').read_text();ns={'__name__':'handwritten_typed_protocol'}
exec(compile(source,'<inert-component-only>','exec'),ns)
for r in ns['STATIC_SOURCE_REFS']:need(ref(r['path'])==r,'Original source pin exact '+r['path'])
rows=sorted([*ns['STATIC_SOURCE_REFS'],ref(D/'global_observation_protocol_v1.py')],key=lambda r:r['path'])
parents=ns['TRAIN'];plans={};seals={};planrefs={};cache={}
for parent in parents:
    def r(name):return dict(path='RAM/'+parent+'/'+name,sha256=hashlib.sha256((parent+name).encode()).hexdigest())
    pr=r('plan.json');planrefs[parent]=pr
    plan=dict(schema='american_c3_whole_tree_block_index_v1',parent_id=parent,plan_ref=pr,observation_ref=r('observation.json'),
        focal_registry_ref=r('focals.npz'),sequence_length_bp=2000000,class_order=list(ns['CLASSES']),haplotypes=1600,
        original_reference_counts=[200,200,200],effective_reference_counts=[200,200,200],focal_records=10000,
        all_trees_and_samples_retained=True,targets_opened=False,trees=2,sites=2,
        blocks=[dict(start_tree=0,stop_tree=1,nodes=1600,sites=1),dict(start_tree=1,stop_tree=2,nodes=1600,sites=1)],
        config=dict(native_preparation_ref=r('native.json'),class_order=list(ns['CLASSES']),expected_haplotypes=1600,
            input_reference_class_mask=None,input_reference_sample_indices=None,allow_missing=False,donor_age_split=350.,
            max_trees_per_block=32,max_nodes_per_block=10000,max_sites_per_block=4096))
    # Two synthetic holds/sites only; focal counters are handwritten metadata, never biological payloads.
    support=ns['verify_observed_plan_support_v1'](plan,tree_rows=[(0,0.,1000000.,1600),(1,1000000.,2000000.,1600)],
        site_positions=[0.,1000000.],focal_holds=[0,1]*5000,focal_sample_columns=[600]*10000,
        registry_sample_columns=[600]*10000,resource_check=lambda event:None)
    identity=dict(tensor_sha256=hashlib.sha256(('SYNTHETIC '+parent).encode()).hexdigest(),tensor_bytes=1,
        focal_identity_sha256=hashlib.sha256(('SYNTHETIC FOCALS '+parent).encode()).hexdigest(),trees=2,sites=2,
        all_global_structural_recipes_before_truth=True)
    metadata=dict(schema='American_TreeLoom_full_observation_preparation_v1',parent_id=parent,split='train',observation_only=True,
        observation_ref=plan['observation_ref'],focal_registry_ref=plan['focal_registry_ref'],native_preparation_ref=plan['config']['native_preparation_ref'],
        focal_identity_sha256=identity['focal_identity_sha256'],focal_records=10000,all_native_trees_nodes_sites_samples_retained=True,
        structural_recipes_global_before_partition=True,artificial_block_boundaries=False,private_truth_opened=False,model_executed=False)
    seals[parent]=ns['make_global_seal_v1'](plan,plan_ref=pr,preparation_metadata=metadata,global_identity=identity,plan_support=support,source_refs=rows)
    plans[parent]=plan;cache[parent]=copy.deepcopy(identity)
barrier=ns['make_all20_barrier_v1'](planrefs,seals,validated_cache_identities=cache)
need(ns['validate_barrier_v1'](barrier)==barrier,'All20 typed seals and cache identities form a complete observation-only barrier')
need(not any(k in seals[parents[0]]for k in ('all_blocks_sealed_before_truth','block_feature_sha256','legacy_block_seal_sha256')),'No fabricated legacy seal fields')
parent=parents[0];plan=plans[parent];seal=seals[parent]
# Verify source derivation restores every target parsing/indexing/identity statement exactly.
target_tree=ast.parse((ROOT/'src/arggnn/american_c3_gnn_targets_v1.py').read_text())
cls=next(n for n in target_tree.body if isinstance(n,ast.ClassDef)and n.name=='TrainTargetJoinV1')
original=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='join')
derived=ns['derive_base_join_v1'](original);restored=copy.deepcopy(derived);restored.body[2]=copy.deepcopy(original.body[2])
need(ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False),'Entire original truth validation/half-open indexing body exact after one observation-gate restoration')
hook_tree=ast.parse((ROOT/'research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/execution_hooks_prefetch_v1.py').read_text())
factory=next(n for n in hook_tree.body if isinstance(n,ast.FunctionDef)and n.name=='make_actual_target_joiner_factory_v1')
actual_class=next(n for n in factory.body if isinstance(n,ast.ClassDef)and n.name=='ActualTrainTargetJoinV1')
actual_join=next(n for n in actual_class.body if isinstance(n,ast.FunctionDef)and n.name=='join')
ad=ns['derive_actual_join_v1'](actual_join);back=copy.deepcopy(ad)
for c in ast.walk(back):
    if isinstance(c,ast.Call)and isinstance(c.func,ast.Name)and c.func.id=='_global_base_join':
        need(c.args.pop(0).id=='self','Only explicit self delegation inserted')
        c.func=ast.parse('super().join',mode='eval').body
need(ast.dump(back,include_attributes=False)==ast.dump(actual_join,include_attributes=False),'Actual path gate and binding receipt exact after one delegation restoration')
# Exercise the two actual derived joins with only tiny metadata doubles; resolver is a sentinel.
class ReachedTruth(Exception):pass
calls=[];registry_mode=['exact']
class FocalCount:
    def __len__(self):return 10000
records=dict(parent_id=SimpleNamespace(item=lambda:parent),class_order=SimpleNamespace(tolist=lambda:list(ns['CLASSES'])),record_ids=FocalCount())
def require(ok,msg):
    if not ok:raise ValueError(msg)
def registry(r):
    require(r==plan['focal_registry_ref'],'same frozen focal registry ref')
    return records
base_scope=dict(require=require,registry=registry,identity_sha=lambda records:seal['focal_identity_sha256']if registry_mode[0]=='exact'else'b'*64,
    CLASSES=ns['CLASSES'],_global_observation_gate=ns['global_observation_gate_v1'])
exec(compile(ast.Module(body=[derived],type_ignores=[]),'<actual-target-body-with-RAM-readers>','exec'),base_scope)
actual_scope=dict(require=require,TRAIN=parents,metadata={parent:dict(binding=dict(inferred_observation_ref=plan['observation_ref']))},_global_base_join=base_scope['join'])
exec(compile(ast.Module(body=[ad],type_ignores=[]),'<actual-path-delegation-with-RAM-readers>','exec'),actual_scope)
def resolver(p):calls.append(p);raise ReachedTruth()
obj=SimpleNamespace(rows={p:{}for p in parents},resolve=resolver,_global_barrier=copy.deepcopy(barrier))
try:actual_scope['join'](obj,parent,split='train',plan=plan,observation_seal=seal)
except ReachedTruth:pass
else:raise AssertionError('Valid exact protocol must reach resolver sentinel')
need(calls==[parent],'Valid new global protocol reaches the original resolver only after complete source/support/focal gates')
cases=[('missing parent',lambda b,s,p:b['parents'].pop(parents[-1])),
 ('cache mismatch',lambda b,s,p:b['validated_cache_identities'][parents[-1]].update(tensor_sha256='b'*64)),
 ('legacy schema',lambda b,s,p:s.update(schema='american_c3_observation_blocks_sealed_v1')),
 ('wrong plan ref',lambda b,s,p:p.update(plan_ref=dict(path='RAM/other.json',sha256='b'*64))),
 ('wrong inferred observation',lambda b,s,p:p.update(observation_ref=dict(path='RAM/other.json',sha256='b'*64))),
 ('wrong native focals',lambda b,s,p:p.update(focal_registry_ref=dict(path='RAM/other.npz',sha256='b'*64))),
 ('reduced sample support',lambda b,s,p:p.update(haplotypes=1000)),
 ('reduced focal support',lambda b,s,p:p.update(focal_records=9999)),
 ('altered focal identity',lambda b,s,p:s.update(focal_identity_sha256='b'*64)),
 ('truth opened',lambda b,s,p:b.update(truth_resolver_calls_before_barrier=1)),
 ('legacy claim',lambda b,s,p:s.update(legacy_block_seal_claimed=True)),
 ('global source mismatch',lambda b,s,p:b['parents'][parents[-1]]['source_refs'][0].update(sha256='b'*64)),
 ('plan node count drift',lambda b,s,p:p['blocks'][0].update(nodes=1601))]
for label,mutate in cases:
    b=copy.deepcopy(barrier);s=copy.deepcopy(seal);p=copy.deepcopy(plan);mutate(b,s,p);obj._global_barrier=b;calls.clear()
    reject(lambda:actual_scope['join'](obj,parent,split='train',plan=p,observation_seal=s),'No truth on '+label)
    need(not calls,'Resolver remains unopened '+label)
obj._global_barrier=barrier;registry_mode[0]='mismatch';calls.clear()
reject(lambda:actual_scope['join'](obj,parent,split='train',plan=plan,observation_seal=seal),'Original focal identity gate remains before truth')
need(not calls,'Original focal mismatch never resolves truth');registry_mode[0]='exact'
for split,pid in (('val','american_c3_val_000'),('test','american_c3_test_000'),('train','american_c3_train_000')):
    calls.clear();reject(lambda:actual_scope['join'](obj,pid,split=split,plan=plan,observation_seal=seal),'Held-out/old parent denied '+pid);need(not calls,'No held-out resolver call '+pid)
# Metadata support witness rejects count/coverage drift, without any encoder or model.
def support_case(**changes):
    x=dict(tree_rows=[(0,0.,1000000.,1600),(1,1000000.,2000000.,1600)],site_positions=[0.,1000000.],focal_holds=[0,1]*5000,
        focal_sample_columns=[600]*10000,registry_sample_columns=[600]*10000,resource_check=lambda event:None);x.update(changes)
    return ns['verify_observed_plan_support_v1'](plan,**x)
for label,change in [('missing tree',dict(tree_rows=[(0,0.,1000000.,1600)])),('extra tree',dict(tree_rows=[(0,0.,1000000.,1600),(1,1000000.,2000000.,1600),(2,2000000.,3000000.,1600)])),('node count',dict(tree_rows=[(0,0.,1000000.,1599),(1,1000000.,2000000.,1600)])),('site count',dict(site_positions=[0.])),('focal hold',dict(focal_holds=[2]*10000)),('query column',dict(focal_sample_columns=[599]*10000)),('sample identity',dict(registry_sample_columns=[601]*10000))]:reject(lambda:support_case(**change),'Observed support witness rejects '+label)
# Global digest body remains the same ordered hash computation; only lazy imports/require alias are added.
old=ast.parse((ROOT/'research/reviews/H060_legacy_fast_full_training_source_20261017_v1/parent_startup_legacy_fast_v1.py').read_text())
g=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='fast_global_identity_v1')
gnew=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='global_identity_v1');norm=copy.deepcopy(gnew);norm.name=g.name;del norm.body[1:3]
need(ast.dump(norm,include_attributes=False)==ast.dump(g,include_attributes=False),'Original ordered global tensor digest body unchanged')
# Exact source-derived callable identity join, using code objects only (never importing target modules).
def code_method(record,path):
    raw=(ROOT/record['path']).read_bytes();code=compile(raw,str(ROOT/record['path']),'exec',dont_inherit=True,optimize=0)
    for name in path:code=next(c for c in code.co_consts if isinstance(c,CodeType)and c.co_name==name)
    module=ModuleType('RAM_exact_method_'+str(len(path)));module.__file__=str(ROOT/record['path'])
    def cell(v):return(lambda:v).__closure__[0]
    fn=FunctionType(code,module.__dict__,closure=tuple(cell(None)for _ in code.co_freevars)or None)
    return module,fn
for record,path in ((ref('src/arggnn/american_c3_gnn_targets_v1.py'),('TrainTargetJoinV1','join')),
    (ref('research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/execution_hooks_prefetch_v1.py'),('make_actual_target_joiner_factory_v1','ActualTrainTargetJoinV1','join'))):
    module,fn=code_method(record,path)
    need(ns['_exact_method_v1'](module,record,fn,path).name=='join','Exact pinned target callable/code-object identity '+str(len(path)))
    fn.__wrapped__=fn
    reject(lambda:ns['_exact_method_v1'](module,record,fn,path),'Wrapped original target method denied '+str(len(path)))
# Explicit new target binding preserves old path/equation lineage without claiming old observation/checkpoint identity.
original_binding=dict(schema='C3_actual_TRAIN_simulation_binding_operational_v1',registry_ref=ns['REGISTRY_REF'],target_equations_unchanged=True,parents={'SYNTHETIC':'original metadata only'})
binding=dict(schema=ns['TARGET_SCHEMA'],registry_ref=ns['REGISTRY_REF'],original_actual_target_binding=copy.deepcopy(original_binding),
    component_source_ref=ref(D/'global_observation_protocol_v1.py'),source_refs=rows,
    global_observation_barrier_sha256=ns['json_sha'](barrier),global_seal_schema=ns['SEAL_SCHEMA'],
    target_equations_unchanged=True,actual_target_paths_unchanged=True,legacy_block_seal_fabricated=False,
    original_observation_protocol_or_checkpoint_identity_claimed=False)
need(ns['validate_target_protocol_binding_v1'](binding,original_binding,barrier=barrier)==binding,'Explicit new observation protocol preserves original actual target binding')
for field,value in (('target_equations_unchanged',False),('actual_target_paths_unchanged',False),('legacy_block_seal_fabricated',True),('original_observation_protocol_or_checkpoint_identity_claimed',True),('global_observation_barrier_sha256','b'*64)):
    bad=copy.deepcopy(binding);bad[field]=value
    reject(lambda:ns['validate_target_protocol_binding_v1'](bad,original_binding,barrier=barrier),'Protocol binding rejects '+field)
badold=copy.deepcopy(original_binding);badold['parents']['SYNTHETIC']='substituted data lineage'
reject(lambda:ns['validate_target_protocol_binding_v1'](binding,badold,barrier=barrier),'Original actual target data lineage substitution denied')
prep=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='prepare_and_seal_global_parent_v1')
prep_calls=[ast.unparse(n.func)for n in ast.walk(prep)if isinstance(n,ast.Call)]
need(prep_calls.count('prepare_from_loaded_observation_v1')==1 and prep_calls.count('global_identity_v1')==1
    and not any('provider' in c.lower()or 'encode_block' in c or c.endswith('.seal')for c in prep_calls),'One original global preparation/hash; no legacy provider or block encoding call')
need(not any(n in sys.modules for n in ('numpy','torch','torch_geometric','tskit')),'No scientific package imports or numerical runtime')
result=dict(schema='H060_global_observation_source_RAM_checks_v1',status='AUTHOR_SOURCE_ONLY_RAM_PASS',positive_count=len(checks),negative_count=len(negative),checks=checks,negative_cases=negative,
    synthetic_fixture='two handwritten holds/sites, abstract10000 focal counters/indices, no neural tensors or true datasets',
    exact_target_equation_AST_restoration=True,exact_actual_path_AST_restoration=True,original_global_digest_body_unchanged=True,
    actual_preparation_target_truth_scientific_payload_numerical_GPU_process_or_accounting_operations=False)
print(json.dumps(result,sort_keys=True,indent=2))
