"""Independent targeted RAM/AST review; no installed runtime or payload reads."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,importlib.util,json,sys,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=Path(__file__).resolve().parent
CAND=ROOT/'research/reviews/native_VAL000_contrastive_saved_MLM_integration_source_20261017_v1'
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(p,name):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def fn(path,name):return copy.deepcopy(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name))
def dump(node):return ast.dump(node,include_attributes=False)
# Redirect only the candidate runner's result write in its in-RAM AST. No source edit.
tree=ast.parse((CAND/'ram_ast_checks_v1.py').read_text())
writes=[n for n in tree.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='write_text']
assert len(writes)==1
writes[0].value.func.value=ast.parse("OUT/'candidate_reproduced_ram_ast_results.v1.json'",mode='eval').body
ast.fix_missing_locations(tree);namespace={'__file__':str(CAND/'ram_ast_checks_v1.py'),'OUT':OUT,'__name__':'_candidate_runner_private_output_only'}
exec(compile(tree,'<unchanged candidate tests except own output destination>','exec'),namespace)
assert namespace['result']['tests']==117
m=load(CAND/'saved_mlm_integration_v1.py','_independent_saved_MLM_integration')
f=load(ROOT/Path(m.PINS['bridge']['path']).parent/'ram_fixture_v1.py','_independent_accepted_bridge_fixtures')
b=f.b;m._BRIDGE=b;b.read_eligibility_v1=lambda:f.eligibility
TESTS=[];FINDINGS=[]
def test(name,action):action();TESTS.append({'name':name,'status':'PASS'})
def rejects(action):
 try:action()
 except (ValueError,KeyError,TypeError,AttributeError):return
 raise AssertionError('Expected rejection')
def exact_clone():
 original=fn(ROOT/m.PINS['bridge']['path'],'run_original_contrastive_train_v1')
 anchor=ast.parse('expected.autoresume=False').body[0]
 idx=[i for i,n in enumerate(original.body)if dump(n)==dump(anchor)];assert len(idx)==1
 original.body.insert(idx[0]+1,ast.parse('expected.save_num_checkpoints_to_keep=1').body[0]);ast.fix_missing_locations(original)
 ns={};exec(compile(ast.Module(body=[original],type_ignores=[]),'<independent exact expected clone>','exec'),ns)
 actual=m.keep1_bridge_function_v1(b)
 assert b._code_shape(actual.__code__)==b._code_shape(ns['run_original_contrastive_train_v1'].__code__)
 assert actual.__globals__ is not b.__dict__
 for name in ('_instrument_training','_load_once_from_exact_FD','_runtime_reference_vocabulary','_authenticate_function'):
  assert actual.__globals__[name] is b.__dict__[name]
test('private keep1 clone differs by exactly one operational AST assignment and preserves all inline checks',exact_clone)

def binding():
 request={'storage_policy_selector_ref':m.PINS['current_selector'],'saved_MLM_integration_source_ref':m.own_ref_v1()}
 m.bind_stage_v1(dict(request),request)
 for role in request:
  bad=dict(request);bad[role]={'path':'RAM/wrong','sha256':'0'*64}
  rejects(lambda bad=bad:m.bind_stage_v1(bad,request))
  rejects(lambda bad=bad:m.bind_stage_v1(request,bad))
test('current selector and exact integration-source binds reject stale request or context',binding)

orig_bytes=m._bytes;byte_map={}
m._bytes=lambda r,metadata=False:byte_map[r['path']]if r['path']in byte_map else orig_bytes(r,metadata=metadata)
def document(name,value):
 doc=f.document('independent_'+name,value);byte_map[doc.ref['path']]=doc.json_bytes;return doc
accept={'schema':'ROOT_native_VAL000_saved_MLM_artifact_acceptance_v1','root_authorizer':'root_agent','saved_MLM_checkpoint_ref':b.CHECKPOINT_REF,'saved_MLM_eligibility':f.eligibility.record(),'historical_MLM_outer_exit_code':1,'contrastive_only_artifact_use_accepted':True,'MLM_replay_authorized':False,'launch_or_admission_authority':False}
def accept_case(value):
 doc=document('RAM_NOT_ACTUAL_acceptance',value);r={'ROOT_saved_MLM_artifact_acceptance_ref':doc.ref}
 return m.saved_acceptance_v1(r,r)
test('RAM explicit artifact descriptor preserves historical1 and no admission authority',lambda:accept_case(accept))
for key,value in [('historical_MLM_outer_exit_code',0),('MLM_replay_authorized',True),('launch_or_admission_authority',True),('contrastive_only_artifact_use_accepted',False),('saved_MLM_checkpoint_ref',{'path':'RAM/other.pt','sha256':'0'*64})]:
 test('artifact descriptor rejects '+key,lambda key=key,value=value:rejects(lambda:accept_case(dict(accept,**{key:value}))))

f.initialization_record.update(original_saved_MLM_bridge_source_ref=m.PINS['bridge'],keep1_compatibility_source_ref=m.own_ref_v1(),operational_checkpoint_retention={'original':-1,'actual':1,'save_interval':'20ba'},all_scientific_configuration_fields_unchanged=True,no_extra_checkpoint_load_or_forward_or_optimizer_probe=True)
def bundle(changes=()):
 initial=f.bundle();values={k:v.value for k,v in initial.items()}
 for role,key,value in changes:values[role][key]=value
 init=document('init',values['initialization']);values['contrastive_completion']['saved_MLM_initialization_ref']=init.ref
 done=document('done',values['contrastive_completion']);values['contrastive_wait']['child_completion_ref']=done.ref
 wait=document('wait',values['contrastive_wait']);values['consumed_outer'].update(actual_wait_ref=wait.ref,child_completion_ref=done.ref)
 outer=document('outer',values['consumed_outer'])
 return dict(initialization_ref=init.ref,contrastive_completion_ref=done.ref,contrastive_wait_ref=wait.ref,consumed_outer_ref=outer.ref)
valid_refs=bundle();qualified=m.qualify_saved_MLM_for_embeddings_v1(**valid_refs)
qualified_doc=document('qualification',qualified)
spec={'parent_id':b.PARENT,'refs':{'saved_MLM_qualification':qualified_doc.ref,'fit_completion':qualified_doc.ref,'preparation':b.PREPARATION_REF,'counts':b.COUNTS_REF}}
test('fully bound separate qualification revalidates complete new contrastive graph',lambda: m.fit_completion_dispatch_v1(qualified_doc.ref,spec))
for role,key,value in [('initialization','operational_checkpoint_retention',{'original':-1,'actual':-1,'save_interval':'20ba'}),('initialization','original_saved_MLM_bridge_source_ref',{'path':'RAM/other.py','sha256':'0'*64}),('initialization','complete_encoder_keys_shapes_dtypes_and_values_equal',False),('contrastive_completion','actual_optimizer_updates',2134),('contrastive_completion','completed_epochs',0),('contrastive_completion','checkpoint_ref',b.CHECKPOINT_REF),('contrastive_wait','actual_direct_child_wait',False),('contrastive_wait','process_group_ESRCH',False),('contrastive_wait','operational_failure','RAM failure'),('consumed_outer','terminal_consumed',False),('consumed_outer','actual_tool_terminal_exit_code',1)]:
 test('fully rebound qualification rejects '+role+'.'+key,lambda role=role,key=key,value=value:rejects(lambda:m.qualify_saved_MLM_for_embeddings_v1(**bundle([(role,key,value)]))))
# Restore the valid graph after adversarial same-name RAM document maps.
valid_refs=bundle();qualified=m.qualify_saved_MLM_for_embeddings_v1(**valid_refs);qualified_doc=document('qualification',qualified)
spec['refs'].update(saved_MLM_qualification=qualified_doc.ref,fit_completion=qualified_doc.ref)

# Check actual embedding gate is before the native callback, with no guard/runtime.
stages=m.load_source_v1('stages');events=[]
stages._context_v1=lambda *a,**k:Path('/RAM_ONLY')
class ReachedNative(Exception):pass
def no_native(ctx):events.append('native_boundary');raise ReachedNative('RAM sentinel only')
stages._native_v1=no_native
orig_load=m.load_source_v1
m.load_source_v1=lambda role:stages if role=='stages'else orig_load(role)
ctx={'storage_policy_selector_ref':m.PINS['current_selector'],'saved_MLM_integration_source_ref':m.own_ref_v1(),'saved_MLM_qualification_spec':spec}
request={'storage_policy_selector_ref':m.PINS['current_selector'],'saved_MLM_integration_source_ref':m.own_ref_v1(),'saved_MLM_qualification_ref':qualified_doc.ref,'view':'query_paths','split':'train'}
def embedding_valid():
 try:m.embedding_v1(ctx,request)
 except ReachedNative:pass
 else:raise AssertionError('Expected RAM native boundary')
 assert events==['native_boundary']
test('actual separate embedding dispatch reaches native boundary only after real qualifier',embedding_valid)
def embedding_invalid():
 events.clear();old=byte_map[valid_refs['consumed_outer_ref']['path']]
 byte_map[valid_refs['consumed_outer_ref']['path']]=b._encoded({'schema':'RAM wrong'})
 try:rejects(lambda:m.embedding_v1(ctx,request));assert events==[]
 finally:byte_map[valid_refs['consumed_outer_ref']['path']]=old
test('actual embedding route rejects corrupt outer bytes before native boundary',embedding_invalid)
m.load_source_v1=orig_load

# Third production join: exact delegated IO source, real original legacy metadata
# prerequisite, all execution/runtime/guard operations replaced by RAM sentinels.
io_path=ROOT/m.PINS['io']['path'];stage_path=ROOT/m.PINS['stages']['path']
legacy_ns={'REFS':stages.REFS,'require':m.require}
for name in ('full_mlm_metadata_prerequisite_v1','full_contrastive_metadata_prerequisite_v1'):
 node=fn(stage_path,name);exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact legacy prerequisite RAM>','exec'),legacy_ns)
legacy=SimpleNamespace(REFS=stages.REFS,full_mlm_metadata_prerequisite_v1=legacy_ns['full_mlm_metadata_prerequisite_v1'],full_contrastive_metadata_prerequisite_v1=legacy_ns['full_contrastive_metadata_prerequisite_v1'])
retrieval_events=[]
def context_stub(c,r,phase,GPU):
 assert callable(c['validated_VAL000_fit_completion_v1']);retrieval_events.append('context_has_new_qualifier');return Path('/RAM_ONLY')
legacy._context_v1=context_stub;legacy._native_v1=lambda c:retrieval_events.append('UNREACHABLE_native')
io_ns={'require':m.require,'_policy_v1':lambda *a:{'resource_failstop_proposal':{'maximum_single_embedding_input_bytes':123}},'_base_v1':lambda:legacy}
node=fn(io_path,'RAM_retrieval_and_seal_v1');exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact delegated RAM retrieval source>','exec'),io_ns)
io_fake=SimpleNamespace(RAM_retrieval_and_seal_v1=io_ns['RAM_retrieval_and_seal_v1'])
m.load_source_v1=lambda role:io_fake if role=='io'else orig_load(role)
truthful_old={k:json.loads(raw)for k,raw in f.blobs.items()}
legacy_metadata=dict(completion=truthful_old['completed'],wait=truthful_old['wait'],outer={'actual_exit_code':1,'actual_terminal_consumed':True,'actual_wait_ref':b.EVIDENCE_PINS['wait']},completion_ref=b.EVIDENCE_PINS['completed'],wait_ref=b.EVIDENCE_PINS['wait'],checkpoint_ref=b.CHECKPOINT_REF)
rctx=dict(ctx,MLM_prerequisite_metadata=legacy_metadata)
rrequest={k:ctx[k]for k in ('storage_policy_selector_ref','saved_MLM_integration_source_ref')}
rconfig={'max_embedding_input_bytes':123}
def reproduce_block():
 retrieval_events.clear()
 try:m.RAM_retrieval_and_seal_v1(rctx,rrequest,spec,rconfig)
 except ValueError as e:
  assert str(e)=='Actual MLM worker0, owned drain and separately consumed genuine outer0'
  assert retrieval_events==['context_has_new_qualifier']
  FINDINGS.append({'id':'B1','status':'REPRODUCED_RAM_SOURCE_PATH','error_type':'ValueError','error':str(e),'new_qualification_accepted_before_error':True,'legacy_MLM_outer1_preserved':True,'native_concat_retrieval_or_runtime_reached':False})
 else:raise AssertionError('Expected exact saved-MLM retrieval blocker')
test('B1 reproduction: valid new qualification still fails delegated historical MLM outer0 gate',reproduce_block)
def reproduce_missing_old():
 retrieval_events.clear()
 try:m.RAM_retrieval_and_seal_v1(ctx,rrequest,spec,rconfig)
 except KeyError as e:assert e.args==('MLM_prerequisite_metadata',)
 else:raise AssertionError('Expected obsolete context field dependency')
 assert retrieval_events==['context_has_new_qualifier']
test('B1 reproduction: new saved-artifact-only context still requires obsolete MLM metadata slot',reproduce_missing_old)
m.load_source_v1=orig_load;m._bytes=orig_bytes

def final_integrity():
 closure=json.loads((OUT/'source_closure.v1.json').read_text())
 for path,sha in closure['verified_source_metadata_pins'].items():assert ref(ROOT/path)['sha256']==sha,path
 assert 'torch' not in sys.modules and not any(k.startswith(('arggnn','composer','omegaconf'))for k in sys.modules)
test('all pinned sources unchanged and no Torch/guard/runtime module import',final_integrity)
result={'schema':'independent_saved_MLM_contrastive_integration_RAM_AST_v1','status':'BLOCKING_SOURCE_FINDING_REPRODUCED','candidate_checks_reproduced':117,'independent_checks':len(TESTS),'total_checks':117+len(TESTS),'tests':TESTS,'findings':FINDINGS,'candidate_handoff_ref':ref(CAND/'source_handoff.v1.json'),'test_source_ref':ref(Path(__file__)),'real_checkpoint_data_truth_runtime_Torch_guard_process_GPU_Git_action':False}
with (OUT/'independent_ram_ast_results.v1.json').open('x')as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'status':result['status'],'candidate_checks':117,'independent_checks':len(TESTS),'total_checks':result['total_checks'],'findings':FINDINGS,'result_ref':ref(OUT/'independent_ram_ast_results.v1.json')},sort_keys=True))
