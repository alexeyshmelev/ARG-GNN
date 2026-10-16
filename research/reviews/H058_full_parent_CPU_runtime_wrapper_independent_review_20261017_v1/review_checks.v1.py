"""Combined H058 review: source plus focused handwritten metadata seams only."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast,builtins,contextlib,copy,hashlib,importlib,io,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');P='research/reviews/H058_full_parent_CPU_execution_source_20261017_v1';O='research/reviews/H058_full_parent_pilot_source_20261016_v2';W='research/reviews/H058_full_parent_CPU_runtime_wrapper_source_20261017_v1';R='research/reviews/H058_full_parent_CPU_runtime_wrapper_independent_review_20261017_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
def need(ok,msg='review check'):
 if not ok:raise ValueError(msg)
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def fns(text):return {n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def extract(path,name,ns):
 n=next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
 exec(compile(ast.Module(body=[n],type_ignores=[]),path,'exec'),ns);return ns[name]
def reject(fn):
 try:fn()
 except ValueError:return True
 raise AssertionError('expected metadata rejection')
def main():
 need(sys.dont_write_bytecode and sys.flags.no_user_site)
 expected={P+'/source_manifest.v1.json':'5263acef1aa73c527d7dbbbf674ec039d0c87c1c399711ba7bf87e3197a9cfa5',P+'/source_handoff.v1.json':'3550de5db0eaf89ace4a593ad87cb4d03a21ffc7a0339794436a3c9838b6bd68',W+'/source_manifest.v1.json':'72839331a5df51f0e7dfd884b80ab838b2b8324bd7c769b4d3de868771c392df',W+'/source_handoff.v1.json':'ba3cbac7dd9407cca56144c9c7c45606ec1b5b3e0653d293c454692289032d8d',W+'/root_driver_v1.py':'be95f7ac042836b8177064b167f87a1113eb1044a51c5be046e1785c0e0ac5a1',W+'/root_contract_v1.py':'6152cd938e29870dc4dc84eafb10c353ec4261864c66e0f29dbe995680850ccb',W+'/worker_entry_v1.py':'b89e1618d0aaa08d68b2d49a0b8d5956bedc1a2716f56c9888fb31198bd6350e'}
 for path,sha in expected.items():need(ref(path)['sha256']==sha,'candidate seal')
 pm=json.loads((ROOT/P/'source_manifest.v1.json').read_text());wm=json.loads((ROOT/W/'source_manifest.v1.json').read_text())
 for record in pm['refs']:need(ref(record['path'])==record,'producer source pin')
 for path,sha in wm['source_sha256'].items():need(ref(path)['sha256']==sha,'wrapper source pin')
 for root in (P,W):
  for path in (ROOT/root).glob('*.py'):compile(path.read_text(),str(path),'exec')
 unchanged_files=('__init__.py','model_v1.py','protocol_v1.py','matrix_v1.py','diagnostics_v1.py')
 for name in unchanged_files:need((ROOT/P/name).read_bytes()==(ROOT/O/name).read_bytes(),'unchanged scientific file')
 old=fns((ROOT/O/'producer_v1.py').read_text());new=fns((ROOT/P/'producer_v1.py').read_text())
 permitted={'runtime_flags_v1','validate_runtime_flags_v1','validate_context_v1','scheduled_step_v1'}
 for name in old.keys()-permitted:need(same(old[name],new[name]),'original numerical/control function '+name)
 scheduled=copy.deepcopy(new['scheduled_step_v1'])
 class StripCache(ast.NodeTransformer):
  def visit_Expr(self,n):
   if isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='validate_scheduled_cache_v1':return None
   return self.generic_visit(n)
  def visit_Assign(self,n):
   if isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='scheduled_CPU_receipt_v1':return None
   return self.generic_visit(n)
 need(same(old['scheduled_step_v1'],StripCache().visit(scheduled)),'scheduled numerical step unchanged')
 oo=fns((ROOT/O/'observations_v1.py').read_text());no=fns((ROOT/P/'observations_v1.py').read_text())
 for name in ('tensor_bytes','identity','null_for_parent'):need(same(oo[name],no[name]),'original observed-only function')
 prep=copy.deepcopy(no['_prepare_payload_once_v1']);prep.name='payload_v1';need(same(oo['payload_v1'],prep),'original packed/F1 construction')
 protocol=importlib.import_module(P.replace('/','.')+'.protocol_v1');TRAIN=protocol.TRAIN;ARMS=protocol.ARMS
 need(protocol.fixed_protocol_v1()['updates_all_new_arms']==1920 and protocol.fixed_protocol_v1()['updates_per_arm']==960)
 cfg=json.loads((ROOT/W/'scientific_binding.v1.json').read_text())['config_defaults']
 counts=json.loads((ROOT/cfg['source_count_ref']['path']).read_text())['parents'];rows=[counts[p]for p in TRAIN]
 forecast=dict(bank_bytes_upper=sum(v['CPU_GPU_original_bank_tensor_bytes_upper']for v in rows),episode_bytes_upper=sum(v['original_H045_episode_tensor_upper_bytes']for v in rows),packed_numeric_bytes_upper=2*sum(v['one_packed_layout_numeric_bytes']for v in rows),packed_tensor_objects_upper=2*sum(v['one_packed_layout_Tensor_objects']for v in rows),target_bytes=sum(v['native_TRAIN_target_FP32_bytes']for v in rows),Python_metadata_allowance_bytes=cfg['CPU_packed_python_allowance_bytes'])
 total=sum(forecast.values())-forecast['packed_tensor_objects_upper'];need(total==19628505574 and total<=32*2**30)
 # Actual pure count preflight and packed-cache class with no imports/tensors/builders.
 cache_ns=dict(json=json,TRAIN=TRAIN,VAL=protocol.VAL,ARMS=ARMS,require=need,stamp=lambda value:value.version)
 preflight=extract(P+'/cpu_cache_v1.py','cache_preflight_v1',cache_ns)
 fake_reader=NS(decode_json=lambda r:{'parents':counts})
 actual_forecast=preflight(dict(cfg=cfg,reader=fake_reader),TRAIN)
 need(actual_forecast['prepared_cache_bytes_upper_with_allowance']==total)
 Packed=extract(P+'/cpu_cache_v1.py','FrozenPackedCacheV1',cache_ns)
 banks={p:object()for p in TRAIN};episodes={p:object()for p in TRAIN}
 def validate(pid,bank,keys):need(bank is banks[pid]and keys==pid)
 observed=dict(banks=banks,keys={p:p for p in TRAIN},episodes=NS(values=episodes,builds=32,bytes=1024,validate_active_v1=validate))
 cache=Packed(dict(cfg=cfg),observed,TRAIN,actual_forecast)
 binding_calls=[]
 for pid in TRAIN:
  for arm in ARMS:
   route=NS(numeric_tensor_bytes=128,tensor_object_count=3,global_query_node=NS(version=0))
   binding=NS(validate=lambda bank,episode,route:binding_calls.append((id(bank),id(episode),id(route))))
   payload=NS(episodes=episodes[pid],route_cpu=route,binding=binding)
   cache.add_v1(pid,arm,payload,{'seconds':0.0})
 cache.seal_v1();pid=TRAIN[0];bank=banks[pid];episode=episodes[pid]
 payload,_=cache.get_v1(pid,ARMS[0],bank,episode);again,_=cache.get_v1(pid,ARMS[0],bank,episode)
 need(payload is again and len(cache.values)==64 and len(binding_calls)==64)
 cache.validate_scheduled_v1(bank,payload);payload.route_cpu.global_query_node.version=1;reject(lambda:cache.validate_scheduled_v1(bank,payload));payload.route_cpu.global_query_node.version=0
 reject(lambda:cache.get_v1(pid,ARMS[0],object(),episode))
 # Actual context factory with handwritten import/reader/guard/CPU-stat seams.
 events=[];scientific_ref={'path':'synthetic/scientific_config.json','sha256':'a'*64};gen_ref={'path':'synthetic/generation.json','sha256':'b'*64};reg_ref={'path':'synthetic/registration.json','sha256':'c'*64}
 original=dict(refs=dict(generation_config=gen_ref,registration=reg_ref),observation_entry_refs={p:{'path':p+'.json','sha256':'d'*64}for p in TRAIN+protocol.VAL},observation_cache_source_refs={})
 registry=[{'parent_id':p}for p in TRAIN+protocol.VAL];target={p:{'truth_npz':{'path':'data/truth/'+p+'.npz','sha256':'e'*64}}for p in TRAIN}
 def readmeta(record):
  if record==scientific_ref:return cfg
  if record==protocol.H050_INPUT_REF:return original
  if record==gen_ref:return {'parent_registry':registry}
  if record==reg_ref:return dict(config_ref=gen_ref,parent_registry=registry)
  if record==protocol.H050_TRAIN_BINDINGS_REF:return {'parents':target}
  if record==cfg['source_count_ref']:return {'parents':counts}
  raise AssertionError('unexpected context read')
 upper=dict(native_nodes=counts[TRAIN[0]]['native_nodes'],native_holds=counts[TRAIN[0]]['native_holds'],haplotypes=952,compact_payload_upper_bound_bytes=1,preparation_named_array_allowance_bytes=1)
 torch=NS(get_num_threads=lambda:32,get_num_interop_threads=lambda:1,get_default_dtype=lambda:'FP32',float32='FP32',cuda=NS(is_initialized=lambda:False),device=lambda x:NS(type=x))
 h45=NS(science=lambda:events.append('science_import_seam'),episode_upper_bounds_v1=lambda bank:upper)
 orig_import=builtins.__import__
 def imports(name,*args,**kwargs):
  if name=='torch':return torch
  if name=='arggnn':return NS(train_ordered_clade_episode_v1=h45)
  return orig_import(name,*args,**kwargs)
 factory_ns={k:getattr(protocol,k)for k in ('TRAIN','VAL','H050_INPUT_REF','H050_TRAIN_BINDINGS_REF','H050_VAL_BINDINGS_REF','OBSERVATION')}
 factory_ns.update(require=need,os=NS(environ={'CUDA_VISIBLE_DEVICES':''},sysconf=lambda key:4096),open=lambda *a:io.StringIO('1 1'),__builtins__=dict(vars(builtins),__import__=imports))
 factory=extract(P+'/context_factory_cpu_v1.py','build_context_v1',factory_ns)
 reader=NS(root=ROOT,read=lambda r:b'',decode_json=readmeta);writer=NS(save=lambda *a:None);guard=lambda:events.append('guard_seam')
 spec=dict(root_authorizer='root_agent',admitted=True,hypothesis_id='H058',phase='train',device='cpu',physical_gpu=None,max_cuda_reserved_bytes=0,refs=dict(config=scientific_ref))
 ctx=factory(spec,reader=reader,output=writer,guard=guard,worker_tmp_directory='synthetic/tmp')
 need(ctx['cfg']is cfg and ctx['reader']is reader and ctx['output']is writer and ctx['guard']is guard and set(ctx['bindings'])==set(TRAIN))
 ctx['before_episode_build'](TRAIN[0],object(),upper);reject(lambda:ctx['before_episode_build'](TRAIN[0],object(),dict(upper,native_nodes=-1)))
 # Actual inactive executable imports expose the blocker without any guard or worker call.
 c=importlib.import_module(W.replace('/','.')+'.root_contract_v1')
 d,w=[importlib.import_module(W.replace('/','.')+'.'+name)for name in ('root_driver_v1','worker_entry_v1')]
 wrong=dict(driver_contract=d.c.HERE,worker_contract=w.c.HERE,expected_contract=c.HERE)
 need(d.c is w.c and d.c is not c and d.HERE=='research/reviews/American_TreeLoom_matched_CPU_execution_source_20261017_v5','blocking actual namespace defect')
 for module in(d,w):
  with contextlib.redirect_stdout(io.StringIO()):need(module.main([])==0)
 # Same actual local serializer algebra; no numerical module imported.
 class Scalar:
  def __init__(self,v):self.v=v
  def item(self):return self.v
 class Array:
  def __init__(self,v):self.v=v
  def tolist(self):return self.v
 class Tensor:
  def __init__(self,v):self.v=v
  def detach(self):return self
  def cpu(self):return self
  def numpy(self):return Array(self.v)
 value={1:(Tensor([1.,2.]),Array([Scalar(3)])),'native':True}
 need(w.jsonable_runtime_v1(value,NS(Tensor=Tensor),NS(ndarray=Array,generic=Scalar))=={'1':[[1.,2.],[3]],'native':True})
 need(not any(n in sys.modules for n in ('torch','numpy','torch_geometric','arggnn.census_tsinfer_fit_v1','scripts.project_budget_v9','arggnn.project_budget_v4')))
 need(c._h._GUARD is None and c.SELECTOR_REF is None and d.c._h._GUARD is None)
 return dict(schema='H058_complete_CPU_independent_source_metadata_check_v1',status='SOURCE_BLOCKED_WRAPPER_ROUTING',producer_source_refs_authenticated=len(pm['refs']),wrapper_source_refs_authenticated=len(wm['source_sha256']),scientific_files_byte_identical=list(unchanged_files),all_original_producer_numerical_train_evaluate_checkpoint_selection_AST_unchanged=True,scheduled_step_unchanged_except_two_cache_checks_and_inline_receipt=True,original_null_and_packed_F1_builder_AST_preserved=True,
  complete32parent_64payload_handwritten_cache_seal_and_same_object_lookup=True,mutated_active_scatter_and_bank_rejected=True,actual_source_count_forecast_bytes=total,packed_numeric_bytes_upper=forecast['packed_numeric_bytes_upper'],packed_tensor_objects_upper=forecast['packed_tensor_objects_upper'],
  actual_context_factory_with_handwritten_runtime_and_reader_guard_stat_seams_passed=True,context_scientific_config_ref_original32TRAINbindings_writer_guard_identity_preserved=True,actual_inactive_driver_worker_contract_routing=wrong,blockers=['R1: both actual executable modules import H060v5 contract, not H058wrapper contract'],
  local_serializer_handwritten_nested_metadata=True,real_Torch_Numpy_models_payload_guards_mutex_admission_resource_process_probe_jobs_Git_install=False,actual_optimizer_updates=0,
  limits='No numerical arrays/model imports or actual resources/lifecycle. Handwritten cache/context seams establish control flow only, not original-data integrity, runtime fit, successful serialization or real wait/drain. Wrapper is blocked before any permission to admit or launch.')
if __name__=='__main__':
 value=main();(ROOT/R/'review_checks.v1.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');print(json.dumps(value,sort_keys=True,indent=2))
