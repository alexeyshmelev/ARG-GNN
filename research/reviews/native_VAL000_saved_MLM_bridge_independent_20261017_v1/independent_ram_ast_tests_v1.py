"""Independent stdlib-only source/AST and RAM checks. No numerical experiment."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys,types,io,pickle
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=Path(__file__).resolve().parent
CAND=ROOT/'research/reviews/native_VAL000_saved_MLM_acceptance_bridge_source_20261017_v1'
spec=importlib.util.spec_from_file_location('_independent_candidate_fixture',CAND/'ram_fixture_v1.py')
f=importlib.util.module_from_spec(spec);sys.modules[spec.name]=f;spec.loader.exec_module(f)
b=f.b
TESTS=[]
def test(name,fn):
 fn();TESTS.append({'name':name,'status':'PASS'})
def rejects(fn,error=Exception):
 try:fn()
 except error:return
 raise AssertionError('Expected rejection')
def source_ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def code_at(path,names):
 code=compile((ROOT/path).read_bytes(),str(ROOT/path),'exec',dont_inherit=True,optimize=0)
 for name in names:
  choices=[c for c in code.co_consts if isinstance(c,types.CodeType) and c.co_name==name]
  assert len(choices)==1;code=choices[0]
 return code
base='.runtime/author_sources/argformer_f7545be7bca0/'
JOINS=[(base+'sequence_contrastive.py',('train',)),(base+'sequence_contrastive.py',('build_contrastive_model',)),(base+'src/flex_bert.py',('create_flex_bert_classification',)),(base+'src/bert_layers/model.py',('FlexBertForSequenceClassification','from_composer'))]
for path,names in JOINS:
 def check(path=path,names=names):
  original=types.FunctionType(code_at(path,names),{'sentinel':object()})
  b._authenticate_function(original,path,names)
  clone=b._clone(original,{'private_replacement':object()})
  assert original.__code__ is clone.__code__ and original.__globals__ is not clone.__globals__
  assert 'private_replacement' not in original.__globals__
 test('exact source code authentication and private clone: '+'.'.join(names),check)

def ast_boundaries():
 tree=ast.parse((CAND/'saved_mlm_bridge_v1.py').read_text())
 allowed={'pathlib','dataclasses','hashlib','io','json','os','pickle','stat','types'}
 assert {n.module if isinstance(n,ast.ImportFrom) else a.name for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) for a in (n.names if isinstance(n,ast.Import) else [None])}<=allowed
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('exec','eval','__import__') for n in ast.walk(tree))
 train=code_at(base+'sequence_contrastive.py',('train',))
 assert train.co_names.index('build_contrastive_model')<train.co_names.index('build_optimizer')<train.co_names.index('Trainer')<train.co_names.index('fit')
 loader=code_at(base+'src/bert_layers/model.py',('FlexBertForSequenceClassification','from_composer'))
 assert loader.co_names.count('load')==1 and 'load_state_dict' in loader.co_names
 assert '_load_once_from_exact_FD' in (CAND/'saved_mlm_bridge_v1.py').read_text()
 old=(ROOT/'src/arggnn/american_c3_native_run_v1.py').read_text()
 assert 'for mode in ("mlm","contrastive")' in old
 assert 'checkpoint=fit["phases"]["contrastive"]["checkpoint_ref"]' in old
 assert 'for view in ("query_paths","contrastive")' in old and 'for split in ("train","val")' in old
 contract=(ROOT/'src/arggnn/american_c3_native_contract_v1.py').read_text()
 assert 'completed_C3_fresh_native_fit' in contract and 'for mode,epochs in (("mlm",5),("contrastive",1))' in contract
 assert 'completed_C3_fresh_native_fit' not in [n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
test('stdlib/no execution hooks, actual author order, unchanged legacy and four-loop source joins',ast_boundaries)

# Execute the genuine from_composer code object with tiny RAM stubs. In particular,
# its real prefix-removal mutation must not erase the bridge source snapshot.
def original_loader_ram():
 events=[];source,target=f.states();torch=f.Torch({'state':{'model':source}},events)
 def consume(state_dict,prefix):
  for k in list(state_dict):
   if k.startswith(prefix):state_dict[k[len(prefix):]]=state_dict.pop(k)
 class OriginalClass:
  def __init__(self,config,*args,**kwargs):self.state=copy.deepcopy(target)
  def state_dict(self):return dict(self.state)
  def load_state_dict(self,value,strict):
   assert strict is False
   missing=set(self.state)-set(value);unexpected=set(value)-set(self.state)
   for k in self.state:
    if k in value:self.state[k]=value[k]
   return sorted(missing),sorted(unexpected)
 env={'torch':torch,'consume_prefix_in_state_dict_if_present':consume,'logger':types.SimpleNamespace(warning=lambda *a:events.append('original_loader_warning'))}
 loader=types.FunctionType(code_at(base+'src/bert_layers/model.py',('FlexBertForSequenceClassification','from_composer')),env,'from_composer',(None,None,False,None))
 OriginalClass.from_composer=classmethod(loader)
 layers=types.SimpleNamespace(FlexBertForSequenceClassification=OriginalClass)
 flex_env={'bert_layers_module':layers,'types':types,'Tokenizer':f.Tokenizer,'vocab':f.reference_vocab()}
 exec("def create_flex_bert_classification(pretrained_checkpoint=None):\n return types.SimpleNamespace(model=bert_layers_module.FlexBertForSequenceClassification.from_composer(pretrained_checkpoint=pretrained_checkpoint,config=None),tokenizer=Tokenizer(vocab))\n",flex_env)
 flex=types.SimpleNamespace(bert_layers_module=layers,create_flex_bert_classification=flex_env['create_flex_bert_classification'])
 build_env={'flex_bert_module':flex}
 exec("def build_contrastive_model(cfg):\n return flex_bert_module.create_flex_bert_classification(pretrained_checkpoint=cfg['pretrained_checkpoint'])\n",build_env)
 train_env={'build_contrastive_model':build_env['build_contrastive_model'],'build_optimizer':lambda *a:events.append('optimizer'),'events':events}
 exec("def train(cfg):\n model=build_contrastive_model(cfg['model'])\n build_optimizer({},model)\n events.append('complete_original_call_placeholder')\n",train_env)
 checked,state=b._instrument_training(train_env['train'],lambda t,p,*a,**k:t.load(p),f.reference_vocab)
 checked({'model':{'pretrained_checkpoint':str(ROOT/b.CHECKPOINT_REF['path'])}})
 assert state['verified']['complete_encoder_key_count']==74 and state['checkpoint_loads']==1
 assert state['source'] is None and events[-2:]==['optimizer','complete_original_call_placeholder']
 assert OriginalClass.from_composer.__func__ is loader
 assert all(not k.startswith('model.') for k in source) # real original prefix handling ran
 assert train_env['build_contrastive_model'].__globals__['flex_bert_module'] is flex
test('real unchanged author loader code runs through private bridge including prefix mutation',original_loader_ram)

for kind in ('source_decoder_nan','target_classifier_nan','source_missing_core','non_tensor_source'):
 def check(kind=kind):
  source,target=f.states()
  if kind=='source_decoder_nan':source['model.decoder.weight']=f.Tensor((float('nan'),))
  if kind=='target_classifier_nan':target['classifier.weight']=f.Tensor((float('nan'),))
  if kind=='source_missing_core':source['model.decoder.extra']=source.pop('model.bert.parameter_0')
  if kind=='non_tensor_source':source['model.decoder.weight']=object()
  rejects(lambda:b._verify_transfer(source,f.Model(target),f.Torch()),(ValueError,TypeError))
 test('independent full tensor rejection: '+kind,check)

def error_propagation():
 checked,state,events,cls,loader=f.fake_training()
 def failed(path):raise OSError('RAM simulated read failure')
 original_torch=loader.__globals__['torch'];original_torch.load=failed
 rejects(lambda:checked({'model':{'pretrained_checkpoint':str(ROOT/b.CHECKPOINT_REF['path'])}}),OSError)
 assert 'original_optimizer_construction' not in events and state['verified'] is None
 assert cls.from_composer.__func__ is loader
test('original loader failure propagates without retry/fallback/optimizer or global edit',error_propagation)

def unsafe_pickle():
 rejects(lambda:b._PrimitiveUnpickler(io.BytesIO(b"cos\nsystem\n.")).load(),ValueError)
test('pinned vocabulary deserialization rejects global lookup without invoking it',unsafe_pickle)

# Fully rebind every changed document to avoid mistaking a stale SHA join failure
# for rejection of the actual scientific or terminal predicate.
def rebound_bundle(changes):
 docs=f.bundle();init=docs['initialization'].value;done=docs['contrastive_completion'].value
 wait=docs['contrastive_wait'].value;outer=docs['consumed_outer'].value
 values={'initialization':init,'contrastive_completion':done,'contrastive_wait':wait,'consumed_outer':outer}
 for role,key,value in changes:values[role][key]=value
 initdoc=f.document('independent_init',init);done['saved_MLM_initialization_ref']=initdoc.ref
 donedoc=f.document('independent_done',done);wait['child_completion_ref']=donedoc.ref
 waitdoc=f.document('independent_wait',wait);outer['actual_wait_ref']=waitdoc.ref;outer['child_completion_ref']=donedoc.ref
 return dict(initialization=initdoc,contrastive_completion=donedoc,contrastive_wait=waitdoc,consumed_outer=f.document('independent_outer',outer))
for role,key,value in [('contrastive_completion','actual_optimizer_updates',2134),('contrastive_completion','completed_epochs',0),('contrastive_completion','parent_id','other_parent'),('contrastive_completion','initializer_checkpoint_ref',{'path':'other.pt','sha256':'0'*64}),('contrastive_completion','uniform_class_weight_fallback_preserved',False),('contrastive_completion','training_config_plumbing_correction',True),('contrastive_completion','query_ancestry_truth_opened',True),('contrastive_wait','returncode',1),('contrastive_wait','process_group_ESRCH',False),('contrastive_wait','operational_failure','failed'),('consumed_outer','terminal_consumed',False),('consumed_outer','actual_tool_terminal_exit_code',1),('initialization','verified_before_optimizer_construction',False),('initialization','all_source_and_destination_floating_tensors_finite',False),('initialization','complete_encoder_keys_shapes_dtypes_and_values_equal',False),('contrastive_completion','checkpoint_ref',b.CHECKPOINT_REF)]:
 def check(role=role,key=key,value=value):
  rejects(lambda:b.qualify_for_native_embeddings_v1(f.eligibility,**rebound_bundle([(role,key,value)])),(ValueError,KeyError,TypeError))
 test('fully rebound downstream rejection: '+role+'.'+key,check)

def successful_bound_graph():
 result=b.qualify_for_native_embeddings_v1(f.eligibility,**rebound_bundle([]))
 assert result['historical_MLM_outer_exit_code']==1 and not result['completed_legacy_full_fit_wrapper_claimed']
 assert result['phases']['contrastive']['checkpoint_ref']!=b.CHECKPOINT_REF
 assert result['requires_all_four_original_single_split_embeddings'] and not result['launch_authority']
test('fully rebound genuine-shape RAM graph preserves separate saved-MLM qualification',successful_bound_graph)

def final_scope():
 assert 'torch' not in sys.modules and not any(k.startswith(('arggnn','composer','omegaconf')) for k in sys.modules)
 closure=json.loads((OUT/'source_closure.v1.json').read_text())
 for path,sha in closure['baseline_sha256'].items():assert source_ref(ROOT/path)['sha256']==sha,path
 fixture=json.loads((CAND/'fixture_inputs.v1.json').read_text())
 for role,pin in b.EVIDENCE_PINS.items():
  p=ROOT/pin['path'];assert p.stat().st_size<=1048576
  raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==pin['sha256']
  assert raw.decode()==fixture['raw_metadata_text_by_role'][role]
test('source closure unchanged and copied small evidence bytes match original receipts; no runtime import',final_scope)
result={'schema':'independent_saved_MLM_bridge_RAM_AST_checks_v1','status':'PASS_SOURCE_AND_RAM_ONLY','candidate_fixture_tests':f.run(),'independent_tests':TESTS,'independent_test_count':len(TESTS),'total_test_count':len(TESTS)+f.run()['count'],'source_ref':source_ref(CAND/'saved_mlm_bridge_v1.py'),'test_source_ref':source_ref(Path(__file__)),'real_checkpoint_opened_or_hashed':False,'real_vocabulary_scientific_data_or_truth_read':False,'installed_runtime_or_torch_imported':False,'guard_admission_process_GPU_or_Git_action':False,'source_activation':False}
with (OUT/'independent_ram_ast_results.v1.json').open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'status':result['status'],'independent_tests':len(TESTS),'total_tests':result['total_test_count'],'result_ref':source_ref(OUT/'independent_ram_ast_results.v1.json')},sort_keys=True))
