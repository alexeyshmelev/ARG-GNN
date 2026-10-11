"""Fresh bounded independent v3 source/fake-guard and small full-reference checks."""
from __future__ import annotations
import ast,copy,hashlib,importlib.machinery,importlib.util,json,os,pickle,sys,types
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
REVIEW=ROOT/'research/reviews/native_argformer_tsinfer_fit_v3_independent'
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONPATH']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load('_independent_native_v3',ROOT/'src/arggnn/native_argformer_tsinfer_fit_v3.py');c=m.policy()
checks=[];details={};counterexamples=[]
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
def reject(name,call,contains=None):
 try:call()
 except (ValueError,RuntimeError,TypeError,KeyError,FileNotFoundError) as error:
  if contains is not None:assert contains in str(error),(name,str(error))
  checks.append(name);return {'type':type(error).__name__,'message':str(error)}
 raise AssertionError('unexpected acceptance: '+name)
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
 return m.reference(path)
expected={
 'src/arggnn/native_argformer_tsinfer_fit_v3.py':'e9d681be39f3038e6c5258e22554fa65bc3554d0d036260f3e6cea6961b4a449',
 'configs/native_argformer_tsinfer_fit_v3.json':'656650f5bb44465fe8964fa8803e53004416973e82c22fe97d1c366eb74855cd',
 'research/reviews/native_argformer_tsinfer_fit_v3_author/source_receipt.json':'b10be6e3948bd3b4110e309b7551537070958a96cb6d4e540bf5c387b3bc33e2',
 'research/reviews/native_argformer_tsinfer_fit_v2_independent/review_receipt.json':'10e487bc8089d6a8713c172d7eb6731de50a7e68c48b8e1f9c61877f04b92295'}
for path,h in expected.items():check('requested_pin_'+path,m.sha(ROOT/path)==h)
author=json.loads((ROOT/'research/reviews/native_argformer_tsinfer_fit_v3_author/source_receipt.json').read_text())
v2receipt=json.loads((ROOT/'research/reviews/native_argformer_tsinfer_fit_v2_author/source_receipt.json').read_text())
for record in author['files']+v2receipt['files']:
 check('sealed_file_'+record['path'],m.sha(ROOT/record['path'])==record['sha256'])
proof=m.ast_proof(c)
check('all30_native_scientific_ASTs_exact',proof==v2receipt['native_ASTs'] and sum(len(v) for v in proof.values())==30)
old_config=json.loads((ROOT/'configs/native_argformer_tsinfer_fit_v2.json').read_text())
for field in ('native_recipe','source_commit','prepared_dispatchers','qualified_corpus_metadata','retrieval_rule','source_AST','runtime'):
 check('unchanged_scientific_contract_'+field,c[field]==old_config[field])
old_tree=ast.parse((ROOT/'src/arggnn/native_argformer_tsinfer_fit_v2.py').read_text());new_tree=ast.parse(m.SCRIPT.read_text())
for name in ('execute_fit','execute_retrieval','verify_completed_fit','verify_memory','retrieval_parent'):
 old=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 new=next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 check('unchanged_scientific_wrapper_'+name,ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False))
for path,h in c['frozen_files_sha256'].items():check('frozen_source_'+path,m.sha(ROOT/path)==h)
recipe=m.read_json(c['native_recipe'])
check('source_MLM512_micro64_5epochs_seed17',all(recipe['pretraining'][k]==v for k,v in {'epochs':5,'seed':17,'logical_batch':512,'microbatch':64}.items()))
check('source_CL4096_fullbatch_1epoch_seed36',all(recipe['contrastive'][k]==v for k,v in {'epochs':1,'seed':36,'logical_batch':4096,'microbatch':4096,'default_execution':'full_batch'}.items()))
check('source_full_reference_k1_k5_no_query_targets',recipe['primary_readout']=='full_reference_corpus_cosine_top1_hard_votes' and recipe['sensitivity_k']==5 and recipe['query_ancestry_targets_read'] is False)
check('source_inactive_and_no_fallback',c['execution_enabled'] is False and c['claims']['gradient_cache_or_fallback_authorized'] is False)
check('real_prepared_helper_exact_bytes_no_actual_import',m.additive_helper_paths(c,{'additive_helpers':c['runtime_inventory_contract']['expected_additive_helpers']})==[])
for parent in c['parent_ids']:
 scope,_,tmp,_=m.scopes(c,parent,'fit','fresh_review')
 check('fresh20GB_scope_and_short_tmp_'+parent,'/native_argformer_tsinfer_fit_v3/' in str(scope) and len(os.fsencode(tmp/'pymp-12345678/listener-12345678'))<=107)
 for stage in m.STAGES:
  t=m.templates(c,parent,stage,'fresh_review')
  check('inactive_separate_stage_'+parent+'_'+stage,t['admission']['active'] is False and t['runtime']['active'] is False and t['storage']['active'] is False)
check('four20GB_parent_scopes',len(c['parent_ids'])==4 and c['storage']['reserved_bytes_per_parent']==20000000000)
for role,record in c['storage']['companion_contract'].items():check('exact_budgetv4_'+role,m.reference(ROOT/record['path'])==record)

fixture=REVIEW/'f';fixture.mkdir()
fake=fixture/'runtime';prefix=fake/c['runtime']['prefix'];site=prefix/'lib/python3.11/site-packages';site.mkdir(parents=True)
binary=prefix/'bin/python3.11';binary.parent.mkdir();binary.write_bytes(b'fresh inert interpreter, never executed');binary.chmod(0o700)
alias=prefix/'bin/python';alias.symlink_to(binary.name)
d={m.distribution_name(k):v for k,v in c['runtime_inventory_contract']['expected_scientific_distributions'].items()}
objects={};module_records={}
for name,dist in c['runtime_inventory_contract']['scientific_modules'].items():
 p=site/(name+'.py');p.write_text('# fresh inert native module identity\n')
 obj=types.ModuleType(name);obj.__file__=str(p);obj.__version__=d[m.distribution_name(dist)]
 obj.__spec__=types.SimpleNamespace(loader=importlib.machinery.SourceFileLoader(name,str(p)));objects[name]=obj
 module_records[name]={'distribution':m.distribution_name(dist),'distribution_version':d[m.distribution_name(dist)],'module_version':obj.__version__,'origin':str(p),'sha256':m.sha(p),'loader':'_frozen_importlib_external.SourceFileLoader'}
with patch.object(m,'ROOT',fake):
 fc=copy.deepcopy(c);fc['runtime_inventory_contract']['expected_additive_helpers']=[];fc['runtime_inventory_contract']['helper_provenance']=None
 rt={'python_executable_sha256':m.sha(alias),'additive_helpers':[]}
 inventory={'kind':c['runtime_inventory_contract']['kind'],'status':c['runtime_inventory_contract']['status'],'prefix':str(prefix),
  'python':{'version':[3,11,9],'executable':str(alias),'resolved_executable':str(binary),'sha256':m.sha(alias)},
  'distributions':d,'scientific_modules':module_records,'import_file_sha256':m.native_import_bytes(prefix),'additive_helpers':[]}
 rt['environment_inventory_receipt']=write(fake/'inventory.json',inventory)
 check('fresh_complete_handmade_inventory_schema',m.parse_runtime_inventory(fc,inventory,rt)==inventory)
 for name,bad in [('empty',{}),('empty_distributions',dict(inventory,distributions={})),('empty_files',dict(inventory,import_file_sha256={})),('missing_module',dict(inventory,scientific_modules={}))]:
  reject('reject_runtime_inventory_'+name,lambda bad=bad:m.parse_runtime_inventory(fc,bad,rt))
 bad=copy.deepcopy(inventory);bad['distributions']['numpy']='99.0'
 reject('reject_pinned_distribution_version',lambda:m.parse_runtime_inventory(fc,bad,rt))
 bad=copy.deepcopy(inventory);bad['scientific_modules']['numpy']['origin']=str(fixture/'outside.py')
 reject('reject_scientific_origin_outside_prefix',lambda:m.parse_runtime_inventory(fc,bad,rt))
 bad=copy.deepcopy(inventory);bad['scientific_modules']['numpy']['sha256']='f'*64
 reject('reject_scientific_origin_bytes_unbound',lambda:m.parse_runtime_inventory(fc,bad,rt))
 with patch.object(sys,'prefix',str(prefix)),patch.object(sys,'executable',str(alias)),patch.object(sys,'version_info',(3,11,9)),patch.object(m,'native_distributions',return_value=d),patch.object(m.importlib,'import_module',side_effect=lambda name:objects[name]),patch.dict(os.environ,{'PYTHONPATH':'','CUDA_VISIBLE_DEVICES':''}):
  check('real_preflight_complete_handmade_identity_comparison',m.runtime_preflight(fc,rt,True)['inventory_matches_actual_child_scientific_runtime'])
  objects['numpy'].__version__='wrong'
  reject('reject_actual_child_module_version_drift',lambda:m.runtime_preflight(fc,rt,True));objects['numpy'].__version__=d['numpy']
  original=objects['numpy'].__file__;objects['numpy'].__file__=str(fixture/'outside.py')
  reject('reject_actual_child_scientific_shadow',lambda:m.runtime_preflight(fc,rt,True));objects['numpy'].__file__=original
  extra=site/'numpy.pyc';extra.write_bytes(b'fresh unexecuted bytecode shadow')
  reject('reject_actual_child_unrecorded_executable_bytecode',lambda:m.runtime_preflight(fc,rt,True));extra.unlink()
  with patch.object(m,'native_distributions',return_value=dict(d,numpy='wrong')):
   reject('reject_actual_child_complete_distribution_drift',lambda:m.runtime_preflight(fc,rt,True))
 helper=fake/'.runtime/native_helpers/review_helper';helper.mkdir(parents=True)
 hp=helper/'threadpoolctl.py';hp.write_text("__version__='independent_fixture_only'\n")
 wheel=fake/'helper.whl';wheel.write_bytes(b'handmade never installed')
 def helper_authority():
  manifest=write(fake/'helper_inventory.json',{'directory':str(helper),'files_sha256':{str(p.relative_to(helper)):m.sha(p) for p in helper.rglob('*') if p.is_file()}})
  h={'module':'threadpoolctl','version':'independent_fixture_only','implementation':m.reference(hp),'wheel':m.reference(wheel),'directory_inventory':manifest}
  fc['runtime_inventory_contract']['expected_additive_helpers']=[h]
  return {'additive_helpers':[h]}
 hr=helper_authority()
 check('helper_has_no_generic_import_search_path',m.additive_helper_paths(fc,hr)==[])
 for name in ('numpy.pyc','torch.pyo','composer.so','nested/__pycache__/numpy.cpython-311.pyc','sitecustomize.pth','torch.dll'):
  p=helper/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'unexecuted shadow')
  hr=helper_authority();reject('reject_exact_manifest_executable_helper_shadow_'+name,lambda:m.additive_helper_paths(fc,hr));p.unlink()
 hr=helper_authority()
 with patch.dict(sys.modules,{},clear=False),patch.dict(os.environ,{'PYTHONPATH':''}):
  sys.modules.pop('threadpoolctl',None);before=list(sys.path)
  found=m.load_exact_helpers(fc,hr)
  check('helper_exact_source_and_discovered_version',found[0]['version']=='independent_fixture_only' and Path(sys.modules['threadpoolctl'].__file__)==hp)
  check('helper_import_leaves_sys_path_unchanged',sys.path==before and str(helper) not in sys.path)
  hr['additive_helpers'][0]['version']='invented';fc['runtime_inventory_contract']['expected_additive_helpers']=hr['additive_helpers']
  reject('reject_actual_helper_version_mismatch',lambda:m.load_exact_helpers(fc,hr))

# True execute() entry, all science and storage/resource boundaries inert.
ec=copy.deepcopy(c);ec['result_root']=str(fixture/'children')
parent=c['parent_ids'][0]
base=write(fixture/'root.json',{'fixture_only_not_authority':True})
admission={'independent_review':{},'runtime':{},'experiment_registration':{},'prepared_dispatcher':{},'storage':{}}
storage=types.SimpleNamespace(check=lambda:{'fixture_only':True})
real_read=m.read_json
for phase in ('root_admission','exec_admission','runtime_preflight','science_exception'):
 run='owned_'+phase;_,out,tmp,_=m.scopes(ec,parent,'retrieve',run);out.mkdir(parents=True)
 write(out/'exec_admission.json',{'implementation_sha256':m.sha(m.SCRIPT),'stage':'retrieve','root_admission':base,'TMPDIR':str(tmp)})
 args=types.SimpleNamespace(parent=parent,stage='retrieve',run_id=run,native=True,expected_source_sha256=m.sha(m.SCRIPT),expected_config_sha256=m.sha(m.CONFIG),admission=str(ROOT/base['path']),expected_admission_sha256=base['sha256'])
 def read(record,expected=None):
  if record==base:return admission
  if record=={}:return {}
  return real_read(record,expected)
 if phase=='exec_admission':(out/'exec_admission.json').write_text('{malformed')
 validate_error=ValueError('fresh independent root admission rejection') if phase=='root_admission' else None
 runtime_error=ValueError('fresh independent runtime mismatch') if phase=='runtime_preflight' else None
 with patch.object(m,'verify_source',return_value=None),patch.object(m,'read_json',side_effect=read),patch.object(m,'validate_admission',side_effect=validate_error,return_value={}),patch.object(m,'prepared_metadata',return_value={}),patch.object(m,'Storage',return_value=storage),patch.object(m,'runtime_preflight',side_effect=runtime_error,return_value={'fixture_only':True}),patch.object(m,'execute_retrieval',side_effect=RuntimeError('fresh independent stage rejection')),patch.dict(os.environ,{'TMPDIR':str(tmp)}):
  result=m.execute(ec,args)
 check('execute_owned_'+phase+'_terminal_receipt_and_exit_status',(out/'receipt.json').is_file() and (out/'exit_status').read_text()=='1\n' and result['exit_code']==1)
 check('execute_owned_'+phase+'_never_native_success',result['root_actual_execution'] is False)
 reject('execute_owned_'+phase+'_terminal_not_reused',lambda:m.execute(ec,args))
terminal=fixture/'bounded_terminal';terminal.mkdir()
compact=m.terminal_receipt(c,terminal,{'exit_code':1,'failure':None,'untrusted':float('nan'),'large':'x'*100000})
check('malformed_large_failure_terminal_bounded',compact['exit_code']==1 and (terminal/'receipt.json').stat().st_size<=65536 and (terminal/'exit_status').read_text()=='1\n')

# Main's policy() runs outside execute's owned terminal handler.
run='entry_policy_failure';_,out,tmp,_=m.scopes(ec,parent,'retrieve',run);out.mkdir(parents=True)
write(out/'exec_admission.json',{'fixture_only_not_authority':True,'implementation_sha256':m.sha(m.SCRIPT),'stage':'retrieve','root_admission':base,'TMPDIR':str(tmp)})
argv=['--execute','--native','--parent',parent,'--stage','retrieve','--run-id',run,'--expected-source-sha256',m.sha(m.SCRIPT),'--expected-config-sha256',m.sha(m.CONFIG),'--admission',str(ROOT/base['path']),'--expected-admission-sha256',base['sha256']]
child_config=fixture/'child_config.json'
child_ref=write(child_config,ec);before=child_config.read_bytes()
argv[argv.index('--expected-config-sha256')+1]=child_ref['sha256']
with patch.object(m,'CONFIG',child_config),patch.object(m,'CONFIG_SHA256',child_ref['sha256']):
 check('entry_fixture_real_policy_initially_exact',m.policy()==ec)
 child_config.write_bytes(before+b'\n')
 failure=reject('main_owned_child_policy_byte_drift_rejected',lambda:m.main(argv),'exact inactive reviewed')
check('counterexample_main_policy_failure_no_terminal',not (out/'receipt.json').exists() and not (out/'exit_status').exists())
counterexamples.append({'id':'OWNED_CHILD_ENTRY_POLICY_FAILURE_NO_TERMINAL','priority':2,'source':'src/arggnn/native_argformer_tsinfer_fit_v3.py','line':884,'failure':failure,'fixture_output':str(out.relative_to(ROOT)),'observation':'Real main(argv --execute --native) and real policy() reject changed bytes in a handmade initially hash-valid config before execute() is called, leaving fresh owned output+exec_admission without receipt.json or exit_status. Only the config path/digest constants are relocated to the contained fixture. Original source/config remain immutable.'})

# Small full-reference native retrieval seal: all280 leaves,96 reference paths,
# all120 query paths,768 dimensions and11,520 exhaustive pairs. No fit wiring.
import numpy as np
import torch
ns=m.compile_retrieval(c)
directory=fixture/'retrieval';directory.mkdir();output=directory/'sealed';output.mkdir()
nodes=[11+7*i for i in range(280)]
roster=[{'sample_index':i,'inferred_tree_node_id':node,'source_observed_node_id':2000+i,
 'role':'pure_source_reference' if i<96 else 'mixed_reference_only' if i<160 else 'query',
 'observed_reference_class':i%4 if i<96 else None} for i,node in enumerate(nodes)]
paths=tuple((node,10000+i) for i,node in enumerate(nodes));occurrences=tuple((i,node,0.,100.) for i,node in enumerate(nodes));namespace='fresh_independent_native:'+('7'*64)
with (directory/'corpus.pkl').open('xb') as stream:pickle.dump({'paths':paths,'occurrences':occurrences,'namespace':namespace},stream,protocol=4)
raw=directory/'embedding/all/raw_all_sequences_and_vocab.pkl';raw.parent.mkdir(parents=True)
with raw.open('xb') as stream:pickle.dump({'raw_sequences':[list(p) for p in paths],'vocab':{}},stream,protocol=4)
joins={'grid_indexes':np.array([0]),'original_positions_bp':np.array([50]),'local_positions_bp':np.array([50]),
 'query_nodes':np.array(nodes[160:]),'reference_nodes':np.array(nodes[:96]),'reference_classes':np.arange(96)%4,
 'query_path_ids':np.arange(160,280,dtype=np.int64)[None,:],'reference_path_ids':np.arange(96,dtype=np.int64)[None,:]}
prepared={'parent_id':parent,'namespace':namespace,'unique_full_paths':280,
 'coordinate_contract':{'retained_fixed100bp_indexes':[0],'coverage_original_bp0':[0,100],'local_to_original_offset_bp':0},
 'files_sha256':{str(p.relative_to(directory)):m.sha(p) for p in directory.rglob('*') if p.is_file()}}
pref=write(directory/'prepared_manifest.json',prepared)
vectors=torch.zeros((280,768),dtype=torch.float32);vectors[:,0]=1;vectors[0]=0
embedding=directory/'handmade_embeddings.pt';torch.save({'embeddings':vectors,'native_pooling':'trained_author_mean_head'},embedding)
fitref=write(directory/'fixture_not_native_fit.json',{'fixture_only':True,'actual_native_fit':False})
meanref=write(directory/'fixture_aux_readout.json',{'fixture_only':True,'primary':False})
state={'prepared':prepared,'records':{'prepared_manifest':pref,'fit_receipt':fitref},'joins':joins,
 'receipt':{'readout':{'native_embedding_sha256':ns['tensor_sha'](vectors)},'fixture_only':True,'actual_native_fit':False}}
parent_record=m.retrieval_parent({'prepared':prepared},joins,roster,np)
metadata={'reference_full_path_count':96,'embeddings':m.reference(embedding),'original_mean_readout':meanref}
seal=ns['seal_parent'](parent_record,state,{'retrieval_rule':c['retrieval_rule']},metadata,output,torch)
diag=seal['diagnostics']
check('fresh_small_full_reference_exhaustive_native_seal',diag['pure_reference_full_paths']==96 and diag['unique_query_paths']==120 and diag['exhaustive_pairs']==11520)
check('native_seal_duplicate_embeddings_zero_row_crossclass_ties',diag['native_embedding_zero_or_epsilon_rows']==1 and diag['top1_cross_class_exact_max_ties']==120)
with np.load(m.bound(seal['neighbors']),allow_pickle=False) as z:
 check('full_reference_IDs_retained',np.array_equal(z['corpus_full_path_ids'],np.arange(96)))
 check('native_exact_tie_original_ID_order',np.array_equal(z['neighbor_full_path_ids'][0],np.arange(1,7)))
with np.load(m.bound({'path':seal['path'],'sha256':seal['sha256']}),allow_pickle=False) as z:
 check('native_k1_k5_all_common_support',z['C4_probabilities_raw_k1'].shape==(1,120,4) and z['C4_probabilities_raw_k5'].shape==(1,120,4))
 check('native_raw_votes_no_probability_floor',np.array_equal(z['C4_probabilities_raw_k1'][0,0],np.array([0,1,0,0])) and np.allclose(z['C4_probabilities_raw_k5'][0,0],[.2,.4,.2,.2]))
check('handmade_retrieval_no_CUDA_or_truth',not torch.cuda.is_initialized() and seal['truth_opened'] is False)
query=np.ones((2,4),dtype=np.float32);corpus=np.ones((8,4),dtype=np.float32);ids=np.array([20,2,17,3,8,7,5,11]);labels=np.arange(8)%4
_,selected,_=ns['exhaustive_neighbors'](query,corpus,ids,labels,1,3,6)
check('native_chunk_merge_exact_score_ties',selected.tolist()==[sorted(ids.tolist())[:6]]*2)
bad=copy.deepcopy(joins);bad['query_path_ids'][0,0]=-1
reject('native_support_rejects_missing_query',lambda:ns['validate_support'](paths,occurrences,namespace,namespace,bad,100))
reject('native_support_rejects_cross_ARG',lambda:ns['validate_support'](paths,occurrences,'other',namespace,joins,100))
bad=copy.deepcopy(roster);bad[160]['observed_reference_class']=0
reject('native_roster_rejects_query_target',lambda:m.retrieval_parent({'prepared':prepared},joins,bad,np))
details['fresh_handmade_retrieval']={'full_paths':280,'observed_leaves':280,'pure_reference_full_paths':96,'query_full_paths':120,'embedding_dim':768,'exhaustive_pairs':11520,'native_fit_evidence':False,'seal':seal}
details['scientific_ASTs']=proof
for path,h in expected.items():check('final_immutable_pin_'+path,m.sha(ROOT/path)==h)
report={'kind':'native_argformer_tsinfer_fit_v3_independent_handmade_checks','status':'completed_with_one_reproduced_blocker',
 'check_count':len(checks),'checks':checks,'counterexamples':counterexamples,'details':details,
 'actual_model_fit_executed':False,'released_model_instantiated':False,'actual_scientific_arrays_opened':False,
 'actual_GPU_inventory_context_or_live_ledger_read':False,'actual_native_runtime_inventory_captured':False,
 'actual_native_helper_import_performed':False,'runtime_prefix_modified':False,
 'genuine_distribution_provider_stubbed_only_for_handmade_guard_tests':True,'authority_activation_performed':False,
 'actual_fit_success_wiring_created':False,'handmade_retrieval_kernel_executed':True}
write(REVIEW/'handmade_checks.json',report)
print(json.dumps({'status':report['status'],'check_count':len(checks),'counterexamples':counterexamples,'fresh_handmade_retrieval':details['fresh_handmade_retrieval']},indent=2))
