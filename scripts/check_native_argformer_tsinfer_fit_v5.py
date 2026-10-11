"""Guard counterexamples and genuine full retrieval on handmade fixtures only."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import types
from unittest.mock import patch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');REVIEW=ROOT/'research/reviews/native_argformer_tsinfer_fit_v5_author'
sys.dont_write_bytecode=True;os.environ['CUDA_VISIBLE_DEVICES']=''
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
m=load('_author_guard_v3',ROOT/'src/arggnn/native_argformer_tsinfer_fit_v5.py');c=m.policy();checks=[];details={};sha=m.sha
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
def reject(name,call):
 try:call()
 except (ValueError,RuntimeError,KeyError,TypeError,FileNotFoundError):checks.append(name);return
 raise AssertionError('unexpected acceptance: '+name)
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return m.reference(path)
def preserved():
 for version in (1,2,3,4):
  receipt=ROOT/f'research/reviews/native_argformer_tsinfer_fit_v{version}_author/source_receipt.json'
  data=json.loads(receipt.read_text())
  for r in data['files']:check('sealed_preserved_'+r['path'],m.sha(ROOT/r['path'])==r['sha256'])
preserved()
check('sealed_REJECT_before_successor',m.sha(ROOT/'research/reviews/native_argformer_tsinfer_fit_v2_independent/review_receipt.json')=='10e487bc8089d6a8713c172d7eb6731de50a7e68c48b8e1f9c61877f04b92295')
v2=json.loads((ROOT/'research/reviews/native_argformer_tsinfer_fit_v2_author/source_receipt.json').read_text())
check('all30_scientific_ASTs_preserved',m.ast_proof(c)==v2['native_ASTs'])
old=json.loads((ROOT/'configs/native_argformer_tsinfer_fit_v2.json').read_text())
check('genuine_root_prepared_helper_bytes_bound_without_import',m.additive_helper_paths(c,{'additive_helpers':c['runtime_inventory_contract']['expected_additive_helpers']})==[])
reject('empty_helpers_cannot_activate_actual_pinned_runtime',lambda:m.additive_helper_paths(c,{'additive_helpers':[]}))
for field in ('native_recipe','source_commit','prepared_dispatchers','qualified_corpus_metadata','retrieval_rule','source_AST','runtime'):
 check('pinned_scientific_contract_'+field,c[field]==old[field])

v2tree=ast.parse((ROOT/'src/arggnn/native_argformer_tsinfer_fit_v2.py').read_text());v3tree=ast.parse(m.SCRIPT.read_text())
for name in ('execute_fit','execute_retrieval','verify_completed_fit','verify_memory','retrieval_parent'):
 oldnode=next(n for n in v2tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 newnode=next(n for n in v3tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 check('scientific_wrapper_AST_unchanged_'+name,ast.dump(oldnode,include_attributes=False)==ast.dump(newnode,include_attributes=False))
for field in ('implementation','config'):
 record=c['storage']['companion_contract'][field]
 check('exact_bound_budget_v4_'+field,m.reference(ROOT/record['path'])==record)
for parent in c['parent_ids']:
 parent_scope,_,tmp,_=m.scopes(c,parent,'probe-memory','root_assign_fresh_run')
 check('real_production_socket_limit_'+parent,len(os.fsencode(tmp/'pymp-12345678/listener-12345678'))<=107)
 check('fresh_v5_parent_storage_scope_'+parent,'/native_argformer_tsinfer_fit_v5/' in str(parent_scope) and '/afts5/' in str(tmp))
 for stage in m.STAGES:
  t=m.templates(c,parent,stage,'root_assign_fresh_run')
  check('inactive_stage_template_'+parent+'_'+stage,t['admission']['active'] is False and t['storage']['active'] is False and t['runtime']['active'] is False and t['experiment_registration']['active'] is False and t['runtime']['additive_helpers']==c['runtime_inventory_contract']['expected_additive_helpers'])

with tempfile.TemporaryDirectory(prefix='guards_',dir=REVIEW) as temp:
 directory=Path(temp);fake=directory/'runtime_fixture';prefix=fake/c['runtime']['prefix'];site=prefix/'lib/python3.11/site-packages';site.mkdir(parents=True)
 binary=prefix/'bin/python3.11';binary.parent.mkdir();binary.write_bytes(b'handmade executable identity, never executed');binary.chmod(0o755)
 alias=prefix/'bin/python';alias.symlink_to(binary.name)
 distributions={m.distribution_name(k):v for k,v in c['runtime_inventory_contract']['expected_scientific_distributions'].items()}
 modules={};objects={}
 for name,dist in c['runtime_inventory_contract']['scientific_modules'].items():
  p=site/(name+'.py');p.write_text('# handmade package identity only\n')
  obj=types.ModuleType(name);obj.__file__=str(p);obj.__version__=distributions[m.distribution_name(dist)];obj.__spec__=types.SimpleNamespace(loader=importlib.machinery.SourceFileLoader(name,str(p)));objects[name]=obj
  modules[name]={'distribution':m.distribution_name(dist),'distribution_version':distributions[m.distribution_name(dist)],'module_version':obj.__version__,'origin':str(p),'sha256':m.sha(p),'loader':'_frozen_importlib_external.SourceFileLoader'}
 with patch.object(m,'ROOT',fake):
  fc=copy.deepcopy(c);fc['runtime_inventory_contract']['expected_additive_helpers']=[];fc['runtime_inventory_contract']['helper_provenance']=None
  runtime={'python_executable_sha256':m.sha(alias),'additive_helpers':[]}
  inventory={'kind':c['runtime_inventory_contract']['kind'],'status':c['runtime_inventory_contract']['status'],'prefix':str(prefix),
   'python':{'version':[3,11,9],'executable':str(alias),'resolved_executable':str(binary),'sha256':m.sha(alias)},'distributions':distributions,'scientific_modules':modules,'import_file_sha256':m.native_import_bytes(prefix),'additive_helpers':[]}
  iref=write(fake/'inventory.json',inventory);runtime['environment_inventory_receipt']=iref
  check('strict_complete_handmade_inventory_parses',m.parse_runtime_inventory(fc,inventory,runtime)==inventory)
  for name,bad in [('empty',{}),('missing_modules',{k:v for k,v in inventory.items() if k!='scientific_modules'}),('empty_files',dict(inventory,import_file_sha256={})),('wrong_prefix',dict(inventory,prefix=str(directory)) )]:
   reject('inventory_reject_'+name,lambda bad=bad:m.parse_runtime_inventory(fc,bad,runtime))
  bad=copy.deepcopy(inventory);bad['distributions']['numpy']='99.0'
  reject('pinned_numpy_distribution_version_mismatch',lambda:m.parse_runtime_inventory(fc,bad,runtime))
  bad=copy.deepcopy(inventory);bad['scientific_modules']['numpy']['origin']=str(directory/'shadow_numpy.py')
  reject('inventory_module_origin_outside_native_prefix',lambda:m.parse_runtime_inventory(fc,bad,runtime))
  bad=copy.deepcopy(inventory);bad['scientific_modules']['numpy']['sha256']='9'*64
  reject('inventory_module_import_bytes_unbound',lambda:m.parse_runtime_inventory(fc,bad,runtime))
  def imported(name):return objects[name]
  with patch.object(sys,'prefix',str(prefix)),patch.object(sys,'executable',str(alias)),patch.object(sys,'version_info',(3,11,9)),patch.dict(os.environ,{'PYTHONPATH':'','CUDA_VISIBLE_DEVICES':''}),patch.object(m,'native_distributions',return_value=distributions),patch.object(m.importlib,'import_module',side_effect=imported):
   proof=m.runtime_preflight(fc,runtime,True)
   check('actual_child_runtime_compares_complete_import_bytes_versions_modules',proof['inventory_matches_actual_child_scientific_runtime'])
   objects['torch'].cuda=types.SimpleNamespace(is_initialized=lambda:False)
   with patch.dict(sys.modules,{'torch':objects['torch']}):
    measured=m.capture_runtime_inventory(fc,runtime)
   check('root_capture_measures_complete_handmade_package_identities_without_model',measured==inventory)
   with patch.object(sys,'executable',str(directory/'different_interpreter')):
    reject('root_capture_rejects_actual_wrong_interpreter',lambda:m.capture_runtime_inventory(fc,runtime))
   shadow=types.ModuleType('numpy');shadow.__file__=str(directory/'outside_numpy.py');Path(shadow.__file__).write_text('# shadow');shadow.__version__='1.26.4';shadow.__spec__=objects['numpy'].__spec__
   objects['numpy']=shadow
   reject('actual_shadowed_package_rejected_even_with_expected_version',lambda:m.runtime_preflight(fc,runtime,True))
   objects['numpy']=types.SimpleNamespace(__file__=modules['numpy']['origin'],__version__=modules['numpy']['module_version'],__spec__=types.SimpleNamespace(loader=importlib.machinery.SourceFileLoader('numpy',modules['numpy']['origin'])))
   objects['numpy'].__version__='wrong_module_version'
   reject('actual_module_version_mismatch_rejected',lambda:m.runtime_preflight(fc,runtime,True));objects['numpy'].__version__=modules['numpy']['module_version']
   p=site/'numpy.py';original=p.read_text();p.write_text('# changed import bytes')
   reject('actual_prefix_import_byte_drift_rejected_before_package_import',lambda:m.runtime_preflight(fc,runtime,True));p.write_text(original)
   p=site/'numpy.pyc';p.write_bytes(b'handmade never executed bytecode')
   reject('unrecorded_actual_native_bytecode_shadow_rejected',lambda:m.runtime_preflight(fc,runtime,True));p.unlink()
   with patch.object(m,'native_distributions',return_value=dict(distributions,numpy='wrong')):
    reject('actual_complete_distribution_map_mismatch_rejected',lambda:m.runtime_preflight(fc,runtime,True))
  helperdir=fake/'.runtime/native_helpers/threadpoolctl_3_7_0_v1';helperdir.mkdir(parents=True)
  hp=helperdir/'threadpoolctl.py';hp.write_text("__version__='handmade_only'\n")
  wheel=helperdir/'threadpoolctl-fixture.whl';wheel.write_bytes(b'handmade never installed')
  def helper_authority():
   href=write(fake/'helper_inventory.json',{'directory':str(helperdir),'files_sha256':{str(p.relative_to(helperdir)):m.sha(p) for p in helperdir.rglob('*') if p.is_file()}})
   value={'additive_helpers':[{'module':'threadpoolctl','version':'handmade_only','implementation':m.reference(hp),'wheel':m.reference(wheel),'directory_inventory':href}]}
   fc['runtime_inventory_contract']['expected_additive_helpers']=value['additive_helpers']
   return value
  hr=helper_authority();check('helper_has_no_generic_scientific_search_path',m.additive_helper_paths(fc,hr)==[])
  for name in ['numpy.pyc','torch.pyo','composer.so','triton.abi3.so','sitecustomize.pth','pkg/__pycache__/numpy.cpython-311.pyc','torch.dll']:
   p=helperdir/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'handmade unexecuted shadow')
   hr=helper_authority();reject('helper_rejects_executable_shadow_'+name,lambda:m.additive_helper_paths(fc,hr));p.unlink()
  hr=helper_authority()
  with patch.dict(sys.modules,{},clear=False),patch.dict(os.environ,{'PYTHONPATH':''}):
   sys.modules.pop('threadpoolctl',None);before=list(sys.path);actual_helpers=m.load_exact_helpers(fc,hr)
   check('captured_helper_version_is_actual_source_version',actual_helpers[0]['version']==sys.modules['threadpoolctl'].__version__)
   check('helper_loaded_by_exact_source_origin',Path(sys.modules['threadpoolctl'].__file__)==hp)
   check('helper_import_never_adds_directory_to_sys_path',before==sys.path and str(helperdir) not in sys.path)
  reject('stale_helper_authority_rejected_by_exact_config',lambda:m.additive_helper_paths(fc,{'additive_helpers':[]}))
  saved=hp.read_text();hp.write_text("__version__='wrong_actual_version'\n")
  hr=helper_authority()
  with patch.dict(sys.modules,{},clear=False):
   sys.modules.pop('threadpoolctl',None)
   wrong={'additive_helpers':[dict(hr['additive_helpers'][0],version='expected_invented')]};fc['runtime_inventory_contract']['expected_additive_helpers']=wrong['additive_helpers']
   reject('genuine_helper_version_discovery_rejects_wrong_authority',lambda:m.load_exact_helpers(fc,wrong))
  hp.write_text(saved)
  hr=helper_authority()
  reject('helper_rejects_stale_isolated_helper_root',lambda:m.additive_helper_paths(dict(fc,runtime_inventory_contract=dict(fc['runtime_inventory_contract'],allowed_additive_helper_roots=['.runtime/isolated_helpers'])),hr))
  # Real bootstrap environment and exec admission, with only execve stubbed.
  fc['runtime_inventory_contract']['expected_additive_helpers']=[]
  ba=types.SimpleNamespace(parent=fc['parent_ids'][0],stage='retrieve',run_id='bootstrap_fixture',admission=str(fake/'root_admission.json'))
  write(Path(ba.admission),{'fixture_only':True})
  bs=types.SimpleNamespace(check=lambda:{'fixture_only':True});ownership={'owns_output':False}
  calls=[]
  def fixture_exec(executable,argv,env):calls.append((str(executable),argv,env));raise RuntimeError('fixture-only exec boundary')
  original_fsencode=os.fsencode
  def fixture_fsencode(path):return original_fsencode(str(path).replace(str(fake),'/handmade_short_root'))
  details['bootstrap_handmade_socket_root_shortened_only_for_fixture_path_length']=True
  with patch.dict(os.environ,{},clear=False),patch.object(os,'execve',side_effect=fixture_exec),patch.object(os,'chdir'),patch.object(os,'fsencode',side_effect=fixture_fsencode):
   reject('complete_bootstrap_reaches_exact_native_exec_stub',lambda:m.bootstrap(fc,ba,{},runtime,bs,ownership))
  _,bo,bt,_=m.scopes(fc,ba.parent,'retrieve',ba.run_id)
  check('complete_bootstrap_owns_fresh_output_and_records_exec_admission',ownership['owns_output'] and (bo/'exec_admission.json').is_file())
  check('complete_bootstrap_never_injects_helper_directory_or_PYTHONPATH',len(calls)==1 and calls[0][2]['PYTHONPATH']=='')
  check('complete_bootstrap_CPU_CUDA_hidden_and_B_interpreter',calls[0][1][:2]==[str(alias),'-B'] and calls[0][2]['CUDA_VISIBLE_DEVICES']=='')

 # Complete owned-child error path with real execute() and terminal writer.
 ec=copy.deepcopy(c);ec['result_root']=str(directory/'child_results');ec['short_tmp_root']='.runtime/tmp/afts5'
 parent=c['parent_ids'][0];a_path=directory/'root_admission.json';aref=write(a_path,{'handmade_only':True})
 arg=types.SimpleNamespace(parent=parent,stage='retrieve',run_id='child_preflight',native=True,expected_source_sha256=m.sha(m.SCRIPT),expected_config_sha256=m.sha(m.CONFIG),admission=str(a_path),expected_admission_sha256=aref['sha256'])
 admission={'independent_review':{},'runtime':{},'experiment_registration':{},'prepared_dispatcher':{},'storage':{}}
 storage=types.SimpleNamespace(check=lambda:{'fixture_only':True})
 def child_case(run,phase):
  arg.run_id=run;_,out,tmp,_=m.scopes(ec,parent,'retrieve',run);out.mkdir(parents=True)
  write(out/'exec_admission.json',{'implementation_sha256':m.sha(m.SCRIPT),'stage':'retrieve','root_admission':aref,'TMPDIR':str(tmp)})
  real_read=m.read_json
  def read(record,expected=None):
   if record==aref:return None if phase=='root_nondict' else admission
   if record in ({},):return [] if phase=='runtime_nondict' else {}
   return real_read(record,expected)
  with patch.object(m,'verify_source',return_value=None),patch.object(m,'validate_admission',return_value={}),patch.object(m,'prepared_metadata',return_value={}),patch.object(m,'Storage',return_value=storage),patch.object(m,'read_json',side_effect=read),patch.object(m,'runtime_preflight',side_effect=ValueError('handmade child runtime mismatch')),patch.dict(os.environ,{'TMPDIR':str(tmp)}):
   if phase=='admission':(out/'exec_admission.json').write_text('{invalid JSON')
   result=m.execute(ec,arg)
  check('owned_'+phase+'_exception_returns_failed_terminal_receipt',result['exit_code']==1 and (out/'receipt.json').is_file() and (out/'exit_status').read_text()=='1\n')
  check('owned_'+phase+'_exception_never_counts_native_success',result['root_actual_execution'] is False)
  reject('owned_'+phase+'_terminal_output_not_reused',lambda:m.execute(ec,arg))
 child_case('child_preflight','runtime');child_case('child_exec_admission','admission');child_case('child_root_nondict','root_nondict');child_case('child_runtime_nondict','runtime_nondict')
 malformed=directory/'malformed_terminal';malformed.mkdir()
 result=m.terminal_receipt(c,malformed,{'status':'failed','exit_code':1,'failure':{'type':'fixture','message':'bad metadata'},'untrusted':float('nan')})
 check('malformed_failure_metadata_cannot_suppress_terminal_evidence',(malformed/'exit_status').read_text()=='1\n' and result['exit_code']==1)
 for tag,bad in [('nondict_failure',{'failure':None,'bad':float('nan')}),('large_context',{'exit_code':1,'failure':'bad','large':'x'*100000}),('nondict_result',None),('bad_exit',{'exit_code':None}),('unicode_context',{'exit_code':1,'action':'😀'*10000,'parent_id':'😀'*10000,'implementation_sha256':'😀'*10000,'config_sha256':'😀'*10000,'failure':{'message':'😀'*10000}}),('nonstring_keys',{('bad','key'):1,'bad':float('nan')})]:
  out=directory/('terminal_'+tag);out.mkdir();result=m.terminal_receipt(c,out,bad)
  check('terminal_evidence_survives_'+tag,(out/'exit_status').read_text()=='1\n' and result['exit_code']==1 and (out/'receipt.json').stat().st_size<=c['storage']['failure_receipt_max_bytes'])

 fixture=directory
 import numpy as np
 import torch
 if importlib.util.find_spec("threadpoolctl") is None:
     helper=ROOT/".runtime/env/lib/python3.11/site-packages/threadpoolctl.py"
     dm=load("threadpoolctl",helper);sys.modules["threadpoolctl"]=dm
     details["test_only_helper"]={"path":str(helper.relative_to(ROOT)),"sha256":sha(helper),"version":dm.__version__}
 directory=fixture/"retrieval_prepared";directory.mkdir()
 nodes=[7+3*i for i in range(280)]
 roster=[{"sample_index":i,"inferred_tree_node_id":n,"source_observed_node_id":i+2000,"role":"pure_source_reference" if i<96 else "mixed_reference_only" if i<160 else "query","observed_reference_class":i//24 if i<96 else None} for i,n in enumerate(nodes)]
 paths=[];occurrences=[];at={}
 for i,node in enumerate(nodes):
     count=48 if i<96 else 2
     for j in range(count):
         pid=len(paths);paths.append((node,10000+j));occurrences.append((pid,node,200*j/count,200*(j+1)/count));at[(i,j)]=pid
 paths=tuple(paths);occurrences=tuple(occurrences);namespace=parent+":"+"8"*64
 corpus={"paths":paths,"occurrences":occurrences,"namespace":namespace}
 with (directory/"corpus.pkl").open("xb") as f:pickle.dump(corpus,f,protocol=4)
 raw=directory/"embedding/all/raw_all_sequences_and_vocab.pkl";raw.parent.mkdir(parents=True)
 vocab={"special":{"[PAD]":0,"[CLS]":1,"[SEP]":2,"[MASK]":3},"node_id_tokens":{"node_"+str(n):i+4 for i,n in enumerate(sorted({n for p in paths for n in p}))}}
 with raw.open("xb") as f:pickle.dump({"raw_sequences":[list(p) for p in paths],"vocab":vocab},f,protocol=4)
 positions=np.asarray([50,150])
 joins={"grid_indexes":np.asarray([0,1]),"original_positions_bp":positions,"local_positions_bp":positions,"query_nodes":np.asarray(nodes[160:]),"reference_nodes":np.asarray(nodes[:96]),"reference_classes":np.asarray([i//24 for i in range(96)]),
 "query_path_ids":np.asarray([[at[(i,j)] for i in range(160,280)] for j in (0,1)]),"reference_path_ids":np.asarray([[at[(i,j)] for i in range(96)] for j in (12,36)])}
 np.savez_compressed(directory/"coordinate_path_joins.npz",**joins);write(directory/"observed_roster.json",roster)
 prepared={"parent_id":parent,"namespace":namespace,"unique_full_paths":len(paths),"coordinate_contract":{"retained_fixed100bp_indexes":[0,1],"coverage_original_bp0":[0,200],"local_to_original_offset_bp":0},"files_sha256":{str(p.relative_to(directory)):sha(p) for p in directory.rglob("*") if p.is_file()}}
 pref=write(directory/"prepared_manifest.json",prepared)
 data={"dispatcher":{"parent_id":parent,"prepared_manifest":pref},"prepared":prepared,"budget":{"MLM_optimizer_updates":5,"contrastive_optimizer_updates":1,"pure_reference_corpus_paths_all_train_plus_val":4608,"actual_counts":{"query_full_paths_scored":240},"exhaustive_query_corpus_pairs":1105920}}
 rc=copy.deepcopy(c);rc["result_root"]=str(fixture/"retrieval_results");rc["short_tmp_root"]=str(fixture/"retrieval_tmp")
 args=types.SimpleNamespace(parent=parent,stage="retrieve",run_id="handmade")
 _,output,_,_=m.scopes(rc,parent,"retrieve","handmade");output.mkdir(parents=True)
 fitdir=Path(rc["result_root"])/parent/"fit"/"handmade";fitdir.mkdir(parents=True)
 vectors=torch.zeros((len(paths),768),dtype=torch.float32);vectors[:,0]=1;vectors[0]=0
 torch.save({"embeddings":vectors,"native_pooling":"trained_author_mean_head"},fitdir/"native_path_embeddings.pt")
 np.savez_compressed(fitdir/"downstream_C4_local_readout.npz",fixture_only=np.asarray([1]))
 fit={"fixture_only_not_native_fit_success":True,"status":"passed","action":"fit","exit_code":0,"root_actual_execution":True,"true_query_ancestry_read":False,"implementation_sha256":sha(m.SCRIPT),"config_sha256":sha(m.CONFIG),"parent_id":parent,"namespace":namespace,"prepared_manifest_sha256":pref["sha256"],"published_native_task_reproduction":False,"classifier_unchanged_unused":True,"encoder_initialization":{"all_encoder_keys_loaded_equal":["fixture_only"],"missing_encoder":[],"unequal_encoder":[]},"readout":{"native_embedding_sha256":hashlib.sha256(vectors.numpy().tobytes()).hexdigest()},"artifacts_sha256":{p.name:sha(p) for p in fitdir.iterdir()}}
 for phase,mode,ep,key in [("pretraining","mlm",5,"MLM_optimizer_updates"),("contrastive","contrastive",1,"contrastive_optimizer_updates")]:
     fit[phase]={"mode":mode,"execution":"full_batch","completed_epochs":ep,"actual_optimizer_updates":data["budget"][key],"backbone_unchanged":True,"true_query_ancestry_read":False,"initial_model_state_sha256":"1"*64,"final_model_state_sha256":"2"*64}
 fitref=write(fitdir/"receipt.json",fit)
 result=m.execute_retrieval(rc,args,{"completed_fit_receipt":fitref},data,storage,{})
 seal=result["prediction_seal"];diag=seal["diagnostics"]
 check("real_retrieval_wrapper_all_pure_full_paths_and_queries",diag["pure_reference_full_paths"]==4608 and diag["unique_query_paths"]==240 and diag["exhaustive_pairs"]==1105920)
 check("duplicate_embeddings_retained_zero_rows_and_crossclass_ties",diag["native_embedding_zero_or_epsilon_rows"]==1 and diag["top1_cross_class_exact_max_ties"]==240)
 with np.load(m.bound(seal["neighbors"]),allow_pickle=False) as z:
     check("exact_original_fullpath_tie_order",np.array_equal(z["neighbor_full_path_ids"][0],np.arange(1,7)))
     check("full_pure_reference_ids_no_query_selfmatch",np.array_equal(z["corpus_full_path_ids"],np.arange(4608)))
 with np.load(m.bound({"path":seal["path"],"sha256":seal["sha256"]}),allow_pickle=False) as z:
     check("real_k1_k5_sealed_all_retained_query_intervals",z["C4_probabilities_raw_k1"].shape==(2,120,4) and z["C4_probabilities_raw_k5"].shape==(2,120,4))
     check("raw_vote_zero_preserved",np.any(z["C4_probabilities_raw_k1"]==0))
     check("query_native_nonconsecutive_IDs_retained",np.array_equal(z["query_nodes"],nodes[160:]))
 check("no_GPU_or_truth_in_handmade_retrieval",not torch.cuda.is_initialized() and result["external_truth_access_authorized"] is False and seal["truth_opened"] is False)
 ns=m.compile_retrieval(c)
 bad=copy.deepcopy(joins);bad["query_path_ids"][0,0]=-1
 reject("real_support_rejects_missing_query_path",lambda:ns["validate_support"](paths,occurrences,namespace,namespace,bad,200))
 reject("real_support_rejects_cross_ARG_namespace",lambda:ns["validate_support"](paths,occurrences,"other",namespace,joins,200))
 bad=copy.deepcopy(joins);bad["reference_path_ids"][0,0]=1
 reject("real_support_rejects_wrong_reference_interval_path",lambda:ns["validate_support"](paths,occurrences,namespace,namespace,bad,200))
 bad=copy.deepcopy(roster);bad[160]["observed_reference_class"]=0
 reject("real_roster_rejects_query_ancestry_label",lambda:m.retrieval_parent(data,joins,bad,np))
 scores=np.asarray([[1,1,0,1,0,1,1]],dtype=np.float32);ids=np.asarray([70,2,9,11,1,5,3])
 v,i=ns["lexicographic_topk"](scores,ids,5)
 check("independent_nonordered_tie_counterexample",i.tolist()==[[2,3,5,11,70]])
 reject("real_topk_rejects_nonfinite_scores",lambda:ns["lexicographic_topk"](np.asarray([[np.nan]],dtype=np.float32),np.asarray([1]),1))
 details["fresh_retrieval_fixture"]={"full_paths":len(paths),"observed_leaves":280,"pure_reference_full_paths":4608,"query_full_paths":240,"exhaustive_pairs":1105920,"seal":seal}

# Real main(argv) entrypoint cases; only handmade ownership metadata is used.
with tempfile.TemporaryDirectory(prefix='entry_',dir=REVIEW) as temp:
 base=Path(temp);ec=copy.deepcopy(c);ec['result_root']=str(base/'outputs');actual_scopes=m.scopes
 def entry_scopes(_config,parent,stage,run):return actual_scopes(ec,parent,stage,run)
 parent=c['parent_ids'][0]
 check('entry_scope_contract_is_pinned_in_source',m.ENTRY_OWNERSHIP['parent_ids']==c['parent_ids'] and m.ENTRY_OWNERSHIP['result_root']==c['result_root'] and m.ENTRY_OWNERSHIP['short_tmp_root']==c['short_tmp_root'] and m.ENTRY_OWNERSHIP['storage']['failure_receipt_max_bytes']==c['storage']['failure_receipt_max_bytes'])
 def main_case(run,mode):
  scope,out,tmp,log=actual_scopes(ec,parent,'retrieve',run)
  out.mkdir(parents=True)
  if mode=='symlink_output':
   out.rmdir();target=base/(run+'_target');target.mkdir();out.symlink_to(target,target_is_directory=True)
  private_config=base/(run+'_config.json')
  if mode=='corrupt_config':private_config.write_text('{}\n')
  config_path=private_config if mode in ('corrupt_config','missing_config') else m.CONFIG
  config_ref={'path':str(config_path.relative_to(ROOT)),'sha256':m.CONFIG_SHA256}
  review={'kind':'native_argformer_tsinfer_fit_v5_independent_review','review_role':'independent_reviewer','verdict':'PASS','implementation':m.reference(m.SCRIPT),'config':config_ref,'handmade_metadata_only':True}
  if mode=='pending_review':review['verdict']='PENDING'
  rr=write(base/(run+'_review.json'),review)
  a=m.admission_bindings(c,parent,'retrieve',run)
  a.update(parent_storage_scope=str(scope),output=str(out),short_tmp=str(tmp),launch_log=str(log),config=config_ref,independent_review=rr,handmade_metadata_only=True)
  if mode=='inactive_root':a['active']=False
  ar=write(base/(run+'_admission.json'),a)
  marker={'implementation_sha256':m.sha(m.SCRIPT),'stage':'retrieve','root_admission':ar,'TMPDIR':str(tmp),'fixture_only':True}
  if mode=='wrong_marker':marker['root_admission']={**ar,'sha256':'f'*64}
  if mode not in ('missing_marker','nonexistent_output'):
   if mode=='symlink_output':(out/'exec_admission.json').write_text(json.dumps(marker)+'\n')
   else:write(out/'exec_admission.json',marker)
  if mode=='nonexistent_output':out.rmdir()
  old=None
  if mode=='existing_receipt':
   (out/'receipt.json').write_text('existing immutable evidence\n');old=(out/'receipt.json').read_bytes()
  if mode=='existing_exit':
   (out/'exit_status').write_text('7\n');old=(out/'exit_status').read_bytes()
  argv=['--execute','--native','--parent',parent,'--stage','retrieve','--run-id',run,'--admission',ar['path'],'--expected-admission-sha256',ar['sha256'],'--expected-source-sha256',m.sha(m.SCRIPT),'--expected-config-sha256',m.CONFIG_SHA256]
  if mode=='wrong_source':argv[argv.index('--expected-source-sha256')+1]='f'*64
  if mode=='wrong_config_argument':argv[argv.index('--expected-config-sha256')+1]='f'*64
  if mode=='bootstrap_not_owned':argv.remove('--native')
  env={'TMPDIR':str(tmp) if mode!='wrong_tmp' else str(base/'wrong_tmp')}
  genuine_policy=mode in ('corrupt_config','missing_config')
  policy_patch=patch.object(m,'CONFIG',config_path) if genuine_policy else patch.object(m,'policy',side_effect=ValueError('handmade actual main policy rejection'))
  before_files={p.name:p.read_bytes() for p in out.iterdir()} if out.exists() else {}
  with patch.object(m,'scopes',side_effect=entry_scopes),patch.dict(os.environ,env),policy_patch:
   if mode in ('injected_policy','corrupt_config','missing_config'):
    code=m.main(argv)
    result=json.loads((out/'receipt.json').read_text())
    check('real_main_owned_'+mode+'_terminal',code==1 and (out/'exit_status').read_text()=='1\n' and result['failure_stage']=='main_policy_preflight' and result['root_actual_execution'] is False and result['rejected_config_used_for_scope_or_limit'] is False)
    check('real_main_owned_'+mode+'_exact_root_marker_bound',result['root_admission']==ar and result['reviewed_config_sha256']==m.CONFIG_SHA256 and (out/'receipt.json').stat().st_size<=c['storage']['failure_receipt_max_bytes'])
    if mode=='missing_config':check('missing_config_terminal_observed_hash_is_null',result['config_sha256'] is None)
   else:
    reject('real_main_'+mode+'_does_not_finalize_unproven_scope',lambda:m.main(argv))
    check('real_main_'+mode+'_never_creates_or_overwrites_terminal',({p.name:p.read_bytes() for p in out.iterdir()} if out.exists() else {})==before_files)
  if mode=='nonexistent_output':check('real_main_never_creates_a_missing_native_output',not out.exists())
  return out
 for mode in ('injected_policy','corrupt_config','missing_config','missing_marker','wrong_marker','inactive_root','pending_review','wrong_source','wrong_config_argument','wrong_tmp','bootstrap_not_owned','existing_receipt','existing_exit','nonexistent_output','symlink_output'):
  main_case('main_'+mode,mode)
 missing=base/'never_created'
 before=set(base.rglob('*'))
 with patch.object(m,'policy',side_effect=ValueError('source-only policy rejection')):
  for argv in ([],['--templates'],['--execute'],['--execute','--native','--run-id','invalid/run']):
   reject('real_main_source_or_unowned_policy_failure_preserved_'+str(argv),lambda argv=argv:m.main(argv))
 check('source_and_unowned_entrypoint_failures_never_create_directories_or_files',set(base.rglob('*'))==before and not missing.exists())
 details['real_main_entrypoint_cases']={'owned_failures':['injected policy failure','actual private-config byte mismatch','missing private config'],'unproven_or_terminal_scopes_not_written':12,'source_only_and_unowned_cases':4,'actual_config_source_mutated':False,'scope_override_only_for_handmade_contained_fixture':True}
# Every preexisting terminal directory entry blocks every terminal addition.
with tempfile.TemporaryDirectory(prefix='terminal_entries_',dir=REVIEW) as temp:
 base=Path(temp);tc=copy.deepcopy(c);tc['result_root']=str(base/'outputs');actual_scopes=m.scopes
 def terminal_scopes(_config,parent,stage,run):return actual_scopes(tc,parent,stage,run)
 parent=c['parent_ids'][0]
 def snapshot(directory):
  values={}
  for p in directory.iterdir():
   if p.is_symlink():values[p.name]=('symlink',os.readlink(p))
   elif p.is_dir():values[p.name]=('directory',tuple(sorted(q.name for q in p.iterdir())))
   else:values[p.name]=('regular',p.read_bytes())
  return values
 for route in ('main','execute','finalizer'):
  for terminal in ('receipt.json','exit_status'):
   for kind in ('regular','symlink','dangling_symlink','directory'):
    run='terminal_'+route+'_'+terminal.replace('.','_')+'_'+kind
    scope,out,tmp,log=actual_scopes(tc,parent,'retrieve',run);out.mkdir(parents=True)
    entry=out/terminal
    if kind=='regular':entry.write_text('existing immutable terminal evidence\n')
    elif kind=='directory':entry.mkdir()
    else:
     target=base/(run+'_target')
     if kind=='symlink':target.write_text('existing target bytes\n')
     entry.symlink_to(target)
    rr=write(base/(run+'_review.json'),{'kind':'native_argformer_tsinfer_fit_v5_independent_review','review_role':'independent_reviewer','verdict':'PASS','implementation':m.reference(m.SCRIPT),'config':m.reference(m.CONFIG),'handmade_only':True})
    a=m.admission_bindings(c,parent,'retrieve',run);a.update(parent_storage_scope=str(scope),output=str(out),short_tmp=str(tmp),launch_log=str(log),independent_review=rr,handmade_only=True)
    ar=write(base/(run+'_admission.json'),a)
    write(out/'exec_admission.json',{'implementation_sha256':m.sha(m.SCRIPT),'stage':'retrieve','root_admission':ar,'TMPDIR':str(tmp),'handmade_only':True})
    args=types.SimpleNamespace(execute=True,parent=parent,stage='retrieve',run_id=run,native=True,expected_source_sha256=m.sha(m.SCRIPT),expected_config_sha256=m.CONFIG_SHA256,admission=ar['path'],expected_admission_sha256=ar['sha256'])
    argv=['--execute','--native','--parent',parent,'--stage','retrieve','--run-id',run,'--admission',ar['path'],'--expected-admission-sha256',ar['sha256'],'--expected-source-sha256',args.expected_source_sha256,'--expected-config-sha256',m.CONFIG_SHA256]
    before=snapshot(out)
    check('lstat_presence_'+route+'_'+terminal+'_'+kind,m.terminal_entries_present(out))
    if route=='main':
     with patch.object(m,'scopes',side_effect=terminal_scopes),patch.dict(os.environ,{'TMPDIR':str(tmp)}),patch.object(m,'policy',side_effect=ValueError('handmade real main policy failure')):
      reject('real_main_terminal_entry_blocks_'+terminal+'_'+kind,lambda:m.main(argv))
    elif route=='execute':
     admission={'independent_review':{},'runtime':{},'experiment_registration':{},'prepared_dispatcher':{},'storage':{}}
     real_read=m.read_json
     def fixture_read(record,expected=None):
      if record==ar:return admission
      if record=={}:return {}
      return real_read(record,expected)
     storage=types.SimpleNamespace(check=lambda:{'handmade_only':True})
     with patch.object(m,'scopes',side_effect=terminal_scopes),patch.object(m,'verify_source',return_value=None),patch.object(m,'validate_admission',return_value={}),patch.object(m,'prepared_metadata',return_value={}),patch.object(m,'Storage',return_value=storage),patch.object(m,'read_json',side_effect=fixture_read),patch.object(m,'runtime_preflight') as preflight:
      reject('real_execute_terminal_entry_blocks_'+terminal+'_'+kind,lambda:m.execute(tc,args))
      check('blocked_execute_never_enters_runtime_'+terminal+'_'+kind,not preflight.called)
    else:
     reject('direct_finalizer_terminal_entry_blocks_'+terminal+'_'+kind,lambda:m.terminal_receipt(c,out,{'exit_code':1,'status':'failed','root_actual_execution':False}))
    check('no_terminal_addition_or_overwrite_'+route+'_'+terminal+'_'+kind,snapshot(out)==before)
 details['all_terminal_directory_entry_routes']={'routes':['real main(argv)','execute()','terminal_receipt()'],'terminal_names':['receipt.json','exit_status'],'preexisting_types':['regular file','symlink to regular file','dangling symlink','directory'],'cases':24,'all_additions_and_overwrites_blocked':True}
preserved()
report={'kind':'native_argformer_tsinfer_fit_v5_author_checks','status':'passed','check_count':len(checks),'checks':checks,'details':details,'implementation':m.reference(m.SCRIPT),'config':m.reference(m.CONFIG),'test_source':m.reference(__file__),'actual_scientific_arrays_opened':False,'actual_native_model_fit_executed':False,'GPU_inventory_or_context_used':False,'live_ledger_read':False,'runtime_prefix_modified':False,'handmade_import_metadata_providers_stubbed_only_in_tests':True,'fresh_full_retrieval_kernel_handmade_only':True}
write(REVIEW/'handmade_checks.json',report)
print(json.dumps({'status':'passed','check_count':len(checks),'implementation':report['implementation'],'config':report['config']},indent=2))
