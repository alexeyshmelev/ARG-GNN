"""Source/JSON-only audit of original C3 all30 matrix/score route. No array/runtime/guard access."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=Path(__file__).resolve().parent
PKG='research/reviews/C3_CPU_checkpoint_schema_repair_source_20261017_v2'
LEGACY='research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1'
BARRIER='research/reviews/research_execution_ROOT_20261017_v23/C3_pretruth_prediction_barrier.root_v1.json'
PINS={};READS=[];TESTS=[];FINDINGS=[]
def require(ok,message):
 if not ok:raise ValueError(message)
def ref(path):
 p=ROOT/path if not Path(path).is_absolute()else Path(path)
 assert p.suffix in ('.py','.json','.md','.sh')and p.resolve()==p and p.is_relative_to(ROOT)
 assert p.stat().st_size<=16*2**20
 raw=p.read_bytes();record={'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest()};PINS[record['path']]=record['sha256'];return record
def read(record):
 assert ref(record['path'])==record,record
 assert Path(record['path']).suffix=='.json'
 READS.append(record['path']);return json.loads((ROOT/record['path']).read_text())
def source(path):ref(path);return (ROOT/path).read_text()
def fn(text,name):return copy.deepcopy(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name))
def functions(text,names,env):
 for name in names:
  n=fn(text,name);exec(compile(ast.Module(body=[n],type_ignores=[]),'<exact source/JSON predicates only>','exec'),env)
 return env
def constant(text,name):
 n=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets))
 def literal(v):
  if isinstance(v,ast.Constant):return v.value
  if isinstance(v,ast.Name):return constant(text,v.id)
  if isinstance(v,ast.BinOp)and isinstance(v.op,ast.Add):return literal(v.left)+literal(v.right)
  if isinstance(v,ast.Dict):return {literal(k):literal(x)for k,x in zip(v.keys,v.values)}
  if isinstance(v,ast.Call)and isinstance(v.func,ast.Name)and v.func.id=='dict'and not v.args:return {k.arg:literal(k.value)for k in v.keywords}
  if isinstance(v,(ast.Tuple,ast.List)):return tuple(literal(x)for x in v.elts)if isinstance(v,ast.Tuple)else[literal(x)for x in v.elts]
  raise ValueError('Nonliteral source constant: '+ast.dump(v))
 return literal(n.value)
def test(name,action):action();TESTS.append({'name':name,'status':'PASS'})
barrier_ref=ref(BARRIER);assert barrier_ref['sha256']=='08104af18de96e5f3cf15d5aca18c5e3495b1488c8654983e65a2b448758d0c5';barrier=read(barrier_ref)
accept_ref=ref('research/reviews/C3_saved600_validator_ROOT_actual_20261016_v1/artifact_acceptance.root_active_v1.json');accept=read(accept_ref);training_ref=accept['training_completion_ref'];training=read(training_ref)
EPOCHS=(5,10,15,20,25,30);VAL=tuple('american_c3_val_%03d'%i for i in range(5));candidates={x['epoch']:x['checkpoint_ref']for x in training['checkpoint_candidates']}
latest_source=source(PKG+'/acceptance_contract_v1.py');legacy_source=source(LEGACY+'/acceptance_contract_v1.py');builder_source=source(PKG+'/prepare_evaluation_root_v1.py');worker_source=source(PKG+'/worker_entry_v1.py')
latest_manifest=read(ref(PKG+'/source_manifest.v1.json'));legacy_manifest=read(ref(LEGACY+'/source_manifest.v1.json'))
for path in ('src/arggnn/american_c3_evaluate_v1.py','src/arggnn/american_c3_gnn_selection_isolated_v2.py','src/arggnn/american_c3_gnn_predict_isolated_v2.py','src/arggnn/american_c3_gnn_train_v1.py','src/arggnn/american_c3_gnn_model_v1.py','src/arggnn/american_c3_gnn_targets_v1.py','src/arggnn/american_c3_gnn_shared30mb_isolated_v2.py'):
 assert ref(path)['sha256']==latest_manifest['source_sha256'][path]
seal_docs={row['seal_ref']['path']:read(row['seal_ref'])for row in barrier['products']}
def snapshot_support():
 assert barrier['authenticated_products']==24 and barrier['required_products']==30 and barrier['all30_barrier_satisfied']is False and barrier['truth_read']is False
 assert barrier['per_parent']==dict(zip(VAL,(4,4,4,6,6)))
 for row in barrier['products']:
  s=seal_docs[row['seal_ref']['path']]
  assert s['prediction_ref']==row['prediction_ref']and s['records']==10000 and s['epoch']==row['epoch']and s['parent_id']==row['parent_id']
  assert s['checkpoint_ref']==candidates[s['epoch']]and s['training_completion_ref']==training_ref and s['protocol_sha256']==training['protocol_sha256']
  assert s['private_truth_opened']is False and s['probabilities_modified']is False
 assert training['epochs']==30 and training['complete_parent_updates']==600 and len(training['trained_parent_ids'])==20
 assert training['train_sequence_lengths_bp']==[2000000]*20 and training['focal_exposures']==6000000
 assert training['VAL_used_for_optimization']is False and training['H043_extension']is None and training['purity_filter']is False
 assert [x['epoch']for x in training['checkpoint_candidates']]==list(EPOCHS)
test('pinned24/30 ROOT snapshot and24 actual seal JSONs match original six-state600-update family; arrays untouched',snapshot_support)

# Capture the real current selector rejection before any guard or accounting call.
current_selector_ref=ref('research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json');current_selector=read(current_selector_ref)
selector_scope=dict(copy=copy,require=require,read=read,ref=ref,SELECTOR_REF=None,SELECTOR_SOURCE_REF=constant(latest_source,'SELECTOR_SOURCE_REF'),ACTIVE_POLICY_REF=constant(latest_source,'ACTIVE_POLICY_REF'),REPAIR_REF={'path':'RAM/not_reached.json','sha256':'0'*64},APPROVED_SELECTOR_SOURCES=PKG+'/approved_selector_sources.v1.json')
functions(latest_source,('select_storage_metadata_v1',),selector_scope)
def current_rejection():
 try:selector_scope['select_storage_metadata_v1'](current_selector_ref)
 except ValueError as e:assert str(e)=='Exact known selected source/manifest/helper digests and explicit ROOT policy'
 else:raise AssertionError('Old source must reject genuine new selector')
 FINDINGS.append({'id':'B1','type':'CURRENT_CHILD_COMPATIBILITY','error':'Exact known selected source/manifest/helper digests and explicit ROOT policy','actual_current_selector_ref':current_selector_ref,'actual_current_source_ref':current_selector['source_ref'],'actual_current_policy_ref':current_selector['policy_ref'],'existing_builder_selector_source_ref':selector_scope['SELECTOR_SOURCE_REF'],'existing_builder_policy_ref':selector_scope['ACTIVE_POLICY_REF'],'guard_or_state_called':False,'fresh_native7_H050_mutations_revoked':True})
test('exact latest selector predicate rejects actual current Relate64 metadata before guard/state/admission',current_rejection)

# Validate every genuine closed plan using the actual current mixed-plan functions,
# but extract only metadata predicates so no module/runtime/guard imports occur.
plan_evidence={};prediction_inputs={};prediction_requests={}
for k,parent in enumerate(VAL):
 tag='C3_full30Mb_eval_ROOT_predict_val_%03d_root01_oct16'%k if k<3 else'C3_full30Mb_eval_ROOT_actual_plan_lineage_checkpoint_schema_predict_val_%03d_root01_oct17'%k
 ir=ref('research/reviews/'+tag+'/input.json');spec=read(ir);request=read(spec['root_stage_request_ref']);prediction_inputs[parent]=(ir,spec);prediction_requests[parent]=request;plan_evidence[parent]=request['plan_evidence']
runtime=prediction_inputs[VAL[4]][1]['refs']['runtime']
legacy_plan_scope=dict(require=require,read=read,VAL=VAL)
functions(legacy_source,('terminal0_v1','plan_ready_v1'),legacy_plan_scope)
plan_scope=dict(ROOT=ROOT,require=require,read=read,ref=ref,VAL=VAL,RUNTIME_REF=runtime,PLAN_ROUTE_REGISTRY=constant(latest_source,'PLAN_ROUTE_REGISTRY'),legacy=SimpleNamespace(plan_ready_v1=legacy_plan_scope['plan_ready_v1']))
functions(latest_source,('plan_producer_lineage_v1','plan_ready_v1'),plan_scope)
plans={}
for parent in VAL:
 def check(parent=parent):
  p=plan_scope['plan_ready_v1'](plan_evidence[parent],parent);plans[parent]=p
  assert p['sequence_length_bp']==30000000 and p['haplotypes']==1600 and p['focal_records']==10000 and p['targets_opened']is False
 test('actual complete closed plan lineage passes metadata predicate: '+parent,check)

# Check actual repaired producer envelopes in the schema-v2 entry and demonstrate
# why the older single-producer builder cannot be used as a fallback.
latest_driver=PKG+'/root_driver_v1.py';latest_entry=PKG+'/worker_entry_v1.py';latest_manifest_path=PKG+'/source_manifest.v1.json'
stage_scope=dict(require=require,read=read,ref=ref,OLD_INPUT_REFS=constant(latest_source,'OLD_INPUT_REFS'),LEGACY_DRIVER_REF=constant(latest_source,'LEGACY_DRIVER_REF'),LEGACY_ENTRY_REF=constant(latest_source,'LEGACY_ENTRY_REF'),LEGACY_MANIFEST_REF=constant(latest_source,'LEGACY_MANIFEST_REF'),DRIVER=latest_driver,ENTRY=latest_entry,MANIFEST=latest_manifest_path,PHASES=('predict_val_003','predict_val_004','matrix','score'),RUNTIME_REF=runtime,SAVED_ACCEPTANCE_REF=accept_ref,manifest_v1=lambda:latest_manifest,legacy=SimpleNamespace(manifest_v1=lambda:legacy_manifest))
functions(latest_source,('stage_source_v1',),stage_scope)
producer_summary={}
for k in (3,4):
 parent=VAL[k];output=prediction_inputs[parent][1]['output_directory'];completion_ref=ref(output+'/completed.json');env=read(completion_ref)
 def check(env=env,k=k):assert stage_scope['stage_source_v1'](env,'predict_val_%03d'%k)==prediction_inputs[VAL[k]][1]
 test('actual repaired completion/input/request/source namespace passes latest stage route: '+parent,check)
 producer_summary[parent]={'completion_ref':completion_ref,'input_ref':env['input_ref'],'root_driver_ref':env['root_driver_ref'],'source_namespace':PKG}
old_builder='research/reviews/C3_full30Mb_eval_admission_readiness_source_20261016_v2/prepare_evaluation_root_v2.py';old_text=source(old_builder)
def older_fallback_rejects():
 parent=VAL[4];env=read(producer_summary[parent]['completion_ref']);wait=read(ref(prediction_inputs[parent][1]['packet_directory']+'/actual_wait.json'))
 fake_outer={'actual_exit_code':0,'actual_terminal_consumed':True}
 c=SimpleNamespace(read=lambda r:env if r==producer_summary[parent]['completion_ref']else fake_outer,terminal0_v1=lambda e,cr:wait,STATUSES={'predict_val_004':env['status']},ref=ref,DRIVER=LEGACY+'/root_driver_v1.py')
 scope=dict(require=require);functions(old_text,('actual_stage_v1',),scope)
 try:scope['actual_stage_v1'](c,{'completion_ref':producer_summary[parent]['completion_ref'],'outer_terminal_ref':{'path':'RAM/ROOT_supplied_consumed0.json','sha256':'0'*64}},'predict_val_004',training_ref)
 except ValueError as e:assert str(e)=='Same reviewed saved600 evaluation producer/training lineage'
 else:raise AssertionError('Old source may not relabel repaired producer')
 FINDINGS.append({'id':'B2','type':'NO_OLDER_PRODUCER_FALLBACK','actual_repaired_driver_ref':env['root_driver_ref'],'older_expected_driver_ref':ref(c.DRIVER),'older_route_rejects_actual_repaired_VAL004':True,'note':'Supporting compatibility finding: the current-child repair must preserve literal old000-002 and schema-v2 actual003/004 producer routes, plus its own fresh matrix/score route; no output migration or producer relabel.'})
test('older admission builder rejects actual repaired VAL004 envelope even with a RAM successful terminal placeholder',older_fallback_rejects)

predict_source=source('src/arggnn/american_c3_gnn_predict_isolated_v2.py');selection_source=source('src/arggnn/american_c3_gnn_selection_isolated_v2.py');train_source=source('src/arggnn/american_c3_gnn_train_v1.py');targets_source=source('src/arggnn/american_c3_gnn_targets_v1.py')
# Pure metadata-only training check from the exact prediction module. No top-level torch import.
meta_scope=dict(require=require,read=read,SCHEMA=constant(train_source,'SCHEMA'),REGISTRY_REF=constant(targets_source,'REGISTRY_REF'),TRAIN=training['trained_parent_ids'],CHECKPOINT_EPOCHS=EPOCHS)
functions(predict_source,('completed_training_v1',),meta_scope)
test('original completed-training schema accepts actual600update metadata without loading checkpoint payloads',lambda:meta_scope['completed_training_v1'](training_ref))
# The actual24-item seal-ref list is rejected before bound() could hash an array.
array_calls=[];matrix_scope=dict(require=require,read=read,CHECKPOINT_EPOCHS=EPOCHS,VAL=VAL,completed_training_v1=meta_scope['completed_training_v1'],bound=lambda r:array_calls.append(r))
functions(predict_source,('seal_validation_matrix_v1',),matrix_scope)
def reject24():
 try:matrix_scope['seal_validation_matrix_v1']([r['seal_ref']for r in barrier['products']],training_completion_ref=training_ref,writer=SimpleNamespace(save=lambda *a:None))
 except ValueError as e:assert str(e)=='Six epochs x original5VAL complete matrix'
 else:raise AssertionError('24/30 must remain blocked')
 assert array_calls==[]
test('exact original matrix entry rejects actual24/30 before any prediction/focal payload binding',reject24)

# No evaluator/selection algorithm is changed; confirm actual call chain, criterion,
# original evaluator pin and absence of checkpoint schema loads in matrix/score.
def scientific_chain():
 worker=ast.parse(worker_source)
 for name in ('matrix_v1','score_v1'):
  n=next(n for n in worker.body if isinstance(n,ast.FunctionDef)and n.name==name)
  assert not any(isinstance(c,ast.Call)and isinstance(c.func,ast.Attribute)and c.func.attr=='load'for c in ast.walk(n))
 assert 'seal_validation_matrix_v1' in ast.unparse(fn(worker_source,'matrix_v1'))
 score=ast.unparse(fn(worker_source,'score_v1'));assert 'validation_matrix_v1(matrix_ref)'in score and 'score_validation_matrix_v1('in score and 'select_original_val_checkpoint_v1('in score
 select=ast.unparse(fn(selection_source,'select_original_val_checkpoint_v1'))
 assert 'mean_per_parent_macro_F1' in select and 'ties_earlier_epoch' in selection_source
 assert 'sum(values[e, p] for p in VAL) / 5' in select or '/ 5' in select
 assert '(-r["mean_per_parent_macro_F1"], r["epoch"])' in select or 'winner = min(table' in select
 assert 'execute as original_score' in selection_source and 'original_C3_external_evaluator_unchanged' in selection_source
 assert ref('src/arggnn/american_c3_evaluate_v1.py')['sha256']=='1ada0ab40fe9f198b89a1953bb7c0688e68f1dd44b34dc118600ac25eefc70d3'
test('actual matrix→original external scoring→fixed equal-parent macroF1/earlier-tie selection chain unchanged; no checkpoint-schema load',scientific_chain)

def final_scope():
 assert not any(k=='torch'or k.startswith(('arggnn','numpy','pandas','composer','omegaconf'))for k in sys.modules)
 for path,sha in list(PINS.items()):assert ref(path)['sha256']==sha,path
 assert not any(Path(p).suffix in ('.pt','.npz','.npy','.parquet','.vcf')for p in PINS)
test('source/JSON-only pins unchanged and no scientific runtime, array, guard or process use',final_scope)
result={'schema':'C3_complete_VAL_scoring_route_source_audit_v1','status':'NOT_READY_CURRENT_CHILD_SOURCE_COMPATIBILITY_BLOCKED','all30_condition':{'barrier_ref':barrier_ref,'authenticated_products':24,'required_products':30,'all30_satisfied':False,'missing':barrier['missing'],'no_early_selection_or_truth':True},'findings':FINDINGS,'checks':TESTS,'check_count':len(TESTS),'five_genuine_closed_plans_metadata_validated':True,'actual_repaired_prediction_producers':producer_summary,'training_ref':training_ref,'training_protocol_sha256':training['protocol_sha256'],'candidate_epochs':list(EPOCHS),'training_condition':'Original AmericanC3WholeTreeGNNV1 saved600 backbone:20x2Mb TRAIN,30epochs,600complete-parent updates,6million focal exposures; not TreeLoom or exact author-paper training','scoring_condition':'Original10000 focal records spanning1000heldout queries per30Mb parent,5parents×6states; original external metrics and greatest equal-parent mean macroF1,earlier exact tie','source_and_metadata_pins':PINS,'JSON_reads':sorted(set(READS)),'test_source_ref':ref(str(Path(__file__).resolve())),'checkpoint_prediction_focal_truth_array_read_or_hash':False,'Torch_tensor_numerical_probe_guard_state_process_GPU_resource_admission_launch_Git_action':False,'execution_authority':False}
with(OUT/'audit_result.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'checks':len(TESTS),'findings':FINDINGS,'all30_satisfied':False,'all5_closed_plans_metadata_passed':True,'result_ref':ref(str(OUT/'audit_result.v1.json'))},sort_keys=True))
