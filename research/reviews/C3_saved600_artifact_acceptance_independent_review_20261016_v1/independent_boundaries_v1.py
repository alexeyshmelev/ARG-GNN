"""Independent saved-state/evidence/30-cell-before-truth checks, fake data only."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');SRC=ROOT/'research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1'
OUT=ROOT/'research/reviews/C3_saved600_artifact_acceptance_independent_review_20261016_v1'
sys.path.insert(0,str(SRC))
import acceptance_contract_v1 as c
import saved_state_core_v1 as core
fs=(SRC/'source_fixture_v1.py').read_text();ft=ast.parse(fs)
nodes=[n for n in ft.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('record','tensor','FakeReader','fixture','run')]
ns=dict(core=core,c=c,hashlib=hashlib,copy=copy)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pure-synthetic-fixture-definitions>','exec'),ns)
checks=[]
def reject(name,fn):
 try:fn()
 except (ValueError,KeyError,TypeError,AssertionError):checks.append(dict(name=name,status='PASS_REJECTED'));return
 raise AssertionError('Unexpected acceptance '+name)
for name,change in [
 ('migration_prefix_other_budget',lambda f:f[1].update(remaining_updates=499)),
 ('optimizer_moment_nonfinite',lambda f:next(iter(f[4].values.values()))['optimizer_state']['state'][0]['exp_avg'].update(finite=False)),
 ('model_dtype_changed',lambda f:next(iter(f[4].values.values()))['model_state']['0'].update(dtype='torch.bfloat16')),
 ('missing_optimizer_parameter_state',lambda f:next(iter(f[4].values.values()))['optimizer_state']['state'].pop(37)),
 ('optimizer_fused_changed',lambda f:next(iter(f[4].values.values()))['optimizer_state']['param_groups'][0].update(fused=True)),
 ('optimizer_learning_rate_changed',lambda f:next(iter(f[4].values.values()))['optimizer_state']['param_groups'][0].update(lr=.002)),
 ('history_truncated_BPTT',lambda f:next(iter(f[4].values.values()))['trainer_state']['history'][0].update(block_reset_or_truncated_BPTT=True)),
 ('focal_exposure_budget_changed',lambda f:next(iter(f[4].values.values()))['trainer_state'].update(focal_exposures=1)),
 ('next_epoch_cursor_changed',lambda f:next(iter(f[4].values.values()))['trainer_state'].update(next_epoch=30)),
 ('RNG_device_kind_changed',lambda f:next(iter(f[4].values.values()))['trainer_state']['rng_state'].update(device_type='cpu')),
 ('TRAIN_source_true_ARG',lambda f:f[5][f[2]['parents'][core.TRAIN[0]]['observation_ref']['path']].update(genealogy_kind='SIMULATED_TRUE_ARG')),
 ('TRAIN_external_targets_majority_fallback',lambda f:f[3]['parents'][core.TRAIN[0]].update(nearest_or_majority_fallback=True)),
 ]:
 f=ns['fixture']();change(f);reject(name,lambda f=f:ns['run'](f))
f=ns['fixture']();result=ns['run'](f)
assert len(f[4].reads)==len(f[4].finishes)==6 and result['optimizer_updates_performed']==0 and result['physical_work_exactly600_claimed'] is False
checks.append(dict(name='all_six_fake_states_and20_parent_provenance_no_optimizer_or_truth_work',status='PASS'))
# Acceptance must retain exact original-history JSON types, not relabel any old exit.
for k,v in [('producer_cli_exit_code',0),('producer_cli_exit_code',True),('original_outer_74136_terminal_consumed',True),
 ('original_worker_drain_manifest',dict(path='SYNTHETIC',sha256='0'*64)),('actual_parent_release_count',520),
 ('observed_failure_cause_asserted',True)]:
 bad=copy.deepcopy(c.ORIGINAL_HISTORY);bad[k]=v
 reject('history_rewrite_'+k,lambda bad=bad:c.history_values_v1(bad))
c.history_values_v1(copy.deepcopy(c.ORIGINAL_HISTORY))
# Real original 30-cell validation and scoring functions with explicit fake metadata,
# hash-binding and scorer. A resolver event must follow all60 payload-ref binds.
r=ns['record'];training=f[0];trainingref=r('SYNTHETIC_training.json')
matrixref=r('SYNTHETIC_matrix.json');seals={};cells=[]
for candidate in training['checkpoint_candidates']:
 for parent in c.VAL:
  stem='SYNTHETIC_'+parent+'_'+str(candidate['epoch'])
  sealref=r(stem+'_seal.json');prediction=r(stem+'_prediction.npz');focal=r('SYNTHETIC_'+parent+'_focals.npz')
  seal=dict(status='sealed_C3_original_heldout_predictions',split='val',epoch=candidate['epoch'],parent_id=parent,
   prediction_ref=prediction,focal_registry_ref=focal,checkpoint_ref=candidate['checkpoint_ref'],records=10000,
   focal_identity_sha256='f'*64,training_completion_ref=trainingref,protocol_sha256=training['protocol_sha256'],
   private_truth_opened=False,purity_filter=False)
  seals[sealref['path']]=seal
  cells.append(dict(epoch=candidate['epoch'],parent_id=parent,prediction_seal_ref=sealref,prediction_ref=prediction,
   focal_registry_ref=focal,checkpoint_ref=candidate['checkpoint_ref'],focal_identity_sha256='f'*64))
matrix=dict(status='sealed_C3_all30_original_VAL_predictions',training_completion_ref=trainingref,
 protocol_sha256=training['protocol_sha256'],validation_parent_ids=list(c.VAL),checkpoint_epochs=list(core.EPOCHS),
 all_predictions_sealed_before_VAL_truth=True,metrics='original_C3_external_evaluator_unchanged',cells=cells)
source=(ROOT/'src/arggnn/american_c3_gnn_selection_isolated_v2.py').read_text()
functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('validation_matrix_v1','score_validation_matrix_v1')]
def score_case(value,corrupt_path=None):
 events=[]
 def read(record):
  if record==matrixref:return value
  return seals[record['path']]
 def bound(record):
  if record['path']==corrupt_path:raise ValueError('SYNTHETIC_SHA_MISMATCH')
  events.append(('bound',record['path']))
 def resolve(parent):events.append(('truth',parent));return r('SYNTHETIC_TRUTH_'+parent+'.npz')
 def scorer(cfg,writer):events.append(('score',cfg['prediction_ref']['path']));return r('SYNTHETIC_score.json')
 class Writer:
  def save(self,name,value):return value
 space=dict(read=read,completed_training_v1=lambda record:training,require=core.require,
  VAL=c.VAL,CHECKPOINT_EPOCHS=core.EPOCHS,bound=bound,EVALUATOR_DEFAULTS={},METHOD='SYNTHETIC',
  _ExternalScoreWriter=lambda *a,**kw:Writer(),original_score=scorer)
 exec(compile(ast.Module(body=copy.deepcopy(functions),type_ignores=[]),'<actual-original-selection-functions>','exec'),space)
 try:out=space['score_validation_matrix_v1'](matrixref,writer=Writer(),resolve_val_truth=resolve,runtime_guard=lambda *a:None)
 except (ValueError,KeyError,TypeError,StopIteration):
  assert not any(x[0]=='truth' for x in events)
  return False,events
 assert out['TEST_truth_resolved'] is False and out['VAL_used_for_optimization'] is False
 return True,events
okay,events=score_case(copy.deepcopy(matrix))
assert okay and [x[0] for x in events]==['bound']*60+['truth']*5+['score']*30
checks.append(dict(name='all30_cell_prediction_and_registry_binds_before_any5_truth_resolvers',status='PASS'))
for name,mutate,corrupt in [
 ('missing_cell',lambda v:v['cells'].pop(),None),
 ('duplicate_cell',lambda v:v['cells'].__setitem__(-1,copy.deepcopy(v['cells'][0])),None),
 ('wrong_candidate_checkpoint',lambda v:v['cells'][0].update(checkpoint_ref=r('WRONG.pt')),None),
 ('wrong_parent_focal_registry',lambda v:v['cells'][0].update(focal_registry_ref=r('WRONG.npz')),None),
 ('corrupt_last_prediction_SHA',lambda v:None,cells[-1]['prediction_ref']['path']),
 ('corrupt_last_focal_registry_SHA',lambda v:None,cells[-1]['focal_registry_ref']['path'])]:
 value=copy.deepcopy(matrix);mutate(value);okay,events=score_case(value,corrupt)
 assert not okay and not any(x[0]=='truth' for x in events)
 checks.append(dict(name=name+'_blocks_all_truth',status='PASS_REJECTED'))
# Independent exact numerical-function comparison and supervisor reuse source selection.
def fun(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
old=fun(ROOT/'research/reviews/C3_full30Mb_evaluation_readiness_source_20261016_v1/worker_entry_v1.py');new=fun(SRC/'worker_entry_v1.py')
assert all(old[n]==new[n] for n in ('predict_v1','matrix_v1','score_v1'))
checks.append(dict(name='prediction_matrix_score_numerical_function_AST_exact',status='PASS_SOURCE_CHECK'))
assert not any(n in sys.modules for n in ('torch','numpy','torch_geometric','composer'))
v=dict(schema='C3_saved600_independent_boundaries_and_truth_order_v1',status='PASS_SOURCE_ONLY',checks=checks,
 actual_checkpoint_truth_or_prediction_payload_access=False,
 actual_guard_probe_accounting_activation_training_validation_or_scoring=False)
(OUT/'independent_boundaries.v1.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status='PASS',checks=len(checks))))
