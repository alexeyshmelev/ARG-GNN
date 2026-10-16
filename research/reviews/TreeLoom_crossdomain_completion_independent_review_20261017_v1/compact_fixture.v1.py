"""Independent compact handmade fixtures only. No archive, checkpoint or benchmark data."""
from pathlib import Path
from types import SimpleNamespace
import dataclasses, hashlib, importlib, json, math, sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE=ROOT/'research/reviews/TreeLoom_crossdomain_completion_source_20261017_v1'
REVIEW=ROOT/'research/reviews/TreeLoom_crossdomain_completion_independent_review_20261017_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(SOURCE/'source_manifest.v1.json')=='d4450065a0c6141470ad188c733ca8e8fbef548034fb51144e268c89301c3d84'
assert sha(SOURCE/'source_handoff.v1.json')=='3a58583103f47f828472da768e949681af1af3db0a30348383598d4f46be86b6'
manifest=json.loads((SOURCE/'source_manifest.v1.json').read_text())
for row in manifest['source_files']+manifest['external_source_and_metadata_refs']:
 assert sha(ROOT/row['path'])==row['sha256'],row['path']
prefix='research.reviews.TreeLoom_crossdomain_completion_source_20261017_v1'
completion=importlib.import_module(prefix+'.completion_v1')
assert 'torch' not in sys.modules and 'arggnn.chickenpox' not in sys.modules
checks={'inert_metadata_import':True,'source_and_metadata_pins_authenticated':41}
def rejected(call):
 try:call()
 except ValueError as error:return str(error)
 raise AssertionError('Rejection required')
seals=[dict(seed=s,arm=a,kind='selected_checkpoint',sha256='0'*64)for s in completion.SEEDS for a in completion.ARMS]
checks['incomplete_test_release_rejected']=rejected(lambda:completion.load_test_v1('DO_NOT_OPEN',release_selections=lambda:seals[:-1]))
checks['duplicate_selection_rejected']=rejected(lambda:completion._seal_set(seals[:-1]+[seals[0]],'selected_checkpoint'))
assert 'arggnn.chickenpox' not in sys.modules
class Unreadable:
 phase='test'
 @property
 def counts(self):raise AssertionError('No target read before complete barrier')
checks['incomplete_test_score_rejected_before_targets']=rejected(lambda:completion.score_test_v1({},Unreadable(),release_prediction_seals=lambda:[]))
import numpy as np
import torch
import torch_geometric
torch.set_num_threads(1)
core=importlib.import_module(prefix+'.generic_core_v1')
family=core.matched_pair_v1(101);full,null=family['full'],family['no_memory']
parameters={arm:sum(p.numel()for p in model.parameters())for arm,model in family.items()}
assert parameters=={'full':25378,'no_memory':24865}
assert sum(p.numel()for p in full.memory.parameters())==513
assert all(torch.equal(value,null.state_dict()[name])for name,value in full.state_dict().items()if not name.startswith('memory.'))
checks['parameters']=parameters;checks['common_initial_tensors_identical']=True
x=torch.arange(2*4*3,dtype=torch.float32).reshape(2,4,3,1)/20
edges=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long)
observed=core.ObservedHistoryV1(x,edges,torch.tensor([1.,2.,.5,1.5]),('SYNTHETIC_A','SYNTHETIC_B','SYNTHETIC_C'),(4,5),((0,1,2,3),(1,2,3,4)))
full.eval();null.eval();calls=[]
hook=full.graph_pass.register_forward_hook(lambda *args:calls.append(1))
with torch.no_grad():
 initial=full(observed);comparison=null(observed)
hook.remove()
assert len(calls)==2 and torch.equal(initial,comparison)
checks['shared_block_calls_per_forward']=2;checks['initial_forecasts_identical']=True
with torch.no_grad():
 full.memory.O.weight.copy_(torch.arange(256,dtype=torch.float32).reshape(32,8)/1000)
 up=torch.arange(4*6*32,dtype=torch.float32).reshape(4,6,32)/1000
 outside=up.flip(0)/3
 forcing=full.memory.U(up+outside).numpy();ell=float(full.memory.ell_weeks)
 forward=np.empty_like(forcing);reverse=np.empty_like(forcing)
 for order,destination in [(range(4),forward),(range(3,-1,-1),reverse)]:
  state=None
  for at in order:
   rho=math.exp(-float(observed.durations[at])/ell)
   state=forcing[at].copy()if state is None else rho*state+(1-rho)*forcing[at]
   destination[at]=state
 expected=(.5*(forward+reverse)-forcing)@full.memory.O.weight.numpy().T
 actual=full.memory(up,outside,observed.durations).numpy()
 assert np.allclose(actual,expected,rtol=2e-5,atol=2e-6)
 checks['nonzero_memory_matches_independent_scalar_recurrence']=True
 original=full(observed)
 assert torch.equal(original,full(observed))
 for b in range(2):
  single=dataclasses.replace(observed,x=x[b:b+1],target_indices=observed.target_indices[b:b+1],content_indices=observed.content_indices[b:b+1])
  assert torch.allclose(original[b],full(single)[0],atol=2e-6,rtol=2e-5)
 permutation=torch.tensor([2,0,1]);inverse=torch.argsort(permutation)
 renamed=dataclasses.replace(observed,x=x[:,:,permutation],entity_ids=tuple(observed.entity_ids[i]for i in permutation.tolist()),edge_index=inverse[edges])
 assert torch.allclose(full(renamed),original[:,permutation],atol=2e-6,rtol=2e-5)
checks['no_cross_window_or_batch_state']=True;checks['entity_permutation_equivariant_with_nonzero_memory']=True
# Hand-generated fixed-shape metadata fixture required by the preparation contract.
links=[(i,j)for i in range(20)for j in range(i+1,20)][:41]
raw_edges=np.asarray([(i,i)for i in range(20)]+links+[(j,i)for i,j in links],dtype=np.int64).T
counts=np.arange(522,dtype=np.float64)[:,None]+3*np.arange(20,dtype=np.float64)[None,:]
data=SimpleNamespace(counts=counts,edge_index=raw_edges,counties=completion.COUNTIES,dates=tuple('SYNTHETIC_%03d'%i for i in range(522)),archive_sha256=completion.ARCHIVE_SHA256)
development=completion._prepare_loaded_v1(data,'development')
poison=counts.copy();poison[365:]+=1e6
poisoned=completion._prepare_loaded_v1(dataclasses.replace(data)if dataclasses.is_dataclass(data)else SimpleNamespace(**{**vars(data),'counts':poison}),'development')
assert np.array_equal(development.mean,poisoned.mean)and np.array_equal(development.scale,poisoned.scale)
assert np.isnan(development.counts[443:]).all()and not development.counts.flags.writeable
assert development.edges.shape==(2,82)and np.array_equal(development.native_edges,raw_edges)
validation=completion.history_batch_v1(development,[365])
assert validation.content_indices==(tuple(range(313,365)),)
assert torch.equal(validation.x,completion.history_batch_v1(poisoned,[365]).x)
released=completion._prepare_loaded_v1(data,'test')
future=counts.copy();future[443:]+=1e8
released_poison=completion._prepare_loaded_v1(SimpleNamespace(**{**vars(data),'counts':future}),'test')
history=completion.history_batch_v1(released,[443]);other=completion.history_batch_v1(released_poison,[443])
assert history.content_indices==(tuple(range(391,443)),)and torch.equal(history.x,other.x)
shuffle=completion.history_batch_v1(released,[443],shuffle_prefix=True)
assert shuffle.content_indices[0][-1]==442 and set(shuffle.content_indices[0])==set(history.content_indices[0])and torch.equal(shuffle.x[:,-1],history.x[:,-1])
checks['train_only_scaler_and_target_future_poisoning_pass']=True
checks['development_TEST_tail_mask_readonly_and_exact_history_pass']=True
checks['native_nonself_and_legacy_edge_representations_pass']=True
checks['fixed_last_shuffle_preserves_content_and_past_only']=True
checks['target_counts']={split:len(completion.target_indices_v1(split))for split in ['train','validation','test']}
assert checks['target_counts']=={'train':313,'validation':78,'test':79}
result={'schema':'TreeLoom_crossdomain_independent_compact_fixture_v1','status':'PASS','checks':checks,'runtime_versions':{'torch':torch.__version__,'torch_geometric':torch_geometric.__version__},'limits':'Handmade compact CPU tensors and source/metadata only. No benchmark archive, real observation/target/prediction tensor, checkpoint, guard, admission, training step, resource probe, process supervisor, GPU query or Git. In-memory fake archive identity is a fixture label, never genuine data authentication. This is not full runtime qualification.'}
(REVIEW/'compact_fixture.result.v1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
