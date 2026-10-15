"""One focused handmade FP32 source check; no real data, optimizer steps or GPU."""
from pathlib import Path
from types import SimpleNamespace
import ast, hashlib, io, json, math, sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1 import direct_readout_cut_memory_v1 as m
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1 import integration_hooks_v1 as hooks
from arggnn.ordered_clade_episode_gnn_v1 import build_clade_episodes_v1

def rejected(call):
    try:
        call()
    except ValueError:
        return True
    raise AssertionError("Expected rejection")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(101)
# Reuse only the existing handmade bank-construction statements, not its tests.
fixture = ROOT/"research/reviews/order2_cut_memory_implementation_20261015_v1/check_CPU_v1.py"
nodes = ast.parse(fixture.read_text()).body
start = next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign)
    and isinstance(n.targets[0],ast.Tuple) and [x.id for x in n.targets[0].elts] == ["H","T"])
stop = next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign)
    and isinstance(n.targets[0],ast.Name) and n.targets[0].id == "bank")
space = dict(torch=torch, math=math, MutationRoutedWholeTreeInputsV1=m.MutationRoutedWholeTreeInputsV1)
exec(compile(ast.Module(body=nodes[start:stop+1],type_ignores=[]),str(fixture),"exec"),space)
bank = space["bank"]
episodes, episode_meta = build_clade_episodes_v1(bank)
cut, cut_meta = m.memory_core.prepare_cut_plan_v1(bank)
arrays, recipe = m.codec.encode_v1(cut, cut_meta)
stream = io.BytesIO(); np.savez_compressed(stream,**arrays)
bundle = m.bind_compact_recipe_v1(bank,episodes,episode_meta,stream.getvalue(),recipe)
assert bundle.validate() is bank
base = m.OrderedCladeEpisodeGNNV1(**m.MODEL_KWARGS,arm="own_state")
# Handmade trained-like nonzero common adapter, not an actual checkpoint payload.
with torch.no_grad():
    base.episode_coupling.O.weight.copy_(torch.linspace(-.02,.03,256).reshape(32,8))
family = m.family_from_base_state_v1(base.state_dict(),checkpoint_memory=False)
assert type(family[m.CONTINUATION]) is m.OrderedCladeEpisodeGNNV1
assert not hasattr(family[m.CONTINUATION],"memory")
assert {a:sum(p.numel() for p in x.parameters()) for a,x in family.items()} == m.PARAMETERS
outputs = {}
with torch.no_grad():
    for arm,model in family.items():
        model.eval()
        outputs[arm] = model.forward_train_holds(bank,
            episodes=m.episode_argument_v1(arm,bundle))
        assert outputs[arm].shape == (7,686,4)
        assert outputs[arm].dtype == torch.float32 and bool(torch.isfinite(outputs[arm]).all())
assert all(torch.equal(value,outputs[m.CONTINUATION]) for value in outputs.values())
# One backward with a manually opened synthetic residual checks the actual splice.
chosen = family[m.MEMORY_ARMS[0]]
with torch.no_grad():chosen.memory.O.weight.copy_(torch.linspace(-.03,.04,256).reshape(32,8))
chosen.train()
value = chosen.forward_train_holds(bank,episodes=bundle)
(value[...,0]*torch.arange(1,8,dtype=torch.float32)[:,None]).mean().backward()
gradients = {name:float(parameter.grad.abs().sum()) for name,parameter in chosen.named_parameters()
    if name in ("memory.U.weight","memory.O.weight","memory.ell_theta","episode_coupling.O.weight")}
assert set(gradients)=={"memory.U.weight","memory.O.weight","memory.ell_theta","episode_coupling.O.weight"}
assert all(x>0 and math.isfinite(x) for x in gradients.values())
assert rejected(lambda:m.validate_arm_model_v1(m.CONTINUATION,chosen))
# Bound input mutation must be rejected, then restore a fresh handmade bundle.
with torch.no_grad():bundle.episodes.member_hold.add_(0)
assert rejected(bundle.validate)

class RAMMetadata:
    def __init__(self):self.values={}
    def save(self,name,packet):
        payload=json.dumps(packet,sort_keys=True,separators=(",",":"))
        record=dict(path="handmade/"+name,sha256=hashlib.sha256(payload.encode()).hexdigest())
        self.values[record["path"]]=json.loads(payload)
        return record
    def decode_json(self,record):return self.values[record["path"]]
ram=RAMMetadata()
parents={p:dict(parent_id=p,positions=np.arange(2),edges=np.arange(3),retained_indexes=np.arange(2),
    query_slots=np.arange(686),raw_query_nodes=np.arange(686),cohorts=np.zeros(686,dtype=np.int64)) for p in hooks.VAL}
banks={p:SimpleNamespace(spans=torch.ones(1),readout_hold=torch.zeros(2,dtype=torch.long)) for p in hooks.VAL}
checkpoints={str(at):{a:dict(path="handmade/%s_%d.pt"%(a,at),sha256="0"*64)
    for a in m.ARMS} for at in hooks.PASSES}
matrix=hooks.H050RAMMatrixV1(ram,ram,parents,banks,dict(RAM_prediction_cap_bytes=4*686*4*192+65536),checkpoint_groups=checkpoints)
keys=sorted(hooks.expected_product_keys_v1());assert len(keys)==192
value=np.full((1,686,4),.25,dtype=np.float32)
for at,arm,pid in keys[:-1]:matrix.seal(at,arm,pid,value,checkpoints[str(at)][arm])
assert rejected(matrix.verify)
first=keys[0]
assert rejected(lambda:matrix.grid(matrix.records[first],parents[first[2]]))
at,arm,pid=keys[-1]
assert rejected(lambda:matrix.seal(at,arm,pid,value,dict(path="wrong",sha256="f"*64)))
matrix.seal(at,arm,pid,value,checkpoints[str(at)][arm])
barrier=matrix.verify()
assert matrix.grid(matrix.records[first],parents[first[2]]).shape==(2,686,4)
assert rejected(lambda:matrix.seal(at,arm,pid,value,checkpoints[str(at)][arm]))
metrics={a:dict(accuracy=.96,boundary_f1=.18) for a in m.ARMS}
metrics[m.MEMORY_ARMS[0]]=dict(accuracy=.962,boundary_f1=.181)
assert hooks.practical_criteria_v1(metrics)["meets_primary"]
metrics[m.CONTINUATION]["accuracy"] = .963
assert not hooks.practical_criteria_v1(metrics)["meets_primary"]
assert not torch.cuda.is_initialized()
receipt=dict(schema="H050_one_focused_handmade_FP32_source_check_v1",status="PASS",
    fixture_source_ref=dict(path=str(fixture.relative_to(ROOT)),sha256=hashlib.sha256(fixture.read_bytes()).hexdigest()),
    reused_bank_construction_source_lines=[nodes[start].lineno,nodes[stop].end_lineno],
    exact_original_control_class=True,no_unused_control_parameters=True,parameter_counts=m.PARAMETERS,
    four_initial_functions_bitwise_equal_on_handmade_bank=True,corrected_codec_v2_handmade_roundtrip=True,
    actual_memory_splice_positive_finite_gradient_L1=gradients,bound_mutation_rejected=True,
    missing191barrier_rejected=True,prebarrier_grid_rejected=True,wrong_checkpoint_rejected=True,
    all192_handmade_seals_required=True,duplicate_after_seal_rejected=True,
    continuation_in_practical_comparisons=True,wrong_arm_model_rejected=True,real_payloads_or_truth_read=False,
    actual_checkpoint_authentication_tested=False,optimizer_steps=0,scientific_training_qualification=False,
    GPU_initialized=False,FP64_campaign=False,guard_job_or_Git=False)
with (HERE/"handmade_FP32_check_receipt_v2.json").open("x") as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps(receipt,indent=2,sort_keys=True))
