"""Bounded H047 synthetic FP32 loss/gradient, unchanged-body and barrier checks."""
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
import copy,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
from arggnn import train_cohort_risk_continuation_v1 as worker
assert "torch" not in worker.__dict__
checks=["worker_import_inert"]
worker.science()
torch,np,core=worker.torch,worker.np,worker.core
torch.set_num_threads(1);torch.manual_seed(101)
def check(name,ok):
    if not bool(ok):raise AssertionError(name)
    checks.append(name)
def rejects(name,fn):
    try:fn()
    except (ValueError,TypeError,KeyError):checks.append(name)
    else:raise AssertionError("Expected rejection: "+name)
cohorts=np.concatenate([np.full(n,k) for k,n in core.COHORT_COUNTS.items()])
present,ancient=core.cohort_columns_v1(cohorts)
check("original_observed_cohort_partition182_504",len(present)==182 and len(ancient)==504
    and len(np.union1d(present,ancient))==686)
bad=cohorts.copy();bad[0]="Yam"
rejects("cohort_counts_changed_rejected",lambda:core.cohort_columns_v1(bad))
logits=torch.randn((2,686,4),dtype=torch.float32,requires_grad=True)
mass=torch.zeros_like(logits);mass[:,:,0]=30.;mass[:,:,1]=70.
legacy=core.exact_span_loss(logits,mass);legacy_grad=torch.autograd.grad(legacy,logits)[0]
control,metrics=core.cohort_objective_v1(logits,mass,present,arm="original_risk")
control_grad=torch.autograd.grad(control,logits)[0]
check("control_loss_exact_existing_FP32_function",torch.equal(control,legacy))
check("control_gradient_exact_existing_FP32_function",torch.equal(control_grad,legacy_grad))
balanced,metrics=core.cohort_objective_v1(logits,mass,present,arm="balanced_risk")
manual=.5*core.exact_span_loss(logits[:,present],mass[:,present])+.5*core.exact_span_loss(logits[:,ancient],mass[:,ancient])
gb=torch.autograd.grad(balanced,logits)[0];gm=torch.autograd.grad(manual,logits)[0]
check("balanced_actual_group_support_loss",torch.allclose(balanced,manual,atol=1e-7,rtol=1e-7))
check("balanced_actual_group_support_gradient",torch.allclose(gb,gm,atol=2e-9,rtol=2e-6))
# Generic unequal query support confirms denominators derive from actual mass.
small=torch.randn((2,6,4),requires_grad=True);weights=torch.tensor([20.,30.,50.,80.,100.,120.])
target=torch.zeros_like(small);target[:,:,2]=weights
loss,components=core.cohort_objective_v1(small,target,np.array([0,1]),arm="balanced_risk")
manual=.5*core.exact_span_loss(small[:,:2],target[:,:2])+.5*core.exact_span_loss(small[:,2:],target[:,2:])
check("normalization_uses_actual_query_bp_not_count_assumption",
    torch.allclose(loss,manual,atol=1e-7) and torch.allclose(components["actual_present_query_bp_weight"],torch.tensor(.125)))
rejects("noninteger_cohort_indexes_rejected",lambda:core.cohort_objective_v1(
    small,target,[0.,1.],arm="balanced_risk"))
# Original-grid/native-hold physical aggregation preserves both objective gradients.
mapping=torch.tensor([0,0,1,1]);widths=torch.tensor([80.,20.,70.,30.])
grid_mass=torch.rand((4,6,4));grid_mass=grid_mass/grid_mass.sum(-1,keepdim=True)*widths[:,None,None]
native_mass=torch.zeros((2,6,4));native_mass.index_add_(0,mapping,grid_mass)
for arm in core.ARMS:
    lo=torch.randn((2,6,4),requires_grad=True)
    native,_=core.cohort_objective_v1(lo,native_mass,[0,1],arm=arm)
    grid,_=core.cohort_objective_v1(lo.index_select(0,mapping),grid_mass,[0,1],arm=arm)
    gn=torch.autograd.grad(native,lo)[0];gg=torch.autograd.grad(grid,lo)[0]
    check("original_grid_native_hold_loss_gradient_"+arm,
        torch.allclose(native,grid,atol=2e-7,rtol=2e-7) and torch.allclose(gn,gg,atol=2e-8,rtol=2e-6))
base=core.AnonymousCladeFlagControlGNNV1(**core.MODEL)
payload=dict(schema="anonymous_clade_flag_checkpoint_v1",hypothesis_id="H042",arm="exact_clade_flags",phase="train",
    pass_index=30,additional_optimizer_steps=960,parameter_count=30788,model_kwargs=core.MODEL,teacher_used=False,
    model_state={k:v.detach().clone() for k,v in base.state_dict().items()})
family,proof=core.make_matched_family_v1(payload,**core.MODEL)
check("exact_H042_class_and30788_trainable_parameters_zero_added",all(type(m) is type(base)
    and sum(p.numel() for p in m.parameters())==30788 and all(p.requires_grad for p in m.parameters())
    for m in family.values()) and proof["added_parameters"]==0)
check("matched_initial_states_and_independent_storage",all(torch.equal(base.state_dict()[k],family[a].state_dict()[k])
    for a in worker.ARMS for k in base.state_dict()) and not(
    {p.data_ptr() for p in family[worker.ARMS[0]].parameters()}&
    {p.data_ptr() for p in family[worker.ARMS[1]].parameters()}))
bad=dict(payload,additional_optimizer_steps=959)
rejects("wrong_warmstart_provenance_rejected",lambda:core.make_matched_family_v1(bad,**core.MODEL))
check("scientific_source_pins_match",all(hashlib.sha256((ROOT/r["path"]).read_bytes()).hexdigest()==r["sha256"]
    for r in worker.SCIENTIFIC_REFS))
class Tiny(torch.nn.Module):
    def __init__(self):super().__init__();self.layer=torch.nn.Linear(4,4)
    def forward_train_holds(self,bank,*,flags,control):
        assert control=="aligned"
        return self.layer(torch.ones((2,686,4)))
models={a:Tiny() for a in worker.ARMS};models[worker.ARMS[1]].load_state_dict(models[worker.ARMS[0]].state_dict())
optimizers={a:torch.optim.AdamW(models[a].parameters(),lr=.001,weight_decay=.0001) for a in worker.ARMS}
bank=SimpleNamespace(parent=torch.tensor([-1]))
steps=[]
for arm in worker.ARMS:
    row=worker.update_v1(models[arm],optimizers[arm],bank,mass,None,present,arm,lambda:None,
        after_optimizer_step=lambda arm=arm:steps.append(arm))
    check("synthetic_update_and_gradient_cleanup_"+arm,row["finite_FP32"]
        and all(p.grad is None for p in models[arm].parameters()))
check("one_actual_synthetic_step_callback_each_arm",tuple(steps)==worker.ARMS)
@dataclass
class TinyBank:
    parent: object
cache=worker.FlagCacheV1(dict(cfg={},guard=lambda:None))
flags=SimpleNamespace(node_flags=torch.zeros((2,2),dtype=torch.uint8))
tiny=TinyBank(torch.tensor([-1,0]))
cache.values["synthetic"]=flags;cache.flag_stamps["synthetic"]=worker.flag_stamp_v1(flags)
original_sha=worker.hashlib.sha256
try:
    worker.hashlib.sha256=lambda *a,**k:(_ for _ in ()).throw(AssertionError("hash called on cache hit"))
    check("cache_hit_uses_stamps_without_content_hash",cache.get("synthetic",tiny,[]) is flags)
finally:worker.hashlib.sha256=original_sha
flags.node_flags.add_(1)
rejects("mutated_cached_flag_tensor_rejected",lambda:cache.get("synthetic",tiny,[]))
class Memory:
    root=ROOT;path=ROOT/"synthetic/H047"
    def __init__(self):self.raw={}
    def save(self,name,value,*args):
        raw=(json.dumps(value,sort_keys=True,separators=(",",":"))+chr(10)).encode()
        record=dict(path="synthetic/H047/"+name,sha256=hashlib.sha256(raw).hexdigest())
        self.raw[record["path"]]=raw;return record
    def read(self,r,**kwargs):
        raw=self.raw[r["path"]];assert hashlib.sha256(raw).hexdigest()==r["sha256"];return raw
    def decode_json(self,r):return json.loads(self.read(r))
memory=Memory();reader=worker.H047PhaseReaderV1(memory,"evaluate",{},virtual_prefix="synthetic/H047/RAM")
parents={pid:dict(parent_id=pid,positions=np.array([50.,150.]),edges=np.array([0.,100.,200.]),
    retained_indexes=np.arange(2),query_slots=np.arange(686),raw_query_nodes=np.arange(686),cohorts=cohorts) for pid in worker.VAL}
banks={pid:SimpleNamespace(readout_hold=torch.tensor([0,0])) for pid in worker.VAL}
ctx=dict(output=memory,cfg=dict(RAM_prediction_cap_bytes=10000000),guard=lambda:None)
ram=worker.DenseRAMMatrixV1(ctx,reader,parents,banks);reader.ram=ram
prob=np.full((2,686,4),.25,dtype=np.float32);cp=dict(path="synthetic/cp.pt",sha256="0"*64)
ram.seal(worker.ARMS[0],worker.PASSES[0],worker.VAL[0],prob,cp)
rejects("incomplete96_matrix_rejected",ram.verify)
for arm in worker.ARMS:
    for at in worker.PASSES:
        for pid in worker.VAL:
            if (arm,at,pid) not in ram.values:ram.seal(arm,at,pid,prob,cp)
ram.verify();check("all96_original_grid_products_sealed_before_truth",ram.verified and ram.bytes==96*prob.nbytes)
truth=dict(path="synthetic/truth/full_truth_intervals.npz",sha256=hashlib.sha256(b"x").hexdigest())
memory.raw[truth["path"]]=b"x"
phase=worker.H047PhaseReaderV1(memory,"evaluate",{worker.VAL[0]:dict(truth_npz=truth)},virtual_prefix="synthetic/else")
rejects("VAL_truth_denied_without_verified_barrier",lambda:phase.read(truth))
phase.truth_gate=ram.verified;check("VAL_truth_gate_only_after96",phase.read(truth)==b"x")
receipt=dict(schema="H047_bounded_synthetic_FP32_CPU_checks_v1",status="passed",checks=checks,check_count=len(checks),
    synthetic_only=True,learned_FP32=True,FP64_loss_or_gradient_checks=False,CUDA_operations=False,
    raw_data_checkpoint_or_ancestry_truth_arrays_read=False,registered_training_or_admission=False,
    synthetic_stub_optimizer_updates_per_arm=1,
    core_sha256=hashlib.sha256((ROOT/worker.CORE_REF["path"]).read_bytes()).hexdigest(),
    worker_sha256=hashlib.sha256((ROOT/worker.SOURCE).read_bytes()).hexdigest(),
    check_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
print(json.dumps(receipt,indent=2))
if "--write-receipt" in sys.argv:
    Path(__file__).with_name("cpu_check_receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+chr(10))
