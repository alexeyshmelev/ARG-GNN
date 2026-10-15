"""H046 focused worker CPU smoke with small synthetic model and in-memory seals."""
from pathlib import Path
from types import SimpleNamespace
import copy,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
from arggnn import train_continuous_clade_flank_v1 as worker
assert "torch" not in worker.__dict__
import numpy as np
import torch
torch.set_num_threads(1)
checks=["worker_import_inert"]
def check(name,ok):
    if not bool(ok):raise AssertionError(name)
    checks.append(name)
def rejects(name,fn):
    try:fn()
    except (ValueError,TypeError,KeyError):checks.append(name)
    else:raise AssertionError("Expected rejection: "+name)
cfg=dict(schema="H046_continuous_clade_flank_execution_config_v1",hypothesis_id="H046",arms=list(worker.ARMS),
    linked_hypothesis_ids=["H042","H044","H045"],seed=101,dtype="float32",passes=30,
    updates_per_pass=32,updates_per_arm=960,checkpoint_passes=list(worker.PASSES),
    TRAIN_parent_ids=list(worker.TRAIN),VAL_parent_ids=list(worker.VAL),
    execution_mode="paired_sequential_one_parent",parent_order="shuffle_seed101_plus_pass",
    model=worker.MODEL,readout=worker.READOUT,parameters_per_arm=31165,base_parameters=30788,added_parameters=377,
    learning_rate=.001,weight_decay=.0001,AdamW_betas=[.9,.999],AdamW_epsilon=1e-8,gradient_clip_norm=1.,
    early_stopping=False,first_step_qualification="within_continuous_training_no_extra_updates",
    VAL_influences_optimization=False,same_VAL_is_development_evidence=True,
    selection=worker.SELECTION,metrics=worker.METRICS,core_ref=worker.CORE_REF,
    scientific_refs=list(worker.SCIENTIFIC_REFS),registration_ref=worker.REGISTRATION_REF,design_ref=worker.DESIGN_REF,
    warmstart_checkpoint_ref=worker.WARM_CHECKPOINT,warmstart_completion_ref=worker.WARM_COMPLETION,
    TRAIN_target_policy="one_complete_original100bp_parent_mass_streamed_once_shared_between_arms",
    invariance_scope="fixed_field_original_focal_support_episode_membership_and_positions_only",
    dense_prediction_arrays_saved=False,flank_arrays_saved_to_disk=False,outer_truth_read=False,
    practical_accuracy_gain_threshold=.001,practical_boundary_F1_loss_tolerance=.01,
    practical_criteria_aggregation="equal_parent_mean_Present182_across8VAL",
    local_callability_strata=["0","1-2","3-5",">=6"],
    local_callability_definition="min_usable_SNPs_left_right_within_original_focal_hold",
    local_callability_diagnostics="selected_models_only_after_complete96_seals_truth_barrier",
    preparation_completion_ref=dict(path="synthetic/preparation.json",sha256="0"*64),
    **{k:10000000 for k in worker.RESOURCE_CAPS})
worker.validate_protocol(cfg);checks.append("exact_registered_protocol_accepted")
for key,value in [("updates_per_arm",961),("parameters_per_arm",31812),("readout",{}),("invariance_scope","wholemodel")]:
    bad=dict(cfg);bad[key]=value
    rejects("protocol_rejects_"+key,lambda bad=bad:worker.validate_protocol(bad))
worker.science()
check("all_frozen_source_pins_match",all(hashlib.sha256((ROOT/ref["path"]).read_bytes()).hexdigest()==ref["sha256"]
      for ref in worker.SCIENTIFIC_REFS))
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(4,4,dtype=torch.float32)
    def forward(self,bundle):
        return self.layer(torch.ones(2,686,4,dtype=torch.float32))
bundle=SimpleNamespace(bank=SimpleNamespace(parent=torch.tensor([-1])),
    markers=SimpleNamespace(bin_widths_bp=torch.tensor([100.,100.],dtype=torch.float32)))
mass=torch.zeros(2,686,4,dtype=torch.float32);mass[:,:,0]=100.
a=Tiny();b=copy.deepcopy(a)
models={"foreign_flank":a,"own_state":b}
optimizers={k:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for k,m in models.items()}
steps=[]
for arm in worker.ARMS:
    row=worker.update_v1(models[arm],optimizers[arm],bundle,mass,lambda:None,
        after_optimizer_step=lambda arm=arm:steps.append(arm))
    check("actual_synthetic_grid_update_finite_"+arm,row["finite_FP32"] and np.isfinite(row["loss"]))
    check("gradients_end_before_next_arm_"+arm,all(p.grad is None for p in models[arm].parameters()))
check("one_actual_update_callback_each_arm",tuple(steps)==worker.ARMS)
check("optimizer_states_independent",optimizers["foreign_flank"].state is not optimizers["own_state"].state)
class Memory:
    root=ROOT
    path=ROOT/"synthetic/H046"
    def __init__(self):self.raw={}
    def save(self,name,value,*args):
        payload=(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()
        ref=dict(path="synthetic/H046/"+name,sha256=hashlib.sha256(payload).hexdigest())
        self.raw[ref["path"]]=payload;return ref
    def read(self,ref,**kwargs):
        data=self.raw[ref["path"]]
        assert hashlib.sha256(data).hexdigest()==ref["sha256"]
        return data
    def decode_json(self,ref):return json.loads(self.read(ref))
memory=Memory()
reader=worker.H046PhaseReaderV1(memory,"evaluate",{},virtual_prefix="synthetic/H046/RAM_packets")
parents={pid:dict(parent_id=pid,positions=np.array([50.,150.]),edges=np.array([0.,100.,200.]),
    retained_indexes=np.arange(2),query_slots=np.arange(686),raw_query_nodes=np.arange(686),
    cohorts=np.full(686,"present_day")) for pid in worker.VAL}
banks={pid:SimpleNamespace(readout_hold=torch.tensor([0,0])) for pid in worker.VAL}
ctx=dict(output=memory,cfg=cfg,guard=lambda:None)
ram=worker.DenseRAMMatrixV1(ctx,reader,parents,banks);reader.ram=ram
p=np.full((2,686,4),.25,dtype=np.float32)
checkpoint=dict(path="synthetic/checkpoint.pt",sha256="0"*64)
at,arm,pid=worker.PASSES[0],worker.ARMS[0],worker.VAL[0]
ram.seal(arm,at,pid,p,checkpoint)
rejects("incomplete_matrix_rejected_before_truth",ram.verify)
rejects("RAM_grid_denied_before_verified_matrix",lambda:ram.grid(ram.records[arm,at,pid],parents[pid]))
for ar in worker.ARMS:
    for pass_index in worker.PASSES:
        for parent_id in worker.VAL:
            if (ar,pass_index,parent_id) not in ram.values:
                ram.seal(ar,pass_index,parent_id,p,checkpoint)
ram.verify()
check("all96_original_grid_products_sealed",ram.verified and len(ram.values)==96
      and ram.bytes==96*p.nbytes)
check("verified_original_grid_shape",ram.grid(ram.records[arm,at,pid],parents[pid]).shape==(2,686,4))
rejects("immutable_seal_replacement_denied",lambda:ram.seal(arm,at,pid,p,checkpoint))
truth=dict(path="synthetic/truth/full_truth_intervals.npz",sha256=hashlib.sha256(b"synthetic").hexdigest())
memory.raw[truth["path"]]=b"synthetic"
phase=worker.H046PhaseReaderV1(memory,"evaluate",{pid:dict(truth_npz=truth)},virtual_prefix="synthetic/other")
rejects("VAL_truth_without_barrier_denied",lambda:phase.read(truth))
phase.truth_gate=ram.verified
check("VAL_truth_gate_after_complete96",phase.read(truth)==b"synthetic")
full={ar:dict(cells=[dict(parent_id=pid,groups={"primary_present_day_182":dict(
    accuracy=.6+(.002 if ar=="foreign_flank" else 0),
    boundaries=dict(f1=.5-(.005 if ar=="foreign_flank" else 0)))}) for pid in worker.VAL])
      for ar in worker.ARMS}
check("registered_joint_practical_criteria",worker.practical_criteria_v1(full)["meets_registered_practical_criteria"])
for row in full["foreign_flank"]["cells"]:row["groups"]["primary_present_day_182"]["boundaries"]["f1"]=.48
check("boundary_loss_failure_not_hidden",not worker.practical_criteria_v1(full)["meets_registered_practical_criteria"])
counts=np.array([[0,1,3,6],[0,2,5,100]],dtype=np.float32)
prob=np.zeros((2,4,4),dtype=np.float32);prob[:,:,0]=1.
mass2=np.zeros_like(prob);mass2[:,:,0]=100.
strata=worker.callability_block_sums_v1(prob,mass2,counts)
check("fixed_callability_strata_partition_exact_support",all(row["denominator_query_bp"]==200 for row in strata.values())
      and sum(row["original_grid_query_cells"] for row in strata.values())==8)
check("callability_exact_span_accuracy_without_floor",all(row["correct_query_bp"]==200
      and row["finite_NLL_sum"]==0 and row["infinite_NLL_query_bp"]==0 for row in strata.values()))
mass2[:,:,0]=0;mass2[:,:,1]=100
strata=worker.callability_block_sums_v1(prob,mass2,counts)
check("stratum_zero_probability_preserves_infinite_NLL",all(row["infinite_NLL_query_bp"]==200 for row in strata.values()))
cells=[dict(parent_id=pid,strata={name:dict(denominator_query_bp=100 if i else 0,
    support_missing=not i,accuracy=.5+i/100 if i else None,nll=1. if i else None,NLL_is_infinite=False)
    for name,_,_ in worker.LOCAL_CALLABILITY_STRATA}) for i,pid in enumerate(worker.VAL)]
summary=worker.aggregate_callability_diagnostics_v1(cells)
check("stratum_missing_support_and_parent_variability_explicit",summary["strata"]["0"]["parents_without_support"]==1
      and summary["strata"]["0"]["accuracy"]["finite_parent_count"]==7
      and summary["strata"]["0"]["accuracy"]["parent_population_std"]>0)
check("boundary_criterion_aggregation_explicit",worker.practical_criteria_v1(full)["aggregation"]
      =="equal_parent_mean_Present182_across8VAL")
# Integration of actual blocked observation-prefix counts with noncontiguous
# Present columns across query chunks; four fixed strata each own100 query-bp.
calls=torch.zeros((12,686),dtype=torch.int8)
for col,number in [(0,0),(31,1),(32,3),(685,6)]:
    calls[:,col]=-1
    calls[:number,col]=0;calls[6:6+number,col]=0
markers=worker.core.MarkerEvidenceBankV1(calls,torch.arange(686),
    torch.arange(12,dtype=torch.float64),torch.tensor([6.],dtype=torch.float64),
    torch.tensor([100.]),torch.tensor([0]),torch.zeros((12,4,4)),torch.ones((12,4),dtype=torch.bool),
    torch.zeros((1,3),dtype=torch.long),torch.zeros((1,3),dtype=torch.long),
    torch.tensor([0]),1,1,"original_allele_cardinality")
observed=SimpleNamespace(validate=lambda:None,markers=markers,
    flanks=SimpleNamespace(window_index=torch.tensor([[0,6,12]],dtype=torch.long)))
selected_columns=np.array([0,31,32,685])
diagnostic_cache=SimpleNamespace(parent_id=worker.VAL[0],columns=selected_columns,
    mass=np.array([[[100.,0,0,0]]*4],dtype=np.float32))
probs=np.zeros((1,686,4),dtype=np.float32);probs[:,:,0]=1.
diagnostic=worker.local_callability_diagnostics_v1(observed,
    dict(parent_id=worker.VAL[0],positions=np.array([6.])),probs,diagnostic_cache,lambda:None)
check("actual_observed_prefix_selected_diagnostics_across_query_chunks",
    all(row["denominator_query_bp"]==100 and row["accuracy"]==1 and row["nll"]==0
        for row in diagnostic["strata"].values()))
receipt=dict(schema="H046_worker_focused_CPU_smoke_v1",status="passed",checks=checks,check_count=len(checks),
    synthetic_only=True,CPU_FP32=True,CUDA_operations=False,production_data_or_checkpoint_reads=False,
    registered_training_executed=False,actual_synthetic_updates_per_arm=1,
    worker_sha256=hashlib.sha256((ROOT/worker.SOURCE).read_bytes()).hexdigest(),
    check_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
print(json.dumps(receipt,indent=2))
if "--write-receipt" in sys.argv:Path(__file__).with_name("worker_cpu_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
