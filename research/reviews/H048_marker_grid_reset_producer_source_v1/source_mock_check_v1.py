"""Source-only fake-schedule and synthetic seal-barrier checks. No Torch/models."""
from pathlib import Path
from types import SimpleNamespace
import ast,contextlib,copy,hashlib,io,json,sys
import numpy as np
from arggnn import train_marker_grid_reset_chain_v1 as d

HERE=Path("research/reviews/H048_marker_grid_reset_producer_source_v1")
assert "torch" not in sys.modules
def ref(path):
    return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
def denied(call):
    try:call()
    except (ValueError,KeyError):return
    raise AssertionError("Expected denial")
cfg=d.fixed_protocol_v1()|dict(execution_enabled=True,producer_ref=ref(d.SOURCE),
    preparation_completion_ref=dict(path="synthetic/prepared.json",sha256="0"*64))
cfg.update({k:10**12 for k in d.RESOURCE_CAPS})
d.validate_protocol(cfg)
for key,value in dict(updates_per_arm=959,parameters_per_arm=31165,passes=29,
        first_step_qualification="disposable",execution_enabled=False).items():
    other=copy.deepcopy(cfg);other[key]=value
    denied(lambda:d.validate_protocol(other))
other=copy.deepcopy(cfg);other["utility_criteria"]["primary_boundary_F1_delta_minimum"]=-.01
denied(lambda:d.validate_protocol(other))
for record in (*d.SCIENTIFIC_REFS,d.REGISTRATION_REF,d.DESIGN_REF):d.bound(record)

class Writer:
    def __init__(self):self.saved={};self.path=d.ROOT/"synthetic/H048"
    def save(self,name,value,kind="json"):
        assert name not in self.saved
        self.saved[name]=(copy.deepcopy(value),kind)
        return dict(path="synthetic/H048/"+name,sha256=hashlib.sha256(name.encode()).hexdigest())
class BaseReader:
    root=d.ROOT
    def __init__(self):self.calls=[]
    def read(self,record,**kwargs):
        self.calls.append(record);return b"{}"

# No scientific optimizer runs: these objects only expose producer interfaces.
class DummyState:
    def detach(self):return self
    def cpu(self):return self
class DummyModel:
    def parameters(self):return [SimpleNamespace(grad=None)]
    def state_dict(self):return {"synthetic":DummyState()}
class DummyOptimizer:
    def state_dict(self):return {"synthetic":True}
proofs={p:dict(logical_bounds=dict(native_nodes=7)) for p in d.TRAIN}
bundle=SimpleNamespace(bank=SimpleNamespace(parent=SimpleNamespace(numel=lambda:7),spans=[1,2]))
class Cache:
    def __init__(self,ctx,reader):
        self.proofs=proofs;self.peak=100;self.builds=32;self.hits=0;self.markers=SimpleNamespace(peak=50)
    def get(self,pid,bank,parent):return bundle
calls=[];warm=[];phase=[]
def observations(ctx,reader,ids,cache):
    assert not reader.truth_gate and tuple(ids)==d.TRAIN
    phase.append(reader)
    return {p:object() for p in ids},{p:object() for p in ids},{"observation_barrier":True}
def warmstart(reader):
    assert not reader.truth_gate
    warm.append(True);return {"synthetic":True}
def update(model,optimizer,b,m,guard,*,qualify,after_optimizer_step):
    assert phase[-1].truth_gate
    calls.append(dict(model=id(model),qualify=qualify))
    after_optimizer_step()
    return dict(loss=.2,gradient_norm=.1,seconds=0.,finite_FP32=True)
originals={name:getattr(d,name) for name in ("MarkerGridCacheV1","load_observations_v1","build_family_v1",
    "device_parent_v1","update_v1","_extra_bytes")}
oldwarm,oldmass=d.h46.load_warmstart_v1,d.h44.original_mass_v1
d.MarkerGridCacheV1=Cache;d.load_observations_v1=observations
d.build_family_v1=lambda *a:({k:DummyModel() for k in d.ARMS},
    {k:DummyOptimizer() for k in d.ARMS},{"synthetic":True},{k:"matched" for k in d.ARMS})
d.device_parent_v1=lambda *a:(bundle,object())
d.update_v1=update;d._extra_bytes=lambda x:100
d.h46.load_warmstart_v1=warmstart
d.h44.original_mass_v1=lambda *a:object()
writer=Writer()
ctx=dict(cfg=cfg,output=writer,guard=lambda:None,device=object(),reader=BaseReader(),
    bindings={p:dict(truth_npz=dict(path="truth/"+p,sha256="0"*64)) for p in d.TRAIN},
    config_ref={"synthetic":True},preparation_completion_ref=cfg["preparation_completion_ref"])
try:
    with contextlib.redirect_stdout(io.StringIO()):
        train=d.run_train_v1(ctx,{"synthetic_input":True})
    assert train["steps"]=={a:960 for a in d.ARMS} and len(calls)==1920
    assert sum(r["qualify"] for r in calls)==2 and len(warm)==1
    assert len(train["history"])==30 and set(train["checkpoint_groups"])=={"5","10","15","20","25","30"}
    assert len([v for v,k in writer.saved.values() if k=="torch"])==12
    for value,kind in writer.saved.values():
        if kind=="torch":
            assert value["hypothesis_id"]=="H048" and value["parameter_count"]==31933
            assert value["additional_optimizer_steps"]==32*value["pass_index"]
            assert value["producer_ref"]==cfg["producer_ref"]
    # A failure after the genuine step callback retains its actual counted step.
    def failing(model,optimizer,b,m,guard,*,qualify,after_optimizer_step):
        after_optimizer_step();raise RuntimeError("synthetic after-step failure")
    d.update_v1=failing;failed_writer=Writer();badctx=ctx|dict(output=failed_writer)
    try:d.run_train_v1(badctx,{"synthetic_input":True})
    except RuntimeError:pass
    else:raise AssertionError("Expected fake failure")
    failed=failed_writer.saved["failed.json"][0]
    assert failed["logical_steps"]=={d.ARMS[0]:1,d.ARMS[1]:0} and failed["automatic_replay"] is False
finally:
    for name,value in originals.items():setattr(d,name,value)
    d.h46.load_warmstart_v1=oldwarm;d.h44.original_mass_v1=oldmass

# Actual NumPy synthetic seals exercise every2x6x8 key with no truth bytes.
d.np=np;d.h42.shared.np=np
bindings={p:dict(truth_npz=dict(path="synthetic/truth/"+p+".npz",sha256="0"*64)) for p in d.VAL}
reader=d.H048PhaseReaderV1(BaseReader(),"evaluate",bindings,virtual_prefix="synthetic/H048/RAM")
parents={p:dict(parent_id=p,positions=np.arange(3,dtype=np.float64),
    edges=np.arange(4,dtype=np.float64),retained_indexes=np.arange(3),
    query_slots=np.arange(686),raw_query_nodes=np.arange(686),cohorts=np.zeros(686,dtype=np.int64)) for p in d.VAL}
banks={p:SimpleNamespace(readout_hold=SimpleNamespace(numpy=lambda:np.zeros(3,dtype=np.int64))) for p in d.VAL}
ram=d.DenseRAMMatrixV1(dict(cfg=cfg,output=Writer(),guard=lambda:None),reader,parents,banks);reader.ram=ram
keylist=[(a,at,p) for a in d.ARMS for at in d.PASSES for p in d.VAL]
value=np.full((3,686,4),.25,dtype=np.float32)
truth=bindings[d.VAL[0]]["truth_npz"]
denied(lambda:reader.read(truth));assert reader.base.calls==[]
for a,at,p in keylist[:-1]:ram.seal(a,at,p,value,dict(path="synthetic/cp",sha256="1"*64))
denied(ram.verify);denied(lambda:reader.read(truth));assert not ram.verified
a,at,p=keylist[-1];ram.seal(a,at,p,value,dict(path="synthetic/cp",sha256="1"*64))
ram.verify();assert ram.verified and len(ram.values)==96 and not reader.truth_gate
denied(lambda:reader.current_probability_seal_v1(ram.records[keylist[0]],parents[d.VAL[0]]))
reader.truth_gate=True
assert reader.current_probability_seal_v1(ram.records[keylist[0]],parents[d.VAL[0]]).shape==(3,686,4)
denied(lambda:ram.seal(*keylist[0],value,{}))
denied(lambda:ram.grid(ram.records[keylist[0]],parents[d.VAL[1]]))
# Verify content mutation is detected even if an external NumPy owner changes flags.
v=ram.values[keylist[0]];v.setflags(write=True);v[0,0,0]=.3;v.setflags(write=False)
denied(lambda:ram.grid(ram.records[keylist[0]],parents[d.VAL[0]]))

def full(delta,boundary):
    return {a:dict(cells=[dict(parent_id=p,groups=dict(primary_present_day_182=dict(
        accuracy=.9+(delta if a==d.ARMS[0] else 0),
        boundaries=dict(f1=.7+(boundary if a==d.ARMS[0] else 0))))) for p in d.VAL]) for a in d.ARMS}
assert d.practical_criteria_v1(full(.002,0))["meets_primary_utility"]
assert not d.practical_criteria_v1(full(.002,-.001))["meets_primary_utility"]
assert d.practical_criteria_v1(full(-.0004,.02))["meets_secondary_boundary_utility"]
assert not d.practical_criteria_v1(full(-.0006,.02))["meets_secondary_boundary_utility"]
assert "torch" not in sys.modules
ast.parse(Path(d.SOURCE).read_text())
receipt=dict(schema="H048_producer_source_mock_check_v1",status="passed_source_and_synthetic_seal_checks",
    producer_ref=ref(d.SOURCE),prototype_ref=d.CORE_REF,registration_ref=d.REGISTRATION_REF,
    actual_scientific_payloads_opened=False,actual_checkpoint_or_truth_opened=False,
    real_model_created=False,actual_optimizer_updates=0,GPU_used=False,
    inherited_sources_bound=len(d.SCIENTIFIC_REFS),mocked_step_calls_per_arm=960,
    six_checkpoint_groups_and12_snapshots=True,first_qualification_only_two_scheduled_mock_calls=True,
    real_update_delegates_unchanged_H046=True,after_step_failure_retains_count=True,
    synthetic_all96_seals_before_truth=True,missing96th_seal_denied=True,
    immutable_same_parent_RAM_packet_checks=True,registration_utility_thresholds_exact=True,
    source_import_Torch_free=True,full_model_real_checkpoint_runtime_qualification="root_pending")
(HERE/"source_mock_check_receipt_v1.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
