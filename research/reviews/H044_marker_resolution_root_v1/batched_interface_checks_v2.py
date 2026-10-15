"""Bounded bulk decode and actual original target/virtual-seal compatibility check."""
import contextlib,hashlib,io,json,runpy,types
from pathlib import Path
import numpy as np
import torch
from arggnn import train_marker_resolution_readout_v1 as worker
from arggnn import marker_resolution_readout_v1 as core
from arggnn import marker_resolution_memory_bounds_v2 as memory_bounds
from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1,sealed_probabilities_v1

# Reuse the already bounded mock-I/O branch fixture. Its products stay in RAM.
with contextlib.redirect_stdout(io.StringIO()):
    fixture=runpy.run_path("research/reviews/H044_marker_resolution_root_v1/preparation_wiring_checks_v2.py")
preparation=fixture["result"];saved=fixture["saved"];counts={}
class BulkReader:
    def read(self,ref,**kwargs):
        counts[ref["path"]]=counts.get(ref["path"],0)+1
        if ref==preparation["coordinates_ref"]:
            out=io.BytesIO();np.savez_compressed(out,**saved["all40_original_marker_coordinates.npz"]);return out.getvalue()
        if ref==preparation["bindings_ref"]:return json.dumps(saved["all40_original_marker_bindings.json"]).encode()
        if ref==fixture["ctx"]["observation_cache_ref"]:
            return json.dumps(dict(parents={p:dict(array_ref=preparation["parents"][p]["original_cache_array_ref"])
                for p in preparation["parents"]})).encode()
        raise AssertionError("Only synthetic bulk observation bytes")
    def decode_json(self,ref):return json.loads(self.read(ref))
worker.np=np;worker.torch=torch;worker.core=core;worker.memory_bounds=memory_bounds
worker.h42.shared.np=np
ctx=dict(preparation=preparation,observation_cache_ref=fixture["ctx"]["observation_cache_ref"],
    cfg=dict(CPU_coordinate_bulk_cap_bytes=1000000,CPU_binding_bulk_cap_bytes=1000000),guard=lambda:None)
cache=worker.MarkerCacheV1(ctx,BulkReader())
assert len(cache.coordinates)==200 and len(cache.bindings)==40
assert counts[preparation["coordinates_ref"]["path"]]==counts[preparation["bindings_ref"]["path"]]==1
for pid,entry in preparation["parents"].items():
    binding=cache.bindings[pid]
    for key,member in entry["coordinates_member_selector"].items():
        assert core.tensor_identity_v1(cache.coordinates[member])==binding["array_identities"][key]

# Synthetic original target archive: one complete physical2Mb truth interval.
truth=io.BytesIO()
np.savez_compressed(truth,intervals_bp=np.array([[0.,2000000.]],dtype=np.float64),
    query_raw_node_ids=np.arange(686,dtype=np.int64),native_paths_1based=np.ones((1,686),dtype=np.int64),
    source_classes_0based=np.zeros((1,686),dtype=np.int8))
truth_bytes=truth.getvalue()
bindings={p:dict(split="val",truth_npz=dict(path="synthetic/"+p+"/truth/full_truth_intervals.npz",
    sha256=hashlib.sha256(truth_bytes).hexdigest())) for p in worker.VAL}
reads=[]
class BaseReader:
    root=Path("/workspace-SR003.nfs2/ARG-GNN")
    def read(self,ref,**kwargs):
        assert ref in [b["truth_npz"] for b in bindings.values()],"Unknown paths are never intercepted or accepted"
        reads.append(ref["path"]);return truth_bytes
class Output:
    path=Path("/workspace-SR003.nfs2/ARG-GNN/synthetic/H044_evaluate")
    def __init__(self):self.calls=[]
    def save(self,name,value,kind="json"):
        self.calls.append((name,value))
        data=(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()
        return dict(path=str(self.path.relative_to(BaseReader.root)/name),sha256=hashlib.sha256(data).hexdigest())
output=Output()
reader=worker.H044PhaseReaderV1(BaseReader(),"evaluate",bindings,
    virtual_prefix="synthetic/H044_evaluate/H044_RAM_probability_packets")
parents={p:dict(parent_id=p,split="val",positions=np.array([50.,150.,250.]),edges=np.array([0.,100.,200.,300.]),
    retained_indexes=np.arange(3),query_slots=np.arange(686),raw_query_nodes=np.arange(686),
    cohorts=np.array(["present_day"]*182+["Neo"]*504)) for p in worker.VAL}
banks={p:types.SimpleNamespace(readout_hold=torch.tensor([0,0,1])) for p in worker.VAL}
ram=worker.DenseRAMMatrixV1(dict(output=output,cfg=dict(RAM_prediction_cap_bytes=4000000),guard=lambda:None),
    reader,parents,banks);reader.ram=ram
try:reader.read(bindings[worker.VAL[0]]["truth_npz"])
except ValueError:pass
else:raise AssertionError("VAL truth allowed before complete matrix")
for arm in worker.ARMS:
    for at in worker.PASSES:
        for pid in worker.VAL:
            ram.seal(arm,at,pid,np.full((3,686,4),.25,dtype=np.float32),
                dict(path="synthetic/"+arm+".pass_%d.pt"%at,sha256="0"*64))
            if len(ram.values)==95:
                try:ram.verify()
                except ValueError:pass
                else:raise AssertionError("95 products passed96 barrier")
                assert not output.calls and not reads and not ram.verified
assert len(ram.values)==96 and not output.calls and not reads
ref=ram.records[worker.ARMS[0],worker.PASSES[0],worker.VAL[0]]
try:reader.read(dict(ref,sha256="f"*64))
except ValueError:pass
else:raise AssertionError("Changed virtual packet hash accepted")
try:reader.read(dict(path="synthetic/unknown_packet.json",sha256="0"*64))
except AssertionError:pass
else:raise AssertionError("Unknown packet intercepted")
ram.verify()
assert ram.verified and len(output.calls)==1 and not reads
manifest=output.calls[0][1]
assert len(manifest["packets"])==len(manifest["seals"])==96
for cell in manifest["packets"]:
    payload=(json.dumps(cell["packet"],sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    assert hashlib.sha256(payload).hexdigest()==cell["record"]["sha256"]
    assert reader.read(cell["record"])==payload
reader.truth_gate=True
selected=[ram.records[worker.ARMS[0],worker.PASSES[0],p] for p in worker.VAL]
target=FullPanelExternalTargetV1(parents[worker.VAL[0]],bindings[worker.VAL[0]],
    "VAL_after_sealed_predictions",reader,sealed_records=selected)
assert len(reads)==1 and target.intervals.shape==(1,2)
value=sealed_probabilities_v1(ref,parents[worker.VAL[0]],reader)
assert value is ram.values[worker.ARMS[0],worker.PASSES[0],worker.VAL[0]]
assert value.shape==(3,686,4) and not value.flags.writeable
source="src/arggnn/train_marker_resolution_readout_v1.py"
print(json.dumps(dict(schema="H044_batched_observation_virtual_seal_interface_checks_v2",status="passed",
    source_ref=dict(path=source,sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest()),
    checks=["all40 bulk coordinates/bindings decoded once under caps","all200 member identities preserved",
        "exact96 immutable dense products with no per-cell writes","95-product barrier rejects",
        "VAL truth denied before one matrix commit","tampered/unknown virtual packet rejected",
        "one original writer matrix commit contains all96 exact packet hashes",
        "actual original FullPanelExternalTarget accepts8 virtual packet refs aftergate",
        "actual original sealed_probabilities returns current dense immutable grid"],
    prediction_shape=[3,686,4],RAM_probability_bytes=ram.bytes,products=96,matrix_writer_calls=1,
    actual_observation_assets_targets_checkpoint_runtime_guards_GPU_queries=False,
    synthetic_target_archive_only=True,original_target_class_executed=True,native_model_constructed=False,
    real_storage_writer_and_observation_IO_mocked=True),sort_keys=True))
