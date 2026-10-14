"""H040 fixed paired TRAIN and frozen native-hold RAM-sealed VAL driver.

No execution/admission by default. Root supplies an independently pinned context
factory that owns the existing resource guard, reader, writer and leased device.
The H040 mechanism modules and all H039 sources remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import fields, replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "src/arggnn/H040_latent_clade_execution_v1.py"
HERE = "research/reviews/H040_latent_clade_execution_source_v1"
ARMS = ("tensor", "event_mlp")
TRAIN = tuple("ap_expanded_train_%03d" % i for i in range(32))
VAL = tuple("ap_expanded_val_%03d" % i for i in range(8))
PASSES = (5, 10, 15, 20, 25, 30)
MODEL = dict(hidden_dim=32, expected_haplotypes=952, site_chunk_size=256, checkpoint_sites=True)
SELECTION = "earliest_maximum_equal_parent_Present182_exact_span_accuracy_then_lower_NLL_then_earlier_pass"
WARM_COMPLETION = dict(path="experiments/results/mutation_routed_full_panel_pyg_v1_aligned_seed101_GPU2/completed.json",
    sha256="a36e449a5ec3042993419eaf1a61628122fa24f15de320f0ad02f94a2b087567")
WARM_CHECKPOINT = dict(path="experiments/results/mutation_routed_full_panel_pyg_v1_aligned_seed101_GPU2/aligned.pass_30.pt",
    sha256="7fb53005867077b4b113097df2b5903e99c74f96f88e8e7c1a43ecef97947eb6")
MECHANISM_REFS = (
    dict(path="src/arggnn/anonymous_clade_witness_v1.py", sha256="0cdb438132e11c86fb61306a02244af0471f468c2ff2a57324644cafb97b6310"),
    dict(path="src/arggnn/anonymous_clade_gnn_v1.py", sha256="2f878e6334079cff083f67b6da8eda129dd30e7beebc62ae2a93d51796c3f137"),
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bound(record):
    require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "Exact project path/SHA required")
    p, digest = Path(record["path"]), record["sha256"]
    require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == record["path"]
        and isinstance(digest, str) and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest), "Canonical project reference required")
    path = ROOT / p
    require(path.resolve() == path and path.is_file(), "Existing project-local file required")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, "Pinned bytes changed: " + record["path"])
    return data


def metadata(record):
    require(record["path"].endswith(".json"), "JSON metadata reference required")
    return json.loads(bound(record))


def science():
    """Initialize original lazy serializers before any guarded evidence write."""
    global np, torch, h038, fit
    import numpy as np
    import torch
    from . import census_tsinfer_fit_v1 as fit
    from . import train_mutation_routed_full_panel_pyg_v1 as h038
    from . import train_mutation_routed_whole_tree_gnn_v1 as original
    original.science_v1()
    fit.np, fit.torch = np, torch


def array_identity(array):
    value = np.ascontiguousarray(array)
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=int(value.nbytes),
        sha256=hashlib.sha256(value.tobytes()).hexdigest())


def witness_bounds(native_holds, native_nodes, depth_bound):
    """Raw array bounds. Python/allocator/autograd RSS is separately guarded."""
    require(type(native_holds) is int and native_holds >= 1
        and type(native_nodes) is int and native_nodes >= 1
        and type(depth_bound) is int and depth_bound >= 0, "Positive observed scalar bounds required")
    e = 5692 * (native_holds - 1)
    lifting_tables = 1 + depth_bound.bit_length()
    return dict(candidate_events=e, audit_arrays_max_bytes=26*e+68320,
        cached_runtime_events_max_bytes=20*e,
        # Conservative simultaneous ndarray preparation/verification allowance:
        # event chunks + concatenation + unique/sorting + degrees + LCA scratch.
        preparation_ndarray_allowance_bytes=128*e+(4*lifting_tables+128)*native_nodes
            +128*16*5692+16*native_holds*952+68320,
        coupling_forward_persistent_tensor_bytes=224*native_nodes+32*native_holds*952,
        event_block_gather_bytes=4*32768*(3*32+6*8))


def validate_protocol(cfg):
    require(cfg.get("schema") == "H040_latent_clade_execution_config_v1"
        and cfg.get("hypothesis_id") == "H040" and cfg.get("linked_hypothesis_id") == "H038"
        and cfg.get("arms") == list(ARMS) and cfg.get("seed") == 101 and cfg.get("dtype") == "float32"
        and cfg.get("passes") == 30 and cfg.get("updates_per_pass") == 32
        and cfg.get("updates_per_arm") == 960 and cfg.get("checkpoint_passes") == list(PASSES)
        and cfg.get("model") == MODEL and cfg.get("parameters_per_arm") == 30788
        and cfg.get("learning_rate") == .001 and cfg.get("weight_decay") == .0001
        and cfg.get("gradient_clip_norm") == 1. and cfg.get("early_stopping") is False
        and cfg.get("VAL_influences_optimization") is False
        and cfg.get("selection") == SELECTION and cfg.get("dense_event_sidecars") is False
        and cfg.get("warmstart_completion_ref") == WARM_COMPLETION
        and cfg.get("warmstart_checkpoint_ref") == WARM_CHECKPOINT
        and cfg.get("mechanism_refs") == list(MECHANISM_REFS),
        "Exact prospective H040 paired 30-pass protocol required")
    for name in ("CPU_bank_cap_bytes", "CPU_target_cap_bytes", "witness_cache_cap_bytes",
        "witness_preparation_cap_bytes", "GPU_residency_cap_bytes", "GPU_event_cap_bytes",
        "RAM_prediction_cap_bytes"):
        require(type(cfg.get(name)) is int and cfg[name] > 0, "Positive root-reviewed cap: " + name)


class PhaseReader:
    """Observation capability plus exact phase truth bindings and a seal barrier."""
    def __init__(self, base, phase, bindings):
        self.base, self.root, self.phase = base, base.root, phase
        self.truth = {(v["truth_npz"]["path"], v["truth_npz"]["sha256"]) for v in bindings.values()}
        self.truth_gate = False
        self.ram = None

    def read(self, record, **kwargs):
        parts = Path(record["path"]).parts
        if "truth" in parts:
            require((record["path"],record["sha256"]) in self.truth and self.truth_gate,
                "Truth denied outside exact phase bindings or before complete observation/prediction barrier")
        return self.base.read(record, **kwargs)

    def decode_json(self, record):
        return json.loads(self.read(record))

    def current_probability_seal_v1(self, record, parent):
        require(self.phase == "evaluate" and self.ram is not None and self.truth_gate,
            "Frozen complete RAM matrix required for scoring")
        return self.ram.grid(record, parent)


class WitnessCache:
    """Bounded CPU LRU, canonical recipes, no witness array writer."""
    def __init__(self, cfg, guard):
        self.cfg, self.guard = cfg, guard
        self.values, self.bytes, self.proofs = OrderedDict(), 0, {}
        self.builds, self.hits, self.build_seconds = 0, 0, 0.
        self.peak_runtime_bytes = 0

    def get(self, pid, bank, keys):
        self.guard()
        if pid in self.values:
            self.hits += 1
            self.values.move_to_end(pid)
            return self.values[pid]
        from .anonymous_clade_witness_v1 import build_event_arrays_v1, verify_event_arrays_v1, AnonymousCladeEventsV1
        t, n = int(bank.sample_index.shape[0]), int(bank.parent.numel())
        # Downward traversal depth supplies an observation-only lifting bound.
        depth = len(bank.downward_levels)
        limits = witness_bounds(t, n, depth)
        require(limits["preparation_ndarray_allowance_bytes"] <= self.cfg["witness_preparation_cap_bytes"],
            "Worst-case witness preparation exceeds admitted ndarray allowance")
        # CPU cache and preparation scratch have separate prospective caps.
        # Keep deterministic witnesses when they fit; no disk sidecars.
        start = time.monotonic()
        arrays, meta = build_event_arrays_v1(bank, keys, pid, hold_block_size=16)
        verify_event_arrays_v1(arrays, bank, block_size=16384)
        events = AnonymousCladeEventsV1.from_verified_arrays(arrays, meta)
        identities = {k:array_identity(arrays[k]) for k in
            ("boundary_index","old_node","new_node","shared_sample","weight")}
        proof = dict(metadata=meta, runtime_array_identities=identities, worst_case=limits)
        if pid in self.proofs:
            require(self.proofs[pid] == proof, "Rebuilt witness recipe/content differs after target access")
        else:
            self.proofs[pid] = proof
        size = events.tensor_bytes()
        require(size <= self.cfg["witness_cache_cap_bytes"], "One complete RAM witness exceeds cache cap")
        while self.values and self.bytes + size > self.cfg["witness_cache_cap_bytes"]:
            _, old = self.values.popitem(last=False)
            self.bytes -= old.tensor_bytes()
        self.values[pid] = events
        self.bytes += size
        self.builds += 1
        self.build_seconds += time.monotonic()-start
        self.peak_runtime_bytes = max(self.peak_runtime_bytes, self.bytes)
        del arrays
        self.guard()
        return events


def stable_keys(bank, entry_ref, reader):
    from .mutation_routed_parent_loader_v1 import ObservationOnlyParentReaderV1
    from .mutation_routed_full_panel_observations_v1 import validate_full_roster_v1
    boundary = ObservationOnlyParentReaderV1(reader)
    entry = boundary.decode_json(entry_ref)
    require(entry.get("observations_only") is True and entry.get("truth_or_oracle_read") is False,
        "Original observed-only entry required")
    rows = boundary.decode_json(entry["refs"]["source_roster"])
    validate_full_roster_v1(rows)
    keys = tuple((row["sample_id"], int(row["haplotype"])) for row in rows)
    classes = np.asarray([row["observed_reference_class"] if row["role"]=="pure_source_reference" else -1
        for row in rows], dtype=np.int64)
    require(np.array_equal(classes, bank.reference_class.numpy()), "Stable observed roster/bank column join")
    return keys


def load_banks(ctx, reader, ids, cache):
    from .mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1, read_full_cache_parent_v1
    index = read_full_cache_index_v1(ctx["observation_cache_ref"], reader,
        registry=ctx["registry"], entry_refs=ctx["entry_refs"], source_refs=ctx["observation_source_refs"])
    banks, parents, keys = {}, {}, {}
    total = 0
    for pid in ids:
        ctx["guard"]()
        item = read_full_cache_parent_v1(pid, index, reader)
        bank, parent = item["model_inputs"], item["scoring"]
        require(parent["split"] == ("train" if pid in TRAIN else "val"), "Observed split differs")
        total += h038.tensor_bytes_v1(bank)
        require(total <= ctx["cfg"]["CPU_bank_cap_bytes"], "Complete CPU observation banks exceed cap")
        banks[pid], parents[pid] = bank, parent
        keys[pid] = stable_keys(bank, ctx["entry_refs"][pid], reader)
        cache.get(pid, bank, keys[pid])
        print(json.dumps(dict(stage="H040_observed_witness_recipe",parent_id=pid,
            native_holds=int(bank.sample_index.shape[0]),events=cache.proofs[pid]["metadata"]["event_count"],
            CPU_bank_bytes=total)),flush=True)
        del item
    seal = ctx["output"].save("all_%d_observed_RAM_witness_recipes.json" % len(ids),
        dict(hypothesis_id="H040",parent_ids=list(ids),observation_cache_ref=ctx["observation_cache_ref"],
            entry_refs={p:ctx["entry_refs"][p] for p in ids},recipes=cache.proofs,
            all_observations_precede_phase_truth=True,event_sidecar_arrays_written=False,
            CPU_bank_tensor_bytes=total,mechanism_refs=list(MECHANISM_REFS)))
    return banks, parents, keys, seal


def load_warmstart(reader):
    completed = reader.decode_json(WARM_COMPLETION)
    require(completed.get("status") == "completed_H038_seed101_accuracy_selected_full_parent_run"
        and completed.get("arm") == "aligned" and completed.get("seed") == 101
        and completed.get("outer_truth_read") is False, "Actual completed aligned H038 required")
    selected = completed["selected"]
    require(selected["checkpoint"] == WARM_CHECKPOINT
        and completed["retained_records"].get(WARM_CHECKPOINT["path"]) == WARM_CHECKPOINT,
        "Selected retained H038 pass30 checkpoint identity required")
    payload = torch.load(io.BytesIO(reader.read(WARM_CHECKPOINT)),map_location="cpu",weights_only=False)
    require(payload.get("arm")=="aligned" and payload.get("seed")==101
        and payload.get("teacher_used") is False and payload.get("pass_index")==selected["pass_index"]
        and payload.get("optimizer_steps")==selected["optimizer_steps"], "Warm-start provenance differs")
    return payload


def state_identity(state):
    digest = hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_one(model, optimizer, bank, mass, events, arm, guard):
    from .biological_gst_control_v1 import exact_span_loss
    guard()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    with torch.autocast(device_type=bank.parent.device.type,enabled=False):
        logits = model.forward_train_holds(bank,boundary=events,interaction=arm)
        loss = exact_span_loss(logits,mass)
    require(bool(torch.isfinite(loss)), "Finite full-parent FP32 TRAIN loss required")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Finite updated parameters required")
    row = dict(loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.monotonic()-start)
    del logits,loss,norm
    optimizer.zero_grad(set_to_none=True)
    guard()
    return row


def run_train(ctx, input_ref):
    from .anonymous_clade_gnn_v1 import build_matched_family_v1
    from .mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1
    cfg, output, guard, device = ctx["cfg"],ctx["output"],ctx["guard"],ctx["device"]
    reader = PhaseReader(ctx["reader"],"train",ctx["bindings"])
    cache = WitnessCache(cfg,guard)
    arm = ctx["arm"]
    require(arm in ARMS, "Exactly one registered arm per TRAIN invocation")
    active_arms = (arm,)
    checkpoints, history, steps = {}, [], {arm:0}
    banks,parents,keys,observation_seal = load_banks(ctx,reader,TRAIN,cache)
    require(set(cache.proofs)==set(TRAIN), "All32 observed recipes before any TRAIN truth")
    reader.truth_gate = True
    targets = {}
    helpers = fit.metric_helpers_v1()
    target_bytes = 0
    for pid in TRAIN:
        guard()
        target = FullPanelExternalTargetV1(parents[pid],ctx["bindings"][pid],"TRAIN_loss",reader)
        mass = h038.native_hold_train_mass_v1(parents[pid],banks[pid],target,helpers)
        target_bytes += int(mass.nbytes)
        require(target_bytes <= cfg["CPU_target_cap_bytes"], "External TRAIN mass exceeds cap")
        targets[pid] = torch.from_numpy(mass)
        del target,mass
    family,optimizers,initialization = build_matched_family_v1(load_warmstart(reader),MODEL,
        device=device,execution_guard=guard)
    initial_hashes = {a:state_identity(m.state_dict()) for a,m in family.items()}
    require(len(set(initial_hashes.values()))==1, "All learned states must start identically")
    for other in ARMS:
        if other != arm:
            del optimizers[other], family[other]
    residency = h038.FullParentResidencyV1(banks,targets,device,cfg["GPU_residency_cap_bytes"])
    output.save("initialization.json",dict(initialization=initialization,initial_state_sha256=initial_hashes,
        observation_seal=observation_seal,external_TRAIN_mass_bytes=target_bytes,
        witness_storage_bytes=0,checkpoint_passes=list(PASSES),VAL_influences_optimization=False,
        arm=arm,matched_initial_states_verified_before_selecting_one_arm=True))
    try:
        for at in range(1,31):
            order = list(TRAIN)
            random.Random(101+at).shuffle(order)
            sums = {arm:0.}
            started = time.monotonic()
            for number,pid in enumerate(order,1):
                guard()
                events_cpu = cache.get(pid,banks[pid],keys[pid])
                require(events_cpu.tensor_bytes()<=cfg["GPU_event_cap_bytes"], "One complete GPU event payload exceeds cap")
                bank,mass = residency.get(pid)
                events = events_cpu.to(device)
                for arm in active_arms:
                    row = train_one(family[arm],optimizers[arm],bank,mass,events,arm,guard)
                    steps[arm] += 1
                    sums[arm] += row["loss"]/32
                del events,bank,mass,events_cpu
                if number%8==0:
                    print(json.dumps(dict(stage="H040_TRAIN_progress",pass_index=at,
                        complete_parents=number,steps=steps.copy(),elapsed_seconds=time.monotonic()-started)),flush=True)
            row = dict(pass_index=at,steps=steps.copy(),mean_TRAIN_NLL=sums,
                full_parent_updates=32,seconds=time.monotonic()-started)
            history.append(row)
            output.save("TRAIN_pass_%02d.json"%at,row)
            if at in PASSES:
                groups = {}
                for arm in active_arms:
                    guard()
                    groups[arm] = output.save("%s.pass_%02d.pt"%(arm,at),dict(
                        schema="H040_frozen_checkpoint_v1",hypothesis_id="H040",phase="train",
                        model_state={k:v.detach().cpu() for k,v in family[arm].state_dict().items()},
                        optimizer_state=optimizers[arm].state_dict(),pass_index=at,
                        additional_optimizer_steps=steps[arm],arm=arm,seed=101,model_kwargs=MODEL,
                        parameter_count=30788,input_ref=input_ref,config_ref=ctx["config_ref"],
                        observation_seal=observation_seal,warmstart_checkpoint_ref=WARM_CHECKPOINT,
                        initial_state_sha256=initial_hashes[arm],teacher_used=False,
                        optimizer_state_reset=True,mechanism_refs=list(MECHANISM_REFS)),"torch")
                checkpoints[str(at)] = groups
                output.save("pass_%02d_checkpoints.json"%at,groups)
        require(steps=={arm:960} and len(checkpoints)==6, "Complete paired30-pass schedule required")
        result = dict(schema="H040_train_completion_v1",status="completed_fixed_30_pass_single_arm_TRAIN",arm=arm,
            hypothesis_id="H040",input_ref=input_ref,config_ref=ctx["config_ref"],
            checkpoint_groups=checkpoints,steps=steps,history=history,initialization=initialization,
            initial_state_sha256=initial_hashes[arm],warmstart_checkpoint_ref=WARM_CHECKPOINT,
            observations_seal=observation_seal,VAL_truth_read=False,outer_truth_read=False,
            witness_arrays_written=False,witness_preparation_builds=cache.builds,
            witness_preparation_seconds=cache.build_seconds,
            peak_cached_witness_tensor_bytes=cache.peak_runtime_bytes,selection_deferred=True)
        output.save("completed.json",result)
        return result
    except Exception as error:
        output.save("failed.json",dict(status="failed_partial_preserved",error_type=type(error).__name__,
            error=str(error),logical_steps=steps,completed_checkpoint_groups=checkpoints,
            automatic_replay=False,VAL_truth_read=False,outer_truth_read=False))
        raise


class RAMMatrix:
    """All96 frozen native-hold predictions in CPU RAM before any VAL truth."""
    def __init__(self, output, reader, parents, banks, cfg):
        self.output,self.reader,self.parents,self.banks,self.cfg=output,reader,parents,banks,cfg
        self.values,self.records,self.lookup,self.packets={}, {}, {}, {}
        self.bytes=0
        self.verified=False

    def seal(self, at, arm, pid, value, checkpoint):
        key=(at,arm,pid)
        require(key not in self.values, "Prediction replacement forbidden")
        value=np.array(value,dtype=np.float32,copy=True,order="C")
        bank,parent=self.banks[pid],self.parents[pid]
        require(value.shape==(len(bank.spans),686,4) and np.isfinite(value).all()
            and np.all(value>=0) and np.allclose(value.sum(-1),1,rtol=1e-5,atol=1e-6),
            "Complete native-hold probabilities required")
        self.bytes+=int(value.nbytes)
        require(self.bytes<=self.cfg["RAM_prediction_cap_bytes"], "Frozen RAM prediction matrix exceeds cap")
        value.setflags(write=False)
        geometry={k:array_identity(parent[k]) for k in
            ("positions","edges","retained_indexes","query_slots","raw_query_nodes","cohorts")}
        geometry["readout_hold"]=array_identity(bank.readout_hold.numpy())
        packet=dict(schema="H040_native_hold_RAM_seal_v1",hypothesis_id="H040",pass_index=at,
            arm=arm,parent_id=pid,checkpoint_ref=checkpoint,probability=array_identity(value),
            geometry=geometry,mechanism_refs=list(MECHANISM_REFS),
            prediction_array_durable=False,targets_opened_before_seal=False)
        ref=self.output.save("%s.pass_%02d.%s.RAM_seal.json"%(arm,at,pid),packet)
        self.values[key],self.records[key],self.packets[key]=value,ref,packet
        self.lookup[ref["path"]]=key

    def verify(self):
        require(set(self.values)=={(at,arm,p) for at in PASSES for arm in ARMS for p in VAL},
            "All6 passes x2 arms x8 VAL predictions required")
        for key,value in self.values.items():
            require(not value.flags.writeable and array_identity(value)==self.packets[key]["probability"]
                and self.reader.decode_json(self.records[key])==self.packets[key],
                "Frozen probability/JSON seal changed")
        self.output.save("all96_VAL_RAM_sealed_before_truth.json",dict(hypothesis_id="H040",
            checkpoint_passes=list(PASSES),arms=list(ARMS),VAL=list(VAL),
            complete_native_RAM_prediction_count=96,native_probability_bytes=self.bytes,
            dense_prediction_arrays_saved=False,all96_sealed_before_any_VAL_truth=True,
            seals=[self.records[k] for k in sorted(self.records)]))
        self.verified=True

    def grid(self,record,parent):
        require(self.verified and record["path"] in self.lookup,"Verified complete matrix required")
        key=self.lookup[record["path"]]
        require(record==self.records[key] and key[2]==parent["parent_id"],"Current same-parent seal required")
        value=self.values[key]
        require(not value.flags.writeable and array_identity(value)==self.packets[key]["probability"],
            "Immutable native probabilities changed")
        return value[self.banks[key[2]].readout_hold.numpy()]


def run_evaluate(ctx,input_ref):
    from .anonymous_clade_gnn_v1 import AnonymousCladeTensorGNNV1
    from .mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1
    cfg,output,guard,device=ctx["cfg"],ctx["output"],ctx["guard"],ctx["device"]
    reader=PhaseReader(ctx["reader"],"evaluate",ctx["bindings"])
    require(set(ctx["train_completion_refs"])==set(ARMS), "Both actual single-arm completion refs")
    trains={a:reader.decode_json(ctx["train_completion_refs"][a]) for a in ARMS}
    for arm, train in trains.items():
        require(train.get("status")=="completed_fixed_30_pass_single_arm_TRAIN"
            and train.get("hypothesis_id")=="H040" and train.get("arm")==arm
            and train.get("steps")=={arm:960} and train.get("outer_truth_read") is False,
            "Actual complete30-pass/960-update single-arm TRAIN required")
    require(trains[ARMS[0]]["initial_state_sha256"]==trains[ARMS[1]]["initial_state_sha256"]
        and trains[ARMS[0]]["initialization"]==trains[ARMS[1]]["initialization"],
        "Both arms must share the original matched initializer")
    states,checkpoints={},{}
    for at in PASSES:
        for arm in ARMS:
            guard()
            train=trains[arm]
            ref=train["checkpoint_groups"][str(at)][arm]
            cp=torch.load(io.BytesIO(reader.read(ref)),map_location="cpu",weights_only=False)
            require(cp.get("schema")=="H040_frozen_checkpoint_v1" and cp.get("arm")==arm
                and cp.get("pass_index")==at and cp.get("additional_optimizer_steps")==32*at
                and cp.get("parameter_count")==30788 and cp.get("model_kwargs")==MODEL
                and cp.get("teacher_used") is False and cp.get("mechanism_refs")==list(MECHANISM_REFS)
                and cp.get("warmstart_checkpoint_ref")==WARM_CHECKPOINT
                and cp.get("input_ref")==train["input_ref"] and cp.get("config_ref")==train["config_ref"],
                "Exact actual frozen H040 checkpoint/provenance required")
            require(all(isinstance(v,torch.Tensor) and v.dtype==torch.float32 for v in cp["model_state"].values()),
                "Raw FP32 learned states required")
            states[at,arm],checkpoints[at,arm]=cp["model_state"],ref
            del cp
    cache=WitnessCache(cfg,guard)
    banks,parents,keys,observation_seal=load_banks(ctx,reader,VAL,cache)
    ram=RAMMatrix(output,reader,parents,banks,cfg)
    reader.ram=ram
    residency=h038.FullParentResidencyV1(banks,{},device,cfg["GPU_residency_cap_bytes"])
    model=AnonymousCladeTensorGNNV1(**MODEL).to(device=device,dtype=torch.float32).eval().requires_grad_(False)
    for pid in VAL:
        guard()
        events_cpu=cache.get(pid,banks[pid],keys[pid])
        require(events_cpu.tensor_bytes()<=cfg["GPU_event_cap_bytes"],"Complete GPU event cap")
        bank,_=residency.get(pid)
        events=events_cpu.to(device)
        for at in PASSES:
            for arm in ARMS:
                guard()
                model.load_state_dict(states[at,arm],strict=True)
                with torch.no_grad(),torch.autocast(device_type="cuda",enabled=False):
                    probability=model.forward_train_holds(bank,boundary=events,interaction=arm).softmax(-1)
                ram.seal(at,arm,pid,probability.cpu().numpy(),checkpoints[at,arm])
                del probability
        del bank,events,events_cpu
    del model,residency,states
    ram.verify()
    reader.truth_gate=True
    helpers=fit.metric_helpers_v1()
    selected,history,full={}, {}, {}
    for arm in ARMS:
        validation=FullPanelPrimaryValidationV1(parents,ctx["bindings"],reader,helpers,cfg)
        rows=[]
        for at in PASSES:
            guard()
            seals=[(p,ram.records[at,arm,p]) for p in VAL]
            cells,means=validation.score(seals)
            row=dict(pass_index=at,cells=cells,equal_parent_metrics=means,
                checkpoint=checkpoints[at,arm],selection_rule=SELECTION)
            rows.append(row)
            output.save("%s.pass_%02d.primary_metrics.json"%(arm,at),row)
        best=min(rows,key=lambda row:h038.selection_key_v1(row["equal_parent_metrics"],row["pass_index"]))
        selected[arm],history[arm]=best,rows
        chosen=[(p,ram.records[best["pass_index"],arm,p]) for p in VAL]
        cells,means=validation.full_final(chosen)
        full[arm]=dict(cells=cells,equal_parent_metrics=means,selection_revised=False)
        output.save(arm+".selected_full_diagnostics.json",full[arm])
    result=dict(schema="H040_frozen_VAL_completion_v1",status="completed_original_H038_selection",
        hypothesis_id="H040",input_ref=input_ref,train_completion_refs=ctx["train_completion_refs"],
        observations_seal=observation_seal,selection_rule=SELECTION,selected=selected,history=history,
        selected_full_diagnostics=full,optimization_performed=False,
        all96_RAM_seals_before_any_VAL_truth=True,RAM_native_probability_bytes=ram.bytes,
        dense_prediction_arrays_saved=False,outer_truth_read=False)
    output.save("completed.json",result)
    return result


def run_context(ctx,input_ref):
    """Root factory must enforce the current lease, storage and resource guards."""
    science()
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
        and ctx.get("phase") in ("train","evaluate") and callable(ctx.get("guard")),
        "Independently admitted root execution context required")
    validate_protocol(ctx["cfg"])
    for ref in MECHANISM_REFS:
        bound(ref)
    require(ctx["reader"].root.resolve()==ROOT and ctx["device"].type=="cuda",
        "Existing project reader and root-qualified leased CUDA device required")
    require(ctx["config_ref"] and ctx["reader"].decode_json(ctx["config_ref"])==ctx["cfg"],
        "Hash-bound root config must equal supplied protocol")
    require(ctx.get("arm") in ARMS if ctx["phase"]=="train" else ctx.get("arm")=="joint_pair",
        "Single-arm TRAIN or jointly sealed paired evaluation required")
    ids=TRAIN if ctx["phase"]=="train" else VAL
    require(set(ctx["bindings"])==set(ids),"Exact phase target keys; no outer targets")
    require(set(ctx["entry_refs"])==set(TRAIN+VAL),"Original40 observed metadata refs, no outer entries")
    for pid,row in ctx["bindings"].items():
        require(row["split"]==("train" if pid in TRAIN else "val")
            and row["truth_npz"]["path"].endswith("/"+pid+"/truth/full_truth_intervals.npz"),
            "Original same-parent phase truth binding required")
    ctx["guard"]()
    return run_train(ctx,input_ref) if ctx["phase"]=="train" else run_evaluate(ctx,input_ref)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_not_executed",hypothesis_id="H040",
            source=SOURCE,arms=list(ARMS),passes=30,updates_per_arm=960,
            checkpoint_passes=list(PASSES),parameters_per_arm=30788,
            witness_array_disk_bytes=0,root_context_required=True),indent=2))
        return 0
    require(args.input and args.expected_input_sha256,"Pinned active root input capability required")
    record=dict(path=args.input,sha256=args.expected_input_sha256)
    spec=metadata(record)
    require(spec.get("schema")=="H040_root_execution_input_v1" and spec.get("root_authorizer")=="root_agent"
        and spec.get("status")=="active" and spec.get("hypothesis_id")=="H040"
        and spec.get("phase") in ("train","evaluate"),"Active root H040 capability required")
    ref=spec["root_context_module_ref"]
    bound(ref)
    require(ref["path"].startswith("research/reviews/") and ref["path"].endswith(".py"),
        "Separate pinned root-owned execution factory required")
    module_spec=importlib.util.spec_from_file_location("H040_root_context",ROOT/ref["path"])
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    context=module.build_context_v1(record,spec)
    require(context["phase"]==spec["phase"] and context["arm"]==spec["arm"],
        "Root factory phase/arm differs from capability")
    run_context(context,record)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
