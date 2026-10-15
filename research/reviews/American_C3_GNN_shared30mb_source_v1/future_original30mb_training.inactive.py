# UNREVIEWED INACTIVE FUTURE DRAFT: outside production closure; no runtime authority or fit readiness.
"""Thin original-30Mb C3 task binding for the existing whole-tree PyG baseline.

Source-only scientific callables. Root owns runtime, resource authority, writer,
process lifecycle and actual execution. No new architecture, admission or CLI.
"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import copy,random,zipfile
import numpy as np
import torch
from .american_c3_common_v1 import require,read,bound,ref,CLASSES
from .american_c3_evaluate_v1 import registry,verify_prediction
from .american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
from .american_c3_gnn_dataset_v1 import (
    AmericanC3WholeTreeDatasetV1,build_plan,shutdown_dataloader)
from .american_c3_gnn_targets_v1 import TrainFocalTargetVectorV1,identity_sha
from .american_c3_gnn_stream_train_v1 import (
    SealedObservationProviderV1,validate_plan_v1,_feature_identity,
    checkpointed_complete_parent_step_v1,json_sha)
from .american_c3_gnn_train_v1 import _cpu_snapshot,_rng_state,_seed_fresh
from .american_c3_gnn_run_v1 import _streamed_focal_predictions_impl

SOURCE="src/arggnn/american_c3_gnn_shared30mb_v1.py"
REGISTRY_REF={"path":"research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json",
    "sha256":"608da56508757646a0584cb653f7f6af4c910f05d560d6da1b99c0c1ea691f9a"}
TRAIN=tuple("american_c3_train_%03d"%i for i in range(20))
VAL=tuple("american_c3_val_%03d"%i for i in range(5))
TEST=tuple("american_c3_test_%03d"%i for i in range(10))
LENGTH=30000000
CONDITION="fresh_C3_original20x30Mb_complete_parents_v1"
SCHEMA="american_c3_shared_original30Mb_GNN_v1"
CHECKPOINT_EPOCHS=(5,10,15,20,25,30)
KNOWN_TRAIN000={
 "observation_ref":{"path":"experiments/results/American_C3_observed_full30mb_supervisor_repair_20261015_v1/observed/manifest.json",
    "sha256":"c3d58a8d10d901a162f406e2e218a10e17e5f1617b3c626503d72bcbd1b4ccee"},
 "focal_registry_ref":{"path":"experiments/results/American_C3_native_paths_train000_full30mb_seed92001_v3/focals.npz",
    "sha256":"853b28cf2383d4e7fdf5f2c24526c64ef8071ce81ec04a07d1d69cd7350c8da5"},
 "native_preparation_ref":{"path":"experiments/results/American_C3_native_paths_train000_full30mb_seed92001_v3/completed.json",
    "sha256":"3cc49fbf55bd4b4b3859a93ce1123e9218f407f7c7299ad5029be81f59c7c6fa"}}
DEFAULTS=dict(schema=SCHEMA,execution_enabled=False,registry_ref=REGISTRY_REF,
 condition=CONDITION,class_order=list(CLASSES),training_length_bp=LENGTH,
 epochs=30,parents_per_epoch=20,total_updates=600,focals_per_parent=10000,focal_exposures=6000000,
 model_seed=70001,parent_order_seed=70002,precision="fp32",
 model_kwargs=dict(hidden_dim=32,class_order=list(CLASSES),expected_haplotypes=1600,
    donor_cohorts=2,site_chunk_size=256,checkpoint_sites=True),
 optimizer=dict(name="torch.optim.AdamW",lr=.001,weight_decay=.0001,betas=[.9,.999],eps=1e-8),
 gradient_clip=1.,scheduler="constant_learning_rate",checkpoint_epochs=list(CHECKPOINT_EPOCHS),
 checkpoint_rule="prespecified_final_epoch30_only_no_VAL_or_TEST_selection",
 early_stopping=False,purity_filter=False,CPU_observation_cache_bytes=536870912,
 focal_chunk_size=1024,max_truth_uncompressed_bytes=None)

def original_registry_v1():
    value=read(REGISTRY_REF);rows=value["parents"]
    expected=[*(("train",p,21001+i,1021001+i,92001+i) for i,p in enumerate(TRAIN)),
        *(("val",p,31001+i,1031001+i,93001+i) for i,p in enumerate(VAL)),
        *(("test",p,41001+i,1041001+i,94001+i) for i,p in enumerate(TEST))]
    require(value["schema"]=="american_c3_parent_split_seeds_v1"
        and len(rows)==35 and [(r["split"],r["parent_id"],r["ancestry_seed"],r["mutation_seed"],r["path_sampling_seed"])
            for r in rows]==expected,"Exact original independent-parent split/seeds; no short-parent substitution")
    return {r["parent_id"]:r for r in rows}

def validate_config_v1(cfg):
    adjustable={"execution_enabled","CPU_observation_cache_bytes","focal_chunk_size","max_truth_uncompressed_bytes"}
    require(set(cfg)==set(DEFAULTS) and all(cfg[k]==v for k,v in DEFAULTS.items() if k not in adjustable)
        and cfg["execution_enabled"] is True,"Fixed source-only protocol requires explicit root activation")
    require(type(cfg["CPU_observation_cache_bytes"]) is int and cfg["CPU_observation_cache_bytes"]>=0
        and type(cfg["focal_chunk_size"]) is int and 0<cfg["focal_chunk_size"]<=10000
        and type(cfg["max_truth_uncompressed_bytes"]) is int and cfg["max_truth_uncompressed_bytes"]>0,
        "Explicit bounded root cache/focal/private-TRAIN-read caps")
    original_registry_v1()

def validate_shared_plan_v1(plan,*,parent_id):
    require(parent_id in TRAIN+VAL+TEST,"Original independent C3 registry membership")
    validate_plan_v1(plan,parent_id=parent_id,length_bp=LENGTH)
    require(plan["config"]["reference_counts"]==[200,200,200]
        and plan["config"]["native_preparation_ref"] is not None,
        "Same native preparation and full600 reference/1000 query panel")
    if parent_id==TRAIN[0]:
        require(all((plan["observation_ref"] if k=="observation_ref" else
            plan["focal_registry_ref"] if k=="focal_registry_ref" else plan["config"][k])==v
            for k,v in KNOWN_TRAIN000.items()),"Exact existing original30Mb TRAIN000 observation/native focals")

def build_shared_plan_v1(cfg):
    """Future observed-only count pass; never reads targets or the simulated ARG."""
    plan=build_plan(cfg)
    validate_shared_plan_v1(plan,parent_id=plan["parent_id"])
    return plan

class Shared30MbProviderV1(SealedObservationProviderV1):
    """Existing immutable-block/LRU/seal machinery with original30Mb membership."""
    def __init__(self,plan_ref,*,parent_id,worker_tmp_directory,cache_bytes):
        require(parent_id in TRAIN and type(cache_bytes) is int and cache_bytes>=0,
            "Original TRAIN-only provider")
        self.dataset=AmericanC3WholeTreeDatasetV1(plan_ref,direction="forward",
            worker_tmp_directory=worker_tmp_directory)
        self.plan=dict(self.dataset.plan,plan_ref=dict(self.dataset.plan_ref))
        validate_shared_plan_v1(self.plan,parent_id=parent_id)
        self.cache_bytes=cache_bytes;self.cache=OrderedDict();self.used=0;self.seal_record=None
    def _read(self,index):
        block=self.dataset[index];spec=self.plan["blocks"][index]
        require(block.parent_id==self.plan["parent_id"] and block.plan_ref==self.plan["plan_ref"]
            and block.block_index==index and block.start_tree==spec["start_tree"]
            and block.stop_tree==spec["stop_tree"] and block.sequence_length_bp==LENGTH,
            "Exact original30Mb physical block on every recomputation")
        require(torch.equal(block.query_column,torch.arange(600,1600))
            and torch.equal(block.reference_class,torch.tensor([2]*200+[0]*200+[1]*200+[-1]*1000)),
            "Original AFR/EUR/EAS reference roster, unknown queries; no fourth class")
        rows=block.focal_rows
        require(rows.dtype==torch.long and rows.ndim==1 and block.focal_hold.shape==rows.shape
            and block.focal_sample_column.shape==rows.shape
            and bool(((rows>=0)&(rows<10000)).all())
            and bool(((block.focal_sample_column>=600)&(block.focal_sample_column<1600)).all())
            and bool(((block.focal_hold>=0)&(block.focal_hold<block.stop_tree-block.start_tree)).all()),
            "Exact common query focal row/sample/tree indices")
        digest,size=_feature_identity(block)
        if self.seal_record is not None:
            require(digest==self.seal_record["block_feature_sha256"][index],"Observation changed after seal")
        return block,digest,size

class OriginalTrainTargetJoinV1:
    """Only separate TRAIN labels after all20 observed-parent seals."""
    def __init__(self,resolve_train_metadata,*,max_truth_uncompressed_bytes,all_parent_seals):
        require(callable(resolve_train_metadata) and type(max_truth_uncompressed_bytes) is int
            and max_truth_uncompressed_bytes>0 and set(all_parent_seals)==set(TRAIN)
            and all(s["parent_id"]==p and s["all_blocks_sealed_before_truth"] is True
                and s["observation_only"] is True for p,s in all_parent_seals.items()),
            "Global all20 observation barrier before any TRAIN private capability")
        self.rows=original_registry_v1();self.resolve=resolve_train_metadata
        self.cap=max_truth_uncompressed_bytes;self.seals=all_parent_seals
    def join(self,parent_id,*,split,plan):
        require(split=="train" and parent_id in TRAIN,"Deny VAL/TEST/short parent before resolving private metadata")
        row=self.rows[parent_id];seal=self.seals[parent_id]
        validate_shared_plan_v1(plan,parent_id=parent_id)
        require(seal["plan_ref"]==plan["plan_ref"],"Same sealed observation plan")
        r=registry(plan["focal_registry_ref"])
        require(r["parent_id"].item()==parent_id and len(r["record_ids"])==10000
            and identity_sha(r)==seal["focal_identity_sha256"],"Exact sealed common focal identities")
        capability=self.resolve(parent_id)
        require(set(capability)=={"simulation_config_ref","simulation_completion_ref"},"Metadata-only TRAIN capability")
        cfg=read(capability["simulation_config_ref"]);done=read(capability["simulation_completion_ref"])
        observed=read(plan["observation_ref"])
        require(cfg["parent_id"]==parent_id and cfg["split"]=="train" and cfg["sequence_length_bp"]==LENGTH
            and cfg["class_order"]==list(CLASSES) and cfg["sample_unit"]=="haploid"
            and cfg["sample_counts"]=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000}
            and cfg["ancestry_seed"]==row["ancestry_seed"] and cfg["mutation_seed"]==row["mutation_seed"]
            and done["status"]=="completed_C3_simulation_reconstruction" and done["parent_id"]==parent_id
            and done["class_order"]==list(CLASSES) and done["sample_unit"]=="haploid"
            and observed["parent_id"]==parent_id and observed["split"]=="train"
            and observed["source_observation_ref"]==done["observation_ref"],
            "Same original TRAIN simulation/config/observed genotype provenance before truth payload")
        truth_ref=done["private_truth_ref"]
        expected=str(Path(capability["simulation_completion_ref"]["path"]).parent/"truth/c3_pulse_ancestry.npz")
        require(truth_ref["path"]==expected,"Exact same-parent simulation truth path")
        path=bound(truth_ref)
        keys={"intervals_bp","query_sample_indices","source_classes_0based","parent_id","class_order"}
        with zipfile.ZipFile(path) as z:
            names=z.namelist()
            require(set(names)=={k+".npy" for k in keys} and len(names)==len(keys)
                and sum(i.file_size for i in z.infolist())<=self.cap,"Exact bounded TRAIN truth NPZ members")
        with np.load(path,allow_pickle=False) as z:t={k:z[k] for k in z.files}
        intervals,samples,labels=t["intervals_bp"],t["query_sample_indices"],t["source_classes_0based"]
        require(set(t)==keys and np.array_equal(t["parent_id"],r["parent_id"])
            and np.array_equal(t["class_order"],r["class_order"])
            and intervals.dtype==np.float64 and intervals.ndim==2 and intervals.shape[1]==2
            and len(intervals)>0 and np.isfinite(intervals).all()
            and intervals[0,0]==0 and intervals[-1,1]==LENGTH and (intervals[:,1]>intervals[:,0]).all()
            and np.array_equal(intervals[:-1,1],intervals[1:,0])
            and samples.dtype==np.int64 and np.array_equal(samples,np.arange(600,1600,dtype=np.int64))
            and labels.dtype==np.uint8 and labels.shape==(len(intervals),1000) and (labels<3).all(),
            "Complete original30Mb half-open C3 TRAIN label intervals/query columns")
        positions=r["positions_bp0"];columns=r["sample_indices"]-600
        require(((positions>=0)&(positions<LENGTH)).all() and ((columns>=0)&(columns<1000)).all(),
            "No truth extrapolation/nearest/majority/eligibility change")
        target=np.asarray(labels[np.searchsorted(intervals[:,1],positions,side="right"),columns],dtype=np.int64)
        receipt=dict(schema="C3_original30Mb_external_TRAIN_join_v1",parent_id=parent_id,split="train",
            truth_ref=truth_ref,simulation_metadata_capability=capability,focal_registry_ref=plan["focal_registry_ref"],
            focal_identity_sha256=identity_sha(r),records=10000,class_order=list(CLASSES),
            labels_in_observation_bank=False,purity_filter=False,simulated_true_tree_opened=False)
        return TrainFocalTargetVectorV1(parent_id,dict(plan["focal_registry_ref"]),identity_sha(r),
            torch.tensor(target.copy(),dtype=torch.long,device="cpu"),receipt)

def shared_complete_parent_step_v1(model,provider,targets,optimizer,**kwargs):
    """Reuse numeric CE/PyG/full-BPTT engine; replace its fixed legacy receipt labels."""
    require(isinstance(provider,Shared30MbProviderV1) and provider.plan["sequence_length_bp"]==LENGTH
        and provider.plan["parent_id"] in TRAIN,"Original full30Mb complete-parent numeric engine binding")
    result=checkpointed_complete_parent_step_v1(model,provider,targets,optimizer,**kwargs)
    result.update(condition=CONDITION,training_length_bp=LENGTH,
        numeric_engine="unchanged_checkpointed_complete_parent_step_v1",
        legacy_engine_return_labels_replaced_from_validated_plan=True)
    return result

def adam_steps_v1(optimizer,expected):
    states=list(optimizer.state.values())
    require(states and len(states)==len(optimizer.param_groups[0]["params"])
        and all("step" in s and int(s["step"].item())==expected for s in states),
        "Actual Adam step counter for every learned parameter")
    return dict(parameters=len(states),minimum_step=expected,maximum_step=expected)

def run_train_v1(cfg,*,parent_plan_refs,writer,device,worker_tmp_directory,
        resolve_train_metadata,runtime_guard):
    """Fresh full fixed600 updates; no resume, short substitute, or heldout capability."""
    validate_config_v1(cfg)
    require(set(parent_plan_refs)==set(TRAIN) and callable(getattr(writer,"save",None))
        and callable(runtime_guard) and callable(resolve_train_metadata),"All20 actual plans and root capabilities")
    actual=torch.device(device)
    require(actual.type in ("cpu","cuda") and (actual.type!="cuda" or actual.index is not None),
        "One explicit admitted device")
    runtime_guard("before_model_or_payload")
    providers={};history=[];scheduled=[];safe=None
    try:
        for p in TRAIN:
            providers[p]=Shared30MbProviderV1(parent_plan_refs[p],parent_id=p,
                worker_tmp_directory=worker_tmp_directory,cache_bytes=cfg["CPU_observation_cache_bytes"])
        seals={p:providers[p].seal(resource_check=runtime_guard) for p in TRAIN}
        barrier=dict(schema="C3_original20x30Mb_all_observations_sealed_v1",registry_ref=REGISTRY_REF,
            parents=seals,all20_ready=True,truth_resolver_calls_before_barrier=0,observation_only=True)
        barrier_ref=writer.save("TRAIN_observation_barrier.json",barrier)
        joiner=OriginalTrainTargetJoinV1(resolve_train_metadata,
            max_truth_uncompressed_bytes=cfg["max_truth_uncompressed_bytes"],all_parent_seals=seals)
        targets={p:joiner.join(p,split="train",plan=providers[p].plan) for p in TRAIN}
        target_ref=writer.save("TRAIN_external_target_joins.json",dict(
            parents={p:targets[p].receipt for p in TRAIN},labels_in_observation_bank=False))
        scientific=dict(cfg);scientific.pop("execution_enabled")
        protocol=dict(config=scientific,parent_plan_refs=parent_plan_refs,source_ref=ref(SOURCE),
            original_registry_ref=REGISTRY_REF,observation_barrier_ref=barrier_ref,target_manifest_ref=target_ref)
        protocol_sha=json_sha(protocol)
        runtime_guard("before_fresh_model")
        require(torch.get_default_dtype()==torch.float32 and not torch.is_autocast_enabled(),"FP32 fresh model")
        _seed_fresh(actual)
        with torch.device("cpu"):model=AmericanC3WholeTreeGNNV1(**cfg["model_kwargs"])
        model=model.to(actual);opts=cfg["optimizer"]
        optimizer=torch.optim.AdamW(model.parameters(),lr=opts["lr"],weight_decay=opts["weight_decay"],
            betas=tuple(opts["betas"]),eps=opts["eps"],foreach=False,fused=False)
        for epoch in range(1,31):
            order=list(TRAIN);random.Random(70002+epoch).shuffle(order)
            for p in order:
                result=shared_complete_parent_step_v1(model,providers[p],targets[p],optimizer,device=actual,
                    resource_check=runtime_guard,gradient_clip=1.,focal_chunk_size=cfg["focal_chunk_size"],
                    first_update=(len(history)==0))
                history.append(dict(result,epoch=epoch,global_step=len(history)+1))
                steps=adam_steps_v1(optimizer,len(history))
                safe=_cpu_snapshot(dict(schema=SCHEMA,training_condition=CONDITION,protocol=protocol,
                    protocol_sha256=protocol_sha,contract=model.checkpoint_contract(),model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),rng_state=_rng_state(actual),
                    trained_parent_ids=[q for q in TRAIN if any(r["parent_id"]==q for r in history)],
                    train_sequence_lengths_bp=[LENGTH]*len({r["parent_id"] for r in history}),
                    completed_parent_updates=len(history),completed_epoch=len(history)//20,
                    Adam_steps=steps,history=history,C4_initialization_or_head_slicing=False,
                    observation_barrier_ref=barrier_ref,target_manifest_ref=target_ref))
            if epoch in CHECKPOINT_EPOCHS:
                checkpoint_ref=writer.save("checkpoint_epoch_%03d.pt"%epoch,safe,kind="torch")
                scheduled.append(dict(epoch=epoch,global_step=len(history),checkpoint_ref=checkpoint_ref))
        require(len(history)==600 and len(scheduled)==6,"Complete fixed training budget")
        return writer.save("completed_training.json",dict(schema=SCHEMA,
            status="completed_C3_original20x30Mb_full_training",protocol_sha256=protocol_sha,
            trained_parent_ids=list(TRAIN),train_sequence_lengths_bp=[LENGTH]*20,
            epochs=30,complete_parent_updates=600,focal_exposures=6000000,
            final_checkpoint_ref=scheduled[-1]["checkpoint_ref"],checkpoint_candidates=scheduled,
            checkpoint_rule=cfg["checkpoint_rule"],Adam_steps=adam_steps_v1(optimizer,600),
            observation_barrier_ref=barrier_ref,target_manifest_ref=target_ref,
            VAL_used_for_optimization_or_selection=False,TEST_used_for_optimization_or_selection=False,
            full_history_gradients=True,training_length_transfer=False,accuracy_or_runtime_claim=False))
    except BaseException as error:
        try:
            boundary=writer.save("failure_last_complete_parent.pt",safe,kind="torch") if safe is not None else None
            writer.save("failed_training.json",dict(status="failed_C3_original30Mb_training",
                exception_type=type(error).__name__,last_complete_parent_checkpoint_ref=boundary,
                full_training_budget_complete=False,partial_update_checkpointed=False))
        except BaseException as seal_error:
            if hasattr(error,"add_note"):error.add_note("Failure receipt also failed: "+type(seal_error).__name__)
        raise
    finally:
        for provider in providers.values():provider.release()

def predict_shared_v1(model,forward_loader,reverse_loader,*,split,parent_id,checkpoint_ref,
        training_completion_ref,writer,device,runtime_guard):
    """Unmodified FP32[N,3] same-registry predictions; no truth capability."""
    require((split=="val" and parent_id in VAL) or (split=="test" and parent_id in TEST),
        "Only original heldout parents; original TRAIN000 remains TRAIN")
    done=read(training_completion_ref)
    require(done["status"]=="completed_C3_original20x30Mb_full_training"
        and done["complete_parent_updates"]==600 and done["final_checkpoint_ref"]==checkpoint_ref
        and done["trained_parent_ids"]==list(TRAIN),"Only full-budget prespecified final checkpoint")
    runtime_guard("before_heldout_prediction")
    fd,rd=forward_loader.dataset,reverse_loader.dataset
    require(fd.plan_ref==rd.plan_ref and fd.direction=="forward" and rd.direction=="reverse"
        and fd.plan["parent_id"]==parent_id,"Exact opposite scans of same original heldout plan")
    validate_shared_plan_v1(fd.plan,parent_id=parent_id)
    require(all(getattr(loader,"_iterator",None) is None for loader in (forward_loader,reverse_loader)),
        "No workers start before prediction guards")
    value=torch.load(bound(checkpoint_ref),map_location="cpu",weights_only=True,mmap=True)
    require(type(model) is AmericanC3WholeTreeGNNV1 and model.class_order==CLASSES
        and value["schema"]==SCHEMA and value["training_condition"]==CONDITION
        and value["contract"]==model.checkpoint_contract() and value["completed_parent_updates"]==600
        and value["trained_parent_ids"]==list(TRAIN) and value["C4_initialization_or_head_slicing"] is False
        and value["protocol_sha256"]==done["protocol_sha256"],"Exact fresh original C3 checkpoint family")
    model.load_state_dict(value["model_state"],strict=True);del value
    try:
        with torch.inference_mode():
            prediction,receipt=_streamed_focal_predictions_impl(model,forward_loader,reverse_loader,device=device)
        verify_prediction(prediction,registry(fd.plan["focal_registry_ref"]))
        runtime_guard("before_prediction_seal")
        prediction_ref=writer.save(parent_id+"_predictions.npz",prediction,kind="npz")
        receipt.update(status="sealed_C3_shared_original30Mb_predictions",split=split,
            prediction_ref=prediction_ref,checkpoint_ref=checkpoint_ref,training_completion_ref=training_completion_ref,
            private_truth_opened=False,probabilities_modified=False,probability_floor=False,purity_filter=False)
        return writer.save(parent_id+"_prediction_seal.json",receipt)
    finally:
        drains=[shutdown_dataloader(x) for x in (forward_loader,reverse_loader)]
        require(all(d["drained"] for d in drains),"Drain every owned prediction worker on all exits")
