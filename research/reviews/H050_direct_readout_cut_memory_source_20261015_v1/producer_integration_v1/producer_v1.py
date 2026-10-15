"""H050 scientific TRAIN/EVALUATE producer; default inert, no launcher or admission.

Root supplies its reviewed GPU framework context, guards, reader/writer and lease.
The H049 root driver/context/dispatcher remain frozen and are not invoked as H050.
"""
from __future__ import annotations
from collections import OrderedDict
from itertools import combinations
from pathlib import Path
import argparse,hashlib,importlib.util,io,json,random,statistics,time
import torch
from arggnn import train_ordered_clade_episode_v1 as h45
from arggnn import H040_latent_clade_execution_v1 as shared
from .. import direct_readout_cut_memory_v1 as model
from .. import integration_hooks_v1 as hooks
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1 import producer_v1 as reviewed_worker

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/producer_v1.py"
ARMS,TRAIN,VAL,PASSES=model.ARMS,hooks.TRAIN,hooks.VAL,hooks.PASSES
RESOURCE_CAPS=("CPU_bank_cap_bytes","CPU_target_cap_bytes","episode_payload_cap_bytes",
    "episode_cache_cap_bytes","bundle_cache_cap_bytes","GPU_bundle_cap_bytes",
    "GPU_residency_cap_bytes","checkpoint_file_cap_bytes","output_cap_bytes",
    "max_cpu_rss_bytes","max_cuda_reserved_bytes","log_cap_bytes","fixed_metadata_cap_bytes",
    "qualification_allowance_bytes")
require=model.require


def actual_ref(path):
    p=ROOT/path
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def validate_protocol(cfg,*,active=True):
    require(cfg.get("schema")=="H050_scientific_execution_config_v1"
        and cfg.get("hypothesis_id")=="H050"
        and cfg.get("scientific_protocol")==hooks.fixed_protocol_v1()
        and cfg.get("execution_enabled") is active, "Exact H050 four-arm full-budget science")
    require(cfg.get("warmstart_checkpoint_ref")==model.WARM_CHECKPOINT_REF
        and cfg.get("warmstart_completion_ref")==model.WARM_COMPLETION_REF
        and cfg.get("producer_ref")==actual_ref(SOURCE), "Exact producer and strong H045 warmstart")
    if active:
        require(cfg.get("root_resource_caps_reviewed") is True
            and all(type(cfg.get(k)) is int and cfg[k]>0 for k in RESOURCE_CAPS),
            "Actual root-reviewed positive operational caps")
    return cfg


def bind_registration(cfg,reader):
    registered=reader.decode_json(cfg["registration_ref"])
    require(registered.get("schema")=="H050_direct_readout_cut_memory_registration_v1"
        and registered.get("hypothesis_id")=="H050"
        and registered.get("status")=="registered_before_execution"
        and registered.get("scientific_protocol")==hooks.fixed_protocol_v1(),
        "Actual H050 preregistration before execution")
    return cfg["registration_ref"]


def state_hash(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class BundleCache:
    """Reuse H045 episode cache and genuine corrected cut recipes; RAM-only LRU."""
    def __init__(self,ctx,reader,episode_cache,keys):
        self.ctx,self.reader,self.episodes,self.keys=ctx,reader,episode_cache,keys
        self.values,self.proofs=OrderedDict(),{}
        self.bytes=self.peak=self.builds=self.hits=0
    def get(self,pid,bank):
        self.ctx["guard"]()
        if pid in self.values:
            value=self.values[pid];value.validate(bank);self.values.move_to_end(pid);self.hits+=1
            return value
        episodes=self.episodes.get(pid,bank,self.keys[pid])
        prepared=self.ctx["cut_preparation"]["parents"][pid]
        receipt=self.reader.decode_json(prepared["receipt_ref"])
        require(receipt.get("lossless_roundtrip_verified") is True
            and receipt.get("targets_read") is False
            and receipt.get("original_observed_entry_ref")==self.ctx["entry_refs"][pid],
            "Genuine same-parent observation-only corrected recipe")
        self.ctx["before_cut_decode"](pid,bank,receipt)
        value=model.bind_compact_recipe_v1(bank,episodes,self.episodes.proofs[pid]["metadata"],
            self.reader.read(prepared["arrays_ref"]),receipt)
        size=value.tensor_bytes();require(size<=self.ctx["cfg"]["bundle_cache_cap_bytes"],"One complete bundle cap")
        proof=dict(cut_recipe=prepared,cut_metadata=receipt["cut_metadata"],
            H045_episode_recipe=self.episodes.proofs[pid],extra_tensor_bytes=size,
            original_observed_entry_ref=self.ctx["entry_refs"][pid],targets_read=False)
        if pid in self.proofs:require(proof==self.proofs[pid],"Same immutable rebuilt observed bundle")
        else:self.proofs[pid]=proof
        while self.values and self.bytes+size>self.ctx["cfg"]["bundle_cache_cap_bytes"]:
            _,old=self.values.popitem(last=False);self.bytes-=old.tensor_bytes()
        self.values[pid]=value;self.bytes+=size;self.peak=max(self.peak,self.bytes);self.builds+=1
        return value


def load_observations(ctx,reader,ids):
    episodes=h45.EpisodeCache(ctx["cfg"],ctx["guard"],ctx["before_episode_build"])
    # This unchanged H045 component receipt identifies the reused episode routine.
    banks,parents,keys,episode_seal=h45.load_banks(ctx,reader,ids,episodes)
    cache=BundleCache(ctx,reader,episodes,keys)
    for pid in ids:cache.get(pid,banks[pid])
    require(set(cache.proofs)==set(ids),"All phase observation recipes before any phase truth")
    seal=ctx["output"].save("all_%d_H050_observation_recipes.json"%len(ids),dict(
        schema="H050_observation_barrier_v1",hypothesis_id="H050",parent_ids=list(ids),
        H045_episode_component_ref=episode_seal,cut_preparation_completion_ref=ctx["cut_preparation_completion_ref"],
        cut_preparation_actual_wait_ref=ctx["cut_preparation_actual_wait_ref"],recipes=cache.proofs,
        observation_cache_ref=ctx["observation_cache_ref"],targets_read=False,all_original_trees_retained=True))
    return banks,parents,cache,seal


def device_parent(ctx,bundle,mass=None):
    ctx["guard"]();bundle.validate()
    require(bundle.tensor_bytes()<=ctx["cfg"]["GPU_bundle_cap_bytes"],"Complete GPU episode/cut cap")
    total=h45.h038.tensor_bytes_v1(bundle.bank)+bundle.tensor_bytes()+(0 if mass is None else mass.numel()*4)
    require(total<=ctx["cfg"]["GPU_residency_cap_bytes"],"One shared original whole-parent residency")
    return bundle.to(ctx["device"]),None if mass is None else mass.to(ctx["device"])


def scheduled_step(arm,models,optimizers,bundle,mass,ctx,steps,*,first):
    chosen=models[arm];gradient_checks={};capture={};handles=[]
    before={k:p.detach().cpu().clone() for k,p in chosen.named_parameters()} if first else {}
    if first:
        def capture_logits(_module,_inputs,value):
            capture["logits"]=value.detach().index_select(1,bundle.bank.query_column).cpu()
        handles.append(chosen.classifier.register_forward_hook(capture_logits))
        for name,parameter in chosen.named_parameters():
            def record(gradient,key=name):
                gradient_checks[key]=dict(dtype=str(gradient.dtype),finite=bool(torch.isfinite(gradient).all()))
                return gradient
            handles.append(parameter.register_hook(record))
    def counted():steps[arm]+=1
    try:
        row=hooks.train_step_v1(arm,chosen,optimizers[arm],bundle,mass,ctx["guard"],
            completed_steps=steps[arm],after_optimizer_step=counted)
    finally:
        for handle in handles:handle.remove()
    if first:
        named=dict(chosen.named_parameters())
        require(set(gradient_checks)==set(named) and all(v["dtype"]=="torch.float32" and v["finite"]
            for v in gradient_checks.values()),"Every trainable tensor has finite FP32 gradient on scheduled step1")
        require(len(optimizers[arm].state)==len(named)
            and all(int(v["step"].item())==1 for v in optimizers[arm].state.values()),"Actual fresh full-model AdamW step1")
        changed={k:not torch.equal(p.detach().cpu(),before[k]) for k,p in named.items()}
        require(any(v for k,v in changed.items() if not k.startswith("memory.")),"Actual trained H045 body updates")
        row["full_model_qualification"]=dict(parameters=model.PARAMETERS[arm],gradient_checks=gradient_checks,
            updated_parameters=changed,actual_optimizer_step=steps[arm],no_extra_updates=True,
            warmstart_checkpoint_ref=model.WARM_CHECKPOINT_REF,scientific_accuracy_measured=False)
        require("logits" in capture,"Initial predictor captured from actual scheduled forward")
    return row,capture.get("logits")


def save_checkpoint(ctx,input_ref,arm,at,models,optimizers,steps,hashes,seal):
    ctx["guard"]()
    cp=dict(schema="H050_direct_readout_checkpoint_v1",hypothesis_id="H050",phase="train",arm=arm,
        pass_index=at,additional_optimizer_steps=steps[arm],seed=101,parameter_count=model.PARAMETERS[arm],
        model_state={k:v.detach().cpu() for k,v in models[arm].state_dict().items()},
        optimizer_state=optimizers[arm].state_dict(),optimizer_state_reset=True,teacher_used=False,
        initial_state_sha256=hashes[arm],input_ref=input_ref,config_ref=ctx["config_ref"],
        observation_seal=seal,registration_ref=ctx["cfg"]["registration_ref"],producer_ref=ctx["cfg"]["producer_ref"],
        scientific_protocol=hooks.fixed_protocol_v1(),warmstart_checkpoint_ref=model.WARM_CHECKPOINT_REF,
        warmstart_completion_ref=model.WARM_COMPLETION_REF,
        cut_preparation_completion_ref=ctx["cut_preparation_completion_ref"],
        cut_preparation_actual_wait_ref=ctx["cut_preparation_actual_wait_ref"])
    buffer=io.BytesIO();torch.save(cp,buffer)
    require(buffer.tell()<=ctx["cfg"]["checkpoint_file_cap_bytes"],"Actual checkpoint serialization cap")
    return ctx["output"].save("%s.pass_%02d.pt"%(arm,at),cp,"torch")


def run_train(ctx,input_ref):
    from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1
    output,guard=ctx["output"],ctx["guard"];reader=h45.PhaseReader(ctx["reader"],"train",ctx["bindings"])
    steps={a:0 for a in ARMS};history=[];groups={};qualified={}
    try:
        banks,parents,cache,seal=load_observations(ctx,reader,TRAIN)
        completion=reader.decode_json(model.WARM_COMPLETION_REF)
        payload=torch.load(io.BytesIO(reader.read(model.WARM_CHECKPOINT_REF)),map_location="cpu",weights_only=False)
        guard();models,optimizers,proof=hooks.build_family_v1(payload,completion,
            checkpoint_ref=model.WARM_CHECKPOINT_REF,completion_ref=model.WARM_COMPLETION_REF,device=ctx["device"])
        del payload
        hashes={a:state_hash(m.state_dict()) for a,m in models.items()}
        require(len({hashes[a] for a in model.MEMORY_ARMS})==1,"Three identical initial memory-model states")
        reader.truth_gate=True;targets={};target_bytes=0;helpers=h45.fit.metric_helpers_v1()
        for pid in TRAIN:
            guard();target=FullPanelExternalTargetV1(parents[pid],ctx["bindings"][pid],"TRAIN_loss",reader)
            mass=h45.h038.native_hold_train_mass_v1(parents[pid],banks[pid],target,helpers)
            target_bytes+=mass.nbytes;require(target_bytes<=ctx["cfg"]["CPU_target_cap_bytes"],"Original CPU target cap")
            targets[pid]=torch.from_numpy(mass)
        output.save("initialization.json",dict(hypothesis_id="H050",initialization=proof,
            initial_state_sha256=hashes,observation_seal=seal,parameters=model.PARAMETERS,
            optimizer_moments_reset=True,VAL_influences_optimization=False))
        for at in range(1,31):
            order=list(TRAIN);random.Random(101+at).shuffle(order)
            means={a:0. for a in ARMS};started=time.monotonic()
            for number,pid in enumerate(order,1):
                bundle,mass=device_parent(ctx,cache.get(pid,banks[pid]),targets[pid]);first=at==1 and number==1
                reference=None
                for arm in ARMS:
                    require(all(p.grad is None for m in models.values() for p in m.parameters()),"No cross-arm gradient lifetime")
                    row,initial=scheduled_step(arm,models,optimizers,bundle,mass,ctx,steps,first=first)
                    means[arm]+=row["loss"]/32.
                    if first:
                        if reference is None:reference=initial
                        require(initial.shape==reference.shape and torch.allclose(initial,reference,rtol=1e-5,atol=1e-6),
                            "All4 initial predictors equal the original H045 continuation on scheduled forwards")
                        qualified[arm]=dict(row,parent_id=pid,part_of_continuous_960_updates=True,
                            initial_predictor_max_abs_difference=float((initial-reference).abs().max()))
                    del initial
                if first:output.save("first_real_parent_step_qualification.json",dict(hypothesis_id="H050",
                    arms=qualified,all4_common_initial_predictors_verified=True,no_extra_forwards_or_updates=True))
                del reference,bundle,mass
                if number%8==0:print(json.dumps(dict(stage="H050_TRAIN",pass_index=at,parents=number,steps=steps)),flush=True)
            require(steps=={a:32*at for a in ARMS},"32 scheduled complete-parent updates perarm/pass")
            row=dict(pass_index=at,steps=steps.copy(),full_parent_updates_per_arm=32,mean_TRAIN_NLL=means,
                parent_order=order,seconds=time.monotonic()-started)
            history.append(row);output.save("TRAIN_pass_%02d.json"%at,row)
            if at in PASSES:
                groups[str(at)]={a:save_checkpoint(ctx,input_ref,a,at,models,optimizers,steps,hashes,seal) for a in ARMS}
        require(steps=={a:960 for a in ARMS} and sum(map(len,groups.values()))==24,"All3840updates/all24checkpoints")
        result=dict(schema="H050_four_arm_train_completion_v1",hypothesis_id="H050",status="completed_fixed30pass_four_arm_TRAIN",
            input_ref=input_ref,config_ref=ctx["config_ref"],arms=list(ARMS),steps=steps,history=history,
            checkpoint_groups=groups,first_step_qualification=qualified,initial_state_sha256=hashes,
            common_initial_predictor_verified=True,observations_seal=seal,parameters_per_arm=model.PARAMETERS,
            registration_ref=ctx["cfg"]["registration_ref"],producer_ref=ctx["cfg"]["producer_ref"],
            scientific_protocol=hooks.fixed_protocol_v1(),warmstart_checkpoint_ref=model.WARM_CHECKPOINT_REF,
            warmstart_completion_ref=model.WARM_COMPLETION_REF,
            cut_preparation_completion_ref=ctx["cut_preparation_completion_ref"],
            cut_preparation_actual_wait_ref=ctx["cut_preparation_actual_wait_ref"],VAL_truth_read=False,outer_truth_read=False)
        output.save("completed.json",result);return result
    except Exception as error:
        output.save("failed.json",dict(hypothesis_id="H050",phase="train",status="failed_partial_preserved",
            logical_steps=steps,completed_checkpoint_groups=groups,error_type=type(error).__name__,error=str(error),
            automatic_replay=False,VAL_truth_read=False,outer_truth_read=False));raise


def load_checkpoints(reader,train,ctx):
    cfg=ctx["cfg"]
    require(train.get("schema")=="H050_four_arm_train_completion_v1" and train.get("hypothesis_id")=="H050"
        and train.get("status")=="completed_fixed30pass_four_arm_TRAIN" and train.get("arms")==list(ARMS)
        and train.get("steps")=={a:960 for a in ARMS} and train.get("common_initial_predictor_verified") is True
        and train.get("VAL_truth_read") is False and train.get("outer_truth_read") is False
        and train.get("scientific_protocol")==hooks.fixed_protocol_v1()
        and train.get("registration_ref")==cfg["registration_ref"] and train.get("producer_ref")==cfg["producer_ref"]
        and train.get("warmstart_checkpoint_ref")==model.WARM_CHECKPOINT_REF
        and train.get("warmstart_completion_ref")==model.WARM_COMPLETION_REF
        and all(train.get(k)==cfg.get(k) for k in ("cut_preparation_completion_ref","cut_preparation_actual_wait_ref"))
        and train.get("parameters_per_arm")==model.PARAMETERS and set(train["first_step_qualification"])==set(ARMS)
        and len(train.get("history",[]))==30 and all(r["pass_index"]==i and r["steps"]=={a:32*i for a in ARMS}
            for i,r in enumerate(train["history"],1)),"Actual complete four-arm scheduled TRAIN")
    train_cfg=reader.decode_json(train["config_ref"]);validate_protocol(train_cfg);validate_protocol(cfg)
    require({k:v for k,v in train_cfg.items() if k not in RESOURCE_CAPS}
        =={k:v for k,v in cfg.items() if k not in RESOURCE_CAPS},"Frozen scientific/provenance config; resource caps alone may differ")
    states,refs={},{}
    require(set(train["checkpoint_groups"])=={str(p) for p in PASSES},"Exactly six scheduled groups")
    for at in PASSES:
        require(set(train["checkpoint_groups"][str(at)])==set(ARMS),"Exactly four checkpoints per group")
        for arm in ARMS:
            record=train["checkpoint_groups"][str(at)][arm]
            cp=torch.load(io.BytesIO(reader.read(record)),map_location="cpu",weights_only=False)
            require(cp.get("schema")=="H050_direct_readout_checkpoint_v1" and cp.get("hypothesis_id")=="H050"
                and cp.get("phase")=="train" and cp.get("arm")==arm and cp.get("pass_index")==at
                and cp.get("additional_optimizer_steps")==32*at and cp.get("seed")==101
                and cp.get("parameter_count")==model.PARAMETERS[arm] and cp.get("optimizer_state_reset") is True
                and cp.get("teacher_used") is False and cp.get("initial_state_sha256")==train["initial_state_sha256"][arm]
                and cp.get("observation_seal")==train["observations_seal"]
                and all(cp.get(k)==train[k] for k in ("input_ref","config_ref","registration_ref","producer_ref",
                    "scientific_protocol","warmstart_checkpoint_ref","warmstart_completion_ref",
                    "cut_preparation_completion_ref","cut_preparation_actual_wait_ref")),"Exact H050 scheduled checkpoint provenance")
            require(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in cp["model_state"].values()),"Finite raw FP32 state")
            states[arm,at],refs[arm,at]=cp["model_state"],record
    return states,refs


def paired_criteria(full):
    result={}
    for control in ARMS[1:]:
        rows=[]
        for pid in VAL:
            cells={a:next(c for c in full[a]["cells"] if c["parent_id"]==pid)["groups"]["primary_present_day_182"]
                for a in (ARMS[0],control)}
            true,other=cells[ARMS[0]],cells[control]
            require(true["boundaries"]["f1"] is not None and other["boundaries"]["f1"] is not None,"Defined Present182 boundaries")
            rows.append(dict(parent_id=pid,accuracy_delta=true["accuracy"]-other["accuracy"],
                boundary_F1_delta=true["boundaries"]["f1"]-other["boundaries"]["f1"]))
        da=statistics.mean(r["accuracy_delta"] for r in rows);df=statistics.mean(r["boundary_F1_delta"] for r in rows)
        result[control]=dict(per_parent=rows,mean_accuracy_delta=da,mean_boundary_F1_delta=df,
            paired_accuracy_sample_SD_pp=100*statistics.stdev(r["accuracy_delta"] for r in rows),
            paired_boundary_F1_sample_SD_pp=100*statistics.stdev(r["boundary_F1_delta"] for r in rows),
            primary=(da>=.001 and df>=0),secondary=(df>=.01 and da>=-.0005))
    return dict(comparisons=result,meets_primary=all(r["primary"] for r in result.values()),
        meets_secondary_boundary=all(r["secondary"] for r in result.values()),one_seed_reused8VAL=True,
        parameter_difference_disclosed=model.PARAMETERS,no_priority_claim=True)


def run_evaluate(ctx,input_ref):
    from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1
    reader=h45.PhaseReader(ctx["reader"],"evaluate",ctx["bindings"]);train=reader.decode_json(ctx["train_completion_ref"])
    states,checkpoints=load_checkpoints(reader,train,ctx)
    banks,parents,cache,seal=load_observations(ctx,reader,VAL)
    ram=hooks.H050RAMMatrixV1(ctx["output"],reader,parents,banks,ctx["cfg"],checkpoint_groups=train["checkpoint_groups"])
    reader.ram=ram
    models={a:(model.OrderedCladeEpisodeGNNV1(**model.MODEL_KWARGS,arm="own_state") if a==model.CONTINUATION
        else model.DirectReadoutCutMemoryGNNV1(memory_arm=a)).to(ctx["device"]).eval().requires_grad_(False) for a in ARMS}
    for pid in VAL:
        bundle,_=device_parent(ctx,cache.get(pid,banks[pid]))
        for arm in ARMS:
            for at in PASSES:
                ctx["guard"]();models[arm].load_state_dict(states[arm,at],strict=True)
                probability=hooks.predict_holds_v1(arm,models[arm],bundle).softmax(-1)
                ram.seal(at,arm,pid,probability.cpu().numpy(),checkpoints[arm,at]);del probability
                print(json.dumps(dict(stage="H050_VAL",sealed_products=len(ram.values),required=192)),flush=True)
        del bundle
    del models,states
    ram.verify();reader.truth_gate=True;ctx["VAL_truth_gate_opened"]=True
    selected,histories,diagnostics,receipts={},{},{},{}
    for arm in ARMS:
        validation=FullPanelPrimaryValidationV1(parents,ctx["bindings"],reader,h45.fit.metric_helpers_v1(),ctx["cfg"])
        rows=[]
        for at in PASSES:
            ctx["guard"]();cells,means=validation.score([(p,ram.records[at,arm,p]) for p in VAL])
            rows.append(dict(pass_index=at,cells=cells,equal_parent_metrics=means,checkpoint=checkpoints[arm,at]))
        best=min(rows,key=lambda r:h45.h038.selection_key_v1(r["equal_parent_metrics"],r["pass_index"]))
        cells,means=validation.full_final([(p,ram.records[best["pass_index"],arm,p]) for p in VAL])
        selected[arm],histories[arm]=best,rows
        diagnostics[arm]=dict(cells=cells,equal_parent_metrics=means,selection_revised=False);receipts[arm]=validation.receipt()
    result=dict(schema="H050_four_arm_VAL_completion_v1",hypothesis_id="H050",status="completed_original8VAL_four_arm_development_selection",
        input_ref=input_ref,train_completion_ref=ctx["train_completion_ref"],observations_seal=seal,
        selected=selected,history=histories,selected_full_diagnostics=diagnostics,metric_receipts=receipts,
        registered_practical_criteria=paired_criteria(diagnostics),parameters_per_arm=model.PARAMETERS,
        scientific_protocol=hooks.fixed_protocol_v1(),registration_ref=ctx["cfg"]["registration_ref"],
        producer_ref=ctx["cfg"]["producer_ref"],complete_probability_matrix_manifest_ref=ram.barrier_ref,
        all192_RAM_seals_before_any_VAL_truth=True,optimization_performed=False,
        native_probability_bytes=ram.bytes,dense_prediction_arrays_saved=False,outer_truth_read=False)
    ctx["output"].save("completed.json",result);return result


def run_context_v1(ctx,input_ref):
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
        and ctx.get("hypothesis_id")=="H050" and ctx.get("arm")=="joint_four_arm"
        and ctx.get("phase") in ("train","evaluate") and ctx.get("physical_gpu") in (4,5)
        and ctx["device"].type=="cuda" and ctx["reader"].root.resolve()==ROOT and all(callable(ctx.get(k)) for k in
            ("guard","before_episode_build","before_cut_decode")),"Actual H050 reviewed-framework GPU root context")
    validate_protocol(ctx["cfg"]);bind_registration(ctx["cfg"],ctx["reader"])
    require(ctx["reader"].decode_json(ctx["config_ref"])==ctx["cfg"]
        and set(ctx["entry_refs"])==set(TRAIN+VAL)
        and set(ctx["bindings"])==set(TRAIN if ctx["phase"]=="train" else VAL),"Exact immutable config/observation/target capabilities")
    h45.science();ctx["guard"]()
    if ctx["phase"]=="train":return run_train(ctx,input_ref)
    try:return run_evaluate(ctx,input_ref)
    except Exception as error:
        ctx["output"].save("failed.json",dict(hypothesis_id="H050",phase="evaluate",status="failed_partial_preserved",
            error_type=type(error).__name__,error=str(error),VAL_truth_gate_opened=ctx.get("VAL_truth_gate_opened",False),
            outer_truth_read=False,automatic_replay=False));raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true")
    p.add_argument("--input");p.add_argument("--expected-input-sha256");args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_no_execution",hypothesis_id="H050",arms=list(ARMS),
            updates=3840,checkpoints=24,pretruth_seals=192,parameters=model.PARAMETERS)));return 0
    require(args.input and args.expected_input_sha256,"Actual root-issued pinned capability")
    record=dict(path=args.input,sha256=args.expected_input_sha256)
    spec=json.loads(reviewed_worker.bound(record))
    require(spec.get("schema")=="H050_direct_readout_root_input_v1" and spec.get("status")=="active"
        and spec.get("hypothesis_id")=="H050" and spec.get("root_authorizer")=="root_agent"
        and spec.get("arm")=="joint_four_arm" and spec.get("phase") in ("train","evaluate")
        and spec.get("physical_gpu") in (4,5) and spec.get("producer_ref")==actual_ref(SOURCE),"Actual separate H050 authority")
    require(spec["source_sha256"].get(SOURCE)==actual_ref(SOURCE)["sha256"],"Actual H050 producer in exact source closure")
    for path,sha in spec["source_sha256"].items():reviewed_worker.bound(dict(path=path,sha256=sha))
    context_ref=spec["root_context_module_ref"]
    require(actual_ref(context_ref["path"])==context_ref
        and spec["source_sha256"].get(context_ref["path"])==context_ref["sha256"],"Exact reviewed root context source")
    loader=importlib.util.spec_from_file_location("_H050_root_context",ROOT/context_ref["path"])
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    ctx=module.build_context_v1(record,spec)
    require(ctx.get("phase")==spec["phase"] and ctx.get("physical_gpu")==spec["physical_gpu"]
        and ctx["cfg"]["producer_ref"]==spec["producer_ref"],"Same actual capability/context/producer")
    return run_context_v1(ctx,record)

if __name__=="__main__":raise SystemExit(main())
