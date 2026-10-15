"""Inactive source-only H044 cached-evaluation successor.

A new actual-worker context/driver/consumer closure is mandatory. Old H044
inputs cannot authorize this producer. Default invocation loads no Torch.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,io,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/H044_cached_evaluation_source_v2/evaluate_cached_v2.py"
CACHE_SOURCE="src/arggnn/marker_resolution_input_cache_v2.py"
FROZEN_SOURCE_REFS=[{'path': 'src/arggnn/marker_resolution_readout_v1.py', 'sha256': '27ecd30683024def2d408bf16250102b3252d0ae112687b9664fca086b141310'}, {'path': 'src/arggnn/train_marker_resolution_readout_v1.py', 'sha256': '386814feb14a64310b8c51c2cfbaf681a59b0273f8be0f72ffde5a222eb08c3c'}]
CACHE_SOURCE_REF={'path': 'src/arggnn/marker_resolution_input_cache_v2.py', 'sha256': 'a1b12a31b1890b84e9033da216ce107bb3ce6c47f687876487514f010e3b1987'}
CACHE_POLICY=dict(schema="H044_evaluation_immutable_input_cache_policy_v2",phase="evaluate",
    limits=dict(maximum_parents=8,maximum_feature_bytes=21_073_920_000),
    query_block=64,release_after_parent=True,targets_cached=False,
    learned_values_cached=False,original_feature_arithmetic=True)

def require(value,message):
    if not value:raise ValueError(message)

def actual_ref(path):
    file=ROOT/path
    require(file.resolve()==file and file.is_file(),"Canonical source file required")
    return dict(path=path,sha256=hashlib.sha256(file.read_bytes()).hexdigest())

def run_evaluate_cached_v2(ctx,input_ref):
    from arggnn import train_marker_resolution_readout_v1 as frozen
    from arggnn.marker_resolution_input_cache_v2 import EvaluationMarkerBlockStoreV2,bind_cached_bundle_v2
    torch,core,h42,h038,fit=frozen.torch,frozen.core,frozen.h42,frozen.h038,frozen.fit
    ARMS,VAL,PASSES,SELECTION,CORE_REF=frozen.ARMS,frozen.VAL,frozen.PASSES,frozen.SELECTION,frozen.CORE_REF
    H044PhaseReaderV1=frozen.H044PhaseReaderV1
    DenseRAMMatrixV1=frozen.DenseRAMMatrixV1
    load_observations_v1=frozen.load_observations_v1
    device_parent_v1=frozen.device_parent_v1
    cache=EvaluationMarkerBlockStoreV2(**ctx["cache_policy"]["limits"])
    from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1
    reader=H044PhaseReaderV1(ctx["reader"],"evaluate",ctx["bindings"],
        virtual_prefix=str(ctx["output"].path.relative_to(ROOT)/"H044_RAM_probability_packets"))
    cfg,output,guard=ctx["cfg"],ctx["output"],ctx["guard"]
    train=reader.decode_json(ctx["train_completion_ref"])
    require(train["schema"]=="H044_marker_paired_train_completion_v1" and train["hypothesis_id"]=="H044"
        and train["status"]=="completed_fixed30pass_paired_TRAIN" and train["arms"]==list(ARMS)
        and train["steps"]=={a:960 for a in ARMS} and train["core_ref"]==CORE_REF
        and train["VAL_truth_read"] is False and train["outer_truth_read"] is False
        and train["preparation_completion_ref"]==ctx["preparation_completion_ref"],
        "Actual complete paired TRAIN with original preparation")
    states,checkpoints={},{}
    for arm in ARMS:
        for at in PASSES:
            guard();ref=train["checkpoint_groups"][str(at)][arm]
            cp=torch.load(io.BytesIO(reader.read(ref)),map_location="cpu",weights_only=False)
            require(cp["schema"]=="H044_marker_checkpoint_v1" and cp["hypothesis_id"]=="H044"
                and cp["arm"]==arm and cp["pass_index"]==at and cp["additional_optimizer_steps"]==32*at
                and cp["parameter_count"]==31428 and cp["seed"]==101 and cp["teacher_used"] is False
                and cp["optimizer_state_reset"] is True and cp["all_H042_weights_trainable"] is True
                and cp["core_ref"]==CORE_REF and cp["readout"]==cfg["readout"]
                and cp["initial_state_sha256"]==train["initial_state_sha256"]
                and cp["input_ref"]==train["input_ref"] and cp["config_ref"]==train["config_ref"]
                and cp["preparation_completion_ref"]==ctx["preparation_completion_ref"]
                and cp["warmstart_checkpoint_ref"]==core.H042_PASS30_REF
                and cp["warmstart_completion_ref"]==core.H042_COMPLETION_REF
                and all(isinstance(v,torch.Tensor) and v.dtype==torch.float32 for v in cp["model_state"].values()),
                "Exact paired FP32 checkpoint provenance")
            states[arm,at],checkpoints[arm,at]=cp["model_state"],ref;del cp
    flags=h42.FlagCache(cfg,guard)
    banks,parents,keys,markers,observation_seal=load_observations_v1(ctx,reader,VAL,flags)
    ram=DenseRAMMatrixV1(ctx,reader,parents,banks);reader.ram=ram
    models={a:core.MarkerResolutionReadoutV1(arm=a,**cfg["readout"]).to(ctx["device"],dtype=torch.float32)
        .eval().requires_grad_(False) for a in ARMS}
    for pid in VAL:
        cpu_bundle=markers.get(pid,banks[pid],parents[pid]);flags_cpu=flags.get(pid,banks[pid],keys[pid])
        bundle,device_flags,_=device_parent_v1(ctx,cpu_bundle,flags_cpu)
        blocks=cache.get(pid,cpu_bundle)
        bundle=bind_cached_bundle_v2(bundle,blocks)
        for arm in ARMS:
            for at in PASSES:
                guard();models[arm].load_state_dict(states[arm,at],strict=True)
                with torch.no_grad(),torch.autocast(device_type="cuda",enabled=False):
                    probabilities=models[arm](bundle,flags=device_flags).softmax(-1)
                ram.seal(arm,at,pid,probabilities.cpu().numpy(),checkpoints[arm,at]);del probabilities
                print(json.dumps(dict(stage="H044_VAL_prediction",sealed_products=len(ram.values),total_products=96)),flush=True)
        del bundle,device_flags,cpu_bundle,flags_cpu,blocks
        cache.release(pid)
    del models,states
    ram.verify();reader.truth_gate=True;ctx["VAL_truth_gate_opened"]=True
    selected,histories,full,metric_receipts={},{},{},{}
    for arm in ARMS:
        validation=FullPanelPrimaryValidationV1(parents,ctx["bindings"],reader,fit.metric_helpers_v1(),cfg)
        rows=[]
        for at in PASSES:
            guard();cells,means=validation.score([(p,ram.records[arm,at,p]) for p in VAL])
            row=dict(pass_index=at,cells=cells,equal_parent_metrics=means,
                checkpoint=checkpoints[arm,at],selection_rule=SELECTION)
            rows.append(row)
        best=min(rows,key=lambda row:h038.selection_key_v1(row["equal_parent_metrics"],row["pass_index"]))
        cells,means=validation.full_final([(p,ram.records[arm,best["pass_index"],p]) for p in VAL])
        full[arm]=dict(cells=cells,equal_parent_metrics=means,selection_revised=False)
        selected[arm],histories[arm],metric_receipts[arm]=best,rows,validation.receipt()
    result=dict(schema="H044_marker_paired_VAL_completion_v1",hypothesis_id="H044",
        status="completed_original8VAL_paired_development_selection",input_ref=input_ref,
        train_completion_ref=ctx["train_completion_ref"],observations_seal=observation_seal,
        selection_rule=SELECTION,selected=selected,history=histories,selected_full_diagnostics=full,
        metric_receipts=metric_receipts,optimization_performed=False,
        all96_RAM_seals_before_any_VAL_truth=True,RAM_dense_probability_bytes=ram.bytes,
        complete_probability_matrix_manifest_ref=ram.barrier_ref,virtual_seals_durable_in_matrix_manifest=True,
        dense_prediction_arrays_saved=False,outer_truth_read=False,same_VAL_is_development_evidence=True)
    result["immutable_input_cache_v2"]=dict(execution_ref=ctx["cached_execution_ref"],
        cache_policy_ref=ctx["cache_policy_ref"],cache=cache.receipt(),
        same_original_probability_products=True,extra_fits=False)
    output.save("completed.json",result);return result

def run_context_v2(ctx,input_ref):
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
        and ctx.get("phase")=="evaluate" and ctx.get("arm")=="joint_pair"
        and callable(ctx.get("guard")),"New admitted evaluation-only root context required")
    require(ctx.get("cache_policy")==CACHE_POLICY and ctx.get("cache_policy_ref") is not None
        and ctx.get("cached_execution_ref")==input_ref
        and ctx.get("train_actual_wait_ref") is not None,
        "Exact immutable input policy and original terminal evidence required")
    for ref in [*FROZEN_SOURCE_REFS,CACHE_SOURCE_REF]:
        require(actual_ref(ref["path"])==ref,"Frozen scientific/cache source changed")
        ctx["reader"].read(ref)
    require(ctx["reader"].decode_json(ctx["cache_policy_ref"])==CACHE_POLICY,
        "Actual cache policy differs from separately admitted policy")
    waited=ctx["reader"].decode_json(ctx["train_actual_wait_ref"])
    require(waited["actual_direct_launcher_wait"] is True and waited["returncode"]==0
        and waited["completed_ref"]==ctx["train_completion_ref"] and waited["owned_group_drained"] is True,
        "Original complete paired TRAIN direct wait0 and owned drain required")
    from arggnn import train_marker_resolution_readout_v1 as frozen
    frozen.validate_protocol_v1(ctx["cfg"])
    require(ctx["reader"].root.resolve()==ROOT and ctx["device"].type=="cuda"
        and ctx["reader"].decode_json(ctx["config_ref"])==ctx["cfg"],
        "Qualified device and unchanged original H044 protocol config")
    require(set(ctx["bindings"])==set(frozen.VAL)
        and set(ctx["entry_refs"])==set(frozen.TRAIN+frozen.VAL),
        "Original eight VAL-only target bindings and forty observed entries")
    frozen.science();ctx["guard"]()
    try:return run_evaluate_cached_v2(ctx,input_ref)
    except Exception as error:
        ctx["output"].save("failed.json",dict(schema="H044_cached_evaluation_failure_v2",
            hypothesis_id="H044",phase="evaluate",error_type=type(error).__name__,error=str(error),
            input_ref=input_ref,VAL_truth_gate_opened=ctx.get("VAL_truth_gate_opened",False),
            outer_truth_read=False,optimization_performed=False,automatic_replay=False))
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256");args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_not_executed",source=SOURCE,phase="evaluate",
            products=96,cache_policy=CACHE_POLICY,new_root_context_driver_closure_required=True)))
        return 0
    require(args.input and args.expected_input_sha256,"Fresh root-issued successor input required")
    path=Path(args.input);path=path if path.is_absolute() else ROOT/path
    require(path.resolve()==path and path.is_relative_to(ROOT),"Canonical root-owned input")
    data=path.read_bytes();require(hashlib.sha256(data).hexdigest()==args.expected_input_sha256,
        "Exact new execution input SHA required")
    record=dict(path=str(path.relative_to(ROOT)),sha256=args.expected_input_sha256)
    spec=json.loads(data)
    require(spec.get("schema")=="H044_cached_evaluation_root_input_v2"
        and spec.get("status")=="active" and spec.get("phase")=="evaluate"
        and spec.get("root_authorizer")=="root_agent" and spec.get("hypothesis_id")=="H044"
        and spec.get("main_source_ref")==actual_ref(SOURCE)
        and spec.get("cache_source_ref")==CACHE_SOURCE_REF
        and spec.get("frozen_source_refs")==FROZEN_SOURCE_REFS
        and spec.get("cache_policy_ref") is not None,
        "Actual source-bound evaluation-only successor capability required")
    context_ref=spec["root_context_module_ref"]
    require(context_ref["path"].startswith("research/reviews/")
        and context_ref["path"].endswith(".py")
        and context_ref["path"] not in (
            "research/reviews/H044_marker_resolution_root_v1/context_factory_v1.py",
            "research/reviews/H044_marker_resolution_root_v1/context_factory_v2.py")
        and actual_ref(context_ref["path"])==context_ref
        and spec.get("root_driver_ref") is not None
        and spec.get("consumer_manifest_ref") is not None,
        "New actual-worker root context/driver/consumer closure required; old context forbidden")
    module_spec=importlib.util.spec_from_file_location("_H044_cached_evaluation_context_v2",ROOT/context_ref["path"])
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    ctx=module.build_context_v2(record,spec)
    require(ctx.get("cached_execution_ref")==record
        and ctx.get("cache_policy_ref")==spec["cache_policy_ref"],"Context preserves exact successor capability")
    return run_context_v2(ctx,record)

if __name__=="__main__":raise SystemExit(main())
