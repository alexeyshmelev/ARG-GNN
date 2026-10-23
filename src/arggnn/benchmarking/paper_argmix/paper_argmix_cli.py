"""Readable stage CLI; Root supplies actual admission and process supervision.

Local finite byte/time failstops do not constitute resource admission.
No stage starts by importing this module.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from . import paper_argmix_generator as g
from .paper_argmix_features import ScopeGuard


def native_relate(config_path,native_work,output,*,guard):
    """Execute only two directly pinned native argv, with real waits."""
    config=g.metadata(config_path)
    commands_path=g.confined(native_work)/"commands.json"
    commands=g.metadata(commands_path)
    g.need(commands["config_ref"]==g.file_ref(config_path) and len(commands["argv"])==2,"Native command recipe differs")
    installation=g.metadata(config["input_refs"]["relate_installation"]["path"])
    for argv,key in zip(commands["argv"],("Relate","RelateFileFormats")):
        g.need(argv[0]==str(g.verify_ref(installation["binaries"][key])),"Native executable drift")
    for reference in commands["input_refs"].values():
        g.verify_ref(reference)
    output=g._fresh_directory(output,guard)
    returncodes=[]
    started=time.monotonic()
    for index,argv in enumerate(commands["argv"]):
        guard(0)
        print(json.dumps(dict(native_argv=argv,cwd=commands["cwd"])),flush=True)
        with (output/f"native_{index}.log").open("xb") as log:
            result=subprocess.run(argv,cwd=g.confined(commands["cwd"]),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=max(1,config["bounds"]["native_timeout_seconds"]-(time.monotonic()-started)),check=False)
        returncodes.append(result.returncode)
        guard(0)
        if result.returncode:
            break
    receipt=dict(schema="paper_argmix_native_terminal_v1",commands_ref=g.file_ref(commands_path),
                 actual_wait_consumed=True,returncodes=returncodes,seconds=time.monotonic()-started,
                 successful=returncodes==[0,0],logs={p.name:g.file_ref(p) for p in output.iterdir()})
    g.Writes(1024*1024,guard).json(output/"terminal.json",receipt)
    g.need(receipt["successful"],"Native Relate failed; preserve partial files and stop")
    return receipt


def assemble_fit(config_path,stage_root,output,*,guard):
    config=g.metadata(config_path)
    stage_root=g.confined(stage_root)
    inputs=dict(schema="paper_argmix_fit_inputs_v1",config_ref=g.file_ref(config_path),train=[])
    for parent in config["parents"]:
        if parent["split"] not in ("train","val"):
            continue
        path=stage_root/parent["parent_id"]
        record=dict(features=g.file_ref(path/"features/completed.json"),labels=g.file_ref(path/"labels/completed.json"))
        if parent["split"]=="train":
            inputs["train"].append(record)
        else:
            inputs["validation"]=record
    output=g._fresh_directory(output,guard)
    g.Writes(1024*1024,guard).json(output/"fit_inputs.json",inputs)
    return inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    stages=("describe","validate","adopt-train0","generate","prepare-relate","native-relate","finalize-relate","sample","labels","extract","assemble-fit","train","predict","barrier","score","release-scratch","treeloom-observation","treeloom-train","treeloom-predict","joint-barrier","ap-predict")
    parser.add_argument("stage",choices=stages)
    parser.add_argument("--config",default=str(g.CONFIG))
    for name in ("parent-id","parent-output","native-work","terminal-receipt","inferred-output","support-output","fit-manifest","selected-model","feature-manifest","stage-root","prediction-barrier","comparator","observation-seal","other-barrier","spec"):
        parser.add_argument("--"+name)
    parser.add_argument("--prediction",action="append",default=[])
    parser.add_argument("--label",action="append",default=[])
    parser.add_argument("--condition",choices=("equation_corrected","author_key_i","hidden128"),default="equation_corrected")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--cpu-threads",type=int,default=1)
    parser.add_argument("--output")
    parser.add_argument("--max-output-bytes",type=int)
    parser.add_argument("--timeout-seconds",type=int)
    args=parser.parse_args()
    if args.stage=="describe":
        print(json.dumps(dict(recipe=str(g.CONFIG),stages=stages,note="Root must admit jobs; no authority is created by this CLI."),indent=2))
        return
    if args.stage=="validate":
        config,registry=g.checked_contract(args.config)
        import msprime
        for parent in config["parents"]:
            local,_=g.parent_record(args.config,parent["parent_id"])
            g.demographic_model(local,msprime).debug()
        print(json.dumps(dict(status="valid",parents=len(registry["parents"]),roster_haplotypes=len(g.roster(config)),recipe=g.file_ref(args.config)),indent=2))
        return
    g.need(args.output and args.max_output_bytes and args.timeout_seconds,"Explicit output, byte cap and deadline required")
    guard=ScopeGuard(args.output,args.max_output_bytes,args.timeout_seconds)
    if args.stage=="native-relate":
        native_guard=ScopeGuard(args.native_work,g.metadata(args.config)["bounds"]["native_relate_bytes"],args.timeout_seconds)
        def combined(required_write_bytes=0):
            guard(required_write_bytes)
            native_guard(0)
        callback=combined
    else:
        callback=guard
    kwargs=dict(guard=callback)
    if args.stage=="adopt-train0":
        from .paper_argmix_reuse import adopt_train0
        result=adopt_train0(args.config,args.output,**kwargs)
    elif args.stage=="generate":
        result=g.generate_parent(args.config,args.parent_id,args.output,**kwargs)
    elif args.stage=="prepare-relate":
        result=g.prepare_relate(args.config,args.parent_id,args.parent_output,args.output,**kwargs)
    elif args.stage=="native-relate":
        from .paper_argmix_relate import run_parallel
        result=run_parallel(args.config,args.native_work,args.output,**kwargs)
    elif args.stage=="finalize-relate":
        result=g.finalize_relate(args.config,args.parent_id,args.parent_output,args.native_work,args.terminal_receipt,args.output,**kwargs)
    elif args.stage=="sample":
        result=g.sample_targets(args.config,args.parent_id,args.inferred_output,args.output,**kwargs)
    elif args.stage=="labels":
        _,parent=g.parent_record(args.config,args.parent_id)
        if parent["split"] in ("iid","misspec"):
            from .paper_argmix_score import verify_paired_barrier
            g.need(args.prediction_barrier,"Evaluation labels require complete prediction barrier")
            verify_paired_barrier(args.config,args.prediction_barrier)
        result=g.label_targets(args.config,args.parent_id,args.parent_output,args.support_output,args.output,**kwargs)
    elif args.stage=="extract":
        from .paper_argmix_features import extract
        result=extract(args.config,args.parent_id,args.inferred_output,args.support_output,args.output,**kwargs)
    elif args.stage=="assemble-fit":
        result=assemble_fit(args.config,args.stage_root,args.output,**kwargs)
    elif args.stage=="train":
        from .paper_argmix_train import train
        result=train(args.config,args.fit_manifest,args.output,condition=args.condition,device_name=args.device,cpu_threads=args.cpu_threads,**kwargs)
    elif args.stage=="predict":
        from .paper_argmix_train import predict
        result=predict(args.config,args.selected_model,args.feature_manifest,args.output,device_name=args.device,cpu_threads=args.cpu_threads,**kwargs)
    elif args.stage=="ap-predict":
        from .paper_argmix_ap_prediction import run_predictions
        spec=g.metadata(args.spec)
        g.need(g.confined(spec["prediction_spec"]["output_directory"])==g.confined(args.output),"AP CLI output differs from frozen spec")
        result=run_predictions(spec,resource_check=lambda *labels,**values: callback(0))
    elif args.stage=="treeloom-observation":
        from .paper_argmix_treeloom import observation_seal
        result=observation_seal(args.config,args.parent_id,args.parent_output,args.inferred_output,args.support_output,args.output,**kwargs)
    elif args.stage=="treeloom-train":
        from .paper_argmix_treeloom import train
        result=train(args.config,args.stage_root,args.output,device_name=args.device,cpu_threads=args.cpu_threads,**kwargs)
    elif args.stage=="treeloom-predict":
        from .paper_argmix_treeloom import predict
        result=predict(args.config,args.selected_model,args.observation_seal,args.output,device_name=args.device,cpu_threads=args.cpu_threads,**kwargs)
    elif args.stage=="joint-barrier":
        from .paper_argmix_score import paired_barrier
        result=paired_barrier(args.config,args.prediction_barrier,args.other_barrier)
        output=g._fresh_directory(args.output,callback)
        g.Writes(1024*1024,callback).json(output/"barrier.json",result)
    elif args.stage=="release-scratch":
        from .paper_argmix_retention import release
        result=release(args.config,args.parent_id,args.parent_output,args.native_work,args.inferred_output,args.output,**kwargs)
    elif args.stage=="barrier":
        from .paper_argmix_score import prediction_barrier
        result=prediction_barrier(args.config,args.prediction)
        output=g._fresh_directory(args.output,callback)
        g.Writes(1024*1024,callback).json(output/"barrier.json",result)
    else:
        from .paper_argmix_score import score
        result=score(args.config,args.prediction_barrier,args.label,args.output,comparator_path=args.comparator,**kwargs)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
