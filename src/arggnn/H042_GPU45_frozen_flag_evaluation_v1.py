"""Evaluation-only H042 runtime successor. Original science is imported unchanged."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
from . import train_anonymous_clade_flag_control_v1 as original

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="src/arggnn/H042_GPU45_frozen_flag_evaluation_v1.py"
CONTEXT="research/reviews/H042_GPU45_frozen_flag_evaluation_source_v1/context_factory_v1.py"
DRIVER="research/reviews/H042_GPU45_frozen_flag_evaluation_source_v1/root_driver.py"
TRAIN_PACKET="research/reviews/H042_GPU45_clade_flags_root_source_v1/admitted_train"
TRAIN_OUTPUT="experiments/results/H042_GPU4_clade_flags_train_seed101_v1"
MEMO_REF={'path': 'research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py', 'sha256': 'c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364'}
V4_REF={'path': 'src/arggnn/project_budget_v4.py', 'sha256': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47'}
ORIGINAL_REF={'path': 'src/arggnn/train_anonymous_clade_flag_control_v1.py', 'sha256': 'cb58ec3f81d81cdcd88637f3ade695a3e3ef97a2d0c10b7713e9bddb43fbb2ed'}
ORIGINAL_CONTEXT_REF={'path': 'research/reviews/H042_GPU45_clade_flags_root_source_v1/context_factory_v1.py', 'sha256': '0c8559045139b56026cab44ee48478a9e59e15bc22886d7206de4e232c97b309'}
ORIGINAL_DRIVER_REF={'path': 'research/reviews/H042_GPU45_clade_flags_root_source_v1/root_driver.py', 'sha256': '5b87957f3667187b97172b00871f3cc5c85981810179554578e9779bcca9d0b0'}
require=original.require

def actual_ref(path):
    return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())

def validate_train_evidence_v1(spec,reader,cfg=None):
    require(spec["train_completion_ref"]["path"]==TRAIN_OUTPUT+"/completed.json"
            and spec["train_launch_ref"]["path"]==TRAIN_PACKET+"/launch.json"
            and spec["train_actual_wait_ref"]["path"]==TRAIN_PACKET+"/actual_wait.json",
            "Exact original active H042 TRAIN completion/launch/direct-wait paths required")
    trained=reader.decode_json(spec["train_completion_ref"])
    launch=reader.decode_json(spec["train_launch_ref"])
    wait=reader.decode_json(spec["train_actual_wait_ref"])
    require(trained.get("schema")=="anonymous_clade_flag_train_completion_v1"
            and trained.get("status")=="completed_fixed_30_pass_flag_TRAIN"
            and trained.get("hypothesis_id")=="H042" and trained.get("control_id")==original.CONTROL_ID
            and trained.get("arm")==original.ARM and trained.get("steps")=={original.ARM:960}
            and set(trained["checkpoint_groups"])=={str(p) for p in original.PASSES}
            and all(set(v)=={original.ARM} for v in trained["checkpoint_groups"].values())
            and len(trained["history"])==30
            and trained.get("VAL_truth_read") is False and trained.get("outer_truth_read") is False
            and trained.get("conditioned_on_H040_VAL") is False
            and trained.get("scientific_refs")==list(original.SCIENTIFIC_REFS),
            "Genuine prospectively fixed30-pass/960-update H042 TRAIN required")
    require(type(wait.get("returncode")) is int and wait["returncode"]==0
            and wait.get("actual_direct_launcher_wait") is True
            and wait.get("completed_ref")==spec["train_completion_ref"]
            and type(launch.get("launcher_pid")) is int and launch["launcher_pid"]>=2
            and wait.get("launcher_pid")==launch["launcher_pid"]
            and launch.get("input_ref")==trained["input_ref"]
            and launch.get("phase")=="train" and launch.get("physical_gpu")==4,
            "Actual same-launcher H042 TRAIN wait0 and completion binding required")
    train_input=reader.decode_json(trained["input_ref"])
    train_cfg=reader.decode_json(trained["config_ref"])
    require(train_input.get("schema")=="anonymous_clade_flag_root_execution_input_v1"
            and train_input.get("phase")=="train" and train_input.get("hypothesis_id")=="H042"
            and train_input.get("main_source_ref")==ORIGINAL_REF
            and train_input.get("root_context_module_ref")==ORIGINAL_CONTEXT_REF
            and train_input["refs"]["config"]==trained["config_ref"],
            "Exact frozen original H042 TRAIN input/config provenance required")
    if cfg is not None:
        restored=dict(cfg,source_manifest_ref=train_cfg["source_manifest_ref"])
        require(restored==train_cfg,"Evaluation may change only source_manifest_ref; every numerical field is frozen")
    return trained,train_input,train_cfg

def validate_capability_v1(spec,reader,cfg=None):
    require(spec.get("schema")=="H042_frozen_flag_evaluation_input_v1"
            and spec.get("root_authorizer")=="root_agent" and spec.get("status")=="active"
            and spec.get("phase")=="evaluate" and spec.get("hypothesis_id")=="H042"
            and spec.get("control_id")==original.CONTROL_ID and spec.get("arm")==original.ARM
            and spec.get("physical_gpu") in (4,5)
            and spec.get("main_source_ref")==actual_ref(SOURCE)
            and spec.get("root_context_module_ref")==actual_ref(CONTEXT)
            and spec.get("root_driver_ref")==actual_ref(DRIVER)
            and spec.get("memo_helper_ref")==MEMO_REF
            and spec.get("transaction_memo_no_throttling") is True,
            "Distinct root-selected GPU4/5 evaluation-only capability and pinned memo helper required")
    needed=[ORIGINAL_REF,ORIGINAL_CONTEXT_REF,ORIGINAL_DRIVER_REF,MEMO_REF,V4_REF,
            actual_ref(SOURCE),actual_ref(CONTEXT),actual_ref(DRIVER)]
    require(all(spec["source_sha256"].get(r["path"])==r["sha256"] for r in needed),
            "Mandatory original and successor source closure pins required")
    evidence=validate_train_evidence_v1(spec,reader,cfg)
    require(all(spec["source_sha256"].get(p)==sha for p,sha in evidence[1]["source_sha256"].items()),
            "Every original TRAIN source pin remains mandatory")
    return evidence

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true");p.add_argument("--input");p.add_argument("--expected-input-sha256")
    args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_inactive",source=SOURCE,phase="evaluate",
            eligible_physical_GPUs=[4,5],TRAIN_execution_available=False,native_RAM_seals=48,memo_helper_ref=MEMO_REF),indent=2));return 0
    require(args.input and args.expected_input_sha256,"Exact root evaluation input path/SHA required")
    record=dict(path=args.input,sha256=args.expected_input_sha256)
    spec=original.metadata(record)
    class Reader:
        def decode_json(self,ref):return original.metadata(ref)
    validate_capability_v1(spec,Reader())
    context_ref=spec["root_context_module_ref"];original.bound(context_ref)
    module_spec=importlib.util.spec_from_file_location("_H042_memo_evaluation_context",ROOT/context_ref["path"])
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    ctx=module.build_context_v1(record,spec)
    require(ctx["phase"]=="evaluate","This successor cannot execute TRAIN")
    original.run_context(ctx,record)
    return 0

if __name__=="__main__":raise SystemExit(main())
