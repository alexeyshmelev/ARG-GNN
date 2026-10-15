"""Minimal current-credit-v3 root context for original FLARE scientific stages."""
from pathlib import Path
import datetime,hashlib,json,os
from arggnn import american_c3_common_v1 as common
ROOT=common.ROOT
SOURCE="research/reviews/FLARE_existing_JDK_source_v1/root_context_v1.py"
GUARD_REF=dict(path="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py",
    sha256="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef")
POLICY_REF=dict(path="research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json",
    sha256="a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",
    sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
require=common.require
class WaitWriter(common.Writer):
    def json(self,name,value):
        if name!="native_actual_wait.json":return super().json(name,value)
        # Original runner reaches here only after direct wait and owned-group ESRCH.
        # Its full pre-Popen planned growth includes16MiB fixed receipt allowance.
        require(value["owned_process_group_ESRCH"] is True and value["actual_direct_native_wait"] is True,
                "Only actual terminal native wait may use its precharged persistence slot")
        data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
        require(len(data)<=65536,"Precharged compact native wait receipt")
        path=self.path/name;require(not path.exists(),"Exclusive actual-wait evidence")
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
        record=common.ref(path)
        self.adapter.budget_check_v9(self.storage,boundary=True)
        return record

def execute_v1(admission_ref,*,adapter):
    """No reservation/scheduler. Root configures exact unchanged adapter and admits."""
    spec=common.read(admission_ref);cfg=common.read(spec["config_ref"])
    require(spec["schema"]=="FLARE_existing_JDK_current_v3_root_admission_v1"
        and spec["root_authorizer"]=="root_agent" and spec["execution_authorized"] is True
        and spec["GPU_authorized"] is False and cfg["execution_enabled"] is True
        and spec["source_ref"]==common.ref(SOURCE) and spec["stage"] in ("build","convert","native","seal","score"),
        "Fresh explicit root stage authority")
    require(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Exact CPU-only project execution")
    expiry=datetime.datetime.fromisoformat(spec["expires_at_utc"].replace("Z","+00:00"))
    require(expiry.tzinfo is not None and datetime.datetime.now(datetime.timezone.utc)<expiry,"Root stage authority expiry")
    require(spec["storage_adapter_ref"]==GUARD_REF and common.ref(GUARD_REF["path"])==GUARD_REF
        and Path(adapter.__file__).resolve()==ROOT/GUARD_REF["path"] and adapter.SOURCE==GUARD_REF["path"]
        and spec["completed_overlay_policy_ref"]==POLICY_REF
        and spec["marker_repair_authority_ref"]==REPAIR_REF,"Actual unchanged current-credit-v3 helper/policy/repair")
    storage=common.read(spec["storage_ref"])
    require(storage["accounting_mode"]=="concurrent_completed_scope_credits_v3"
        and storage["completed_overlay_policy_ref"]==POLICY_REF
        and storage["budget_components"]["consumer_adapter"]==GUARD_REF,"Current root storage only")
    bindings=storage["reservation"]["consumer_bindings"]
    required=[spec["source_ref"],spec["stage_source_ref"],spec["config_ref"],spec["runtime_ref"],
        spec["source_manifest_ref"],GUARD_REF,POLICY_REF,REPAIR_REF,*spec["mandatory_helper_refs"],*spec["source_closure_refs"]]
    if spec["stage"]=="build":required.append(spec["installed_JDK_qualification_ref"])
    else:required += [cfg["count_manifest_ref"],cfg["registration_ref"]]
    if spec["stage"]=="native":required.append(cfg["binary_provenance_ref"])
    require(len(bindings)<=64 and all(r in bindings for r in required),"Every actual source/runtime/helper/authority/count ref reserved")
    manifest=common.read(spec["source_manifest_ref"])
    require(all(r in bindings for r in manifest["required_metadata_refs"]),"Actual source-plan/tool/count metadata bindings")
    for path,digest in manifest["source_sha256"].items():
        require(common.ref(path)["sha256"]==digest,"Frozen actual source closure")
    require(spec["source_closure_refs"]==[dict(path=p,sha256=h) for p,h in manifest["source_sha256"].items()],
            "Exact complete ordered source closure")
    runtime=common.read(spec["runtime_ref"])
    actual=common.runtime_report()
    require(runtime["prefix"]==actual["prefix"] and runtime["CUDA_VISIBLE_DEVICES"]==""
        and all(actual["packages"].get(k)==v for k,v in runtime.get("packages",{}).items()),
        "Existing root-qualified CPU runtime")
    adapter.validate_storage_v9(spec,cfg,storage,source_ref=spec["source_ref"],
        config_ref=spec["config_ref"],runtime_ref=spec["runtime_ref"])
    if spec["stage"]=="build":
        from research.reviews.FLARE_existing_JDK_source_v1 import build_v1
        require(spec["stage_source_ref"]==common.ref(build_v1.SOURCE),"Actual installed-JDK builder binding")
        return build_v1.execute_v1(spec,storage,guard=lambda **kw:adapter.budget_check_v9(storage,**kw))
    count=common.read(cfg["count_manifest_ref"]);registration=common.read(cfg["registration_ref"])
    require(registration["root_authorizer"]=="root_agent" and registration["observation_ref"]==count["source_observation_ref"]
        and registration["scientific_role"]==cfg["scientific_role"]
        and registration["focal_registry_ref"]==cfg["focal_registry_ref"],"Root registration binds exact observation/focals/role")
    require(count["schema"]=="American_C3_full30Mb_observation_count_root_v1" and count["truth_opened"] is False
        and type(count["num_sites"]) is int and count["num_sites"]>0
        and count["num_samples"]==1600 and count["sequence_length"]==30000000
        and cfg["max_sites_bound"]==count["num_sites"],"Actual full30Mb observed count closure")
    observed=common.read(count["source_observation_ref"])
    role=cfg["scientific_role"]
    require(observed["query_ancestry_truth_in_inputs"] is False
        and ((observed["split"]=="train" and role=="TRAIN_full30Mb_inference_canary_development_not_heldout")
             or (observed["split"] in ("val","test") and role=="registered_heldout_full30Mb_comparison")),
        "TRAIN canary and heldout comparison require distinct actual observation/count registrations")
    if spec["stage"]!="convert":
        if spec["stage"]=="native":conversion=common.read(cfg["conversion_ref"])
        elif spec["stage"]=="seal":
            native=common.read(cfg["native_run_ref"]);conversion=common.read(native["conversion_ref"])
        else:
            sealed=common.read(cfg["sealed_prediction_completion_ref"])
            conversion=common.read(sealed["conversion_ref"])
        require(conversion["observation_ref"]==count["source_observation_ref"]
            and conversion["focal_registry_ref"]==cfg["focal_registry_ref"]
            and conversion["input_sites"]==count["num_sites"],"Every stage uses the same observed data/focals/counts")
    writer=WaitWriter(dict(spec,storage=storage),adapter)
    if spec["stage"]=="convert":
        from arggnn import american_c3_flare_convert_v1 as module
        require(cfg["observation_ref"]==count["source_observation_ref"] and cfg["external_truth_ref"] is None,
                "Same exact observed data; no query truth")
    elif spec["stage"]=="native":
        from arggnn import american_c3_flare_run_v2 as module
    else:
        from arggnn import american_c3_flare_evaluate_v1 as module
        require(cfg["mode"]==("seal_native" if spec["stage"]=="seal" else "score"),"Exact original sealing/scoring mode")
        if spec["stage"]=="seal":require(cfg["external_truth_ref"] is None,"Seal before private truth")
    require(spec["stage_source_ref"]==common.ref(module.SOURCE),"Actual unchanged scientific producer")
    return module.execute(cfg,writer)
