"""Inactive guarded CPU scheduling successor. No guard or scientific import at import."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import copy,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_CPU_interleaved_guarded_integration_source_20261016_v1"
SOURCE=HERE+"/acceptance_contract_v1.py"
DRIVER=HERE+"/root_driver_v1.py"
ENTRY=CPU_MAIN=HERE+"/worker_entry_v1.py"
BUILDER=HERE+"/prepare_evaluation_root_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json"
HANDOFF=HERE+"/source_handoff.v1.json"
LEGACY_PACKAGE="research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1"
LEGACY_CONTRACT_REF=dict(path=LEGACY_PACKAGE+"/acceptance_contract_v1.py",sha256="3b9a60267873f4c84fe1ca03010088df510547c92c1c61cad20dcd4d02f6f1c2")
LEGACY_MANIFEST_REF=dict(path=LEGACY_PACKAGE+"/source_manifest.v1.json",sha256="a76e6eb58fef091dd80ca8ba918c4873b1a31b7f28f28b2e6e84104ec910af35")
LEGACY_HANDOFF_REF=dict(path=LEGACY_PACKAGE+"/source_handoff.v1.json",sha256="d57045ec59223ebbea48efcff9f67550717be84090efa4560855d110b31ee010")
LEGACY_DRIVER_REF=dict(path=LEGACY_PACKAGE+"/root_driver_v1.py",sha256="0e03cdd1d6ff1176385409db46cce550d6a9c7a1e93de3089a5bef3898a01fbd")
LEGACY_ENTRY_REF=dict(path=LEGACY_PACKAGE+"/worker_entry_v1.py",sha256="32a679e2209e173fbca75651aa113b5b5e26abb604b05636a4a0f26bc8b64a5b")
LEGACY_MAIN_REVIEW_REF=dict(path="research/reviews/C3_saved600_artifact_acceptance_independent_review_20261016_v1/review.v1.json",sha256="b89ce51b5fc08420bb7b275a21817146a54791438c4d8fa74ccdf810b4f1fe51")
BUILDER_V2_REF=dict(path="research/reviews/C3_full30Mb_eval_admission_readiness_source_20261016_v2/prepare_evaluation_root_v2.py",sha256="ce7747e562b56849bb9839d40c899a0da802e71c65dff73ced67c5d94632e1ec")
BUILDER_V2_HANDOFF_REF=dict(path="research/reviews/C3_full30Mb_eval_admission_readiness_source_20261016_v2/source_handoff.v1.json",sha256="cdc5770fd6779bb231b05b7657a48335d191d62aec700668131aecf44fbca7fc")
BUILDER_V2_REVIEW_REF=dict(path="research/reviews/C3_full30Mb_eval_admission_readiness_independent_review_20261016_v2/review.v2.json",sha256="81fae191708f9de9264dcd6f1adb9f98c75fb57c52e361ebf55daa1cf3675f6a")
PREP_HELPER_REF=dict(path="research/reviews/C3_saved600_validator_ROOT_preparation_source_20261016_v1/prepare_root_v1.py",sha256="810177e841b7bc2e3ca3ea29332826442122bd28ad0dd917661f13c7cab33781")
SCANNER_REF=dict(path="research/reviews/C3_CPU_six_candidate_interleave_source_20261016_v1/six_candidate_scan_v1.py",sha256="4da2db9c217b7e4cfe25f0b9c9cf6caeaf228b36e8ced978d7d7299af3607255")
BRIDGE_REF=dict(path="research/reviews/C3_CPU_six_candidate_interleave_source_20261016_v1/prediction_bridge_v1.py",sha256="a11553bd356c6926ae44c4fd069d23f6e1bc57fef1271e19698021c12de31b76")
SCANNER_HANDOFF_REF=dict(path="research/reviews/C3_CPU_six_candidate_interleave_source_20261016_v1/source_handoff.v1.json",sha256="79fffb44dbfd55a18d97948b5ae17139cf5b93c80ac56bd0d1ed620bb9d7e89a")
SCANNER_REVIEW_REF=dict(path="research/reviews/C3_CPU_six_candidate_interleave_independent_review_20261016_v1/review.v1.json",sha256="59fe9754d1b453acccf8857f408b4e64288091c234b80e482e923d384f93772b")
SCANNER_INIT_REF={'path': 'research/reviews/C3_CPU_six_candidate_interleave_source_20261016_v1/__init__.py', 'sha256': '3f4b9b86cc002fbcb0961770ee2469fc168a57752e9e633a0197ac60e6164e39'}
SCANNER_MANIFEST_REF=dict(path="research/reviews/C3_CPU_six_candidate_interleave_source_20261016_v1/source_manifest.v1.json",sha256="b2967474d09f888ee079d75c0d0955a3a76c75deaaae9f9c52aa5f837c3948e3")
SAVED_ACCEPTANCE_REF=dict(path="research/reviews/C3_saved600_validator_ROOT_actual_20261016_v1/artifact_acceptance.root_active_v1.json",sha256="5065ce083306a30a7cbf75d42f77b44db5a3d1c121b0b2e5406ac76c22f474ce")
OLD_INPUT_REFS={
 "predict_val_000":dict(path="research/reviews/C3_full30Mb_eval_ROOT_predict_val_000_root01_oct16/input.json",sha256="9318a6c9a0f01487803632894ed92f4287600a5ad808bc88bb2d65a36bead869"),
 "predict_val_001":dict(path="research/reviews/C3_full30Mb_eval_ROOT_predict_val_001_root01_oct16/input.json",sha256="8ac934c50f764358aa1f61347d6832416731d6772fa927503535016280aba2c1"),
 "predict_val_002":dict(path="research/reviews/C3_full30Mb_eval_ROOT_predict_val_002_root01_oct16/input.json",sha256="6b3d657757268b1fb40f5196afe37e781fc47fdad316d6b5085f7e4375f404df")}
def _legacy_v1():
    p=ROOT/LEGACY_CONTRACT_REF["path"]
    if hashlib.sha256(p.read_bytes()).hexdigest()!=LEGACY_CONTRACT_REF["sha256"]:
        raise ValueError("Exact sealed original acceptance contract")
    spec=importlib.util.spec_from_file_location("_C3_interleaved_original_saved600_contract",p)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
legacy=_legacy_v1()
for _name in ("require","local","ref","read","save","load","active","PARENT","VAL","EPOCHS",
    "GPU_PHASES","GPU_MODULE","GUARD_REF","POLICY_REF","REPAIR_REF","RUNTIME_REF","CORE","BASE",
    "CAPS","CORE_SOURCE","FROZEN","BASE_DRIVER_REF","OLD_CONTRACT_REF","OLD_WORKER_REF",
    "ORIGINAL_HISTORY","ACCEPT_PHASE","original_guard","consumer_guard","terminal0_v1",
    "plan_ready_v1","make_config","review_values_v1","acceptance_inputs_v1"):
    globals()[_name]=getattr(legacy,_name)
PHASES=("predict_val_003","predict_val_004","matrix","score")
STATUSES={p:legacy.STATUSES[p] for p in PHASES}
_ACTIVE_SPEC=None
DIRECTIVE="ROOT authorizes one fresh interleaved C3 CPU stage with the fixed mixed producer route and unchanged original evaluation science"

def caps_v1(phase):
    require(phase in PHASES,"Future VAL003/004 prediction or compatible all5VAL matrix/score only")
    return legacy.caps_v1(phase)

def training_ready_v1(record):
    require(record==SAVED_ACCEPTANCE_REF,"Existing actual saved600 acceptance is unchanged")
    return legacy.training_ready_v1(record)

def effective_extra_refs_v1():
    return [ref(SOURCE),ref(DRIVER),ref(ENTRY),ref(BUILDER),ref(HERE+"/__init__.py"),SCANNER_REF,BRIDGE_REF,SCANNER_INIT_REF,SCANNER_MANIFEST_REF,
        LEGACY_CONTRACT_REF,LEGACY_DRIVER_REF,LEGACY_ENTRY_REF,LEGACY_MANIFEST_REF,
        LEGACY_HANDOFF_REF,LEGACY_MAIN_REVIEW_REF,BUILDER_V2_REF,BUILDER_V2_HANDOFF_REF,
        BUILDER_V2_REVIEW_REF,PREP_HELPER_REF,SCANNER_HANDOFF_REF,SCANNER_REVIEW_REF]

def manifest_v1():
    base=legacy.manifest_v1()
    require(ref(legacy.MANIFEST)==LEGACY_MANIFEST_REF,"Unchanged complete original81-source mapping")
    m=read(ref(MANIFEST))
    require(m["schema"]=="C3_CPU_interleaved_guarded_source_manifest_v1"
        and m["source_only"] is True and m["legacy_manifest_ref"]==LEGACY_MANIFEST_REF,
        "Exact independently reviewed effective scheduling successor")
    for key,value in base.items():
        if key not in ("schema","source_sha256"):
            require(m.get(key)==value,"No inherited original manifest contract removed or weakened")
    expected=dict(base["source_sha256"])
    for record in effective_extra_refs_v1():
        require(ref(record["path"])==record,"Exact old/new source or sealed provenance ref")
        require(record["path"] not in expected or expected[record["path"]]==record["sha256"],
            "Never replace an original frozen scientific source")
        expected[record["path"]]=record["sha256"]
    require(m["source_sha256"]==expected,"Complete exact original plus scheduling source closure")
    return m

def integration_review_v1(record):
    value=read(record)
    require(value.get("schema")=="C3_full30Mb_eval_preparation_independent_review_v1"
        and value.get("independent") is True and value.get("verdict") in ("PASS","SOURCE_PASS")
        and value.get("source_ref")==ref(BUILDER)
        and value.get("source_handoff_ref")==ref(HANDOFF)
        and value.get("main_manifest_ref")==LEGACY_MANIFEST_REF
        and value.get("validator_preparation_source_ref")==PREP_HELPER_REF
        and value.get("integration_contract_ref")==ref(SOURCE)
        and value.get("integration_worker_ref")==ref(ENTRY)
        and value.get("integration_driver_ref")==ref(DRIVER)
        and value.get("integration_manifest_ref")==ref(MANIFEST)
        and value.get("reused_builder_v2_ref")==BUILDER_V2_REF
        and value.get("scanner_handoff_ref")==SCANNER_HANDOFF_REF
        and value.get("scanner_review_ref")==SCANNER_REVIEW_REF
        and value.get("fixed_old_prediction_input_refs")==OLD_INPUT_REFS,
        "Independent complete exact integration, mixed route, scanner and builder review")
    return value

def stage_source_v1(env,phase):
    if phase in OLD_INPUT_REFS:
        expected_input=OLD_INPUT_REFS[phase]
        require(env.get("input_ref")==expected_input
            and env.get("root_driver_ref")==LEGACY_DRIVER_REF,
            "Exact already-admitted original VAL000/001/002 input and driver")
        spec=read(expected_input)
        expected_driver,expected_entry=LEGACY_DRIVER_REF,LEGACY_ENTRY_REF
        expected_manifest=LEGACY_MANIFEST_REF
        source_map=legacy.manifest_v1()["source_sha256"]
    else:
        require(phase in PHASES,"No general producer whitelist or fallback")
        spec=read(env["input_ref"])
        expected_driver,expected_entry=ref(DRIVER),ref(ENTRY)
        expected_manifest=ref(MANIFEST)
        source_map=manifest_v1()["source_sha256"]
        require(env.get("root_driver_ref")==expected_driver
            and spec["packet_directory"].startswith("research/reviews/C3_full30Mb_eval_ROOT_interleaved_"),
            "Exact new guarded future producer and separately owned scopes")
    require(spec["phase"]==phase and spec["root_driver_ref"]==expected_driver
        and spec["main_source_ref"]==expected_entry and spec["source_sha256"]==source_map
        and spec["refs"]["runtime"]==RUNTIME_REF,
        "Exact phase-specific unchanged/effective source map and runtime")
    req=read(spec["root_stage_request_ref"])
    require(req["phase"]==phase and req["source_manifest_ref"]==expected_manifest
        and req["training_evidence_ref"]==SAVED_ACCEPTANCE_REF,
        "Same saved600 family and fixed phase-specific request source")
    return spec

def actual_stage_v1(c,e,phase,training_ref):
    require(isinstance(e,dict) and e.get("outer_terminal_ref") is not None,
        "Actual stage completion/direct wait/consumed outer terminal refs required")
    env=c.read(e["completion_ref"]);wait=c.terminal0_v1(e,e["completion_ref"])
    terminal=c.read(e["outer_terminal_ref"])
    require(type(wait.get("returncode")) is int and wait["returncode"]==0
        and type(terminal.get("actual_exit_code")) is int and terminal["actual_exit_code"]==0,
        "Genuine integer stage wait0 and consumed outer0")
    require(env.get("stage")==phase and env.get("status")==legacy.STATUSES[phase]
        and env.get("training_completion_ref")==training_ref,
        "Original saved600 completed phase/training lineage")
    c.stage_source_v1(env,phase)
    return env,c.read(env["result_ref"])

def prediction_metadata_v1(c,parent,six,training_ref,candidates):
    require(six.get("status")=="sealed_C3_one_original_VAL_all6_candidates" and six.get("parent_id")==parent
        and six.get("training_completion_ref")==training_ref and six.get("private_truth_opened") is False
        and len(six["prediction_seal_refs"])==6,"One actual original VAL parent/all6 candidates before truth")
    seen=set()
    for record in six["prediction_seal_refs"]:
        seal=c.read(record);epoch=seal["epoch"]
        require(epoch in candidates and epoch not in seen and seal.get("parent_id")==parent
            and seal.get("status")=="sealed_C3_original_heldout_predictions" and seal.get("split")=="val"
            and seal.get("records")==10000 and seal.get("private_truth_opened") is False
            and seal.get("training_completion_ref")==training_ref
            and seal.get("checkpoint_ref")==candidates[epoch] and seal.get("probabilities_modified") is False,
            "All6 unique original metadata seals, raw probabilities and exact checkpoint lineage")
        seen.add(epoch)
    require(seen==set(c.EPOCHS),"No final-only or missing-checkpoint prediction packet")

def matrix_metadata_v1(c,matrix,training_ref,candidates,plans):
    expected={(e,p) for e in c.EPOCHS for p in c.VAL}
    require(matrix.get("status")=="sealed_C3_all30_original_VAL_predictions"
        and matrix.get("training_completion_ref")==training_ref and matrix.get("all_predictions_sealed_before_VAL_truth") is True
        and matrix.get("TEST_deferred") is True and matrix.get("checkpoint_epochs")==list(c.EPOCHS)
        and matrix.get("validation_parent_ids")==list(c.VAL) and len(matrix["cells"])==30,
        "Actual complete original5VALx6checkpoint pretruth matrix")
    seen=set()
    for cell in matrix["cells"]:
        key=cell["epoch"],cell["parent_id"]
        require(key in expected and key not in seen and cell["checkpoint_ref"]==candidates[key[0]],
            "Unique original30 Cartesian cells and exact checkpoint refs")
        seal=c.read(cell["prediction_seal_ref"])
        require(seal.get("status")=="sealed_C3_original_heldout_predictions" and seal.get("split")=="val"
            and seal.get("epoch")==key[0] and seal.get("parent_id")==key[1] and seal.get("records")==10000
            and seal.get("private_truth_opened") is False and seal.get("probabilities_modified") is False
            and seal.get("training_completion_ref")==training_ref
            and seal.get("checkpoint_ref")==cell["checkpoint_ref"]
            and seal.get("prediction_ref")==cell["prediction_ref"]
            and seal.get("focal_registry_ref")==cell["focal_registry_ref"]==plans[key[1]]["focal_registry_ref"],
            "All30 actual seals and same closed-plan focal registries before score capability")
        seen.add(key)
    require(seen==expected,"Full original30cells required before truth capability")

def evidence_request_fields_v1(c,evidence,phase):
    common={"schema","phase","ROOT_artifact_acceptance_ref","closed_plan_evidence_by_parent"}
    additional=({"VAL_predictions"} if phase=="matrix" else
        {"matrix","VAL_simulation_completion_refs"} if phase=="score" else set())
    require(set(evidence)==common|additional
        and evidence.get("schema")=="C3_full30Mb_eval_actual_stage_evidence_v1"
        and evidence.get("phase")==phase,"Exact phase-specific actual evidence;no extra truth/stage/future capability")
    b,training=c.training_ready_v1(evidence["ROOT_artifact_acceptance_ref"])
    training_ref=b["training_completion_ref"];candidates={x["epoch"]:x["checkpoint_ref"] for x in training["checkpoint_candidates"]}
    supplied=evidence["closed_plan_evidence_by_parent"]
    required_parents=({c.VAL[int(phase[-3:])]} if phase.startswith("predict_val_") else set(c.VAL))
    require(set(supplied)==required_parents,
        "Prediction requires only its own actual closed plan;matrix/score require all5 actual closed full30Mb plans")
    plans={p:c.plan_ready_v1(supplied[p],p) for p in c.VAL if p in required_parents}
    fields=dict(training_evidence_ref=evidence["ROOT_artifact_acceptance_ref"])
    if phase.startswith("predict_val_"):
        parent=c.VAL[int(phase[-3:])];fields["plan_evidence"]=copy.deepcopy(supplied[parent])
    elif phase=="matrix":
        require(set(evidence["VAL_predictions"])==set(c.VAL),"All5 actually completed prediction stages")
        for p,e in evidence["VAL_predictions"].items():
            _,six=actual_stage_v1(c,e,"predict_val_"+p[-3:],training_ref)
            prediction_metadata_v1(c,p,six,training_ref,candidates)
        fields["VAL_predictions"]=copy.deepcopy(evidence["VAL_predictions"])
        fields["plan_evidence_by_parent"]=copy.deepcopy(supplied)
    else:
        _,matrix=actual_stage_v1(c,evidence["matrix"],"matrix",training_ref)
        matrix_metadata_v1(c,matrix,training_ref,candidates,plans)
        simulations=evidence["VAL_simulation_completion_refs"]
        require(set(simulations)==set(c.VAL),"Only5 actual same-parent simulation completion refs supplied by ROOT")
        for p,r in simulations.items():
            sim=c.read(r);obs=c.read(plans[p]["observation_ref"])
            require(sim.get("status")=="completed_C3_simulation_reconstruction" and sim.get("parent_id")==p
                and sim.get("class_order")==["EUR","EAS","AFR"] and obs["source_observation_ref"]==sim["observation_ref"],
                "Actual same-parent simulation metadata;private truth payload is never opened by admission")
        fields.update(matrix=copy.deepcopy(evidence["matrix"]),plan_evidence_by_parent=copy.deepcopy(supplied),
            VAL_truth_capability=copy.deepcopy(simulations))
    return fields,b,training

def validate_request(record):
    req=read(record);active(req);phase=req["phase"]
    require(phase in PHASES,"Never relaunch or migrate original VAL000/001/002")
    common={"schema","active","root_authorizer","execution_authorized","created_at_utc",
        "expires_at_utc","phase","device","GPU_authorized","physical_gpu","runtime_ref",
        "source_manifest_ref","packet_directory","output_directory","caps_ref",
        "training_evidence_ref","integration_review_ref","ROOT_directive_ref","actual_stage_evidence_ref"}
    additional=({"plan_evidence"} if phase.startswith("predict_val_") else
        {"VAL_predictions","plan_evidence_by_parent"} if phase=="matrix" else
        {"matrix","plan_evidence_by_parent","VAL_truth_capability"})
    require(set(req)==common|additional,"Exact phase capabilities, no extra truth or TEST route")
    require(req["schema"]=="C3_full30Mb_evaluation_ROOT_request_v1"
        and req["device"]=="cpu" and req["GPU_authorized"] is False
        and req["physical_gpu"] is None and req["runtime_ref"]==RUNTIME_REF
        and req["source_manifest_ref"]==ref(MANIFEST),
        "Exact current CPU runtime and effective reviewed source")
    require(req["packet_directory"].startswith("research/reviews/C3_full30Mb_eval_ROOT_interleaved_")
        and req["output_directory"].startswith("experiments/results/C3_full30Mb_eval_interleaved_")
        and not local(req["packet_directory"]).is_relative_to(local(HERE)),
        "Fresh separated future output and administrative scopes")
    manifest_v1();integration_review_v1(req["integration_review_ref"])
    caps=read(req["caps_ref"]);require(caps==caps_v1(phase),"Every original cap stays exact")
    directive=read(req["ROOT_directive_ref"])
    require(directive.get("root_authorizer")=="root_agent" and directive.get("active") is True
        and directive.get("execution_authorized") is True
        and directive.get("phase")==phase and directive.get("directive")==DIRECTIVE
        and directive.get("builder_review_ref")==req["integration_review_ref"]
        and directive.get("actual_stage_evidence_ref")==req["actual_stage_evidence_ref"]
        and directive.get("expires_at_utc")==req["expires_at_utc"]
        and directive.get("actual_ROOT_artifact_acceptance_ref")==SAVED_ACCEPTANCE_REF
        and directive.get("main_review_ref")==LEGACY_MAIN_REVIEW_REF
        and directive.get("original_history")==ORIGINAL_HISTORY
        and directive.get("all5VALx6checkpoint_barrier_preserved") is True,
        "Exact separate future ROOT stage directive and reviewed evidence")
    evidence=read(req["actual_stage_evidence_ref"])
    access=SimpleNamespace(**globals())
    fields,accepted,training=evidence_request_fields_v1(access,evidence,phase)
    require(all(req[key]==value for key,value in fields.items()),
        "Request is exactly the actual own-plan/all5-plan predecessor evidence")
    require(req["training_evidence_ref"]==SAVED_ACCEPTANCE_REF,
        "Original actual accepted saved artifacts, never a new training lineage")
    return req,caps,read(RUNTIME_REF)

def validate_input_v1(record):
    spec=read(record);active(spec);req,caps,_=validate_request(spec["root_stage_request_ref"])
    require(spec["schema"]=="C3_full30Mb_evaluation_active_input_v1"
            and spec["phase"]==req["phase"] and spec["root_driver_ref"]==ref(DRIVER)
            and spec["main_source_ref"]==ref(ENTRY) and spec["GPU_authorized"] is False
            and spec["physical_gpu"] is None and spec["output_directory"]==req["output_directory"]
            and spec["packet_directory"]==req["packet_directory"] and spec["refs"]["runtime"]==RUNTIME_REF
            and spec["source_sha256"]==manifest_v1()["source_sha256"],"Actual reviewed CPU producer/source/input/runtime")
    storage=read(spec["refs"]["storage"])
    require(storage["accounting_mode"]=="native14_stopped_retention_v3"
            and storage["completed_overlay_policy_ref"]==POLICY_REF
            and storage["job_id"]==spec["job_id"] and storage["output_directory"]==spec["output_directory"]
            and storage["reserved_new_bytes"]==caps["reservation_cap_bytes"]
            and storage["reservation"]["output"]==str(local(spec["output_directory"]))
            and storage["reservation"]["auxiliary_scopes"]==[
                str(local(spec["packet_directory"]+"/worker_aux")),str(local(spec["packet_directory"]+"/worker_tmp"))]
            and len(storage["reservation"]["consumer_bindings"])<=64
            and all(r in storage["reservation"]["consumer_bindings"] for r in required_direct_refs_v1(spec,req)),
            "Fresh exact current-child per-stage charge and every required direct role")
    require(read(spec["refs"]["config"])==make_config(req,caps),"Exact operational-only config")
    return spec,req,caps

def set_active_spec_v1(record):
    global _ACTIVE_SPEC
    require(_ACTIVE_SPEC is None,"One ROOT supervisor invocation")
    _ACTIVE_SPEC,req,caps=validate_input_v1(record)
    return _ACTIVE_SPEC,req,caps

def phase_paths(phase):
    require(_ACTIVE_SPEC is not None and phase==_ACTIVE_SPEC["phase"],"Exact activated phase")
    return local(_ACTIVE_SPEC["packet_directory"]),_ACTIVE_SPEC["output_directory"]

def required_direct_refs_v1(spec,req):
    # Preserve every original mandatory direct role and every returned guard role.
    # Historical review/handoff provenance is also authenticated through manifest_v1.
    rows=legacy.required_direct_refs_v1(spec,req)+[
        ref(SOURCE),ref(DRIVER),ref(ENTRY),ref(BUILDER),ref(MANIFEST),
        SCANNER_REF,BRIDGE_REF,BUILDER_V2_REF,PREP_HELPER_REF,
        LEGACY_HANDOFF_REF,LEGACY_MAIN_REVIEW_REF,ref(HANDOFF),
        req["integration_review_ref"],req["ROOT_directive_ref"],req["actual_stage_evidence_ref"]]
    indexed={}
    for record in rows:
        require(Path(record["path"]).suffix in (".py",".json",".sh"),
            "Only exact original supported direct roles")
        require(record["path"] not in indexed or indexed[record["path"]]==record,
            "No conflicting original/new direct role")
        indexed[record["path"]]=record
    require(len(indexed)<=64,"Original64 direct-role cap, never prune or raise it")
    return list(indexed.values())
