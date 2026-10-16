"""SOURCE ONLY integration API; private frozen-trainer binding, no global patch."""
from pathlib import Path
import copy,hashlib,importlib.util,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_shared_context_training_source_20261016_v1"
SOURCE=HERE+"/integration_api_v1.py"
TRAINER_REF=dict(path="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/trainer_prefetch_v1.py",
    sha256="b2f08a3e84bd5fb65c5f3d95566ab082fb71a2dd0a826180a10aa518aa35025c")
STEP_REF=dict(path=HERE+"/shared_context_step_v1.py",
    sha256="bfea21ea1f18d01aa0b01644295b539c25ac80e42928a471cf1a8e3d922ab786")

def _require(ok,message):
    if not ok:raise ValueError(message)

def _ref(path):
    p=ROOT/path
    _require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.suffix in (".py",".json")
        and p.stat().st_size<=1048576,"Canonical bounded source/metadata")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def _load(record,name):
    _require(_ref(record["path"])==record,"Exact source before private import")
    spec=importlib.util.spec_from_file_location(name,ROOT/record["path"])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def _caps(record):
    _require(isinstance(record,dict) and set(record)=={"path","sha256"}
        and record["path"].endswith(".json") and _ref(record["path"])==record,"Exact root bounds metadata")
    value=json.loads((ROOT/record["path"]).read_text())
    expected={"schema","active","root_authorizer","phase","step_source_ref","precision","autocast",
        "TF32","memory_caps","execution_authority_granted"}
    _require(set(value)==expected and value["schema"]=="C3_shared_context_tensor_caps_v1"
        and value["active"] is True and value["root_authorizer"]=="root_agent" and value["phase"]=="train"
        and value["step_source_ref"]==STEP_REF and value["precision"]=="fp32"
        and value["autocast"] is False and value["TF32"] is False
        and value["execution_authority_granted"] is False,"Root prospective bounds only; separate admission required")
    keys={"max_context_cpu_bytes","max_metadata_and_boundary_cpu_bytes","max_block_copy_bytes",
        "max_context_device_copy_bytes"}
    _require(isinstance(value["memory_caps"],dict) and set(value["memory_caps"])==keys
        and all(type(v)is int and v>0 for v in value["memory_caps"].values()),"Explicit positive root tensor caps")
    return value

def make_training_callable_v1(memory_caps_ref):
    """Root-deferred: requires original guarded admission/device before invocation.

    Returns the original 600-update/30-epoch trainer with only its private step
    binding and protocol description changed. All model/data/optimizer/checkpoint
    candidate logic and run_train_v1 codeobject remain original. New source refs
    and caps are sealed into its protocol, so old-family resumes are rejected.
    """
    caps_value=_caps(memory_caps_ref)
    step=_load(STEP_REF,"_C3_shared_context_step_private")
    trainer=_load(TRAINER_REF,"_C3_shared_context_frozen_trainer_private")
    original_run_code=trainer.run_train_v1.__code__
    original_signature=trainer._signature
    source_binding=dict(schema="C3_shared_context_same_objective_execution_binding_v1",
        original_trainer_ref=TRAINER_REF,original_step_ref=step.ORIGINAL_STEP_REF,
        step_source_ref=STEP_REF,integration_source_ref=_ref(SOURCE),memory_caps_ref=copy.deepcopy(memory_caps_ref),
        objective="unchanged_all10000_focal_mean_CE_complete_parent",
        recurrence="unchanged_full_directional_state_carry_no_detach_or_truncation",
        encoder_gradient="sum_block_J_encoder_transpose_times_both_direction_plus_direct_context_adjoints",
        model_optimizer_schedule_candidate_selection_unchanged=True,
        no_cross_update_learned_context_cache=True,precision="fp32",autocast=False,TF32=False,
        architecture_change=False,private_module_globals_only=True,current_running_job_modified=False)
    trainer.SOURCES=tuple(dict.fromkeys((*trainer.SOURCES,STEP_REF["path"],SOURCE,memory_caps_ref["path"])))
    def signature(cfg,parent_plan_refs):
        value,_=original_signature(cfg,parent_plan_refs)
        value["exact_gradient"]="shared_CPU_context_adjoint_then_original_encoder_VJP_without_recurrent_detach"
        value["same_objective_execution_binding"]=copy.deepcopy(source_binding)
        return value,trainer.json_sha(value)
    def bound_step(model,provider,targets,optimizer,**kwargs):
        return step.shared_context_complete_parent_step_v1(model,provider,targets,optimizer,
            memory_caps=copy.deepcopy(caps_value["memory_caps"]),**kwargs)
    trainer._signature=signature
    trainer.checkpointed_complete_parent_step_v1=bound_step
    def run_train_v1(*args,**kwargs):
        _require(_caps(memory_caps_ref)==caps_value and _ref(STEP_REF["path"])==STEP_REF
            and _ref(TRAINER_REF["path"])==TRAINER_REF,"Same future source/caps family on invocation")
        _require(trainer.run_train_v1.__code__ is original_run_code,"Original full trainer body")
        return trainer.run_train_v1(*args,**kwargs)
    run_train_v1.execution_binding=copy.deepcopy(source_binding)
    return run_train_v1
