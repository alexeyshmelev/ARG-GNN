"""Only deferred original full-state migration, never a model or optimizer call."""
from pathlib import Path
import hashlib,importlib.util,io,json,os,stat,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/C3_full_state_migration_CPU_runtime_source_20261016_v1/migration_callback_v1.py"
VALIDATOR_REF={"path":"research/reviews/C3_efficient_resume_migration_source_20261016_v1/checkpoint_migration_validator_v1.py",
 "sha256":"31414598d0f69533864cf600a9a6f0c24ef5f0d81f9ca131fb5f9612f761fafd"}
_RAN=False

def need(ok,message):
    if not ok:raise ValueError(message)

def run_migration_v1(request,*,resource_check,guard,storage):
    """Original validator owns every scientific check and CPU checkpoint read."""
    global _RAN
    need(not _RAN and callable(resource_check),"One explicit original migration callback")
    _RAN=True
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and Path.cwd().resolve()==ROOT,
        "Explicit CPU-only project worker")
    plan=request["checkpoint_byte_plan"];intent_ref=request["migration_intent_ref"]
    expected=plan["copies"]
    need(storage["reserved_new_bytes"]==plan["reserved_new_bytes"]==8*2**20
        and plan["planned_complete_scope_bound_bytes"]<=storage["reserved_new_bytes"],"Actual8MiB complete migration allocation")
    # These are future actual stat checks, never executed by the source author.
    for row in expected:
        resource_check("before_actual_post_terminal_checkpoint_length")
        p=ROOT/row["source_ref"]["path"]
        need(p.resolve()==p and not p.is_symlink(),"Canonical original checkpoint")
        s=p.lstat()
        need(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size==row["actual_source_bytes"],
            "Actual stopped checkpoint bytes match the pre-admission count-derived plan")
    resource_check("before_original_guarded_output_writer")
    output=guard.PilotOutputV9(storage["output_directory"],storage,types.SimpleNamespace(root=ROOT))
    path=ROOT/VALIDATOR_REF["path"]
    need(path.is_file() and path.resolve()==path and hashlib.sha256(path.read_bytes()).hexdigest()==VALIDATOR_REF["sha256"],
        "Already reviewed exact original migration validator")
    sp=importlib.util.spec_from_file_location("_C3_actual_original_complete_state_migration",path)
    validator=importlib.util.module_from_spec(sp);sys.modules[sp.name]=validator;sp.loader.exec_module(validator)
    planned={x["name"]:x for x in expected};written=set()
    def persist_checkpoint(relative_path,value):
        target=ROOT/relative_path
        need(target.parent==output.path and target.name in planned and target.name not in written
            and not target.exists(),"Only exact predeclared fresh separate checkpoint copies")
        resource_check("before_original_guarded_torch_serialization")
        import torch
        probe=io.BytesIO();torch.save(value,probe);size=probe.tell();del probe
        need(0<size<=planned[target.name]["maximum_migrated_serialized_bytes"],
            "Actual migrated serialized bytes fit declared per-copy bound before write")
        need(output.used+size+8192<=sum(x["maximum_guarded_write_charge_bytes"] for x in expected),
            "Actual total guarded copy bytes fit the original count-derived cap")
        # Exact original configured guard writer, with its original boundary,
        # filesystem-free and O_EXCL checks. No custom checkpoint persistence.
        result=output.save(target.name,value,kind="torch")
        need(target.stat().st_size==size,"Original serializer used the prechecked actual byte count")
        written.add(target.name)
        resource_check("after_original_guarded_torch_copy")
        return result
    result=validator.migrate_complete_boundary_v1(intent_ref,
        resource_check=resource_check,persist_checkpoint=persist_checkpoint)
    need(written==set(planned) and result.get("passed") is True
        and result.get("actual_migration_tool_terminal_consumed") is False
        and result.get("actual_migration_tool_exit_code") is None
        and result.get("optimizer_updates_performed")==0,
        "Original complete-state result remains unconsumed, no optimizer updates")
    resource_check("after_original_full_state_migration_callback")
    return result
