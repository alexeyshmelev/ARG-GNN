"""Root completion-only recovery of the already computed empirical bank."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.reviews.H018_empirical_preparation_root_v3 import root_driver as old
from research.reviews.H047_GPU45_cohort_risk_root_source_v1 import root_common_v1 as common
HERE=old.HERE
def main():
    import torch,numpy
    from arggnn import census_tsinfer_fit_v1 as codec
    codec.torch=torch
    codec.np=numpy
    reader=old.Reader()
    admitted=reader.decode_json(old.ref(HERE/"admitted.json"))
    spec=reader.decode_json(admitted["input_ref"])
    storage=reader.decode_json(spec["storage_ref"])
    wait_ref=old.ref(HERE/"actual_wait.json")
    wait=reader.decode_json(wait_ref)
    assert wait["returncode"]==1 and wait["actual_direct_Popen_wait"] is True
    pid=wait["pid"]
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("Original preparation child remains present")
    output=ROOT/spec["output_directory"]
    assert not (output/"completed.json").exists()
    bank_ref=old.ref(output/"observation_bank.pt")
    bank=torch.load(ROOT/bank_ref["path"],map_location="cpu",weights_only=False)
    assert bank["schema"]=="H018_empirical_CPU_observation_bank_v1"
    assert bank["input_ref"]==admitted["input_ref"]
    tensors=bank["bank_tensors"]
    assert all(not v.is_cuda for v in tensors.values() if isinstance(v,torch.Tensor))
    flags=bank["flag_node_flags"]
    assert flags.shape[0]==tensors["parent"].numel()
    rss=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
    assert rss<2000000000
    cfg=reader.decode_json(spec["config_ref"])
    m,memo=common.load_guard_v1(consumer=True)
    m.validate_storage_v9(spec,cfg,storage,source_ref=spec["producer_ref"],
        config_ref=spec["config_ref"],runtime_ref=spec["runtime_ref"])
    source_ref=old.ref(Path(__file__))
    authority=dict(schema="H018_completion_only_root_recovery_authority_v1",root_authorizer="root_agent",
        active=True,source_ref=source_ref,original_storage_ref=spec["storage_ref"],
        original_failed_wait_ref=wait_ref,existing_bank_ref=bank_ref,
        all_original_preparation_launch_resume_write_capabilities_revoked=True,
        original_child_targeted_ESRCH=True,new_preparation_or_model_execution_authorized=False,
        new_reservation_or_guard_change_authorized=False,
        output_write="exclusive completed.json only, within original20MB charge",
        recorded_utc=datetime.now(timezone.utc).isoformat())
    authority_ref=old.save("completion_recovery_authority_v1.json",authority)
    done=dict(schema="H018_empirical_CPU_preparation_completed_v1",
        status="completed_with_root_metadata_recovery",input_ref=admitted["input_ref"],
        producer_ref=spec["producer_ref"],bank_ref=bank_ref,metadata=bank["metadata"],
        original_preparation_process_returncode=1,
        original_metadata_serialization_failure="census serializer torch global uninitialized after bank save",
        completion_only_recovery_authority_ref=authority_ref,recovery_current_RSS_bytes=rss,
        original_preparation_peak_RSS_unavailable=True,no_model_or_checkpoint_executed=True,
        no_query_ancestry_truth=True,no_GPU_used=True,
        completed_utc=datetime.now(timezone.utc).isoformat())
    raw=m.budget.v4.encoded(done)+b"\n"
    m.budget_check_v9(storage,boundary=True,required_write_bytes=len(raw)+8192)
    target=output/"completed.json"
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    m.budget_check_v9(storage,boundary=True)
    receipt=dict(event="H018_actual_prepared_bank_completion_only_recovery",
        completed_ref=old.ref(target),bank_ref=bank_ref,metadata=bank["metadata"],
        recovery_current_RSS_bytes=rss,original_preparation_returncode=1,
        model_or_GPU_execution=False)
    old.save("completion_recovery_receipt_v1.json",receipt)
    print(json.dumps(receipt),flush=True)
if __name__=="__main__":
    main()
