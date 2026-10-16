from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os,stat,signal,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/native_v4_preserved_checkpoint_resume_root_metadata_20261016_v1"
p=ROOT/"experiments/results/native_VAL000_full30mb_borrowed32_GPU0_v2/same_charge_owned_roster_v4/checkpoints/ep0-ba6500-rank0.pt"
for pid in (3611198,3610904):
    try:os.kill(pid,0)
    except ProcessLookupError:pass
    else:raise RuntimeError("Original current-attempt process remains present")
try:os.killpg(3611198,0)
except ProcessLookupError:pass
else:raise RuntimeError("Original worker group remains present")
listed=p.lstat()
assert stat.S_ISREG(listed.st_mode) and listed.st_nlink==1 and 0<listed.st_size<=2*2**30
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
try:
    before=os.fstat(fd)
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
    assert identity(before)==identity(listed)
    started=time.monotonic();digest=hashlib.sha256();read_bytes=0
    while True:
        b=os.read(fd,8*2**20)
        if not b:break
        digest.update(b);read_bytes+=len(b)
    assert read_bytes==before.st_size and identity(os.fstat(fd))==identity(before)
    os.lseek(fd,0,os.SEEK_SET)
    import torch
    assert not torch.cuda.is_initialized()
    value=torch.load(f"/proc/self/fd/{fd}",map_location="cpu",weights_only=False,mmap=True)
    assert not torch.cuda.is_initialized()
    state=value["state"];timestamp=state["timestamp"]["Timestamp"]
    assert type(timestamp["epoch"]) is int and timestamp["epoch"]==0
    assert type(timestamp["batch"]) is int and timestamp["batch"]==6500
    assert timestamp["batch_in_epoch"]==6500
    child=ROOT/"research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py"
    tree=ast.parse(child.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("_int_time","optimizer_counter_evidence")]
    assert len(functions)==2
    ns={};exec(compile(ast.Module(body=functions,type_ignores=[]),str(child),"exec"),ns)
    counters=ns["optimizer_counter_evidence"](state["optimizers"])
    assert counters["unique_step_values"]==[6500]
    assert isinstance(state["model"],dict) and state["model"]
    assert "schedulers" in state and "callbacks" in state
    assert "rng" in value
    after=os.fstat(fd)
    assert identity(after)==identity(before)==identity(p.lstat())
    receipt=dict(schema="ROOT_preserved_native_full_state_checkpoint_metadata_v1",
        observed_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_ref=dict(path=str(p.relative_to(ROOT)),sha256=digest.hexdigest()),
        checkpoint_identity=dict(device=before.st_dev,inode=before.st_ino,bytes=before.st_size,mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns,links=before.st_nlink),
        full_byte_read_and_sha256=True,full_bytes_read=read_bytes,hash_and_metadata_elapsed_seconds=time.monotonic()-started,
        own_readonly_no_follow_FD_mmap=True,pre_post_file_and_FD_identity_equal=True,
        Composer_nested_Timestamp=timestamp,actual_saved_Adam_counters=counters,actual_verified_optimizer_updates=6500,
        full_update_budget=227990,full_budget_complete=False,
        top_level_keys=sorted(map(str,value.keys())),state_keys=sorted(map(str,state.keys())),
        model_parameter_entry_count=len(state["model"]),scheduler_state_present=True,callback_state_present=True,global_RNG_state_present=True,
        resume_runtime_or_bitwise_equivalence_qualified=False,
        original_worker_and_supervisor_numeric_PID_and_worker_group_absent=True,
        original_failure_preserved=True,checkpoint_modified=False,CUDA_context_created=False,
        no_training_launch_accounting_close_or_credit=True,ancestry_score=None,
        verifier_ref=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with (HERE/"preserved_checkpoint_metadata.root_v1.json").open("x") as h:
        json.dump(receipt,h,indent=2,sort_keys=True);h.write("\n")
    print(json.dumps(receipt),flush=True)
finally:os.close(fd)
