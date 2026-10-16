from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,hashlib,json,os,re,stat,time,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/native_v10_checkpoint_progress_root_20261016_v1"
SCOPE=ROOT/"experiments/results/native_VAL000_full30mb_borrowed32_GPU0_v2/same_charge_full_state_resume_v10/checkpoints"
def run():
    started=time.monotonic()
    choices=[p for p in SCOPE.glob("ep*-ba*-rank0.pt") if stat.S_ISREG(p.lstat().st_mode) and p.lstat().st_nlink==1 and time.time()-p.lstat().st_mtime>=5]
    assert choices,"No quiescent regular saved checkpoint yet"
    p=max(choices,key=lambda v:int(re.search(r"-ba(\d+)-",v.name).group(1)))
    named=p.lstat();assert 0<named.st_size<=2*2**30
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
        assert ident(before)==ident(named)
        digest=hashlib.sha256();read_bytes=0
        while True:
            b=os.read(fd,8*2**20)
            if not b:break
            digest.update(b);read_bytes+=len(b)
        assert read_bytes==before.st_size and ident(os.fstat(fd))==ident(before)
        import torch
        assert not torch.cuda.is_initialized()
        value=torch.load(f"/proc/self/fd/{fd}",map_location="cpu",weights_only=False,mmap=True)
        assert not torch.cuda.is_initialized()
        state=value["state"];timestamp=state["timestamp"]["Timestamp"];step=timestamp["batch"]
        assert type(step) is int and 6500<step<=227990
        assert step==int(re.search(r"-ba(\d+)-",p.name).group(1))
        child="research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py"
        manifest=ROOT/"research/reviews/native_VAL000_full_state_resume_source_20261016_v10/source_manifest.v1.json"
        assert hashlib.sha256(manifest.read_bytes()).hexdigest()=="9ab34741ee948a2ddc9fe986abb0bd72533ead0a788c79270fe3064307193719"
        refs=[]
        def find(v):
            if isinstance(v,dict):
                if v.get("path")==child and "sha256" in v:refs.append(v["sha256"])
                for item in v.values():find(item)
            elif isinstance(v,list):
                for item in v:find(item)
        find(json.loads(manifest.read_text()))
        assert len(set(refs))==1 and hashlib.sha256((ROOT/child).read_bytes()).hexdigest()==refs[0]
        tree=ast.parse((ROOT/child).read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("_int_time","optimizer_counter_evidence")]
        assert len(functions)==2
        ns={};exec(compile(ast.Module(body=functions,type_ignores=[]),child,"exec"),ns)
        counters=ns["optimizer_counter_evidence"](state["optimizers"])
        assert counters["unique_step_values"]==[step] and counters["parameter_state_count"]==77
        assert state["model"] and "schedulers" in state and "callbacks" in state and "rng" in value
        assert ident(os.fstat(fd))==ident(before)==ident(p.lstat())
        ts=json.loads(json.dumps(timestamp,default=lambda v:dict(type="timedelta",seconds=v.total_seconds()) if isinstance(v,timedelta) else str(type(v))))
        return dict(schema="ROOT_live_native_saved_checkpoint_progress_v1",status="VERIFIED_SAVED_PROGRESS",observed_utc=datetime.now(timezone.utc).isoformat(),checkpoint_ref=dict(path=str(p.relative_to(ROOT)),sha256=digest.hexdigest()),checkpoint_bytes=read_bytes,full_byte_hash_and_FD_mmap=True,pre_post_named_and_FD_identity_equal=True,actual_saved_optimizer_counters=counters,actual_saved_optimizer_updates=step,additional_saved_updates_beyond6500=step-6500,Composer_timestamp=ts,model_parameter_entries=len(state["model"]),scheduler_callbacks_RNG_present=True,ancestry_score=None,full_budget_complete=step==227990,full_update_budget=227990,CUDA_context_created=False,checkpoint_modified=False,training_or_model_forward_executed=False,prior_failures_preserved=True,live_updates_after_saved_checkpoint_unknown=True,elapsed_seconds=time.monotonic()-started,verifier_ref=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    finally:os.close(fd)
try:
    receipt=run()
except BaseException as exc:
    receipt=dict(schema="ROOT_live_native_saved_checkpoint_progress_attempt_v1",status="NOT_VERIFIED",error=repr(exc),traceback=traceback.format_exc(),no_training_restart_or_checkpoint_change=True,CUDA_context_created=False)
with (HERE/"checkpoint_progress.root_v1.json").open("x") as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps(receipt),flush=True)
raise SystemExit(0 if receipt["status"]=="VERIFIED_SAVED_PROGRESS" else 1)
