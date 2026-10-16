from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os,stat,re,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
directory=ROOT/"experiments/results/native_VAL000_full30mb_borrowed32_GPU0_v2/same_charge_owned_roster_v4/checkpoints"
alias=directory/"latest-rank0.pt"
assert alias.is_symlink()
name=os.readlink(alias)
assert Path(name).parent==Path(".") and re.fullmatch(r"ep(\d+)-ba(\d+)-rank0\.pt",name)
p=directory/name
listed=p.lstat();batch=int(re.fullmatch(r"ep(\d+)-ba(\d+)-rank0\.pt",name).group(2))
assert stat.S_ISREG(listed.st_mode) and listed.st_nlink==1
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);before=os.fstat(fd)
assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(listed.st_dev,listed.st_ino,listed.st_size,listed.st_mtime_ns)
assert 0<before.st_size<=3*2**30 and before.st_nlink==1
import torch
assert not torch.cuda.is_initialized()
value=torch.load(f"/proc/self/fd/{fd}",map_location="cpu",weights_only=False,mmap=True)
assert not torch.cuda.is_initialized()
child=ROOT/"research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py"
tree=ast.parse(child.read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("_int_time","optimizer_counter_evidence")]
assert len(functions)==2
ns={};exec(compile(ast.Module(body=functions,type_ignores=[]),str(child),"exec"),ns)
state=value["state"];timestamp=state["timestamp"]
counters=ns["optimizer_counter_evidence"](state["optimizers"])
actual_batch=ns["_int_time"](timestamp["batch"])
assert counters["unique_step_values"]==[actual_batch] and actual_batch==batch
assert isinstance(state["model"],dict) and len(state["model"])>0
after=os.fstat(fd)
assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
receipt=dict(schema="ROOT_native_roster_v4_actual_saved_optimizer_progress_v1",observed_utc=datetime.now(timezone.utc).isoformat(),actual_tool_session_id=24814,checkpoint_path=str(p.relative_to(ROOT)),checkpoint_opened_without_symlink_following=True,checkpoint_identity=dict(device=before.st_dev,inode=before.st_ino,bytes=before.st_size,mtime_ns=before.st_mtime_ns,link_count_before=before.st_nlink,link_count_after=after.st_nlink),checkpoint_mmap_via_own_readonly_fd=True,checkpoint_full_bytes_sha256_computed=False,checkpoint_may_rotate_after_observation=True,actual_Composer_timestamp={k:ns["_int_time"](timestamp[k]) for k in ("epoch","batch","batch_in_epoch")},actual_saved_Adam_counters=counters,actual_verified_optimizer_updates=actual_batch,source_verifier_path=str(Path(__file__).relative_to(ROOT)),source_verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),counter_function_source_ref=dict(path=str(child.relative_to(ROOT)),sha256=hashlib.sha256(child.read_bytes()).hexdigest()),Torch_CPU_checkpoint_read_only=True,CUDA_context_created=False,training_modified=False,ancestry_score=None,full_budget_updates=227990,full_budget_complete=False)
out=ROOT/"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_owned_roster_v4_saved_optimizer_progress.root_v1.json"
with out.open("x") as h:json.dump(receipt,h,indent=2,sort_keys=True);h.write("\n")
print(json.dumps(receipt),flush=True)
os.close(fd)
