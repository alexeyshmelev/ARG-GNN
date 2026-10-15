"""Independent v9 source/metadata and in-memory/mocked CPU review; no runtime imports."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import ast,copy,hashlib,json,os,stat,collections
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"research/reviews/American_C3_native_MLM_amendment_v9"
HERE=BASE/"native_v9_independent_review"
OLD=ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"
checks=[]
def check(n,ok):
    if not ok:raise AssertionError(n)
    checks.append(n)
def reject(n,call):
    try:call()
    except (ValueError,KeyError,UnicodeError):checks.append(n);return
    raise AssertionError(n+" accepted")
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source(p):return p.read_text()
def fn(s,name):return next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
def segment(s,name):return ast.get_source_segment(s,fn(s,name))
def take(s,names,ns):
    nodes=[n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"<review-extracted-helpers>","exec"),ns)
    return ns
def require(ok,msg):
    if not ok:raise ValueError(msg)
closure=read(BASE/"execution_source_closure.json");handoff=read(BASE/"source_handoff.json")
check("exact frozen handoff",digest(BASE/"source_handoff.json")=="36624d8602a19b758e1c8afae468b252c6a4a3cf4a1f0504d60094bfeb95fb0f")
check("exact frozen source closure",digest(BASE/"execution_source_closure.json")=="36f97e6ae03097d42926679975d3adcbf3ae1817c027f991cb88718cdbc4a95c")
check("all47 source pins match",len(closure["source_sha256"])==47 and all(digest(ROOT/p)==h for p,h in closure["source_sha256"].items()))
oldinput=read(OLD/"admitted_mlm_v1/input.json");newinput=read(BASE/"input.json")
check("all31 original pins exact preserved",len(oldinput["source_sha256"])==31 and all(closure["source_sha256"].get(p)==h for p,h in oldinput["source_sha256"].items()))
check("exact16 truthful actualnew execution pins",len(set(closure["source_sha256"])-set(oldinput["source_sha256"]))==16)
allowed={"main_source_ref","actual_launcher_ref","root_context_module_ref","source_sha256","transitive_source_closure_ref","input_binding_mode"}
check("only input execution provenance differs",{k:v for k,v in oldinput.items() if k not in allowed}=={k:v for k,v in newinput.items() if k not in allowed})
refs=read(BASE/"original_refs.json")
check("all fixed original metadata ref hashes exact",all(digest(ROOT/r["path"])==r["sha256"] for r in refs.values()))
storage=read(ROOT/refs["ORIGINAL_STORAGE_REF"]["path"])
check("exact original14GiB and job scopes",storage["reserved_new_bytes"]==15032385536 and storage["job_id"]=="American-C3-native-MLM-full30mb-mlm-fit-v8"
 and storage["reservation"]["key"]=="H041/American-C3-native-MLM-full30mb-mlm-fit-v8")
check("original absolute expiry unchanged",newinput["expires_at_utc"]==oldinput["expires_at_utc"]==storage["expires_at_utc"]=="2026-10-18T11:47:27.217781+00:00")
auth=read(BASE/"execution_authority.inactive.json")
check("authority inactive pending actualroot evidence",auth["active"] is False and auth["old_writer_absence_ref"] is None and auth["archive_receipt_ref"] is None)
wait=read(ROOT/auth["actual_old_outer_wait_ref"]["path"])
check("genuine failed v8 outer wait remains1",digest(ROOT/auth["actual_old_outer_wait_ref"]["path"])==auth["actual_old_outer_wait_ref"]["sha256"] and wait["actual_direct_outer_supervisor_wait"] is True and wait["pid"]==2817234 and wait["returncode"]==1)
check("unknown old native result not invented",read(BASE/"old_writer_absence.inactive.json")["native_wait_exit"]=="UNKNOWN"
 and read(BASE/"old_writer_absence.inactive.json")["actual_optimizer_updates"]=="UNKNOWN")
amend=source(BASE/"amendment_v9.py");driver=source(BASE/"root_native_driver_v1.py");launch=source(BASE/"launcher_v9.py")
supervisor=source(BASE/"supervisor_v1.py");child=source(BASE/"native_child_v1.py");names=source(BASE/"terminal_namespace_v9.py")
archive=source(BASE/"root_archive_v9.py");capture=source(BASE/"terminal_capture_v9.py")
check("no reserve apply path in new driver",all(x not in driver+amend for x in ["apply_reserve_v1","operation_draft_v1(\"reserve\"","--admit"]))
activation=segment(amend,"activate")
check("same existing guard precharge before Popen","guard.check_storage_locked_v1(storage,boundary=True,required_write_bytes=134217728)" in activation and activation.index("guard.check_storage_locked_v1")<activation.index("manifest=prepare("))
check("singleuse exclusive lifetime and no replay","not LOCK.exists() and not ATTEMPT.exists()" in activation and "os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW" in activation and "fcntl.LOCK_EX|fcntl.LOCK_NB" in activation)
check("lifetime descendants exactowner and writable independent probe","descends_from(os.getpid(),token[\"owner\"])" in amend and "probe=os.open(LOCK,os.O_RDWR|os.O_NOFOLLOW)" in amend)
check("cooperative revocation honestly explicit",auth["immutable_old_code_mechanical_rejection_claimed"] is False and 'auth["all_v8_launch_resume_write_capabilities_cooperatively_revoked"] is True' in amend)
check("fresh actual old ESRCH check before activation",activation.count("observe_old_absence(auth)")>=2 and '0<=(now-checked).total_seconds()<=300' in amend and 'os.kill(row["pid"],0)' in amend)
check("exclusive GPU4 exactjob logicalidentity", "SCHEDULABLE_GPUS = (4,)" in launch and "fcntl.LOCK_EX | fcntl.LOCK_NB" in launch
 and 'args.module!=amendment.ENTRY or args.gpu!=4 or args.job!=amendment.JOB' in launch
 and '"ARGGNN_LOGICAL_JOB_NAME": args.job' in launch)
check("no pool registry modification route",all(x not in launch for x in ["register_logical","pool_register","shared_pool"]))
check("archive explicit authority and actual callback before mutation",'authority["archive_allowance_bytes"]==5242880' in archive
 and archive.index("require(prove_old_writers_absent()")<archive.index("archive.mkdir()")
 and "os.rename(root,dest)" in archive and "stat.S_ISSOCK" in archive)
check("archive preserves namespace metadata before socket cleanup",archive.index('admin_write(archive/"before_rename.json"')<archive.index("clean_dead_sockets(")
 and "native_wait_exit=\"UNKNOWN\",actual_optimizer_updates=\"UNKNOWN\"" in archive)
check("socket cleanup narrowly ownpymp/listener and preservedfirst",'p.parent.name.startswith("pymp-")' in names and 'p.name.startswith("listener-")' in names
 and names.index('preserve(dict(schema="native_MLM_dead_IPC')<names.index("p.unlink()"))
check("fresh exact socket kernel check each unlink","No live exact socket immediately before cleanup" in names and "prove_drained() and stamp(p.lstat())" in names)
check("latest only ownrelative regular independent samebytes afterdrain","rel.parent==Path(\".\")" in names and 'alias=directory/"latest-rank0.pt"' in names and "os.O_RDONLY|os.O_NOFOLLOW" in names
 and "ds.st_ino!=ts.st_ino" in names and "copied.hexdigest()==h.hexdigest()" in names
 and names.index("preserve(metadata)")<names.index("os.replace(temp,alias)") and "prove_drained()" in names)
native=segment(supervisor,"_run_child");outer=segment(driver,"launch")
check("native direct wait before risky postprocessing",native.index("returncode=proc.wait();capture_direct(returncode)")<native.index("clean_dead_sockets(")<native.index('ctx["boundary_guard"]("after_native_child_actual_wait_and_drain"'))
check("native capture failures still reach finallydrain","except BaseException as error:\n            guard_error=guard_error or error" in native and "drain(force=True)" in native)
check("outer direct wait before completion parse and boundary",outer.index("rc=proc.wait();capture_direct(rc)")<outer.index('complete_path=ROOT/output/"completed.json"')<outer.index('m.budget_check_v9(storage,boundary=True,required_write_bytes=6*'))
check("launcher direct wait before drain and file guard",launch.index("returncode=child.wait()")<launch.index('terminal.event(0,')<launch.index("while ownership.live() or _children():")<launch.rindex("save_wait_evidence_locked(record, account, child, returncode,"))
check("wait print before fixedcapture all3layers",native.index('event="actual_direct_native_wait_v9"')<native.index("terminal.event(0,")
 and outer.index('event="actual_direct_launcher_wait_v9"')<outer.index("terminal.event(0,")
 and launch.index('event="actual_direct_entry_wait_v9"')<launch.index("terminal.event(0,"))
check("fixed rawtail caps match planning",read(BASE/"write_cap_followup.json")["capture_fixed_total_bytes"]==24117248
 and read(BASE/"write_cap_followup.json")["auxiliary_planned_sum_bytes"]<134217728)
oldchild=source(OLD/"native_child_v1.py")
for name in ["_io_config","optimizer_counter_evidence","main"]:
    check("unchanged scientific child function "+name,ast.dump(fn(child,name),include_attributes=False)==ast.dump(fn(oldchild,name),include_attributes=False))
check("owned lifecycle byteexact old",digest(BASE/"owned_lifecycle_v2.py")==digest(OLD/"owned_lifecycle_v2.py"))
check("terminal bounds byteexact old",digest(BASE/"terminal_bounds_v1.py")==digest(OLD/"terminal_bounds_v1.py"))
# In-memory capture: real production helper, only mocked syscalls, no filesystem writes.
ns=take(capture,["require","_identity","_regular","_pwrite_all","Capture","decode_slot","WaitRelay"],dict(Path=Path,hashlib=hashlib,json=json,os=os,stat=stat,
 HEADER=128,WAIT_BYTES=1048576,ROLES=("native","launcher","outer"),TAIL_CAPS={"native":4194304,"outer":16777216}))
raw=bytearray(1048576);obj=ns["Capture"].__new__(ns["Capture"])
obj.role="native";obj.fds={"wait":90};obj.committed=set();obj.manifest={"files":{"native.wait":[1,2,1048576]}}
st=NS(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_dev=1,st_ino=2,st_size=1048576)
def write(fd,data,offset):raw[offset:offset+len(data)]=data;return len(data)
with patch.object(os,"fstat",return_value=st),patch.object(os,"pwrite",side_effect=write),patch.object(os,"fsync",side_effect=OSError("synthetic fsync failure")):
    try:obj.event(0,dict(returncode=1,actual_direct_child_wait=True))
    except OSError:pass
reject("torn preheader write is not evidence",lambda:ns["decode_slot"](bytes(raw),0))
check("torn event not marked committed",obj.committed==set())
with patch.object(os,"fstat",return_value=st),patch.object(os,"pwrite",side_effect=write),patch.object(os,"fsync"):
    obj.event(0,dict(returncode=1,actual_direct_child_wait=True,drain_state="not_yet_certified"))
check("failed direct wait preserved independently of drain",ns["decode_slot"](bytes(raw),0)["returncode"]==1)
reject("drain slot not fabricated from wait slot",lambda:ns["decode_slot"](bytes(raw),1))
events=[];relay=ns["WaitRelay"](emit=events.append)
for _ in range(20):relay.feed(b'{"event":"actual_direct_native_wait_v9","returncode":1}\n')
check("authentic ownpipe wait relay bounded8",len(events)==8)
relay.feed(b"x"*20000);check("relay unterminated buffer bounded8192",len(relay.pending)<=8192)
# Child checkpoint helper with fake paths, fake CPU loader and serialized counters only.
cs=take(child,["_int_time","optimizer_counter_evidence","_checkpoint_entries","_final_checkpoint"],dict(Path=Path,stat=stat,os=os))
class FakePath:
    def __init__(self,name,mode):self.name=name;self.suffix=Path(name).suffix;self.mode=mode
    def lstat(self):return NS(st_mode=self.mode,st_nlink=1,st_dev=1,st_ino=3 if self.name.startswith("latest") else 4,st_size=128,st_mtime_ns=1,st_ctime_ns=1)
    def __str__(self):return "/synthetic/checkpoints/"+self.name
target=FakePath("ep5-ba224775-rank0.pt",stat.S_IFREG|0o600);alias=FakePath("latest-rank0.pt",stat.S_IFLNK|0o777)
class FakeDirectory:
    def iterdir(self):return iter([alias,target])
    def __truediv__(self,name):
        if str(name)==target.name:return target
        raise ValueError("unexpected target")
directory=FakeDirectory();loaded=[]
def payload(step=224775,epoch=5):
    return dict(state=dict(timestamp=dict(epoch=epoch,batch=224775,batch_in_epoch=0),model={"synthetic.weight":0},
      optimizers={"DecoupledAdamW":{"state":{0:dict(step=step,exp_avg=0,exp_avg_sq=0)}}}))
torch=NS(load=lambda path,**kw:(loaded.append(path.name) or payload()))
with patch.object(os,"readlink",return_value=target.name):
    result=cs["_final_checkpoint"](directory,epochs=5,updates=224775,max_files=2,max_bytes=2950918528,torch=torch,reference=lambda path:dict(path=str(path),sha256="a"*64))
check("semantic actualfull counters checked before alias normalization",result["actual_optimizer_updates"]==224775 and result["completed_epochs"]==5
 and result["checkpoint_latest_alias"]["status"]=="observed_validated_alias_pending_postdrain_normalization")
check("latest alias never opened by child",loaded==[target.name] and result["checkpoint_latest_alias"]["alias_followed_or_opened"] is False)
torch.load=lambda path,**kw:payload(step=224774)
with patch.object(os,"readlink",return_value=target.name):
    reject("short actual Adam state rejected",lambda:cs["_final_checkpoint"](directory,epochs=5,updates=224775,max_files=2,max_bytes=2950918528,torch=torch,reference=lambda path:dict(path=str(path),sha256="a"*64)))
torch.load=lambda path,**kw:payload(epoch=4)
with patch.object(os,"readlink",return_value=target.name):
    reject("short actualepoch rejected",lambda:cs["_final_checkpoint"](directory,epochs=5,updates=224775,max_files=2,max_bytes=2950918528,torch=torch,reference=lambda path:dict(path=str(path),sha256="a"*64)))
receipt=dict(schema="native_v9_independent_source_and_mocked_CPU_checks_v1",status="PASS",checks=len(checks),check_names=checks,
 actual_runtime_guard_model_native_GPU_process_probe_or_admission_executed=False,
 actual_scientific_payload_checkpoint_or_empirical_array_read=False,actual_rename_socket_unlink_or_checkpoint_copy_performed=False,
 namespace_and_capture_IO_mocked=True,original_storage_ref=refs["ORIGINAL_STORAGE_REF"],
 original_outer_wait_ref=auth["actual_old_outer_wait_ref"],old_native_result_and_update_count="UNKNOWN",
 reviewed_source_handoff_ref=dict(path=str((BASE/"source_handoff.json").relative_to(ROOT)),sha256=digest(BASE/"source_handoff.json")),
 reviewed_execution_closure_ref=dict(path=str((BASE/"execution_source_closure.json").relative_to(ROOT)),sha256=digest(BASE/"execution_source_closure.json")))
(HERE/"checks.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(status=receipt["status"],checks=len(checks))))
