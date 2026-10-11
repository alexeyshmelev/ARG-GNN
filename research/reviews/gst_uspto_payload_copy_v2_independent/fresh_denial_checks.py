"""Fresh finite GST V2 denial/diagnostic checks on newly made opaque ZIP bytes."""
from pathlib import Path
import os,sys,io,json,hashlib,stat,struct,zipfile,datetime,contextlib
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=Path(__file__).resolve().parent
CASE_ROOT=OUT/"fresh_fixtures"
CASE_ROOT.mkdir()
CHECKS=[];CASES=[];FORBIDDEN=[];active_state=None
original_os_write=os.write
def tracked_write(fd,payload):
 if active_state is not None and active_state["denied"]:
  active_state["post_denial_os_writes"].append({"fd":fd,"bytes_requested":len(payload)})
 return original_os_write(fd,payload)
os.write=tracked_write
def audit(event,args):
 if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
  name=os.fsdecode(args[0]);path=Path(name)
  blocked=(name=="data/paper_native_raw_acquisition_v1/gst_native_snapshots/USPTO15K_data.zip"
   or (path.is_absolute() and (name.startswith(str(ROOT/"data")+"/") or name.startswith("/proc/")
    or name in {str(ROOT/".runtime/gpup/storage_reservations.json"),str(ROOT/".runtime/storage/project_budget_v4.json")}
    or name.startswith(str(ROOT/".runtime/conda")+"/"))))
  if blocked:FORBIDDEN.append({"event":event,"path":name});raise AssertionError("actual raw/data/proc/ledger/native prefix forbidden")
  flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
  mode=args[1] if len(args)>1 else None
  writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)) or (isinstance(mode,str) and any(c in mode for c in "wax+"))
  if writing and path.is_absolute() and not path.is_relative_to(OUT):
   FORBIDDEN.append({"event":event,"path":name});raise AssertionError("write outside independent folder")
 if event=="import" and args and str(args[0]).split(".")[0] in {"torch","tensorflow","numpy","scipy","pandas","rdkit","tsinfer","tsdate"}:
  FORBIDDEN.append({"event":event,"module":str(args[0])});raise AssertionError("scientific/model import forbidden")
 if event in {"subprocess.Popen","os.kill","os.killpg","socket.connect"}:
  FORBIDDEN.append({"event":event});raise AssertionError("external process/network action forbidden")
sys.addaudithook(audit)
# These modules have only stdlib imports; actual archive/ledger APIs remain blocked.
from arggnn import gst_uspto_payload_copy_v2 as p
from arggnn import gst_uspto_archive_inventory_v2 as inv
real_event=p.CopyScopeV2.event;real_finish=p.CopyScopeV2.finish
def marked_event(scope,kind,**fields):
 prior=active_state["phase"] if active_state is not None else None
 if active_state is not None:active_state["phase"]=(kind,fields.get("archive_order"))
 try:return real_event(scope,kind,**fields)
 finally:
  if active_state is not None:active_state["phase"]=prior
def marked_finish(scope,result):
 prior=active_state["phase"] if active_state is not None else None
 if active_state is not None:active_state["phase"]=("finish",None)
 try:return real_finish(scope,result)
 finally:
  if active_state is not None:active_state["phase"]=prior
p.CopyScopeV2.event=marked_event;p.CopyScopeV2.finish=marked_finish
def ck(label,value):
 CHECKS.append({"check":label,"passed":bool(value)})
 if not value:raise AssertionError(label)
def snapshot(dst):
 files={};dirs=[]
 if dst.exists():
  for item in sorted(dst.rglob("*")):
   name=str(item.relative_to(dst))
   if item.is_dir():dirs.append(name)
   else:
    data=item.read_bytes();files[name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
 return {"files":files,"directories":dirs}
def make_zip(method=zipfile.ZIP_STORED):
 data=[("data/",b""),("data/connor_original.tsv",bytes(range(256))*300+b"\x00\r\n\n"),
  ("data/test.txt",b"independent opaque test\r\n"),("data/train.txt",b"independent opaque train\n"),
  ("data/valid.txt",b"independent opaque valid\r\n")]
 buffer=io.BytesIO()
 with zipfile.ZipFile(buffer,"w") as archive:
  for name,payload in data:
   item=zipfile.ZipInfo(name,date_time=(2020,4,5,6,7,8));item.create_system=3
   item.external_attr=((stat.S_IFDIR|0o755)<<16)|0x10 if name.endswith("/") else (stat.S_IFREG|0o644)<<16
   item.compress_type=zipfile.ZIP_STORED if name.endswith("/") else method
   archive.writestr(item,payload)
 return buffer.getvalue(),data
def corrupt_payload(raw):
 with zipfile.ZipFile(io.BytesIO(raw)) as archive:item=archive.infolist()[1]
 header=struct.unpack_from("<IHHHHHIIIHH",raw,item.header_offset)
 changed=bytearray(raw);changed[item.header_offset+30+header[-2]+header[-1]+19]^=3
 return bytes(changed)
def state_for(dst,target):
 return {"destination":dst,"target":target,"phase":None,"denied":False,"denied_snapshot":None,
  "guard_calls":0,"guard_calls_at_denial":None,"post_denial_os_writes":[]}
def guard_for(state):
 def guard(required):
  state["guard_calls"]+=1
  if state["phase"]==state["target"] and not state["denied"]:
   state["denied_snapshot"]=snapshot(state["destination"])
   state["guard_calls_at_denial"]=state["guard_calls"];state["denied"]=True
   raise ValueError("independent explicit authority refusal at "+state["target"][0])
 return guard
def inspect_denied(label,state,result,receipt):
 after=snapshot(state["destination"])
 ck(label+" reached fresh intended denial",state["denied"] and state["denied_snapshot"] is not None)
 ck(label+" no OS writes after authority refusal",state["post_denial_os_writes"]==[])
 ck(label+" exact destination bytes/entries preserved after refusal",after==state["denied_snapshot"])
 ck(label+" recovered guard callback never reconsulted",state["guard_calls"]==state["guard_calls_at_denial"])
 ck(label+" final receipt refused",receipt is None and result["final_receipt_written"] is False and not (state["destination"]/"payload_manifest.json").exists())
 ck(label+" reports sticky authority denial",result["authority_rejected"] is True)
 total=sum(value["bytes"] for name,value in after["files"].items() if name!="progress.jsonl")
 ck(label+" exact truthful payload total",total==result["total_payload_bytes_written"])
 ck(label+" no native parse/model/scoring admission",not result["dataset_rows_parsed"] and not result["native_model_run"] and not result["scoring_performed"] and not result["dataset_accepted_for_parsing"])
 CASES.append({"case":label,"denial_phase":list(state["target"]),"destination_at_denial":state["denied_snapshot"],
  "destination_after":after,"OS_writes_after_denial":0,"guard_calls":state["guard_calls"],
  "payload_bytes_written":result["total_payload_bytes_written"],"status":result["status"],
  "error":result.get("error"),"failure_event_error":result.get("failure_event_error"),"receipt_sha256":receipt})
raw,contents=make_zip()
for label,target in (
 ("deny_copy_start",("copy_started",None)),("deny_member_open",("member_opened",1)),
 ("deny_first_short_prefix",("member_prefix_written",1)),("deny_member_complete",("member_complete",1)),
 ("deny_copy_complete",("copy_complete",None)),("deny_final_receipt",("finish",None))):
 dst=CASE_ROOT/label;state=state_for(dst,target);active_state=state
 writer=(lambda fd,payload:tracked_write(fd,payload[:31])) if label=="deny_first_short_prefix" else tracked_write
 result,receipt=p.copy_payload_v2(raw,inv.inventory(raw),dst,guard=guard_for(state),payload_writer=writer)
 inspect_denied(label,state,result,receipt)
 if label=="deny_first_short_prefix":
  partial=(dst/contents[1][0]).read_bytes();row=result["members"][1]
  ck("first short-prefix denial preserves31 actual bytes",partial==contents[1][1][:31] and row["payload_bytes_written"]==31)
  ck("first short-prefix SHA exact",row["payload_prefix_sha256"]==hashlib.sha256(partial).hexdigest())
  ck("first short-prefix later members never reached",all(r["status"]=="unreached" for r in result["members"][2:]))
 active_state=None
# An ordinary corrupted handmade CRC still reports its original cause when diagnostics are denied.
corrupt=corrupt_payload(raw);dst=CASE_ROOT/"deny_CRC_failure_event"
state=state_for(dst,("copy_failed",None));active_state=state
result,receipt=p.copy_payload_v2(corrupt,inv.inventory(corrupt),dst,guard=guard_for(state),payload_writer=tracked_write)
inspect_denied("deny_CRC_failure_event",state,result,receipt)
ck("CRC cause survives later authority failure","Actual member CRC32 differs from frozen inventory" in result["error"] and result["error_type"]=="ValueError")
ck("CRC logging refusal retained separately","explicit authority refusal" in result["failure_event_error"])
active_state=None
# A real short write followed by I/O failure and denied failure logging stays exact.
dst=CASE_ROOT/"deny_IO_failure_event";state=state_for(dst,("copy_failed",None));active_state=state;write_calls=[]
def short_then_fail(fd,payload):
 if write_calls:raise OSError("independent original short-write I/O failure")
 count=tracked_write(fd,payload[:13]);write_calls.append(count);return count
result,receipt=p.copy_payload_v2(raw,inv.inventory(raw),dst,guard=guard_for(state),payload_writer=short_then_fail)
inspect_denied("deny_IO_failure_event",state,result,receipt)
ck("original I/O error survives denied diagnostics",result["error_type"]=="OSError" and result["error"]=="independent original short-write I/O failure")
ck("I/O denial exact13 real bytes",(dst/contents[1][0]).read_bytes()==contents[1][1][:13] and result["total_payload_bytes_written"]==13)
active_state=None
# With valid authority, ordinary CRC failure continues to have bounded, truthful diagnostics.
dst=CASE_ROOT/"CRC_valid_authority"
result,receipt=p.copy_payload_v2(corrupt,inv.inventory(corrupt),dst)
manifest=json.loads((dst/"payload_manifest.json").read_bytes())
events=[json.loads(line) for line in (dst/"progress.jsonl").read_text().splitlines()]
ck("valid authority CRC records original bounded failure",receipt is not None and manifest==result and result["error_type"]=="ValueError" and result["status"]=="failed_partial_preserved")
ck("valid authority CRC writes guarded failure event",any(e["event"]=="copy_failed" for e in events) and result["authority_rejected"] is False)
ck("valid authority CRC never admits parsing",not result["complete_payload_copy"] and not result["dataset_accepted_for_parsing"])

# CLI: handmade hooks only. Caller stdout keeps the CRC cause and measured prefix after failure-event refusal.
saved=(p.admit_v2,inv._opaque_bytes,p.budget.bound,p.budget.check_reservation)
dst=CASE_ROOT/"CLI_CRC_denied_diagnostic";state=state_for(dst,("copy_failed",None));active_state=state
now=datetime.datetime.now(datetime.timezone.utc)
admission={"created_utc":now.isoformat(),"expires_utc":(now+datetime.timedelta(minutes=20)).isoformat(),
 "implementation":{"handmade":True},"config":{"handmade":True},"registration":{"handmade":True},"independent_review":{"handmade":True}}
cfg={"budget_source":{"sha256":"1"*64},"budget_config":{"sha256":"2"*64},"inventory":{"handmade":True}}
p.admit_v2=lambda reference:(admission,cfg,inv.inventory(corrupt),{"handmade":True},dst,{"handmade":True})
inv._opaque_bytes=lambda:corrupt;p.budget.bound=lambda *args,**kwargs:None
guard=guard_for(state)
p.budget.check_reservation=lambda *args,**kwargs:guard(kwargs["required_write_bytes"])
buffer=io.StringIO()
try:
 with contextlib.redirect_stdout(buffer):code=p.main(["--execute","--admission","handmade_only","--expected-admission-sha256","3"*64])
finally:
 p.admit_v2,inv._opaque_bytes,p.budget.bound,p.budget.check_reservation=saved
 active_state=None
report=json.loads(buffer.getvalue());diag=report["failure_diagnostics"]
ck("CLI denied diagnostic returns failure without receipt",code==1 and report["receipt_sha256"] is None and diag["authority_rejected"])
ck("CLI retains original CRC error and separate refusal","Actual member CRC32 differs" in diag["error"] and "explicit authority refusal" in diag["failure_event_error"])
ck("CLI reports measured payload prefix count",diag["total_payload_bytes_written"]==len(contents[1][1]) and diag["member_prefixes"][1]["payload_bytes_written"]==len(contents[1][1]))
ck("CLI reports measured actual payload SHA",diag["member_prefixes"][1]["payload_prefix_sha256"]==hashlib.sha256((dst/contents[1][0]).read_bytes()).hexdigest())
ck("CLI denied diagnostic no subsequent destination writes",snapshot(dst)==state["denied_snapshot"] and state["post_denial_os_writes"]==[])
ck("CLI sticky latch prevents guard retry",state["guard_calls"]==state["guard_calls_at_denial"])
CASES.append({"case":"CLI_CRC_denied_diagnostic","caller_stdout_report":report,"destination_at_denial":state["denied_snapshot"],"destination_after":snapshot(dst),"OS_writes_after_denial":0,"actual_admission_or_raw_used":False})
# Reject before first output creation: no fallback retry when exact authority already refused.
for label,opaque_raises in (("CLI_initial_guard_denied",False),("CLI_opaque_error_then_guard_denied",True)):
 dst=CASE_ROOT/label;state=state_for(dst,("CLI_guard",None));active_state=state
 saved=(p.admit_v2,inv._opaque_bytes,p.budget.bound,p.budget.check_reservation)
 p.admit_v2=lambda reference:(admission,cfg,inv.inventory(raw),{"handmade":True},dst,{"handmade":True})
 def opaque():
  if opaque_raises:raise OSError("independent original opaque preflight error")
  return raw
 inv._opaque_bytes=opaque;p.budget.bound=lambda *args,**kwargs:None
 def deny_check(*args,**kwargs):
  state["guard_calls"]+=1;state["denied"]=True;state["denied_snapshot"]=snapshot(dst);state["guard_calls_at_denial"]=state["guard_calls"]
  raise ValueError("independent original authority guard denial")
 p.budget.check_reservation=deny_check;buffer=io.StringIO()
 try:
  with contextlib.redirect_stdout(buffer):code=p.main(["--execute","--admission","handmade_only","--expected-admission-sha256","4"*64])
 finally:
  p.admit_v2,inv._opaque_bytes,p.budget.bound,p.budget.check_reservation=saved;active_state=None
 report=json.loads(buffer.getvalue());diag=report["failure_diagnostics"]
 expected="independent original opaque preflight error" if opaque_raises else "independent original authority guard denial"
 ck(label+" refuses creation and retry",not dst.exists() and state["guard_calls"]==1 and code==1)
 ck(label+" retains original stdout cause",diag["error"]==expected and diag["authority_rejected"] and report["receipt_sha256"] is None)
 ck(label+" no OS writes",state["post_denial_os_writes"]==[])
 CASES.append({"case":label,"caller_stdout_report":report,"OS_writes_after_denial":0,"destination_created":False,"guard_calls":1})
p.CopyScopeV2.event=real_event;p.CopyScopeV2.finish=real_finish;os.write=original_os_write
ck("no forbidden raw/data/runtime/process/budget access",FORBIDDEN==[])
result={"kind":"gst_payload_copy_v2_fresh_independent_denial_checks","verdict":"PASS",
 "checks":CHECKS,"check_count":len(CHECKS),"cases":CASES,"forbidden_access_attempts":FORBIDDEN,
 "byte_exact_progress_and_payload_preservation_after_refusal":True,"zero_tracked_OS_writes_after_refusal":True,
 "original_crc_IO_and_preflight_causes_preserved":True,"caller_stdout_member_prefix_hashes_and_counts_verified":True,
 "handmade_ZIP_and_explicit_CLI_hook_stubs_only":True,"actual_archive_or_payload_or_native_parser_opened":False,
 "actual_admission_runtime_model_GPU_budget_proc_launch_or_Git_actions":False}
with (OUT/"fresh_denial_checks.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
print(json.dumps({"verdict":"PASS","checks":len(CHECKS),"fresh_denial_cases":len(CASES),"OS_writes_after_refusal":0,"forbidden_access_attempts":len(FORBIDDEN)},indent=2))
