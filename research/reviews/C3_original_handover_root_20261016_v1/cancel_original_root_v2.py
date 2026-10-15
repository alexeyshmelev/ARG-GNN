"""Root-only exact-pidfd cancellation of the original C3 supervisor."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,signal,importlib.util,sys,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=Path("research/reviews/C3_original_handover_root_20261016_v1")
ROSTER=dict(path=str(HERE/"pre_cancellation_owned_roster.root_v1.json"),sha256="694f71cae0f9fc70b13aa294dd78d1c2ebd0e59a737be659ba8700cc51eaa32d")
def ref(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--cancel",action="store_true");a=parser.parse_args()
 if not a.cancel:print(json.dumps(dict(status="source_only_no_signal")));return 0
 if Path.cwd()!=ROOT or ref(ROSTER["path"])!=ROSTER:raise ValueError("Exact root cwd/immutable owned roster")
 d=json.loads((ROOT/ROSTER["path"]).read_text())
 expected=next(r for r in d["identities"] if r["pid"]==3042044)
 if expected["start_ticks"]!=476961689 or expected["UID"]!=os.getuid() or d["physical_gpu"]!=4:raise ValueError("Original root identity")
 if (ROOT/HERE/"signal_root_SIGTERM_actual_v2.json").exists():raise ValueError("No repeated cancellation")
 backend_ref={'path': 'research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py', 'sha256': '61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca'}
 if ref(backend_ref["path"])!=backend_ref:raise ValueError("Exact already-reviewed original pidfd backend")
 spec=importlib.util.spec_from_file_location("_root_C3_cancel_pidfd_v2",ROOT/backend_ref["path"])
 backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
 pid=expected["pid"];fd=backend.pidfd_open_v1(pid,0)
 try:
  base=Path("/proc")/str(pid)
  values=(base/"stat").read_text().rsplit(")",1)[1].split()
  cmd=(base/"cmdline").read_bytes()
  if int(values[19])!=expected["start_ticks"] or base.stat().st_uid!=expected["UID"] or hashlib.sha256(cmd).hexdigest()!=expected["cmdline_sha256"] or int(values[2])!=3042044 or int(values[3])!=3042044:raise ValueError("Original identity changed after pidfd open")
  backend.pidfd_send_signal_v1(fd,signal.SIGTERM,None,0)
  result=dict(schema="C3_original_exact_root_pidfd_SIGTERM_actual_v1",root_authorizer="root_agent",signal_returned_successfully=True,
   signalled_pid=pid,signalled_start_ticks=expected["start_ticks"],signal="SIGTERM",delivery="original reviewed pidfd backend after exact UID/start/cmdline/PGID/SID revalidation",
   roster_ref=ROSTER,source_ref=ref(HERE/"cancel_original_root_v2.py"),backend_ref=backend_ref,native_GPU5_signalled=False,other_PIDs_signalled_by_this_tool=False,
   original_supervisor_owns_descendant_drain=True,actual_supervisor_terminal_pending=True,actual_direct_launcher_wait_pending=True,
   observed_utc=datetime.now(timezone.utc).isoformat())
 finally:os.close(fd)
 out=ROOT/HERE/"signal_root_SIGTERM_actual_v2.json"
 with out.open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps(dict(signal_receipt_ref=ref(out.relative_to(ROOT)),native_GPU5_preserved=True)),flush=True)
 return 0
if __name__=="__main__":
 try:raise SystemExit(main())
 except Exception:
  traceback.print_exc(file=sys.stdout)
  raise SystemExit(1)
