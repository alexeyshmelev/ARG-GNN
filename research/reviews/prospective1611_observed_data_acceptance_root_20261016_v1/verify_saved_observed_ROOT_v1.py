"""Root read-only byte verification of completed observed exports after child wait0."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,stat,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/prospective1611_observed_data_acceptance_root_20261016_v1"
OBS=ROOT/"data/ARGMix_prospective1611_first_train_H020v3_generated_train000_oct16_root_v1/operational_successor_train000_live_boundary_root_v1/observed/observed_generation_receipt.json"
TERM=ROOT/"research/reviews/ARGMix_prospective1611_live_boundary_successor_source_20261016_v3/actual_tool_terminal.root_v1.json"
def ref(p):
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def ident(p):
 s=p.lstat()
 assert p.resolve()==p and p.is_relative_to(ROOT) and stat.S_ISREG(s.st_mode) and s.st_nlink==1
 return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def absent(pid):
 try:os.kill(pid,0)
 except ProcessLookupError:return dict(pid=pid,absent=True,observed_utc=datetime.now(timezone.utc).isoformat())
 raise RuntimeError("Exact owned process is not absent")
def main():
 assert Path.cwd()==ROOT
 start=time.monotonic()
 assert ref(OBS)["sha256"]=="60e2f59041578f90207f7c41cee81f628582a7394ae6e0ce3acb524a163b9659"
 term=json.loads(TERM.read_text())
 assert term["actual_terminal_consumed"] and term["actual_exit_code"]==1 and term["actual_numerical_child_returncode"]==0
 wait_path=ROOT/term["actual_wait_ref"]["path"]
 assert ref(wait_path)==term["actual_wait_ref"]
 wait=json.loads(wait_path.read_text())
 assert wait["returncode"]==0 and wait["all_owned_processes_and_groups_drained"] is True
 assert wait["operational_failure"] is not None
 before_absence=[absent(pid) for pid in (3477332,3477878)]
 observed=json.loads(OBS.read_text())
 assert observed["complete"] is True and observed["contains_genealogy_or_founder_truth"] is False
 assert (observed["diploids"],observed["haplotypes"],observed["observed_sites"])==(1611,3222,311599)
 rows=[]
 for name,entry in observed["files"].items():
  p=Path(entry["path"])
  assert p.parent==OBS.parent
  before=ident(p)
  assert before["bytes"]==entry["bytes"]
  h=hashlib.sha256()
  with p.open("rb") as f:
   for block in iter(lambda:f.read(4*2**20),b""):h.update(block)
  after=ident(p)
  assert before==after and h.hexdigest()==entry["sha256"]
  rows.append(dict(manifest_key=name,path=str(p.relative_to(ROOT)),sha256=h.hexdigest(),bytes=before["bytes"],identity_before_and_after=before))
 assert len(rows)==7
 after_absence=[absent(pid) for pid in (3477332,3477878)]
 result=dict(schema="prospective1611_saved_observed_independent_byte_verification_root_v1",
  status="PASS",root_authorizer="root_agent",observed_utc=datetime.now(timezone.utc).isoformat(),
  source_ref=ref(Path(__file__)),observed_receipt_ref=ref(OBS),actual_outer_terminal_ref=ref(TERM),actual_child_wait_ref=ref(wait_path),
  numerical_child_returncode=0,parent_outer_returncode=1,operational_failure_preserved=wait["operational_failure"],
  exact_owned_root_and_child_absent_before=before_absence,exact_owned_root_and_child_absent_after=after_absence,
  observed_products_independently_stream_hashed=rows,observed_bytes_stream_hashed=sum(x["bytes"] for x in rows),
  private_truth_payload_opened=False,simulation_rerun=False,optimizer_updates=0,CUDA_context_created=False,
  independent_sample_spec_semantics_recomputed=False,full_run_peak_RSS_certified=False,
  data_acceptance_and_downstream_execution_authorized=False,storage_close_or_credit_authorized=False,
  elapsed_seconds=time.monotonic()-start)
 with (HERE/"byte_integrity.root_v1.json").open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps(dict(status=result["status"],observed_products=len(rows),observed_bytes=result["observed_bytes_stream_hashed"],elapsed_seconds=result["elapsed_seconds"],receipt_ref=ref(HERE/"byte_integrity.root_v1.json"),data_acceptance_pending=True)),flush=True)
if __name__=="__main__":main()
