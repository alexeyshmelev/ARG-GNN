"""Root closes successful complete native prep, retaining all original native files."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse,hashlib,json,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/American_C3_native_paths_root_v3"
PACKET=HERE/"admitted_native_paths_train000_full30mb_v3"
def require(ok,message):
 if not ok:raise ValueError(message)
def ref(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(rec):
 require(ref(rec["path"])==rec,"Exact original native evidence ref required")
 return json.loads((ROOT/rec["path"]).read_bytes())
def save(folder,name,value):
 raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
 require(len(raw)<=65536,"Compact root close certificate")
 p=folder/name
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return ref(p)
def absent(pid,group=False):
 try:
  if group:os.killpg(pid,0)
  else:os.kill(pid,0)
 except ProcessLookupError:
  return dict(**({"process_group":pid} if group else {"pid":pid}),absent=True,
    method="targeted os.killpg(group,0) ESRCH" if group else "targeted os.kill(pid,0) ESRCH",
    checked_utc=datetime.now(timezone.utc).isoformat())
 raise ValueError("Original native PID/group is present, reused or zombie")
def run():
 from research.reviews.H044_marker_resolution_root_v1.root_driver_v4 import context,load_guard,CORE,BASE
 ar,lr,wr=ref(PACKET/"admission.json"),ref(PACKET/"launch.json"),ref(PACKET/"actual_wait.json")
 admission,launch,wait=read(ar),read(lr),read(wr)
 complete_ref=wait["completed_ref"];completed=read(complete_ref);dispatch=read(ref(HERE/"root_dispatch_20261015_v1.json"))
 own=admission["storage"]["reservation"]
 require(own["reserved_bytes"]==6000000000 and own["key"]=="H041/American-C3-native-paths-train000-full30mb-92001-v3",
   "Exact original fully charged native v3 scope")
 require(wait["returncode"]==0 and wait["actual_direct_worker_wait"] is True and wait["owned_worker_group_ESRCH"] is True
   and wait["operational_failure"] is None and not wait["cleanup_errors"] and not wait["post_wait_errors"] and not wait["supervisor_errors"],
   "Genuine native original direct wait0 and complete group drain required")
 require(wait["pid"]==wait["worker_pid"]==launch["pid"]==2677571
   and wait["supervisor_pid"]==launch["supervisor_pid"]==dispatch["root_driver_pid"]==2677398
   and wait["owned_PID_roster_observed"]==[2677571],"Complete original native writer roster required")
 expected=dict(status="completed_native_C3_path_preparation",unique_paths=25574587,path_tokens=316672003,
   path_occurrences=122980800,vocabulary_nodes=106769,max_raw_path_nodes=29,
   reference_unique_paths=9583977,reference_path_tokens=116080787,query_focal_count=10000,
   query_ancestry_truth_opened=False,contrastive_labels_reference_only=True,native_split_seed=42,native_train_fraction=.9)
 require(all(completed.get(k)==v for k,v in expected.items()),"All native full-corpus scientific counts and split/control evidence")
 require(len(completed["native_dataset_refs"])==6 and len(completed["full_path_id_order_refs"])==6,
   "All six native datasets and row-order sidecars required")
 pids=[2677398,2677571];observations=[absent(x) for x in pids];groups=[absent(2677571,True)]
 directory=HERE/"completed_scope_close_root_v1";directory.mkdir()
 revocation=save(directory,"scope_write_resume_revocation.json",dict(root_authorizer="root_agent",
   key=own["key"],all_original_scope_writes_resumes_retries_revoked=True,
   future_native_training_must_use_fresh_cache_and_output_scopes=True,
   read_only_native_corpus_reuse_preserved=True,preserve_all_files=True,
   source_driver_ref=admission["driver_ref"],actual_wait_ref=wr,pids=pids,observations=observations,worker_groups=groups))
 m=load_guard(context())
 observation=save(directory,"root_observation.json",dict(schema="concurrent_completed_scope_targeted_root_observation_v1",
   root_authorizer="root_agent",key=own["key"],reservation_sha256=m.budget.digest(own),
   storage_ref=ar,launch_ref=lr,direct_wait_ref=wr,completed_ref=complete_ref,launcher_pid=launch["pid"],
   all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
   completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
   external_NFS_writers_outside_observation=True,pids=pids,observations=[absent(x) for x in pids],
   worker_group_observations=[absent(2677571,True)],scope_revocation_ref=revocation))
 request=dict(key=own["key"],reservation_sha256=m.budget.digest(own),storage_ref=ar,storage_locator=["storage"],
   launch_ref=lr,launch_pid_locator=["pid"],direct_wait_ref=wr,wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
   launch_identity={k:launch[k] for k in ("pid","worker_pid","process_group","supervisor_pid","admission_ref","driver_ref","consumer_wrapper_ref","command","stage")},
   wait_identity={k:wait[k] for k in ("returncode","pid","worker_pid","process_group","supervisor_pid","actual_direct_worker_wait",
     "owned_worker_group_ESRCH","operational_failure","owned_PID_roster_observed","admission_ref","driver_ref","completed_ref",
     "cleanup_errors","post_wait_errors","supervisor_errors","no_scientific_subsampling_or_cap_fallback")},
   completed_ref=complete_ref,completed_identity=expected,observation_ref=observation)
 with m.budget.v4.mutex(ROOT):
  state=m._state_locked(CORE,BASE,fresh=True);m.assert_open(own["key"],state[0],state[1])
  require(state[0][own["key"]]==own,"Original native reservation unchanged")
  authority=m.operation_draft_v1("close_completed",request,expected_before=m._pins(state[2]),bindings=state[3])
 authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
 receipt=m.apply_close_completed_v1(save(directory,"close_authority.json",authority),expected_source_sha256=CORE,expected_launcher_sha256=BASE)
 saved=save(directory,"close_receipt.json",receipt)
 print(json.dumps(dict(receipt_ref=saved,retained_bytes=receipt["retained_bytes"],credit_bytes=receipt["credit_bytes"],
   projected_bytes=receipt["projected_bytes"],artifacts_deleted=0)),flush=True)
 return 0
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true");a=p.parse_args()
 if not a.execute:print('{"status":"inactive_root_only_original_wait_and_absence_required"}')
 else:raise SystemExit(run())
