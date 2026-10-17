"""Saved outer000 observed-only validation and new acceptance. Default import performs no IO.
Original34504outer1, UNKNOWN worker wait/drain and lost monitor interval are immutable.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib,json,os,stat,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H059_saved_outer000_acceptance_and_continuation_source_20261017_v1"
SOURCE=HERE+"/saved_outer000_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json"
BASE="data/expanded_full_panel_v1/ap_expanded_outer_000"
PARENT="ap_expanded_outer_000"
ROOT_METADATA={"path":"research/reviews/research_execution_ROOT_20261017_v25/H059_outer000_observed_completion_metadata.root_v1.json","sha256":"c219f2e6bb2b8eec58fdf3c8646d13dbdc3d334c283ed8d5d32421f488ecc428"}
COMPLETION={"path":BASE+"/expanded_parent_completed.json","sha256":"3f9664808857c58bbc2e723e6a564de61305bcf12776004339b96719bdfff8ed"}
INPUT={"path":"research/reviews/European_outer000_ROOT_actual_Relate64_20261017_v1/input.json","sha256":"3561b5898cb6abc553b2a8bff4528baf02f6a2a02c416d9c696f7355376b7010"}
OBSERVED=("genotypes.vcf","sample_manifest.json","sample_order.txt","sample_ages.generations","population_labels.txt",
 "site_coordinate_manifest.json","uniform_recombination_map.txt","scoring_coordinates.npz","observation_manifest.json")
HISTORY=dict(original_tool=34504,original_outer_exit_code=1,original_worker_PID=3998053,
 original_worker_OS_wait="UNKNOWN",original_owned_descendant_drain="UNKNOWN",preserved_missing_monitoring_interval=True,
 full_run_peak_RSS_certified=False,old_scope_completion_or_ordinary_close_or_credit=False)
KNOWN_IDENTITIES=((3998053,488962157),(4036548,489505214),(4077391,489552358))
KNOWN_GROUPS=(3998053,4036548)
LABELS=("convert_vcf","annotate_observed_cohorts","relate_initial","five_population_size_iterations","convert_post_size")
def need(ok,message):
 if not ok:raise ValueError(message)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def metadata_v1(read):
 root=read(ROOT_METADATA);completed=read(COMPLETION);inp=read(INPUT);cfg=read(inp["config_ref"])
 need(root["schema"]=="ROOT_H059_observed_artifact_metadata_v1"and root["completed_ref"]==COMPLETION
  and root["original_worker_OS_wait"]=="UNKNOWN"and root["original_owned_descendant_drain"]=="UNKNOWN"
  and root["original_supervisor_outer_exit_code"]==1 and root["preserved_missing_monitoring_interval"]is True,"Exact genuine ROOT observation; no historical success upgrade")
 need(completed["schema"]=="expanded_full_panel_v1_parent_completion"and completed["parent_id"]==PARENT
  and completed["split"]=="outer"and completed["downstream_role"]=="UNTOUCHED_OUTER"
  and completed["generation_completed"]is True and completed["Relate_completed"]is True
  and completed["outer_truth_evaluated"]is False and completed["models_prepared"]is False and completed["GPU_used"]is False
  and completed["source_ref"]==inp["source_ref"]and completed["config_content_sha256"]==inp["config_content_sha256"]==digest(cfg)
  and completed["parent_registry_sha256"]==inp["parent_registry_sha256"]==digest(cfg["parent_registry"]),"Exact original generation metadata/config/source/registry")
 rows=[x for x in cfg["parent_registry"]if x["parent_id"]==PARENT]
 need(len(rows)==1 and rows[0]["split"]=="outer"and rows[0]["evaluation_authorized"]is False,"Original reserved outer000 seed row")
 generation=read(completed["generation_ref"])
 need(generation["parent_id"]==PARENT and generation["split"]=="outer"and generation["query_haplotypes"]==686
  and generation["status"]=="observations_and_external_targets_prepared","Original observed generation metadata")
 return completed,inp,cfg,rows[0],generation
def native_metadata_v1(completed,read):
 inference=read(completed["inference_ref"]);entry=read(completed["observation_entry_ref"])
 commands_ref=inference["commands"];commands=read(commands_ref);steps=commands["steps"]
 need(inference["parent_id"]==entry["parent_id"]==PARENT and inference["split"]==entry["split"]=="outer"
  and inference["status"]=="post_five_iterations_inferred"and inference["truth_or_oracle_read"]is False
  and inference["outer_command_returncodes"]==[0]*5 and [x["label"]for x in steps]==list(LABELS)
  and len(steps)==5 and all(type(x["returncode"])is int and x["returncode"]==0 and x["status"]=="complete"
   and x["command_sha256"]==digest(x["argv"])for x in steps),"Five genuine source-bound outer helper wait0 receipts")
 need(inference["native_status_scope"]=="outer helper returncodes; bash-x trace retained; child statuses not independently asserted",
  "Outer helper statuses never become complete historical shell-descendant evidence")
 need(entry["schema"]=="mutation_routed_full_panel_observed_entry_v1"and entry["observations_only"]is True
  and entry["truth_or_oracle_read"]is False and (entry["samples"],entry["query_haplotypes"],entry["pure_reference_haplotypes"])==(952,686,266)
  and entry["original_native_inference_receipt"]==completed["inference_ref"]and entry["refs"]["native_tree"]==inference["inferred_tree"],
  "Original full952 observed-only handoff")
 return inference,entry,commands_ref
def known_absence_v1():
 rows=[];groups=[]
 for pid,start in KNOWN_IDENTITIES:
  try:os.kill(pid,0)
  except ProcessLookupError:pass
  else:raise ValueError("Known numeric PID exists or was reused")
  rows.append(dict(pid=pid,start_ticks=start,absent=True,method="numeric ESRCH",checked_utc=datetime.now(timezone.utc).isoformat()))
 for pgid in KNOWN_GROUPS:
  try:os.killpg(pgid,0)
  except ProcessLookupError:pass
  else:raise ValueError("Known original group exists or was reused")
  groups.append(dict(pgid=pgid,absent=True,method="numeric group ESRCH",checked_utc=datetime.now(timezone.utc).isoformat()))
 return dict(known_PID_observations=rows,known_group_observations=groups,complete_historical_descendant_absence_claimed=False)
def validate_absence_values_v1(value):
 need(set(value)=={"known_PID_observations","known_group_observations","complete_historical_descendant_absence_claimed"}
  and value["complete_historical_descendant_absence_claimed"]is False,"Known observations preserve incomplete history")
 need(len(value["known_PID_observations"])==len(KNOWN_IDENTITIES)and len(value["known_group_observations"])==len(KNOWN_GROUPS),"Every sealed known root/group")
 for row,(pid,start)in zip(value["known_PID_observations"],KNOWN_IDENTITIES):
  need(row["pid"]==pid and row["start_ticks"]==start and row["absent"]is True and row["method"]=="numeric ESRCH","Known exact PID/start ESRCH row")
 for row,pgid in zip(value["known_group_observations"],KNOWN_GROUPS):
  need(row["pgid"]==pgid and row["absent"]is True and row["method"]=="numeric group ESRCH","Known exact group ESRCH row")
 now=datetime.now(timezone.utc)
 for row in value["known_PID_observations"]+value["known_group_observations"]:
  t=datetime.fromisoformat(row["checked_utc"]);need(t.tzinfo and 0<=(now-t).total_seconds()<=120,"Fresh known absence required")
 return value
def stream_observed_v1(record,guard,deadline,*,maximum=4000000000):
 path=Path(record["path"])
 need(not path.is_absolute()and ".."not in path.parts and path.parts[:3]==tuple(Path(BASE).parts)
  and not any(x.lower()in("truth","oracle")for x in path.parts),"Only contained outer000 observed/native inputs")
 p=ROOT/path;need(p.resolve()==p,"No linked observed input ancestor")
 def fields(s):return(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 guard();before=p.lstat();need(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and 0<before.st_size<=maximum,"Bounded regular observed artifact")
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);h=hashlib.sha256();count=0
 with os.fdopen(fd,"rb")as f:
  need(fields(os.fstat(f.fileno()))==fields(before),"Opened exact observed object")
  while True:
   need(time.monotonic()<deadline,"Finite saved-observed validator");guard()
   block=f.read(2**20)
   if not block:break
   h.update(block);count+=len(block);need(count<=maximum,"Observed artifact byte cap")
  need(fields(os.fstat(f.fileno()))==fields(before)==fields(p.lstat()),"Stable observed object before/after stream")
 need(count==before.st_size and h.hexdigest()==record["sha256"],"Exact saved observed bytes")
 return dict(ref=record,bytes=count,identity=list(fields(before)))
def validate_saved_v1(read,ref,guard,*,wall_seconds=600):
 """Future separately admitted CPU validator only. No generation, native command, fit or truth read."""
 deadline=time.monotonic()+wall_seconds
 before=validate_absence_values_v1(known_absence_v1())
 completed,inp,cfg,parent,generation=metadata_v1(read)
 inference,entry,commands_ref=native_metadata_v1(completed,read)
 expected={BASE+"/observed/"+name:generation["files"]["observed/"+name]["sha256"]for name in OBSERVED}
 for r in [inference["inferred_tree"],inference["observed_sample_manifest"],completed["inference_ref"],
  completed["observation_entry_ref"],entry["refs"]["inference_receipt"],commands_ref]:
  need(r["path"]not in expected or expected[r["path"]]==r["sha256"],"No conflicting observed reference")
  expected[r["path"]]=r["sha256"]
 rows=[stream_observed_v1(dict(path=p,sha256=h),guard,deadline)for p,h in expected.items()]
 need(sum(x["bytes"]for x in rows)<=4000000000,"Original whole scientific output cap; no fit certification")
 # Original observed semantics and full coordinate/query contracts. Imports occur only in this admitted worker.
 original=importlib.import_module("arggnn.prepare_ancestralpaths_full_panel_discovery_v1")
 need(ref("src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py")==completed["original_producer_ref"],"Pinned unchanged observed-assets validator")
 original.observed_assets(cfg,parent,ROOT/BASE)
 validator=importlib.import_module("research.reviews.European_outer8_preparation_source_20261016_v3.outer_prediction_validator_v2")
 support=validator.observed_support(PARENT,completed["observation_entry_ref"],validator.ObservedReaderV2(guard))
 need(len(support["query_keys"])==686 and len(support["source"])==len(support["derived"])==952,"Original full952/686 semantics")
 after=validate_absence_values_v1(known_absence_v1());guard()
 # Identity revalidation after semantic readers; no old files are written.
 for row in rows:
  p=ROOT/row["ref"]["path"];s=p.lstat()
  need(list((s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns))==row["identity"],"Observed object unchanged through semantic validation")
 need(time.monotonic()<deadline,"Finite final saved-observed validation")
 return dict(schema="H059_saved_outer000_observed_validation_result_v1",status="VALIDATED_SAVED_OBSERVED_ONLY",
  parent_id=PARENT,completion_ref=COMPLETION,observation_entry_ref=completed["observation_entry_ref"],
  original_input_ref=INPUT,original_history=HISTORY,observed_files=rows,observed_bytes=sum(x["bytes"]for x in rows),
  full952_roster_686queries_and_original_clipped_grid_validated=True,
  known_absence_before=before,known_absence_after=after,original_guard_monitoring_interval_recovered=False,
  private_truth_opened=False,native_rerun=False,optimizer_updates=0,CUDA_context_created=False,
  original_scope_completion_or_close_or_credit_authorized=False)
def acceptance_values_v1(record,read,ref):
 a=read(record);need(a.get("schema")=="H059_saved_outer000_ROOT_artifact_acceptance_v1"and a.get("active")is True
  and a.get("root_authorizer")=="root_agent"and a.get("accepted_for_observed_inference_only")is True
  and a.get("original_history")==HISTORY and a.get("completion_ref")==COMPLETION
  and a.get("source_ref")==ref(SOURCE)and a.get("source_manifest_ref")==ref(MANIFEST)
  and a.get("ordinary_close_or_credit_authorized")is False,"Separate ROOT observed-only acceptance; old history immutable")
 review=read(a["source_review_ref"])
 need(review.get("independent")is True and review.get("verdict")in("SOURCE_PASS","PASS")and review.get("blockers")==[]
  and review.get("saved_observed_source_ref")==ref(SOURCE)and review.get("source_manifest_ref")==ref(MANIFEST),"Actual complete saved-observed source review")
 result=read(a["validation_result_ref"]);wait=read(a["validator_wait_ref"]);terminal=read(a["validator_ROOT_terminal_ref"])
 driver_ref=ref(HERE+"/validation_driver_v1.py");completion=read(a["validator_completion_ref"]);admission=read(a["validator_input_ref"])
 validator_request=read(admission["root_request_ref"])
 need(review.get("validator_driver_source_ref")==driver_ref
  and admission["schema"]=="H059_saved_outer000_validator_admission_v1"
  and admission["source_ref"]==driver_ref and admission["operational_source_ref"]==driver_ref
  and admission["operational_manifest_ref"]==ref(MANIFEST)
  and completion["schema"]=="H059_saved_outer000_validator_completion_v1"
  and completion["input_ref"]==a["validator_input_ref"]and completion["driver_source_ref"]==driver_ref
  and completion["validation_result_ref"]==a["validation_result_ref"]and completion["original_history"]==HISTORY
  and validator_request["old_writer_revocation_ref"]==a["old_writer_revocation_ref"]==result["old_writer_revocation_ref"],
  "Actual dedicated reviewed validator input and completion bind the result")
 need(result["schema"]=="H059_saved_outer000_observed_validation_result_v1"and result["status"]=="VALIDATED_SAVED_OBSERVED_ONLY"
  and result["original_history"]==HISTORY and result["completion_ref"]==COMPLETION
  and result["full952_roster_686queries_and_original_clipped_grid_validated"]is True and result["private_truth_opened"]is False
  and result["original_scope_completion_or_close_or_credit_authorized"]is False,"Genuine dedicated validator result")
 need(wait["schema"]=="H059_saved_outer000_validator_actual_wait_v1"and type(wait["returncode"])is int and wait["returncode"]==0
  and wait["actual_OS_wait_status"]==0 and wait["actual_direct_worker_wait"]is True
  and wait["all_owned_processes_and_groups_drained"]is True and wait["complete_owned_roster_known"]is True
  and wait["operational_failure"]is None and wait["input_ref"]==a["validator_input_ref"]and wait["completion_ref"]==a["validator_completion_ref"]and wait["driver_source_ref"]==driver_ref
  and terminal["root_authorizer"]=="root_agent"and terminal["actual_tool_terminal_consumed"]is True
  and type(terminal["actual_tool_exit_code"])is int and terminal["actual_tool_exit_code"]==0
  and terminal["actual_wait_ref"]==a["validator_wait_ref"]and terminal["input_ref"]==a["validator_input_ref"]
  and terminal["validation_result_ref"]==a["validation_result_ref"]and terminal["completion_ref"]==a["validator_completion_ref"],"New dedicated validator genuine wait0/drain/consumed outer0")
 # These are later historical observations, not claims of current or historical complete absence.
 need(result["known_absence_before"]["complete_historical_descendant_absence_claimed"]is False
  and result["known_absence_after"]["complete_historical_descendant_absence_claimed"]is False
  and result["original_guard_monitoring_interval_recovered"]is False,"Lost monitoring and original descendant uncertainty preserved")
 rev=read(a["old_writer_revocation_ref"])
 need(rev["root_authorizer"]=="root_agent"and rev["active"]is True and rev["original_input_ref"]==INPUT
  and rev["all_original_launch_resume_retry_write_routes_permanently_cooperatively_revoked"]is True
  and rev["all_original_files_charges_clocks_and_failures_preserved"]is True
  and rev["read_only_artifact_reuse_preserved"]is True,"Exact permanent cooperative old-writer revocation")
 return a
