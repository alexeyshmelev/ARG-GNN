"""Inactive future ordinary completed close for successful full30Mb TRAIN001 simulation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/frozen_TRAIN001_simulator_close_ready_20261017_v1"
SOURCE=HERE+"/close_ROOT_v1.py";MANIFEST=HERE+"/source_manifest.v1.json"
def need(ok,message):
 if not ok:raise ValueError(message)
def local(value):
 p=Path(value);p=p if p.is_absolute()else ROOT/p
 need(p!=ROOT and p.is_relative_to(ROOT)and".."not in p.parts and p.resolve()==p,"Canonical contained path");return p
def ref(value):
 p=local(value);s=p.lstat();need(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=2**20,"Bounded singly linked source/metadata")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(record):
 need(type(record)is dict and set(record)=={"path","sha256"}and ref(record["path"])==record,"Exact immutable source/metadata ref")
 return local(record["path"]).read_bytes()
def read(record):return json.loads(raw(record))
def pins():return read(ref(HERE+"/pins.v1.json"))
def save(directory,name,value):
 data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode();need(len(data)<=65536,"Original compact operation metadata")
 p=directory/name
 existing=list(directory.iterdir());need(len(existing)<8 and all(x.is_file()and not x.is_symlink()for x in existing),"Bounded flat own close records")
 need(4096+sum(x.stat().st_size+4096 for x in existing)+len(data)+4096<=384*1024,"Own384KiB admin bound plus original64KiB event fits512KiB")
 with p.open("xb")as stream:need(stream.write(data)==len(data),"Complete exclusive record");stream.flush();os.fsync(stream.fileno())
 return ref(str(p.relative_to(ROOT)))
def metadata_join(p):
 inp=read(p["input"]);wait=read(p["wait"]);done=read(p["completed"]);outer=read(p["outer"])
 launch=read(p["launch"]);storage=read(p["storage"]);q=read(p["original_request"]);own=storage["reservation"]
 cfg=read(p["config"]);caps=read(inp["caps_ref"]);supervisor_wait=read(p["supervisor_wait"])
 need(inp["schema"]=="frozen_TRAIN001_current_Census_simulator_input_v1"
  and inp["parent_id"]==q["parent_id"]==wait["parent_id"]==done["parent_id"]=="american_c3_train_001"
  and inp["phase"]==q["phase"]==wait["phase"]=="simulate_train_001"
  and inp["stage"]==q["stage"]=="simulate"and inp["GPU_authorized"]is False
  and inp["query_truth_access_authorized"]is False and inp["TEST_access_authorized"]is False
  and inp["GNN_optimization_authorized"]is False,"Exact original full30Mb TRAIN001 CPU simulator")
 need(inp["root_driver_ref"]==inp["main_source_ref"]==p["producer_source"]
  and inp["source_sha256"].get(p["producer_source"]["path"])==p["producer_source"]["sha256"]
  and inp["root_stage_request_ref"]==p["original_request"]and q["source_manifest_ref"]==p["producer_manifest"]
  and q["source_review_ref"]==p["producer_review"]and q["storage_policy_selector_ref"]==p["selector"],
  "Exact reviewed v5 source and genuine admitted input lineage")
 for name in("producer_source","producer_manifest","producer_review"):raw(p[name])
 need(inp["refs"]["config"]==p["config"]and cfg==q["scientific_config"]==q["bindings"]["scientific_config"]
  and cfg["sequence_length_bp"]==30000000 and cfg["parent_id"]=="american_c3_train_001"
  and cfg["split"]=="train"and cfg["sample_counts"]==dict(ADMIX=1000,AFR=200,ASIA=200,EUR=200)
  and cfg["class_order"]==["EUR","EAS","AFR"]and cfg["sample_unit"]=="haploid"
  and cfg["ancestry_seed"]==21002 and cfg["mutation_seed"]==1021002
  and cfg["output_cap_bytes"]==6500000000 and caps["reservation_bytes"]==7000000000,
  "Original registered full-length science and7GBwhole reservation unchanged")
 need(inp["refs"]["storage"]==p["storage"]and inp["storage"]==storage
  and storage["accounting_mode"]=="Census_observation_two_scope_retention_v1"
  and storage["completed_overlay_policy_ref"]==p["policy"]and storage["budget_components"]["consumer_adapter"]==p["guard"]
  and storage["budget_components"]["implementation"]==p["implementation"]and storage["budget_components"]["launcher"]==p["launcher"]
  and own["key"]==p["reservation_key"]and own["reserved_bytes"]==storage["reserved_new_bytes"]==7000000000
  and own["output"]==str(local(inp["output_directory"]))and own["auxiliary_scopes"]==[str(local(inp["worker_auxiliary_scope"]))]
  and 1<=len(own["consumer_bindings"])<=64,"Exact existing original7GBscope and all original bindings retained")
 need(wait["schema"]=="American_C3_full_original_CPU_actual_wait_v1"and wait["input_ref"]==launch["input_ref"]==p["input"]
  and launch["pid"]==wait["pid"]and type(launch["pid"])is int and launch["pid"]>=2
  and wait["completed_ref"]==p["completed"]and type(wait["returncode"])is int and wait["returncode"]==0
  and type(wait["actual_OS_wait_status"])is int and wait["actual_OS_wait_status"]==0
  and wait["actual_direct_worker_wait"]is True and wait["actual_direct_launcher_wait"]is True
  and wait["all_owned_processes_and_groups_drained"]is True and wait["cleanup_errors"]==[]
  and wait["operational_failure"]is None and all(type(r["returncode"])is int and r["returncode"]==0 for r in wait["adopted_waits"]),
  "Genuine workerOSwait0 and full original owned drain required")
 need(supervisor_wait.get("actual_direct_wait")is True and type(supervisor_wait.get("actual_returncode"))is int
  and supervisor_wait["actual_returncode"]==0 and supervisor_wait["pid"]==wait["supervisor_pid"],
  "Separate genuine direct supervisorOSwait0")
 need(outer.get("schema")=="ROOT_frozen_TRAIN001_simulator_actual_outer_terminal_v1"
  and outer.get("root_authorizer")=="root_agent"and outer.get("actual_terminal_consumed")is True
  and type(outer.get("actual_exit_code"))is int and outer["actual_exit_code"]==0
  and type(outer.get("actual_tool_session_id"))is int and outer["actual_tool_session_id"]>0
  and outer.get("actual_worker_wait_ref")==p["wait"]and outer.get("actual_supervisor_wait_ref")==p["supervisor_wait"]
  and outer.get("completed_ref")==p["completed"]and outer.get("input_ref")==p["input"]
  and outer.get("all_owned_processes_and_groups_drained")is True and outer.get("new_ancestry_score")is False,
  "Genuine ROOT consumed simulator outer0, never a completion-file inference")
 need(done["status"]=="completed_C3_simulation_reconstruction"and done["sample_unit"]=="haploid"
  and done["class_order"]==["EUR","EAS","AFR"]and type(done["sites"])is int and 0<done["sites"]<=cfg["max_sites"]
  and type(done["actual_tables_bytes"])is int and 0<done["actual_tables_bytes"]<=cfg["max_tables_bytes"],
  "Actual complete original simulation receipt, without scientific payload reads")
 obs=read(done["observation_ref"])
 need(obs["schema"]=="american_c3_observed_v1"and obs["parent_id"]=="american_c3_train_001"
  and obs["sequence_length_bp"]==30000000 and obs["counts_haploid"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
  and obs["query_ancestry_truth_in_inputs"]is False and obs["migrations_and_internal_population_labels_removed"]is True,
  "Full30Mb/all1600sanitized observed metadata, private truth files unopened")
 need(launch["ownership"]["signals_use_pidfd_only"]is True and launch["ownership"]["no_numeric_group_signals"]is True
  and len(launch["ownership"]["owned_identities"])==1
  and launch["ownership"]["owned_identities"][0]["pid"]==launch["pid"]
  and type(launch["ownership"]["owned_identities"][0]["start"])is int,"Original exact launch PID/start format")
 group_rows=wait["group_observations"]
 need(isinstance(group_rows,list)and group_rows and all(r["absent"]is True for r in group_rows)
  and set(wait["known_owned_process_groups"])=={r["pgid"]for r in group_rows},"All original recorded process groups genuinely drained")
 need(all(type(r.get("pid"))is int and r["pid"]in wait["known_owned_pids"]for r in wait["adopted_waits"])
  and wait["supervisor_process_group"]==wait["supervisor_pid"]and launch["pid"]in wait["known_owned_process_groups"],"Every actual adopted wait and final original session group included")
 pids=sorted(set(wait["known_owned_pids"]+[wait["supervisor_pid"],launch["pid"]]))
 groups=sorted(set(wait["known_owned_process_groups"]+[wait["supervisor_process_group"]]))
 need(1<=len(pids)<=256 and launch["pid"]in wait["known_owned_pids"]and all(type(x)is int and x>=2 for x in pids)
  and 1<=len(groups)<=256 and all(type(x)is int and x>=2 for x in groups),"Complete finite original worker/helper/supervisor roster")
 need(local(p["completed"]["path"])==local(inp["output_directory"])/"completed.json","Original output completed.json only")
 return inp,wait,done,launch,storage,own,pids,groups

def current(value):
 now=datetime.now(timezone.utc);a=datetime.fromisoformat(value["created_at_utc"]);b=datetime.fromisoformat(value["expires_at_utc"])
 need(a.tzinfo and b.tzinfo and a<=now<b and 0<(b-a).total_seconds()<=3600,"Fresh finite ROOT one-shot close authority")
def request_value(p,own,launch,observation_ref):
 return dict(key=own["key"],reservation_sha256=p["reservation_sha256"],storage_ref=p["storage"],storage_locator=[],
  launch_ref=p["launch"],launch_pid_locator=["pid"],launch_identity=dict(pid=launch["pid"],input_ref=p["input"]),
  direct_wait_ref=p["wait"],wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
  wait_identity=dict(pid=launch["pid"],input_ref=p["input"],returncode=0,actual_OS_wait_status=0,
   actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,completed_ref=p["completed"]),
  completed_ref=p["completed"],completed_identity=dict(status="completed_C3_simulation_reconstruction",parent_id="american_c3_train_001"),
  observation_ref=observation_ref)
def close_once(authorization_ref):
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0,"Fresh ROOT -B -s nonoptimized interpreter")
 base=pins();m=read(ref(MANIFEST))
 for r in m["source_refs"]:raw(r)
 auth=read(authorization_ref);current(auth)
 p=dict(base);p.update(auth["evidence_refs"]);p.update(reservation_key=auth["reservation_key"],reservation_sha256=auth["reservation_sha256"])
 need(set(auth["evidence_refs"])=={"input","launch","wait","completed","outer","supervisor_wait","storage","original_request","config"},"Exact genuine future evidence refs only")
 review=read(auth["source_review_ref"])
 need(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"and review.get("blockers")==[]
  and review.get("source_ref")==ref(SOURCE)and review.get("source_manifest_ref")==ref(MANIFEST),"Fresh independent exact close source pass")
 need(auth.get("schema")=="ROOT_frozen_TRAIN001_simulator_ordinary_close_once_authorization_v1"
  and auth.get("root_authorizer")=="root_agent"and auth.get("active")is True and auth.get("once_authorized")is True
  and auth.get("operation")=="close_completed"and auth.get("source_manifest_ref")==ref(MANIFEST)
  and auth.get("reservation_key")==p["reservation_key"]
  and auth.get("reservation_sha256")==p["reservation_sha256"]and auth.get("storage_policy_selector_ref")==p["selector"]
  and auth.get("ROOT_has_serialized_all_mutations_and_resolved_prior_inflight_actions")is True
  and auth.get("preserve_all_files_and_original_reservations")is True and auth.get("no_retry_retirement_or_new_quota_framework")is True,"Exact ROOT ordinary close once authority")
 inp,wait,done,launch,storage,own,pids,group_ids=metadata_join(p)
 rev=read(auth["permanent_route_revocation_ref"])
 need(rev.get("schema")=="ROOT_frozen_TRAIN001_simulator_permanent_route_revocation_v1"
  and rev.get("root_authorizer")=="root_agent"and rev.get("active")is True
  and rev.get("input_ref")==p["input"]and rev.get("direct_wait_ref")==p["wait"]and rev.get("completed_ref")==p["completed"]
  and rev.get("actual_outer_terminal_ref")==p["outer"]and rev.get("reservation_key")==own["key"]
  and rev.get("reservation_sha256")==p["reservation_sha256"]and rev.get("output_directory")==inp["output_directory"]
  and rev.get("worker_auxiliary_scope")==inp["worker_auxiliary_scope"]and rev.get("all_original_launch_resume_retry_write_routes_permanently_revoked")is True
  and rev.get("preserve_all_files_original_clocks_failures_and_reservations")is True,"Separate exact permanent ROOT original-route revocation")
 stamp=datetime.fromisoformat(rev["revoked_at_utc"]);need(stamp.tzinfo and stamp<=datetime.now(timezone.utc),"Actual nonfuture permanent revocation time")
 directory=local(auth["run_directory"]);need(str(directory.relative_to(ROOT)).startswith(HERE+"/ROOT_actual_success_"),"Fresh ROOT close administrative scope")
 directory.mkdir() # Exclusive: consumes this invocation before guard import, state or probes. Never reuse after uncertainty.
 save(directory,"invocation.json",dict(authorization_ref=authorization_ref,source_manifest_ref=ref(MANIFEST),automatic_retry=False))
 raw(p["guard"]);sp=importlib.util.spec_from_file_location("_ROOT_frozen_TRAIN001_simulator_exact_current_Census_close",local(p["guard"]["path"]))
 module=importlib.util.module_from_spec(sp);sys.modules[sp.name]=module;sp.loader.exec_module(module)
 g=module.load_selected_guard_v1(p["selector"])
 need(g is module and g.MODE=="Census_observation_two_scope_retention_v1"and g._POLICY_REF==p["policy"],"Exact selected current guard, never old Relate64 configuration")
 need(g.budget.digest(own)==p["reservation_sha256"],"Original budget reservation digest")
 current(auth);rows=[g._absent(pid)for pid in pids];group_rows=[]
 for group in group_ids:
  try:os.killpg(group,0)
  except ProcessLookupError:group_rows.append(dict(pgid=group,absent=True))
  else:raise ValueError("Recorded worker/supervisor group present, reused or unknown; no close")
 obs=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",key=own["key"],
  reservation_sha256=p["reservation_sha256"],storage_ref=p["storage"],launch_ref=p["launch"],direct_wait_ref=p["wait"],completed_ref=p["completed"],
  launcher_pid=launch["pid"],all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
  completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,
  pids=pids,observations=rows,actual_outer_terminal_ref=p["outer"],permanent_route_revocation_ref=auth["permanent_route_revocation_ref"],
  groups=group_ids,group_observations=group_rows,original_failures_and_clocks_preserved=True)
 observation_ref=save(directory,"targeted_completion_observation.json",obs);request=request_value(p,own,launch,observation_ref)
 with g.budget.v4.mutex(ROOT):
  state=g._state_locked(p["implementation"]["sha256"],p["launcher"]["sha256"],fresh=True)
  need(len(state[3])==47 and state[0][own["key"]]==own and own["key"]not in state[1],"Exact open original reservation and all47 current roles")
  complete=g.consumer_bindings_v1(own["consumer_bindings"],list(state[3].values()))
  need(complete==own["consumer_bindings"]and len(complete)<=64,"No role omission, reservation rebind or original64cap increase")
  current(auth);authority=g.operation_draft_v1("close_completed",request,expected_before=g._pins(state[2]),bindings=state[3])
  authority.update(active=True,expires_at_utc=auth["expires_at_utc"])
  authority_ref=save(directory,"ordinary_close_authority.json",authority)
 print(json.dumps(dict(ordinary_close_authority_ref=authority_ref,automatic_retry=False)),flush=True)
 current(auth)
 receipt=g.apply_close_completed_v1(authority_ref,expected_source_sha256=p["implementation"]["sha256"],expected_launcher_sha256=p["launcher"]["sha256"])
 receipt_ref=save(directory,"ordinary_close_receipt.json",receipt)
 result=dict(status="ORDINARY_CLOSE_APPLIED",receipt_ref=receipt_ref,authority_ref=authority_ref,retained_bytes=receipt["retained_bytes"],
  credit_bytes=receipt["credit_bytes"],projected_bytes=receipt["projected_bytes"],all_files_preserved=True,scientific_score=False,
  ROOT_actual_outer_terminal_consumption_still_required=True,automatic_retry=False)
 save(directory,"result.json",result);return result
def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--authorization-ref");args=parser.parse_args(argv)
 if args.authorization_ref is None:print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",guard_imported=False,operation=False)));return 0
 print(json.dumps(close_once(json.loads(args.authorization_ref)),sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
