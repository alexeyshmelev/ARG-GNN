"""Inactive one-shot ordinary completed close for exact actual FLARE VAL004 native."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_VAL004_native_current_Census_close_source_20261017_v1"
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
 with p.open("xb")as stream:need(stream.write(data)==len(data),"Complete exclusive record");stream.flush();os.fsync(stream.fileno())
 return ref(str(p.relative_to(ROOT)))
def metadata_join(p):
 inp=read(p["input"]);wait=read(p["wait"]);done=read(p["completed"]);outer=read(p["outer"])
 launch=read(p["launch"]);storage=read(p["storage"]);q=read(p["original_request"]);own=storage["reservation"]
 cfg=read(p["config"]);count=read(p["count"]);native_wait=read(p["native_wait"]);conversion=read(p["conversion"])
 need(inp["parent_id"]==q["parent_id"]==wait["parent_id"]==done["parent_id"]==count["parent_id"]=="american_c3_val_004"
  and inp["phase"]==q["phase"]==wait["phase"]=="native_american_c3_val_004"
  and inp["stage"]==q["stage"]=="native","Exact original VAL004 native")
 need(inp["root_driver_ref"]==p["producer_driver"]and inp["main_source_ref"]==p["producer_worker"]
  and inp["source_sha256"].get(p["producer_driver"]["path"])==p["producer_driver"]["sha256"]
  and inp["source_sha256"].get(p["producer_worker"]["path"])==p["producer_worker"]["sha256"]
  and inp["root_stage_request_ref"]==p["original_request"]and q["source_manifest_ref"]==p["producer_manifest"]
  and q["source_review_ref"]==p["producer_review"],"Actual same-source admitted producer lineage")
 raw(p["producer_driver"]);raw(p["producer_worker"]);raw(p["producer_manifest"]);raw(p["producer_review"])
 need(inp["refs"]["config"]==p["config"]and cfg["count_manifest_ref"]==p["count"]
  and count["schema"]=="American_C3_full30Mb_observation_count_root_v1"and count["num_samples"]==1600
  and count["num_sites"]==cfg["max_sites_bound"]==229643 and count["sequence_length"]==30000000
  and count["truth_opened"]is False and cfg["external_truth_ref"]is None
  and q["caps"]==p["original_caps"]and q["caps"]["reservation_bytes"]==2003235369
  and cfg["output_cap_bytes"]==q["caps"]["output_bytes"]==1994846761
  and done["native_options"]==cfg["native_options"]==p["native_options"]
  and done["conversion_ref"]==cfg["conversion_ref"]==q["conversion_ref"]==p["conversion"]
  and conversion["parent_id"]=="american_c3_val_004"and conversion["status"]=="completed_C3_FLARE_observation_conversion"
  and conversion["input_sites"]==229643 and conversion["reference_haplotypes"]==600
  and conversion["query_haplotypes"]==1000 and conversion["common_focal_records"]==10000
  and conversion["unsupported_sites_dropped"]==0 and conversion["native_expected_markers"]==26532
  and conversion["query_ancestry_truth_opened"]is False and conversion["common_focal_registry_unchanged"]is True
  and done["focal_registry_ref"]==cfg["focal_registry_ref"],"Exact original counts, caps, scientific options and full native input refs")
 need(wait["schema"]=="FLARE_phase_CPU_actual_wait_v2"and wait["input_ref"]==launch["input_ref"]==p["input"]
  and wait["completed_ref"]==p["completed"]and type(wait["returncode"])is int and wait["returncode"]==0
  and type(wait["actual_OS_wait_status"])is int and wait["actual_OS_wait_status"]==0
  and wait["actual_direct_worker_wait"]is True and wait["actual_direct_launcher_wait"]is True
  and wait["all_owned_processes_and_groups_drained"]is True and wait["cleanup_errors"]==[]
  and wait["operational_failure"]is None and all(type(r["returncode"])is int and r["returncode"]==0 for r in wait["adopted_waits"]),"Genuine direct OSwait0 and complete original owned drain")
 # Consume the genuine VAL004 ROOT schema directly; missing VAL002 wrapper fields are not synthesized.
 need(outer["schema"]=="ROOT_FLARE_VAL004_native_actual_terminal_consumed_v1"
  and outer["root_authorizer"]=="root_agent"and outer["actual_terminal_consumed"]is True
  and type(outer["actual_exit_code"])is int and outer["actual_exit_code"]==0
  and type(outer["actual_tool_session_id"])is int and outer["actual_tool_session_id"]==40429
  and outer["actual_direct_wait_ref"]==p["direct_supervisor_wait"]
  and outer["actual_worker_wait_ref"]==p["wait"]and outer["completed_ref"]==p["completed"]
  and outer["all_owned_processes_and_groups_drained"]is True
  and outer["no_new_ancestry_score"]is True and outer["input_ref"]==p["input"]
  and type(outer["actual_outer_exit_code"])is int and outer["actual_outer_exit_code"]==0
  and type(outer["actual_direct_worker_returncode"])is int and outer["actual_direct_worker_returncode"]==0
  and outer["query_ancestry_truth_opened"]is False,"Exact genuine consumed ROOT40429 outer0 and both actual wait refs")
 need(done["status"]=="completed_C3_native_FLARE"and done["query_ancestry_truth_opened"]is False
  and launch["pid"]==wait["pid"]==wait["actual_worker_scientific_entry"]["pid"]
  and wait["actual_worker_scientific_entry"]["input_ref"]==p["input"]
  and done["native_actual_wait_ref"]==p["native_wait"]
  and native_wait["schema"]=="C3_native_FLARE_actual_wait_v1"
  and type(native_wait["returncode"])is int and native_wait["returncode"]==0
  and native_wait["actual_direct_native_wait"]is True and native_wait["owned_process_group_ESRCH"]is True
  and native_wait["operational_failure"]is None,"Actual native completion, original child wait0 and same worker/input")
 spawn=read(p["direct_spawn"]);direct=read(p["direct_supervisor_wait"])
 admission=read(p["admission_receipt"]);admission_invocation=read(p["admission_invocation"]);admission_wait=read(p["admission_direct_wait"])
 need(p["producer_driver"]==p["producer_worker"],"Actual native driver and embedded worker consolidate in same authenticated source")
 module=p["producer_driver"]["path"][:-3].replace("/",".")
 expected_argv=["setsid",str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",module,"--execute-ref"]
 argv=spawn["argv"]
 need(spawn.get("actual_Popen")is True and type(spawn.get("pid"))is int
  and spawn.get("start_new_session")is False and "proc_stat"not in spawn
  and argv[:-1]==expected_argv and json.loads(argv[-1])==p["input"]
  and admission.get("execution_argv")==argv and admission.get("admission_only")is True
  and admission.get("input_ref")==p["input"]and admission.get("consumer_binding_count")==63,
  "Genuine Popen/setsid argv and actual admitted input; original raw spawn schema retained")
 need(admission_invocation.get("ROOT_all_mutations_serialized")is True
  and admission_invocation.get("automatic_retry")is False
  and admission_invocation["argv"][:-1]==[str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",module,"--admit-ref"]
  and json.loads(admission_invocation["argv"][-1])==p["original_request"]
  and admission_wait.get("actual_direct_wait")is True and type(admission_wait.get("actual_returncode"))is int
  and admission_wait["actual_returncode"]==0 and admission_wait["pid"]==169566,
  "Genuine serialized exact-once admission invocation and direct wait0")
 admission_outer=read(p["admission_outer"]);issued=read(p["issued_once"])
 need(admission_outer["root_authorizer"]=="root_agent"and admission_outer["actual_terminal_consumed"]is True
  and type(admission_outer["actual_exit_code"])is int and admission_outer["actual_exit_code"]==0
  and admission_outer["actual_tool_session_id"]==14772 and admission_outer["input_ref"]==p["input"]
  and admission_outer["actual_direct_wait_ref"]==p["admission_direct_wait"]
  and admission_outer["consumer_binding_count"]==63
  and issued["root_authorizer"]=="root_agent"and issued["input_ref"]==p["input"]
  and issued["all_ROOT_mutations_serialized"]is True and issued["no_automatic_retry"]is True
  and issued["actual_once_execution_argv"]==argv
  and datetime.fromisoformat(admission_outer["utc"])<=datetime.fromisoformat(issued["utc"])<=datetime.fromisoformat(spawn["utc"]),
  "Genuine consumed admission14772 outer0 and exact issued-once execution argv")
 # This observation is after Popen but before setsid. Its group/session are not the final supervisor group.
 post=spawn["actual_post_Popen_stat_observation"]
 need(post.get("observed_after_Popen")is True and post.get("pid")==spawn["pid"]==175310
  and post.get("start_jiffies")==492310847 and post.get("pgrp")==post.get("session")==175309
  and post["pgrp"]!=wait["supervisor_process_group"]
  and direct.get("pid")==spawn["pid"]==wait["supervisor_pid"]==175310
  and wait["supervisor_process_group"]==175310
  and direct.get("actual_direct_wait")is True and type(direct.get("actual_returncode"))is int
  and direct["actual_returncode"]==0
  and datetime.fromisoformat(spawn["utc"])<=datetime.fromisoformat(direct["utc"])<=datetime.fromisoformat(outer["utc"]),
  "Genuine supervisor PID/start and pre-setsid175309 group preserved separately from final175310 group in worker wait")
 launch_rows=launch["ownership"]["owned_identities"]
 need(launch["ownership"]["signals_use_pidfd_only"]is True and launch["ownership"]["no_numeric_group_signals"]is True
  and launch_rows==[dict(pid=176516,start=492327635)]
  and launch["pid"]==176516
  and {r["pid"]for r in launch_rows}<=set(wait["known_owned_pids"]),"Actual producer launch ownership, no fabricated identity format")
 groups=wait["group_observations"]
 need(groups and all(r["absent"]is True for r in groups)
  and set(wait["known_owned_process_groups"])=={r["pgid"]for r in groups},"Actual all-recorded-group drain")
 pids=sorted(set(wait["known_owned_pids"]+[wait["supervisor_pid"],launch["pid"]]))
 group_ids=sorted(set(wait["known_owned_process_groups"]+[wait["supervisor_process_group"]]))
 need(pids==group_ids==[175310,176516,178698,181613]
  and 1<=len(pids)<=256 and len(pids)==len(set(pids))and all(type(n)is int and n>=2 for n in pids)
  and all(type(n)is int and n>=2 for n in group_ids),"Exact bounded actual worker/supervisor PID and group roster")
 need(inp["refs"]["storage"]==p["storage"]and storage["storage_policy_selector_ref"]==q["storage_policy_selector_ref"]==p["selector"]
  and storage["completed_overlay_policy_ref"]==p["policy"]and storage["budget_components"]["consumer_adapter"]==p["guard"]
  and storage["budget_components"]["implementation"]==p["implementation"]and storage["budget_components"]["launcher"]==p["launcher"]
  and storage["accounting_mode"]=="Census_observation_two_scope_retention_v1","Exact current child original storage lineage")
 need(own["key"]==p["reservation_key"]and own["reserved_bytes"]==storage["reserved_new_bytes"]==2003235369
  and own["output"]==str(local(inp["output_directory"]))and own["auxiliary_scopes"]==[str(local(inp["packet_directory"]))]
  and q["output_directory"]==inp["output_directory"]and q["packet_directory"]==inp["packet_directory"]
  and len(own["consumer_bindings"])==63,"Entire original immutable reservation and both scopes preserved")
 return inp,wait,done,launch,storage,own,pids,group_ids

def current(value):
 now=datetime.now(timezone.utc);a=datetime.fromisoformat(value["created_at_utc"]);b=datetime.fromisoformat(value["expires_at_utc"])
 need(a.tzinfo and b.tzinfo and a<=now<b and 0<(b-a).total_seconds()<=3600,"Fresh finite ROOT one-shot close authority")
def request_value(p,own,launch,observation_ref):
 return dict(key=own["key"],reservation_sha256=p["reservation_sha256"],storage_ref=p["storage"],storage_locator=[],
  launch_ref=p["launch"],launch_pid_locator=["pid"],launch_identity=dict(pid=launch["pid"],input_ref=p["input"]),
  direct_wait_ref=p["wait"],wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
  wait_identity=dict(pid=launch["pid"],input_ref=p["input"],returncode=0,actual_OS_wait_status=0,
   actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,completed_ref=p["completed"]),
  completed_ref=p["completed"],completed_identity=dict(status="completed_C3_native_FLARE",parent_id="american_c3_val_004"),
  observation_ref=observation_ref)
def close_once(authorization_ref):
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0,"Fresh ROOT -B -s nonoptimized interpreter")
 p=pins();m=read(ref(MANIFEST))
 for r in m["source_refs"]:raw(r)
 auth=read(authorization_ref);current(auth);review=read(auth["source_review_ref"])
 need(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"and review.get("blockers")==[]
  and review.get("source_ref")==ref(SOURCE)and review.get("source_manifest_ref")==ref(MANIFEST),"Fresh independent exact close source pass")
 need(auth.get("schema")=="ROOT_FLARE_VAL004_native_current_Census_ordinary_close_once_authorization_v1"
  and auth.get("root_authorizer")=="root_agent"and auth.get("active")is True and auth.get("once_authorized")is True
  and auth.get("operation")=="close_completed"and auth.get("source_manifest_ref")==ref(MANIFEST)
  and auth.get("input_ref")==p["input"]and auth.get("direct_wait_ref")==p["wait"]and auth.get("completed_ref")==p["completed"]
  and auth.get("actual_outer_terminal_ref")==p["outer"]and auth.get("reservation_key")==p["reservation_key"]
  and auth.get("reservation_sha256")==p["reservation_sha256"]and auth.get("storage_policy_selector_ref")==p["selector"]
  and auth.get("ROOT_has_serialized_all_mutations_and_resolved_prior_inflight_actions")is True
  and auth.get("preserve_all_files_and_original_reservations")is True and auth.get("no_retry_retirement_or_new_quota_framework")is True,"Exact ROOT ordinary close once authority")
 inp,wait,done,launch,storage,own,pids,group_ids=metadata_join(p)
 rev=read(auth["permanent_route_revocation_ref"])
 need(rev.get("schema")=="ROOT_FLARE_VAL004_completed_native_permanent_route_revocation_v1"
  and rev.get("root_authorizer")=="root_agent"and rev.get("active")is True
  and rev.get("input_ref")==p["input"]and rev.get("direct_wait_ref")==p["wait"]and rev.get("completed_ref")==p["completed"]
  and rev.get("actual_outer_terminal_ref")==p["outer"]and rev.get("reservation_key")==own["key"]
  and rev.get("reservation_sha256")==p["reservation_sha256"]and rev.get("output_directory")==inp["output_directory"]
  and rev.get("packet_directory")==inp["packet_directory"]and rev.get("all_original_launch_resume_retry_write_routes_permanently_revoked")is True
  and rev.get("preserve_all_files_original_clocks_failures_and_reservations")is True,"Separate exact permanent ROOT original-route revocation")
 stamp=datetime.fromisoformat(rev["revoked_at_utc"]);need(stamp.tzinfo and stamp<=datetime.now(timezone.utc),"Actual nonfuture permanent revocation time")
 directory=local(auth["run_directory"]);need(str(directory.relative_to(ROOT)).startswith("research/reviews/FLARE_VAL004_native_ROOT_close_Census_"),"Fresh ROOT close administrative scope")
 directory.mkdir() # Exclusive: consumes this invocation before guard import, state or probes. Never reuse after uncertainty.
 save(directory,"invocation.json",dict(authorization_ref=authorization_ref,source_manifest_ref=ref(MANIFEST),automatic_retry=False))
 raw(p["guard"]);sp=importlib.util.spec_from_file_location("_ROOT_FLARE_VAL004_native_exact_current_Census_close",local(p["guard"]["path"]))
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
