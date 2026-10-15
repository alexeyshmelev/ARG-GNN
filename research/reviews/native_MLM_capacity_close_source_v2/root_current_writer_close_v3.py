"""Root-owned current-writer observations and unchanged completed-scope closures."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"research/reviews/native_MLM_capacity_close_source_v2"
def ref(path):
    path=Path(path);return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def stamp():return datetime.now(timezone.utc).isoformat()
def save(name,obj):
    p=BASE/name
    with p.open("x") as f:json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    return ref(p)
def load_adapter():
    p=BASE/"root_close_adapter_v3.py"
    assert ref(p)["sha256"]=="217ad30e934efa058c96789450abb7de55f3ead2ada44b657bbf734530956f4a"
    spec=importlib.util.spec_from_file_location("_root_capacity_close_adapter",p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def snapshot(item):
    entries=[];excluded=[];own=os.getpid()
    targets=[item["original_reservation"]["output"],item["candidate"]["key"]]
    targets += [r["path"] for r in item["revocation_target_refs"]]
    targets += list(item["original_reservation"].get("auxiliary_scopes",[]))
    groups=set(item["known_process_group_ids"])
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            args=[a.decode(errors="replace") for a in (proc/"cmdline").read_bytes().split(b"\0") if a]
            row=(proc/"stat").read_text().rsplit(")",1)[1].split()
            state=dict(pid=int(proc.name),ppid=int(row[1]),pgid=int(row[2]),start_ticks=int(row[19]))
            try:cwd=os.readlink(proc/"cwd")
            except PermissionError:
                assert args and args[0]=="ssh-agent" and args[1:] in (["-s"],["-a",str(ROOT/".runtime/ssh/agent.sock"),"-s"]),("unresolved denied process",state,args)
                excluded.append(dict(**state,reason="ssh-agent, immutable signing/socket service rather than experiment writer"))
                continue
            except FileNotFoundError:continue
            candidate=(state["pgid"] in groups or any(t in a for t in targets for a in args))
            project=(cwd==str(ROOT) or cwd.startswith(str(ROOT)+"/") or any(str(ROOT) in a for a in args))
            if not (candidate or project):continue
            assert not candidate,("live completed-scope capability",state,args)
            if state["pid"]==own:
                reason="current root observer, no historical scope write capability"
            elif state["pgid"]==1844431 and state["pid"] in (1844431,1844434,1844436):
                assert args and ((state["pid"]==1844431 and args[:2]==["zsh","-c"] and "grep -Rnl" in args[2] and "signed_difference" in args[2]) or
                    (state["pid"]==1844434 and args[0]=="grep") or (state["pid"]==1844436 and args==["head","-28"]))
                reason="source-identified stale text search, read-only grep/head"
            elif state["pgid"]==2060822 and state["pid"] in (2060822,2060998,2060999,2061000):
                assert (state["pid"]==2060822 and args==[".runtime/env/bin/python","-B","research/reviews/mutation_routed_pyg_dataset_v1/handmade_spawn_check.py"]) or (
                    state["ppid"]==2060822 and len(args)>=5 and any(x.startswith("from multiprocessing.") for x in args))
                reason="unrelated synthetic dataset check and support children, source limits writes to its handmade/receipt scopes"
            else:
                raise ValueError("Unclassified live project process: "+repr((state,args)))
            entries.append(dict(**state,cwd=cwd,argv=args,classification=reason))
        except (ProcessLookupError,FileNotFoundError):continue
    return dict(checked_utc=stamp(),targeting_rules="same-uid /proc cmdline,cwd,group matching exact completed output/aux/key/revocation source paths; project-cwd processes separately classified",
        payload_or_environment_reads=False,scope_files_inventoried=False,project_processes=entries,
        denied_cwd_exclusions=excluded,matched_live_completed_scope_writers=[])
def absent(values,group=False):
    rows=[]
    for number in values:
        try:(os.killpg if group else os.kill)(number,0)
        except ProcessLookupError:pass
        else:raise ValueError("Owned identity or reused number remains present")
        rows.append(dict(**{("pgid" if group else "pid"):number},absent=True,
            method="targeted os.killpg(pgid,0) ESRCH" if group else "targeted os.kill(pid,0) ESRCH",checked_utc=stamp()))
    return rows
def main(tag):
    m=load_adapter();audit,policy=m.audit_v1();item=audit["candidates"][tag]
    m.original_evidence_v1(item)
    current=snapshot(item)
    resolution=save(tag+".root_current_writer_resolution_v2.json",dict(
        schema="root_completed_scope_current_writer_resolution_v1",root_authorizer="root_agent",key=item["candidate"]["key"],
        roster_reconstruction_complete=True,current_cooperative_writer_roster_complete=True,
        historical_process_tree_reconstructed=False,historical_group_ids_unrecorded=item.get('historical_group_ids_unrecorded',False),historical_unrecorded_helper_and_support_PIDs_remain_unknown=True,
        historical_unknown_PIDs_not_fabricated=True,unresolved_roster_items=[],
        known_historical_PIDs=item["known_pids"],source_backed_groups=item["known_process_group_ids"],
        actual_current_process_observation=current,wait_provenance=item["wait_provenance"],
        source_audit_ref=m.AUDIT_REF,synthetic_unrelated_source_ref=ref(ROOT/"research/reviews/mutation_routed_pyg_dataset_v1/handmade_spawn_check.py"),
        source_reasoning="Actual original wait evidence is authenticated. Bound post-wait helper paths contain no completed-scope writes or relaunch. Recorded groups, if any, are absent. Unrecorded inherited historical groups remain unknown. No omitted current scope-capable writer survives in the bounded project process observation. Extinct unrecorded helper identities remain unknown.",
        revoked_at_utc=stamp(),revoked_original_entrypoint_refs=item["revocation_target_refs"],
        revocation="ROOT permanently revokes every old/original/resume/write capability for these exact original output/auxiliary scopes. All files are retained. Future launches on these scopes are prohibited by the unchanged close primitive and ROOT scheduling.",
        preserve_all_files=True,changed_historical_files=False))
    obs=json.loads((BASE/(tag+".observation.inactive.json")).read_text())
    obs["root_close_adapter_ref"]=m.source_ref_v1()
    for k in ("active","all_root_owned_launcher_supervisor_worker_pids_listed","completed_scope_all_old_original_resume_write_capabilities_revoked","direct_wait_observed","roster_reconstruction_complete"):obs[k]=True
    obs.update(historical_group_ids_unrecorded=item.get("historical_group_ids_unrecorded",False),unresolved_roster_items=[],roster_resolution_refs=[resolution],revoked_original_entrypoint_refs=item["revocation_target_refs"],
        observations=absent(obs["pids"]),process_group_observations=absent(obs["process_group_ids"],True))
    observation_ref=save(tag+".observation.root_v2.json",obs)
    draft_result=m.root_operation_v1(tag,observation_ref,output_path=str((BASE/(tag+".operation.inactive.root_v2.json")).relative_to(ROOT)))
    draft=json.loads((ROOT/draft_result["operation_ref"]["path"]).read_text())
    assert draft["operation"]=="close_completed" and draft["active"] is False and draft["request"]["key"]==item["candidate"]["key"]
    draft.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(minutes=10)).isoformat())
    auth_ref=save(tag+".operation.active.root_v2.json",draft)
    result=m.root_operation_v1(tag,observation_ref,output_path=str((BASE/(tag+".close_receipt.root_v2.json")).relative_to(ROOT)),authority_ref=auth_ref)
    receipt=json.loads((ROOT/result["receipt_ref"]["path"]).read_text())["receipt"]
    print(json.dumps(dict(candidate=tag,result=result,retained_bytes=receipt["retained_bytes"],credit_bytes=receipt["credit_bytes"],projected_bytes=receipt["projected_bytes"])),flush=True)
if __name__=="__main__":main(sys.argv[1])
