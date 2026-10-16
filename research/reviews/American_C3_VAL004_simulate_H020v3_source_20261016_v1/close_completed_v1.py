"""Root-only completed phase close; default bounded actual metadata only."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse,fcntl,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.reviews.American_C3_VAL004_simulate_H020v3_source_20261016_v1 import root_packet_v1 as packet
packet.install_renewal_v1()
HERE=packet.REL
SOURCE=HERE+"/close_completed_v1.py"
SCOPES=packet.STAGES
def need(ok,msg):
    if not ok:raise ValueError(msg)

def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT) and p!=ROOT,"Canonical project path")
    return p

def ref(value):
    p=local(value);need(p.suffix in (".py",".json") and p.stat().st_size<=262144,"Source/JSON only; no payload hashing")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(r):
    p=local(r["path"]);need(p.suffix==".json" and p.stat().st_size<=131072 and ref(p)==r,"Exact bounded actual JSON")
    return json.loads(p.read_text())

def save(folder,name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(len(raw)<=65536 and Path(name).name==name,"Compact flat root close certificate")
    fd=os.open(folder/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return ref(folder/name)

def absence(pid,group=False):
    need(type(pid) is int and pid>=2,"Exact known original PID/group")
    try:
        if group:os.killpg(pid,0)
        else:os.kill(pid,0)
    except ProcessLookupError:
        return dict(**({"process_group":pid} if group else {"pid":pid}),absent=True,
            method="targeted os.killpg(group,0) ESRCH" if group else "targeted os.kill(pid,0) ESRCH",
            checked_utc=datetime.now(timezone.utc).isoformat())
    raise ValueError("Original PID/group present, reused, zombie or inaccessible")

def collect(scope):
    need(scope in SCOPES,"One original remaining-VAL completed preparation stage")
    pins=packet.frozen();folder,output,job=packet.identity(scope)
    admission_ref=ref(folder/"admission.json");spec=read(admission_ref)
    row=dict(stage=scope,cap=dict(simulate=6500000000,infer=12500000000,native_paths=16000000000)[scope],
        admission_ref=admission_ref,storage_ref=admission_ref,storage_locator=["storage"],
        config_ref=spec["config_ref"],launch_ref=ref(folder/"launch.json"),
        direct_wait_ref=ref(folder/"actual_wait.json"),minimal_wait_ref=ref(folder/"actual_wait_minimal.json"),
        completed_ref=ref(output+"/completed.json"),IO_source_ref=ref(packet.HERE/"root_execution_v1.py"),
        producer_ref=pins[packet.PRODUCERS[scope]],execution_authority_ref=ref(folder/"root_execution_authority.json"),
        reserve_receipt_ref=ref(folder/"reserve_receipt.json"))
    components={k:pins[p] for k,p in (("guard",packet.OVERLAY),("policy",packet.POLICY),
        ("repair",packet.REPAIR),("budget",packet.BUDGET),("launcher",packet.LAUNCHER))}
    manifest=dict(components=components,target=row)
    for r in [*components.values(),*[v for k,v in row.items() if k.endswith("_ref")]]:
        need(ref(r["path"])==r,"Actual pinned source/JSON only")
    storage=spec["storage"];own=storage["reservation"];cfg=read(row["config_ref"])
    launch=read(row["launch_ref"]);wait=read(row["direct_wait_ref"])
    minimal=read(row["minimal_wait_ref"]);completed=read(row["completed_ref"])
    reserved=read(row["reserve_receipt_ref"]);authority=read(row["execution_authority_ref"])
    need(spec["driver_ref"]==ref(packet.SOURCE) and spec["IO_supervisor_ref"]==row["IO_source_ref"]
        and spec["source_ref"]==row["producer_ref"] and spec["parent_id"]==packet.PARENT
        and spec["parent_seed_row"]==packet.SEED_ROW and spec["stage"]==scope and spec["split"]=="val"
        and spec["GPU_authorized"] is False and spec["GNN_optimization_authorized"] is False
        and spec["query_ancestry_truth_payload_access_authorized"] is False,
        "Exact separately admitted original remaining-VAL science and CPU route")
    need(own["key"]=="H041/"+job and spec["job_id"]==storage["job_id"]==job
        and own["output"]==str(ROOT/output) and spec["output_directory"]==storage["output_directory"]==output
        and storage["reserved_new_bytes"]==own["reserved_bytes"]==cfg["output_cap_bytes"]==row["cap"]
        and storage["accounting_mode"]=="prospective_980gb_H020_v3"
        and storage["completed_overlay_policy_ref"]==spec["completed_overlay_policy_ref"]==components["policy"]
        and storage["budget_components"]["consumer_adapter"]==spec["storage_adapter_ref"]==components["guard"]
        and reserved["value"]==own and reserved["initial_ledger_pins"]==storage["initial_ledger_pins"],
        "Original exact stage cap/storage/reservation; no new accounting")
    need(wait["actual_direct_worker_wait"] is True and wait["returncode"]==0
        and wait["owned_worker_group_ESRCH"] is True and wait["operational_failure"] is None
        and wait["parent_id"]==packet.PARENT and wait["stage"]==scope
        and wait["admission_ref"]==launch["admission_ref"]==row["admission_ref"]
        and wait["completed_ref"]==row["completed_ref"] and wait["launch_ref"]==row["launch_ref"]
        and wait["minimal_wait_ref"]==row["minimal_wait_ref"],"Genuine same-parent direct worker wait0/drain")
    need(minimal["actual_direct_worker_wait"] is True and minimal["returncode"]==0
        and minimal["stage"]==scope and minimal["admission_ref"]==row["admission_ref"]
        and minimal["pid"]==minimal["process_group"]==wait["pid"]==launch["pid"]==launch["worker_pid"]==launch["process_group"]
        and launch["IO_supervisor_ref"]==wait["root_IO_source_ref"]==row["IO_source_ref"],
        "Actual direct/minimal worker wait and separately admitted supervisor source")
    need(authority["active"] is True and authority["root_authorizer"]=="root_agent"
        and authority["actual_IO_supervisor_ref"]==row["IO_source_ref"]
        and authority["actual_admission_ref"]==row["admission_ref"] and authority["producer_ref"]==row["producer_ref"]
        and authority["CPU_only"] is True and authority["storage_unchanged"] is True
        and authority["new_reservation_or_cap_change"] is False
        and authority["scientific_module_arguments_and_config_unchanged"] is True
        and authority["operational_memo_bootstrap_added"] is True,"Original-science operational execution authority")
    status=dict(simulate="completed_C3_simulation_reconstruction",infer="completed_current_C3_inference",
        native_paths="completed_native_C3_path_preparation")[scope]
    need(completed["status"]==status,"Actual unchanged stage completion")
    obs=read(completed["observation_ref"])
    need(obs["parent_id"]==packet.PARENT and obs["split"]=="val" and obs["sequence_length_bp"]==30000000
        and obs["counts_haploid"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
        and obs["query_ancestry_truth_in_inputs"] is False,"Same full30Mb1600-haplotype observed parent")
    if scope=="simulate":
        need(completed["parent_id"]==packet.PARENT and completed["sample_unit"]=="haploid"
            and cfg["parent_id"]==packet.PARENT and cfg["ancestry_seed"]==packet.SEED_ROW["ancestry_seed"]
            and cfg["mutation_seed"]==packet.SEED_ROW["mutation_seed"],"Actual original registry simulation")
        ci={k:completed[k] for k in ("status","parent_id","sample_unit","observation_ref")}
    elif scope=="infer":
        need(completed["tsinfer_num_threads"]==cfg["num_threads"]==8
            and completed["unary_preprocessing_policy"]==cfg["unary_preprocessing_policy"],
            "Unchanged current tsinfer/tsdate and explicit unary preprocessing")
        ci={k:completed[k] for k in ("status","tsinfer_num_threads","unary_preprocessing_policy","observation_ref")}
    else:
        need(completed["parent_id"]==packet.PARENT and completed["observation_ref"]==cfg["observation_ref"]
            and completed["query_ancestry_truth_opened"] is False and completed["contrastive_labels_reference_only"] is True
            and completed["query_embedding_view_prespecified_focals_only"] is True
            and completed["query_focal_count"]==cfg["query_focal_count"]==10000
            and cfg["path_sampling_seed"]==packet.SEED_ROW["path_sampling_seed"]
            and completed["native_split_seed"]==42 and completed["native_train_fraction"]==.9,
            "Original native full-path preparation and common10k label-blind focals")
        ci={k:completed[k] for k in ("status","parent_id","observation_ref","query_ancestry_truth_opened",
            "query_focal_count","native_split_seed","native_train_fraction")}
    pids={launch["supervisor_pid"],wait["pid"],*[int(x) for x in wait["observed_owned_pid_identities"]]}
    groups={launch["process_group"]}
    need(len(pids)<=256 and all(type(x) is int and x>=2 for x in pids|groups),
         "Bounded saved exact source-owned supervisor/worker roster only")
    li={k:launch[k] for k in ("pid","worker_pid","process_group","supervisor_pid","admission_ref","IO_supervisor_ref")}
    wi={k:wait[k] for k in ("actual_direct_worker_wait","returncode","owned_worker_group_ESRCH",
        "operational_failure","parent_id","stage","pid","admission_ref","completed_ref","minimal_wait_ref","root_IO_source_ref")}
    return manifest,row,spec,storage,own,launch,wait,completed,sorted(pids),sorted(groups),"pid",li,wi,ci

def execute(args):
    need(Path.cwd().resolve()==ROOT,"Root repository execution")
    manifest,row,spec,storage,own,launch,wait,completed,pids,groups,launch_pid,li,wi,ci=collect(args.scope)
    # Exact known source-owned roots and groups must be absent, including the outer supervisor.
    observations=[absence(x) for x in pids];group_observations=[absence(x,True) for x in groups]
    components=manifest["components"];sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    ms=importlib.util.spec_from_file_location("_VAL_FLARE_root980_close",ROOT/components["guard"]["path"])
    m=importlib.util.module_from_spec(ms);ms.loader.exec_module(m)
    m.configure_v2(components["repair"],policy_ref=components["policy"])
    need(m.MODE=="prospective_980gb_H020_v3" and all(getattr(m.budget.v4,n).__module__==m.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Exact actual980 original unmemoized close API")
    folder=local(args.output_packet or HERE+"/closed_"+args.scope+"_v1");need(not folder.exists(),"Fresh close packet; preserve prior attempts")
    # Cooperate with the live wave admission lock without touching its source/configs.
    lock=packet.ADMISSION_LOCK
    with lock.open("a+") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        with m.budget.v4.mutex(ROOT):
            state=m._state_locked(components["budget"]["sha256"],components["launcher"]["sha256"],fresh=True)
            m.assert_open(own["key"],state[0],state[1]);need(state[0][own["key"]]==own,"Exact still-open original reservation")
        folder.mkdir()
        revoke=save(folder,"root_scope_revocation.json",dict(root_authorizer="root_agent",key=own["key"],
            all_original_input_config_storage_source_launch_resume_retry_write_capabilities_revoked=True,
            read_only_reuse_preserved=True,preserve_all_files=True,all_known_source_owned_root_worker_pids=pids,
            PID_observations=observations,group_observations=group_observations,
            actual_evidence_refs=row,root_close_helper_ref=ref(SOURCE),no_outer_remote_direct_Popen_wait_claim=True,
            simulation_unseeded_RSS_zero_is_invalid=bool(row.get("RSS_correction_ref"))))
        observation=save(folder,"root_observation.json",dict(schema="concurrent_completed_scope_targeted_root_observation_v1",
            root_authorizer="root_agent",key=own["key"],reservation_sha256=m.budget.digest(own),
            storage_ref=row["storage_ref"],launch_ref=row["launch_ref"],direct_wait_ref=row["direct_wait_ref"],
            completed_ref=row["completed_ref"],launcher_pid=launch[launch_pid],
            all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
            completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
            external_NFS_writers_outside_observation=True,pids=pids,observations=[absence(x) for x in pids],
            worker_group_observations=[absence(x,True) for x in groups],scope_revocation_ref=revoke,
            actual_evidence_refs=row,root_close_helper_ref=ref(SOURCE)))
        request=dict(key=own["key"],reservation_sha256=m.budget.digest(own),storage_ref=row["storage_ref"],
            storage_locator=row["storage_locator"],launch_ref=row["launch_ref"],launch_pid_locator=[launch_pid],
            launch_identity=li,direct_wait_ref=row["direct_wait_ref"],wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
            wait_identity=wi,completed_ref=row["completed_ref"],completed_identity=ci,observation_ref=observation)
        with m.budget.v4.mutex(ROOT):
            state=m._state_locked(components["budget"]["sha256"],components["launcher"]["sha256"],fresh=True)
            m.assert_open(own["key"],state[0],state[1]);need(state[0][own["key"]]==own,"Reservation unchanged before close")
            authority=m.operation_draft_v1("close_completed",request,expected_before=m._pins(state[2]),bindings=state[3])
        authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
        receipt=m.apply_close_completed_v1(save(folder,"close_authority.json",authority),
            expected_source_sha256=components["budget"]["sha256"],expected_launcher_sha256=components["launcher"]["sha256"])
        rr=save(folder,"close_receipt.json",receipt)
        print(json.dumps(dict(closed_key=own["key"],receipt_ref=rr,retained_bytes=receipt["retained_bytes"],
            credit_bytes=receipt["credit_bytes"],projected_bytes=receipt["projected_bytes"],files_deleted=0)),flush=True)
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true")
    p.add_argument("--parent",choices=packet.PARENTS,required=True)
    p.add_argument("--scope",choices=SCOPES,required=True);p.add_argument("--output-packet")
    a=p.parse_args();packet.select_parent(a.parent)
    a.output_packet=a.output_packet or str(packet.close_folder(a.scope))
    if a.execute:return execute(a)
    row=collect(a.scope)
    print(json.dumps(dict(status="source_only_actual_completed_metadata_no_PID_guard_or_close",
        parent_id=packet.PARENT,scope=a.scope,key=row[4]["key"]),sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
