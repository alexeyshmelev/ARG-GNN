"""Root-only close of original VAL000 native_paths16GB scope; default bounded metadata only."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse,fcntl,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/VAL000_native_paths_completed_close_20261015_v1"
SOURCE=HERE+"/close_native_paths_v1.py"
REFS_REF={'path': 'research/reviews/VAL000_native_paths_completed_close_20261015_v1/actual_refs.json', 'sha256': '0cebdbe4aa1834927cd9578de0f7db17d133186a4f665ca6aaf4a98e9755fa41'}
SCOPES=("VAL000_native_paths",)
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
    need(scope in SCOPES,"Only original completed VAL000 native_paths reservation")
    manifest=read(REFS_REF);components=manifest["components"];row=manifest["target"]
    need(all(ref(r["path"])==r for r in components.values()),"Exact current980 guard/policy/repair/core/launcher")
    need(ref(manifest["basis_helper_ref"]["path"])==manifest["basis_helper_ref"],"Frozen accepted close execution pattern")
    for name,r in row.items():
        if name.endswith("_ref"):need(ref(r["path"])==r,"Every actual source/JSON evidence pin")
    spec=read(row["admission_ref"]);storage=read(row["storage_ref"])
    for key in row["storage_locator"]:storage=storage[key]
    own=storage["reservation"];cfg=read(row["config_ref"])
    launch=read(row["launch_ref"]);wait=read(row["direct_wait_ref"])
    minimal=read(row["minimal_wait_ref"]);completed=read(row["completed_ref"])
    execution=read(row["execution_authority_ref"]);reserved=read(row["reserve_receipt_ref"])
    need(spec["stage"]==row["stage"]=="native_paths" and spec["parent_id"]=="american_c3_val_000"
        and spec["split"]=="val" and spec["GPU_authorized"] is False
        and spec["GNN_optimization_authorized"] is False
        and spec["query_ancestry_truth_payload_access_authorized"] is False
        and spec["source_ref"]==row["producer_ref"] and spec["driver_ref"]==row["admission_driver_ref"]
        and spec["config_ref"]==row["config_ref"] and spec["storage"]==storage
        and spec["job_id"]==storage["job_id"]=="American-C3-val000-full30mb-native_paths-shared-980-v1"
        and own["key"]==row["reservation_key"]=="H041/American-C3-val000-full30mb-native_paths-shared-980-v1",
        "Exact original same-parent CPU-only native_paths admission")
    need(storage["accounting_mode"]=="prospective_980gb_v1"
        and storage["completed_overlay_policy_ref"]==spec["completed_overlay_policy_ref"]==components["policy"]
        and storage["budget_components"]["consumer_adapter"]==spec["storage_adapter_ref"]==components["guard"]
        and storage["reserved_new_bytes"]==own["reserved_bytes"]==cfg["output_cap_bytes"]==row["cap"]==16000000000
        and own["output"]==str(ROOT/storage["output_directory"])
        and row["completed_ref"]["path"]==storage["output_directory"]+"/completed.json"
        and reserved["value"]==own and reserved["initial_ledger_pins"]==storage["initial_ledger_pins"],
        "Exact original16GB reservation/storage/output and genuine reserve receipt")
    need(wait["actual_direct_worker_wait"] is True and wait["returncode"]==0
        and wait["owned_worker_group_ESRCH"] is True and wait["operational_failure"] is None
        and wait["parent_id"]=="american_c3_val_000" and wait["stage"]=="native_paths"
        and wait["admission_ref"]==launch["admission_ref"]==row["admission_ref"]
        and wait["completed_ref"]==row["completed_ref"] and wait["launch_ref"]==row["launch_ref"]
        and wait["minimal_wait_ref"]==row["minimal_wait_ref"],
        "Genuine original native_paths direct worker wait0/drain and completed lineage")
    need(minimal["actual_direct_worker_wait"] is True and minimal["returncode"]==0
        and minimal["stage"]=="native_paths" and minimal["admission_ref"]==row["admission_ref"]
        and minimal["pid"]==minimal["process_group"]==wait["pid"]==launch["pid"]
            ==launch["worker_pid"]==launch["process_group"]==row["worker_pid"]
        and launch["supervisor_pid"]==row["supervisor_pid"]
        and launch["IO_supervisor_ref"]==wait["root_IO_source_ref"]==row["IO_source_ref"],
        "Actual direct Popen/minimal wait and exact original source-owned worker/supervisor")
    need(execution["active"] is True and execution["root_authorizer"]=="root_agent"
        and execution["actual_IO_supervisor_ref"]==row["IO_source_ref"]
        and execution["actual_admission_ref"]==row["admission_ref"]
        and execution["producer_ref"]==row["producer_ref"]
        and execution["existing_owned_RSS_wait_helper_ref"]==row["owned_wait_helper_ref"]
        and execution["CPU_only"] is True and execution["storage_unchanged"] is True
        and execution["original_scientific_argv_and_config_unchanged"] is True
        and execution["new_reservation_or_cap_change"] is False,
        "Preserved actual root execution-only supervisor authority")
    need(completed["status"]=="completed_native_C3_path_preparation"
        and completed["parent_id"]=="american_c3_val_000"
        and completed["observation_ref"]==cfg["observation_ref"]==row["observation_ref"]
        and completed["source_author_function_ref"]==row["author_source_ref"]
        and completed["query_ancestry_truth_opened"] is False
        and completed["contrastive_labels_reference_only"] is True
        and completed["query_embedding_view_prespecified_focals_only"] is True
        and completed["query_focal_count"]==cfg["query_focal_count"]==10000
        and completed["class_order"]==["EUR","EAS","AFR"]
        and completed["native_split_seed"]==42 and completed["native_train_fraction"]==.9
        and completed["author_Table1_exact_reproduction"] is False
        and all(0<completed[k]<=cfg[v] for k,v in
            (("unique_paths","max_unique_paths"),("path_tokens","max_path_tokens"),
             ("path_occurrences","max_path_occurrences"),("vocabulary_nodes","max_vocabulary_nodes"))),
        "Actual complete unchanged native path/focal preparation; no accuracy claim")
    obs=read(row["observation_ref"])
    need(obs["parent_id"]=="american_c3_val_000" and obs["split"]=="val"
        and obs["sequence_length_bp"]==30000000
        and obs["counts_haploid"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
        and obs["genealogy_kind"]==completed["genealogy_kind"]==cfg["genealogy_kind"]
        and obs["query_ancestry_truth_in_inputs"] is False,
        "Original full30Mb1600-haplotype pre-truth inferred observation")
    pids={launch["supervisor_pid"],wait["pid"],*[int(x) for x in wait["observed_owned_pid_identities"]]}
    groups={launch["process_group"]}
    need(sorted(pids)==row["known_source_owned_roster"] and len(pids)<=256
        and all(type(x) is int and x>=2 for x in pids|groups),
        "Bounded exact saved source-owned supervisor/worker roster; no unrelated audit")
    li={k:launch[k] for k in ("pid","worker_pid","process_group","supervisor_pid","admission_ref","IO_supervisor_ref")}
    wi={k:wait[k] for k in ("actual_direct_worker_wait","returncode","owned_worker_group_ESRCH",
        "operational_failure","parent_id","stage","pid","admission_ref","completed_ref","minimal_wait_ref","root_IO_source_ref")}
    ci={k:completed[k] for k in ("status","parent_id","observation_ref","source_author_function_ref",
        "query_ancestry_truth_opened","contrastive_labels_reference_only","query_focal_count","unique_paths","path_tokens","path_occurrences")}
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
    need(m.MODE=="prospective_980gb_v1" and all(getattr(m.budget.v4,n).__module__==m.budget.v4.__name__
        for n in ("canonical","file_bytes","mutex")),"Exact actual980 original unmemoized close API")
    folder=local(args.output_packet or HERE+"/closed_"+args.scope+"_v1");need(not folder.exists(),"Fresh close packet; preserve prior attempts")
    # Cooperate with the live wave admission lock without touching its source/configs.
    lock=ROOT/"research/reviews/American_C3_short_train2mb_wave_source_v1/scheduler/admission.lock"
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
    p.add_argument("--scope",choices=SCOPES);p.add_argument("--output-packet");a=p.parse_args()
    if a.execute:return execute(a)
    scopes=[a.scope] if a.scope else SCOPES
    rows=[dict(scope=s,key=collect(s)[4]["key"],actual_completed_metadata_validated=True) for s in scopes]
    print(json.dumps(dict(status="source_only_metadata_review_no_guard_PID_probe_or_close",scopes=rows),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
