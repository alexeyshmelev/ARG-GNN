"""Metadata-only TRAIN binding and inactive cap proposal; no activation/guard/science."""
from pathlib import Path
import argparse,datetime,hashlib,json,math
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=Path(__file__).resolve().parent
PACKET=HERE.parent
REGISTRY=ROOT/"research/reviews/American_C3_short_train2mb_wave_source_v1/parent_registry.inactive.json"
SOURCE_FILES_READ={}
GiB=2**30;MiB=2**20
TRAIN=tuple("american_c3_short_train_%03d"%i for i in range(20))

def require(ok,message):
    if not ok:raise ValueError(message)
def local(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    require(p.resolve()==p and ".." not in p.parts,"Canonical nonsymlink project path")
    p.relative_to(ROOT);return p
def file_ref(path):
    p=local(path)
    require(p.suffix in (".json",".py",".sh",".md"),"Metadata/source files only; never arrays/checkpoints")
    require(p.stat().st_size<=4*MiB,"Bounded metadata/source file")
    b=p.read_bytes();value=dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(b).hexdigest())
    SOURCE_FILES_READ[value["path"]]=value["sha256"];return value
def read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact metadata ref")
    require(local(record["path"]).suffix==".json" and file_ref(record["path"])==record,"Actual JSON ref/hash")
    return json.loads(local(record["path"]).read_text())
def get(path):return read(file_ref(path))
def save(folder,name,value):
    p=folder/name
    with p.open("x") as stream:stream.write(json.dumps(value,indent=2,sort_keys=True)+"\n")
    return file_ref(p)
def successful_wait(record,completion_ref):
    w=read(record)
    require(w["returncode"]==0 and w["completed_ref"]==completion_ref and w.get("operational_failure") is None
        and any(w.get(x) is True for x in ("actual_direct_worker_wait","actual_direct_launcher_wait"))
        and any(w.get(x) is True for x in ("owned_process_group_ESRCH","worker_process_group_drained","all_owned_processes_and_groups_drained")),"Genuine direct wait0 and drain")
    return w
def focal_wait_path(i,parent):
    if i==0:return ROOT/"research/reviews/C3_next_plan_root_handoff_20261015_v1"/("admitted_"+parent)/"root_retry2_actual_wait.json"
    if i<6:return ROOT/"research/reviews/C3_focal_plan_fanout_root_source_20261015_v1/run_v1"/parent/"execute/actual_wait.json"
    run="run006_015_v1" if i<16 else "run016_017_v1"
    return ROOT/"research/reviews/C3_focal_plan_continuation_memo2_20261015_v1"/run/parent/"execute/actual_wait.json"
def one_parent(i,row):
    parent=TRAIN[i];origin="American_C3_short_train2mb_wave_supervision_v2" if i<14 else "American_C3_short_train2mb_wave_980_source_v1"
    p=ROOT/"research/reviews"/origin/"admitted"/parent
    result=dict(parent_id=parent,actual_train_binding=None,parent_plan_ref=None,parent_plan_completion_ref=None,
        parent_plan_wait_ref=None,lineage_evidence=None,plan_metadata=None,pending=[])
    if i>=18:
        result["pending"]=["Final pair intentionally remains null pending genuine simulation/inference/focal-plan completion and wait evidence"]
        return result
    sim_path=ROOT/"experiments/results/American_C3_short_train2mb_wave_v2"/parent/"simulation/completed.json"
    infer_path=ROOT/"experiments/results/American_C3_short_train2mb_wave_v2"/parent/"inference/completed.json"
    paths=[sim_path,infer_path,p/"simulate/config.json",p/"simulate/actual_wait.json",p/"infer/actual_wait.json"]
    if not all(x.is_file() for x in paths):
        result["pending"]=["Missing genuine completed simulation/inference metadata"];return result
    sim_ref,inf_ref,cfg_ref,sw_ref,iw_ref=[file_ref(x) for x in paths]
    sim=read(sim_ref);inf=read(inf_ref);cfg=read(cfg_ref);template=read(row["simulation_config_ref"])
    successful_wait(sw_ref,sim_ref);successful_wait(iw_ref,inf_ref)
    science=("schema","species","model","parent_id","split","sample_unit","sample_counts","sequence_length_bp","mutation_rate","recombination_rate","admixture_generation","ancestry_seed","mutation_seed","class_order")
    require(all(cfg[x]==template[x] for x in science) and cfg["execution_enabled"] is True,"Unchanged actual short TRAIN science/seeds")
    require(sim["status"]=="completed_C3_simulation_reconstruction" and sim["parent_id"]==parent
        and sim["class_order"]==["EUR","EAS","AFR"] and sim["sample_unit"]=="haploid","Actual same-parent simulation")
    obs_ref=inf["observation_ref"];obs=read(obs_ref)
    require(inf["status"]=="completed_current_C3_inference" and obs["parent_id"]==parent and obs["split"]=="train"
        and obs["sequence_length_bp"]==2000000 and obs["counts_haploid"]=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000}
        and obs["class_order"]==["EUR","EAS","AFR"] and obs["sample_unit"]=="haploid"
        and obs["source_observation_ref"]==sim["observation_ref"] and obs["query_ancestry_truth_in_inputs"] is False
        and obs["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG","Actual inferred lineage, never true ARG")
    result["actual_train_binding"]=dict(simulation_completion_ref=sim_ref,simulation_config_ref=cfg_ref,inferred_observation_ref=obs_ref)
    result["lineage_evidence"]=dict(simulation_wait_ref=sw_ref,inference_completion_ref=inf_ref,inference_wait_ref=iw_ref,
        private_truth_payload_opened=False)
    done_path=ROOT/"experiments/results/C3_short_TRAIN_focal_plans_v1"/parent/"completed.json"
    wait_path=focal_wait_path(i,parent)
    if not done_path.is_file() or not wait_path.is_file():
        result["pending"]=["Focal/plan final completion or actual successful direct wait pending"];return result
    done_ref=file_ref(done_path);wait_ref=file_ref(wait_path);done=read(done_ref);successful_wait(wait_ref,done_ref)
    require(done["status"]=="completed_C3_short_TRAIN_native_focal_and_PyG_plan_v1" and done["parent_id"]==parent
        and done["observation_ref"]==obs_ref and done["focal_records"]==10000 and done["query_ancestry_truth_opened"] is False
        and done["all_trees_and_samples_retained"] is True,"Genuine complete focal/plan metadata")
    plan_ref=done["block_index_ref"];plan=read(plan_ref);native=read(done["native_preparation_ref"])
    require(plan["schema"]=="american_c3_whole_tree_block_index_v1" and plan["parent_id"]==parent and plan["sequence_length_bp"]==2000000
        and plan["haplotypes"]==1600 and plan["focal_records"]==10000 and plan["targets_opened"] is False
        and plan["class_order"]==["EUR","EAS","AFR"] and plan["all_trees_and_samples_retained"] is True
        and plan["observation_ref"]==obs_ref and plan["focal_registry_ref"]==done["focal_registry_ref"]
        and plan["config"]["native_preparation_ref"]==done["native_preparation_ref"]
        and native["observation_ref"]==obs_ref and native["focals_ref"]==plan["focal_registry_ref"],"Same exact observed native focal/plan identities")
    require(all(plan["config"][k]==v for k,v in dict(max_trees_per_block=32,max_nodes_per_block=250000,max_sites_per_block=4096,reference_counts=[200,200,200]).items()),"Original block recipe")
    declared=plan["feature_tensor_byte_planning_bound_per_block"]
    formula=256*250000+32*32*1600+(1600+256)*4096
    require(declared==formula and len(plan["blocks"])==done["blocks"],"Source-declared feature planning arithmetic")
    blocks=plan["blocks"]
    result.update(parent_plan_ref=plan_ref,parent_plan_completion_ref=done_ref,parent_plan_wait_ref=wait_ref,
        plan_metadata=dict(trees=plan["trees"],sites=plan["sites"],blocks=len(blocks),focals=plan["focal_records"],
            max_block_nodes=max(b["nodes"] for b in blocks),max_block_sites=max(b["sites"] for b in blocks),
            max_block_trees=max(b["stop_tree"]-b["start_tree"] for b in blocks),
            feature_tensor_byte_planning_bound_per_block=declared,
            sum_of_per_block_planning_bounds_bytes=declared*len(blocks),native_preparation_ref=done["native_preparation_ref"]))
    return result

def old_val000():
    rel="research/reviews/C3_GNN_current980_executable_root_20261015_v1"
    stage="plan_val_000";folder=ROOT/rel/("admitted_"+stage+"_v1")
    paths=[ROOT/"experiments/results/C3_GNN_current980_plan_val_000_v1/completed.json",folder/"actual_wait.json",folder/"closed/close_receipt.json"]
    result=dict(parent_id="american_c3_val_000",ready=False,plan_stage_evidence=None,bindings=None)
    if not all(x.is_file() for x in paths):return result
    er,wr,cr=[file_ref(p) for p in paths];envelope=read(er);successful_wait(wr,er);close=read(cr)
    require(envelope["status"]=="completed_C3_GNN_current980_plan_val_000" and envelope["stage"]==stage,"Actual old plan stage")
    v=read(envelope["result_ref"]);plan_ref=v["block_index_ref"];plan=read(plan_ref)
    require(v["status"]=="completed_C3_observation_only_block_plan" and v["parent_id"]=="american_c3_val_000"
        and v["targets_opened"] is False and plan["sequence_length_bp"]==30000000,"Original observation-only30Mb plan")
    require(close["kind"]=="concurrent_completed_scope_close_receipt_v1" and close["value"]["future_writes_and_relaunch_revoked"] is True
        and close["value"]["request"]["completed_ref"]==er and close["value"]["request"]["direct_wait_ref"]==wr,"Actually closed old plan")
    obs=read(close["value"]["request"]["observation_ref"]);approval=read(obs["root_approval_ref"])
    require(approval["root_actually_consumed_supervisor_terminal_wait0"] is True
        and approval["all_prior_launch_resume_write_capabilities_revoked"] is True,"Actual root terminal consumption/revocation")
    old_req=get(ROOT/rel/"request_plan_val_000.root_active_v1.json")
    evidence=dict(route="current980_zero_worker",completion_ref=er,actual_wait_ref=wr,close_receipt_ref=cr)
    b=dict(parent_id="american_c3_val_000",plan_ref=plan_ref,native_preparation_ref=old_req["bindings"]["native_preparation_ref"],
        preparation_wait_ref=old_req["bindings"]["preparation_wait_ref"],plan_stage_evidence=evidence,training=None)
    result.update(ready=True,plan_stage_evidence=evidence,bindings=b,plan_metadata=dict(trees=plan["trees"],sites=plan["sites"],blocks=len(plan["blocks"])))
    return result

def snapshot(folder):
    require(folder.resolve()==folder and folder.parent==HERE and not folder.exists(),"Fresh immediate snapshot child only")
    registry=get(REGISTRY);require([x["parent_id"] for x in registry["parents"]]==list(TRAIN),"Original fixed20 roster")
    rows=[one_parent(i,row) for i,row in enumerate(registry["parents"])]
    plans=[x["plan_metadata"] for x in rows if x["plan_metadata"] is not None]
    require(plans,"Actual verified plan metadata required")
    cap_block=128*MiB;cache=512*MiB;workers=2;worker_rss=8*GiB;aggregate=120*GiB
    max_declared=max(x["feature_tensor_byte_planning_bound_per_block"] for x in plans)
    min_aggregate=cache+(workers+2)*cap_block+workers*worker_rss
    require(cap_block>=max_declared and aggregate>=min_aggregate,"Explicit conservative planning inequality")
    val=old_val000()
    folder.mkdir()
    caps=dict(schema="C3_GNN_prefetch_current980_operational_caps_v1",active=False,phase="train",
        bound_method="operational_fail_stop_not_measured_fit_or_timing",reservation_cap_bytes=2*GiB,
        phase_write_budget_bytes=GiB,temp_allowance_bytes=512*MiB,outer_auxiliary_cap_bytes=128*MiB,
        root_packet_overhead_cap_bytes=4*MiB,outer_log_cap_bytes=4*MiB,child_log_ram_bytes=4*MiB,
        max_cpu_rss_bytes=128*GiB,max_cuda_reserved_bytes=40*GiB,operational_deadline_seconds=86400,
        observation_prefetch=dict(num_workers=workers,max_block_bytes=cap_block,worker_rss_cap_bytes=worker_rss,
            aggregate_rss_cap_bytes=aggregate,task_timeout_seconds=900,drain_timeout_seconds=30))
    caps_ref=save(folder,"caps.train.proposed.inactive.json",caps)
    config=get(PACKET/"training_config.inactive.json");require(config["CPU_observation_cache_bytes"]==cache,"Existing scientific config cache unchanged")
    mappings={k:{x["parent_id"]:x[v] for x in rows} for k,v in [
        ("actual_train_bindings","actual_train_binding"),("parent_plan_refs","parent_plan_ref"),
        ("parent_plan_completion_refs","parent_plan_completion_ref"),("parent_plan_wait_refs","parent_plan_wait_ref")]}
    save(folder,"TRAIN_bindings.available.v1.json",dict(schema="C3_GNN_available_TRAIN_binding_maps_v1",bindings=mappings,
        fully_ready=sum(not x["pending"] for x in rows),all20_ready=all(not x["pending"] for x in rows)))
    req=get(PACKET/"request_train.inactive.json");req["bindings"]=mappings;req["caps_template_ref"]=caps_ref
    require(req["active"] is False and req["physical_gpu"] is None and req["GPU_UUID"] is None,"No activation/device choice")
    save(folder,"request_train.available.proposed.inactive.json",req)
    save(folder,"TRAIN_metadata_evidence.v1.json",dict(schema="C3_GNN_TRAIN_metadata_snapshot_v1",
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parents=rows,private_arrays_or_checkpoints_opened=False))
    save(folder,"VAL000_closed_plan_reuse.v1.json",val)
    proposal=dict(schema="C3_GNN_persistent_prefetch_cap_rationale_v1",proposal_only=True,actual_admission=False,
        chosen_workers=2,reason="One persistent pool and one task per worker; limit duplicated native observation state and shared-host contention before any measured throughput result",
        verified_plan_count=len(plans),verified_training_input_count=sum(x["actual_train_binding"] is not None for x in rows),
        max_declared_block_bytes=max_declared,declared_formula="256*250000 +32*32*1600 +(1600+256)*4096",
        max_parent_blocks=max(x["blocks"] for x in plans),max_parent_trees=max(x["trees"] for x in plans),max_parent_sites=max(x["sites"] for x in plans),
        max_actual_block_nodes=max(x["max_block_nodes"] for x in plans),max_actual_block_sites=max(x["max_block_sites"] for x in plans),
        max_sum_per_parent_feature_planning_bytes=max(x["sum_of_per_block_planning_bounds_bytes"] for x in plans),
        block_cap_headroom_ratio=cap_block/max_declared,cache_bytes=cache,minimum_aggregate_planning_bytes=min_aggregate,
        proposed_aggregate_bytes=aggregate,proposed_root_owned_RSS_bytes=128*GiB,root_helper_RSS_headroom_bytes=8*GiB,
        caps_ref=caps_ref,source_formula_ref=file_ref(ROOT/"src/arggnn/american_c3_gnn_dataset_v1.py"),
        proven_components=["Exact JSON identity/lineage/wait checks for available inputs","Original full-plan declared feature byte formula and dimensions","Original tiny actual PyG/shared-tensor transport mechanism receipt; no persistent lifecycle measurement"],
        unproven_fail_stop_proposals=["Per-worker8GiB peak and aggregate120GiB/root128GiB RSS","GPU reserved40GiB subject to root actual device/UUID/free-memory admission","2GiB storage reservation with1GiB product budget and512MiB TMP allowance","900s task/30s release-drain/86400s overall deadlines","128MiB block cap headroom; actual full Dataset/persistent-context behavior and600-update performance"],
        notes=["Feature planning bounds do not bound autograd/native allocation peak","Peak RSS sums are historical and conservative; state release does not reduce them",
            "Final018019 plans must be checked against block cap; all20 waits/refs plus physicalGPU/UUID must be genuinely supplied before binding",
            "No shorter run, fallback, retry, partial optimizer completion or accuracy claim is implied"])
    save(folder,"caps_rationale.v1.json",proposal)
    save(folder,"metadata_source_inventory.v1.json",dict(schema="C3_GNN_metadata_only_source_inventory_v1",
        refs=[dict(path=p,sha256=s) for p,s in sorted(SOURCE_FILES_READ.items()) if not p.startswith(str(folder.relative_to(ROOT))+"/")],
        forbidden_payload_suffixes=[".npz",".npy",".trees",".pt",".pth",".vcf",".bcf"],payloads_opened=False))
    print(json.dumps(dict(snapshot=str(folder.relative_to(ROOT)),available_input_bindings=proposal["verified_training_input_count"],
        available_plan_bindings=len(plans),pending=[dict(parent_id=x["parent_id"],reasons=x["pending"]) for x in rows if x["pending"]],
        max_declared_block_bytes=max_declared,minimum_aggregate_planning_bytes=min_aggregate,VAL000_closed_plan_ready=val["ready"],
        active=False,actual_admission=False),indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--snapshot",action="store_true");p.add_argument("--output-directory")
    a=p.parse_args()
    if not a.snapshot:print(json.dumps(dict(status="metadata_source_only",actual_bind=False,actual_admission=False)));return
    require(a.output_directory,"Explicit new snapshot path required")
    snapshot(local(a.output_directory))
if __name__=="__main__":main()
