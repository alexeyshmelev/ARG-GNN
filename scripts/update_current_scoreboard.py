"""Refresh the current evidence-based scoreboard without training or truth access."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, os, statistics
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
RESEARCH=ROOT/"research"
PRIMARY="primary_present_day_182"
COHORTS=[PRIMARY,"age_stratum_Bronze_Age","age_stratum_BAA","age_stratum_Neo","age_stratum_Yam","secondary_ancient_504","all_queries_686"]
LABELS=["Present182","BronzeAge162","BAA14","Neo302","Yam26","Ancient504","All686"]
def read(p):
    return json.loads((ROOT/p).read_text())
def ref(p):
    return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def stat(mean,sd=None):
    return dict(mean=mean,sample_SD=sd)
def normalize(row):
    result={}
    for k,v in row.items():
        if isinstance(v,dict) and "mean" in v:
            result[k]=stat(v["mean"],v.get("sample_SD",v.get("sample_sd")))
            for nk in ("n","defined_parents","defined_parent_count","undefined_parent_count"):
                if nk in v:
                    result[k][nk]=v[nk]
    if "boundaries" in row:
        for k,v in row["boundaries"].items():
            result["boundary_"+k]=stat(v["mean"],v.get("sample_SD",v.get("sample_sd")))
            for nk in ("n","defined_parents","defined_parent_count","undefined_parent_count"):
                if nk in v:
                    result["boundary_"+k][nk]=v[nk]
    if "boundary_F1" in result:
        result["boundary_f1"]=result.pop("boundary_F1")
    return result
def fmt(v,percent=False):
    if not v or v.get("mean") is None:
        return "—"
    if isinstance(v["mean"],str):
        return v["mean"]
    mul=100 if percent else 1
    number=v["mean"]*mul
    if not math.isfinite(number):
        return "∞" if number>0 else "undefined"
    text=f"{number:.4f}" if percent else f"{number:.6f}"
    if v.get("sample_SD") is not None:
        text+=f" ± {v['sample_SD']*mul:.4f}" if percent else f" ± {v['sample_SD']:.6f}"
    n=v.get("defined_parents",v.get("defined_parent_count",v.get("n")))
    if n is not None and n<8:
        text+=f" (n={n})"
    return text

def maybe_read(p):
    path=ROOT/p
    try:return json.loads(path.read_text()) if path.is_file() else None
    except (FileNotFoundError,json.JSONDecodeError):return None
def known_process(pid):
    """Read only the previously recorded project PID, never a host process inventory."""
    try:
        base=Path("/proc")/str(pid)
        fields=(base/"stat").read_text().rsplit(")",1)[1].split()
        title=(base/"cmdline").read_bytes().decode(errors="replace")
        try:wait_channel=(base/"wchan").read_text().strip()
        except OSError:wait_channel="unavailable"
        return dict(present=True,required_title_present="AI Auto Research Pipeline Test" in title,
            start_ticks=int(fields[19]),CPU_seconds=(int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK"),
            current_RSS_bytes=int(fields[21])*os.sysconf("SC_PAGE_SIZE"),wait_channel=wait_channel)
    except (OSError,IndexError,ValueError):return dict(present=False,required_title_present=False)
def training_job(hid,gpu,output,packet):
    rows=sorted((ROOT/output).glob("TRAIN_pass_*.json"))
    last=maybe_read(str(rows[-1].relative_to(ROOT))) if rows else None
    last=last or {}
    title=maybe_read(packet+"/NVTOP_title_verification_root_v1.json") or {}
    process=known_process(title.get("own_pid",-1))
    updates=last.get("steps",{})
    qualification=maybe_read(output+"/first_real_parent_step_qualification.json")
    if not updates and qualification:
        updates={arm:r["optimizer_step"] for arm,r in qualification.get("arms",{}).items() if "optimizer_step" in r}
    complete=maybe_read(output+"/completed.json")
    wait=maybe_read(packet+"/actual_wait.json")
    inp=maybe_read(packet+"/input.json") or {}
    config_ref=inp.get("refs",{}).get("config")
    cfg=maybe_read(config_ref["path"]) if config_ref else {}
    cfg=cfg or {}
    status=("TRAIN completed; evaluation pending" if complete else
        "failed; partial artifacts preserved" if maybe_read(output+"/failed.json") or (wait and wait.get("returncode")!=0) else
        "training; completed-pass update counts are a lower bound" if process["present"] and updates else
        "runtime started; optimizer updates not yet verified" if process["present"] and (ROOT/output/"started.json").exists() else
        "admitted; runtime startup pending" if inp and process["present"] else
        "terminal status requires actual wait/completion review")
    if not complete and process["present"] and updates and process.get("wait_channel") in ("locks_lock_inode_wait","nfs4_retry_setlk"):
        status="training process live, currently waiting for an NFS/file lock; completed-pass counts remain lower bounds"
    return dict(hypothesis=hid,GPU=gpu,worker_pid=title.get("own_pid"),verified_process_present=process["present"] and process["required_title_present"],
        process_observation=process,runtime_status=status,passes_completed=last.get("pass_index",0),target_passes=30,
        updates_per_arm=updates,updates_are_verified_receipt_lower_bounds=True,target_updates_per_arm=960,
        completed=complete is not None,actual_TRAIN_wait0=bool(wait and wait.get("actual_direct_launcher_wait") is True and wait.get("returncode")==0),
        accuracy_available=False,evaluation_completed=False,validation_score="pending",
        last_pass_ref=ref(str(rows[-1].relative_to(ROOT))) if rows else None,
        first_step_receipt_ref=ref(output+"/first_real_parent_step_qualification.json") if qualification else None,
        started_ref=ref(output+"/started.json") if (ROOT/output/"started.json").exists() else None,
        title_verification_ref=ref(packet+"/NVTOP_title_verification_root_v1.json") if title else None,
        config_ref=config_ref,training=dict(optimizer="AdamW",learning_rate=cfg.get("learning_rate"),weight_decay=cfg.get("weight_decay"),
            seed=cfg.get("seed"),dtype=cfg.get("dtype"),passes=cfg.get("passes"),updates_per_arm=cfg.get("updates_per_arm"),
            parameters_per_arm=cfg.get("parameters_per_arm")))
def cpu_stage(output,packet,flare=False):
    completed=maybe_read(output+"/completed.json")
    wait_name="worker_aux/outer_wait.json" if flare else "actual_wait.json"
    wait=maybe_read(packet+"/"+wait_name)
    launch=maybe_read(packet+("/worker_aux/launch_monitor_seed.json" if flare else "/launch.json")) or {}
    pid=launch.get("worker_pid",launch.get("pid"))
    minimal=maybe_read(wait["actual_wait_ref"]["path"]) if wait and flare else None
    success=bool(completed and wait and wait.get("returncode")==0
        and ((flare and wait.get("error") is None and wait.get("observed_owned_groups_ESRCH") is True
              and minimal and minimal.get("actual_direct_wait") is True and minimal.get("returncode")==0)
             or (not flare and wait.get("actual_direct_worker_wait") is True
                 and wait.get("owned_worker_group_ESRCH",wait.get("worker_process_group_drained")) is True and wait.get("operational_failure") is None
                 and wait.get("completed_ref")==ref(output+"/completed.json"))))
    live=bool(pid and (Path("/proc")/str(pid)).exists())
    status=("completed; actual wait0 and owned-group drain verified" if success else
        "failed or incomplete terminal receipt; preserve artifacts" if wait else
        "running on CPU" if live else "admitted; worker launch pending" if (ROOT/packet/"admission.json").exists() else "not admitted")
    return dict(status=status,completed_and_wait0_verified=success,recorded_worker_process_present=live,
        completion_ref=ref(output+"/completed.json") if completed else None,
        actual_wait_ref=ref(packet+"/"+wait_name) if wait else None)

def root_CPU_stage(output,packet):
    done=maybe_read(output+"/completed.json");wait=maybe_read(packet+"/actual_wait.json")
    launch=maybe_read(packet+"/launch.json") or {}
    success=bool(done and wait and wait.get("actual_direct_launcher_wait") is True
        and wait.get("returncode")==0 and wait.get("all_owned_processes_and_groups_drained") is True
        and wait.get("operational_failure") is None and wait.get("completed_ref")==ref(output+"/completed.json"))
    pid=launch.get("pid")
    live=bool(pid and (Path("/proc")/str(pid)).exists())
    status=("completed; actual wait0 and owned drain verified" if success else
        "failed or terminal evidence requires review" if wait else "running on CPU" if live else
        "admitted; worker startup pending" if maybe_read(packet+"/admitted.json") else "not admitted")
    return dict(status=status,completed_and_wait0_verified=success,
        completion_ref=ref(output+"/completed.json") if done else None,
        actual_wait_ref=ref(packet+"/actual_wait.json") if wait else None)
def main():
    import importlib.util
    normalization_path=ROOT/"research/reviews/current_score_reporting_audit_20261015_v1/normalization_reference_v1.py"
    normalization_spec=importlib.util.spec_from_file_location("_scored_cells_reporting_normalization",normalization_path)
    normalization=importlib.util.module_from_spec(normalization_spec)
    normalization_spec.loader.exec_module(normalization)
    all_scored_details=normalization.load_normalized()
    native_path=ROOT/"research/reviews/current_score_reporting_audit_20261015_v1/native_metric_overrides_v1.py"
    native_spec=importlib.util.spec_from_file_location("_native_scored_metrics_reporting",native_path)
    native_normalization=importlib.util.module_from_spec(native_spec)
    native_spec.loader.exec_module(native_normalization)
    native_details=native_normalization.load_native_overrides()
    models=[]
    def add(name,source,cohorts,training,notes=""):
        models.append(dict(name=name,source_ref=ref(source),cohorts=cohorts,training=training,notes=notes))
    p="research/reviews/native_common_cohort_diagnostics_20261015_v1/same_support_comparison.json"
    d=read(p)
    for key,label in [("aligned","H038 aligned whole-tree PyG"),("zero_concordance","H038 zero-concordance control")]:
        if key in d["models"]:
            add(label,p,{r["cohort"]:normalize(r) for r in d["models"][key]["rows"]},
                "30 passes / 960 updates, seed101")
    p="research/reviews/native_common_cohort_diagnostics_20261015_v1/native_cohort_metrics_mean_SD.json"
    for key,label in [("native_ARGMix","Current native ARGMix"),("native_AP","Native AncestralPaths derived C4")]:
        rows=read(p)["models"][key]["rows"]
        cs={r["cohort"]:normalize(r) for r in rows}
        for cohort,values in native_details[key]["cohorts"].items():
            cs.setdefault(cohort,{}).update(values)
        add(label,p,cs,("11 epochs, native early stopping" if key=="native_ARGMix" else "5 epochs, native fixed schedule"),
            ("Native inputs/objective/selection retained. Current author ARGMix release." if key=="native_ARGMix" else "Native inputs/objective/selection retained. AP six-path probabilities collapsed to C4."))
    p="research/reviews/H040_completed_comparison_root_20261015_v1/summary.json"
    for key,r in read(p)["rows"].items():
        cs=all_scored_details["H040"]["arms"][key]["cohorts"]
        add("H040 "+key,all_scored_details["H040"]["source_ref"]["path"],cs,
            "H038 warmstart +30 passes /960 updates, seed101")
    p="research/reviews/H042_completed_cohort_diagnostics_root_20261015_v1/summary.json"
    add("H042 exact local-clade flags",p,{c:normalize(v) for c,v in read(p)["cohorts"].items()},
        "H038 warmstart +30 passes /960 updates, seed101")
    p="research/reviews/H043_completed_comparison_root_20261015_v1/summary.json"
    for key,r in read(p)["models"].items():
        add("H043 "+key,p,{c:normalize(v) for c,v in r["cohorts"].items()},
            "H042 warmstart +30 passes /960 updates per arm, seed101")
    p="research/reviews/H044_completed_comparison_root_20261015_v1/summary.json"
    for key,r in read(p)["arms"].items():
        cs=all_scored_details["H044"]["arms"][key]["cohorts"]
        add("H044 "+key,all_scored_details["H044"]["source_ref"]["path"],cs,"H042 warmstart +30 passes /960 updates per arm, seed101",
            "Written scores independently verified. Original post-completion CLI exit1 preserved.")
    p="research/reviews/H045_completed_comparison_root_20261015_v1/summary.json"
    for key,r in read(p)["arms"].items():
        add("H045 "+key,p,{c:normalize(v) for c,v in r["groups"].items()},
            "H042 warmstart +30 passes /960 updates per arm, seed101",
            "Foreign-minus-own accuracy gain0.009568pp, below0.10pp prespecified criterion.")
    h047_summary="research/reviews/H047_completed_comparison_root_20261015_v1/summary.json"
    if (ROOT/h047_summary).exists():
        for key,r in read(h047_summary)["arms"].items():
            add("H047 "+key,h047_summary,{c:normalize(v) for c,v in r["cohorts"].items()},
                "H042 warmstart +30 passes /960 updates per arm, seed101",
                "Balanced-minus-original accuracy -0.020029pp. Registered practical criterion not met.")
    h046_summary="research/reviews/H046_completed_comparison_root_20261015_v1/summary.json"
    h046=maybe_read(h046_summary)
    h046_wait_ref=h046.get("actual_launcher_wait_ref",{}) if h046 else {}
    h046_wait=maybe_read(h046_wait_ref["path"]) if h046_wait_ref else None
    h046_ready=bool(h046 and h046_wait and ref(h046_wait_ref["path"])==h046_wait_ref
        and h046_wait.get("actual_direct_launcher_wait") is True and h046_wait.get("returncode")==0
        and h046_wait.get("logging_error") is None and h046_wait.get("supervisor_error") is None
        and h046_wait.get("completed_ref")==h046["completion_ref"]
        and ref(h046["completion_ref"]["path"])==h046["completion_ref"]
        and h046.get("all96_products_sealed_before_truth") is True and h046.get("outer_test_accessed") is False)
    h046_outcome="H046 evaluation and accuracy pending."
    h046_scores="pending"
    if h046_ready:
        effect=h046["paired_effects"]
        h046_outcome=("H046 foreign-flank minus own-state Present182 accuracy gain "
            +f"{effect['mean_accuracy_gain']*100:+.6f} ± {effect['sample_SD_accuracy_gain']*100:.6f} percentage points "
            +"(sample SD of paired differences across eight VAL parents). "
            +f"Mean boundary F1 difference {effect['mean_boundary_F1_delta']*100:+.6f} percentage points. "
            +"The registered utility rule "+("passed" if effect["meets_registered_practical_criteria"] else "failed")
            +": at least +0.10 accuracy points and at most 1.00 point mean boundary F1 loss. "
            +"Both selected checkpoints are original pass 30; one optimizer seed. This is reused development validation, not independent test or novelty evidence.")
        h046_scores="; ".join(fmt(normalize(h046["arms"][a]["cohorts"][PRIMARY])["accuracy"],True)+"% "+a
            for a in ("foreign_flank","own_state"))
        for key,r in h046["arms"].items():
            add("H046 "+key,h046_summary,{c:normalize(v) for c,v in r["cohorts"].items()},
                "H042 warmstart +30 passes /960 updates per arm, seed101",
                "Foreign-minus-own accuracy +0.011573 ± 0.069124pp paired sample SD; registered utility criterion not met.")
    h048_summary="research/reviews/H048_completed_comparison_root_20261015_v1/summary.json"
    h048_result=maybe_read(h048_summary)
    h048_wait_ref=h048_result.get("actual_launcher_wait_ref",{}) if h048_result else {}
    h048_wait=maybe_read(h048_wait_ref["path"]) if h048_wait_ref else None
    h048_ready=bool(h048_result and h048_wait and ref(h048_wait_ref["path"])==h048_wait_ref
        and h048_wait.get("actual_direct_launcher_wait") is True and h048_wait.get("returncode")==0
        and h048_wait.get("logging_error") is None and h048_wait.get("supervisor_error") is None
        and h048_wait.get("completed_ref")==h048_result["completion_ref"]
        and ref(h048_result["completion_ref"]["path"])==h048_result["completion_ref"]
        and h048_result.get("all96_products_sealed_before_truth") is True and h048_result.get("outer_test_accessed") is False)
    h048_scores="pending"
    if h048_ready:
        h048_scores="; ".join(fmt(v["cohorts"][PRIMARY]["accuracy"],True)+"% "+a for a,v in h048_result["arms"].items())
        for a,v in h048_result["arms"].items():
            add("H048 "+a,h048_summary,{c:normalize(vv) for c,vv in v["cohorts"].items()},
                "H042 warmstart +30passes/960updates per arm, seed101",
                "Both registered utility criteria failed; graph-transition accuracy delta -0.008640pp.")
            models[-1]["training_budget_lineage"]={k:v[k] for k in ("completed_extra_updates","cumulative_model_lineage_updates","selected_model_lineage_updates","selected_pass")}
    audit_path="research/reviews/current_score_reporting_audit_20261015_v1/all17_allcohort_metrics.actual.json"
    audited=read(audit_path)
    for original in audited["models"]:
        matched=next((r for r in models if r["name"]==original["name"]),None)
        if matched is None:continue
        raw_ref={k:original["raw_score_ref"][k] for k in ("path","sha256")}
        summary_ref={k:original["score_summary_ref"][k] for k in ("path","sha256")}
        assert ref(raw_ref["path"])==raw_ref and ref(summary_ref["path"])==summary_ref
        assert matched["source_ref"] in (raw_ref,summary_ref)
        matched.update(cohorts=original["cohorts"],raw_score_ref=raw_ref,
            training_budget_lineage={k:original[k] for k in (
                "training_budget","completed_extra_updates","cumulative_model_lineage_updates",
                "selected_model_lineage_updates","selected_pass")})
    models.sort(key=lambda r:r["cohorts"][PRIMARY]["accuracy"]["mean"],reverse=True)
    jobs=[]
    for hid,gpu,output,packet in [
        ("H046",5,"experiments/results/H046_GPU5_flank_train_seed101_v2","research/reviews/H046_GPU45_flank_root_source_v2/admitted_train"),
        ("H047",4,"experiments/results/H047_GPU4_cohort_risk_train_seed101_v1","research/reviews/H047_GPU45_cohort_risk_root_source_v1/admitted_train"),
        ("H048",4,"experiments/results/H048_GPU4_grid_reset_train_seed101_v1","research/reviews/H048_GPU45_grid_root_source_v1/admitted_train")]:
        row=training_job(hid,gpu,output,packet)
        if hid=="H047" and (ROOT/h047_summary).exists():
            row.update(accuracy_available=True,evaluation_completed=True,summary_ref=ref(h047_summary),
                validation_score="96.7777% original-risk, 96.7576% balanced-risk",runtime_status="TRAIN and VAL completed")
        if hid=="H046" and h046_ready:
            row.update(accuracy_available=True,evaluation_completed=True,summary_ref=ref(h046_summary),
                actual_evaluation_wait0=True,evaluation_wait_ref=h046_wait_ref,
                validation_score=h046_scores,runtime_status="TRAIN and VAL completed")
        if hid=="H048":
            ep="research/reviews/H048_GPU45_grid_root_source_v1/admitted_evaluate"
            ew=maybe_read(ep+"/actual_wait.json")
            el=maybe_read(ep+"/launch.json")
            ea=maybe_read(ep+"/admitted.json")
            row.update(evaluation_admission_ref=ref(ep+"/admitted.json") if ea else None,
                evaluation_launch_ref=ref(ep+"/launch.json") if el else None,
                evaluation_wait_ref=ref(ep+"/actual_wait.json") if ew else None)
            if row["completed"] and row["actual_TRAIN_wait0"]:
                row["updates_are_verified_receipt_lower_bounds"]=False
                row["runtime_status"]=("TRAIN completed30passes/960updates per arm with actual wait0; evaluation terminal requires review" if ew else
                    "TRAIN completed30passes/960updates per arm with actual wait0; registered evaluation launched, score pending" if el else
                    "TRAIN completed30passes/960updates per arm with actual wait0; evaluation admitted, startup pending" if ea else
                    "TRAIN completed30passes/960updates per arm with actual wait0; evaluation pending")
            if h048_ready:
                row.update(accuracy_available=True,evaluation_completed=True,summary_ref=ref(h048_summary),
                    actual_evaluation_wait0=True,validation_score=h048_scores,runtime_status="TRAIN and VAL completed; both registered utility criteria failed")
            boundary_path="research/reviews/H048_boundary_contention_root_20261015_v1/actual_scheduler_lock_observation.root_v1.json"
            boundary=maybe_read(boundary_path)
            if boundary and row["passes_completed"]==boundary["no_new_pass_receipt_count"] and row["worker_pid"]==boundary["pid"] and row["process_observation"].get("start_ticks")==boundary["start_ticks"]:
                row.update(latest_boundary_observation_ref=ref(boundary_path),runtime_status="training process alive; latest observation at shared scheduler mutex boundary; compute may precede receipt counts")
        jobs.append(row)
    gstpath="research/reviews/GST_training_readiness_20261015_v1/readiness.json"
    gst=read(gstpath)["existing_observations"]["models"]
    empirical_path="data/H018_empirical_observation_bank_root_v3/completed.json"
    empirical=read(empirical_path) if (ROOT/empirical_path).exists() else None
    real_status=("Observed bank prepared, 2074 sites, 355 trees, 40 haplotypes. Metadata recovered from the saved bank after original worker exit1. Model application and PCA pending."
        if empirical else "2074-site observed bridge and355-tree graph complete. Model application and PCA pending.")
    flare_build=cpu_stage("data/native_FLARE_build_existing_JDK_root_v1","research/reviews/FLARE_existing_JDK_build_root_v1",True)
    flare_conversion=cpu_stage("data/FLARE_train000_convert_980_root_v1","research/reviews/FLARE_train000_convert_980_root_v1",True)
    flare_native=cpu_stage("data/FLARE_train000_native_980_root_v1","research/reviews/FLARE_train000_native_980_root_v1",True)
    seal_version="v2" if (ROOT/"research/reviews/FLARE_train000_seal_980_root_v2").exists() else "v1"
    seal_packet="research/reviews/FLARE_train000_seal_980_root_"+seal_version
    flare_seal=cpu_stage("data/FLARE_train000_seal_980_root_"+seal_version,seal_packet,True)
    seal_rejection_path="research/reviews/FLARE_train000_seal_980_root_v1/root_failed_admission_observation_v1.json"
    seal_rejection=maybe_read(seal_rejection_path)
    if seal_rejection and not (ROOT/seal_packet/"admission.json").exists():
        flare_seal.update(status="pending; initial stale-prefix admission rejected before reserve append; fresh "+seal_version+" admission pending",
            initial_rejection_ref=ref(seal_rejection_path))
    flare_close_path="research/reviews/FLARE_native_completed_close_20261015_v1/closed_FLARE_native_v1/close_receipt.json"
    flare_close=maybe_read(flare_close_path)
    flare_close_status=(" Native completed allowance closed; "+str(flare_close["credit_bytes"])+" unused bytes credited with all files preserved."
        if flare_close else "")
    flare_status=("Author build and TRAIN000 full-site conversion completed with actual wait0. Native inference: "+flare_native["status"]
        +"; sealing: "+flare_seal["status"]+"."+flare_close_status+" Accuracy pending; TRAIN000 is development, not held-out.")
    val_simulation=cpu_stage("experiments/results/American_C3_val000_full30mb_shared_v1/simulation",
        "research/reviews/prospective_980GB_capacity_source_v1/admitted_simulate_v1")
    val_inference=cpu_stage("experiments/results/American_C3_val000_full30mb_shared_v1/inference",
        "research/reviews/prospective_980GB_capacity_source_v1/admitted_infer_v1")
    val_native_paths=cpu_stage("experiments/results/American_C3_val000_full30mb_shared_v1/native_paths",
        "research/reviews/prospective_980GB_capacity_source_v1/admitted_native_paths_v1")
    flare_val_conversion=cpu_stage("data/FLARE_val000_convert_memo2_980_root_v2",
        "research/reviews/FLARE_val000_convert_memo2_980_root_v2",True)
    flare_val_old_stop="research/reviews/FLARE_VAL000_consumer_IO_repair_root_20261015_v1/original68464_actual_terminal_v1.json"
    val_status=("Original 30 Mb VAL000 simulation: "+val_simulation["status"]+"; inferred observation: "+val_inference["status"]
        +"; shared native-path/10k-focal preparation: "+val_native_paths["status"]+". R600/Q1000 identities stay common. No VAL ancestry score.")
    if all(stage["completed_and_wait0_verified"] for stage in (val_simulation,val_inference,val_native_paths)):
        val_status+=" Original full30Mb VAL000 native preparations are ready; method predictions and common-truth scores remain pending."
    val_flare_target_path="research/reviews/FLARE_VAL000_bridge_source_20261015_v1/bound_target_v1/target.json"
    val_flare_target=maybe_read(val_flare_target_path)
    val_flare_binding=None
    if val_flare_target and val_flare_target.get("parent_id")=="american_c3_val_000" and val_flare_target.get("external_truth_ref") is None:
        val_flare_counts=read(val_flare_target["count_manifest_ref"]["path"])
        val_flare_binding=dict(target_ref=ref(val_flare_target_path),count_manifest_ref=val_flare_target["count_manifest_ref"],
            sites=val_flare_counts["num_sites"],reference_haplotypes=600,query_haplotypes=1000,score_status=val_flare_target["score_status"])
        val_status+=" Native FLARE VAL000 target bound: "+str(val_flare_counts["num_sites"])+" sites, common R600/Q1000. Fresh memo2 conversion: "+flare_val_conversion["status"]+"; predictions/scoring pending. Earlier conversion deliberately stopped to repair scheduler contention, with all artifacts retained."
    flare_val_native=cpu_stage("data/FLARE_val000_native_namespace_980_root_v1","research/reviews/FLARE_val000_native_namespace_980_root_v1",True)
    flare_val_seal=cpu_stage("data/FLARE_val000_seal_namespace_980_root_v1","research/reviews/FLARE_val000_seal_namespace_980_root_v1",True)
    remaining_val_phases=[]
    remaining_val_root="research/reviews/American_C3_remaining_VAL_full30mb_root_source_20261015_v1"
    for index in range(1,5):
        for stage,dirname in (("simulate","simulation"),("infer","inference"),("native_paths","native_paths")):
            state=cpu_stage("experiments/results/American_C3_val%03d_full30mb_shared_v1/"%index+dirname,
                (remaining_val_root if index==1 else "research/reviews/American_C3_remaining_VAL_namespace_memo_root_20261015_v1")+"/admitted_val%03d_"%index+stage+"_v1")
            remaining_val_phases.append(dict(parent_id="american_c3_val_%03d"%index,stage=stage,**state))
    remaining_val_status="; ".join(r["parent_id"][-3:]+"/"+r["stage"]+": "+r["status"] for r in remaining_val_phases if r["status"]!="not admitted")
    remaining_val_status=remaining_val_status or "Original remaining VAL001–004 source ready; actual execution pending."
    native_launch="research/reviews/American_C3_native_MLM_amendment_v10/root_execution_admin_v2/worker_metadata/v11_root_outer_launch.json"
    native_status="Full30Mb corpus prepared. Native fit pending."
    native_device="GPU4/5 pending"
    native_execution=dict(optimizer_updates=None,optimizer_updates_verified=False,parent_id="american_c3_train_000",split="train",scientific_role="engineering_only_not_held_out")
    if (ROOT/native_launch).exists():
        native_pid=read(native_launch)["pid"]
        try:
            native_live="root_native_driver_v1.py" in (Path("/proc")/str(native_pid)/"cmdline").read_bytes().decode(errors="replace")
        except OSError:
            native_live=False
        native_status=("Full5epoch fit launched on GPU4. Startup/storage checks active. No native optimizer progress verified yet."
            if native_live else "Native full-fit root terminal. Actual wait and counters need inspection.")
    native_wait="research/reviews/American_C3_native_MLM_amendment_v10/root_execution_admin_v2/worker_metadata/v11_root_outer_actual_wait.json"
    if (ROOT/native_wait).exists():
        native_rc=read(native_wait)["returncode"]
        native_status=f"Full5epoch fit attempt terminal, actual root exit{native_rc}. No native optimizer progress verified. Execution repair pending."
    native_v12="research/reviews/American_C3_native_MLM_amendment_v10/root_execution_admin_v2/worker_metadata/v12_root_outer"
    v12_launch=maybe_read(native_v12+"_launch.json")
    v12_wait=maybe_read(native_v12+"_actual_wait.json")
    if v12_launch:
        bound_gpu=v12_launch.get("physical_gpu")
        if bound_gpu in (4,5):native_device="GPU"+str(bound_gpu)
        native_process=known_process(v12_launch.get("pid",-1))
        native_live=bool(native_process["present"] and native_process.get("start_ticks")==v12_launch.get("start_ticks"))
        native_execution.update(launch_ref=ref(native_v12+"_launch.json"),physical_gpu=bound_gpu,
            authority_ref=v12_launch.get("authority_ref"),recorded_root_process_present=native_live,
            process_observation=native_process)
    if v12_wait:
        native_status="V12 full-corpus attempt terminal, actual root exit "+str(v12_wait.get("returncode"))+". Native optimizer progress and completion require saved counters; prior v11 updates remain unknown."
        native_execution["actual_wait_ref"]=ref(native_v12+"_actual_wait.json")
    elif v12_launch:
        native_status=("V12 full-corpus launch recorded on "+native_device+". "
            +("Recorded root process present. " if native_live else "Recorded root process not currently observed; actual wait pending. ")
            +"Optimizer updates unverified; no native score. Prior v11 root exit1 and unknown updates preserved.")
        native_audit_root="research/reviews/native_ARGformer_v12_progress_source_audit_20261015_v1"
        native_audit_path=native_audit_root+"/observation_v1.json"
        native_audit=maybe_read(native_audit_path)
        if native_audit and v12_launch.get("pid")==2943908:
            progress=native_audit["root_reported_console"]
            checkpoints=[r for r in native_audit["checkpoint_metadata_only"] if not r["symlink"]]
            native_execution.update(training_activity_observed=True,
                source_progress_audit_ref=ref(native_audit_path),
                source_progress_report_ref=ref(native_audit_root+"/REPORT.md"),
                root_observed_Composer_progress=progress,checkpoint_metadata_only=checkpoints,
                saved_Adam_step_count_verified=False,quiet_console_expected_after_first8_batch_banners=True)
            native_status=("Native v12 author MLM training observed on "+native_device
                +"; Composer epoch "+str(progress["epoch_one_based"])+" through batch "+str(progress["batch_in_epoch"])
                +"/"+str(progress["dataloader_batches"])+" at about "+str(progress["batches_per_second_approx_range"][0])
                +"–"+str(progress["batches_per_second_approx_range"][1])+" batches/s. "
                +("Recorded root process present. " if native_live else "Actual root wait pending. ")
                +"Checkpoint metadata: "+", ".join(r["name"]+" ("+str(r["bytes"])+" bytes)" for r in checkpoints)
                +"; latest symlink present. Quiet console after the first eight batch banners is expected. "
                +"Five-epoch completion and saved Adam-step verification pending; no native ancestry score. Prior v11 exit1/unknown updates preserved.")
    elif (ROOT/"research/reviews/American_C3_native_MLM_amendment_v12/source_handoff.json").exists():
        cleanup=maybe_read("research/reviews/American_C3_native_MLM_amendment_v10/root_execution_admin_v2/worker_metadata/v12_namespace_attempt1/cleanup_receipt.json")
        prepared=bool(cleanup and cleanup.get("active") is True and cleanup.get("cleanup_failure") is None and len(cleanup.get("per_node_events",[]))==16)
        native_status=("V11 failed with actual root exit1; native optimizer updates unknown. V12 operational source reviewed; "
            +("exact namespace cleanup completed, GPU binding/activation and launch pending." if prepared else "root preparation/activation pending, no actual v12 launch receipt."))
    native_latest_path=("research/reviews/native_ARGformer_checkpoint_metadata_20261016_v1.json" if (ROOT/"research/reviews/native_ARGformer_checkpoint_metadata_20261016_v1.json").is_file() else "research/reviews/H048_boundary_contention_root_20261015_v1/native_MLM_progress_metadata_actual_v3.json")
    if (ROOT/"research/reviews/native_ARGformer_checkpoint_metadata_20261016_v2.json").is_file():native_latest_path="research/reviews/native_ARGformer_checkpoint_metadata_20261016_v2.json"
    if (ROOT/"research/reviews/native_ARGformer_checkpoint_metadata_20261016_v3.json").is_file():native_latest_path="research/reviews/native_ARGformer_checkpoint_metadata_20261016_v3.json"
    native_latest=maybe_read(native_latest_path)
    if native_latest and v12_launch and not v12_wait:
        latest_checkpoints=[r for r in native_latest["checkpoints"] if not r["symlink"]]
        native_execution.update(latest_checkpoint_metadata_observation_ref=ref(native_latest_path),checkpoint_metadata_only=latest_checkpoints)
        native_status=("Engineering TRAIN000: native author MLM training on "+native_device+". Checkpoint metadata observed at "+native_latest["observed_utc"]+": "+", ".join(r["name"] for r in latest_checkpoints)+". Full five-epoch/224775-update budget and final saved-Adam verification remain pending. No native ancestry score.")
    epoch_path=next("research/reviews/native_ARGformer_epoch%d_actual_console_20261015_v1.json"%ep for ep in (5,4,3,2) if (ROOT/("research/reviews/native_ARGformer_epoch%d_actual_console_20261015_v1.json"%ep)).is_file())
    epoch_event=maybe_read(epoch_path)
    if epoch_event and v12_launch and not v12_wait:
        progress=epoch_event["actual_console_payload"]
        native_execution.update(actual_author_console_progress_ref=ref(epoch_path),
            actual_author_console_epoch=progress["epoch_one_based"],actual_author_console_batch=progress["batch_in_epoch"])
        native_status+=" Actual unchanged author console entered epoch"+str(progress["epoch_one_based"])+"/5, batch"+str(progress["batch_in_epoch"])+"."
    if v12_launch and not native_live and not v12_wait:
        native_status="Recorded native root process cannot currently be confirmed. Actual terminal/completion evidence still requires review. Historical checkpoint/epoch observations are retained; do not infer termination or restart from this observation."
    gnn_VAL000_plan=root_CPU_stage("experiments/results/C3_GNN_current980_plan_val_000_v1",
        "research/reviews/C3_GNN_current980_executable_root_20261015_v1/admitted_plan_val_000_v1")
    counts_packet="research/reviews/C3_native_counts980_root_handoff_20261015_v1/admitted_counts980_v1"
    counts_output="experiments/results/American_C3_native_train000_full30mb_counts980_v1"
    counts_wait=maybe_read(counts_packet+"/actual_wait.json")
    counts_done=maybe_read(counts_output+"/completed.json")
    counts_launch=maybe_read(counts_packet+"/launch.json")
    counts_ready=bool(counts_wait and counts_done and counts_wait.get("returncode")==0 and counts_wait.get("actual_direct_launcher_wait") is True and counts_wait.get("all_owned_processes_and_groups_drained") is True and counts_wait.get("operational_failure") is None and counts_wait.get("completed_ref")==ref(counts_output+"/completed.json"))
    counts_status=("completed; actual count producer wait0 and drain verified" if counts_ready else "terminal receipt requires review" if counts_wait else "CPU count producer launched; completion pending" if counts_launch else "admitted; root CPU launch started, scientific child startup pending" if (ROOT/counts_packet/"admitted.json").exists() else "source ready")
    counts_execution=dict(status=counts_status,parent_id="american_c3_train_000",split="train",scientific_role="engineering_only_not_held_out",MLM_dependency=False,ancestry_score=None,completion_ref=ref(counts_output+"/completed.json") if counts_done else None,actual_wait_ref=ref(counts_packet+"/actual_wait.json") if counts_wait else None)
    val000_native_counts=root_CPU_stage("experiments/results/American_C3_native_val000_full30mb_counts980_namespace_v1",
        "research/reviews/C3_native_VAL000_counts_namespace_root_20261015_v1/admitted_counts980_v1")
    val000_native_counts.update(parent_id="american_c3_val_000",split="val",scientific_role="held_out_per_ARG_input_readiness",ancestry_score=None)
    if val000_native_counts["completed_and_wait0_verified"]:
        count_result=read("experiments/results/American_C3_native_val000_full30mb_counts980_namespace_v1/completed.json")
        val000_native_counts.update(actual_dataset_counts=count_result["datasets"],planned_MLM_optimizer_updates=count_result["MLM_optimizer_updates"],
            planned_contrastive_optimizer_updates=count_result["contrastive_optimizer_updates"],actual_fit_updates=None,
            native_training_eligible=count_result["native_training_eligible"],query_truth_opened=count_result["query_ancestry_truth_opened"])
    empirical_output=ROOT/"experiments/results/H018_H045_empirical_application_root_v1"
    empirical_aux=ROOT/"research/reviews/H018_H045_empirical_CPU_root_v1/worker_aux"
    empirical_products=all((empirical_output/n).exists() for n in ["probabilities_all355_holds.npz","query_bpweighted_proportions_and_PCA.tsv","query_genotype_PCA.svg"])
    empirical_wait=empirical_aux/"outer_wait.json"
    empirical_verified=(empirical_output/"completed.json").exists() and empirical_wait.exists() and json.loads(empirical_wait.read_text()).get("returncode")==0
    if empirical_products:
        real_status=("H045 empirical probabilities, bp-weighted proportions and original-genotype PCA written. "
            +("Actual successful wait and completion verified." if empirical_verified else "Root completion/wait verification pending.")
            +" Descriptive24query haplotypes/12subjects only. No known ancestry truth.")
    pilot_source="experiments/results/native_argformer_tsinfer_fit_v7_summary_v2/summary.json"
    pilot=read(pilot_source)
    pilot_cohorts=["primary_present_day_64","secondary_ancient_56","all_queries_120"]
    pilot_rows=[]
    for k in ["1","5"]:
        cs={}
        for c in pilot_cohorts:
            raw=[r["results_by_k"][k]["exact_support_metrics"][c] for r in pilot["rows"]]
            cs[c]={metric:stat(statistics.mean(r[metric] for r in raw),statistics.stdev(r[metric] for r in raw))
                for metric in ["accuracy","macro_f1","balanced_accuracy","brier","calibration_ece","dosage_mse_per_diploid_bp_class","dosage_mae_per_diploid_bp_class"]}
            cs[c]["nll"]=stat("∞")
        pilot_rows.append(dict(name="Adapted ARGformer top"+k,cohorts=cs))
    pilot_lines=["## Earlier adapted ARGformer pilot","",
        "Two European VAL simulations,280observed haplotypes,64present-day/56ancient queries,2Mb,tsinfer/tsdate trees. Native per-ARG5epoch MLM plus1reference-only contrastive epoch. This differs from both the952input panel above and the original American benchmark. SD is across two VAL simulations. Raw hard-vote NLL is infinite. No probability floor or recalibration.","",
        "| Setup | Present64 accuracy % ± SD | Ancient56 accuracy % ± SD | All120 accuracy % ± SD |",
        "|---|---:|---:|---:|"]
    for r in pilot_rows:
        pilot_lines.append("| "+r["name"]+" | "+" | ".join(fmt(r["cohorts"][c]["accuracy"],True) for c in pilot_cohorts)+" |")
    pilot_lines+=["","[Actual summary with all metrics]("+("../" if False else "")+pilot_source+")."]
    focal_packet="research/reviews/C3_next_plan_root_handoff_20261015_v1/admitted_american_c3_short_train_000"
    focal_output="experiments/results/C3_short_TRAIN_focal_plans_v1/american_c3_short_train_000"
    focal_prefix="root_retry2_" if maybe_read(focal_packet+"/root_retry2_actual_launch.json") else "root_"
    focal_launch=maybe_read(focal_packet+"/"+focal_prefix+"actual_launch.json")
    focal_wait=maybe_read(focal_packet+"/"+focal_prefix+"actual_wait.json")
    focal_done=maybe_read(focal_output+"/completed.json")
    focal_status=("completed with actual wait0" if focal_done and focal_wait and focal_wait.get("returncode")==0
        and focal_wait.get("actual_direct_worker_wait") is True and focal_wait.get("owned_process_group_ESRCH") is True else
        "terminal; actual receipt review required" if focal_wait else
        "running on CPU" if focal_launch and (Path("/proc")/str(focal_launch["pid"])).exists() else
        "admitted; launch pending" if (ROOT/focal_packet/"admission.json").exists() else "not admitted")
    if focal_prefix=="root_retry2_":focal_status+="; explicit retry after missing-parent directory repair, original failure preserved"
    if focal_done and focal_wait and focal_wait.get("returncode")==0:
        focal_status+="; "+str(focal_done["focal_records"])+" focals, "+str(focal_done["trees"])+" trees, "+str(focal_done["sites"])+" sites, "+str(focal_done["blocks"])+" blocks"
    fanout_root="research/reviews/C3_focal_plan_fanout_root_source_20261015_v1"
    fanout_run=fanout_root+"/run_v1"
    fanout_started=maybe_read(fanout_run+"/started.json")
    fanout_result=maybe_read(fanout_run+"/fanout_result.json")
    fanout_completed=[]
    for index in range(1,14):
        parent="american_c3_short_train_%03d"%index
        wait_path=fanout_run+"/"+parent+"/execute/actual_wait.json"
        done_path="experiments/results/C3_short_TRAIN_focal_plans_v1/"+parent+"/completed.json"
        fw=maybe_read(wait_path)
        if fw and fw.get("actual_direct_worker_wait") is True and fw.get("returncode")==0 and fw.get("owned_process_group_ESRCH") is True and (ROOT/done_path).is_file() and fw.get("completed_ref")==ref(done_path):
            fanout_completed.append(parent)
    fanout_status=("remaining TRAIN001–013 fanout "+str(len(fanout_completed))+"/13 completed with actual wait0"
        +("; terminal "+fanout_result["status"] if fanout_result else "; launched, completion pending") if fanout_started else
        "remaining TRAIN001–013 fanout source ready; root launch pending")
    fanout_stop_path="research/reviews/H048_boundary_contention_root_20261015_v1/stop_fanout_additions.root_v1.json"
    fanout_stop=maybe_read(fanout_stop_path)
    if fanout_stop:
        fanout_status+="; root requested cooperative stop of new additions to relieve GPU training mutex contention, current scientific children drain unchanged"
    continuation_root="research/reviews/C3_focal_plan_continuation_memo2_20261015_v1"
    continuation_run=continuation_root+"/run006_015_v1"
    continuation_started=maybe_read(continuation_run+"/started.json")
    continuation_result=maybe_read(continuation_run+"/fanout_result.json")
    continuation_completed=[]
    for index in range(6,16):
        parent="american_c3_short_train_%03d"%index
        wait=maybe_read(continuation_run+"/"+parent+"/execute/actual_wait.json")
        completed_path="experiments/results/C3_short_TRAIN_focal_plans_v1/"+parent+"/completed.json"
        if wait and wait.get("actual_direct_worker_wait") is True and wait.get("returncode")==0 and wait.get("owned_process_group_ESRCH") is True and not wait.get("errors") and (ROOT/completed_path).is_file() and wait.get("completed_ref")==ref(completed_path):
            continuation_completed.append(parent)
    if continuation_started:
        fanout_status+=". Fresh controlled TRAIN006–015 continuation "+str(len(continuation_completed))+"/10 completed with genuine wait0; max2CPU producers; "+("terminal "+continuation_result["status"] if continuation_result else "launched, completion pending")
    additional_continuation_runs=[]
    for run_name,indices in (("run016_017_v1",range(16,18)),("run018_v1",range(18,19)),("run019_v1",range(19,20)),("run018_019_v1",range(18,20)),("run016_019_v1",range(16,20))):
        directory=continuation_root+"/"+run_name
        started=maybe_read(directory+"/started.json")
        if not started:continue
        done=maybe_read(directory+"/fanout_result.json")
        verified=[]
        for index in indices:
            parent="american_c3_short_train_%03d"%index
            wait=maybe_read(directory+"/"+parent+"/execute/actual_wait.json")
            completed_path="experiments/results/C3_short_TRAIN_focal_plans_v1/"+parent+"/completed.json"
            if wait and wait.get("actual_direct_worker_wait") is True and wait.get("returncode")==0 and wait.get("owned_process_group_ESRCH") is True and not wait.get("errors") and (ROOT/completed_path).is_file() and wait.get("completed_ref")==ref(completed_path):verified.append(parent)
        continuation_completed.extend(p for p in verified if p not in continuation_completed)
        additional_continuation_runs.append(dict(run=run_name,verified_completed_parents=verified,started_ref=ref(directory+"/started.json"),result_ref=ref(directory+"/fanout_result.json") if done else None))
        fanout_status+=". "+run_name+": "+str(len(verified))+"/"+str(len(indices))+" genuine completed waits; "+(done["status"] if done else "active")
    focal_preparation=dict(first000_status=focal_status,first000_completion_ref=ref(focal_output+"/completed.json") if focal_done else None,
        first000_actual_wait_ref=ref(focal_packet+"/"+focal_prefix+"actual_wait.json") if focal_wait else None,
        fanout_status=fanout_status,fanout_completed_parents=fanout_completed,
        controlled_continuation_completed_parents=continuation_completed,
        additional_continuation_runs=additional_continuation_runs,
        controlled_continuation_started_ref=ref(continuation_run+"/started.json") if continuation_started else None,
        fanout_source_ref=ref(fanout_root+"/fanout_v1.py"),fanout_handoff_ref=ref(fanout_root+"/HANDOFF.md"))
    h048=next((r for r in read("research/experiment_decisions.json")["decisions"] if r["id"]=="H048"),None)
    h048_job=next(j for j in jobs if j["hypothesis"]=="H048")
    h048_status=("GPU4 "+h048_job["runtime_status"]+". "+str(h048_job["passes_completed"])+"/30 completed passes; "
        +(str(h048_job["updates_per_arm"])+" verified update lower bounds." if h048_job["updates_per_arm"] else "optimizer updates unverified.")
        +(" Validation: "+h048_scores+". Both registered utility criteria failed." if h048_ready else " No validation accuracy yet."))
    novelty_path="research/reviews/H048_novelty_crossdomain_fresh_20261015_v1/report.md"
    novelty_evidence_path="research/reviews/H048_novelty_crossdomain_fresh_20261015_v1/evidence.json"
    novelty_status=("Fresh independent H048 audit: reset transitions have RFMix precedent; a distinct graph-collection operator is unestablished. "
        "GST reaction-center sets have no native chain order, so whole-H048 transfer is not justified. "
        "PKU-MMD is only a possible future decoder study after native label-overlap semantics are resolved; no transfer result exists.")
    gst_full=read("research/reviews/GST_full_TRAIN_count_root_v1/counts.json")
    gst_checkpoint=read("research/reviews/GST_full_TRAIN_CPU_source_v1/checkpoint.json")
    gst_status=("Only the saved TRAIN 1000/VALID 1495, seed 0 CPU pilot is scored: four models, AdamW lr=0.001, weight_decay=0.0001, batch=16, 20 epochs/1260 updates per fit. "
        "Full TRAIN has "+str(gst_full["retained_rows"])+"/"+str(gst_full["raw_rows"])+" retained reactions; this count is data readiness, not a fit. "
        "The full-data multi-seed fit is inactive and unfinished. Our molecular adapter and native paired TEST comparison remain unfinished; TEST withheld. GST inputs are unordered reactant sets with no native chain order, so whole-H048 transfer is not justified.")
    wave_status=[]
    for index in range(14,20):
        parent="american_c3_short_train_%03d"%index
        for stage,dirname in [("simulate","simulation"),("infer","inference")]:
            state=cpu_stage("experiments/results/American_C3_short_train2mb_wave_v2/"+parent+"/"+dirname,
                "research/reviews/American_C3_short_train2mb_wave_980_source_v1/admitted/"+parent+"/"+stage)
            wave_status.append(dict(parent_id=parent,stage=stage,**state))
    recovery_path="research/reviews/C3_wave014_capacity_recovery_20261015_v1/root014_recovery_exec_launch_identity_actual_v1.json"
    recovery_launch=maybe_read(recovery_path)
    recovery=dict(status="recovery launch not recorded")
    if recovery_launch:
        rp=known_process(recovery_launch["remote_supervisor_pid"])
        recovery.update(status="recovery supervisor running" if rp["present"] and rp.get("start_ticks")==recovery_launch["remote_supervisor_start_ticks"] else "recovery launch recorded; actual stage terminal evidence required",
            launch_ref=ref(recovery_path),exec_session_id=recovery_launch.get("exec_session_id"),
            process_observation=rp,original_failed_pair_ref=ref("research/reviews/American_C3_short_train2mb_wave_980_source_v1/root_exec_session50286_terminal_actual_v1.json"))
    recovery_stage=next(r for r in wave_status if r["parent_id"]=="american_c3_short_train_014" and r["stage"]=="infer")
    if recovery_stage["completed_and_wait0_verified"]:
        recovery.update(status="inference recovery completed with actual stage wait0",
            stage_completion_ref=recovery_stage["completion_ref"],stage_actual_wait_ref=recovery_stage["actual_wait_ref"])
    recovery_terminal_path="research/reviews/C3_wave014_capacity_recovery_20261015_v1/root014_recovery_exec_terminal_actual_v1.json"
    recovery_terminal=maybe_read(recovery_terminal_path)
    if recovery_terminal and recovery_terminal.get("actual_exec_session_terminal") is True:
        recovery.update(root_exec_terminal_ref=ref(recovery_terminal_path),
            actual_root_exec_session_exit_code=recovery_terminal["actual_exec_session_exit_code"],
            remote_direct_Popen_wait_claimed=False)
        recovery["status"]+="; actual outer exec session76279 exit"+str(recovery_terminal["actual_exec_session_exit_code"])
    wave_text="TRAIN000–013 original 2 Mb preparations completed. "+recovery["status"]+" for TRAIN014; successful TRAIN015 reused; original session50286 failure preserved. "
    wave_text+="; ".join(r["parent_id"][-3:]+"/"+r["stage"]+": "+r["status"] for r in wave_status if r["parent_id"].endswith(("014","015")))
    close014015_path="research/reviews/C3_wave014_capacity_recovery_20261015_v1/root_actual_close_loop_terminal_v1.json"
    close014015=maybe_read(close014015_path)
    pair1617_path="research/reviews/American_C3_short_train2mb_wave_980_source_v1/root_pair016_017_exec_launch_identity_actual_v1.json"
    pair1617=maybe_read(pair1617_path)
    if close014015 and close014015.get("actual_exec_session_terminal") is True and close014015.get("actual_SSH_propagated_returncode")==0 and len(close014015.get("actual_four_completed_scope_close_receipts",[]))==4 and all(ref(r["path"])==r for r in close014015["actual_four_completed_scope_close_receipts"]):
        wave_text+=". All four014/015 scope closes completed, actual root exec19862 exit0."
        if pair1617:
            pair_observation=known_process(pair1617["remote_supervisor_pid"])
            pair_present=pair_observation["present"] and pair_observation.get("start_ticks")==pair1617["remote_supervisor_start_ticks"]
            pair_completed=all(r["completed_and_wait0_verified"] for r in wave_status if r["parent_id"].endswith(("016","017")))
            wave_text+=" TRAIN016/017 parallel preparation "+("completed with genuine per-stage wait0" if pair_completed else "running" if pair_present else "launched; terminal pending")+", root exec"+str(pair1617["exec_session_id"])+"."
    for pair_start in (16,18):
        close_paths=["research/reviews/American_C3_short_train2mb_wave_980_close_source_v1/closed_%03d_%s_v1/close_receipt.json"%(i,stage) for i in (pair_start,pair_start+1) for stage in ("simulate","infer")]
        closed=sum(maybe_read(p) is not None for p in close_paths)
        if closed:wave_text+=" Original TRAIN%03d/%03d completed-scope closes %d/4, preserving all files."%(pair_start,pair_start+1,closed)
    last_pair=[r for r in wave_status if r["parent_id"].endswith(("018","019"))]
    if any(r["status"]!="not admitted" for r in last_pair):
        wave_text+=" "+"; ".join(r["parent_id"][-3:]+"/"+r["stage"]+": "+r["status"] for r in last_pair)
    else:wave_text+=" TRAIN018/019 pending original016/017 scope closure."
    pair1819_terminal="research/reviews/American_C3_short_train2mb_wave_980_source_v1/root_pair018_019_exec_terminal_actual_v1.json"
    if maybe_read(pair1819_terminal):
        wave_text+=" Actual parent supervisor session2356 returned1. TRAIN018 simulation/inference andTRAIN019 simulation completed with genuine wait0. TRAIN019 inference has partial pre-admission metadata, recovery under investigation. Original failure and completed artifacts preserved."
    recovered019="research/reviews/American_C3_short_train2mb_wave_980_source_v1/train019_infer_recovery_v1/root_finish_remote_direct_wait_actual_v1.json"
    recovered019_wait=maybe_read(recovered019)
    if recovered019_wait and recovered019_wait.get("actual_remote_direct_Popen_wait_observed") is True and recovered019_wait.get("actual_remote_direct_Popen_returncode")==0:
        wave_text="All20/20 original inferred2Mb TRAIN inputs completed with genuine workerwait0/drain. Final019 recovery has genuine durable remotePopenwait0 and helpercompletion0. Original parent terminal1 and independentSSH255 transportfailure remain recorded. Exact focalplan readiness is reported separately."
    wave_text+=" At most two wave parents."
    table1_audit_path="research/reviews/ARGformer_Table1_downstream_resolution_20261016_v1/RESOLUTION.md"
    table1_fidelity="ARGformer Table1 reports PCA10/clustering and nearest-reference retrieval as separate classifiers. The exact PCA cluster/mapping evaluator remains unavailable. We retain the unchanged author-release head_norm/top1 retrieval as a disclosed common-data reconstruction. No native ancestry score is available yet."
    stamp=datetime.now(timezone.utc).isoformat()
    h049=next((r for r in read("research/experiment_decisions.json")["decisions"] if r["id"]=="H049"),None)
    h049_base="research/reviews/H049_order2_cut_memory_root_source_20261015_v1"
    h049_done=maybe_read("experiments/results/H049_CPU_order2_cut_memory_prepare_v1/completed.json")
    h049_wait=maybe_read(h049_base+"/admitted_prepare/actual_wait.json")
    h049_launch=maybe_read(h049_base+"/admitted_prepare/launch.json") or {}
    h049_prepare_verified=bool(h049_done and h049_wait and h049_wait.get("actual_direct_launcher_wait") is True and h049_wait.get("returncode")==0 and h049_wait.get("owned_group_drained") is True and h049_wait.get("error") is None and h049_wait.get("completed_ref")==ref("experiments/results/H049_CPU_order2_cut_memory_prepare_v1/completed.json"))
    h049_prepare_status=("CPU compact preparation completed with actual wait0 and drain" if h049_prepare_verified else "first CPU preparation failed before any recipe; codec correction sealed. Bounded-v3 admission failed at protocol metadata validation before any reservation/job; compatibility repair active" if h049_wait and h049_wait.get("returncode")==1 else "preparation terminal evidence requires review" if h049_wait else "CPU preparation launched" if h049_launch else "root supervisor launched; scientific child startup pending" if maybe_read(h049_base+"/root_prepare_exec_launch_identity_actual_v1.json") else "CPU preparation admitted; launch pending" if maybe_read(h049_base+"/admitted_prepare/admitted.json") else "root preparation caps approved; admission pending" if maybe_read(h049_base+"/root_prepare_resource_review_actual_v1.json") else "source ready; root preparation-cap review pending")
    h049v3="research/reviews/H049_bounded_prepare_root_source_20261015_v3/admitted_prepare"
    h049v3_launch=maybe_read(h049v3+"/launch.json")
    h049v3_admit=maybe_read(h049v3+"/admitted.json")
    h049v3_wait=maybe_read(h049v3+"/actual_wait.json")
    h049v3_done=maybe_read("experiments/results/H049_CPU_order2_cut_memory_prepare_bounded_v3/completed.json")
    if h049v3_admit:
        h049_prepare_status=("corrected bounded CPU preparation completed with actual wait0/drain" if h049v3_wait and h049v3_done and h049v3_wait.get("returncode")==0 and h049v3_wait.get("owned_group_drained") is True else
            "corrected bounded CPU preparation terminal requires review" if h049v3_wait else
            "corrected bounded CPU preparation launched; original failed attempt preserved" if h049v3_launch else
            "corrected bounded CPU preparation admitted, startup pending; inactive-config admission mistake preserved and corrected")
    h049v4="research/reviews/H049_bounded_actual_metadata_allocation_repair_20261015_v1/admitted_prepare"
    h049v4_launch=maybe_read(h049v4+"/launch.json")
    h049v4_admit=maybe_read(h049v4+"/admitted.json")
    h049v4_wait=maybe_read(h049v4+"/actual_wait.json")
    h049v4_done=maybe_read("experiments/results/H049_CPU_order2_cut_memory_prepare_bounded_v4/completed.json")
    if h049v3_wait and h049v3_wait.get("returncode")==1:
        h049_prepare_status="bounded-v3 CPU preparation failed before any parent recipe at the NFS metadata-allocation bound; no fit or accuracy"
    if h049v4_admit:
        h049_prepare_verified=bool(h049v4_wait and h049v4_done and h049v4_wait.get("returncode")==0 and h049v4_wait.get("owned_group_drained") is True and h049v4_wait.get("error") is None and h049v4_wait.get("completed_ref")==ref("experiments/results/H049_CPU_order2_cut_memory_prepare_bounded_v4/completed.json"))
        h049_prepare_status=("fresh bounded-v4 CPU preparation completed with genuine wait0/drain" if h049_prepare_verified else
            "fresh bounded-v4 CPU preparation failed; original artifacts preserved" if h049v4_wait and h049v4_wait.get("returncode")!=0 else
            "fresh bounded-v4 CPU preparation launched; corrected allocation allowance, unchanged science" if h049v4_launch else
            "fresh bounded-v4 CPU preparation admitted, launch pending; prior failure retained")
    elif maybe_read("research/reviews/H049_bounded_actual_metadata_allocation_repair_20261015_v1/root_source_cap_approval_v1.json"):
        h049_prepare_status+=". Fresh bounded-v4 allocation repair approved, admission pending"
    h049v5="research/reviews/H049_bounded_directory_metadata_repair_20261015_v1/admitted_prepare"
    h049v5_admit=maybe_read(h049v5+"/admitted.json")
    h049v5_launch=maybe_read(h049v5+"/launch.json")
    h049v5_wait=maybe_read(h049v5+"/actual_wait.json")
    h049v5_done=maybe_read("experiments/results/H049_CPU_order2_cut_memory_prepare_bounded_v5/completed.json")
    if h049v5_admit:
        h049_prepare_verified=bool(h049v5_wait and h049v5_done and h049v5_wait.get("returncode")==0 and h049v5_wait.get("owned_group_drained") is True and h049v5_wait.get("error") is None and h049v5_wait.get("completed_ref")==ref("experiments/results/H049_CPU_order2_cut_memory_prepare_bounded_v5/completed.json"))
        h049_prepare_status=("bounded-v5 CPU preparation completed with genuine wait0/drain" if h049_prepare_verified else
            "bounded-v5 CPU preparation failed; original artifacts and charge preserved" if h049v5_wait and h049v5_wait.get("returncode")!=0 else
            "bounded-v5 CPU preparation launched with corrected directory allowance; prior v4 failure after38/40 logged preparations preserved" if h049v5_launch else
            "bounded-v5 CPU preparation admitted, launch pending; unchanged science and full256MiB cap")
    c3_failed_output="experiments/results/C3_GNN_prefetch_current980_train_v1/failed_training.json"
    c3_wait_path="research/reviews/C3_GNN_prefetch_current980_root_20261015_v1/admitted_train_v1/actual_wait.json"
    c3_failed=maybe_read(c3_failed_output)
    c3_wait=maybe_read(c3_wait_path)
    c3_status=("TRAIN failed before the first optimizer update on all-isolated samples at chromosome flanks. Full20-parent observation inventory found only boundary forests. Scoped encoder repair in progress, unchanged full600-update budget" if c3_failed and c3_wait and c3_wait.get("returncode")==1 else "full600-update training route ready, ancestry score pending")
    c3_execution=dict(status=c3_status,actual_optimizer_updates=0 if c3_failed else None,ancestry_score=None,failed_training_ref=ref(c3_failed_output) if c3_failed else None,actual_wait_ref=ref(c3_wait_path) if c3_wait else None)
    c3_new_packet="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2"
    c3_new_admit=maybe_read(c3_new_packet+"/admitted_train_v1/admitted.json")
    c3_new_launch=maybe_read(c3_new_packet+"/root_train_exec_launch_actual_v1.json")
    if c3_new_admit:
        c3_status="Isolated-boundary encoder repair installed. Fresh full600-update campaign admitted for GPU4; prior before-update failure and charge preserved"
        if c3_new_launch:
            c3_status+=". Root launch issued, completed optimizer updates not yet verified"
        c3_execution.update(status=c3_status,actual_optimizer_updates=None,prior_failed_campaign_verified_updates=0,new_admitted_ref=ref(c3_new_packet+"/admitted_train_v1/admitted.json"),new_root_launch_ref=ref(c3_new_packet+"/root_train_exec_launch_actual_v1.json") if c3_new_launch else None)
    elif maybe_read(c3_new_packet+"/root_source_resource_approval_v1.json"):
        c3_status="Isolated-boundary encoder repair installed and root source/caps approved; fresh full600-update GPU4 admission in progress, prior failure preserved"
        c3_execution["status"]=c3_status
    c3_start_path="experiments/results/C3_GNN_prefetch_isolated980_train_v1/started.json"
    c3_start=maybe_read(c3_start_path)
    if c3_start:
        runtime=c3_start.get("runtime",{})
        title_verified=runtime.get("actual_process_title")=="AI Auto Research Pipeline Test" and runtime.get("nvtop",{}).get("actual_pid_title_and_index_verified") is True
        c3_observed=known_process(runtime.get("pid",-1))
        c3_live=bool(c3_observed["present"] and c3_observed.get("start_ticks")==476966238 and c3_observed["required_title_present"])
        barrier_path="experiments/results/C3_GNN_prefetch_isolated980_train_v1/TRAIN_observation_barrier.json"
        target_path="experiments/results/C3_GNN_prefetch_isolated980_train_v1/TRAIN_external_target_joins.json"
        barrier_ready=(ROOT/barrier_path).is_file() and (ROOT/target_path).is_file()
        current_wait=maybe_read(c3_new_packet+"/admitted_train_v1/actual_wait.json")
        current_done=maybe_read("experiments/results/C3_GNN_prefetch_isolated980_train_v1/completed.json")
        if current_wait:
            c3_status=("Full600-update TRAIN completion recorded; exact counts/evaluation require review" if current_done and current_wait.get("returncode")==0 else "Repaired campaign terminal; actual wait and partial artifacts require review")
        elif c3_live:
            c3_status=("Full600-update campaign active on GPU4. All20 observation seals and TRAIN target joins completed; optimizer update counts not yet verified" if barrier_ready else "Full600-update repaired campaign started CUDA on GPU4; observation loading active, optimizer update counts not yet verified")
        else:
            c3_status="Recorded GPU4 runtime is no longer observed; actual wait and optimizer counts require review"
        c3_execution.update(status=c3_status,started_ref=ref(c3_start_path),process_title_verified_in_NVTOP=title_verified,process_observation=c3_observed,worker_identity_present=c3_live,
            all20_observation_barrier_complete=barrier_ready,observation_barrier_ref=ref(barrier_path) if barrier_ready else None,TRAIN_target_manifest_ref=ref(target_path) if barrier_ready else None)
    c3_device="GPU4" if c3_execution.get("worker_identity_present") else "GPU4 planned"
    flare_val_seal["outer_tool_terminal_or_final_guard_success_claimed"]=False
    if flare_val_seal["completed_and_wait0_verified"]:
        flare_val_seal["status"]="prediction worker completed; direct wait0 and owned-group drain verified, outer tool terminal/final guard unconfirmed"
    val002_attempt_path="research/reviews/American_C3_remaining_VAL_namespace_memo_root_20261015_v1/root_val002_exec_attempt_actual_v1.json"
    if maybe_read(val002_attempt_path):
        correction=maybe_read("research/reviews/American_C3_remaining_VAL_namespace_memo_root_20261015_v1/root_val002_environment_correction_active_v1.json")
        remaining_val_status+=(". Original VAL002 pre-admission import failure preserved; corrected project-root import environment authorized and root launch issued" if correction else ". VAL002 parent launcher returned1 before useful output; preserved attempt, early cause under investigation, no replay")
    val002_infer_attempt_path="research/reviews/American_C3_remaining_VAL_namespace_memo_root_20261015_v1/root_val002_parent_terminal_actual_v1.json"
    if maybe_read(val002_infer_attempt_path):
        recovery_path=("research/reviews/VAL002_infer_unconsumed_admission_recovery_20261015_v2/root_recovery_admit_actual_terminal_v1.json" if (ROOT/"research/reviews/VAL002_infer_unconsumed_admission_recovery_20261015_v2/root_recovery_admit_actual_terminal_v1.json").is_file() else "research/reviews/VAL002_infer_unconsumed_admission_recovery_20261015_v1/root_recovery_admit_actual_terminal_v1.json")
        recovery=maybe_read(recovery_path)
        remaining_val_status+=(". Original parent session92429 returned1 after successful simulate/close; fresh same-science inference admission recovered, original failure preserved" if recovery and recovery.get("returncode")==0 else ". Parent session92429 returned1 after successful simulate/close, next inference admission stopped before a worker; capacity recovery in progress")
    h050=next((d for d in read("research/experiment_decisions.json")["decisions"] if d["id"]=="H050"),None)
    h050_binding_path="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/bound_root_v1/canonical_metadata_binding.json"
    h050_bound=maybe_read(h050_binding_path)
    h050_execution_status=("Root-bound canonical config and full caps reviewed. GPU admission/launch pending; current route requires an exclusive lease" if h050_bound else "Registered; source integration pending")
    shared_root="research/reviews/GPU45_C3cache_H050_migration_gate_successor_20261016_v2"
    shared_snapshot_path=shared_root+"/root_live_status_latest_v1.json"
    shared_snapshot=maybe_read(shared_snapshot_path)
    shared_c3_admitted=maybe_read(shared_root+"/admitted_C3_shared_train_v1/admitted.json")
    shared_h050_admitted_path="research/reviews/H050_shared_admission_source_metadata_closure_fix_20261016_v1/admitted_H050_shared_train_v2/admitted.json"
    shared_h050_admitted=maybe_read(shared_h050_admitted_path)
    old_c3_terminal_path="research/reviews/C3_original_handover_root_20261016_v1/original_tool_terminal_actual_v1.json"
    old_c3_terminal=maybe_read(old_c3_terminal_path)
    if old_c3_terminal:
        c3_status="Original slow campaign stopped through its exact supervisor; actual tool exit1 and launcher wait-15 retained. Original optimizer count unknown. Cached replacement launch pending"
        c3_execution.update(original_campaign_terminal_ref=ref(old_c3_terminal_path),original_optimizer_updates=None,
            original_campaign_stopped=True,original_campaign_completion_claimed=False)
    if shared_c3_admitted:
        c3_status="Cached full600-update TRAIN admitted for shared GPU4 execution; runtime launch pending"
        c3_execution.update(shared_admitted_ref=ref(shared_root+"/admitted_C3_shared_train_v1/admitted.json"))
        c3_device="GPU4"
    if shared_h050_admitted:
        h050_execution_status="Four-arm TRAIN admitted for shared GPU4 execution; runtime launch pending"
    if shared_snapshot and shared_snapshot.get("schema")=="root_GPU4_shared_live_status_v1":
        c3_job=shared_snapshot.get("jobs",{}).get("C3-GNN-cachev3-shared-TRAIN-v1")
        h050_job=shared_snapshot.get("jobs",{}).get("H050-direct-readout-shared-TRAIN-v2")
        if c3_job:
            state=c3_job["state"]
            lower=c3_job.get("verified_optimizer_updates_lower_bound")
            c3_status="Cached full600-update TRAIN "+state+" on shared GPU4. "+("Successful optimizer-update lower bound: "+str(lower)+"/600" if lower is not None else "Optimizer count not yet verified")
            c3_execution.update(shared_live_status_ref=ref(shared_snapshot_path),shared_logical_job=c3_job,
                actual_optimizer_updates_lower_bound=lower,original_optimizer_count_unknown_not_zero=True)
            c3_device="GPU4"
        if h050_job:
            h050_execution_status="Four-arm TRAIN "+h050_job["state"]+" on shared GPU4. "+("Verified optimizer-update lower bounds: "+str(h050_job["updates_per_arm"]) if h050_job.get("updates_per_arm") else "Optimizer counts not yet verified")
    if shared_snapshot and shared_snapshot.get("controller_status"):
        c3_status+=". Shared controller: "+shared_snapshot["controller_status"]
        h050_execution_status+=". Shared controller: "+shared_snapshot["controller_status"]
    c3_execution["status"]=c3_status
    current_gpu_policy_path="research/reviews/GPU0_only_user_transition_root_20261016_v1/GPU0_only_policy.root_active_v1.json"
    current_gpu_policy=maybe_read(current_gpu_policy_path)
    gpu0_snapshot=None
    if current_gpu_policy and current_gpu_policy.get("allowed_physical_GPUs")==[0]:
        native_status="Historical GPU5 engineering TRAIN000 stopped at latest user request. Actual native/entry/launcher waits-15, roottool48073exit1 with expired-marker postdrain accounting error. Saved ep4-ba206000 checkpoint preserved, metadata only; full fiveepochs/savedAdam count and ancestry score unverified. Known own roots and GPU4/5contexts absent. Future experiments use onlyGPU0."
        native_execution.update(current_status=native_status,user_requested_stop=True,running=False,
            actual_user_stop_ref=ref("research/reviews/GPU0_only_user_transition_root_20261016_v1/actual_stop_and_absence.root_v1.json"),new_GPU0_execution_pending=True)
        native_device="GPU0 pending"
        c3_device="GPU0 pending"
        c3_status="Historical GPU4 cache run failed with mmap allocation error before a verified optimizer update. Private-owned cache repair passed its CPU fixture. Fresh GPU0 full600-update run pending."
        h050_execution_status="Historical GPU4 run interrupted by GPU0-only instruction after genuine first scheduled FP32 steps in all four arms. Fresh GPU0 full960-updates-per-arm run pending. No validation score."
        c3_execution.update(status=c3_status,running=False,new_GPU0_execution_pending=True)
        gpu0_snapshot_path="research/reviews/GPU0_shared_root_integration_20261016_v1/runtime_status.root_latest.json"
        gpu0_snapshot=maybe_read(gpu0_snapshot_path)
        if gpu0_snapshot:
            for kind,entry in gpu0_snapshot.get("jobs",{}).items():
                if kind=="H050":h050_execution_status=entry["summary"]
                elif kind=="native_ARGformer":
                    native_status=entry["summary"]
                    native_device="GPU0"
                    native_execution.update(current_status=native_status,GPU0_runtime_snapshot_ref=ref(gpu0_snapshot_path),running=entry.get("state")=="running")
                elif kind=="C3":
                    c3_status=entry["summary"]
                    c3_execution.update(status=c3_status,GPU0_runtime_snapshot_ref=ref(gpu0_snapshot_path),running=entry.get("state")=="running")
                    c3_device="GPU0"
    tgb_results_path="experiments/results/H052_TGB_trade_causal_native_baselines_v1/results.json"
    tgb_terminal_path="research/reviews/TGB_trade_causal_baselines_root_20261016_v1/actual_tool_terminal.root_v1.json"
    tgb_results=maybe_read(tgb_results_path);tgb_terminal=maybe_read(tgb_terminal_path)
    tgb_ready=bool(tgb_results and tgb_terminal and tgb_terminal.get("actual_exit_code")==0
        and tgb_terminal.get("actual_terminal_consumed") is True and tgb_terminal.get("result_ref")==ref(tgb_results_path))
    tgb_status="Native strictly causal baselines pending"
    tgb_lines=["", "## TGB annual trade causal diagnostic", ""]
    if tgb_ready:
        tgb_status="Native persistence TEST NDCG@10 0.8469±0.0144, EMA-window7 0.8090±0.0049. SD across3 serially dependent target years2014–2016,675country-years. Unchanged native classes with disclosed forecast-before-current-year-edge release and1986initialization. No graph-model score or exact-paper reproduction."
        tgb_lines += ["255country catalogue,468245edges,1986–2016. Strictly prior-year observations, all255source forecasts sealed before target joining. Fixed author persistence/EMA classes. No model fitting or hyperparameter selection.", "", "| Method | TRAIN NDCG@10 ± year SD | VAL NDCG@10 ± year SD | TEST NDCG@10 ± year SD |", "|---|---:|---:|---:|"]
        for name,arm in tgb_results["results"].items():
            values=[f"{arm['splits'][split]['mean_ndcg_at10']:.4f} ± {arm['splits'][split]['sample_SD_over_years']:.4f}" for split in ("TRAIN","VAL","TEST")]
            tgb_lines.append("| "+name+" | "+" | ".join(values)+" |")
        tgb_lines += ["", "TRAIN1987–2009/23years, VAL2010–2013/4years, TEST2014–2016/3years and675queriedcountry-years. SD is descriptive across serially dependent years, not training seeds. This causal diagnostic adapts scheduler and1986initialization and is distinct from the published leaderboard. Our ordered graph model remains pending.", "", "[All per-year results](../"+tgb_results_path+") · [Frozen protocol](../research/reviews/TGB_trade_causal_baselines_root_20261016_v1/protocol.root_frozen_v1.json)", ""]
    else:tgb_lines += [tgb_status, ""]
    support_path="research/reviews/TGB_trade_H051_data_integration_root_20261016_v1/actual_TRAIN_operator_support.root_v1.json"
    support=maybe_read(support_path)
    ordered_support_note=""
    if support and support.get("schema")=="H053_actual_TRAIN_observation_operator_support_v1":
        assert support["target_label_member_opened"] is False and support["GPU_context_created"] is False
        ordered_support_note=("**Ordered-tree mechanism status:** The frozen TRAIN-only trade audit found "
            +str(support["cut_change_count"])+" partition changes and "
            +str(support["nonzero_conditional_increment_years"])+" nonzero conditional-innovation years. "
            "This establishes active input support, without a predictive-accuracy result. "
            "[Actual audit]("+support_path+"). "
            "A [fresh independent literature review](research/reviews/H051_independent_priority_audit_20261016_v2/REPORT.md) "
            "identifies established conditional-mean and higher-order constructions. Generic operator novelty remains unestablished.")
        tgb_lines += [ordered_support_note.replace("](research/","](../research/"), ""]
    h055_root="research/reviews/H055_root_actual_CPU_20261016_v1"
    h055_output="experiments/results/H055_Danish_genotype_corroboration_CPU_v1"
    h055_launch=maybe_read(h055_root+"/actual_worker_launch_observation.root_v1.json")
    h055_terminal=maybe_read(h055_root+"/actual_supervisor_tool_terminal.root_v1.json")
    h055_complete=maybe_read(h055_output+"/completed.json")
    h055_failure=maybe_read(h055_output+"/failed.json")
    h055_summary=None
    h055_status="Registered fixed-cohort Danish genotype corroboration. Acquisition pending."
    h055_live=False
    if h055_launch:
        observed=known_process(h055_launch["pid"])
        h055_live=bool(observed["present"] and observed.get("start_ticks")==h055_launch["start_ticks"])
        h055_status=("CPU acquisition running for the fixed ten Danish queries and eight proxies over chr20/21/22 [20,30)Mb. GP-only analysis, no neural-model or ancestry-accuracy result yet."
            if h055_live else "CPU caller terminal requires root review. No completed biological estimate claimed.")
    if h055_terminal and h055_terminal.get("actual_terminal_consumed") is True:
        if h055_terminal.get("actual_exit_code")==0 and h055_complete:
            assert h055_terminal["completed_ref"]==ref(h055_output+"/completed.json")
            assert h055_terminal["summary_ref"]==ref(h055_output+"/summary.json")
            h055_summary=read(h055_output+"/summary.json")
            h055_status=("Completed GP-only Danish corroboration over all three fixed windows, "
                +str(h055_summary["primary_SNPs"])+" common-support SNPs. Pooled f4: "
                +", ".join(f"{x:.6g}" for x in h055_summary["pooled_equal_SNP_means"])
                +". Descriptive genotype affinity only, without independent ancestry truth or model accuracy.")
        elif h055_terminal.get("actual_exit_code")!=0:
            h055_status="CPU acquisition failed. Partial outputs preserved. No completed biological estimate."
    if h055_summary and not any(h055_summary["positive_direction_observed"]):
        h055_status+=" Neither prespecified positive-direction hypothesis was supported."
    h055_evidence=dict(status=h055_status,running=h055_live,completed_summary=h055_summary,
        actual_wait0=bool(h055_terminal and h055_terminal.get("actual_terminal_consumed") is True and h055_terminal.get("actual_exit_code")==0),
        launch_ref=ref(h055_root+"/actual_worker_launch_observation.root_v1.json") if h055_launch else None,
        terminal_ref=ref(h055_root+"/actual_supervisor_tool_terminal.root_v1.json") if h055_terminal else None,
        protocol_ref=ref("research/reviews/Danish_genotype_corroboration_source_20261016_v2/protocol.source_frozen.inactive.v2.json"),
        genotype_corroboration_is_not_model_validation=True,model_accuracy=None)
    result=dict(schema="ARG_GNN_current_scientific_scoreboard_v1",updated_utc=stamp,
        H055_biological_corroboration=h055_evidence,
        GPU0_current_execution=gpu0_snapshot,
        American_C3_GNN_execution=c3_execution,
        GPU4_shared_execution=shared_snapshot,
        current_GPU_execution_policy=current_gpu_policy,
        TGB_trade_causal_diagnostic=dict(status=tgb_status,complete=tgb_ready,results=tgb_results if tgb_ready else None,result_ref=ref(tgb_results_path) if tgb_ready else None,terminal_ref=ref(tgb_terminal_path) if tgb_ready else None,our_GNN_score=None),
        H050_registered_candidate=dict(status=h050["status"],execution_status=h050_execution_status,canonical_metadata_binding_ref=ref(h050_binding_path) if h050_bound else None,training_budget="4arms x30passes x32parents =3840updates",actual_optimizer_updates=None,verified_optimizer_update_lower_bounds=(gpu0_snapshot.get("jobs",{}).get("H050",{}).get("optimizer_steps_verified") if gpu0_snapshot else (shared_snapshot.get("jobs",{}).get("H050-direct-readout-shared-TRAIN-v2",{}).get("updates_per_arm") if shared_snapshot else None)),accuracy=None,parameters_per_arm=h050["parameters_per_arm"],VAL_products_required_before_truth=192,source_handoff_ref=h050["source_handoff_ref"],utility_criteria=h050["prospective_utility_criteria"]) if h050 else None,
        H049_registered_candidate=dict(status=h049["status"],parameters_per_arm=32446,
            arms=h049["execution_budget"]["arms"],training_budget="30passes/960updates per arm",
            actual_optimizer_updates=None,actual_optimizer_updates_status="no_training_launch_or_verified_step_receipt",accuracy=None,preparation_status=h049_prepare_status,preparation_completion_wait0_verified=h049_prepare_verified,registered_criteria=h049["prospective_utility_criteria"]) if h049 else None,
        dataset=dict(name="European C4 shared development",length_bp=2000000,haplotypes=952,queries=686,
            TRAIN_simulations=32,VAL_simulations=8,SD_unit="eight VAL simulations, one optimizer seed per model",
            original_paper_reproduction=False,untouched_outer_test_completed=False),
        models=models,live_jobs=jobs,GST_TRAIN1000_seed0_VALID_AP=gst,
        H046_completed_comparison=dict(summary_ref=ref(h046_summary),paired_effects=h046["paired_effects"],status=h046_outcome) if h046_ready else None,
        original_American_C3=dict(our_GNN_score=None,native_ARGformer_score=None,native_FLARE_score=None,
            status=val_status+" Native ARGformer, FLARE and our C3 adapter proceed in parallel.",shared_VAL_preparation_status=val_status,native_ARGformer_execution_status=native_status),
        real_data=dict(predictions_available=empirical_verified,PCA_available=empirical_verified,products_written=empirical_products,completion_verified=empirical_verified,observed_bank_prepared=empirical is not None,
            prepared_bank_completion_ref=ref(empirical_path) if empirical else None,status=real_status),
        adapted_ARGformer_pilot=dict(source_ref=ref(pilot_source),rows=pilot_rows,original_paper_reproduction=False),
        H048_registered_candidate=dict(status=h048_status,trained=True if h048_job["updates_per_arm"] else None,accuracy={a:r["cohorts"][PRIMARY]["accuracy"] for a,r in h048_result["arms"].items()} if h048_ready else None,
            runtime_started=h048_job["started_ref"] is not None,verified_update_lower_bounds=h048_job["updates_per_arm"],evaluation_completed=h048_ready,completed_comparison_ref=ref(h048_summary) if h048_ready else None,validation_scores=h048_scores,
            launch_ref=ref("research/reviews/H048_GPU45_grid_root_source_v1/admitted_train/launch.json"),
            registration_ref=ref("research/experiment_decisions.json")) if h048 else None,
        novelty_and_cross_domain=dict(status=novelty_status,report_ref=ref(novelty_path),evidence_ref=ref(novelty_evidence_path),
            whole_H048_cross_domain_transfer_validated=False),
        GST_status=dict(status=gst_status,full_TRAIN_count_ref=ref("research/reviews/GST_full_TRAIN_count_root_v1/counts.json"),
            full_TRAIN_source_checkpoint_ref=ref("research/reviews/GST_full_TRAIN_CPU_source_v1/checkpoint.json")),
        CPU_stage_status=dict(VAL_simulation=val_simulation,VAL_inference=val_inference,VAL_native_paths=val_native_paths,
            FLARE_author_build=flare_build,FLARE_TRAIN_conversion=flare_conversion,FLARE_TRAIN_native=flare_native,FLARE_TRAIN_seal=flare_seal,
            FLARE_VAL_conversion=flare_val_conversion,FLARE_VAL_native=flare_val_native,FLARE_VAL_seal=flare_val_seal,FLARE_VAL_original_deliberate_stop_ref=ref(flare_val_old_stop),C3_original_VAL000_GNN_plan=gnn_VAL000_plan),
        American_remaining_VAL_preparation=dict(status=remaining_val_status,phases=remaining_val_phases),
        American_short_TRAIN_remaining_preparation=wave_status,
        American_short_TRAIN014_recovery=recovery,C3_focal_plan_preparation=focal_preparation,
        native_ARGformer_execution=native_execution,native_ARGformer_counts_execution=counts_execution,native_ARGformer_VAL000_counts=val000_native_counts,ARGformer_Table1_fidelity=dict(status=table1_fidelity,report_ref=ref(table1_audit_path)),VAL000_FLARE_target_binding=val_flare_binding,
        FLARE_native_completion_close=dict(receipt_ref=ref(flare_close_path),credit_bytes=flare_close["credit_bytes"],retained_bytes=flare_close["retained_bytes"]) if flare_close else None,
        native_FLARE_build_status=flare_status,
        source_only_or_engineering_checks_are_not_training_results=True)
    (RESEARCH/"current_scoreboard.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=["# Current scientific scoreboard","",f"Updated UTC: {stamp}","",
        "[All measured cohort details](#detailed-cohort-metrics) · [Complete historical receipt inventory](all_recorded_metric_receipts.md) · [Fresh H048 novelty/cross-domain audit](../"+novelty_path+")","",
        "## European C4 shared development","",
        "32 TRAIN and eight VAL simulations, 2 Mb, 952 input haplotypes and 686 query haplotypes. Scores use common physical-span support. SD is across eight VAL simulations, with one optimizer seed. This is development evidence. Native method input representations, objectives and training schedules are preserved and disclosed. Architecture-only attribution and original-paper reproduction are unproven.","",
        "| Setup | Present accuracy % ± SD | Present NLL | Present boundary F1 % ± SD | Training |",
        "|---|---:|---:|---:|---|"]
    for r in models:
        q=r["cohorts"][PRIMARY]
        lines.append("| "+r["name"]+" | "+fmt(q.get("accuracy"),True)+" | "+fmt(q.get("nll"))+" | "+fmt(q.get("boundary_f1"),True)+" | "+r["training"]+" |")
    lines+=["","## Selected training lineage","",
        "| Setup | Completed cumulative updates | Selected pass | Selected cumulative updates |",
        "|---|---:|---:|---:|"]
    for r in models:
        lineage=r.get("training_budget_lineage",{})
        if lineage:
            lines.append("| "+r["name"]+" | "+str(lineage.get("cumulative_model_lineage_updates","—"))+" | "+str(lineage.get("selected_pass","—"))+" | "+str(lineage.get("selected_model_lineage_updates","—"))+" |")
    lines+=["","## All measured cohort accuracies","",
        "| Setup | "+" | ".join(LABELS)+" |","|---|"+"---:|"*len(COHORTS)]
    for r in models:
        lines.append("| "+r["name"]+" | "+" | ".join(fmt(r["cohorts"].get(c,{}).get("accuracy"),True) for c in COHORTS)+" |")
    lines+=["","BronzeAge is not asserted identical to the paper's Post-Neolithic cohort. A dash means unavailable or undefined in the current aggregate, not zero. An n below8 identifies a metric defined for fewer simulations.","",
        "## Detailed cohort metrics","",
        "| Setup / cohort | Accuracy % ± SD | NLL ± SD | Macro F1 % ± SD | Boundary F1 % ± SD | Brier ± SD | ECE ± SD |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in models:
        for c,label in zip(COHORTS,LABELS):
            q=r["cohorts"].get(c,{})
            lines.append("| "+r["name"]+" / "+label+" | "+" | ".join(fmt(q.get(k),pct) for k,pct in [("accuracy",True),("nll",False),("macro_f1",True),("boundary_f1",True),("brier",False),("calibration_ece",False)])+" |")
    lines+=["","## Additional available cohort metrics","",
        "| Setup / cohort | Balanced accuracy % ± SD | Boundary precision % ± SD | Boundary recall % ± SD | Dosage MSE ± SD | Dosage MAE ± SD |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in models:
        for c,label in zip(COHORTS,LABELS):
            q=r["cohorts"].get(c,{})
            lines.append("| "+r["name"]+" / "+label+" | "+" | ".join(fmt(q.get(k),pct) for k,pct in [
                ("balanced_accuracy",True),("boundary_precision",True),("boundary_recall",True),
                ("dosage_mse_per_diploid_bp_class",False),("dosage_mae_per_diploid_bp_class",False)])+" |")
    lines+=tgb_lines
    lines += ["", "## Danish registered genotype corroboration", "", h055_status, ""]
    if h055_summary:
        lines += ["| Support | Common13 SNPs | f4(Sidelkino,Bar8) | f4(KK1,Bar8) |", "|---|---:|---:|---:|"]
        for row in h055_summary["chromosomes"]:
            lines.append("| chr"+row["chromosome"]+" | "+str(row["SNPs"])+" | "+f"{row['means'][0]:.8f}"+" | "+f"{row['means'][1]:.8f}"+" |")
        lines.append("| Pooled | "+str(h055_summary["primary_SNPs"])+" | "+f"{h055_summary['pooled_equal_SNP_means'][0]:.8f}"+" | "+f"{h055_summary['pooled_equal_SNP_means'][1]:.8f}"+" |")
        lines += ["", "Both contrasts use later-minus-earlier and the same13-person complete GP support over the fixed three windows. Descriptive chromosome SDs: "+", ".join(f"{v:.8f}" for v in h055_summary["descriptive_chromosome_SD"])+". These are not independent uncertainty estimates. DeletingNEO92 flips the Sidelkino pooled sign. No neural model or ancestry truth was used.", "", "[Actual report and registered outcome](../research/reviews/H055_root_actual_CPU_20261016_v1/REPORT.md) · [Post-outcome biological context audit](../research/reviews/H055_actual_biological_interpretation_audit_20261016_v1/)", ""]
    lines += ["H056 separately registers the frozenH045 model on the same10query subjects and three nominal windows. Source/cardinality qualification is pending. Existing H045 real-data output covers7of10people onchr22 only. No H056 model score exists.", ""]
    lines+=["","## H046 matched continuous-flank outcome","",h046_outcome,
        "[Completed H046 comparison and paired effects](../research/reviews/H046_completed_comparison_root_20261015_v1/REPORT.md).","",
        "## Live full-budget training","",
        "| Run | GPU | Verified process present | Passes | Actual updates per arm | Planned updates per arm | Validation accuracy |",
        "|---|---:|---|---:|---|---:|---|"]
    for j in jobs:
        lines.append(f"| {j['hypothesis']} | {j['GPU']} | {j['verified_process_present']} | {j['passes_completed']}/30 | {j['updates_per_arm'] or 'unverified'} | 960 | {j['validation_score']} |")
    lines+=["","Update counts below are verified receipt lower bounds, not live optimizer counters.",""]
    for j in jobs:
        lines.append("- "+j["hypothesis"]+": "+j["runtime_status"]+". AdamW lr="+str(j["training"]["learning_rate"])+", weight_decay="+str(j["training"]["weight_decay"])+", seed="+str(j["training"]["seed"])+", dtype="+str(j["training"]["dtype"])+".")
    lines+=["","## Registered next architecture experiment","",h048_status,"","[Fresh independent novelty/cross-domain report](../"+novelty_path+"). "+novelty_status,"","H048 retains 31,933 parameters per arm and a fixed 30-pass/960-update schedule. Its matched comparison places the same graph signal into transitions or confidence. The reset-chain components have prior art. No novelty claim is established. Actual H048 scores are recorded above.","","## American C3 benchmark","",
        "| Method | Current ancestry scores | Status |","|---|---|---|",
        "| Native ARGformer | unavailable | "+native_status+" |",
        "| Native ARGformer dataset counts | not an accuracy run | "+counts_status+" |",
        "| Native FLARE | unavailable | "+flare_status+" |",
        "| Our C3 GNN | unavailable | TRAIN000 exact native focal/plan preparation: "+focal_status+". "+fanout_status+". Shared 30 Mb VAL000 inference completed. Remaining TRAIN parents, checkpoint and full prespecified VAL gate pending |",
        "","The TRAIN000 conversion is development evidence. Independent VAL000 inference now has its own actual completion/wait receipts; its method predictions and common-truth scoring remain pending. "+val_status+" Original-paper simulation assumptions and source versions remain explicitly documented.","",
        table1_fidelity+" [Exact fidelity audit](../"+table1_audit_path+").","",
        "## GST molecular graph-set pilot","",
        gst_status+" AP is pooled per-atom average precision. One seed supplies no training-seed SD.","",
        "| Native setup | VALID AP | Selected epoch |","|---|---:|---:|"]
    for k,v in gst.items():
        lines.append(f"| {k} | {v['selected_VALID_AP']:.8f} | {v['selected_epoch']} |")
    lines+=["","Full TRAIN parsing retains10,439/10,500 reactions. This is data readiness, not a new model score. [Fresh task-semantics/transfer assessment](../"+novelty_path+").",""]+pilot_lines+["",
        "## Real data and paper","",
        real_status+" No empirical ancestry accuracy exists. The AAAI draft is8pages and remains unfinished.","",
        "## Evidence and complete recorded inventory",""]
    for r in models:
        p=r["source_ref"]["path"]
        lines.append(f"- {r['name']}: [{p}](../{p})")
    lines+=["","[All recorded metric receipts, including historical pilots](all_recorded_metric_receipts.md) · [Machine-readable current scoreboard](current_scoreboard.json)"]
    (RESEARCH/"current_scoreboard.md").write_text("\n".join(lines).replace("]("+pilot_source,"](../"+pilot_source)+"\n")
    ledger=read("research/experiment_ledger.json")
    inventory=[r for r in ledger["result_inventory"] if r.get("metrics")]
    if tgb_ready and not any(r.get("artifact")==tgb_results_path for r in inventory):
        for name,arm in tgb_results["results"].items():
            inventory.append(dict(study="H052/"+name,kind="native_cross_domain_predictive_metrics",status="completed_causal_diagnostic",artifact=tgb_results_path,metrics=arm["splits"]))
    (RESEARCH/"all_recorded_metric_receipts.json").write_text(json.dumps(dict(updated_utc=stamp,
        warning="Distinct datasets/conditions are not interchangeable. Engineering measurements are explicitly tagged and are not accuracy results.",records=inventory),indent=2,sort_keys=True)+"\n")
    hist=["# All recorded metric receipts","",
        "This inventory includes historical pilots, selected-model scores, timing and parameter records. The kind column distinguishes engineering records. It does not rank incompatible datasets. Use the current scoreboard for the shared development comparison.","",
        "| Study | Kind | Status | Recorded metrics | Source |","|---|---|---|---|---|"]
    for r in inventory:
        p=r["artifact"]
        metrics=json.dumps(r["metrics"],sort_keys=True,ensure_ascii=False).replace("|","\\|")
        hist.append(f"| {r['study']} | {r['kind']} | {r.get('status','')} | {metrics} | [{Path(p).name}](../{p}) |")
    (RESEARCH/"all_recorded_metric_receipts.md").write_text("\n".join(hist)+"\n")
    table=["<!-- CURRENT_SCOREBOARD_START -->",f"**Scoreboard updated UTC: {stamp}.** [Every current setup, all cohorts and detailed metrics](research/current_scoreboard.md) · [Native class metrics and training-lineage audit, frozen17-setup snapshot](research/reviews/root_score_report_20261015_v1/REPORT.md) · [All recorded historical/engineering metric receipts](research/all_recorded_metric_receipts.md)","",
        "| Current run | Device | Progress | Validation score |","|---|---:|---|---|"]
    for j in jobs:
        if j["evaluation_completed"]:continue
        updates=(str(min(j["updates_per_arm"].values()))+"/960 verified updates per arm"+(" (receipt lower bound)" if j["updates_are_verified_receipt_lower_bounds"] else "")) if j["updates_per_arm"] else "optimizer updates not yet verified"
        table.append(f"| {j['hypothesis']} | GPU{j['GPU']} | {j['passes_completed']}/30 completed passes; {updates}; {j['runtime_status']} | {j['validation_score']} |")
    short_ready=14+sum(r["stage"]=="infer" and r["completed_and_wait0_verified"] for r in wave_status)
    focal_ready=1+len(fanout_completed)+len(continuation_completed)
    checkpoints=native_execution.get("checkpoint_metadata_only",[])
    native_brief=("Engineering TRAIN000 full-corpus MLM training. Latest observed checkpoint: "+", ".join(r["name"] for r in checkpoints)+
        ". Five epochs / 224,775 planned updates. Final completion pending. This is engineering TRAIN000, with held-out per-ARG models still pending." if checkpoints and not v12_wait and native_live else native_status)
    if epoch_event and v12_launch and not v12_wait and native_live:
        native_brief+=" Actual author console entered epoch "+str(epoch_event["actual_console_payload"]["epoch_one_based"])+"/5."
    table+=["| American TRAIN preparation | CPU | "+str(short_ready)+"/20 inferred 2 Mb TRAIN parents ready. Exact 10k-focal plans: "+str(focal_ready)+"/20 completed; controlled continuation uses at most two CPU producers. | no ancestry score |",
        "| Original American VAL000 | CPU | Full 30 Mb simulation, inference and 10k-focal/native-path preparation completed. Same 600 references and 1,000 queries. | predictions pending |",
        "| Original American VAL001–004 | CPU | "+remaining_val_status+" | no ancestry score |",
        "| Native ARGformer | "+native_device+" | "+native_brief+" | pending |",
        "| Native ARGformer held-out VAL000 | CPU | Original per-ARG six-dataset counts: "+val000_native_counts["status"]+". Separate native fit follows actual counts. | no ancestry score |",
        "| Native FLARE | CPU | Author build, TRAIN000 full-site inference and prediction seal completed. Held-out VAL000 conversion: "+flare_val_conversion["status"]+"; native inference: "+flare_val_native["status"]+"; seal: "+flare_val_seal["status"]+". | held-out score pending |",
        "| Our American C3 GNN | "+c3_device+" | "+str(focal_ready)+"/20 exact TRAIN plans completed. "+c3_status.rstrip(".")+". Original VAL000 full-length GNN block plan: "+gnn_VAL000_plan["status"]+". | no ancestry score |",
        "| H049 cut-memory candidate | CPU preparation | "+h049_prepare_status+". Three matched 32,446-parameter arms. Full 30 passes / 960 updates per arm planned. | no fit or accuracy yet |",
        "| H050 stronger direct-readout candidate | "+"GPU0"+" | "+h050_execution_status.rstrip(".")+". H045 predictor retained initially. Three 32,325-parameter memory arms and an unmodified 31,812-parameter continuation. 960 updates per arm, 192 prediction seals planned. Original45run interrupted byGPU0-only instruction, no validation accuracy. | full fit incomplete |",
        "| Empirical data | CPU | H045 probabilities, ancestry proportions and genotype PCA completed for 24 query haplotypes / 12 subjects. Descriptive only, without known ancestry truth. | completed |","",
        "**Completed H048:** "+h048_scores+". Both registered utility criteria failed. [Actual scores, all cohorts and provenance](research/reviews/H048_completed_comparison_root_20261015_v1/REPORT.md)." if h048_ready else "**H048 evaluation:** pending.","",
        "**Current measured development scores:** H045 foreign-episode **96.8270 ± 0.5870%** Present182 span accuracy, matched own-state **96.8175 ± 0.6142%**. H046: "+h046_scores+". "+h046_outcome+" H047 original-risk **96.7777 ± 0.5970%**, balanced-risk **96.7576 ± 0.5287%**. SD is across eight VAL simulations, one optimizer seed. H048: "+h048_scores+". [All "+str(len(models))+" measured setups, all seven cohorts, NLL/F1/boundaries/Brier/ECE and training budgets](research/current_scoreboard.md#detailed-cohort-metrics).","",
        "[Fresh independent H048 novelty/cross-domain assessment]("+novelty_path+"). "+novelty_status,"",
        "**ARGformer paper fidelity:** "+table1_fidelity+" [Audit]("+table1_audit_path+").","",
        "**Biological comparison scope:** Our current AncestralPaths comparator is the released Dense workflow with derived four-source probability collapse. Native six-path ancestry and BAA-excluded four-path retraining are distinct endpoints. [Exact labels, cohorts and protocol differences](docs/biological_comparison_scope.md).", "",
        "**Remaining biological error:** H045 recovers22.59% of true Present182 changes at1kb tolerance despite96.827% span accuracy. Across8VAL parents its selected predictions match905 changes, miss3103 and add5063extra calls. [Audited residual-error table and scope limits](research/reviews/residual_biological_errors_20261015_v1/REPORT.md).", "",
        "**Comparison contract:** Use the same observed genomes, reference/query identities, truth and scored support. Each method keeps its native representation, such as VCFs, ancestral paths or PyG tensors. Extra inputs, training changes and reconstruction differences are disclosed. Native-format differences do not postpone a shared-data comparison.","",
        "**Next ARGMix comparison, not yet run:** A prospective 1,611-diploid paired panel with explicit reference/query roles. The official HapMapII GRCh37 chr22 map has been acquired and the generator map contract passed. Simulation and model comparison remain pending. This is a new common-data comparison, distinct from the unavailable original-table inputs. [Frozen plan and remaining adapters](research/reviews/ARGMix_published_panel_fair_comparison_source_plan_20261016_v1/HANDOFF.md).","",
        "**Paper readiness:** The compiled eight-page AAAI draft now includes H045 and its matched control. It is not submission-ready. American ARGformer/FLARE scores, a larger ARGMix comparison, independent held-out confirmation, useful operator novelty, real-data corroboration and learned cross-domain evidence remain unfinished. [Draft scope](paper/README.md).", "",
        "**Danish biological corroboration:** "+h055_status+" [Actual results and limits](research/reviews/H055_root_actual_CPU_20261016_v1/REPORT.md).","",
        "**Current GPU policy: physical GPU0 only.** Our GPU4/5 jobs have stopped and their contexts/known roots are absent. Parallel jobs onGPU0 are permitted after fresh device-bound admission. [Policy and actual stop evidence](research/reviews/GPU0_only_user_transition_root_20261016_v1/GPU0_only_policy.root_active_v1.json).","",
        "**GST and cross-domain status:** "+gst_status,"",
        "**TGB ordered cross-domain diagnostic:** "+tgb_status+" [Frozen protocol and actual per-year results](research/current_scoreboard.md#tgb-annual-trade-causal-diagnostic).","",
        ordered_support_note,"",
        "Parallel ownership: two workers implement the frozen H056 same-cohort H045 application sources. One worker updates the development manuscript proposal. Root owns execution, outcomes, metrics and Git.","","<!-- CURRENT_SCOREBOARD_END -->"]
    p=ROOT/"README.md"
    text=p.read_text()
    a="<!-- CURRENT_SCOREBOARD_START -->"
    b="<!-- CURRENT_SCOREBOARD_END -->"
    if a in text:
        text=text[:text.index(a)]+"\n".join(table)+text[text.index(b)+len(b):]
    else:
        marker="## Latest status\n"
        assert marker in text
        text=text.replace(marker,marker+"\n"+"\n".join(table)+"\n",1)
    # Refresh the directly visible main comparison and population tables.
    import re
    main_table=["| Pipeline | Present182 accuracy, mean ± SD (%) | Mean NLL | Boundary F1 at1kb (%) |",
        "|---|---:|---:|---:|"]
    for r in models:
        q=r["cohorts"][PRIMARY]
        main_table.append("| "+r["name"]+" | "+fmt(q.get("accuracy"),True)+" | "+fmt(q.get("nll"))+" | "+fmt(q.get("boundary_f1"),True)+" |")
    text,count=re.subn(r"\| Pipeline \| Present182 accuracy[^\n]*\n\|[^\n]*\n(?:\|[^\n]*\n)+",
        "\n".join(main_table)+"\n",text,count=1)
    assert count==1
    pop_table=["| Setup | "+" | ".join(LABELS)+" |","|---|"+"---:|"*len(COHORTS)]
    for r in models:
        pop_table.append("| "+r["name"]+" | "+" | ".join(fmt(r["cohorts"].get(c,{}).get("accuracy"),True) for c in COHORTS)+" |")
    text,count=re.subn(r"\| Cohort \| Native AP accuracy[^\n]*\n\|[^\n]*\n(?:\|[^\n]*\n)+",
        "\n".join(pop_table)+"\n",text,count=1)
    # Successive refreshes update the generated version, too.
    if count==0:
        text,count=re.subn(r"\| Setup \| Present182 \| BronzeAge162[^\n]*\n\|[^\n]*\n(?:\|[^\n]*\n)+",
            "\n".join(pop_table)+"\n",text,count=1)
    assert count==1
    # This whole status section is generated so old queue and running claims cannot linger.
    section_start="### Running and next experiments"
    section_end="### Original-paper and empirical validation"
    if section_start in text and section_end in text:
        running=[section_start,"",
            "**H048, GPU4:** "+h048_status+" Fixed budget:30 FP32 passes and960 updates per arm. [Registered evaluation route](research/reviews/H048_evaluate_root_handoff_20261015_v1/HANDOFF.md).","",
            "**Native ARGformer, "+native_device+":** "+native_status,"",
            "**Native FLARE:** "+flare_status+" Held-out VAL000 conversion: "+flare_val_conversion["status"]+".","",
            "**Original remaining American VAL:** "+remaining_val_status+".","",
            "**Completed H046:** "+h046_outcome+" [Results](research/reviews/H046_completed_comparison_root_20261015_v1/REPORT.md).","",
            "**Completed H047:** original-risk96.7777±0.5970%, balanced-risk96.7576±0.5287%. Both30passes/960updates. Registered accuracy benefit not met. [Results](research/reviews/H047_completed_comparison_root_20261015_v1/REPORT.md).","",
            "**American GNN preparation:** "+fanout_status+". "+wave_text+" "+c3_status+".","",
            "**Architecture work:** H049 sample-witness order-2 cut memory is implemented and registered with matched null/leaf-memory controls. "+h049_prepare_status+". Predictive utility and a distinct new contribution remain unproved. [Design and prior-art limits](research/reviews/ordered_graph_operator_research_20261015_v1/REPORT.md).","",
            "Root orchestrates execution and publishing. Three workers handle full prospective1611 generation, shared-run evaluation integration, and operator novelty/scaling in parallel. Only physical GPU0 is eligible for future experiments, with process title AI Auto Research Pipeline Test. Historical GPU4/5 attempts are stopped.","",""]
        text=text[:text.index(section_start)]+"\n".join(running)+text[text.index(section_end):]
    pilot_start="<!-- ADAPTED_ARGFORMER_PILOT_START -->"
    pilot_end="<!-- ADAPTED_ARGFORMER_PILOT_END -->"
    pilot_block=pilot_start+"\n"+"\n".join(pilot_lines)+"\n"+pilot_end
    if pilot_start in text:
        text=text[:text.index(pilot_start)]+pilot_block+text[text.index(pilot_end)+len(pilot_end):]
    else:
        text+="\n"+pilot_block+"\n"
    stale_table_start="| Benchmark | Completed evidence | What is still needed |"
    if stale_table_start in text:
        rows=[stale_table_start,"| --- | --- | --- |",
            "| European common-input development panel | Native AP, current native ARGMix and our GNN on eight shared VAL simulations, population SDs and confusion/boundary diagnostics | Independent test confirmation and training-seed variation |",
            "| ARGMix published Tables 1 and 2 | Primary-source protocol and published values audited | Published-panel IID and misspecification experiments. Paper-era code and recipe details remain unavailable |",
            "| ARGformer American Table 1 | Full30Mb TRAIN and VAL000 simulated/inferred inputs and native paths completed. Native MLM status: "+native_status+". FLARE TRAIN000 calls sealed. | Finish native per-target fits, original remaining VAL001–004, all matched predictions and common EUR/EAS/AFR metrics. Exact Table1 downstream settings remain unresolved |",
            "| Empirical ancestry and PCA | H045 predictions, bp-weighted proportions and original-genotype PCA completed for24query haplotypes/12subjects on the355-tree,2074-site panel | Descriptive biological/domain assessment. No known empirical truth or accuracy |"]
        text,count=re.subn(r"\| Benchmark \| Completed evidence[^\n]*\n\|[^\n]*\n(?:\|[^\n]*\n)+",
            "\n".join(rows)+"\n",text,count=1)
        assert count==1
    text=text.replace("Fourteen independent 2 Mb TRAIN preparations are complete.",
        "TRAIN000–015 independent2Mb preparations are complete; TRAIN016/017 are running and018/019 remain pending.")
    text=text.replace("TRAIN000–015 independent2Mb preparations are complete; TRAIN016/017 are running and018/019 remain pending.",
        str(short_ready)+"/20 independent2Mb TRAIN inferred inputs and"+str(focal_ready)+"/20 exact focal plans are complete. Original TRAIN019 inference recovery completed with genuine remote wait0, preserving the earlier supervisor/transport failures.")
    text=text.replace("The 40-haplotype panel and pre-size/domain differences still need an explicit adapter and assessment. No ancestry or PCA result.",
        "The variable-panel adapter and H045 empirical application have completed, with ancestry probabilities, bp-weighted proportions and genotype PCA for24query haplotypes/12subjects. These are descriptive outputs without known ancestry truth. Panel/domain assessment remains necessary. [Completed application](experiments/results/H018_H045_empirical_application_root_v1/completed.json).")
    p.write_text(text)
    print(json.dumps(dict(current_setups=len(models),cohorts=len(COHORTS),recorded_metric_receipts=len(inventory),live_jobs=jobs,updated_utc=stamp)),flush=True)
if __name__=="__main__":
    main()
