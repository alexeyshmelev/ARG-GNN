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
        return dict(present=True,required_title_present="AI Auto Research Pipeline Test" in title,
            start_ticks=int(fields[19]),CPU_seconds=(int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK"),
            current_RSS_bytes=int(fields[21])*os.sysconf("SC_PAGE_SIZE"))
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

def main():
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
        if key=="native_AP":
            cs[PRIMARY]["nll"]=stat("∞")
        add(label,p,cs,("11 epochs, native early stopping" if key=="native_ARGMix" else "5 epochs, native fixed schedule"),
            "Native inputs/objective/selection retained. AP six-path probabilities collapsed to C4.")
    p="research/reviews/H040_completed_comparison_root_20261015_v1/summary.json"
    for key,r in read(p)["rows"].items():
        cs={c:dict(accuracy=stat(v["accuracy_mean_percent"]/100,v["accuracy_sample_SD_percentage_points"]/100)) for c,v in r["populations"].items()}
        cs[PRIMARY]["nll"]=stat(r["nll"])
        add("H040 "+key,p,cs,"H038 warmstart +30 passes /960 updates, seed101")
    p="research/reviews/H042_completed_cohort_diagnostics_root_20261015_v1/summary.json"
    add("H042 exact local-clade flags",p,{c:normalize(v) for c,v in read(p)["cohorts"].items()},
        "H038 warmstart +30 passes /960 updates, seed101")
    p="research/reviews/H043_completed_comparison_root_20261015_v1/summary.json"
    for key,r in read(p)["models"].items():
        add("H043 "+key,p,{c:normalize(v) for c,v in r["cohorts"].items()},
            "H042 warmstart +30 passes /960 updates per arm, seed101")
    p="research/reviews/H044_completed_comparison_root_20261015_v1/summary.json"
    for key,r in read(p)["arms"].items():
        cs={c:dict(accuracy=stat(v["accuracy_percent"]/100,v["accuracy_sample_SD_pp"]/100),nll=stat(v["mean_nll"])) for c,v in r["cohorts"].items()}
        b=r["primary_boundaries_percent"]
        for k in ("precision","recall","f1"):
            cs[PRIMARY]["boundary_"+k]=stat(b[k]/100,b.get(k+"_sample_SD_pp",0)/100 if k+"_sample_SD_pp" in b else None)
        add("H044 "+key,p,cs,"H042 warmstart +30 passes /960 updates per arm, seed101",
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
        val_status+=" Native FLARE VAL000 target bound: "+str(val_flare_counts["num_sites"])+" sites, common R600/Q1000; predictions and scoring pending."
    native_launch="research/reviews/American_C3_native_MLM_amendment_v10/root_execution_admin_v2/worker_metadata/v11_root_outer_launch.json"
    native_status="Full30Mb corpus prepared. Native fit pending."
    native_device="GPU4/5 pending"
    native_execution=dict(optimizer_updates=None,optimizer_updates_verified=False)
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
    focal_preparation=dict(first000_status=focal_status,first000_completion_ref=ref(focal_output+"/completed.json") if focal_done else None,
        first000_actual_wait_ref=ref(focal_packet+"/"+focal_prefix+"actual_wait.json") if focal_wait else None,
        fanout_status=fanout_status,fanout_completed_parents=fanout_completed,
        fanout_source_ref=ref(fanout_root+"/fanout_v1.py"),fanout_handoff_ref=ref(fanout_root+"/HANDOFF.md"))
    h048=next((r for r in read("research/experiment_decisions.json")["decisions"] if r["id"]=="H048"),None)
    h048_job=next(j for j in jobs if j["hypothesis"]=="H048")
    h048_status=("GPU4 "+h048_job["runtime_status"]+". "+str(h048_job["passes_completed"])+"/30 completed passes; "
        +(str(h048_job["updates_per_arm"])+" verified update lower bounds." if h048_job["updates_per_arm"] else "optimizer updates unverified.")
        +" No validation accuracy yet.")
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
    wave_text+=". TRAIN016/017 then018/019 require actual prior-pair closure; at most two wave parents."
    stamp=datetime.now(timezone.utc).isoformat()
    result=dict(schema="ARG_GNN_current_scientific_scoreboard_v1",updated_utc=stamp,
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
        H048_registered_candidate=dict(status=h048_status,trained=True if h048_job["updates_per_arm"] else None,accuracy=None,
            runtime_started=h048_job["started_ref"] is not None,verified_update_lower_bounds=h048_job["updates_per_arm"],
            launch_ref=ref("research/reviews/H048_GPU45_grid_root_source_v1/admitted_train/launch.json"),
            registration_ref=ref("research/experiment_decisions.json")) if h048 else None,
        novelty_and_cross_domain=dict(status=novelty_status,report_ref=ref(novelty_path),evidence_ref=ref(novelty_evidence_path),
            whole_H048_cross_domain_transfer_validated=False),
        GST_status=dict(status=gst_status,full_TRAIN_count_ref=ref("research/reviews/GST_full_TRAIN_count_root_v1/counts.json"),
            full_TRAIN_source_checkpoint_ref=ref("research/reviews/GST_full_TRAIN_CPU_source_v1/checkpoint.json")),
        CPU_stage_status=dict(VAL_simulation=val_simulation,VAL_inference=val_inference,VAL_native_paths=val_native_paths,
            FLARE_author_build=flare_build,FLARE_TRAIN_conversion=flare_conversion,FLARE_TRAIN_native=flare_native,FLARE_TRAIN_seal=flare_seal),
        American_short_TRAIN_remaining_preparation=wave_status,
        American_short_TRAIN014_recovery=recovery,C3_focal_plan_preparation=focal_preparation,
        native_ARGformer_execution=native_execution,VAL000_FLARE_target_binding=val_flare_binding,
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
    lines+=["","## Registered next architecture experiment","",h048_status,"","[Fresh independent novelty/cross-domain report](../"+novelty_path+"). "+novelty_status,"","H048 retains 31,933 parameters per arm and a fixed 30-pass/960-update schedule. Its matched comparison places the same graph signal into transitions or confidence. The reset-chain components have prior art. No novelty or accuracy claim is established.","","## American C3 benchmark","",
        "| Method | Current ancestry scores | Status |","|---|---|---|",
        "| Native ARGformer | unavailable | "+native_status+" |",
        "| Native FLARE | unavailable | "+flare_status+" |",
        "| Our C3 GNN | unavailable | TRAIN000 exact native focal/plan preparation: "+focal_status+". "+fanout_status+". Shared 30 Mb VAL000 inference completed. Remaining TRAIN parents, checkpoint and full prespecified VAL gate pending |",
        "","The TRAIN000 conversion is development evidence. Independent VAL000 inference now has its own actual completion/wait receipts; its method predictions and common-truth scoring remain pending. "+val_status+" Original-paper simulation assumptions and source versions remain explicitly documented.","",
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
    table=["<!-- CURRENT_SCOREBOARD_START -->",f"**Scoreboard updated UTC: {stamp}.** [Every current setup, all cohorts and detailed metrics](research/current_scoreboard.md) · [All recorded historical/engineering metric receipts](research/all_recorded_metric_receipts.md)","",
        "| Current run | Device | Progress | Validation score |","|---|---:|---|---|"]
    for j in jobs:
        updates=str(min(j["updates_per_arm"].values()))+"/960 verified updates per arm (receipt lower bound)" if j["updates_per_arm"] else "optimizer updates not yet verified"
        table.append(f"| {j['hypothesis']} | GPU{j['GPU']} | {j['passes_completed']}/30 completed passes; {updates}; {j['runtime_status']} | {j['validation_score']} |")
    table+=["| C3 remaining TRAIN014–019 | CPU | "+wave_text+" | no ancestry score |","| Original American VAL000 | CPU | "+val_status+" | no ancestry score |","| Native ARGformer | "+native_device+" | "+native_status+" | pending |",
        "| Native FLARE | CPU | "+flare_status+" | pending |",
        "| Our C3 GNN | CPU preparation | TRAIN000 exact native focal/plans: "+focal_status+". "+fanout_status+". VAL000 inference completed; remaining TRAIN parents, checkpoint and full prespecified VAL gate pending | pending |",
        "| Empirical data | CPU | "+real_status+" | completion "+("verified" if empirical_verified else "pending")+" |","",
        "**Current measured development scores:** H045 foreign-episode **96.8270 ± 0.5870%** Present182 span accuracy, matched own-state **96.8175 ± 0.6142%**. H046: "+h046_scores+". "+h046_outcome+" H047 original-risk **96.7777 ± 0.5970%**, balanced-risk **96.7576 ± 0.5287%**. SD is across eight VAL simulations, one optimizer seed. H048 accuracy remains pending. [All "+str(len(models))+" measured setups, all seven cohorts, NLL/F1/boundaries/Brier/ECE and training budgets](research/current_scoreboard.md#detailed-cohort-metrics).","",
        "[Fresh independent H048 novelty/cross-domain assessment]("+novelty_path+"). "+novelty_status,"",
        "**GST and cross-domain status:** "+gst_status,"",
        "Parallel ownership: worker1 handles training startup and throughput, worker2 handles scores and reporting, and worker3 handles architecture plus C3 focal/plan integration. Root owns executions and Git. Independent worker tasks proceed in parallel, with completed handoffs integrated by root.","","<!-- CURRENT_SCOREBOARD_END -->"]
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
    pilot_start="<!-- ADAPTED_ARGFORMER_PILOT_START -->"
    pilot_end="<!-- ADAPTED_ARGFORMER_PILOT_END -->"
    pilot_block=pilot_start+"\n"+"\n".join(pilot_lines)+"\n"+pilot_end
    if pilot_start in text:
        text=text[:text.index(pilot_start)]+pilot_block+text[text.index(pilot_end)+len(pilot_end):]
    else:
        text+="\n"+pilot_block+"\n"
    p.write_text(text)
    print(json.dumps(dict(current_setups=len(models),cohorts=len(COHORTS),recorded_metric_receipts=len(inventory),live_jobs=jobs,updated_utc=stamp)),flush=True)
if __name__=="__main__":
    main()
