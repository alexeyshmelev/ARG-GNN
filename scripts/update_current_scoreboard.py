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
    models.sort(key=lambda r:r["cohorts"][PRIMARY]["accuracy"]["mean"],reverse=True)
    jobs=[]
    for hid,gpu,output,packet in [
        ("H046",5,"experiments/results/H046_GPU5_flank_train_seed101_v2","research/reviews/H046_GPU45_flank_root_source_v2/admitted_train"),
        ("H047",4,"experiments/results/H047_GPU4_cohort_risk_train_seed101_v1","research/reviews/H047_GPU45_cohort_risk_root_source_v1/admitted_train")]:
        rows=sorted((ROOT/output).glob("TRAIN_pass_*.json"))
        last=json.loads(rows[-1].read_text()) if rows else {}
        title=read(packet+"/NVTOP_title_verification_root_v1.json")
        pid=title["own_pid"]
        live=(Path("/proc")/str(pid)).exists()
        row=dict(hypothesis=hid,GPU=gpu,worker_pid=pid,verified_process_present=live,
            passes_completed=last.get("pass_index",0),target_passes=30,updates_per_arm=last.get("steps",{}),
            target_updates_per_arm=960,completed=(ROOT/output/"completed.json").exists(),accuracy_available=False,
            last_pass_ref=ref(str(rows[-1].relative_to(ROOT))) if rows else None)
        if hid=="H047" and (ROOT/h047_summary).exists():
            row["accuracy_available"]=True
            row["evaluation_completed"]=True
            row["summary_ref"]=ref(h047_summary)
            row["validation_score"]="96.7777% original-risk, 96.7576% balanced-risk"
        else:
            row["evaluation_completed"]=False
            row["validation_score"]="pending"
        jobs.append(row)
    gstpath="research/reviews/GST_training_readiness_20261015_v1/readiness.json"
    gst=read(gstpath)["existing_observations"]["models"]
    empirical_path="data/H018_empirical_observation_bank_root_v3/completed.json"
    empirical=read(empirical_path) if (ROOT/empirical_path).exists() else None
    real_status=("Observed bank prepared, 2074 sites, 355 trees, 40 haplotypes. Metadata recovered from the saved bank after original worker exit1. Model application and PCA pending."
        if empirical else "2074-site observed bridge and355-tree graph complete. Model application and PCA pending.")
    flare_packet=ROOT/"research/reviews/FLARE_existing_JDK_build_root_v1"
    flare_wait=flare_packet/"worker_aux/actual_wait_minimal.json"
    flare_done=ROOT/"data/native_FLARE_build_existing_JDK_root_v1/completed.json"
    flare_status=("Native author-source build completed. Conversion and inference pending." if flare_done.exists()
        else "Native author-source build running. Conversion and inference pending." if not flare_wait.exists()
        else "Native build terminal. Inspect preserved actual wait before use.")
    val_folder=ROOT/"research/reviews/prospective_980GB_capacity_source_v1/admitted_simulate_v1"
    val_launch=val_folder/"launch.json"
    val_wait=val_folder/"actual_wait.json"
    val_sim_complete=ROOT/"experiments/results/American_C3_val000_full30mb_shared_v1/simulation/completed.json"
    val_status="Original30Mb VAL000 preparation pending."
    if val_launch.exists():
        val_pid=json.loads(val_launch.read_text())["pid"]
        if (Path("/proc")/str(val_pid)).exists():
            val_status="Original30Mb independent VAL000 simulation running on CPU."
        elif val_wait.exists() and val_sim_complete.exists() and json.loads(val_wait.read_text())["returncode"]==0:
            val_status="Original30Mb independent VAL000 simulation completed. Inferred observations and comparisons pending."
        else:
            val_status="Original30Mb VAL000 simulation terminal, inspect actual wait and completion."
    native_launch="research/reviews/American_C3_native_MLM_amendment_v10/root_execution_admin_v2/worker_metadata/v11_root_outer_launch.json"
    native_status="Full30Mb corpus prepared. Native fit pending."
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
    flare_convert=ROOT/"research/reviews/FLARE_train000_convert_980_root_v1"
    if (flare_convert/"admission.json").exists():
        flare_status="Native author build completed. Original30Mb TRAIN000 conversion admitted and execution requested. Not a held-out accuracy result."
    infer_packet=ROOT/"research/reviews/prospective_980GB_capacity_source_v1/admitted_infer_v1"
    if (infer_packet/"launch.json").exists():
        infer_launch=json.loads((infer_packet/"launch.json").read_text())
        if (Path("/proc")/str(infer_launch["pid"])).exists():
            val_status="Original30Mb independent VAL000 simulation completed. Genealogy inference running on CPU."
    elif (infer_packet/"admission.json").exists():
        val_status="Original30Mb independent VAL000 simulation completed. Genealogy inference admitted, startup checks active."
    stamp=datetime.now(timezone.utc).isoformat()
    result=dict(schema="ARG_GNN_current_scientific_scoreboard_v1",updated_utc=stamp,
        dataset=dict(name="European C4 shared development",length_bp=2000000,haplotypes=952,queries=686,
            TRAIN_simulations=32,VAL_simulations=8,SD_unit="eight VAL simulations, one optimizer seed per model",
            original_paper_reproduction=False,untouched_outer_test_completed=False),
        models=models,live_jobs=jobs,GST_TRAIN1000_seed0_VALID_AP=gst,
        original_American_C3=dict(our_GNN_score=None,native_ARGformer_score=None,native_FLARE_score=None,
            status=val_status+" Native ARGformer, FLARE and our C3 adapter proceed in parallel.",shared_VAL_preparation_status=val_status,native_ARGformer_execution_status=native_status),
        real_data=dict(predictions_available=empirical_verified,PCA_available=empirical_verified,products_written=empirical_products,completion_verified=empirical_verified,observed_bank_prepared=empirical is not None,
            prepared_bank_completion_ref=ref(empirical_path) if empirical else None,status=real_status),
        adapted_ARGformer_pilot=dict(source_ref=ref(pilot_source),rows=pilot_rows,original_paper_reproduction=False),
        native_FLARE_build_status=flare_status,
        source_only_or_engineering_checks_are_not_training_results=True)
    (RESEARCH/"current_scoreboard.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=["# Current scientific scoreboard","",f"Updated UTC: {stamp}","",
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
    lines+=["","## Live full-budget training","",
        "| Run | GPU | Verified process present | Passes | Actual updates per arm | Planned updates per arm | Validation accuracy |",
        "|---|---:|---|---:|---|---:|---|"]
    for j in jobs:
        lines.append(f"| {j['hypothesis']} | {j['GPU']} | {j['verified_process_present']} | {j['passes_completed']}/30 | {j['updates_per_arm']} | 960 | {j['validation_score']} |")
    lines+=["","## American C3 benchmark","",
        "| Method | Current ancestry scores | Status |","|---|---|---|",
        "| Native ARGformer | unavailable | "+native_status+" |",
        "| Native FLARE | unavailable | "+flare_status+" |",
        "| Our C3 GNN | unavailable | Shared30Mb adapter source ready. Completed training checkpoint and actual VAL observation required |",
        "","The complete 30 Mb observation currently available is a TRAIN parent. It must not be presented as a held-out performance comparison. "+val_status+" Original-paper simulation assumptions and source versions remain explicitly documented.","",
        "## GST molecular graph-set pilot","",
        "First1,000 retained TRAIN reactions, full1,495 VALID, seed0, 20 epochs per mode. AP is pooled per-atom average precision. Our molecular adapter and final paired TEST comparison are unfinished.","",
        "| Native setup | VALID AP | Selected epoch |","|---|---:|---:|"]
    for k,v in gst.items():
        lines.append(f"| {k} | {v['selected_VALID_AP']:.8f} | {v['selected_epoch']} |")
    lines+=["","Full TRAIN parsing retains10,439/10,500 reactions. This is data readiness, not a new model score.",""]+pilot_lines+["",
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
        table.append(f"| {j['hypothesis']} | GPU{j['GPU']} | {j['passes_completed']}/30 passes, {min(j['updates_per_arm'].values()) if j['updates_per_arm'] else 0}/960 updates per arm, TRAIN complete={j['completed']}, VAL complete={j['evaluation_completed']} | {j['validation_score']} |")
    table+=["| Original American VAL000 | CPU | "+val_status+" | no ancestry score |","| Native ARGformer | GPU4 | "+native_status+" | pending |",
        "| Native FLARE | CPU | "+flare_status+" | pending |",
        "| Our C3 GNN | source ready | Actual shared VAL observations and completed training checkpoint required | pending |",
        "| Empirical data | CPU | "+real_status+" | completion "+("verified" if empirical_verified else "pending")+" |","","Parallel ownership: worker1 handles benchmark/evaluation execution sources, worker2 handles comparison figures and runtime bottlenecks, worker3 handles architecture and the American C3 PyG adapter. Root owns live runs, integration, Git and score publication. Independent work proceeds concurrently.","","<!-- CURRENT_SCOREBOARD_END -->"]
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
