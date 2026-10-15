"""Read existing score and completion metadata; write only this audit packet."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,importlib.util,json,math,statistics,io
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/current_score_reporting_audit_20261015_v1"
PRIMARY="primary_present_day_182"
COHORTS=[PRIMARY,"age_stratum_Bronze_Age","age_stratum_BAA","age_stratum_Neo","age_stratum_Yam","secondary_ancient_504","all_queries_686"]
LABELS=["Present182","BronzeAge162","BAA14","Neo302","Yam26","Ancient504","All686"]
def ref(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p;b=p.read_bytes()
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(path):return json.loads((ROOT/path).read_text())
def save(name,value):(D/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def module(name):
    s=importlib.util.spec_from_file_location("_audit_"+name,D/(name+".py"))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
normalize=module("normalization_reference_v1").normalize_scored_cells
native_overrides=module("native_metric_overrides_v1").load_native_overrides()
board=read("research/current_scoreboard.json")
for path,name in [("README.md","README.snapshot.md"),("research/current_scoreboard.md","scoreboard.snapshot.md"),("research/current_scoreboard.json","scoreboard.snapshot.json")]:
    (D/name).write_bytes((ROOT/path).read_bytes())
completions={
 "H040":"experiments/results/H040_GPU45_frozen_evaluation_seed101_v3/completed.json",
 "H042":"experiments/results/H042_GPU4_clade_flags_evaluate_seed101_memo_v1/completed.json",
 "H043":"experiments/results/H043_GPU5_transport_evaluate_seed101_v1/completed.json",
 "H044":"experiments/results/H044_marker_cached_evaluate_seed101_v2/completed.json",
 "H045":"experiments/results/H045_GPU5_episode_evaluate_seed101_v1/completed.json",
 "H046":"experiments/results/H046_GPU5_flank_evaluate_seed101_eval980_v1/completed.json",
 "H047":"experiments/results/H047_GPU4_cohort_risk_evaluate_seed101_v1/completed.json"}
data={k:read(v) for k,v in completions.items()}
raw_sources=[ref(v) for v in completions.values()]
h038paths={a:"experiments/results/mutation_routed_full_panel_pyg_v1_"+a+"_seed101_GPU"+g+"/final_selected_full_diagnostics.json"
    for a,g in (("aligned","2"),("zero_concordance","3"))}
h038={a:read(p) for a,p in h038paths.items()};raw_sources.extend(ref(p) for p in h038paths.values())
native_path="research/reviews/native_common_cohort_diagnostics_20261015_v1/per_parent_metrics.json"
native=read(native_path);raw_sources.append(ref(native_path))
source_labels={r["path"]:"S"+str(i+1) for i,r in enumerate(raw_sources)}
models=[];checks=[];missing=[];hash_checks=[]
for m in board["models"]:
    name=m["name"];family=name.split()[0];arm=name.split(" ",1)[1]
    if family in completions:
        key="exact_clade_flags" if family=="H042" else arm
        cells=data[family]["selected_full_diagnostics"][key]["cells"];source=completions[family]
        selected=data[family]["selected"]
        selected_pass=(selected.get(key,{}) or {}).get("pass_index")
    elif family=="H038":
        key="aligned" if "aligned" in name else "zero_concordance"
        cells=h038[key]["cells"];source=h038paths[key];selected_pass=h038[key]["selected_pass"]
    else:
        key="native_AP" if name.startswith("Native AncestralPaths") else "native_ARGMix"
        cells=native[key];source=native_path;selected_pass=None
    cohorts=normalize(cells)
    if family not in completions and family!="H038":
        for c in COHORTS:cohorts[c].update(native_overrides[key]["cohorts"][c])
    actual_source=ref(m["source_ref"]["path"])
    assert actual_source["sha256"]==m["source_ref"]["sha256"],name
    hash_checks.append(dict(name=name,source_ref=m["source_ref"],verified=True))
    for c in COHORTS:
        for metric,record in m["cohorts"][c].items():
            raw=cohorts[c].get(metric)
            if raw is None:continue
            for stat in ("mean","sample_SD"):
                old=record.get(stat);new=raw.get(stat)
                if type(old) in (int,float) and type(new) in (int,float):
                    checks.append(dict(model=name,cohort=c,metric=metric,stat=stat,absolute_delta=abs(old-new)))
                    assert abs(old-new)<=1e-6,(name,c,metric,stat,old,new)
        for metric,record in cohorts[c].items():
            prior=m["cohorts"][c].get(metric)
            if prior is None or (prior.get("mean") is None and record.get("mean") is not None) or (prior.get("sample_SD") is None and record.get("sample_SD") is not None):
                missing.append(dict(model=name,cohort=c,metric=metric,scoreboard_value=prior,available_scored_value=record))
    if family=="H038":
        trained_passes=30;new_updates=960;lineage_updates=960;selected_updates=32*selected_pass
        budget="fresh30pass/960updates; selectedpass"+str(selected_pass)
    elif family in ("H040","H042"):
        trained_passes=30;new_updates=960;lineage_updates=1920;selected_updates=1920
        budget="H038960 +960updates; 30extra passes"
    elif family in completions:
        trained_passes=30;new_updates=960;lineage_updates=2880;selected_updates=2880
        budget="H0421920 +960updates; 30extra passes"
    elif key=="native_AP":
        trained_passes=None;new_updates=None;lineage_updates=None;selected_updates=None
        budget="5epochs,batch32,5,190,874rows; finalepoch5; seed101"
    else:
        trained_passes=None;new_updates=None;lineage_updates=None;selected_updates=None
        budget="11epochs; native earlystop,selectedepoch9; seed50"
    models.append(dict(name=name,cohorts=cohorts,score_summary_ref=m["source_ref"],raw_score_ref=ref(source),
        source_label=source_labels[source],training_budget=budget,completed_extra_updates=new_updates,
        cumulative_model_lineage_updates=lineage_updates,selected_model_lineage_updates=selected_updates,
        selected_pass=selected_pass,source_notes=m.get("notes","")))
best=[]
for c,label in zip(COHORTS,LABELS):
    m=max(models,key=lambda r:r["cohorts"][c]["accuracy"]["mean"])
    best.append(dict(cohort=c,label=label,setup=m["name"],accuracy=m["cohorts"][c]["accuracy"],
        source_ref=m["raw_score_ref"],source_label=m["source_label"]))
save("all17_allcohort_metrics.actual.json",dict(schema="all17_existing_scored_scalar_metrics_audit_v1",
    no_new_inference_or_truth_arrays=True,SD_unit="sample SD across eight independent VAL simulations; one optimizer seed per setup",
    boundary_tolerance_bp=1000,cohort_labels=dict(zip(COHORTS,LABELS)),models=models,best_accuracy_by_cohort=best))
save("verification.actual.json",dict(schema="current_score_reporting_verification_v1",
    audited_utc=datetime.now(timezone.utc).isoformat(),scoreboard_updated_utc=board["updated_utc"],
    snapshot_refs=[ref(D/n) for n in ("README.snapshot.md","scoreboard.snapshot.md","scoreboard.snapshot.json")],
    all17_scoreboard_source_hashes=hash_checks,raw_source_crosswalk=raw_sources,
    numeric_scoreboard_scalar_comparisons=len(checks),max_absolute_delta=max(r["absolute_delta"] for r in checks),
    published_numeric_metrics_agree_with_raw_receipts_within=1e-6,
    omitted_or_incompletely_reported_metrics=missing,
    actual_models_inference_truth_arrays_native_runtime_GPU_admission_accounting_Git_executed=False))
fields=["setup","cohort","queries","source_label","source_path","training_budget"]
metric_names=("accuracy","balanced_accuracy","macro_f1","nll","boundary_precision","boundary_recall","boundary_f1","brier","calibration_ece","dosage_mse_per_diploid_bp_class","dosage_mae_per_diploid_bp_class")
for k in metric_names:fields.extend([k+"_mean",k+"_sample_SD",k+"_defined_parents",k+"_undefined_parents"])
with (D/"all17_allcohort_metrics.actual.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for m in models:
        for c,label in zip(COHORTS,LABELS):
            row=dict(setup=m["name"],cohort=c,queries=int("".join(x for x in label if x.isdigit())),source_label=m["source_label"],source_path=m["raw_score_ref"]["path"],training_budget=m["training_budget"])
            for k in metric_names:
                r=m["cohorts"][c][k];row.update({k+"_mean":r["mean"],k+"_sample_SD":r["sample_SD"],k+"_defined_parents":r["defined_parent_count"],k+"_undefined_parents":r["undefined_parent_count"]})
            w.writerow(row)
def fmt(r,pct=False):
    v=r.get("mean");sd=r.get("sample_SD");scale=100 if pct else 1
    if v is None:return "undefined"
    if isinstance(v,str):return v
    t=f"{v*scale:.4f}" if pct else f"{v:.6f}"
    if sd is not None:t+=f" ± {sd*scale:.4f}" if pct else f" ± {sd:.6f}"
    if r.get("defined_parent_count",8)<8:t+=" (n="+str(r["defined_parent_count"])+")"
    return t
print(json.dumps(dict(status="scalar_audit_rebuilt",models=len(models),cohort_rows=len(models)*7,numeric_checks=len(checks))))
