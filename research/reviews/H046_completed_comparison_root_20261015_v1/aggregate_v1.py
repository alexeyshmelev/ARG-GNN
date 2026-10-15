"""H046 metric-JSON aggregation only. No scored output until explicit aggregate and actual wait0."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,statistics
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H046_completed_comparison_root_20261015_v1"
SOURCE=HERE/"aggregate_v1.py"
LINEAGE_REF={'path': 'research/reviews/H046_completed_comparison_root_20261015_v1/lineage_refs.json', 'sha256': 'd2bd6f20b08624866dde5f73b3056fb9b78d25849674c7fda9984f4cb01a2541'}
PACKET="research/reviews/H046_GPU45_flank_eval980_source_v1/admitted_evaluate"
OUTPUT="experiments/results/H046_GPU5_flank_evaluate_seed101_eval980_v1"
ARMS=("foreign_flank","own_state")
PARENTS=tuple("ap_expanded_val_%03d"%i for i in range(8))
COHORTS=("primary_present_day_182","secondary_ancient_504","all_queries_686",
         "age_stratum_Bronze_Age","age_stratum_BAA","age_stratum_Neo","age_stratum_Yam")
METRICS=("accuracy","nll","macro_f1","boundary_F1","brier","calibration_ece")
SUPPORT=("query_count","query_bp","denominator_query_bp","original_parent_query_bp",
         "coverage_original_bp0","physical_coverage_fraction","represented_bin_support_bp","represented_span_bp")
def need(ok,message):
    if not ok:raise ValueError(message)
def path(rel):
    p=Path(rel);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT) and p.suffix in (".json",".py"),"Exact project source/metric JSON only")
    return p
def ref(rel):
    p=path(rel);need(p.stat().st_size<=16000000,"Bounded source/JSON metadata")
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
    p=path(r["path"]);need(p.suffix==".json" and ref(p)==r,"Exact actual metric/lineage JSON hash")
    return json.loads(p.read_text())
def clean(x):
    if isinstance(x,float) and math.isinf(x):return "∞" if x>0 else "-∞"
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    return x
def stats(values):
    defined=[x for x in values if x is not None]
    return dict(mean=statistics.mean(defined) if defined else None,
        sample_SD=statistics.stdev(defined) if len(defined)>1 and all(math.isfinite(x) for x in defined) else None,
        defined_parent_count=len(defined),undefined_parent_count=len(values)-len(defined),
        infinite_parent_count=sum(math.isinf(x) for x in defined))
def metric(group,name):
    value=(group["boundaries"]["f1"] if name=="boundary_F1" else group[name])
    if name=="nll" and (group.get("NLL_is_infinite") or group.get("infinite_nll_query_bp",0)):value=math.inf
    need(value is None or (type(value) in (int,float) and not math.isnan(value)),"Numeric metric or explicit undefined")
    return value
def delta(a,b):
    return a-b if a is not None and b is not None and math.isfinite(a) and math.isfinite(b) else None
def fmt(stat,scale=1):
    if stat["mean"] is None:return "undefined (n=0)"
    mean=stat["mean"];sd=stat["sample_SD"]
    value=("∞" if math.isinf(mean) else f"{mean*scale:.6f}")+" ± "+("undefined" if sd is None else f"{sd*scale:.6f}")
    return value+(f" (n={stat['defined_parent_count']})" if stat["defined_parent_count"]!=8 else "")
def aggregate(args):
    need(Path.cwd().resolve()==ROOT,"Run from actual root repository")
    need(args.expected_eval_wait_sha256,"Explicit actual evaluation wait SHA required")
    need(not (HERE/"summary.json").exists() and not (HERE/"REPORT.md").exists(),"Preserve any already-produced report")
    pins=read(LINEAGE_REF)
    need(all(ref(r["path"])==r for r in pins["refs"].values()),"Frozen actual source, registration and TRAIN/eval lineage")
    rs=pins["refs"];spec=read(rs["eval_input"]);launch=read(rs["eval_launch"])
    # Check the genuine wait before opening evaluation metrics.
    wait_ref=dict(path=PACKET+"/actual_wait.json",sha256=args.expected_eval_wait_sha256)
    wait=read(wait_ref)
    need(wait["actual_direct_launcher_wait"] is True and wait["returncode"]==0
        and wait["logging_error"] is None and wait["supervisor_error"] is None
        and wait["launcher_pid"]==launch["launcher_pid"] and wait["phase"]==launch["phase"]=="evaluate"
        and wait["physical_gpu"]==launch["physical_gpu"]==spec["physical_gpu"]==5
        and launch["input_ref"]==rs["eval_input"],"Actual same-launcher evaluation wait0 without errors")
    completed_ref=wait["completed_ref"]
    need(completed_ref["path"]==OUTPUT+"/completed.json","Actual expected evaluation completion")
    result=read(completed_ref)
    need(result["schema"]=="H046_paired_VAL_completion_v1"
        and result["status"]=="completed_original8VAL_paired_development_selection"
        and result["hypothesis_id"]=="H046" and result["input_ref"]==rs["eval_input"]
        and result["train_completion_ref"]==rs["train_completion"]==spec["train_completion_ref"]
        and result["optimization_performed"] is False and result["outer_truth_read"] is False
        and result["all96_RAM_seals_before_any_VAL_truth"] is True
        and result["same_VAL_is_development_evidence"] is True,"Genuine complete frozen H046 development evaluation")
    train=read(rs["train_completion"]);tw=read(rs["train_wait"]);tl=read(rs["train_launch"]);cfg=read(rs["config"])
    need(train["status"]=="completed_fixed30pass_paired_TRAIN" and train["steps"]==dict.fromkeys(ARMS,960)
        and tw["actual_direct_launcher_wait"] is True and tw["returncode"]==0
        and tw["logging_error"] is None and tw["supervisor_error"] is None
        and tw["completed_ref"]==rs["train_completion"] and tw["launcher_pid"]==tl["launcher_pid"],
        "Actual original paired960-update TRAIN and matching successful direct wait")
    need(set(result["selected_full_diagnostics"])==set(ARMS),"Exactly both original arms")
    arms={};values={};supports={}
    for arm in ARMS:
        full=result["selected_full_diagnostics"][arm];cells=full["cells"];selected=result["selected"][arm]
        need(full["selection_revised"] is False and len(cells)==8 and {c["parent_id"] for c in cells}==set(PARENTS)
            and selected["pass_index"] in (5,10,15,20,25,30)
            and selected["checkpoint"]==train["checkpoint_groups"][str(selected["pass_index"])][arm],
            "Eight unique genuine metric parents and original selected checkpoint reference")
        receipt=result["metric_receipts"][arm]
        need(receipt["VAL_parents"]==8 and receipt["all_VAL_seals_before_truth"] is True
            and receipt["target_mass_model_input"] is False and receipt["target_mass_VAL_loss"] is False,
            "Original pre-truth sealing and external-target reporting receipt")
        values[arm]={};supports[arm]={}
        for cell in cells:
            pid=cell["parent_id"];need(cell["split"]=="val" and set(cell["groups"])==set(COHORTS),"Original seven VAL cohorts")
            values[arm][pid]={};supports[arm][pid]={}
            for cohort in COHORTS:
                group=cell["groups"][cohort]
                values[arm][pid][cohort]={m:metric(group,m) for m in METRICS}
                supports[arm][pid][cohort]={k:group[k] for k in SUPPORT}
        cohorts={}
        for cohort in COHORTS:
            ss={m:stats([values[arm][p][cohort][m] for p in PARENTS]) for m in METRICS}
            ss["boundaries"]={"f1":ss.pop("boundary_F1")};cohorts[cohort]=ss
        arms[arm]=dict(selected_pass=selected["pass_index"],checkpoint_ref=selected["checkpoint"],
            cohorts=cohorts,per_parent_metric_values=values[arm])
    need(supports[ARMS[0]]==supports[ARMS[1]],"Identical paired query/physical scoring support")
    effects={}
    for cohort in COHORTS:
        cells=[dict(parent_id=p,**{m:delta(values[ARMS[0]][p][cohort][m],values[ARMS[1]][p][cohort][m])
            for m in METRICS}) for p in PARENTS]
        effects[cohort]=dict(direction="foreign_flank_minus_own_state",cells=cells,
            metrics={m:stats([c[m] for c in cells]) for m in METRICS})
    primary=effects[COHORTS[0]];cells=primary["cells"]
    need(all(c["accuracy"] is not None for c in cells),"All8 primary paired accuracy values")
    gain=sum(c["accuracy"] for c in cells)/8
    bd=[c["boundary_F1"] for c in cells];boundary=sum(bd)/8 if all(x is not None for x in bd) else None
    registration=read(rs["registration"])["decision"]["prospective_criteria"]
    need(registration["mean_Present182_accuracy_gain"]==.001 and registration["maximum_boundary_F1_loss"]==.01,
        "Original preregistered utility thresholds")
    meets=gain>=.001 and boundary is not None and boundary>=-.01
    original=result["registered_practical_criteria"]
    need(math.isclose(original["mean_Present182_accuracy_gain"],gain,rel_tol=1e-12,abs_tol=1e-12)
        and ((original["mean_boundary_F1_delta"] is None and boundary is None) or
             (original["mean_boundary_F1_delta"] is not None and boundary is not None and
              math.isclose(original["mean_boundary_F1_delta"],boundary,rel_tol=1e-12,abs_tol=1e-12)))
        and original["meets_registered_practical_criteria"]==meets,"Independent arithmetic matches genuine producer utility result")
    utility=dict(original,mean_accuracy_gain=gain,mean_boundary_F1_delta=boundary,
        sample_SD_accuracy_gain=primary["metrics"]["accuracy"]["sample_SD"],
        positive_parents=sum(c["accuracy"]>0 for c in cells),negative_parents=sum(c["accuracy"]<0 for c in cells),
        boundary_criterion_requires_all8_defined=True,statistical_unit="independent_ARG_parent",
        paired_direction="foreign_flank_minus_own_state")
    conclusion=("H046 meets the preregistered development utility rule." if meets else
        "H046 does not meet the preregistered development utility rule.")
    conclusion+=" Independent-test confirmation, training-seed variation and useful novelty remain unestablished."
    summary=dict(schema="H046_actual_completed_comparison_root_v1",created_utc=datetime.now(timezone.utc).isoformat(),
        root_authorizer="root_agent",source_ref=ref(SOURCE),lineage_manifest_ref=LINEAGE_REF,
        completion_ref=completed_ref,actual_launcher_wait_ref=wait_ref,train_completion_ref=rs["train_completion"],
        dataset="European C4 shared development;32TRAIN/8VAL;2Mb;952observed inputs/686queries;original clipped100bp grid",
        statistical_unit="eight independent VAL simulation parents; one optimizer seed; equal parent mean",
        SD_definition="sample standard deviation ddof1 across defined independent VAL parents; no zero fill",
        dataset_support_by_parent_and_cohort=supports[ARMS[0]],metric_configuration=cfg["metrics"],
        training=dict(optimizer="AdamW",learning_rate=.001,weight_decay=.0001,optimizer_seed=101,
            dtype="float32",passes_per_arm=30,actual_updates_per_arm=960,parameters_per_arm=31165,
            independent_reset_optimizers=True,identical_H042_pass30_initialization=True),
        selection_rule=result["selection_rule"],arms=arms,all_cohort_paired_effects=effects,
        paired_effects=utility,conclusion=conclusion,outer_test_accessed=False,
        all96_products_sealed_before_truth=True,novel_architecture_claim=False)
    lines=["# H046 continuous foreign-clade flanks","",conclusion,"",
        f"Foreign-flank minus own-state Present182 accuracy: {gain*100:+.6f} percentage points. "
        +("Boundary F1 difference is undefined." if boundary is None else f"Mean boundary F1 difference: {boundary*100:+.6f} points.")
        +" The rule requires at least +0.10 accuracy points and no more than 1.00 point mean boundary F1 loss, with all eight primary boundary values defined.","",
        "Both arms completed 30 FP32 passes and 960 whole-parent updates from identical H042 pass-30 initialization, "
        "with independent reset AdamW optimizers (learning rate 0.001, weight decay 0.0001, seed 101), 31,165 parameters each. "
        "The original rule selected checkpoint passes "+str(arms[ARMS[0]]["selected_pass"])+" (foreign flank) and "
        +str(arms[ARMS[1]]["selected_pass"])+" (own state).","",
        "All values below are equal-parent means ± sample SD across eight independent shared VAL simulations. "
        "Undefined values are excluded with explicit defined-parent counts. This reused VAL supports development selection. "
        "It supplies no independent outer-test result, training-seed SD or significance claim. "
        "Raw six-metric values, exact scoring support, paired parent effects and receipt hashes are in [summary.json](summary.json).","",
        "Cohorts contain 182 present-day, 504 ancient and 686 total queries, with ancient strata Bronze Age 162, BAA 14, Neo 302 and Yam 26. "
        "Scores integrate exact external intervals over the frozen clipped 100 bp bin support. NLL uses natural logs, boundary tolerance is 1,000 bp, "
        "Brier sums squared error over four classes, and ECE uses 15 bins with unchanged raw probabilities and no probability floor.","",
        "| Cohort | Arm | Accuracy % ± SD | NLL ± SD | Macro F1 % ± SD | Boundary F1 % ± SD | Brier ± SD | ECE ± SD |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cohort in COHORTS:
        for arm in ARMS:
            ss=arms[arm]["cohorts"][cohort]
            lines.append("| "+cohort+" | "+arm+" | "+" | ".join(fmt(ss["boundaries"]["f1"] if m=="boundary_F1" else ss[m],
                100 if m in ("accuracy","macro_f1","boundary_F1") else 1) for m in METRICS)+" |")
    lines+=["","Paired differences below are foreign flank minus own state within each parent, then mean ± sample SD. "
        "Accuracy/F1 differences use percentage points. Lower NLL/Brier/ECE differences favor foreign flank.","",
        "| Cohort | Δ accuracy pp | Δ NLL | Δ macro F1 pp | Δ boundary F1 pp | Δ Brier | Δ ECE |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for cohort in COHORTS:
        lines.append("| "+cohort+" | "+" | ".join(fmt(effects[cohort]["metrics"][m],
            100 if m in ("accuracy","macro_f1","boundary_F1") else 1) for m in METRICS)+" |")
    lines+=["","All 96 prediction products were sealed before VAL truth; all reporting reads metric JSON only. "
        "No checkpoint, probability, query-truth payload or scientific algorithm was opened or changed by this aggregator.",""]
    with (HERE/"summary.json").open("x") as stream:json.dump(clean(summary),stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
    (HERE/"REPORT.md").open("x").write("\n".join(lines))
    print(json.dumps(dict(summary_ref=ref(HERE/"summary.json"),registered_utility_met=meets,
        Present182_accuracy_gain_pp=gain*100,conclusion=conclusion),ensure_ascii=False))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--aggregate",action="store_true")
    p.add_argument("--expected-eval-wait-sha256");args=p.parse_args()
    if args.aggregate:aggregate(args)
    else:print(json.dumps(dict(status="source_prepared_no_evaluation_metrics_read_or_scored_outputs",
        expected_eval_wait_path=PACKET+"/actual_wait.json",cohorts=7,metrics=METRICS)))
if __name__=="__main__":main()
