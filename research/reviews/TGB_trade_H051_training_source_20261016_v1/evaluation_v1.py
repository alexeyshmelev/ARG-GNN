"""After-seal split evaluation; never called automatically by the trainer."""
from __future__ import annotations
import hashlib
from .targets_and_metrics_v1 import validate_target_v1,score_year_v1,summarize_years_v1,require

def _difference_summary(rows):
    result=summarize_years_v1(rows)
    result["mean_delta_ndcg_at10"]=result.pop("mean_ndcg_at10")
    return result

def evaluate_sealed_split_v1(bundle,*,split,load_target_year,baseline_results,
        baseline_result_ref,writer,runtime_check,protocol):
    require(split in ("VAL","TEST") and bundle.seal_ref and bundle.fit_receipt_ref,"Actual sealed primary bundle and explicit split")
    require(baseline_result_ref==protocol["source_refs"]["H052_results"],"Frozen native baseline result reference")
    expected=tuple(range(2010,2014)) if split=="VAL" else tuple(range(2014,2017))
    current_sha=hashlib.sha256(memoryview(bundle.scores.numpy().reshape(-1)).cast("B")).hexdigest()
    require(current_sha==bundle.score_payload_sha256,"Exact sealed score payload unchanged")
    runtime_check("before_explicit_"+split+"_target_release")
    labels={year:validate_target_v1(load_target_year(year),year) for year in expected}
    results={};by_key={}
    for si,seed in enumerate(bundle.seeds):
        for ai,arm in enumerate(bundle.arms):
            rows=[]
            for year in expected:
                runtime_check("evaluate_%s_%d"%(split,year))
                row=score_year_v1(bundle.scores[si,ai,bundle.years.index(year)],labels[year])
                rows.append(row)
            key="%s.seed%d"%(arm,seed);by_key[key]=rows
            results[key]=dict(summary=summarize_years_v1(rows),by_year=rows)
    native={};differences={}
    for name in ("native_persistence","native_EMA_window7"):
        rows=[row for row in baseline_results["results"][name]["by_year"] if row["target_year"] in expected]
        require(tuple(row["target_year"] for row in rows)==expected,"Same H052 target years")
        require(all(row["queried_countries"]==int(labels[row["target_year"]].source_present.sum()) for row in rows),
            "Identical official source support")
        native[name]=dict(summary=summarize_years_v1(rows),by_year=rows)
        for key,model_rows in by_key.items():
            delta=[dict(target_year=x["target_year"],queried_countries=x["queried_countries"],
                ndcg_at10=x["ndcg_at10"]-y["ndcg_at10"]) for x,y in zip(model_rows,rows)]
            differences[key+" minus "+name]=_difference_summary(delta)
    architecture={}
    for seed in bundle.seeds:
        for left,right in (("conditional_triple","first_order"),("first_order","tree_only")):
            rows=[dict(target_year=x["target_year"],queried_countries=x["queried_countries"],
                ndcg_at10=x["ndcg_at10"]-y["ndcg_at10"])
                for x,y in zip(by_key["%s.seed%d"%(left,seed)],by_key["%s.seed%d"%(right,seed)])]
            architecture["%s minus %s.seed%d"%(left,right,seed)]=_difference_summary(rows)
    result=dict(schema="TGB_trade_H051_sealed_split_results_v1",split=split,models=results,
        native_baselines=native,paired_model_minus_native=differences,paired_architecture=architecture,
        prediction_seal_ref=bundle.seal_ref,H052_result_ref=baseline_result_ref,
        metric="Exact sklearn1.9.1 ndcg_score k10 ignore_tiesFalse; equal queried countries within year, equal years",
        final_epoch=100,model_selection_performed=False,seed_count=1,confirmatory_result=False,
        year_SD_is_not_optimizer_seed_SD=True,novelty_or_paper_reproduction_claim=False)
    runtime_check("before_evaluation_write_"+split)
    return writer.write_evaluation(split,result)
