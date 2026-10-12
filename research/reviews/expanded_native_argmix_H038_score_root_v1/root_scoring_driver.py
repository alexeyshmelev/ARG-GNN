"""Source-only ARGMix/H038 scorer; root alone admits10MB and executes on CPU.

Reuse the pinned AP scoring-only runtime, admission, guarded JSON writer and
actual-wait launcher. Preserve sealed native probabilities without modification.
"""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
AP_DRIVER=ROOT/'research/reviews/expanded_native_AP_score_root_assistance_v1/root_scoring_driver.py'
AP_DRIVER_SHA='58d9e927db01a39f5c6ad95a4750a90a37ab8476b5499554825c68cdcaf81d6a'
AP_HELPER=ROOT/'research/reviews/expanded_native_AP_score_root_assistance_v1/root_score_helper.py'
AP_HELPER_SHA='735ce1ea8f5c14ee260b642472434f425284edf367b99144c18d412aa7d8fb05'
REPORT_FUNCTION=r"""def source_report_v1():
    aligned_ref=ref(ALIGNED);completed_ref=ref(COMPLETED)
    require(completed_ref["sha256"]=="9e27f2c188d74ed652ecdf10f5fdf51d23e6ffd20742f35d9832bdefcf3c698a",
            "Exact completed eight-seal native ARGMix run changed")
    aligned,completed=read_json(aligned_ref),read_json(completed_ref)
    require(aligned["schema"]=="mutation_routed_full_panel_pyg_input_v1" and aligned["arm"]=="aligned",
            "Exact existing aligned H038 metadata")
    require(completed["schema"]=="expanded_native_argmix_GPU_fit_root_v1_completed"
            and completed["status"]=="completed_native_fit_all8_VAL"
            and completed["TRAIN_parents"]==32 and completed["VAL_parents"]==8
            and completed["outer_parents"]==0 and completed["truth_opened_for_external_score"] is False,
            "Complete native ARGMix metadata required")
    fit_input=read_json(completed["input_ref"]);cfg_ref=fit_input["config_ref"]
    cfg=read_json(cfg_ref);fit=read_json(completed["fit_ref"])
    require(cfg["class_order"]==["ANA","CHG","WHG","EHG"] and fit["config_sha256"]==cfg_ref["sha256"]
            and fit["kind"]=="expanded_native_argmix_stock_fit_v1","Original C4 native fit/config lineage differs")
    sources=dict(aligned["source_sha256"])
    extra=[SELF,HELPER,UTILITY,ROOT/"src/arggnn/project_budget_v9_consumer.py",
           ROOT/"research/reviews/expanded_native_AP_score_root_assistance_v1/root_scoring_driver.py",
           ROOT/"research/reviews/expanded_native_AP_score_root_assistance_v1/root_score_helper.py"]
    for p,h in completed["VAL_seals"][0]["frame"]["owned_code_sha256"].items():
        require(ref(p)["sha256"]==h,"Frozen ARGMix native source changed")
        sources[p]=h
    for p in extra:
        r=ref(p)
        if r["path"] in sources:require(sources[r["path"]]==r["sha256"],"Original source pin changed")
        sources[r["path"]]=r["sha256"]
    for p,h in sources.items():
        require(Path(p).suffix==".py" and ref(p)["sha256"]==h,"Scoring source closure differs")
    metadata={"aligned_input":aligned_ref,"ARGMix_completed":completed_ref,
        "H038_config":aligned["refs"]["config"],"generation_config":aligned["refs"]["generation_config"],
        "registration":aligned["refs"]["registration"],"observation_cache":aligned["refs"]["observation_cache"],
        "target_bindings":aligned["refs"]["target_bindings"],"ARGMix_config":cfg_ref,"ARGMix_fit":completed["fit_ref"]}
    for r in metadata.values():
        require(Path(r["path"]).suffix==".json","Metadata consumer role required");bound(r)
    return dict(status="source_only_no_admission_execution_or_arrays",schema=SCHEMA,source_ref=ref(SELF),
        scoring_helper_ref=ref(HELPER),storage_adapter_ref=ref(GUARD),source_sha256=sources,metadata_refs=metadata,
        output_cap_bytes=CAP,GPU_authorized=False,model_execution_authorized=False,
        outer_truth_authorized=False,TRAIN_truth_authorized=False,hypothesis_id="H038",new_hypothesis=False)
"""


def namespace_v1():
    text=AP_DRIVER.read_text()
    if hashlib.sha256(text.encode()).hexdigest()!=AP_DRIVER_SHA or hashlib.sha256(AP_HELPER.read_bytes()).hexdigest()!=AP_HELPER_SHA:
        raise ValueError('Pinned AP scoring infrastructure changed')
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='source_report_v1')
    old=ast.get_source_segment(text,node)
    if text.count(old)!=1:raise ValueError('Exact AP source report function required')
    text=text.replace(old,'__ARGMIX_SOURCE_REPORT__')
    replacements=[
      ('"research/reviews/expanded_native_AP_score_root_assistance_v1"','"research/reviews/expanded_native_argmix_H038_score_root_v1"'),
      ('"data/expanded_native_ancestralpaths_v2/completed.json"','"experiments/results/expanded_native_argmix_full32TRAIN8VAL_fit_v1/completed.json"'),
      ('"Present182_FP32_C4_primary_plus_separate_native6_SNP_focals"','"Present182_FP32_common_C4_primary"'),
      ('action.add_argument("--run", action="store_true")','action.add_argument("--execute","--run",dest="run",action="store_true")'),
    ]
    for before,after in replacements:
        if text.count(before)!=1:raise ValueError('Exact pinned AP infrastructure literal required: '+before)
        text=text.replace(before,after)
    for before,after in [('expanded_native_AP_H038','expanded_native_ARGMix_H038'),
                         ('native-AP-H038','native-ARGMix-H038'),('AP_score','ARGMix_score'),
                         ('AP_parent_scored','ARGMix_parent_scored'),('AP_completed','ARGMix_completed'),
                         ('all_eight_AP_seals_before_truth','all_eight_ARGMix_seals_before_truth'),
                         ('native6_scope','prediction_scope'),('native AP/H038','native ARGMix/H038'),
                         ('AP scoring failed','ARGMix scoring failed')]:
        text=text.replace(before,after)
    text=text.replace('__ARGMIX_SOURCE_REPORT__',REPORT_FUNCTION)
    ns=dict(__name__='_root_native_ARGMix_scoring_infrastructure',__file__=str(AP_DRIVER))
    exec(compile(text,str(AP_DRIVER)+':ARGMix_scoring_only_IO','exec'),ns)
    return ns


def main():
    namespace_v1()['main']()


if __name__=='__main__':main()
