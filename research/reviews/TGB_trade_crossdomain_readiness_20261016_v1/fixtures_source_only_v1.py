"""Synthetic metadata/scheduling checks only. No TGB import, dataset or model."""
from pathlib import Path
import runpy,copy,json,ast
HERE=Path(__file__).resolve().parent
ns=runpy.run_path(str(HERE/"causal_partition_contract_v1.py"),run_name="__contract_fixture__")
h=ns["COUNTRIES"]
record=dict(schema=ns["SCHEMA"],forecast_year=2001,observation_year=2000,
    country_catalogue_sha256="a"*64,leaf_keys=list(range(h)),observed_edge_years=[1999,2000,2000],
    parent=[h]*h+[-1],merge_height=[0.]*h+[1.],cut_membership=list(range(h)),
    cut_threshold=.25,duration_years=1,missing_flags=[True]*h,
    features_observation_max_year=2000,labels_used=False)
results=[]
def passed(name,ok=True):assert ok,name;results.append(dict(case=name,passed=True))
def reject(name,mutate):
    value=copy.deepcopy(record);mutate(value)
    try:ns["validate_partition_observation_v1"](value)
    except (ValueError,TypeError):passed(name)
    else:raise AssertionError(name)
passed("full255_leaf_star_and_missing_singletons_valid",ns["validate_partition_observation_v1"](record)["cut_cells"]==255)
grouped=copy.deepcopy(record);grouped.update(parent=[255,255]+[256]*253+[256,-1],
    merge_height=[0.]*255+[.1,1.],cut_membership=[0,0]+list(range(2,255)),
    missing_flags=[False,False]+[True]*253)
passed("whole_polytomy_and_observed_shared_cut_valid",ns["validate_partition_observation_v1"](grouped)["cut_cells"]==254)
reject("target_year_edge_rejected",lambda r:r["observed_edge_years"].append(2001))
reject("future_feature_timestamp_rejected",lambda r:r.update(features_observation_max_year=2001))
reject("labels_cannot_enter_features",lambda r:r.update(labels_used=True))
reject("extra_target_field_rejected",lambda r:r.update(target=[0.]*255))
reject("fake_ARG_node_times_rejected",lambda r:r.update(node_time=[350.]*256))
reject("350_generation_cut_not_trade_units",lambda r:r.update(cut_threshold=350.))
reject("future_presence_panel_selection_rejected",lambda r:r.update(leaf_keys=list(range(254))))
reject("catalogue_without_digest_rejected",lambda r:r.update(country_catalogue_sha256="unknown"))
reject("partial_tree_leaf_drop_rejected",lambda r:r["parent"].__setitem__(0,-1))
reject("anonymous_cut_label_cannot_merge_unrelated_leaves",lambda r:r["cut_membership"].__setitem__(1,0))
reject("nonmonotone_distance_rejected",lambda r:r["merge_height"].__setitem__(0,1.1))
reject("multi_year_gap_needs_separate_contract",lambda r:r.update(duration_years=2))
reject("forecast_boundary_is_next_year",lambda r:r.update(observation_year=1999))
grouped_bad=copy.deepcopy(grouped);grouped_bad["missing_flags"][0]=True
try:ns["validate_partition_observation_v1"](grouped_bad)
except ValueError:passed("missing_histories_do_not_create_shared_cohort")
else:raise AssertionError("missing cohort")
years=[1]*7+[2]*3+[3]*3+[4]*7
masks=ns["split_masks_v1"](years)
for year in set(years):
    rows=[i for i,y in enumerate(years) if y==year]
    assert len({(masks["train"][i],masks["val"][i],masks["test"][i]) for i in rows})==1
passed("split_quantiles_never_split_tied_years")
passed("split_masks_partition_every_row",all(sum(int(masks[k][i]) for k in ("train","val","test"))==1 for i in range(len(years))))
groups=ns["causal_year_groups_v1"]([2000,2001,1999,2001,2000,2002],2001)
passed("all_target_year_ties_excluded",groups==[(1999,(2,)),(2000,(0,4))])
passed("causal_input_year_multiset_invariant_to_row_order",
    [(y,len(rows)) for y,rows in groups]==[(y,len(rows)) for y,rows in ns["causal_year_groups_v1"]([2001,2000,2002,2000,2001,1999],2001)])
passed("current_contract_default_has_no_download_model_or_job_imports",
    all(not isinstance(n,ast.ImportFrom) or n.module not in ("torch","torch_geometric","tgb") for n in ast.parse((HERE/"causal_partition_contract_v1.py").read_text()).body))
print(json.dumps(dict(schema="TGB_trade_causal_contract_synthetic_fixtures_v1",cases=results,cases_passed=len(results),
    actual_TGB_dataset_downloads_imports_models_or_jobs=0),indent=2,sort_keys=True))
