"""Inactive ROOT-callable freeze/predict/score bridge to unchanged reviewed science.

No payload or numerical library is imported at module import. Only the genuinely
admitted finite CPU worker may call run_stage_v1. No fitting of core models,
optimizer, new initialization seed, test tuning, clipping or fallback exists.
"""
import importlib,io
from research.reviews.H061_postselection_execution_source_20261017_v1 import root_contract_v1 as c
def _archive(b):
 p=c.local(b["archive_ref"]["path"]);c.require(p.stat().st_size==14765,"Exact existing small archive")
 c.raw(b["archive_ref"]);return p
def _state(row,core,torch):
 raw=c.raw({k:row[k]for k in("path","sha256")})
 c.require(len(raw)<=c.CAPS["checkpoint_serialized_bytes"],"Bounded selected state authentication")
 state=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
 c.require(isinstance(state,dict)and state and all(isinstance(t,torch.Tensor)and t.device.type=="cpu"
  and t.dtype==torch.float32 and bool(torch.isfinite(t).all())for t in state.values()),
  "Only actual finite FP32 CPU state tensors; no optimizer/resume object")
 model=core.TreeLoomGenericForecastV1(with_memory=row["arm"]=="full")
 model.load_state_dict(state,strict=True)
 meta=c.read(row["selected_metadata_ref"])
 c.require(model.interface_contract()==meta["family_contract"]
  and sum(p.numel()for p in model.parameters())==meta["parameters"],"Exact reviewed constructor/arm/state interface")
 model.eval();return model
def _array(value,np,shape):
 a=np.asarray(value,dtype=np.float64)
 c.require(a.shape==shape and bool(np.isfinite(a).all()),"Complete finite same-support float64 product")
 return a
def _scaler(prepared,selections,np):
 scaler=selections["scaler"]
 c.require(tuple(prepared.counties)==tuple(c.binding()["counties"])
  and np.array_equal(prepared.mean,np.asarray(scaler["mean"],dtype=np.float64))
  and np.array_equal(prepared.scale,np.asarray(scaler["scale"],dtype=np.float64)),
  "Exact frozen TRAIN-only scaler and physical county join")
def _common(selections,b):
 return dict(selection_barrier_ref=selections["barrier_ref"],source_protocol_ref=b["protocol_ref"],
  registration_ref=b["registration_ref"],legacy_GRU_status="unavailable_payload_not_accepted")
def _complete(stage,barrier,writer,**extra):
 return writer.save("completed_"+stage+".json",dict(status="completed_H061_"+stage,stage=stage,
  barrier_ref=barrier,new_core_fits=0,new_optimizer_updates=0,
  legacy_GRU_status="unavailable_payload_not_accepted",automatic_next_stage=False,
  automatic_retry_resume_or_fallback=False,**extra))
def _freeze(q,selections,completion,core,writer,guard,np,torch,b):
 guard("before_TRAIN_only_comparator_freeze")
 development=completion.load_development_v1(_archive(b));_scaler(development,selections,np)
 # Authenticate every actual saved model with its exact reviewed constructor.
 # No forward pass, optimization, probe, new seed or model product is performed.
 for row in selections["products"]:
  guard("authenticate_selected_state_seed%d_%s"%(row["seed"],row["arm"]))
  model=_state(row,core,torch);del model
 guard("before_original_frozen_alpha1000_TRAIN_coefficient_reconstruction")
 ridge=completion.fit_frozen_ridge_v1(development)
 guard("after_original_frozen_alpha1000_TRAIN_coefficient_reconstruction")
 c.require(ridge.history==52 and ridge.fit_targets==tuple(range(52,365))
  and ridge.scaler.fit_end==365,"Original registered TRAIN-only frozen ridge support, no alpha search")
 state=writer.save("frozen_comparator_state.json",dict(
  schema="H061_fixed_comparator_state_v1",comparators=list(c.COMPARATORS),ridge_alpha=1000.,
  coefficients=_array(ridge.coefficients,np,(1040,20)).tolist(),
  feature_mean=_array(ridge.feature_mean,np,(1040,)).tolist(),
  target_mean=_array(ridge.target_mean,np,(20,)).tolist(),
  TRAIN_only_scaler=selections["scaler"],history_weeks=52,fit_targets=list(ridge.fit_targets),
  last_value=dict(method="last_value",observed_offset=-1,fitted_parameters=False),
  seasonal_lag52=dict(method="seasonal_lag52",observed_offset=-52,fitted_parameters=False),
  TEST_accessed=False,TEST_scored=False,alpha_search=False,core_fitting=False,
  **_common(selections,b)))
 barrier=writer.save("stage_barrier.json",dict(
  schema="H061_complete_selection_comparator_barrier_v1",selected_products=selections["products"],
  TRAIN_only_scaler=selections["scaler"],comparators=list(c.COMPARATORS),comparator_state_ref=state,
  selected_configurations_and_complete_histories_authenticated=True,all_six_state_constructors_strictly_loaded=True,
  TEST_accessed=False,TEST_scored=False,chronology_diagnostic="candidates_only_fixed_source_permutation",
  legacy_GRU_reason="Retained payload not accepted by ROOT; historical constructor alone is not payload acceptance.",
  legacy_GRU_fit_time_clean_worktree_attestation="missing_disclosed",**_common(selections,b)))
 return _complete("freeze",barrier,writer,TEST_accessed=False,TEST_scored=False)
def _ridge(frozen,selections,np):
 from arggnn.chickenpox import LinearRidgeForecaster,TrainingScaler
 c.require(frozen.get("schema")=="H061_fixed_comparator_state_v1"and frozen.get("comparators")==list(c.COMPARATORS)
  and frozen.get("ridge_alpha")==1000. and frozen.get("history_weeks")==52
  and frozen.get("fit_targets")==list(range(52,365))and frozen.get("TRAIN_only_scaler")==selections["scaler"]
  and frozen.get("TEST_accessed")is False and frozen.get("TEST_scored")is False
  and frozen.get("alpha_search")is False and frozen.get("core_fitting")is False
  and frozen.get("selection_barrier_ref")==selections["barrier_ref"]
  and frozen.get("last_value")==dict(method="last_value",observed_offset=-1,fitted_parameters=False)
  and frozen.get("seasonal_lag52")==dict(method="seasonal_lag52",observed_offset=-52,fitted_parameters=False),
  "Exact previously frozen three-comparator state; never reconstruction after TEST release")
 scaler=TrainingScaler(_array(selections["scaler"]["mean"],np,(20,)),
  _array(selections["scaler"]["scale"],np,(20,)),365)
 return LinearRidgeForecaster(_array(frozen["coefficients"],np,(1040,20)),
  _array(frozen["feature_mean"],np,(1040,)),_array(frozen["target_mean"],np,(20,)),
  scaler,52,tuple(range(52,365)))
def _prediction(value,writer,name,np,b,*,kind,freeze_ref,**identity):
 array=_array(value,np,(79,20))
 ref=writer.save(name,dict(kind=kind,**identity,prediction_cases=array.tolist(),
  target_indices=list(range(443,522)),county_order=b["counties"],freeze_barrier_ref=freeze_ref,
  prediction_clipping="none",TEST_scored=False))
 return dict(kind=kind,**identity,**ref)
def _predict(q,selections,completion,core,writer,guard,np,torch,b):
 freeze_ref,freeze=c.frozen_barrier(q,selections)
 # All comparator bytes and selected state bytes were frozen before this TEST load.
 ridge=_ridge(c.read(freeze["comparator_state_ref"]),selections,np)
 guard("before_separately_ROOT_released_TEST_decode")
 prepared=completion.load_test_v1(_archive(b),release_selections=lambda:selections["products"])
 _scaler(prepared,selections,np)
 predictions=[];chronology=[]
 for row in selections["products"]:
  seed,arm=row["seed"],row["arm"];model=_state(row,core,torch)
  for shuffled,destination,prefix,kind in((False,predictions,"prediction","test_prediction"),
    (True,chronology,"chronology","chronology_prediction")):
   consumed=[]
   def boundary(event):
    guard("scheduled_"+prefix+"_batch")
    c.require(event["event"]=="predict_batch"and event["shuffled"]is shuffled
     and event["targets"]==list(range(443,522))[len(consumed):len(consumed)+len(event["targets"])]
     and 1<=len(event["targets"])<=16,"Exact complete ordered TEST prediction batches")
    consumed.extend(event["targets"])
   value=completion.predict_test_v1(model,prepared,boundary=boundary,shuffle_prefix=shuffled)
   c.require(consumed==list(range(443,522)),"All79TESTweeks/all20counties; no omitted target")
   destination.append(_prediction(value,writer,"%s_seed%d_%s.json"%(prefix,seed,arm),np,b,
    kind=kind,freeze_ref=freeze_ref,seed=seed,arm=arm,selected_checkpoint_ref={k:row[k]for k in("path","sha256")}))
  del model
 guard("before_full_fixed_classical_predictions")
 classical=completion.classical_predictions_v1(prepared)
 controls=[]
 for name,original in(("last_value","last_value"),("seasonal_lag52","seasonal_lag52")):
  controls.append(_prediction(classical[original],writer,"comparator_"+name+".json",np,b,
   kind="comparator_prediction",freeze_ref=freeze_ref,comparator=name))
 ridge_targets=[]
 def ridge_boundary(event):
  guard("scheduled_frozen_ridge_prediction")
  c.require(event["event"]=="frozen_ridge_prediction"and event["target"]==443+len(ridge_targets),
   "Original complete ordered79week frozen ridge predictions")
  ridge_targets.append(event["target"])
 value=completion.predict_frozen_ridge_v1(ridge,prepared,boundary=ridge_boundary)
 c.require(ridge_targets==list(range(443,522)),"All fixed ridge forecast targets")
 controls.append(_prediction(value,writer,"comparator_ridge_alpha1000_from_prior_pilot.json",np,b,
  kind="comparator_prediction",freeze_ref=freeze_ref,comparator="ridge_alpha1000_from_prior_pilot"))
 barrier=writer.save("stage_barrier.json",dict(schema="H061_complete_prediction_barrier_v1",
  freeze_barrier_ref=freeze_ref,predictions=predictions,chronology_predictions=chronology,
  comparator_predictions=controls,target_indices=list(range(443,522)),county_order=b["counties"],
  chronology_permutation="Original history_batch_v1: Random(9031+target), first51shuffled, finalknownweekfixed",
  comparator_shuffle_authorized=False,TEST_accessed=True,TEST_scored=False,
  no_metrics_or_truth_scorer_called=True,**_common(selections,b)))
 return _complete("predict",barrier,writer,TEST_accessed=True,TEST_scored=False)
def _load_predictions(barrier,freeze_ref,selections,np,b):
 core={};chronology={};controls={}
 selected={(r["seed"],r["arm"]):{k:r[k]for k in("path","sha256")}for r in selections["products"]}
 # Authenticate and validate every byte/product before returning any score inputs.
 for field,kind,destination in(("predictions","test_prediction",core),
   ("chronology_predictions","chronology_prediction",chronology),
   ("comparator_predictions","comparator_prediction",controls)):
  for row in barrier[field]:
   record=c.read({k:row[k]for k in("path","sha256")})
   c.require(record.get("kind")==kind and record.get("freeze_barrier_ref")==freeze_ref
    and record.get("target_indices")==list(range(443,522))and record.get("county_order")==b["counties"]
    and record.get("TEST_scored")is False and record.get("prediction_clipping")=="none",
    "Exact externally sealed prediction product on full identical support")
   if kind=="comparator_prediction":
    key=row["comparator"];c.require(record.get("comparator")==key and key in c.COMPARATORS,"Fixed comparator identity")
   else:
    key=row["seed"],row["arm"]
    c.require(record.get("seed")==key[0]and record.get("arm")==key[1]
     and record.get("selected_checkpoint_ref")==selected[key],"Exact selected core state/seed/arm")
   c.require(key not in destination,"Unique complete sealed product")
   destination[key]=_array(record["prediction_cases"],np,(79,20))
 required={(s,a)for s in c.SEEDS for a in c.ARMS}
 c.require(set(core)==set(chronology)==required and set(controls)==set(c.COMPARATORS),"All15sealed complete products before any scorer")
 return core,chronology,controls
def _score(q,selections,completion,writer,guard,np,b):
 freeze_ref,_=c.frozen_barrier(q,selections)
 prediction_ref,barrier=c.prediction_barrier(q,selections,freeze_ref)
 core,chronology,controls=_load_predictions(barrier,freeze_ref,selections,np,b)
 guard("all15sealed_products_authenticated_before_separate_score_release")
 prepared=completion.load_test_v1(_archive(b),release_selections=lambda:selections["products"])
 _scaler(prepared,selections,np)
 # This is the first TEST truth scorer in the concrete route.
 scores=completion.score_test_v1(core,prepared,release_prediction_seals=lambda:barrier["predictions"])
 truth=prepared.counts[list(range(443,522))]
 control_scores={name:completion.metrics_v1(value,truth)for name,value in controls.items()}
 diagnostic_scores={key:completion.metrics_v1(value,truth)for key,value in chronology.items()}
 def spread(values):
  a=np.asarray(values,dtype=np.float64)
  c.require(a.shape==(3,)and bool(np.isfinite(a).all()),"Three fixed initialization seeds, no denominator changes")
  return dict(mean=float(a.mean()),sample_SD=float(a.std(ddof=1)),seeds=list(c.SEEDS))
 aggregate={arm:{metric:spread([scores[(seed,arm)][metric]for seed in c.SEEDS])
  for metric in("mae_cases","rmse_cases")}for arm in c.ARMS}
 diagnostic={arm:dict(mae_cases=spread([diagnostic_scores[(seed,arm)]["mae_cases"]for seed in c.SEEDS]),
  rmse_cases=spread([diagnostic_scores[(seed,arm)]["rmse_cases"]for seed in c.SEEDS]),
  mean_MAE_change_from_chronological=float(np.mean([diagnostic_scores[(seed,arm)]["mae_cases"]-scores[(seed,arm)]["mae_cases"]for seed in c.SEEDS])))
  for arm in c.ARMS}
 guard("before_original_paired13week_10000draw_interval")
 interval=completion.paired_block_interval_v1(np.stack([core[(s,"full")]for s in c.SEEDS]),
  np.stack([core[(s,"no_memory")]for s in c.SEEDS]),truth)
 guard("after_original_paired13week_10000draw_interval")
 full,null=aggregate["full"],aggregate["no_memory"]
 criteria=dict(mean_MAE_at_most_098_NoMem=full["mae_cases"]["mean"]<=.98*null["mae_cases"]["mean"],
  upper95_paired_delta_below_zero=interval["interval95_cases"][1]<0,
  mean_RMSE_no_worse=full["rmse_cases"]["mean"]<=null["rmse_cases"]["mean"])
 passed=all(criteria.values())
 score_ref=writer.save("scores.json",dict(schema="H061_complete_score_report_v1",prediction_barrier_ref=prediction_ref,
  core_scores=[dict(seed=s,arm=a,**scores[(s,a)])for s in c.SEEDS for a in c.ARMS],
  comparator_scores=control_scores,aggregate=aggregate,paired_block_interval=interval,
  utility_criteria=criteria,memory_utility_gate_passed=passed,
  result="predeclared_memory_utility_gate_passed"if passed else"negative_completion_retained_without_tuning_or_replay",
  chronology_scores=[dict(seed=s,arm=a,**diagnostic_scores[(s,a)])for s in c.SEEDS for a in c.ARMS],
  chronology_aggregate=diagnostic,chronology_interpretation="Reliance diagnostic only; no tuning or comparator shuffle.",
  uncertainty_scope="Three initialization seeds conditional on one fixed county graph/time series; circular13week temporal blocks.",
  legacy_GRU_fit_time_clean_worktree_attestation="missing_disclosed",all15products_authenticated_before_scoring=True,
  target_indices=list(range(443,522)),county_order=b["counties"],TEST_scored=True,**_common(selections,b)))
 release=writer.save("stage_barrier.json",dict(schema="H061_score_release_barrier_v1",
  prediction_barrier_ref=prediction_ref,scores_ref=score_ref,TEST_scored=True,**_common(selections,b)))
 return _complete("score",release,writer,TEST_accessed=True,TEST_scored=True)
def run_stage_v1(q,*,writer,runtime_guard):
 c.validate_config(q["scientific_config"]);b=c.binding()
 c.raw(b["completion_ref"]);c.raw(b["generic_core_ref"])
 selections=c.train_selections(q,authenticate_payloads=True)
 # All scientific imports occur only inside the already admitted CPU worker.
 import numpy as np
 import torch
 completion=importlib.import_module(b["completion_module"])
 if q["stage"]=="score":return _score(q,selections,completion,writer,runtime_guard,np,b)
 core=importlib.import_module(b["completion_module"].rsplit(".",1)[0]+".generic_core_v1")
 if q["stage"]=="freeze":return _freeze(q,selections,completion,core,writer,runtime_guard,np,torch,b)
 c.require(q["stage"]=="predict"and q["TEST_access_authorized"]is True and q["scoring_authorized"]is False,
  "Prediction release never authorizes a scorer")
 return _predict(q,selections,completion,core,writer,runtime_guard,np,torch,b)
