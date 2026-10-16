"""TRAIN/VALID-only bridge to the reviewed six-fit scientific source.

No scientific import, data read, guard, process or output operation occurs on
import. Only the admitted worker calls run_train_v1 after its owned gate.
"""
from pathlib import Path
import importlib,math,time
from research.reviews.TreeLoom_Chickenpox_CPU_execution_source_20261017_v1 import root_contract_v1 as c

def run_train_v1(config, *, writer, runtime_guard):
 c.validate_config(config);b=c.binding()
 c.raw(b["completion_ref"]);c.raw(b["generic_core_ref"])
 # Exact immutable original archive authentication happens only inside the real worker.
 archive=c.local(b["archive_ref"]["path"])
 c.require(archive.stat().st_size==14765,"Exact existing small archive; no download")
 c.raw(b["archive_ref"])
 completion=importlib.import_module(b["completion_module"])
 runtime_guard("before_development_archive_decode")
 development=completion.load_development_v1(archive)
 c.require(development.phase=="development"and tuple(development.counties)==tuple(b["counties"])
  and development.counts.shape==(522,20),"Complete development support")
 counts={};validation_batches=0;first_started=None;first_seconds=None;fits=[];seen=set()
 def boundary(event):
  nonlocal validation_batches,first_started,first_seconds
  runtime_guard("scheduled_"+event["event"])
  if event["event"]=="train_batch":
   seed,arm,epoch,update=(event[k]for k in("seed","arm","epoch","update"))
   c.require(seed in c.SEEDS and arm in c.ARMS and 1<=epoch<=40
    and update==counts.get((seed,arm),0)+1 and epoch==(update-1)//20+1
    and 1<=len(event["targets"])<=16 and all(52<=t<365 for t in event["targets"]),
    "Actual scheduled fixed complete TRAIN update sequence")
   counts[(seed,arm)]=update
   if (seed,arm,update)==(101,"full",1):
    c.require(first_started is None,"One real first scheduled update")
    first_started=time.monotonic()
   if (seed,arm,update)==(101,"full",2):
    c.require(first_started is not None and first_seconds is None,"First real update completed before next boundary")
    first_seconds=time.monotonic()-first_started
    c.require(math.isfinite(first_seconds)and first_seconds>0,"Finite measured first scheduled update time")
  elif event["event"]=="predict_batch":
   c.require(event["shuffled"]is False and 1<=len(event["targets"])<=16
    and all(365<=t<443 for t in event["targets"]),"Validation predictions only; no TEST/shuffle")
   validation_batches+=1
  else:
   c.require(event["event"]=="save_selected_checkpoint"and event["seed"]in c.SEEDS
    and event["arm"]in c.ARMS,"Only scheduled TRAIN/VAL/selection callbacks")
  return None
 def sink(seed,arm,state,record):
  key=(seed,arm)
  c.require(key not in seen and seed in c.SEEDS and arm in c.ARMS
   and record["seed"]==seed and record["arm"]==arm
   and record["epochs_completed"]==40 and record["optimizer_updates"]==800
   and record["training_targets"]==list(range(52,365))
   and record["validation_targets"]==list(range(365,443))
   and record["test_accessed"]is False and len(record["history"])==40
   and counts.get(key)==800,"Exactly one full40epoch/800update selected state per arm/seed")
  c.require([x["epoch"]for x in record["history"]]==list(range(1,41))
   and [x["updates"]for x in record["history"]]==list(range(20,801,20))
   and all(x["forecast_weeks"]==78 for x in record["history"])
   and record["parameters"]==({"full":25378,"no_memory":24865}[arm]),
   "Original complete validation support, parameter counts and every epoch")
  scores=[x["mae_cases"]for x in record["history"]]
  c.require(all(math.isfinite(v)for v in scores)
   and record["selected_epoch"]==scores.index(min(scores))+1
   and record["selected_validation_mae_cases"]==min(scores),"Minimum VAL MAE with earliest exact tie")
  c.require(isinstance(state,dict)and state
   and all(getattr(t,"device",None).type=="cpu"for t in state.values()),
   "Selected CPU model state only; no optimizer or resume object")
  checkpoint=writer.save("selected_seed%d_%s.pt"%(seed,arm),state,kind="torch")
  metadata=writer.save("selection_seed%d_%s.json"%(seed,arm),dict(record,
   selected_checkpoint_ref=checkpoint,scientific_config=config,
   source_protocol_ref=b["protocol_ref"],registration_ref=b["registration_ref"],
   state_kind="selected_cpu_model_state_only",optimizer_state_saved=False,
   TEST_release_granted=False))
  seen.add(key)
  fits.append({**{k:record[k]for k in("seed","arm","selected_epoch","selected_validation_mae_cases",
   "epochs_completed","optimizer_updates","parameters","test_accessed")},
   "selected_checkpoint_ref":checkpoint,"selected_metadata_ref":metadata})
  return checkpoint
 for seed in c.SEEDS:
  runtime_guard("before_full_seed_pair_%d"%seed)
  pair=completion.fit_development_pair_v1(development,seed=seed,boundary=boundary,checkpoint_sink=sink)
  c.require(set(pair)==set(c.ARMS)and all(pair[a]["selected_checkpoint_ref"]==
   next(x["selected_checkpoint_ref"]for x in fits if x["seed"]==seed and x["arm"]==a)for a in c.ARMS),
   "Reviewed runner returned the actual bounded selections")
 required={(seed,arm)for seed in c.SEEDS for arm in c.ARMS}
 c.require(seen==required and set(counts)==required and all(n==800 for n in counts.values())
  and sum(counts.values())==4800 and validation_batches==1200 and first_seconds is not None,
  "All six real800update fits and all1200complete validation batches; no probe/replay/reduction")
 runtime_guard("after_full_six_selection_budget")
 products=[dict(seed=x["seed"],arm=x["arm"],kind="selected_checkpoint",
  **x["selected_checkpoint_ref"],selected_metadata_ref=x["selected_metadata_ref"])for x in fits]
 barrier=writer.save("six_selection_barrier.json",dict(
  schema="H061_TRAIN_six_selection_barrier_v1",status="six_TRAIN_selected_states_and_configurations_frozen",
  selected_products=products,scientific_config=config,source_protocol_ref=b["protocol_ref"],
  registration_ref=b["registration_ref"],completion_source_ref=b["completion_ref"],generic_core_ref=b["generic_core_ref"],
  archive_ref=b["archive_ref"],data_source_manifest_ref=b["data_source_manifest_ref"],county_order=b["counties"],
  TRAIN_only_scaler=dict(fit_rows=[0,365],mean=development.mean.tolist(),scale=development.scale.tolist()),
  TEST_release_granted=False,TEST_accessed=False,scoring_authorized=False,
  comparator_states="deferred_to_separate_ROOT_release",legacy_GRU_status="unavailable_payload_not_accepted",
  chronology_diagnostic="deferred_to_separate_ROOT_release"))
 return writer.save("completed_TRAIN.json",dict(
  schema="H061_TRAIN_six_selection_result_v1",status="completed_TreeLoom_Chickenpox_TRAIN_six_selections",
  seeds=list(c.SEEDS),arms=list(c.ARMS),fits=fits,total_optimizer_updates=4800,selected_count=6,
  validation_prediction_batches=validation_batches,county_count=20,history_weeks=52,
  selection_barrier_ref=barrier,first_scheduled_update_timed_inline=True,
  first_scheduled_update_wall_seconds=first_seconds,
  timing_scope="First full seed101 real update1 forward/backward/clip/AdamW step through update2 boundary; no extra numerical work",
  standalone_numerical_probe=False,TEST_accessed=False,TEST_scored=False,test_predictions_sealed=0,
  separate_ROOT_TEST_release_required=True,legacy_GRU_status="unavailable_payload_not_accepted",
  comparator_states="deferred_to_separate_ROOT_release",chronology_diagnostic="deferred_to_separate_ROOT_release",
  automatic_retry_resume_or_fallback=False))
