from pathlib import Path
import os,stat,json,hashlib,time,datetime
REVIEW=Path("research/reviews/scientific_progress_fresh_oct17_v1")
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def identity(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink]
def read_stable(p,decode=False):
 p=Path(p);fd=os.open(str(p),os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd);assert stat.S_ISREG(before.st_mode)
  sha=hashlib.sha256();chunks=[]
  while True:
   block=os.read(fd,1024*1024)
   if not block:break
   sha.update(block)
   if decode:chunks.append(block)
  after=os.fstat(fd);named=p.lstat()
  assert identity(before)==identity(after)==identity(named),str(p)
  ref=dict(path=str(p),sha256=sha.hexdigest(),bytes=before.st_size,identity=identity(before))
  return (ref,json.loads(b"".join(chunks))) if decode else ref
 finally:os.close(fd)
def maybe_json(p):
 try:return read_stable(p,True)
 except FileNotFoundError:return None,None
def save(name,value):
 data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 with (REVIEW/name).open("xb") as f:f.write(data)
 return dict(path=str(REVIEW/name),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
def proc(pid,start):
 try:
  raw=Path("/proc/%d/stat"%pid).read_text();a=raw[raw.rfind(")")+2:].split()
  result=dict(pid=pid,expected_start=start,actual_start=int(a[19]),identity_matches=int(a[19])==start,state=a[0],cpu_ticks=int(a[11])+int(a[12]),rss_bytes=int(a[21])*os.sysconf("SC_PAGE_SIZE"),observed_utc=now())
  if not result["identity_matches"]:return result
  result["cmdline"]=Path("/proc/%d/cmdline"%pid).read_bytes().replace(b"\0",b" ").decode()
  result["wchan"]=Path("/proc/%d/wchan"%pid).read_text().strip()
  status=Path("/proc/%d/status"%pid).read_text().splitlines()
  result["status_fields"]={line.split(":",1)[0]:line.split(":",1)[1].strip() for line in status if line.split(":",1)[0] in ["Threads","Cpus_allowed_list","VmHWM","voluntary_ctxt_switches","nonvoluntary_ctxt_switches"]}
  return result
 except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,expected_start=start,absent=True,observed_utc=now())
specs=[
 dict(name="H058_CPU24151",packet="research/reviews/H058_CPU_ROOT_contextfix_20261017_v1",sha="4a42dfb9ca7341427fcc8a15b48f690220791d51075228b4dd73ae022118b1e4",pid=3936924,start=488124815,arms=["same_marker_aligned","same_marker_alignment_null"],target=960,runner="research/reviews/H058_context_terminal_successor_source_20261017_v1/producer_v1.py"),
 dict(name="ordered_marker_CPU95424",packet="research/reviews/ordered_marker_CPU_ROOT_root01_oct17_v1",sha="99e64923b31aab758966ad4a9505f3fc5ad8198bc455843fa0f5c964cc3b7776",pid=3943291,start=488211434,arms=["ordered_marker_tree_feedback","ordered_marker_sample_readout"],target=960,runner="research/reviews/ordered_marker_learning_integration_source_20261017_v2/producer_v2.py"),
 dict(name="H060_old_CPU36502",packet="research/reviews/H060_isolated_CPU_ROOT_actual_tx_20261017_v1",sha="dfb5cd92d697b8f6359268e98db585d691c6450e6a550a265b29e8bc3921445d",pid=3889463,start=487457986,arms=["stable_leaf_memory","no_added_memory"],target=600,runner="research/reviews/H060_isolated_hold_compatibility_source_20261017_v1/warmstart_train_v2.py"),
 dict(name="H060_fast_CPU97638",packet="research/reviews/H060_isolated_CPU_ROOT_actual_fast_observed_root01_oct17_v1",sha="e79abe3bb140676c35ada8a8d32f06490f5f6fe758ba9df8fd626746ea0a6d3e",pid=3965209,start=488505215,arms=["stable_leaf_memory","no_added_memory"],target=600,runner="research/reviews/H060_fast_observed_full_training_source_20261017_v1/warmstart_train_v2.py")]
start_utc=now();sample_start=time.monotonic()
activity_before={s["name"]:proc(s["pid"],s["start"]) for s in specs}
activity_before["native_GPU0_70445"]=proc(3752065,485504904)
jobs=[]
for s in specs:
 inp_ref,inp=read_stable(s["packet"]+"/input.json",True);assert inp_ref["sha256"]==s["sha"]
 output=Path(inp["output_directory"])
 row=dict(name=s["name"],input_ref=inp_ref,output_directory=str(output),target_new_optimizer_updates_per_arm=s["target"],observed_utc=now(),output_inventory=[dict(name=p.name,bytes=p.lstat().st_size,directory=p.is_dir()) for p in sorted(output.iterdir())])
 source_refs=[]
 for p in [inp["main_source_ref"]["path"],s["runner"]]:
  source_ref=read_stable(p);expected=inp["source_sha256"].get(p);source_ref["input_pin_sha256"]=expected;source_ref["matches_input_pin"]=source_ref["sha256"]==expected;assert source_ref["matches_input_pin"];source_refs.append(source_ref)
 row["source_refs"]=source_refs
 row["new_optimizer_updates_verified_lower_bound"]={a:0 for a in s["arms"]};row["first_scheduled_updates"]={}
 for arm in s["arms"]:
  ref,j=maybe_json(output/(arm+".first_scheduled_step.json"))
  if j is not None:
   row["first_scheduled_updates"][arm]=dict(receipt_ref=ref,optimizer_steps=j["first_scheduled_step"]["optimizer_steps"],finite_FP32=j["finite_FP32"],seconds=j.get("seconds"),parent_id=j.get("parent_id"),parameter_count=j["first_scheduled_step"]["parameter_count"],loss=j.get("loss"),gradient_norm=j.get("gradient_norm"),parent_seconds_to_receipt=j.get("parent_seconds_to_receipt"),main_peak_RSS_bytes=j["first_scheduled_step"].get("CPU_runtime_cache",{}).get("main_peak_RSS_bytes"))
   row["new_optimizer_updates_verified_lower_bound"][arm]=j["first_scheduled_step"]["optimizer_steps"]
 row["completed_pass_receipts"]=[]
 for at in range(1,31):
  ref,j=maybe_json(output/("TRAIN_pass_%02d.json"%at))
  if j is not None:
   row["completed_pass_receipts"].append(dict(receipt_ref=ref,pass_index=at,steps=j.get("steps"),seconds=j.get("seconds")))
   for arm,v in j.get("steps",{}).items():row["new_optimizer_updates_verified_lower_bound"][arm]=max(row["new_optimizer_updates_verified_lower_bound"].get(arm,0),v)
 row["receipts"]={}
 for name in ["ROOT_started.json","started.json","initialization.json","CPU_complete_phase_preparation_cache.json","TRAIN_observation_barrier.json","TRAIN_external_target_joins.json","matched_initialization.json","completed.json","ROOT_completed.json","completed_training.json","failed.json","worker_failure.json","failed_training.json"]:
  ref,j=maybe_json(output/name)
  if j is None:continue
  if name=="CPU_complete_phase_preparation_cache.json":
   summary={k:j.get(k) for k in ["parents","episode_builds","parent_arm_payloads","packed_build_seconds","actual_packed_numeric_bytes","numeric_model_operations_or_optimizer_updates","forecast"]}
  elif name=="initialization.json":summary={k:j.get(k) for k in ["hypothesis_id","initialization","observations_seal"]}
  elif name in ["ROOT_started.json","started.json"]:summary=j
  elif name=="TRAIN_observation_barrier.json":summary={k:j.get(k) for k in ["all20_ready","observation_only","truth_resolver_calls_before_barrier"]}
  else:summary={k:j.get(k) for k in ["status","steps","logical_steps","error_type","error","exception_type","message","additional_updates_per_arm","full_scientific_training_complete"] if k in j}
  row["receipts"][name]=dict(ref=ref,summary=summary)
 row["aux_terminal_evidence"]={}
 for name in ["actual_wait.json","worker_log.json"]:
  ref,j=maybe_json(Path(inp["worker_auxiliary_scope"])/name)
  if j is not None:row["aux_terminal_evidence"][name]=dict(ref=ref,content=j)
 row["completion_file_present"]=any(n in row["receipts"] for n in ["completed.json","completed_training.json","ROOT_completed.json"])
 if "H060" in s["name"]:
  row["inherited_saved600_backbone_is_not_new_updates"]=True
  row["observation_barrier_present"]="TRAIN_observation_barrier.json" in row["receipts"]
  row["phase_evidence"]="before_durable_all20_observation_barrier_and_before_new_optimizer_updates" if not row["observation_barrier_present"] else "barrier_present_inspect_later_products"
 else:row["phase_evidence"]="actual_first_scheduled_optimizer_update_attested" if row["first_scheduled_updates"] else "before_observation_preparation_or_first_update_receipt"
 row["exact_live_optimizer_count_available"]=False
 row["zero_lower_bound_is_not_CPU_activity_inference"]=True
 jobs.append(row)
prior_ref,prior=read_stable("research/reviews/research_execution_ROOT_20261017_v14/C3_prediction_products.root_v1.json",True)
c3=[]
for spec in prior["parents"]:
 inp_ref,inp=read_stable(spec["input_ref"]["path"],True);assert inp_ref["sha256"]==spec["input_ref"]["sha256"]
 output=Path(spec["output_directory"]);assert str(output)==inp["output_directory"]
 parent=spec["parent_id"];row=dict(parent_id=parent,input_ref=inp_ref,output_directory=str(output),products=[],missing_epochs=[],unsealed_product_epochs=[],expected_count=6,observed_utc=now())
 for epoch in [5,10,15,20,25,30]:
  base=parent+"_epoch%03d"%epoch;seal_ref,seal=maybe_json(output/(base+"_prediction_seal.json"))
  product=output/(base+"_predictions.npz")
  if seal is None:
   row["missing_epochs"].append(epoch)
   if product.exists():row["unsealed_product_epochs"].append(epoch)
   continue
  pred_ref=read_stable(product)
  assert seal["prediction_ref"]["path"]==str(product) and seal["prediction_ref"]["sha256"]==pred_ref["sha256"]
  assert seal["parent_id"]==parent and seal["epoch"]==epoch and seal["focal_records"]==10000 and seal["evaluation_length_bp"]==30000000 and seal["private_truth_opened"] is False
  row["products"].append(dict(epoch=epoch,seal_ref=seal_ref,prediction_ref=pred_ref,focal_records=seal["focal_records"],evaluation_length_bp=seal["evaluation_length_bp"],private_truth_opened=seal["private_truth_opened"],global_step=seal.get("global_step")))
 row["sealed_product_count"]=len(row["products"])
 row["completion_receipts"]={}
 for name in ["completed.json","ROOT_completed.json"]:
  ref,j=maybe_json(output/name)
  if j is not None:row["completion_receipts"][name]=dict(ref=ref,status=j.get("status"))
 row["completed_file_exists"]=spec["completed_file_exists"] if not row["completion_receipts"] else True
 c3.append(row)
native_ref,native=read_stable("research/reviews/research_execution_ROOT_20261017_v14/saved_native_optimizer_progress.root_v1.json",True)
prior_cpu_ref=read_stable("research/reviews/research_execution_ROOT_20261017_v14/targeted_owned_CPU_observation.root_v2.json")
remaining=20-(time.monotonic()-sample_start)
if remaining>0:time.sleep(remaining)
activity_after={s["name"]:proc(s["pid"],s["start"]) for s in specs};activity_after["native_GPU0_70445"]=proc(3752065,485504904)
elapsed=time.monotonic()-sample_start;activity={}
for name,before in activity_before.items():
 after=activity_after[name];row=dict(before=before,after=after)
 if before.get("identity_matches") and after.get("identity_matches"):
  row["CPU_seconds"]=float(after["cpu_ticks"]-before["cpu_ticks"])/os.sysconf("SC_CLK_TCK");row["average_CPU_cores"]=row["CPU_seconds"]/elapsed
 activity[name]=row
activity_ref=save("targeted_runtime_activity.v1.json",dict(schema="targeted_readonly_owned_worker_activity_v1",observed_from_utc=start_utc,observed_to_utc=now(),elapsed_monotonic_seconds=elapsed,clock_ticks_per_second=os.sysconf("SC_CLK_TCK"),workers=activity,CPU_activity_does_not_establish_optimizer_updates=True,all_owned_tree_or_total_capacity_claimed=False))
result=dict(schema="scientific_progress_readonly_fresh_oct17_v1",observed_from_utc=start_utc,observed_to_utc=now(),scope="Exact named current output/input/worker products only; metadata and small JSON reads, stable streamed prediction hashing, targeted known PID identities; no scientific imports or checkpoint payload loads",jobs=jobs,C3=dict(parents=c3,sealed_product_count=sum(r["sealed_product_count"] for r in c3),expected_product_count=30,all30_pretruth_barrier_complete=sum(r["sealed_product_count"] for r in c3)==30,prior_observation_ref=prior_ref,prediction_arrays_or_truth_loaded=False,all_seals_match_stable_prediction_hashes=True,ancestry_accuracy_or_checkpoint_selection_performed=False),native=dict(prior_saved_optimizer_audit_ref=native_ref,audit_observed_utc=native["observed_utc"],saved_optimizer_steps=native["optimizer_unique_step_values"],saved_batch=native["saved_batch"],saved_epoch=native["saved_epoch"],saved_batch_in_epoch=native["saved_batch_in_epoch"],optimizer_parameter_state_count=native["optimizer_parameter_state_count"],fresh_checkpoint_payload_load_performed=False,current_count_not_extrapolated_from_activity=True),runtime_activity_ref=activity_ref,prior_runtime_activity_ref=prior_cpu_ref,source_mutation_or_job_control_or_accounting_mutation=False,scientific_success_claimed=False)
result_ref=save("status.v1.json",result)
print(json.dumps(dict(status_ref=result_ref,runtime_activity_ref=activity_ref,jobs=[dict(name=j["name"],verified_new_steps_lower_bound=j["new_optimizer_updates_verified_lower_bound"],files=[p["name"] for p in j["output_inventory"]],phase=j["phase_evidence"]) for j in jobs],C3_total=result["C3"]["sealed_product_count"],C3_parents=[dict(parent_id=r["parent_id"],count=r["sealed_product_count"],missing=r["missing_epochs"]) for r in c3],activity_cores={k:v.get("average_CPU_cores") for k,v in activity.items()},observed_to_utc=result["observed_to_utc"]),sort_keys=True))
