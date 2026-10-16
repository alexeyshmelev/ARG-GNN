"""Inactive H061 stage driver; finite execute body is the sealed H061 lifecycle."""
from pathlib import Path
import argparse,json,os,sys
from research.reviews.H061_postselection_execution_source_20261017_v1 import root_contract_v1 as c
def validate_completion(value,q):
 c.require(value.get("status")==c.STATUSES[c.PHASE]and value.get("parent_id")==c.PARENT
  and value.get("stage")==q["stage"]and value.get("root_driver_ref")==c.ref(c.DRIVER)
  and value.get("scientific_config")==q["scientific_config"]and value.get("upstream")==c.upstream(q),
  "Exact completed admitted stage envelope")
 result=c.read(value["result_ref"]);stage=q["stage"];directory=q["output_directory"]
 c.require(value["result_ref"]["path"]==directory+"/completed_"+stage+".json"
  and result.get("status")=="completed_H061_"+stage and result.get("stage")==stage
  and result.get("new_core_fits")==0 and result.get("new_optimizer_updates")==0
  and result.get("legacy_GRU_status")=="unavailable_payload_not_accepted","Exact completed scientific stage; no new fits")
 barrier=c.read(result["barrier_ref"])
 c.require(result["barrier_ref"]["path"]==directory+"/stage_barrier.json"
  and barrier.get("TEST_scored")is(stage=="score"),"Actual final barrier has exact scoring scope")
 if stage=="freeze":
  c.require(result.get("TEST_accessed")is False and barrier.get("schema")=="H061_complete_selection_comparator_barrier_v1"
   and len(barrier["selected_products"])==6 and barrier["comparators"]==list(c.COMPARATORS),"All six selections and three controls frozen")
  products=[barrier["comparator_state_ref"]]
 elif stage=="predict":
  c.require(result.get("TEST_scored")is False and barrier.get("schema")=="H061_complete_prediction_barrier_v1"
   and len(barrier["predictions"])==6 and len(barrier["chronology_predictions"])==6
   and len(barrier["comparator_predictions"])==3,"Full six predictions, six chronology controls and three fixed comparators")
  products=barrier["predictions"]+barrier["chronology_predictions"]+barrier["comparator_predictions"]
 else:
  c.require(result.get("TEST_scored")is True and barrier.get("schema")=="H061_score_release_barrier_v1"
   and barrier.get("prediction_barrier_ref")is not None,"Separately released score completion")
  products=[barrier["scores_ref"]]
 for record in products:
  c.require(Path(record["path"]).parent==c.local(directory).relative_to(c.ROOT),"All stage products remain in actual charged output")
  c.require(len(c.raw({k:record[k]for k in("path","sha256")}))<=c.CAPS["total_JSON_bytes"],"Every actual product hash and bounded bytes")
def admit(request_ref):
 q,caps,_=c.validate_request(request_ref);packet=c.local(q["packet_directory"]);output=q["output_directory"]
 c.require(set(q["cpu_ids"])<=os.sched_getaffinity(0),"ROOT supplies exact CPU223 capacity")
 c.require(not packet.exists()and not c.local(output).exists(),"Fresh exclusive one-shot packet/output")
 g=c.guard(False);preparation=c.local(q["preparation_directory"]);stem=packet.name
 cfgref=c.put(preparation/(stem+".config.json"),c.make_config(q,caps));sources=c.sources()["source_sha256"]
 closure=c.put(preparation/(stem+".source_and_input_closure.json"),dict(source_sha256=sources,root_stage_request_ref=request_ref))
 implementation=c.ref(g.budget.SOURCE_REL);launcher=c.ref(g.budget.LAUNCHER_REL)
 with g.budget.v4.mutex(c.ROOT):state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
 consumers=g.consumer_bindings_v1(list(state[3].values()),c.direct_roles(request_ref,q,cfgref,closure))
 job="H061-postselection-"+q["stage"]+"-"+packet.name
 reserve=dict(key="H061/"+job,output=str(c.ROOT/output),auxiliary_scopes=[str(packet)],
  reserved_bytes=caps["reservation_bytes"],consumer_bindings=consumers)
 # Retain reviewed V4 whole-scope/role preflight BEFORE authority/accounting mutation.
 paths=g.budget.v4.scope_spec(c.ROOT,reserve)
 c.require(reserve["key"]not in state[0]and reserve["key"]not in g._policy()["never_credit_keys"]
  and not any(p.exists()for p in paths),"New key/fresh scopes; complete role union must fit64")
 authority=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(state[2]),bindings=state[3])
 authority.update(active=True,expires_at_utc=q["expires_at_utc"])
 ar=c.put(preparation/(stem+".reserve_authority.json"),authority)
 receipt=g.apply_reserve_v1(ar,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
 c.require(not packet.exists()and not c.local(output).exists(),"Paths remain absent through real reserve")
 packet.mkdir();rr=c.put(packet/"reserve_receipt.json",receipt)
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,
  job_id=job,output_directory=output,reserved_new_bytes=caps["reservation_bytes"],reservation=receipt["value"],
  initial_ledger_pins=receipt["initial_ledger_pins"],created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
  completed_overlay_policy_ref=c.POLICY_REF,storage_policy_selector_ref=c.SELECTOR_REF,
  budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=c.GUARD_REF))
 sr=c.put(packet/"storage.json",storage)
 s=dict(schema="H061_postselection_admitted_input_v1",status="active",active=True,root_authorizer="root_agent",execution_authorized=True,
  GPU_authorized=False,physical_gpu=None,parent_id=c.PARENT,phase=c.PHASE,stage=q["stage"],job_id=job,packet_directory=q["packet_directory"],
  output_directory=output,worker_auxiliary_scope=q["packet_directory"]+"/worker_aux",root_stage_request_ref=request_ref,
  root_driver_ref=c.ref(c.DRIVER),main_source_ref=c.ref(c.ENTRY),source_sha256=sources,transitive_source_closure_ref=closure,
  refs=dict(config=cfgref,runtime=q["runtime_ref"],storage=sr),runtime_ref=q["runtime_ref"],storage=storage,
  created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"])
 ir=c.put(packet/"input.json",s);c.input_spec(ir)
 c.put(packet/"admitted.json",dict(input_ref=ir,driver_ref=c.ref(c.DRIVER),storage=storage,reserve_receipt_ref=rr,
  consumer_binding_count=len(consumers),scientific_execution_started=False,stage=q["stage"]))
 return dict(input_ref=ir,consumer_binding_count=len(consumers),admission_only=True,
  execution_argv=["setsid",str(c.ROOT/".runtime/env/bin/python"),"-B","-s","-m",c.HERE.replace("/",".")+".root_driver_v1","--execute-ref",json.dumps(ir)])
def execute(record):
 # Isolated namespace: all finite pidfd/wait/drain/time/RSS/scan algorithms are the
 # byte-identical previously reviewed H061 driver body; only contract bindings differ.
 stage=c.read(record)["stage"]
 def recorded_put(path,value,cap=2**20):
  if Path(path).name=="actual_wait.json":
   value=dict(value,postselection_stage=stage,TEST_scoring_authorized=stage=="score",
    TEST_scoring_completed=stage=="score"and value.get("completed_ref")is not None
     and value.get("operational_failure")is None,
    outer_truth_scored_scope="finite_supervisor_process_only")
  return c.put(path,value,cap=cap)
 base=c.load(c.binding()["lifecycle_driver_ref"],"_H061_postselection_sealed_finite_lifecycle")
 for name,value in dict(c=c,ROOT=c.ROOT,HERE=c.HERE,SOURCE=c.DRIVER,PIDFD=c.PIDFD,OWNED=c.OWNED,
  need=c.require,local=c.local,ref=c.ref,read=c.read,load=c.load,put=recorded_put,current=c.current,guard=c.guard,
  validate_completion=validate_completion).items():setattr(base,name,value)
 result=base.execute(record)
 result.update(postselection_stage=stage,TEST_scoring_authorized=stage=="score",
  TEST_scoring_completed=stage=="score"and result.get("returncode")==0 and result.get("completed_ref")is not None,
  outer_truth_scored_scope="finite_supervisor_process_only")
 return result
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group();m.add_argument("--admit-ref");m.add_argument("--execute-ref");a=p.parse_args(argv)
 if not(a.admit_ref or a.execute_ref):
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False,training_updates=0)));return 0
 c.require(Path.cwd()==c.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"Exact ROOT -B -s repository interpreter")
 result=admit(json.loads(a.admit_ref))if a.admit_ref else execute(json.loads(a.execute_ref))
 print(json.dumps(result,sort_keys=True),flush=True);return result.get("returncode",0)
if __name__=="__main__":raise SystemExit(main())
