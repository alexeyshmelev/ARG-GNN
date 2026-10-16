"""Narrow ROOT stage admission; reuses exact existing owned CPU execute() algorithm."""
from pathlib import Path
import argparse,importlib,json,os,sys
from research.reviews.FLARE_original_remaining_CPU_execution_source_20261017_v1 import root_contract_v1 as c
def validate_completion(value,q):
 c.require(value.get("status")==c.STATUS[q["stage"]]and value.get("parent_id")==q["parent_id"]
  and value.get("query_ancestry_truth_opened")is False,"Exact original FLARE completed metadata")
 row=c.completion()._context(q["context_ref"],parent_id=q["parent_id"])["targets"][q["parent_id"]]
 c.require(value.get("class_order")==["EUR","EAS","AFR"]and value.get("focal_registry_ref")==row["focal_registry_ref"],"Exact original classes/focals")
 if q["stage"]=="convert":
  c.require(value.get("observation_ref")==row["observation_ref"]and value.get("input_sites")==row["input_sites"]
   and value.get("reference_haplotypes")==600 and value.get("query_haplotypes")==1000
   and value.get("common_focal_records")==10000 and value.get("unsupported_sites_dropped")==0
   and value.get("native_options")==c.completion().OPTIONS and value.get("native_MAC_threshold")==50,"Complete unchanged converter")
 elif q["stage"]=="native":
  wait=c.read(value["native_actual_wait_ref"])
  c.require(value.get("conversion_ref")==q["conversion_ref"]and value.get("native_options")==c.completion().OPTIONS
   and value.get("native_outputs_modified")is False and wait.get("actual_direct_native_wait")is True
   and wait.get("returncode")==0 and wait.get("owned_process_group_ESRCH")is True
   and wait.get("operational_failure")is None,"Actual unchanged native0/drain")
 else:
  c.require(value.get("native_run_ref")==q["native_run_ref"]and value.get("conversion_ref")==q["conversion_ref"]
   and value.get("records")==10000 and value.get("native_AN1_AN2_calls_used_unchanged")is True
   and value.get("rounded_ANP_renormalized")is False and value.get("raw_rounded_ANP_modified")is False
   and value.get("native_VCF_modified")is False,"Original AN1/AN2 hard calls and raw ANP preserved")
def framework():
 # Import authenticates only compact source/dependency metadata. No scientific or guard calls.
 c.raw(c.FRAMEWORK)
 m=importlib.import_module("research.reviews.American_TreeLoom_matched_CPU_execution_source_20261017_v4.root_driver_v1")
 m.c=c;m.ROOT=c.ROOT;m.HERE=c.HERE;m.SOURCE=c.DRIVER;m.PIDFD=c.PIDFD;m.OWNED=c.OWNED
 for name in ("local","ref","read","load","put","current","guard"):setattr(m,name,getattr(c,name))
 m.need=c.require;m.validate_completion=validate_completion
 return m
def admit(request_ref):
 q,caps,_=c.validate_request(request_ref);packet=c.local(q["packet_directory"]);output=q["output_directory"]
 c.require(set(q["cpu_ids"])<=os.sched_getaffinity(0),"Four currently available ROOT CPU identities")
 c.require(not packet.exists()and not c.local(output).exists(),"Reserved targets absent before reserve")
 prep=c.local(q["preparation_directory"]);stem=packet.name
 cfgref=c.put(prep/(stem+".config.json"),c.make_config(q));sources=c.sources()["source_sha256"]
 closure=c.put(prep/(stem+".source_and_input_closure.json"),dict(source_sha256=sources,root_stage_request_ref=request_ref))
 g=c.guard(False);impl=c.ref(g.budget.SOURCE_REL);launcher=c.ref(g.budget.LAUNCHER_REL)
 with g.budget.v4.mutex(c.ROOT):state=g._state_locked(impl["sha256"],launcher["sha256"],fresh=True)
 consumers=g.consumer_bindings_v1(list(state[3].values()),c.direct_roles(request_ref,q,cfgref,closure))
 job="FLARE-original-"+q["stage"]+"-"+packet.name
 reserve=dict(key="H041/"+job,output=str(c.ROOT/output),auxiliary_scopes=[str(packet)],
  reserved_bytes=caps["reservation_bytes"],consumer_bindings=consumers)
 paths=g.budget.v4.scope_spec(c.ROOT,reserve)
 c.require(reserve["key"]not in state[0]and reserve["key"]not in g._policy()["never_credit_keys"]
  and not any(p.exists()for p in paths),"Actual original scope schema/all refs preflight; new key/fresh paths")
 authority=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(state[2]),bindings=state[3])
 authority.update(active=True,expires_at_utc=q["expires_at_utc"])
 ar=c.put(prep/(stem+".reserve_authority.json"),authority)
 receipt=g.apply_reserve_v1(ar,expected_source_sha256=impl["sha256"],expected_launcher_sha256=launcher["sha256"])
 c.require(not packet.exists()and not c.local(output).exists(),"Target paths absent through actual successful reserve")
 packet.mkdir();rr=c.put(packet/"reserve_receipt.json",receipt)
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,
  job_id=job,output_directory=output,reserved_new_bytes=caps["reservation_bytes"],reservation=receipt["value"],
  initial_ledger_pins=receipt["initial_ledger_pins"],created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
  completed_overlay_policy_ref=c.POLICY_REF,storage_policy_selector_ref=c.SELECTOR_REF,
  budget_components=dict(implementation=impl,launcher=launcher,consumer_adapter=c.GUARD_REF))
 sr=c.put(packet/"storage.json",storage)
 s=dict(schema="FLARE_remaining_CPU_admitted_input_v1",active=True,status="active",root_authorizer="root_agent",execution_authorized=True,
  GPU_authorized=False,physical_gpu=None,parent_id=q["parent_id"],phase=q["phase"],stage=q["stage"],job_id=job,packet_directory=q["packet_directory"],
  output_directory=output,worker_auxiliary_scope=q["packet_directory"]+"/worker_aux",root_stage_request_ref=request_ref,
  root_driver_ref=c.ref(c.DRIVER),main_source_ref=c.ref(c.ENTRY),source_sha256=sources,transitive_source_closure_ref=closure,
  refs=dict(config=cfgref,runtime=q["runtime_ref"],storage=sr),runtime_ref=q["runtime_ref"],storage=storage,
  created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"])
 ir=c.put(packet/"input.json",s);c.input_spec(ir)
 c.put(packet/"admitted.json",dict(input_ref=ir,driver_ref=c.ref(c.DRIVER),storage=storage,reserve_receipt_ref=rr,
  consumer_binding_count=len(consumers),scientific_execution_started=False))
 return dict(input_ref=ir,consumer_binding_count=len(consumers),admission_only=True,
  execution_argv=["setsid",str(c.ROOT/".runtime/env/bin/python"),"-B","-s","-m",c.HERE.replace("/",".")+".root_driver_v1","--execute-ref",json.dumps(ir)])
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group();m.add_argument("--admit-ref");m.add_argument("--execute-ref")
 a=p.parse_args(argv)
 if not(a.admit_ref or a.execute_ref):
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False,truth=False)));return 0
 c.require(Path.cwd()==c.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"ROOT exact -B -s CPU interpreter")
 result=admit(json.loads(a.admit_ref))if a.admit_ref else framework().execute(json.loads(a.execute_ref))
 print(json.dumps(result,sort_keys=True),flush=True);return result.get("returncode",0)
if __name__=="__main__":raise SystemExit(main())
