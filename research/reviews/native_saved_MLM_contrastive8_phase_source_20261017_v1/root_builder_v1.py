"""Inactive-by-default ROOT metadata builder. Actual operations require exact active reviewed ROOT requests."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,json,os,re,stat,sys
from . import contract_v1 as c
from . import science_binding_v1 as science
from .worker_context_v1 import check_external_prerequisites_v1,prerequisite_metadata_v1,PendingPrerequisites
EVENT_METADATA_ALLOWANCE_BYTES=524288
RESERVE_EVENT_JSON_MAX_BYTES=65536
RESERVE_EVENT_NODE_CHARGE_BYTES=4096
MAX_ROOT_RECORD_BYTES=EVENT_METADATA_ALLOWANCE_BYTES-RESERVE_EVENT_JSON_MAX_BYTES-RESERVE_EVENT_NODE_CHARGE_BYTES
MAX_ROOT_RECORD_FILES=64
CONFIG=c.HERE+"/scientific_config.FROZEN.v1.json"
def _request_v1(record,action):
 q=c.read(record);c.active(q)
 c.require(q.get("schema")=="native_saved_MLM_contrastive8_ROOT_request_v1"and q.get("action")==action
  and q.get("source_manifest_ref")==c.ref(c.MANIFEST),"Exact reviewed ROOT builder request action/source")
 review=c.read(q["source_review_ref"])
 candidate=review.get("candidate_manifest_ref",review.get("candidate_source_manifest_ref",review.get("source_manifest_ref")))
 c.require(candidate==c.ref(c.MANIFEST)and review.get("independent")is True
  and review.get("verdict")in("SOURCE_PASS","PASS_SOURCE_ONLY","PASS")
  and review.get("blockers",review.get("blocking_findings"))==[],"Whole operational integration review, no cap-only authority")
 c.configure_selector_v1(q["storage_policy_selector_ref"])
 c.require(q["capacity_guard_ref"]==c.GUARD and q["capacity_policy_ref"]==c.POLICY,
  "ROOT explicitly supplies same actual activated Census policy and selector, no inferred route")
 c.require(q.get("accounting_coordinator_ref")==c.COORDINATOR_REF and q.get("batch_input_ref")==c.BATCH_INPUT_REF,
  "Exact actual v3 accounting coordinator and genuine batch input, no guessed lock")
 return q
def _record_directory_v1(instance):
 c.require(type(instance)is str and re.fullmatch("[a-z][a-z0-9_]{0,39}",instance)is not None,"Exact fresh bounded ROOT instance")
 return c.local(c.HERE+"/ROOT_records_"+instance)
def _record_budget_v1(directory,additional=0):
 total=0;count=0
 def scan(p,depth):
  nonlocal total,count
  total+=4096
  c.require(depth<=1 and p.is_dir()and not p.is_symlink(),"Bounded canonical own ROOT metadata directory")
  for child in p.iterdir():
   c.require(not child.is_symlink(),"No linked ROOT record")
   if child.is_dir():scan(child,depth+1);continue
   st=child.lstat();c.require(stat.S_ISREG(st.st_mode)and child.suffix==".json"and st.st_nlink==1 and st.st_size<=65536,"Only bounded own JSON records")
   total+=st.st_size+4096;count+=1
 if directory.exists():scan(directory,0)
 c.require(total+additional<=MAX_ROOT_RECORD_BYTES and count<MAX_ROOT_RECORD_FILES,
  "ROOT records plus worst original64KiB journal event and4096B node stay within unchanged512KiB event allowance")
def _save_v1(directory,name,value):
 raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 c.require(len(raw)<=65536,"Bounded ROOT metadata, not scientific output")
 _record_budget_v1(directory.parent if directory.name.startswith("phase_")else directory,len(raw)+8192)
 return c.save(directory,name,value)
def _science_request_bindings_v1(q):
 return dict(ROOT_saved_MLM_artifact_acceptance_ref=q["ROOT_saved_MLM_artifact_acceptance_ref"],
  saved_MLM_scientific_integration_ref=c.SAVED_INTEGRATION_REF,
  storage_policy_selector_ref=q["storage_policy_selector_ref"])
def qualification_spec_v1(record):
 return dict(parent_id=c.PARENT,refs=dict(preparation=c.PREP_REF,preparation_config=c.PREP_CONFIG_REF,counts=c.COUNTS_REF,
  saved_MLM_qualification=record,fit_completion=record))
def master_metadata_v1(q,*,config_ref,storage_ref,request_ref):
 instance=q["instance"];job="native-saved-MLM-contrastive8-"+instance
 return dict(schema="native_VAL000_downstream_ROOT_master_input_v1",root_authorizer="root_agent",status="active",template_only=False,
  execution_authorized=True,source_review_passed=True,source_review_ref=q["source_review_ref"],source_manifest_ref=c.ref(c.MANIFEST),
  created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],job_id=job,
  output_directory="experiments/results/native_saved_MLM_contrastive8_"+instance,
  worker_auxiliary_scope=".runtime/tmp/native_saved_MLM_contrastive8_"+instance,
  future_storage_reserved_bytes=8*2**30,original_MLM7GiB_scope_extended=False,
  capacity_guard_ref=c.GUARD,capacity_policy_ref=c.POLICY,storage_policy_selector_ref=q["storage_policy_selector_ref"],
  config_ref=config_ref,runtime_ref=c.original.RUNTIME_REF,storage_ref=storage_ref,ROOT_master_request_ref=request_ref,
  ROOT_record_directory=str(_record_directory_v1(instance).relative_to(c.ROOT)),
  saved_MLM_route=True,saved_MLM_scientific_integration_ref=c.SAVED_INTEGRATION_REF,
  saved_MLM_scientific_binding_bridge_ref=c.ref(c.SCIENCE_BINDING),
  ROOT_saved_MLM_artifact_acceptance_ref=q["ROOT_saved_MLM_artifact_acceptance_ref"],
  MLM_prerequisites=dict(saved_artifact_acceptance_ref=q["ROOT_saved_MLM_artifact_acceptance_ref"],historical_MLM_outer_exit_code=1),
  operational_IO_source_ref=c.IO_SOURCE_REF,operational_IO_policy_ref=c.IO_POLICY_REF,owned_roster_policy_ref=c.OWNED_ROSTER_POLICY_REF)
def phase_metadata_v1(q,master,master_ref):
 phase=q["phase"];c.require(phase in c.PHASES and q.get("prerequisites")=={} and q.get("saved_MLM_qualification_ref")is None,"Only contrastive stage with accepted savedMLM, no later products")
 c.require(datetime.fromisoformat(master["created_at_utc"])<=datetime.fromisoformat(q["created_at_utc"])
  and datetime.fromisoformat(q["expires_at_utc"])<=datetime.fromisoformat(master["expires_at_utc"]),
  "Fresh phase interval remains inside genuine master, no clock extension")
 cap=q["max_cuda_reserved_bytes"]
 c.require(type(cap)is int and(phase=="contrastive"and 0<cap<=74*2**30),
  "Exact finite ROOT GPU0 allocator request; no fit claim")
 prerequisites=copy.deepcopy(q["prerequisites"]);prerequisites["MLM"]=copy.deepcopy(master["MLM_prerequisites"])
 qual=q.get("saved_MLM_qualification_ref")
 request=dict(schema="native_VAL000_downstream_ROOT_phase_authority_v1",root_authorizer="root_agent",status="active",template_only=False,
  execution_authorized=True,source_review_passed=True,source_review_ref=q["source_review_ref"],source_manifest_ref=c.ref(c.MANIFEST),
  master_input_ref=master_ref,created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],phase=phase,
  job_id=master["job_id"]+"/"+phase,runtime_ref=master["runtime_ref"],config_ref=master["config_ref"],
  capacity_guard_ref=c.GUARD,capacity_policy_ref=c.POLICY,storage_policy_selector_ref=master["storage_policy_selector_ref"],
  caps=dict(c.BASE_CAPS,max_cuda_reserved_bytes=cap),automatic_restart_authorized=False,partial_scientific_success_authorized=False,
  operational_IO_source_ref=c.IO_SOURCE_REF,operational_IO_policy_ref=c.IO_POLICY_REF,ROOT_context_source_ref=c.ref(c.CONTEXT),
  keep1_rotation_of_new_scope_authorized=True,original_author_recipe_claimed_exact=False,prerequisites=prerequisites,
  ROOT_saved_MLM_artifact_acceptance_ref=master["ROOT_saved_MLM_artifact_acceptance_ref"],
  saved_MLM_scientific_integration_ref=c.SAVED_INTEGRATION_REF,saved_MLM_qualification_ref=qual,
  saved_MLM_qualification_spec=None if qual is None else qualification_spec_v1(qual),
  direct_attachment_authority_ref=q.get("direct_attachment_authority_ref"),controller_binding_ref=q.get("controller_binding_ref"))
 return request
def prepare_master_v1(request_ref):
 q=_request_v1(request_ref,"prepare_master");directory=_record_directory_v1(q["instance"])
 c.require(not directory.exists(),"Fresh immutable ROOT master records, no retry")
 config_ref=c.ref(CONFIG);c.require(c.read(config_ref)==c.EXPECTED_CONFIG,"Actual frozen8GiB config and all original scientific predicates")
 # Source-pinned saved-artifact eligibility is metadata only; never hash/load its checkpoint.
 pre=dict(phase="contrastive",prerequisites={},**_science_request_bindings_v1(q))
 check_external_prerequisites_v1(pre)
 g=c.selected_guard_v1(consumer=False)
 c.require(g.MAX_JSON==RESERVE_EVENT_JSON_MAX_BYTES and g.META_CHARGE==EVENT_METADATA_ALLOWANCE_BYTES,
  "Exact original serialized-event maximum and unchanged per-event accounting charge")
 implementation=g._json(c.POLICY)["implementation_ref"];launcher=g._json(c.POLICY)["launcher_ref"]
 c.raw(implementation);c.raw(launcher)
 c.require(not directory.exists(),"Still fresh before ROOT metadata creation");directory.mkdir()
 master=master_metadata_v1(q,config_ref=config_ref,storage_ref=None,request_ref=request_ref)
 reservation=dict(key="H056/"+master["job_id"],output=str(c.local(master["output_directory"])),
  auxiliary_scopes=[str(c.local(master["worker_auxiliary_scope"]))],reserved_bytes=8*2**30)
 manifest=c.read(c.ref(c.MANIFEST))
 for r in manifest["executed_source_refs"]+manifest["source_authentication_refs"]:c.raw(r)
 direct=[c.GUARD,c.ref(c.MANIFEST),config_ref,master["runtime_ref"],request_ref,q["source_review_ref"],
  q["ROOT_saved_MLM_artifact_acceptance_ref"],c.SAVED_INTEGRATION_REF,c.SAVED_SCIENTIFIC_ENTRY_REF,c.SAVED_SCIENCE_REVIEW_REF]
 direct+=manifest["executed_source_refs"]+[c.COORDINATOR_REF,c.BATCH_INPUT_REF,c.OWNER4_SOURCE_REF,c.OWNER4_MANIFEST_REF,c.OWNER4_REVIEW_REF]
 coordinator=c.load(c.COORDINATOR_REF,"_contrastive8_actual_v3_accounting_coordinator")
 with coordinator.accounting_transaction(c.BATCH_INPUT_REF):
  with g.budget.v4.mutex(c.ROOT):
   state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
   reservation["consumer_bindings"]=g.consumer_bindings_v1(list(state[3].values()),direct)
   authority=g.operation_draft_v1("reserve",reservation,expected_before=g._pins(state[2]),bindings=state[3])
   authority.update(active=True,expires_at_utc=q["expires_at_utc"])
   authority_ref=_save_v1(directory,"reserve_authority.json",authority)
  receipt=g.apply_reserve_v1(authority_ref,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
 receipt_ref=_save_v1(directory,"reserve_receipt.json",receipt)
 own=receipt["value"]
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
  accounting_mode=g.MODE,job_id=master["job_id"],output_directory=master["output_directory"],
  reserved_new_bytes=8*2**30,reservation=own,created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
  budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=c.GUARD),
  completed_overlay_policy_ref=c.POLICY,storage_policy_selector_ref=q["storage_policy_selector_ref"],
  initial_ledger_pins=receipt["initial_ledger_pins"])
 storage_ref=_save_v1(directory,"storage.json",storage);master["storage_ref"]=storage_ref
 master_ref=_save_v1(directory,"master_input.json",master)
 admitted=_save_v1(directory,"admitted.json",dict(schema="native_saved_MLM_contrastive8_master_admitted_v1",
  master_input_ref=master_ref,reserve_receipt_ref=receipt_ref,consumer_binding_count=len(own["consumer_bindings"]),
  whole_scope_reservation_bytes=8*2**30,scientific_execution_started=False,measured_fit=False,
  metadata_allocation=dict(ROOT_record_cap_bytes=MAX_ROOT_RECORD_BYTES,reserve_event_JSON_max_bytes=RESERVE_EVENT_JSON_MAX_BYTES,
   reserve_event_node_charge_bytes=RESERVE_EVENT_NODE_CHARGE_BYTES,combined_event_allowance_bytes=EVENT_METADATA_ALLOWANCE_BYTES)))
 return dict(status="MASTER_ADMITTED_ONLY",master_input_ref=master_ref,admitted_ref=admitted,reserve_receipt_ref=receipt_ref,
  scientific_execution_started=False,automatic_phase_execution=False)
def _master_v1(record,q):
 master=c.read(record);c.active(master)
 c.require(master["source_manifest_ref"]==c.ref(c.MANIFEST)and master["storage_policy_selector_ref"]==q["storage_policy_selector_ref"]
  and master["capacity_guard_ref"]==c.GUARD and master["capacity_policy_ref"]==c.POLICY
  and master["saved_MLM_route"]is True and master["future_storage_reserved_bytes"]==8*2**30
  and master["source_review_ref"]==q["source_review_ref"] and master["source_review_passed"]is True
  and master["config_ref"]==c.ref(CONFIG) and c.read(master["config_ref"])==c.EXPECTED_CONFIG
  and master["runtime_ref"]==c.original.RUNTIME_REF
  and master["saved_MLM_scientific_integration_ref"]==c.SAVED_INTEGRATION_REF
  and master["saved_MLM_scientific_binding_bridge_ref"]==c.ref(c.SCIENCE_BINDING)
  and master["MLM_prerequisites"]==dict(saved_artifact_acceptance_ref=master["ROOT_saved_MLM_artifact_acceptance_ref"],historical_MLM_outer_exit_code=1),
  "Exact actual same master/selector/source and8GiB charge")
 storage=c.read(master["storage_ref"])
 c.require(storage["accounting_mode"]==c.CENSUS_MODE and storage["storage_policy_selector_ref"]==q["storage_policy_selector_ref"]
  and storage["reserved_new_bytes"]==storage["reservation"]["reserved_bytes"]==8*2**30
  and storage["reservation"]["auxiliary_scopes"]==[str(c.local(master["worker_auxiliary_scope"]))],
  "Same actual contrastive8 whole-scope stored reservation")
 g=c.selected_guard_v1(consumer=False)
 g.validate_storage_v9(master,dict(output_cap_bytes=8*2**30),storage,source_ref=c.ref(c.DRIVER),
  config_ref=master["config_ref"],runtime_ref=master["runtime_ref"])
 return master
def prepare_phase_v1(request_ref):
 q=_request_v1(request_ref,"prepare_phase");master=_master_v1(q["master_input_ref"],q)
 authority=phase_metadata_v1(q,master,q["master_input_ref"])
 from . import direct_gpu0_v1 as direct
 direct.configure_controller_binding_v1(authority)
 check_external_prerequisites_v1(authority)
 directory=c.local(master["ROOT_record_directory"])/("phase_"+q["phase"])
 c.require(not directory.exists(),"Fresh single phase preparation; no replay");directory.mkdir()
 authority_ref=_save_v1(directory,"phase_authority.json",authority)
 phase=q["phase"]
 spec=dict(schema="native_VAL000_downstream_ROOT_phase_input_v1",root_authorizer="root_agent",status="active",template_only=False,
  execution_authorized=True,created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
  root_stage_request_ref=authority_ref,job_id=authority["job_id"],phase=phase,
  output_directory=master["output_directory"]+"/"+phase,worker_auxiliary_scope=master["worker_auxiliary_scope"]+"/"+phase,
  storage_policy_selector_ref=q["storage_policy_selector_ref"],
  refs=dict(config=master["config_ref"],runtime=master["runtime_ref"],storage=master["storage_ref"]))
 input_ref=_save_v1(directory,"input.json",spec)
 c.set_amendment_v1(input_ref);c.validate_input_v1(input_ref)
 argv=["setsid",str(c.ROOT/".runtime/env/bin/python"),"-B","-s","-m",c.HERE.replace("/",".")+".execution_v1",
  "--execute","--input-ref",json.dumps(input_ref),"--amendment-ref",json.dumps(input_ref)]
 prepared=_save_v1(directory,"prepared.json",dict(schema="native_saved_MLM_contrastive8_phase_prepared_v1",input_ref=input_ref,
  phase=phase,execute_argv=argv,scientific_execution_started=False,measured_fit=False))
 return dict(status="PHASE_PREPARED_ONLY",input_ref=input_ref,prepared_ref=prepared,execute_argv=argv,automatic_execution=False)
def seal_producer_lineage_v1(request_ref):
 q=_request_v1(request_ref,"seal_producer_lineage");master=_master_v1(q["master_input_ref"],q)
 bundle=q["contrastive_prerequisite"];metadata=prerequisite_metadata_v1(bundle,"contrastive")
 done,wait,outer=metadata["completion"],metadata["wait"],metadata["outer"]
 phase_ref=wait["input_ref"];phase=c.read(phase_ref);authority=c.read(phase["root_stage_request_ref"])
 c.require(authority["master_input_ref"]==q["master_input_ref"]and phase["phase"]=="contrastive"
  and bundle["completion_ref"]["path"]==master["output_directory"]+"/contrastive/completed.json"
  and bundle["wait_ref"]["path"]==master["worker_auxiliary_scope"]+"/contrastive/actual_wait.json"
  and done["saved_MLM_initialization_ref"]==q["initialization_ref"],
  "Same genuine admitted contrastive child, source, initialization, actual wait and consumed outer0")
 c.set_amendment_v1(phase_ref);c.validate_input_v1(phase_ref);c.validate_child_completion_v1(done,phase,authority)
 integration=science.integration_v1(q["storage_policy_selector_ref"])
 qualification=integration.qualify_saved_MLM_for_embeddings_v1(initialization_ref=q["initialization_ref"],
  contrastive_completion_ref=bundle["completion_ref"],contrastive_wait_ref=bundle["wait_ref"],consumed_outer_ref=bundle["outer_terminal_ref"])
 g=c.MasterStorageGuard(c.selected_guard_v1(consumer=False));storage=c.read(master["storage_ref"])
 g.validate_storage_v9(phase,dict(output_cap_bytes=8*2**30),storage,source_ref=c.ref(c.DRIVER),
  config_ref=master["config_ref"],runtime_ref=master["runtime_ref"])
 directory=c.local(master["ROOT_record_directory"])
 c.active_amendment_v1();c.charged_census_v1(phase,required_write_bytes=131072)
 g.budget_check_v9(storage,boundary=True,required_write_bytes=131072)
 qualified_ref=_save_v1(directory,"saved_MLM_qualification.json",qualification)
 lineage=dict(schema="native_saved_MLM_contrastive8_actual_producer_lineage_v1",
  status="completed_native_saved_MLM_contrastive_only_producer",parent_id=c.PARENT,
  source_manifest_ref=c.ref(c.MANIFEST),source_review_ref=q["source_review_ref"],
  master_input_ref=q["master_input_ref"],phase_input_ref=phase_ref,contrastive_prerequisite=bundle,
  checkpoint_ref=done["checkpoint_ref"],saved_MLM_initialization_ref=q["initialization_ref"],
  saved_MLM_qualification_ref=qualified_ref,ROOT_saved_MLM_artifact_acceptance_ref=master["ROOT_saved_MLM_artifact_acceptance_ref"],
  historical_MLM_outer_exit_code=1,contrastive_actual_direct_wait0=True,contrastive_consumed_outer0=True,
  all_owned_processes_and_groups_drained=True,actual_optimizer_updates=2135,completed_epochs=1,
  reservation_bytes=8*2**30,whole_cap_includes_all_output_AUX_TMP_cache_logs_and_checkpoints=True,
  embedding_retrieval_or_full_pipeline_finalization_performed=False,MLM_rerun_performed=False,
  future_consumers_require_separate_admission_and_explicit_actual_producer_manifest_join=True,
  future_writes_and_relaunch_revoked=False,no_storage_close_or_credit_applied=True,query_ancestry_truth_opened=False)
 encoded=(json.dumps(lineage,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 c.require(len(encoded)<=65536 and not(c.local(master["output_directory"])/"completed.json").exists(),
  "Fresh bounded truthful operational phase-only producer completion")
 c.active_amendment_v1();c.charged_census_v1(phase,required_write_bytes=len(encoded)+8192)
 g.budget_check_v9(storage,boundary=True,required_write_bytes=len(encoded)+8192)
 completed=c.save(c.local(master["output_directory"]),"completed.json",lineage)
 c.charged_census_v1(phase)
 record=_save_v1(directory,"producer_lineage.json",dict(lineage,operational_producer_completion_ref=completed))
 return dict(status="CONTRASTIVE_ONLY_PRODUCER_LINEAGE_SEALED",producer_lineage_ref=record,
  operational_producer_completion_ref=completed,qualification_ref=qualified_ref,historical_MLM_outer_exit_code=1,
  scientific_execution_authorized=False,close_or_credit_authorized=False,
  ordinary_close_requires_separate_ROOT_revocation_and_actual_child_wait_join=True)
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group()
 a.add_argument("--prepare-master-ref");a.add_argument("--prepare-phase-ref");a.add_argument("--seal-producer-lineage-ref")
 args=p.parse_args(argv)
 if not any((args.prepare_master_ref,args.prepare_phase_ref,args.seal_producer_lineage_ref)):
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)));return 0
 c.require(sys.platform.startswith("linux")and sys.dont_write_bytecode and not any(n in sys.modules for n in("torch","torch_geometric")),
  "Fresh ROOT Linux stdlib metadata interpreter")
 try:
  if args.prepare_master_ref:result=prepare_master_v1(json.loads(args.prepare_master_ref))
  elif args.prepare_phase_ref:result=prepare_phase_v1(json.loads(args.prepare_phase_ref))
  else:result=seal_producer_lineage_v1(json.loads(args.seal_producer_lineage_ref))
 except PendingPrerequisites as exc:
  print(json.dumps(dict(status="PENDING_GENUINE_PREREQUISITES",reason=str(exc),no_launch=True,no_automatic_continuation=True)));return 2
 print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
