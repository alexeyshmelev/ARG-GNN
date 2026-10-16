"""Future ROOT-only compact context/count preparation. No resource or scientific work."""
from pathlib import Path
import argparse,copy,json,sys
from research.reviews.FLARE_original_remaining_CPU_execution_source_20261017_v1 import root_contract_v1 as c
def bind_selected_context_v1(record):
 b=c.read(record)
 c.require(b.get("schema")=="FLARE_remaining_ROOT_context_binding_v1"and b.get("active")is True
  and b.get("root_authorizer")=="root_agent"and b.get("metadata_preparation_only")is True
  and b.get("admission_authorized")is False and b.get("remaining_VAL_truth_not_previously_opened")is True
  and b.get("source_manifest_ref")==c.ref(c.MANIFEST)and b.get("parent_id")in c.VAL,
  "Explicit ROOT metadata-only original target binding")
 c.sources();c.reviewed(b["source_review_ref"]);c.scientific_review();c.current(b,300)
 prep=c.local(b["preparation_directory"])
 c.require(prep==c.local(record["path"]).parent and prep.is_dir()
  and b["preparation_directory"].startswith("research/reviews/FLARE_remaining_ROOT_preparation_"),
  "Existing separate ROOT metadata preparation only")
 core=c.completion()
 original=b.get("previous_context_ref")or c.ref("research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1/context.INACTIVE.v1.json")
 ctx=copy.deepcopy(c.read(original));parent=b["parent_id"]
 row=copy.deepcopy(b["target_row"])
 frozen=c.DEPS["ORIGINAL_SIMULATION_CONFIGS"].get(parent)
 if frozen is not None:c.require(b["simulation_config_ref"]==frozen,"Exact original completed simulation config from inventory")
 plan=c.read(row["plan_ref"])
 c.require(plan.get("schema")=="american_c3_whole_tree_block_index_v1"and plan.get("parent_id")==parent
  and plan.get("sequence_length_bp")==30000000 and plan.get("haplotypes")==1600
  and plan.get("focal_records")==10000 and plan.get("all_trees_and_samples_retained")is True
  and plan.get("targets_opened")is False and plan.get("class_order")==["EUR","EAS","AFR"]
  and plan.get("observation_ref")==row["observation_ref"]and plan.get("focal_registry_ref")==row["focal_registry_ref"]
  and type(plan.get("sites"))is int and plan["sites"]>0,
  "Actual completed original whole-tree plan JSON count source; no array/truth reading")
 row.update(parent_id=parent,haplotypes=1600,input_sites=plan["sites"],simulation_config_ref=b["simulation_config_ref"],
  count_source_kind="completed_original_C3_plan_JSON_metadata_no_array_read")
 count=c.put(prep/(parent+".FLARE_count.json"),dict(schema="American_C3_full30Mb_observation_count_root_v1",
  source_observation_ref=row["observation_ref"],num_sites=plan["sites"],num_samples=1600,sequence_length=30000000,
  truth_opened=False,count_source_ref=row["plan_ref"],parent_id=parent))
 row["count_manifest_ref"]=count;ctx["targets"][parent]=row
 ctx.update(root_authorizer="root_agent",remaining_VAL_truth_not_previously_opened=True,algorithm_options_unchanged_after_VAL000=True)
 context_ref=c.put(prep/(parent+".FLARE_context.json"),ctx)
 core._sources();core._context(context_ref,parent_id=parent)
 registration=c.put(prep/(parent+".FLARE_registration.json"),dict(schema="FLARE_root_target_registration_v1",
  root_authorizer="root_agent",parent_id=parent,observation_ref=row["observation_ref"],focal_registry_ref=row["focal_registry_ref"],
  count_manifest_ref=count,scientific_role="registered_heldout_full30Mb_comparison",query_ancestry_truth_opened=False,score_authorized=False,
  hypothesis_id="H041",missing_original_cell=True,VAL000_rerun=False))
 configs=core.inactive_stage_configs_v1(context_ref,parent)
 return dict(status="BOUND_METADATA_ONLY_NOT_ADMITTED",context_ref=context_ref,registration_ref=registration,
  conversion_output_bytes=configs["conversion"]["output_cap_bytes"],conversion_whole_reservation_bytes=configs["conversion"]["output_cap_bytes"]+8*2**20,
  native_before_conversion_all_input_marker_output_bound=configs["native"]["output_cap_bytes"],
  capacity_proven=False,admission_or_execution=False,all_four_ready=False)
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--bind-ref");a=p.parse_args(argv)
 if not a.bind_ref:print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",writes=False,admission=False)));return 0
 c.require(Path.cwd()==c.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"Exact ROOT -B -s interpreter")
 print(json.dumps(bind_selected_context_v1(json.loads(a.bind_ref)),sort_keys=True));return 0

def remaining_cell_evidence_v1(stage_evidence_ref):
 """Return adapter-compatible ROOT certification only after real closed seal lifecycle."""
 e=c.read(stage_evidence_ref);parent=e["parent_id"];sealed_ref=e["completion_ref"]
 c.stage_evidence(stage_evidence_ref,stage="seal",parent=parent,completion_ref=sealed_ref)
 sealed=c.read(sealed_ref);native=c.read(sealed["native_run_ref"])
 c.require(sealed.get("status")==c.STATUS["seal"]and sealed.get("parent_id")==parent
  and native.get("parent_id")==parent,"Exact genuine remaining original seal/native cell")
 return dict(schema="FLARE_remaining_cell_ROOT_terminal_evidence_v1",root_authorizer="root_agent",parent_id=parent,
  seal_completion_ref=sealed_ref,native_run_ref=sealed["native_run_ref"],native_wait_ref=native["native_actual_wait_ref"],
  genuine_seal_direct_wait0=True,all_owned_drained=True,actual_outer_terminal_consumed0=True,
  seal_actual_wait_ref=e["actual_wait_ref"],seal_outer_terminal_ref=e["outer_terminal_ref"],
  seal_completion_envelope_ref=sealed_ref,closed_stage_evidence_ref=stage_evidence_ref)

if __name__=="__main__":raise SystemExit(main())
