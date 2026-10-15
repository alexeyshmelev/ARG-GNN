"""Materialize only the genuine VAL000 observation-plan request; no bind/admit/launch."""
from pathlib import Path
import copy,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/C3_GNN_current980_executable_root_20261015_v1"
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/REL)]
import root_contract_v1 as c
PREP_REF=dict(path="experiments/results/American_C3_val000_full30mb_shared_v1/native_paths/completed.json",
 sha256="40a3b588d0aa2b474c3f4aab492e34feb0a753a0f8cb73c20d9c0059d8a8c16b")
WAIT_REF=dict(path="research/reviews/prospective_980GB_capacity_source_v1/admitted_native_paths_v1/actual_wait.json",
 sha256="3b7e0d9a2e0b83c968c8e2340b850d44e27db1783990a07e933cfa09b58975b8")
PLAN_TEMPLATE_REF=dict(path="research/reviews/American_C3_PyG_execution_source_20261015_v1/VAL_plan_configs.inactive.json",
 sha256="2b351e69b58302f77a07d38ca52125670f5c132f4b2d711258af7dcff0a3bb1d")
def main():
 parent="american_c3_val_000";prep=c.validate_preparation_wait(PREP_REF,WAIT_REF)
 c.require(prep["parent_id"]==parent and prep["status"]=="completed_native_C3_path_preparation"
  and prep["query_focal_count"]==10000 and prep["query_ancestry_truth_opened"] is False,"Genuine original VAL000 prep")
 cfg=copy.deepcopy(c.read(PLAN_TEMPLATE_REF)["parents"][parent])
 cfg.update(execution_enabled=True,observation_ref=prep["observation_ref"],
  focal_registry_ref=prep["focals_ref"],native_preparation_ref=PREP_REF)
 req=c.read(c.ref(REL+"/request_plan_val_000.inactive.json"))
 req["bindings"]=dict(parent_id=parent,plan_config=cfg,native_preparation_ref=PREP_REF,preparation_wait_ref=WAIT_REF)
 c.validate_bindings("plan_val_000",req["bindings"])
 path=ROOT/REL/"request_plan_val_000.actual_inactive.json"
 if path.exists():
  c.require(c.read(c.ref(path))==req,"Preserve exact original inactive metadata binding")
  result=c.ref(path)
 else:result=c.save(ROOT/REL,path.name,req)
 print(json.dumps(dict(status="genuine_VAL000_plan_metadata_only_no_activation",request_ref=result,
  native_MLM_dependency=False,TRAIN_plans_dependency=False,scientific_payload_or_truth_opened=False,actual_admission=False)))
 return 0
if __name__=="__main__":raise SystemExit(main())
