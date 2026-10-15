#!/usr/bin/env python3
"""Explicit human-authorized sharing, preserving frozen storage, leases and waits."""
from pathlib import Path
import hashlib,json,os
import importlib.util
from arggnn import project_storage_terminal_retired_scopes_v1 as previous
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="scripts/launch_reserved_gpu_authorized_GPU45_980_v1.py"
ORIGINAL=ROOT/"scripts/launch_reserved_gpu_full_storage_v1.py"
ORIGINAL_SHA256="d7e5b2bcb464b9d3863b4f644e69aaa1c73c18aaa2550e44ecd021e2199f6dff"
TERMINAL_SHA256="bbbdf26206e4e1725c79d2558b4426af848df4cbbd5ac2d0f2bc7051c0606bab"
AUTH_PATH="research/reviews/H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json"
AUTH_SHA256="0231ae8edb05f8dd34ce127a0de0b97c1dd64fa461ad081e7a83bb92c558d26c"

HELPER_PATH='research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py'
HELPER_SHA256="fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56"
REPAIR_REF={"path":'research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json',"sha256":"f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db"}

POLICY_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}

def main():
 global storage
 if hashlib.sha256((ROOT/HELPER_PATH).read_bytes()).hexdigest()!=HELPER_SHA256:
  raise ValueError("Actual narrow repair helper changed")
 spec=importlib.util.spec_from_file_location("root_shared_marker_storage",ROOT/HELPER_PATH)
 storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
 if hashlib.sha256((ROOT/POLICY_REF["path"]).read_bytes()).hexdigest()!=POLICY_REF["sha256"]:
  raise ValueError("Exact actual980 policy source")
 if json.loads(os.environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"])!=POLICY_REF:
  raise ValueError("Exact actual980 policy environment")
 storage.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
 if hashlib.sha256((ROOT/previous.SOURCE).read_bytes()).hexdigest()!=TERMINAL_SHA256:
  raise ValueError("Frozen storage helper changed")
 if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()!=ORIGINAL_SHA256:
  raise ValueError("Frozen lease launcher changed")
 raw=(ROOT/AUTH_PATH).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=AUTH_SHA256:raise ValueError("Sharing authorization changed")
 auth=json.loads(raw)
 if (auth.get("schema")!="project_external_GPU_sharing_authorization_v1"
  or auth.get("status")!="authorized" or auth.get("root_authorizer")!="root_agent"
  or auth.get("physical_GPUs")!=[4,5] or auth.get("maximum_distinct_project_GPUs")!=2
  or auth.get("preserve_external_processes") is not True
  or auth.get("max_project_cuda_reserved_bytes")!=50000000000
  or auth.get("minimum_additional_free_bytes")!=4000000000):
  raise ValueError("Explicit bounded human GPU-sharing authority required")
 text=ORIGINAL.read_text().replace('"full_reserved_no_closed_scope_credit"','"prospective_980gb_v1"')
 namespace={"__name__":"_arggnn_authorized_shared_lease_launcher","__file__":str(ORIGINAL)}
 exec(compile(text,str(ORIGINAL),"exec"),namespace)
 def credited_check(record,source_sha,launcher_sha,*,required_write_bytes=0):
  if (record["budget_components"]["implementation"]["sha256"]!=source_sha
   or record["budget_components"]["launcher"]["sha256"]!=launcher_sha):
   raise ValueError("Frozen core and lease pins differ")
  bindings=record["reservation"]["consumer_bindings"]
  for path in (SOURCE,"scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py",AUTH_PATH,HELPER_PATH,POLICY_REF["path"],REPAIR_REF["path"],"research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py","research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py","src/arggnn/project_budget_v9_consumer.py"):
   actual=namespace["budget"].v4.reference(ROOT,path)
   if actual not in bindings:raise ValueError("Sharing wrapper, original wrapper and authorization must be reserved")
  account=storage.check_storage_locked_v1(record,boundary=True,required_write_bytes=required_write_bytes)
  return dict(account,actual_launcher_ref=namespace["budget"].v4.reference(ROOT,SOURCE),external_GPU_sharing_authorization_ref={"path":AUTH_PATH,"sha256":AUTH_SHA256})
 def eligible_authorized_share(record):
  available=(record["memory_total_mib"]-record["memory_used_mib"])*1048576
  return (record["index"] in auth["physical_GPUs"]
   and available>=auth["max_project_cuda_reserved_bytes"]+auth["minimum_additional_free_bytes"])
 namespace["SCHEDULABLE_GPUS"]=(4,5)
 namespace["SOURCE"]=SOURCE
 namespace["full_storage_check_locked"]=credited_check
 namespace["eligible_free"]=eligible_authorized_share
 return namespace["main"]()
if __name__=="__main__":raise SystemExit(main())
