#!/usr/bin/env python3
"""Explicit human-authorized sharing, preserving frozen storage, leases and waits."""
from pathlib import Path
import hashlib,json
from arggnn import project_storage_terminal_retired_scopes_v1 as storage
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="scripts/launch_reserved_gpu_authorized_external_share_v1.py"
ORIGINAL=ROOT/"scripts/launch_reserved_gpu_full_storage_v1.py"
ORIGINAL_SHA256="d7e5b2bcb464b9d3863b4f644e69aaa1c73c18aaa2550e44ecd021e2199f6dff"
TERMINAL_SHA256="bbbdf26206e4e1725c79d2558b4426af848df4cbbd5ac2d0f2bc7051c0606bab"
AUTH_PATH="research/reviews/H039_train_stage_root_v1/external_GPU_sharing_authorization.json"
AUTH_SHA256="56fe3e47542530d45613de6f4356adaebd0cf66f2d21ba2b0b426c7f996684d1"

def main():
 if hashlib.sha256((ROOT/storage.SOURCE).read_bytes()).hexdigest()!=TERMINAL_SHA256:
  raise ValueError("Frozen storage helper changed")
 if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()!=ORIGINAL_SHA256:
  raise ValueError("Frozen lease launcher changed")
 raw=(ROOT/AUTH_PATH).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=AUTH_SHA256:raise ValueError("Sharing authorization changed")
 auth=json.loads(raw)
 if (auth.get("schema")!="project_external_GPU_sharing_authorization_v1"
  or auth.get("status")!="authorized" or auth.get("root_authorizer")!="root_agent"
  or auth.get("physical_GPUs")!=[2,3] or auth.get("maximum_distinct_project_GPUs")!=2
  or auth.get("preserve_external_processes") is not True
  or auth.get("max_project_cuda_reserved_bytes")!=50000000000
  or auth.get("minimum_additional_free_bytes")!=4000000000):
  raise ValueError("Explicit bounded human GPU-sharing authority required")
 text=ORIGINAL.read_text().replace('"full_reserved_no_closed_scope_credit"','"journal_terminal_retired_scopes_credit_v1"')
 namespace={"__name__":"_arggnn_authorized_shared_lease_launcher","__file__":str(ORIGINAL)}
 exec(compile(text,str(ORIGINAL),"exec"),namespace)
 def credited_check(record,source_sha,launcher_sha,*,required_write_bytes=0):
  if (record["budget_components"]["implementation"]["sha256"]!=source_sha
   or record["budget_components"]["launcher"]["sha256"]!=launcher_sha):
   raise ValueError("Frozen core and lease pins differ")
  bindings=record["reservation"]["consumer_bindings"]
  for path in (SOURCE,"scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py",AUTH_PATH):
   actual=namespace["budget"].v4.reference(ROOT,path)
   if actual not in bindings:raise ValueError("Sharing wrapper, original wrapper and authorization must be reserved")
  account=storage.check_storage_locked_v1(record,boundary=True,required_write_bytes=required_write_bytes)
  return dict(account,actual_launcher_ref=namespace["budget"].v4.reference(ROOT,SOURCE),external_GPU_sharing_authorization_ref={"path":AUTH_PATH,"sha256":AUTH_SHA256})
 def eligible_authorized_share(record):
  available=(record["memory_total_mib"]-record["memory_used_mib"])*1048576
  return (record["index"] in auth["physical_GPUs"]
   and available>=auth["max_project_cuda_reserved_bytes"]+auth["minimum_additional_free_bytes"])
 namespace["SOURCE"]=SOURCE
 namespace["full_storage_check_locked"]=credited_check
 namespace["eligible_free"]=eligible_authorized_share
 return namespace["main"]()
if __name__=="__main__":raise SystemExit(main())
