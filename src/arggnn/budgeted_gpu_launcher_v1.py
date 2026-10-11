"""Use a reviewed supplemental budget for the unchanged project GPU launcher.

Only storage preflight changes. GPU selection, locks, inherited FD and child
launch remain the original launch_gpu.py main body. Default is source-only.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
LAUNCHER="scripts/launch_gpu.py"
LAUNCHER_SHA="f07324053d8e649e27d66cbca9799bb6ea237c92c247853e2d45e6d440a2c094"
BUDGET="src/arggnn/project_budget_v4.py"
BUDGET_SHA="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47"
CONFIG_SHA="5788f87d11825cf4fbe94a25abe056132107b155848111354e70637334e91d63"
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def launch(root,authority_ref):
 root=Path(root)
 if digest(root/BUDGET)!=BUDGET_SHA:raise ValueError("Budget implementation changed")
 budget=load_module("budgeted_launcher_companion",root/BUDGET)
 a=budget.bound(root,authority_ref,True)
 source=Path(__file__).resolve()
 required={"kind":"budgeted_gpu_launcher_v1_root_admission","root_authorizer":"root_agent","active":True,
           "implementation":{"path":str(source.relative_to(root)),"sha256":digest(source)},
           "launcher":{"path":LAUNCHER,"sha256":LAUNCHER_SHA},
           "budget_source_sha256":BUDGET_SHA,"budget_config_sha256":CONFIG_SHA,
           "GPU_authorized":True,"scientific_method_changes":False,
           "storage_preflight":"supplemental_combined_projection_v4"}
 if any(budget.encoded(a.get(k))!=budget.encoded(v) for k,v in required.items()):raise ValueError("Exact root launcher authority required")
 budget.bound(root,a["launcher"])
 review=budget.bound(root,a["independent_review"],True)
 expected={"kind":"budgeted_gpu_launcher_v1_independent_review","review_role":"independent_reviewer",
           "verdict":"PASS","implementation":required["implementation"],"launcher":required["launcher"],
           "budget_source_sha256":BUDGET_SHA,"actual_GPU_launch_performed":False}
 if any(budget.encoded(review.get(k))!=budget.encoded(v) for k,v in expected.items()):raise ValueError("Exact independent adapter review required")
 command=a.get("command")
 if not isinstance(command,dict) or set(command)!={"gpu","job","module","args"}:raise ValueError("Exact command schema required")
 if type(command["gpu"]) is not int or command["gpu"] not in (2,3,4,5):raise ValueError("Eligible physical GPU required")
 if not isinstance(command["job"],str) or not command["job"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in command["job"]):raise ValueError("Exact logical job name required")
 if not isinstance(command["module"],str) or not command["module"].startswith("arggnn."):raise ValueError("Project module required")
 if not isinstance(command["args"],list) or any(not isinstance(v,str) or "\0" in v for v in command["args"]):raise ValueError("Exact frozen argument vector required")
 runtime=load_module("budgeted_launcher_original",root/LAUNCHER)
 original_preflight=runtime.require_storage
 original_argv=sys.argv[:]
 first=True
 def preflight(headroom_bytes=2000000000):
  nonlocal first
  if type(headroom_bytes) is not int or headroom_bytes!=2000000000:raise ValueError("Original launcher headroom changed")
  view=budget.check_reservation(root,a["reservation"],expected_source_sha256=BUDGET_SHA,
       expected_config_sha256=CONFIG_SHA,initial_ledger_pins=a["initial_ledger_pins"] if first else None,
       boundary=True,required_write_bytes=headroom_bytes)
  first=False
  info=os.statvfs(root)
  if info.f_bavail*info.f_frsize<=a["reservation"]["reserved_bytes"]+headroom_bytes:raise ValueError("Filesystem headroom exhausted")
  # Original lease.storage_bytes is now this conservative accounting projection.
  # This resource-only semantic change is explicitly disclosed in root authority.
  return view["projected_bytes"]
 try:
  runtime.require_storage=preflight
  sys.argv=[str(root/LAUNCHER),"--gpu",str(command["gpu"]),"--job",command["job"],command["module"],*command["args"]]
  return runtime.main()
 finally:
  runtime.require_storage=original_preflight
  sys.argv=original_argv
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--launch",action="store_true");ap.add_argument("--admission");ap.add_argument("--expected-admission-sha256")
 args=ap.parse_args()
 if not args.launch:
  print(json.dumps({"status":"source_only","launcher_sha256":LAUNCHER_SHA,"budget_source_sha256":BUDGET_SHA,"GPU_launched":False}));return
 if Path.cwd().resolve()!=ROOT or not args.admission or not args.expected_admission_sha256:raise ValueError("Exact root launch opt-in required")
 raise SystemExit(launch(ROOT,{"path":args.admission,"sha256":args.expected_admission_sha256}))
if __name__=="__main__":main()
