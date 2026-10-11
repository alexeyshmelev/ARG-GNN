import json,pathlib,resource,time,os
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"research/reviews/anchor_parent_producer_actual_train000_v1_root"
from arggnn import anchor_parent_producer_v1 as p
binding=json.loads((BASE/"observed_registration.json").read_text())["binding"]
started=time.monotonic()
assemblies,report=p.qualify_observed_v1(binding)
report["elapsed_seconds"]=time.monotonic()-started
report["maximum_self_RSS_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
report["RSS_measurement_scope"]="Linux process lifetime through observed input reconstruction/assembly, no numerical model"
print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
