from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os
p=Path("research/reviews/research_execution_ROOT_20261017_v34")
def ref(q):return dict(path=str(q),sha256=hashlib.sha256(Path(q).read_bytes()).hexdigest())
def save(name,v):
 q=p/name
 with q.open("x")as f:json.dump(v,f,sort_keys=True,indent=2);f.write("\n")
 return ref(q)
assert not(p/"TreeLoom_qualification_owner4_execution_invocation.root_v1.json").exists()
assert not(p/"TreeLoom_qualification_owner4_admission_actual_outer0_consumed.root_v1.json").exists()
save("TreeLoom_qualification_owner4_prelaunch_failure_correction.root_v2.json",dict(root_authorizer="root_agent",all_failed_ROOT_launchers_before_scientific_Popen=True,prior_shell_failure_attribution_not_established=True,at_least_one_prior_Python_wrapper_ran=True,confirmed_ROOT_error="Launch check used nonexistent admitted.status instead of actual admitted.schema; subsequent exclusive failure-record creation also blocks replay",original_records_preserved=True,scientific_namespace_replayed=False,utc=datetime.now(timezone.utc).isoformat()))
w=json.loads((p/"TreeLoom_qualification_owner4_admission_actual_wait.root_v1.json").read_text());assert w["actual_direct_returncode"]==0 and w["actual_direct_wait"]is True
q=Path("research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/ROOT_qualification_owner4_records_root01_owner4_oct17/admitted.json");a=json.loads(q.read_text());assert a["schema"]=="ROOT_TreeLoom_qualification_admitted_v1" and a["input_ref"]["sha256"]=="49e9094f2b7497df38e8006b04e241c498d9341f18c2542e5ec48a1102ecec8b"
save("TreeLoom_qualification_owner4_admission_actual_outer0_consumed.root_v1.json",dict(root_authorizer="root_agent",actual_tool_session_id=83706,actual_tool_terminal_exit_code=0,actual_terminal_consumed=True,direct_wait_ref=ref(p/"TreeLoom_qualification_owner4_admission_actual_wait.root_v1.json"),admitted_ref=ref(q),utc=datetime.now(timezone.utc).isoformat()))
with(p/"TreeLoom_qualification_owner4_execution.stdout.root_v1.txt").open("x")as f:
 child=subprocess.Popen(a["execute_argv"],env=dict(os.environ,**a["execute_environment"]),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 v=dict(actual_pid=child.pid,actual_argv=a["execute_argv"],admitted_ref=ref(q),scientific_input_ref=a["input_ref"],utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False)
 save("TreeLoom_qualification_owner4_execution_invocation.root_v1.json",v);print(json.dumps(v),flush=True);rc=child.wait()
v=dict(actual_direct_returncode=rc,actual_direct_wait=True,admitted_ref=ref(q),scientific_input_ref=a["input_ref"],utc=datetime.now(timezone.utc).isoformat())
save("TreeLoom_qualification_owner4_execution_actual_wait.root_v1.json",v)
print(json.dumps(v),flush=True);print((p/"TreeLoom_qualification_owner4_execution.stdout.root_v1.txt").read_text()[-18000:],flush=True)
raise SystemExit(rc)
