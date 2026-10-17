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
save("TreeLoom_qualification_owner4_shell_pre_Popen_failure.root_v1.json",dict(root_authorizer="root_agent",actual_tool_exit_codes=[1,1],scientific_Popen_reached=False,scientific_input_replayed=False,original_cause_not_confirmed=True,explicit_bash_c_environment_and_Python_probe_returned0=True,utc=datetime.now(timezone.utc).isoformat()))
w=json.loads((p/"TreeLoom_qualification_owner4_admission_actual_wait.root_v1.json").read_text());assert w["actual_direct_returncode"]==0 and w["actual_direct_wait"]is True
q=Path("research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/ROOT_qualification_owner4_records_root01_owner4_oct17/admitted.json");a=json.loads(q.read_text());assert a["status"]=="QUALIFICATION_ADMITTED_ONLY"
save("TreeLoom_qualification_owner4_admission_actual_outer0_consumed.root_v1.json",dict(root_authorizer="root_agent",actual_tool_session_id=83706,actual_tool_terminal_exit_code=0,actual_terminal_consumed=True,direct_wait_ref=ref(p/"TreeLoom_qualification_owner4_admission_actual_wait.root_v1.json"),admitted_ref=ref(q),utc=datetime.now(timezone.utc).isoformat()))
with(p/"TreeLoom_qualification_owner4_execution.stdout.root_v1.txt").open("x")as f:
 child=subprocess.Popen(a["execute_argv"],env=dict(os.environ,**a["execute_environment"]),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 v=dict(actual_pid=child.pid,actual_argv=a["execute_argv"],admitted_ref=ref(q),scientific_input_ref=a["input_ref"],utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False)
 save("TreeLoom_qualification_owner4_execution_invocation.root_v1.json",v);print(json.dumps(v),flush=True);rc=child.wait()
v=dict(actual_direct_returncode=rc,actual_direct_wait=True,admitted_ref=ref(q),scientific_input_ref=a["input_ref"],utc=datetime.now(timezone.utc).isoformat())
save("TreeLoom_qualification_owner4_execution_actual_wait.root_v1.json",v)
print(json.dumps(v),flush=True);print((p/"TreeLoom_qualification_owner4_execution.stdout.root_v1.txt").read_text()[-18000:],flush=True)
raise SystemExit(rc)
