"""Root retirement of two inactive historical scopes, with retained artifacts."""
from pathlib import Path
import sys
sys.path.insert(0,"/workspace-SR003.nfs2/ARG-GNN")
from datetime import datetime,timezone,timedelta
import errno,json,os,traceback
from arggnn import project_storage_terminal_retired_scopes_v1 as m
R=m.ROOT;P=R/"research/reviews/expanded_full_panel_v1_root"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e";BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
def ref(p):return m._ref(p)
def read(p):return json.loads((R/p).read_text())
def save(packet,name,v):
 packet.mkdir(parents=True,exist_ok=True);p=packet/name
 with p.open("x") as f:json.dump(v,f,indent=2,sort_keys=True);f.write("\n")
 return ref(p)
def inspect():return m.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=BASE)
def absent(pid):
 try:os.kill(pid,0)
 except OSError as e:
  if e.errno!=errno.ESRCH:raise
 else:return False
 return True
def authority(op):
 base=m.old
 return dict(kind="project_budget_terminal_retired_root_admission_v1",root_authorizer="root_agent",active=True,operation=op,accounting_mode=m.MODE,
  bindings=dict(implementation=ref(m.budget.SOURCE_REL),launcher=ref(m.budget.LAUNCHER_REL),
   cpu_overlay=ref("src/arggnn/project_storage_cpu_closed_scopes_v1.py"),terminal_overlay=ref("src/arggnn/project_storage_terminal_scopes_v1.py"),terminal_success_overlay=ref("src/arggnn/project_storage_terminal_success_scopes_v1.py"),terminal_retired_overlay=ref(m.SOURCE)),
  expected_before=m._pins(inspect()),all_budget_writers_use_scheduler_mutex=True,all_consumers_and_launchers_use_terminal_retired_guard=True,
  all_terminal_success_writers_finished_before_retirement_activation=True,historical_records_mutation_authorized=False,global_rebase_authorized=False,
  scientific_execution_authorized=False,GPU_authorized=False,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=23)).isoformat())
def run():
 assert (P/"terminal_completed_VAL_credits_v1/completed.json").exists()
 assert not (P/"full_panel_pyg_fits_v2").exists()
 for pid in [251730,252265,252351,252353,252356,252357,252358,252359,252360,252361,252362,300699,322742,374268]:
  if not absent(pid):raise RuntimeError("Earlier writer still present: "+str(pid))
 packet=P/"obsolete_scope_retirements_v1"
 if not (packet/"writer_barrier.json").exists():save(packet,"writer_barrier.json",dict(kind="root_retirement_writer_barrier_v1",H038_GPU_fits_admitted=False,prior_writers_observed_absent=True,
  source_cache_stop=ref(P/"cache_v4_superseded_for_efficiency_v1.json"),cache_actual_wait=ref(P/"pyg_observation_preparation_v4/launcher_exit.json"),
  orphan_VAL_credit_completion=ref(P/"terminal_completed_VAL_credits_v1/completed.json"),checked_utc=datetime.now(timezone.utc).isoformat()))
 for index,(key,case) in enumerate(m.CASES.items()):
  proof=packet/("scope_"+str(index))
  if (proof/"receipt.json").exists():continue
  refs={name:ref(path) for name,path in case["evidence"].items()}
  pids=case["pids"]
  observations=[]
  for pid in pids:
   if not absent(pid):raise RuntimeError("Retired original writer present: "+str(pid))
   observations.append(dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=datetime.now(timezone.utc).isoformat()))
  kind=case["kind"]
  ob=ref(proof/"observation.json") if (proof/"observation.json").exists() else save(proof,"observation.json",dict(kind="project_budget_obsolete_scope_terminal_observation_v1",root_authorizer="root_agent",key=key,retirement_kind=kind,evidence_refs=refs,pids=pids,observations=observations,all_original_and_resume_capabilities_revoked=True))
  a=authority("retire_obsolete_scope");a.update(scope_write_authority_revoked_for_all_cooperative_project_jobs=True,closed_scope_future_writes_and_relaunch_revoked=True,
   all_original_and_resume_capabilities_for_retired_scope_revoked=True,external_NFS_writers_outside_observation=True,
   retirement=dict(key=key,retirement_kind=kind,evidence_refs=refs,observation_ref=ob))
  ar=save(proof,"authority.json",a);receipt=m.apply_retire_obsolete_scope_v1(ar,expected_source_sha256=CORE,expected_launcher_sha256=BASE);save(proof,"receipt.json",receipt)
  print(json.dumps(dict(stage="obsolete_scope_retired",key=key,retained_bytes=receipt["retained_bytes"],credit_bytes=receipt["credit_bytes"])),flush=True)
 save(packet,"completed.json",dict(view=inspect(),scientific_completion_claimed=False,artifacts_deleted=False,root_scopes_retired=2))
 print(json.dumps(inspect()),flush=True)
if __name__=="__main__":
 try:run()
 except Exception:print(traceback.format_exc(),flush=True);raise SystemExit(1)
