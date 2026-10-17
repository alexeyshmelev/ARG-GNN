"""Inactive bounded original C3 remaining-cell contract; all runtime operations require ROOT authority."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,os,stat,sys,time
from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2 import common_v1 as original
ROOT=original.ROOT
HERE="research/reviews/C3_remaining_cells_GPU0_source_20261017_v1"
DRIVER=HERE+"/execution_v1.py";DIRECT=HERE+"/direct_gpu0_v1.py";WORKER=HERE+"/worker_v1.py";MANIFEST=HERE+"/source_manifest.v1.json";BUILDER=HERE+"/contract_v1.py"
PARENT="American_C3_short20_TRAIN_original5VAL"
RUNTIME_REF={'path': 'research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json', 'sha256': 'e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602'}
VAL=tuple("american_c3_val_%03d"%i for i in range(5));EPOCHS=(5,10,15,20,25,30)
GPU_PHASES=tuple("predict_val_%03d"%i for i in range(3));PHASES=GPU_PHASES+("matrix",)
BASE_CAPS=dict(CPU_threads=32,dataloader_workers=0,reservation_bytes=32*2**20,log_bytes=256*2**10,max_seconds=86400,max_cpu_rss_bytes=128*2**30,headroom_bytes=4*2**30,cleanup_seconds=30,work_cutoff_margin_seconds=300,terminal_evidence_margin_seconds=150,max_checkpoint_bytes=3*2**30,max_checkpoint_files=2)
CAPS=dict(BASE_CAPS,max_cuda_reserved_bytes=0)
OWNED_ROSTER_CAP=4096;MINIMAL_WAIT_CAP_BYTES=65536;TERMINAL_RECEIPT_CAP_BYTES=8*2**20
OWNED_ROSTER_POLICY_REF=original.OWNED_ROSTER_POLICY_REF if hasattr(original,"OWNED_ROSTER_POLICY_REF")else {"path":"research/reviews/native_VAL000_GPU0_owned_roster_source_20261016_v4/owned_roster_policy.INACTIVE.v4.json","sha256":"ea1ddb59ab4248b8ba95dbbe77d25db30636cc318a4fdaebde1e1ae6e5a7344f"}
GPU0_UUID=original.GPU0_UUID;BASE_POOL=original.BASE_POOL;GPU0_GUARD=original.GPU0_GUARD;PIDFD=original.PIDFD;OWNED=original.OWNED;NATIVE_PYTHON_REF=dict(path=".runtime/env/bin/python3.11",runtime_metadata_ref=RUNTIME_REF)
require=original.require;local=original.local;read=original.read;raw=original.raw;ref=original.ref;load=original.load;save=original.save
PINS=None;ROOT_INPUT_REF=None;STATE=None;GUARD=None;POLICY=None;SELECTOR_REF=None;_GUARD=None;_MEMO=None
def pins_v1():
 global PINS
 if PINS is None:PINS=read(ref(HERE+"/pins.v1.json"))
 return PINS
def active(v):
 now=datetime.now(timezone.utc)
 require(v.get("root_authorizer")=="root_agent"and v.get("status",v.get("active"))in("active",True)and v.get("template_only")is not True and datetime.fromisoformat(v["created_at_utc"])<=now<datetime.fromisoformat(v["expires_at_utc"]),"Exact finite active ROOT metadata")
def configure_selector_v1(q):
 global GUARD,POLICY,SELECTOR_REF,_GUARD
 r=q["storage_policy_selector_ref"];p=pins_v1()["pins"]
 require(r==p["CENSUS_SELECTOR"]and q["Census_verification_ref"]==p["CENSUS_VERIFICATION_OUTER"],"Explicit exact already verified Census selector and consumed verification receipt")
 verified=read(q["Census_verification_ref"]);inner=read(p["CENSUS_VERIFICATION"])
 require(verified.get("schema")=="ROOT_Census_observation_fresh_child_verification_actual_outer_consumed_v1"and verified.get("selector_ref")==r and verified.get("verification_ref")==p["CENSUS_VERIFICATION"]and verified.get("actual_terminal_consumed")is True and type(verified.get("actual_outer_exit_code"))is int and verified["actual_outer_exit_code"]==0 and type(verified.get("tool_session_id"))is int and verified["tool_session_id"]>0 and inner.get("schema")=="ROOT_Census_observation_fresh_child_verification_v1"and inner.get("selector_ref")==r and inner.get("actual_direct_guard_roles")==47 and inner.get("all_parent_pins_and_46roles_preserved")is True,"Genuine consumed ROOT verification0 and full47 roles, no inferred activation")
 p=pins_v1()["pins"];v=read(r)
 require(v.get("active")is True and v.get("root_authorizer")=="root_agent"and v.get("source_ref")==v.get("selector_source_ref")==p["CENSUS_GUARD"]and v.get("accounting_mode")=="Census_observation_two_scope_retention_v1"and verified.get("policy_ref")==v["policy_ref"],"Exact reviewed two-Census selector/policy; no discovery or fallback")
 if _GUARD is not None:
  require(SELECTOR_REF==r and POLICY==v["policy_ref"]and GUARD==p["CENSUS_GUARD"],"Immutable selector/policy for interpreter");return _GUARD
 g=load(p["CENSUS_GUARD"],"_C3_remaining_exact_Census")
 require(g.load_selected_guard_v1(r)is g,"Actual selected child singleton")
 SELECTOR_REF=copy.deepcopy(r);POLICY=copy.deepcopy(v["policy_ref"]);GUARD=copy.deepcopy(p["CENSUS_GUARD"]);_GUARD=g;return g
def selected_guard_v1(*,consumer):
 global _MEMO
 require(type(consumer)is bool and _GUARD is not None,"Explicit selected guard role")
 if not consumer:
  require(_MEMO is None and all(getattr(_GUARD.budget.v4,n).__module__==_GUARD.budget.v4.__name__ for n in("canonical","file_bytes","mutex")),"Fresh unwrapped ROOT mutation interpreter");return _GUARD
 if _MEMO is None:
  memo=load(original.NAMESPACE_MEMO,"_C3_remaining_read_only_memo")
  _MEMO=memo.install_consumer_v1(_GUARD.budget.v4,_GUARD.budget,consumer_role="read_only_consumer",expected_source_path=ROOT/original.V4["path"],expected_source_sha256=original.V4["sha256"])
 return _GUARD
def review_v1(r):
 v=read(r)
 require(v.get("independent")is True and v.get("verdict")in("SOURCE_PASS","PASS_SOURCE_ONLY","PASS")and v.get("candidate_manifest_ref",v.get("source_manifest_ref"))==ref(MANIFEST)and v.get("blockers")==[],"Exact complete independent source review")
 return v
def manifest_v1():
 m=read(ref(MANIFEST))
 for r in m["executed_source_refs"]+m["source_authentication_refs"]:raw(r)
 return m
def training_v1():
 p=pins_v1()["pins"];legacy=load(p["SAVED_CONTRACT"],"_C3_remaining_original_saved600_contract")
 bundle,training=legacy.training_ready_v1(p["ARTIFACT_ACCEPTANCE"])
 frozen=read(p["FROZEN"])
 require(training["checkpoint_candidates"]==frozen["checkpoint_candidates"]and [x["epoch"]for x in training["checkpoint_candidates"]]==list(EPOCHS),"All6 original saved600 checkpoints remain frozen")
 return bundle,training
def original_terminal_values_v1(e,parent):
 require(set(e)=={"input_ref","launch_ref","wait_ref","outer_ref","permanent_revocation_ref"},"Exact original CPU disposition evidence")
 expected=(pins_v1()["old_CPU_inputs"]|pins_v1()["completed_inputs"])[parent]
 require(e["input_ref"]==expected,"Exact original already-admitted CPU input")
 inp=read(expected);launch=read(e["launch_ref"]);wait=read(e["wait_ref"]);outer=read(e["outer_ref"]);rev=read(e["permanent_revocation_ref"])
 phase="predict_val_"+parent[-3:];driver=inp["root_driver_ref"]
 require(launch.get("input_ref")==wait.get("input_ref")==expected and launch.get("driver_ref")==wait.get("driver_ref")==driver and launch.get("phase")==wait.get("phase")==phase and launch.get("actual_launch")is True and launch.get("pid")==wait.get("pid")and wait.get("launch_ref")==e["launch_ref"]and type(wait.get("returncode"))is int and wait.get("actual_direct_launcher_wait")is True and wait.get("all_owned_processes_and_groups_drained")is True,"Genuine original direct terminal and complete drain; nonzero stays nonzero")
 require(type(outer.get("actual_exit_code"))is int and outer.get("actual_terminal_consumed")is True and outer.get("actual_wait_ref")==e["wait_ref"]and outer.get("actual_direct_worker_returncode")==wait["returncode"]and type(outer.get("actual_tool_session_id"))is int and outer["actual_tool_session_id"]>0,"Actual consumed original terminal, preserving its own exit and direct status separately")
 require(rev.get("root_authorizer")=="root_agent"and rev.get("input_ref")==expected and rev.get("future_writes_and_relaunch_permanently_revoked")is True and rev.get("preserve_all_files_and_original_charge")is True and rev.get("old_wait_ref")==e["wait_ref"]and rev.get("old_outer_ref")==e["outer_ref"]and datetime.fromisoformat(rev["revoked_at_utc"])>=datetime.fromisoformat(wait["finished_utc"]),"Explicit permanent original-writer revocation without old success or credit")
 return inp,wait,outer,rev
def election_values_v1(record):return election_value_v1(read(record))
def election_value_v1(e):
 active(e)
 require(e.get("schema")=="C3_ROOT_poststop_retained_cells_acceptance_v1"and e.get("saved600_acceptance_ref")==pins_v1()["pins"]["ARTIFACT_ACCEPTANCE"]and e.get("inventory_sealed_after_all_original_terminals_and_drains")is True and e.get("private_truth_opened")is False and e.get("original_nonzero_waits_preserved")is True and set(e["parents"])==set(VAL),"Separate truthful ROOT post-stop artifact acceptance, original5VAL and no truth")
 _,t=training_v1();candidates={x["epoch"]:x["checkpoint_ref"]for x in t["checkpoint_candidates"]}
 retained={};pending={}
 for parent,row in e["parents"].items():
  inp,wait,outer,rev=original_terminal_values_v1(row["original_disposition"],parent)
  require(datetime.fromisoformat(row["inventory_sealed_at_utc"])>=datetime.fromisoformat(rev["revoked_at_utc"]),"Inventory sealed only after permanent revocation")
  refs=row["retained_prediction_seal_refs"];cells={}
  for r in refs:
   seal=read(r);epoch=seal["epoch"]
   require(epoch in candidates and epoch not in cells and seal.get("parent_id")==parent and seal.get("status")=="sealed_C3_original_heldout_predictions"and seal.get("split")=="val"and seal.get("records")==10000 and seal.get("checkpoint_ref")==candidates[epoch]and seal.get("training_completion_ref")==tref_v1()and seal.get("private_truth_opened")is False and seal.get("probabilities_modified")is False and seal["prediction_ref"]["path"].startswith(inp["output_directory"]+"/")and r["path"].startswith(inp["output_directory"]+"/"),"Retained exact original immutable cell metadata/provenance; payload is not opened here")
   cells[epoch]=r
  missing=sorted(set(EPOCHS)-set(cells))
  require((parent in pins_v1()["old_CPU_inputs"]and set(missing)<=set((25,30))and set((5,10,15,20))<=set(cells))or(parent in pins_v1()["completed_inputs"]and not missing),"Only post-stop missing25/30 may run; every answered cell reused")
  require(row["pending_epochs"]==missing,"ROOT sealed inventory determines exact pending subset")
  retained[parent]=cells;pending[parent]=missing
 return e,retained,pending
def draft_election_v1(request_ref):
 q=read(request_ref);active(q)
 require(q.get("schema")=="C3_ROOT_draft_poststop_cell_election_request_v1"and q.get("source_manifest_ref")==ref(MANIFEST),"Exact reviewed metadata-only election producer")
 review_v1(q["source_review_ref"]);manifest_v1()
 inventory=read(q["inventory_ref"])
 require(inventory.get("schema")=="C3_ROOT_poststop_original_prediction_inventory_v1"and inventory.get("root_authorizer")=="root_agent"and inventory.get("private_truth_opened")is False and set(inventory["parents"])==set(VAL),"Actual ROOT post-stop sealed all-parent inventory")
 rows=copy.deepcopy(inventory["parents"])
 for parent,row in rows.items():
  epochs=[read(r)["epoch"]for r in row["retained_prediction_seal_refs"]]
  require(len(epochs)==len(set(epochs)),"No duplicate chosen old cell")
  row["pending_epochs"]=sorted(set(EPOCHS)-set(epochs))
 value=dict(schema="C3_ROOT_poststop_retained_cells_acceptance_v1",root_authorizer="root_agent",status="active",created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],saved600_acceptance_ref=pins_v1()["pins"]["ARTIFACT_ACCEPTANCE"],source_manifest_ref=ref(MANIFEST),source_review_ref=q["source_review_ref"],poststop_inventory_ref=q["inventory_ref"],inventory_sealed_after_all_original_terminals_and_drains=True,private_truth_opened=False,original_nonzero_waits_preserved=True,parents=rows,accepts_saved_cells_only=True,original_invocation_success_claimed=False,storage_close_or_credit_authorized=False)
 election_value_v1(value)
 return value
def tref_v1():return read(pins_v1()["pins"]["FROZEN"])["training_completion_ref"]
def peer_values_v1(record):
 e=read(record);active(e);p=pins_v1()["pins"]
 require(e.get("schema")=="C3_ROOT_exact_live_H058_peer_v1"and e.get("H058_input_ref")==p["H058_INPUT"]and e.get("H058_launch_ref")==p["H058_LAUNCH"]and e.get("H058_request_ref")==p["H058_REQUEST"]and e.get("controller_terminal_qualification_ref")==p["H058_TERMINAL"]and e.get("controller_capability_ref")==p["CONTROLLER_CAP"]and e.get("H058_full_cap_plus_margin_bytes")==30*2**30 and e.get("C3_reserved_bytes")==26*2**30 and e.get("headroom_bytes")==4*2**30 and e.get("H058_source_or_registry_mutation_authorized")is False and e.get("terminal_peer_absence_preserves_full30GiB")is True,"Exact independently qualified peer launch and unchanged complete reservations")
 inp=read(p["H058_INPUT"]);launch=read(p["H058_LAUNCH"]);q=read(p["H058_REQUEST"])
 require(inp["root_stage_request_ref"]==p["H058_REQUEST"]and launch["input_ref"]==p["H058_INPUT"]and launch["borrowed_physical_inode"]is True and q["caps"]["max_cuda_reserved_bytes"]==28*2**30 and q["controller_terminal_qualification_ref"]==p["H058_TERMINAL"],"Actual peer input/launch/source/caps lineage")
 rows=launch["ownership"]["owned_identities"]
 require(len([r for r in rows if r["pid"]==launch["pid"]])==1 and e["worker_identity"]==next(r for r in rows if r["pid"]==launch["pid"]),"Exact peer PID/start from genuine launch, never guessed")
 return e,q
def caps_v1(phase):return dict(BASE_CAPS,max_cuda_reserved_bytes=8*2**30 if phase in GPU_PHASES else 0)
def config_v1(phase):return dict(schema="C3_remaining_cells_operational_config_v1",output_cap_bytes=32*2**20,phase=phase,caps=caps_v1(phase),scientific_body_changed=False)
def new_stage_values_v1(e, parent, election, pending):
    require(set(e) == {'input_ref', 'completion_ref', 'wait_ref', 'outer_ref'}, 'Exact new successful remaining stage evidence')
    s = read(e['input_ref'])
    q = read(s['root_stage_request_ref'])
    done = read(e['completion_ref'])
    wait = read(e['wait_ref'])
    outer = read(e['outer_ref'])
    require(s['source_manifest_ref'] == ref(MANIFEST) and q['source_manifest_ref'] == ref(MANIFEST) and (s['phase'] == q['phase'] == 'predict_val_' + parent[-3:]) and (q['cell_election_ref'] == election) and (q['parent_id'] == parent) and (q['pending_epochs'] == pending), 'Exact same source/election/parent/pending actual new producer')
    require(done['status'] == 'completed_C3_remaining_cells' and done['parent_id'] == PARENT and (done['original_VAL_parent_id'] == parent) and (done['child_input_ref'] == e['input_ref']) and (done['actual_execution_source_manifest_ref'] == ref(MANIFEST)) and (done['cell_election_ref'] == election) and (done['epochs'] == pending) and (done['training_completion_ref'] == tref_v1()) and (done['query_ancestry_truth_opened'] is False), 'Genuine own exact completed new cells')
    require(type(wait.get('returncode')) is int and wait['returncode'] == 0 and (wait['actual_direct_worker_wait'] is True) and (wait['actual_OS_wait_status'] == 0) and (wait['all_owned_processes_and_groups_drained'] is True) and (wait['complete_owned_roster_known'] is True) and (wait['completed_ref'] == e['completion_ref']) and (wait['input_ref'] == e['input_ref']) and (wait['adapter_manifest_ref'] == ref(MANIFEST)) and (wait['operational_failure'] is None) and wait['group_observations'] and all((x['absent'] is True for x in wait['group_observations'])), 'New GPU cells require actual OSwait0 and complete owned drain')
    require(outer['root_authorizer'] == 'root_agent' and type(outer['actual_exit_code']) is int and (outer['actual_exit_code'] == 0) and (outer['actual_terminal_consumed'] is True) and (outer['actual_wait_ref'] == e['wait_ref']) and (outer['actual_direct_worker_returncode'] == 0) and (type(outer['actual_tool_session_id']) is int) and (outer['actual_tool_session_id'] > 0), 'Genuine consumed new outer0')
    result = read(done['result_ref'])
    require(result['status'] == 'sealed_C3_original_remaining_cells' and result['parent_id'] == parent and (result['epochs'] == pending) and (result['training_completion_ref'] == tref_v1()) and (result['cell_election_ref'] == election) and (result['private_truth_opened'] is False) and (len(result['prediction_seal_refs']) == len(pending)), 'Only actually completed selected new cells')
    rows = {}
    for r in result['prediction_seal_refs']:
        seal = read(r)
        epoch = seal['epoch']
        require(epoch in pending and epoch not in rows and (seal['parent_id'] == parent) and (seal['training_completion_ref'] == tref_v1()) and (seal['private_truth_opened'] is False) and r['path'].startswith(s['output_directory'] + '/') and seal['prediction_ref']['path'].startswith(s['output_directory'] + '/'), 'Own immutable new seal provenance')
        rows[epoch] = r
    require(set(rows) == set(pending), 'Exact complete requested subset')
    return rows

def validate_request_v1(record):
 q=read(record);active(q)
 require(0<(datetime.fromisoformat(q["expires_at_utc"])-datetime.fromisoformat(q["created_at_utc"])).total_seconds()<=86400,"At most original24h scientific interval")
 require(type(q["job_id"])is str and q["output_directory"].startswith("experiments/results/C3_remaining_cells_GPU0_")and q["worker_auxiliary_scope"].startswith("research/reviews/C3_remaining_cells_GPU0_ROOT_")and q["worker_auxiliary_scope"].endswith("/worker_aux"),"Fresh own output/packet roots outside every old producer")
 require(q.get("schema")=="C3_remaining_cells_ROOT_request_v1"and q.get("execution_authorized")is True and q.get("source_manifest_ref")==ref(MANIFEST)and q.get("phase")in PHASES and "truth"not in q and "VAL_truth_capability"not in q and "TEST"not in q,"Exact inactive-until-ROOT original remaining-cell or matrix authority")
 review_v1(q["source_review_ref"]);manifest_v1();configure_selector_v1(q)
 require(q["capacity_guard_ref"]==GUARD and q["capacity_policy_ref"]==POLICY and q["caps"]==caps_v1(q["phase"]),"Exact selected Census and original fixed finite8GiB evaluator/CPU matrix caps")
 e,retained,pending=election_values_v1(q["cell_election_ref"])
 require(q.get("original_CPU_jobs_or_live_H058_mutation_authorized")is False and q.get("automatic_restart_or_fallback")is False,"No live job/controller changes or automatic work")
 if q["phase"]in GPU_PHASES:
  parent=VAL[int(q["phase"][-3:])]
  require(q["parent_id"]==parent and q["pending_epochs"]==pending[parent]and q["pending_epochs"]and q["plan_evidence"]["parent_id"]==parent,"Only exact post-stop pending cells for this original parent")
  legacy=load(pins_v1()["pins"]["SAVED_CONTRACT"],"_C3_remaining_original_plan_contract")
  require(q["plan_evidence"]==read(read(pins_v1()["old_CPU_inputs"][parent])["root_stage_request_ref"])["plan_evidence"],"Exact original admitted same-parent plan evidence and input support")
  legacy.plan_ready_v1(q["plan_evidence"],parent)
  peer_values_v1(q["peer_launch_capsule_ref"])
 else:
  require(q["pending_epochs"]==[]and q["parent_id"]==PARENT,"CPU-only cellwise matrix")
  needed={parent for parent in VAL if pending[parent]}
  require(set(q["new_stage_evidence"])==needed,"Complete actual new-stage wait/drain/outer0 evidence before matrix admission")
  for parent in needed:new_stage_values_v1(q["new_stage_evidence"][parent],parent,q["cell_election_ref"],pending[parent])
 return q,retained,pending
def set_amendment_v1(r):
 global ROOT_INPUT_REF
 require(ROOT_INPUT_REF is None or ROOT_INPUT_REF==r,"One exact current stage per interpreter");ROOT_INPUT_REF=r
def current_amendment_ref_v1():require(ROOT_INPUT_REF is not None,"Actual input installed");return ROOT_INPUT_REF
def active_amendment_v1():
 s=read(current_amendment_ref_v1())
 if STATE is None:q,_,_=validate_request_v1(s["root_stage_request_ref"])
 else:
  require(s==STATE[0],"Immutable input for current interpreter");q=read(s["root_stage_request_ref"]);require(q==STATE[2],"Immutable ROOT request for current interpreter")
 active(s);active(q)
 require(s["source_manifest_ref"]==ref(MANIFEST)and s["storage_policy_selector_ref"]==SELECTOR_REF and s["created_at_utc"]==q["created_at_utc"]and s["expires_at_utc"]==q["expires_at_utc"],"Exact current input/request/source/clock")
 return q,s,s
def validate_input_v1(r):
 global STATE
 require(r==ROOT_INPUT_REF,"Exact current input");q,s,_=active_amendment_v1()
 require(s.get("schema")=="C3_remaining_cells_actual_input_v1"and s.get("execution_authorized")is True and s["phase"]==q["phase"]and s["job_id"]==q["job_id"]and s["output_directory"]==q["output_directory"]and s["worker_auxiliary_scope"]==q["worker_auxiliary_scope"]and s["refs"]["runtime"]==RUNTIME_REF,"Exact admitted output/AUX/runtime")
 storage=read(s["refs"]["storage"]);cfg=read(s["refs"]["config"])
 require(cfg==config_v1(q["phase"])and storage["accounting_mode"]=="Census_observation_two_scope_retention_v1"and storage["storage_policy_selector_ref"]==SELECTOR_REF and storage["completed_overlay_policy_ref"]==POLICY and storage["job_id"]==s["job_id"]and storage["output_directory"]==s["output_directory"]and storage["reservation"]["output"]==str(local(s["output_directory"]))and storage["reserved_new_bytes"]==storage["reservation"]["reserved_bytes"]==32*2**20 and storage["reservation"]["auxiliary_scopes"]==[str(local(s["worker_auxiliary_scope"]))],"Actual fresh separate whole32MiB scope and exact Census charge")
 CAPS.clear();CAPS.update(q["caps"]);STATE=(copy.deepcopy(s),copy.deepcopy(storage),copy.deepcopy(q),copy.deepcopy(s))
 return s,q,cfg,storage,cfg
def guard_v1(q):configure_selector_v1(q);return selected_guard_v1(consumer=True)
def work_deadline_monotonic_v1():
 q,s,_=active_amendment_v1();end=min(datetime.fromisoformat(q["created_at_utc"]).timestamp()+CAPS["max_seconds"],datetime.fromisoformat(q["expires_at_utc"]).timestamp())
 return time.monotonic()+end-time.time()-CAPS["work_cutoff_margin_seconds"]
def require_work_window_v2(s):require(time.monotonic()<work_deadline_monotonic_v1(),"No fresh work within cleanup/evidence reserve")
def charged_census_v1(s,*,required_write_bytes=0):
 sizes=original._sizes(local(s["output_directory"]))+original._sizes(local(s["worker_auxiliary_scope"]))
 total=sum(n for _,n,_ in sizes);require(total+required_write_bytes<=32*2**20,"Whole scientific output+AUX/TMP/cache/log32MiB failstop");return {s["output_directory"]:total}
def compact_v1(s,name,value,g,storage,*,cap=2**20):
 active_amendment_v1();data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 require(len(data)<=cap,"Bounded operational metadata")
 charged_census_v1(s,required_write_bytes=len(data)+8192);g.budget_check_v9(storage,boundary=True,required_write_bytes=len(data)+8192)
 return save(local(s["worker_auxiliary_scope"]),name,value)
def worker_env_paths_v1(s):return original.worker_env_paths_v1(s)
def prepare_master_parents_v1(s,g,storage):
 require(local(s["output_directory"]).parent.is_dir()and local(s["worker_auxiliary_scope"]).parent.is_dir(),"Existing canonical fresh stage parents")
def validate_child_completion_v1(v,s,q):
 require(v.get("parent_id")==PARENT and v.get("child_input_ref")==ROOT_INPUT_REF and v.get("actual_execution_source_manifest_ref")==ref(MANIFEST)and v.get("query_ancestry_truth_opened")is False,"Exact source/input truth-free child")
 require(v.get("status")==("completed_C3_remaining_cells"if s["phase"]in GPU_PHASES else"completed_C3_remaining_cellwise_matrix"),"Genuine exact completed stage")
 if s["phase"]in GPU_PHASES:require(v["epochs"]==q["pending_epochs"],"Only requested pending cells completed")

ROOT_RECORD_CAP=454656
def record_budget_v1(directory,extra=0):
 total=4096;count=0
 if directory.exists():
  for p in directory.iterdir():
   require(p.is_file()and not p.is_symlink()and p.suffix==".json"and p.lstat().st_nlink==1 and p.stat().st_size<=65536,"Own bounded flat ROOT metadata only")
   total+=p.stat().st_size+4096;count+=1
 require(total+extra<=ROOT_RECORD_CAP and count<64,"Combined ROOT records plus65536event+4096node within524288 allowance")
def root_save_v1(directory,name,v):
 data=(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 require(len(data)<=65536,"Bounded actual ROOT JSON");record_budget_v1(directory,len(data)+8192)
 return save(directory,name,v)
def direct_refs_v1(q,request_ref,config_ref):
 p=pins_v1()["pins"];m=read(ref(MANIFEST))
 # Five whole operational files (including inert __init__) plus the complete closure
 # anchor, review and actual request/config. Fixed library/history refs stay exact.
 rows=list(m["executed_source_refs"])+[ref(MANIFEST),q["source_review_ref"],request_ref,config_ref,RUNTIME_REF,p["CENSUS_GUARD"],GPU0_GUARD,BASE_POOL,OWNED,PIDFD,original.NAMESPACE_MEMO,p["ARTIFACT_ACCEPTANCE"],p["CONTROLLER_CAP"],p["H058_INPUT"],p["H058_TERMINAL"],p["H058_DIRECT"],q["cell_election_ref"]]
 if q["phase"]in GPU_PHASES:rows.extend([q["direct_attachment_authority_ref"],q["peer_launch_capsule_ref"]])
 return rows
def prepare_v1(request_ref):
 q,_,_=validate_request_v1(request_ref);g=selected_guard_v1(consumer=False)
 require(g.MAX_JSON==65536 and g.META_CHARGE==524288,"Exact combined metadata bound against pinned child")
 directory=local(q["worker_auxiliary_scope"]).parent
 require(not directory.exists()and not local(q["output_directory"]).exists(),"Fresh separately admitted packet/output")
 policy=g._json(POLICY);implementation=policy["implementation_ref"];launcher=policy["launcher_ref"]
 raw(implementation);raw(launcher);directory.mkdir()
 config_ref=root_save_v1(directory,"config.json",config_v1(q["phase"]))
 reservation=dict(key="H056/"+q["job_id"],output=str(local(q["output_directory"])),auxiliary_scopes=[str(local(q["worker_auxiliary_scope"]))],reserved_bytes=32*2**20)
 with g.budget.v4.mutex(ROOT):
  state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
  reservation["consumer_bindings"]=g.consumer_bindings_v1(list(state[3].values()),direct_refs_v1(q,request_ref,config_ref))
  authority=g.operation_draft_v1("reserve",reservation,expected_before=g._pins(state[2]),bindings=state[3])
  authority.update(active=True,expires_at_utc=q["expires_at_utc"]);authority_ref=root_save_v1(directory,"reserve_authority.json",authority)
 receipt=g.apply_reserve_v1(authority_ref,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
 receipt_ref=root_save_v1(directory,"reserve_receipt.json",receipt)
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,job_id=q["job_id"],output_directory=q["output_directory"],reserved_new_bytes=32*2**20,reservation=receipt["value"],created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=GUARD),completed_overlay_policy_ref=POLICY,storage_policy_selector_ref=SELECTOR_REF,initial_ledger_pins=receipt["initial_ledger_pins"])
 storage_ref=root_save_v1(directory,"storage.json",storage)
 spec=dict(schema="C3_remaining_cells_actual_input_v1",root_authorizer="root_agent",status="active",execution_authorized=True,created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],phase=q["phase"],job_id=q["job_id"],output_directory=q["output_directory"],worker_auxiliary_scope=q["worker_auxiliary_scope"],source_manifest_ref=ref(MANIFEST),storage_policy_selector_ref=SELECTOR_REF,root_stage_request_ref=request_ref,refs=dict(config=config_ref,runtime=RUNTIME_REF,storage=storage_ref))
 input_ref=root_save_v1(directory,"input.json",spec);set_amendment_v1(input_ref);s,a,cfg,st,_=validate_input_v1(input_ref)
 g.validate_storage_v9(s,dict(output_cap_bytes=32*2**20),st,source_ref=ref(DRIVER),config_ref=config_ref,runtime_ref=RUNTIME_REF)
 admitted_ref=root_save_v1(directory,"admitted.json",dict(input_ref=input_ref,reserve_receipt_ref=receipt_ref,source_manifest_ref=ref(MANIFEST),scientific_execution_started=False))
 argv=["setsid",str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",HERE.replace("/",".")+".execution_v1","--execute","--input-ref",json.dumps(input_ref),"--amendment-ref",json.dumps(input_ref)]
 return dict(status="ADMITTED_NOT_EXECUTED",input_ref=input_ref,admitted_ref=admitted_ref,execute_argv=argv,automatic_execution=False)
def main(argv=None):
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group();actions.add_argument("--prepare-ref");actions.add_argument("--draft-election-ref");args=parser.parse_args(argv)
 if args.prepare_ref is None and args.draft_election_ref is None:print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)));return 0
 require(sys.platform.startswith("linux")and sys.dont_write_bytecode and "torch"not in sys.modules and "torch_geometric"not in sys.modules,"Fresh isolated stdlib ROOT preparation")
 print(json.dumps(draft_election_v1(json.loads(args.draft_election_ref))if args.draft_election_ref is not None else prepare_v1(json.loads(args.prepare_ref)),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
