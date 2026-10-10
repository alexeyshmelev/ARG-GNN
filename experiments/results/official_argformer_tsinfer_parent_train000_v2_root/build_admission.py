#!/usr/bin/env python3
"""Root-owned admission builder for the first native ARGformer observed TRAIN canary."""
import datetime as dt, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
os.chdir(ROOT)
OUT=ROOT/"experiments/results/official_argformer_tsinfer_parent_train000_v2_root"
PARENT="bio_paths_pilot_train_000"
SOURCE="scripts/prepare_official_argformer_tsinfer_parent_v2.py"
CONFIG="configs/official_argformer_tsinfer_parent_v2.json"
EXPECTED_SOURCE="c75ef5231b30fd7faa34f5d9517f2047aeecd6f884aa1741f7e2840cf91be815"
EXPECTED_CONFIG="7e0ef8bdab828201c28709afe7213b805355f3683e859a6f11cb47bed90aa76f"
def digest(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def ref(p): return {"path":str(Path(p).relative_to(ROOT) if Path(p).is_absolute() else p),"sha256":digest(p)}
def load(p): return json.loads((ROOT/p).read_bytes())
def save(name,data):
 p=OUT/name
 with p.open("x") as f: f.write(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
 return ref(p)
assert digest(SOURCE)==EXPECTED_SOURCE and digest(CONFIG)==EXPECTED_CONFIG
c=load(CONFIG)
independent="research/reviews/official_argformer_tsinfer_parent_v2_independent/review_receipt.json"
assert digest(independent)=="4e28f31dfa29e1b7b8315edd95ebbde3fe3c30a43f25aad733d72aa41a842a07"
review=load(independent)
assert review["verdict"]=="PASS"
decisions=load("research/experiment_decisions.json")
matches=[(h["id"],r) for h in decisions["decisions"] for r in h.get("registered_runs",[]) if r.get("id")=="official_argformer_tsinfer_parent_train000_v2_lock_repair_canary"]
assert len(matches)==1 and matches[0][0]=="H013"
native=ROOT/c["runtime"]["interpreter"]
probe='import sys,hashlib,importlib.metadata,json; print(json.dumps(dict(prefix=sys.prefix,executable=sys.executable,python_version=list(sys.version_info[:3]),python_executable_sha256=hashlib.sha256(open(sys.executable,"rb").read()).hexdigest(),distributions={n:importlib.metadata.version(n) for n in ["numpy","tskit","torch"]})))'
env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
r=subprocess.run([str(native),"-c",probe],env=env,text=True,capture_output=True)
save("runtime_probe.json",{"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr,"command":[str(native),"-c",probe],"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"array_imports":False})
assert r.returncode==0
actual=json.loads(r.stdout)
assert actual["python_version"]==c["runtime"]["python_version"] and actual["distributions"]==c["runtime"]["distributions"] and actual["prefix"]==str(ROOT/c["runtime"]["prefix"])
templates=Path("research/reviews/official_argformer_tsinfer_parent_v2_author")
runtime=load(templates/"CPU_runtime_template.json")
runtime.pop("template_not_runtime_evidence")
runtime["python_executable_sha256"]=actual["python_executable_sha256"]
runtime["created_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
runtime_ref=save("CPU_runtime.json",runtime)
registration=load(templates/"experiment_registration_template.json")
registration.update(status="registered_CPU_preparation_and_feasibility_only",hypothesis_id=matches[0][0],registered_run_id=matches[0][1]["id"],existing_registration=matches[0][1],decision_bytes_at_root_admission=ref("research/experiment_decisions.json"),created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
registration_ref=save("experiment_registration.json",registration)
from arggnn.gpu_pool_v1 import StorageLedger,snapshot_record
p=c["storage"]
snapshot=snapshot_record(ROOT,str(ROOT/p["snapshot"]["path"]),p["snapshot"]["sha256"])
def available():
 v=os.statvfs(ROOT); return v.f_bavail*v.f_frsize
ledger=StorageLedger(ROOT,snapshot,p["external_reserve_bytes"],p["warning_bytes"],available)
import fcntl
from arggnn.gpu_pool_v1 import lock_file
with lock_file(ROOT,"scheduler.lock") as locked:
 fcntl.flock(locked,fcntl.LOCK_EX)
 state=ledger._load()
 assert state["jobs"][p["reservation_key_prefix"]+"/"+PARENT]==dict(reserved_bytes=p["reserved_bytes"],output=str(ROOT/c["output_root"]/PARENT))
 account=ledger._check(state)
save("existing_reservation_reuse.json",{"key":p["reservation_key_prefix"]+"/"+PARENT,"reserved_bytes":p["reserved_bytes"],"new_reservation_created":False,"retained_failed_v1_slot_reused":True,"source_v2_changes_only_lock_path_and_namespace":True,"initial_failure_immutable_snapshot":"research/reviews/official_argformer_tsinfer_parent_v2_author/initial_v1_snapshot"})
storage=load(templates/"storage_template.json")
storage.pop("template_not_storage_admission")
storage["ledger_sha256_at_admission"]=digest(p["ledger_path"])
storage["root_storage_accounting"]=account
storage["created_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
storage_ref=save("storage.json",storage)
a=load(templates/"root_admission_template.json")
a.update(status="root_authorized_CPU_single_parent_preparation",active=True,actual_input_tree_roster_grid_validation_authorized=True,independent_review=ref(independent),runtime=runtime_ref,storage=storage_ref,experiment_registration=registration_ref,created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
admission=save("admission.json",a)
save("root_prelaunch.json",{"admission":admission,"storage":account,"runtime":actual,"registered_experiment":"H013/official_argformer_tsinfer_parent_train000_v2_lock_repair_canary","scientific_fallback":False,"query_truth_used":False,"paper_reproduction":False,"status":"ready_CPU_preparation_only"})
print(json.dumps({"admission":admission,"storage":account},indent=2))
