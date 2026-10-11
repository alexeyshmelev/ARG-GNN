#!/usr/bin/env python3
"""Root admission for unchanged native ARGformer preparation with repaired json binding."""
import datetime as dt, hashlib, json, os, subprocess, fcntl
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
os.chdir(ROOT)
OUT=ROOT/"experiments/results/official_argformer_tsinfer_parent_train001_v3_root"
PARENT="bio_paths_pilot_train_001"
RUN_ID="official_argformer_tsinfer_parent_train001_v3_native_preparation"
SOURCE="scripts/prepare_official_argformer_tsinfer_parent_v3.py"
CONFIG="configs/official_argformer_tsinfer_parent_v3.json"
EXPECTED_SOURCE="3b767b7e5e0d86291173f80f2df1feff4122592753262de8950aed9a724b1bd3"
EXPECTED_CONFIG="686356c3ceb0e15ed7b500f00d97bcae4fe2d91d13b3c8317362392f09414616"
EXPECTED_LEDGER="c7749bf01317bd1676268318ac68680a4a624626a1a408b8144af35ab69db1d1"
def digest(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def ref(p):
 p=Path(p)
 return {"path":str(p.relative_to(ROOT) if p.is_absolute() else p),"sha256":digest(p)}
def load(p): return json.loads((ROOT/p).read_bytes())
def save(name,data):
 p=OUT/name
 with p.open("x") as f: f.write(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
 return ref(p)
assert digest(SOURCE)==EXPECTED_SOURCE and digest(CONFIG)==EXPECTED_CONFIG
c=load(CONFIG)
independent="research/reviews/official_argformer_tsinfer_parent_v3_independent/review_receipt.json"
assert digest(independent)=="ecaad4f91eb4e4774e6169ebd14770044bf5684013b3bbb8f54fe1188a73ae0a"
assert load(independent)["verdict"]=="PASS"
assert digest(c["combined_observed_manifest"]["path"])==c["combined_observed_manifest"]["sha256"]
assert not (ROOT/c["output_root"]/PARENT).exists()
assert not (ROOT/c["launch_log_root"]/(PARENT+".launch.log")).exists()
decisions=load("research/experiment_decisions.json")
matches=[(h["id"],r) for h in decisions["decisions"] for r in h.get("registered_runs",[]) if r.get("id")==RUN_ID]
assert len(matches)==1 and matches[0][0]=="H013"
native=ROOT/c["runtime"]["interpreter"]
probe='import sys,hashlib,importlib.metadata,json; print(json.dumps(dict(prefix=sys.prefix,executable=sys.executable,python_version=list(sys.version_info[:3]),python_executable_sha256=hashlib.sha256(open(sys.executable,"rb").read()).hexdigest(),distributions={n:importlib.metadata.version(n) for n in ["numpy","tskit","torch"]})))'
env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
r=subprocess.run([str(native),"-c",probe],env=env,text=True,capture_output=True,timeout=120)
save("runtime_probe.json",{"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr,"command":[str(native),"-c",probe],"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"array_imports":False})
assert r.returncode==0
actual=json.loads(r.stdout)
assert actual["python_version"]==c["runtime"]["python_version"] and actual["distributions"]==c["runtime"]["distributions"] and actual["prefix"]==str(ROOT/c["runtime"]["prefix"])
templates=Path("research/reviews/official_argformer_tsinfer_parent_v3_author")
runtime=load(templates/"CPU_runtime_template.json")
runtime.pop("template_not_runtime_evidence")
runtime.update(python_executable_sha256=actual["python_executable_sha256"],created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
runtime_ref=save("CPU_runtime.json",runtime)
registration=load(templates/"experiment_registration_template.json")
registration["scope_parents"]=[PARENT]
registration.update(status="registered_CPU_preparation_and_feasibility_only",hypothesis_id="H013",registered_run_id=RUN_ID,existing_registration=matches[0][1],decision_bytes_at_root_admission=ref("research/experiment_decisions.json"),created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
registration_ref=save("experiment_registration.json",registration)
from arggnn.gpu_pool_v1 import StorageLedger,snapshot_record,lock_file
p=c["storage"]
snapshot=snapshot_record(ROOT,str(ROOT/p["snapshot"]["path"]),p["snapshot"]["sha256"])
def available():
 v=os.statvfs(ROOT)
 return v.f_bavail*v.f_frsize
ledger=StorageLedger(ROOT,snapshot,p["external_reserve_bytes"],p["warning_bytes"],available)
with lock_file(ROOT,"scheduler.lock") as locked:
 fcntl.flock(locked,fcntl.LOCK_EX)
 assert digest(p["ledger_path"])==EXPECTED_LEDGER
 before=ledger._load()
 assert p["reservation_key_prefix"]+"/"+PARENT not in before["jobs"]
 before_account=ledger._check(before)
account=ledger.job(p["reservation_key_prefix"],{"job":PARENT,"storage_bytes":p["reserved_bytes"],"output":str(ROOT/c["output_root"]/PARENT)})
with lock_file(ROOT,"scheduler.lock") as locked:
 fcntl.flock(locked,fcntl.LOCK_EX)
 state=ledger._load()
 key=p["reservation_key_prefix"]+"/"+PARENT
 assert state["jobs"][key]==dict(reserved_bytes=p["reserved_bytes"],output=str(ROOT/c["output_root"]/PARENT))
 assert set(state["jobs"])==set(before["jobs"])|{key}
 assert all(state["jobs"][k]==v for k,v in before["jobs"].items())
 assert state["controllers"]==before["controllers"]
 ledger_sha=digest(p["ledger_path"])
save("new_reservation.json",{"key":key,"before_ledger_sha256":EXPECTED_LEDGER,"after_ledger_sha256":ledger_sha,"before_account":before_account,"after_account":account,"new_reservation_created":True,"retained_failed_v1_v2_slots_untouched":True,"reserved_bytes":p["reserved_bytes"]})
storage=load(templates/"storage_template.json")
storage.update(scope_parents=[PARENT],reservation_key=p["reservation_key_prefix"]+"/"+PARENT,output=str(ROOT/c["output_root"]/PARENT),write_scopes=[str(ROOT/c["output_root"]/PARENT),str(ROOT/c["launch_log_root"]/(PARENT+".launch.log"))])
storage.pop("template_not_storage_admission")
storage.update(ledger_sha256_at_admission=ledger_sha,root_storage_accounting=account,created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
storage_ref=save("storage.json",storage)
a=load(templates/"root_admission_template.json")
a.update(scope_parents=[PARENT],output=str(Path(c["output_root"])/PARENT),launch_log=str(Path(c["launch_log_root"])/(PARENT+".launch.log")))
a.update(status="root_authorized_CPU_single_parent_preparation",active=True,actual_input_tree_roster_grid_validation_authorized=True,independent_review=ref(independent),runtime=runtime_ref,storage=storage_ref,experiment_registration=registration_ref,created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
admission=save("admission.json",a)
save("root_prelaunch.json",{"admission":admission,"storage":account,"runtime":actual,"registered_experiment":"H013/"+RUN_ID,"scientific_fallback":False,"query_truth_used":False,"paper_reproduction":False,"status":"ready_CPU_preparation_only"})
print(json.dumps({"admission":admission,"storage":account},indent=2))
