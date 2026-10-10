#!/usr/bin/env python3
"""Root-owned admission builder for the first native ARGformer observed TRAIN canary."""
import datetime as dt, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
os.chdir(ROOT)
OUT=ROOT/"experiments/results/official_argformer_tsinfer_parent_train000_v1_root"
PARENT="bio_paths_pilot_train_000"
SOURCE="scripts/prepare_official_argformer_tsinfer_parent_v1.py"
CONFIG="configs/official_argformer_tsinfer_parent_v1.json"
EXPECTED_SOURCE="9ac0ca01b214e86841808fee951fb55e74bf57bac7227656d6ee7410d9c8a8ed"
EXPECTED_CONFIG="9e5212c6de07bb6c28e04f644b336faade04f1e06d6d488f5af49739e2529c55"
def digest(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def ref(p): return {"path":str(Path(p).relative_to(ROOT) if Path(p).is_absolute() else p),"sha256":digest(p)}
def load(p): return json.loads((ROOT/p).read_bytes())
def save(name,data):
 p=OUT/name
 with p.open("x") as f: f.write(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
 return ref(p)
assert digest(SOURCE)==EXPECTED_SOURCE and digest(CONFIG)==EXPECTED_CONFIG
c=load(CONFIG)
independent="research/reviews/official_argformer_tsinfer_parent_v1_independent/repair_v2/review_receipt.json"
assert digest(independent)=="c2c9c894f87b53b40f6621d8ce5cc1c12728dd5e077ce24e5c36e1322a4a9e90"
review=load(independent)
assert review["verdict"]=="PASS"
decisions=load("research/experiment_decisions.json")
matches=[(h["id"],r) for h in decisions["decisions"] for r in h.get("registered_runs",[]) if r.get("id")=="official_argformer_tsinfer_parent_train000_v1_preparation_intent"]
assert len(matches)==1 and matches[0][0]=="H013"
native=ROOT/c["runtime"]["interpreter"]
probe='import sys,hashlib,importlib.metadata,json; print(json.dumps(dict(prefix=sys.prefix,executable=sys.executable,python_version=list(sys.version_info[:3]),python_executable_sha256=hashlib.sha256(open(sys.executable,"rb").read()).hexdigest(),distributions={n:importlib.metadata.version(n) for n in ["numpy","tskit","torch"]})))'
env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
r=subprocess.run([str(native),"-c",probe],env=env,text=True,capture_output=True)
save("runtime_probe.json",{"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr,"command":[str(native),"-c",probe],"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"array_imports":False})
assert r.returncode==0
actual=json.loads(r.stdout)
assert actual["python_version"]==c["runtime"]["python_version"] and actual["distributions"]==c["runtime"]["distributions"] and actual["prefix"]==str(ROOT/c["runtime"]["prefix"])
templates=Path("research/reviews/official_argformer_tsinfer_parent_v1_author")
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
account=ledger.job(p["reservation_key_prefix"],dict(job=PARENT,storage_bytes=p["reserved_bytes"],output=str(ROOT/c["output_root"]/PARENT)))
storage=load(templates/"storage_template.json")
storage.pop("template_not_storage_admission")
storage["ledger_sha256_at_admission"]=digest(p["ledger_path"])
storage["root_storage_accounting"]=account
storage["created_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
storage_ref=save("storage.json",storage)
a=load(templates/"root_admission_template.json")
a.update(status="root_authorized_CPU_single_parent_preparation",active=True,actual_input_tree_roster_grid_validation_authorized=True,independent_review=ref(independent),runtime=runtime_ref,storage=storage_ref,experiment_registration=registration_ref,created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
admission=save("admission.json",a)
save("root_prelaunch.json",{"admission":admission,"storage":account,"runtime":actual,"registered_experiment":"H013/official_argformer_tsinfer_parent_train000_v1_preparation_intent","scientific_fallback":False,"query_truth_used":False,"paper_reproduction":False,"status":"ready_CPU_preparation_only"})
print(json.dumps({"admission":admission,"storage":account},indent=2))
