"""Root-only single-batch H018 metadata and actual regional map qualification."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,fcntl,hashlib,importlib.util,json,os,resource,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/H018_metadata_root_execution_20261015_v1"
HERE=ROOT/REL
SOURCE=REL+"/root_metadata_driver.py"
OUTPUT="research/reviews/H018_genealogy_next_source_v1/root_metadata_stage_basis_v1"
PACKET=REL+"/admission_v1"
JOB="H018-regional-map-and-date-metadata-v1"
KEY="H018/regional_map_and_date_metadata_v1"
CAP=5000000
BASE_PATH="research/reviews/H018_map_acquisition_root_source_v1/root_map_driver.py"
BASE_SHA="4c957db44a1bf6c9cd8e76fd6cd5e31b979b7c618554c6a73c57cce8099af65d"
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module
def common():
 assert hashlib.sha256((ROOT/BASE_PATH).read_bytes()).hexdigest()==BASE_SHA
 return load("_H018_unchanged_map_admission_primitives",BASE_PATH)
def verify(b,r):
 assert r["schema"]=="H018_metadata_root_request_v1" and r["status"]=="active"
 assert r["root_authorizer"]=="root_agent" and r["registered_before_execution"] is True
 assert r["output_directory"]==OUTPUT and r["packet_directory"]==PACKET
 assert r["reserved_bytes"]==CAP and r["job_id"]==JOB and r["GPU_authorized"] is False
 assert r["driver_ref"]==b.ref(SOURCE)
 for rec in r["source_refs"].values(): assert b.ref(rec["path"])==rec
 sem=b.read(r["semantics_ref"])
 assert sem["status"]=="root_reviewed_source_semantics"
 assert sem["position_basis"]=="absolute_bp1" and sem["source_format"]=="pos_chr_cM"
 assert sem["exact_header_tokens"]==["pos","chr","cM"]
 b.read(r["registration_ref"])
 acquisition=b.read(r["acquisition_completion_ref"])
 assert acquisition["status"]=="completed_exact3_source_assets_only"
 assert acquisition["assets"]["chr22.b37.gmap.gz"]["sha256"]==r["map_ref"]["sha256"]
 assert r["map_ref"]["sha256"]=="75b0026033139f7c2515eda80c330d93d1eb728374a31e0a713589c0d413b24f"
 return sem
def admit(record):
 b=common();r=b.read(record);verify(b,r)
 packet=b.local(PACKET)
 assert not packet.exists() and not b.local(OUTPUT).exists()
 packet.mkdir()
 expiry=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat()
 runtime=b.save(packet,"runtime.json",dict(schema="H018_metadata_CPU_runtime_v1",job_id=JOB,
     device="cpu",physical_gpu=None,python=sys.executable,python_version=sys.version,
     max_control_address_space_bytes=1000000000,output_directory=OUTPUT,expires_at_utc=expiry))
 # Coordinate source refs are data evidence verified by the unchanged helper.
 # Direct bindings seal executable/config/authority closure under the unchanged guard.
 m=b.guard()
 lock=ROOT/"research/reviews/American_C3_short_train2mb_wave_source_v1/scheduler/admission.lock"
 with lock.open("a+") as stream:
  fcntl.flock(stream,fcntl.LOCK_EX)
  view=m.inspect_v1(expected_source_sha256=b.CORE,expected_launcher_sha256=b.BASE)
  refs=[record,r["driver_ref"],runtime,r["semantics_ref"],r["registration_ref"],
        *r["source_refs"].values(),b.GUARD,b.POLICY,b.REPAIR,*view["bindings"].values()]
  refs=list({x["path"]:x for x in refs}.values())
  assert len(refs)<=64
  op=m.operation_draft_v1("reserve",dict(key=KEY,output=str(ROOT/OUTPUT),auxiliary_scopes=[],
       reserved_bytes=CAP,consumer_bindings=refs),expected_before=m._pins(view),bindings=view["bindings"])
  op.update(active=True,expires_at_utc=expiry)
  receipt=m.apply_reserve_v1(b.save(packet,"reserve_authority.json",op),
       expected_source_sha256=b.CORE,expected_launcher_sha256=b.BASE)
 b.save(packet,"reserve_receipt.json",receipt)
 storage=b.save(packet,"storage.json",dict(kind="project_budget_v9_storage_record",active=True,
     consumer_uses_v9_guard=True,accounting_mode=m.MODE,job_id=JOB,output_directory=OUTPUT,
     reserved_new_bytes=CAP,reservation=receipt["value"],expires_at_utc=expiry,
     initial_ledger_pins=receipt["initial_ledger_pins"],completed_overlay_policy_ref=b.POLICY,
     budget_components=dict(implementation=b.ref(m.budget.SOURCE_REL),launcher=b.ref(m.budget.LAUNCHER_REL),consumer_adapter=b.GUARD)))
 result=b.save(packet,"admitted.json",dict(schema="H018_metadata_root_admitted_v1",
     request_ref=record,driver_ref=b.ref(SOURCE),runtime_ref=runtime,storage_ref=storage,
     output_directory=OUTPUT,job_id=JOB,reserved_bytes=CAP,expires_at_utc=expiry))
 print(json.dumps(dict(admitted_ref=result,reserved_bytes=CAP)),flush=True)
 return result
def run(record):
 b=common();ad=b.read(record);r=b.read(ad["request_ref"]);sem=verify(b,r)
 assert ad["driver_ref"]==b.ref(SOURCE) and record["path"]==PACKET+"/admitted.json"
 assert ad["job_id"]==JOB and ad["output_directory"]==OUTPUT and ad["reserved_bytes"]==CAP
 assert datetime.fromisoformat(ad["expires_at_utc"])>datetime.now(timezone.utc)
 runtime=b.read(ad["runtime_ref"]);storage=b.read(ad["storage_ref"])
 assert runtime["python"]==sys.executable and runtime["python_version"]==sys.version
 assert not b.local(OUTPUT).exists()
 b.save(b.local(PACKET),"run_claim.json",dict(pid=os.getpid(),started_utc=b.utc(),admitted_ref=record))
 os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 resource.setrlimit(resource.RLIMIT_AS,(1000000000,1000000000))
 helper=load("_H018_original_metadata_and_map_helper",r["source_refs"]["helper"]["path"])
 m=b.guard();calls=[]
 def actual_guard(amount):
  assert amount==4000000 and not calls
  began=time.monotonic()
  m.budget_check_v9(storage,first=True,boundary=True,required_write_bytes=CAP)
  calls.append(dict(required_write_bytes=CAP,elapsed_seconds=time.monotonic()-began))
 start=time.monotonic()
 result=helper.write_root_metadata_stage_v1(Path(OUTPUT).name,guard=actual_guard,
     map_ref=r["map_ref"],semantics=sem)
 assert len(calls)==1
 report=b.read(b.ref(OUTPUT+"/map_qualification.json"))
 assert report["source_semantics_qualified"] is True
 assert report["output_rows"]==12280 and report["native_source_knots_inside_region"]==12278
 assert abs(report["regional_distance_cM"]-23.129082505768428)<1e-10
 assert report["uniform_map_substitution"] is False
 roster=b.read(b.ref(OUTPUT+"/planned_haplotype_roster.json"))
 assert len(roster)==40 and sum(x["role"]=="reference" for x in roster)==16
 assert all(x["query_ancestry_truth"] is None for x in roster)
 assets={p.name:b.ref(p) for p in b.local(OUTPUT).iterdir() if p.is_file()}
 completed=b.save(b.local(OUTPUT),"completed.json",dict(schema="H018_metadata_map_completed_v1",
     status="completed_source_qualified_map_and_sample_metadata",admitted_ref=record,
     assets=assets,map_rows=report["output_rows"],regional_distance_cM=report["regional_distance_cM"],
     helper_result=result,guard_calls=calls,elapsed_seconds=time.monotonic()-start,
     peak_control_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
     real_genealogy_inference=False,model_prediction=False,ancestry_accuracy=False,
     no_internal_OS_exit_or_wait0_claim=True,finished_utc=b.utc()))
 print(json.dumps(dict(completed_ref=completed)),flush=True)
 return 0
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("action",choices=["admit","run"]);p.add_argument("--execute",action="store_true")
 p.add_argument("--record");p.add_argument("--sha256");a=p.parse_args()
 if not a.execute: print('{"status":"inactive_source_only"}');return 0
 assert a.record and a.sha256 and Path.cwd().resolve()==ROOT
 rec=dict(path=a.record,sha256=a.sha256)
 if a.action=="admit":admit(rec);return 0
 return run(rec)
if __name__=="__main__":raise SystemExit(main())
