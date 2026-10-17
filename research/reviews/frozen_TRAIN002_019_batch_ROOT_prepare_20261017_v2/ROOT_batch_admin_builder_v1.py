"""ROOT metadata builder for the reviewed v3 batch. Default inert; no resource probes or guard imports."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/frozen_TRAIN002_019_batch_ROOT_prepare_20261017_v2"
BATCH="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3"
MANIFEST={"path":BATCH+"/source_manifest.v1.json","sha256":"5b6f1bb18747bc7c3630be210187419e2d7c0167d4b10b8f22c4a8364abe5d1e"}
REVIEW={"path":"research/reviews/frozen_TRAIN002_019_batch_independent_20261017_v3/review.v3.json","sha256":"a4cef04a6aa780da68e9559b89177afa8fdbb17c8254ca74942c6c0614fef422"}
RECIPE={"path":BATCH+"/ROOT_RECIPE.v1.md","sha256":"fbf16b1812afd39a2c138e4ce2e6683c8ffc14542fc67618600df82ce857dc82"}
TEMPLATE=BATCH+"/ROOT_batch_authority.template.inactive.v1.json"
MODULE=BATCH.replace("/",".")+".batch_ROOT_v1"
def need(ok,message):
 if not ok:raise ValueError(message)
def local(value):
 p=Path(value);p=p if p.is_absolute()else ROOT/p
 need(p!=ROOT and p.is_relative_to(ROOT)and".."not in p.parts and p.resolve()==p,"Canonical contained metadata")
 return p
def ref(value):
 p=local(value);st=p.lstat()
 need(stat.S_ISREG(st.st_mode)and st.st_nlink==1 and st.st_size<=2**20,"Bounded singly linked metadata/source")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(rr):
 need(type(rr)is dict and set(rr)=={"path","sha256"}and ref(rr["path"])==rr,"Exact immutable metadata/source ref")
 return json.loads(local(rr["path"]).read_text())
def static_packet():
 m=read(MANIFEST);r=read(REVIEW)
 for rr in m["refs"]:need(ref(rr["path"])==rr,"Exact sealed batch source closure")
 expected=[ref(BATCH+"/"+n)for n in("worker_entry_v1.py","batch_ROOT_v1.py","close_unit_ROOT_v1.py")]
 need(r["independent"]is True and r["verdict"]=="SOURCE_PASS"and r["blockers"]==[]and r["source_manifest_ref"]==MANIFEST
  and all(x in r["reviewed_sources"]for x in expected),"Exact independent batch source pass")
 need(ref(RECIPE["path"])==RECIPE and local(RECIPE["path"]).read_text().count("-m "+MODULE)==2,"Exact v3 recipe with both actual module commands")
 return read(ref(TEMPLATE))
def authority_value(parameters,template,now):
 need(type(parameters)is dict and set(parameters)=={"schema","root_authorizer","once_authorized","scope_id","CPU_lanes","expires_at_utc",
  "actual_capacity_observed_at_utc","admissible_new_bytes","actual_current_capacity_observed","source_only_estimate",
  "ROOT_exclusive_initial_admin_admission_until_coordinator_registered","ROOT_every_manual_mutation_uses_same_coordinator","use_corrected_v2_commands"},
  "Exact ROOT actual-parameter schema")
 need(parameters["schema"]=="ROOT_TRAIN002_019_batch_actual_parameters_v1"and parameters["root_authorizer"]=="root_agent"
  and parameters["once_authorized"]is True and parameters["actual_current_capacity_observed"]is True and parameters["source_only_estimate"]is False
  and parameters["ROOT_exclusive_initial_admin_admission_until_coordinator_registered"]is True
  and parameters["ROOT_every_manual_mutation_uses_same_coordinator"]is True and parameters["use_corrected_v2_commands"]is True,
  "Explicit ROOT actual capacity and exclusive/shared coordinator handover")
 need(now.tzinfo is not None,"Actual aware ROOT creation clock")
 observed=datetime.fromisoformat(parameters["actual_capacity_observed_at_utc"]);end=datetime.fromisoformat(parameters["expires_at_utc"])
 need(observed.tzinfo and end.tzinfo and 0<=(now-observed).total_seconds()<=300
  and 16200<(end-now).total_seconds()<=172800,"Actual capacity within5min and finite batch deadline<=48h")
 scope=parameters["scope_id"];lanes=parameters["CPU_lanes"];capacity=parameters["admissible_new_bytes"]
 need(type(scope)is str and 1<=len(scope)<=48 and all(c.isalnum()or c in "_-"for c in scope),"Fresh bounded ROOT scope_id")
 need(type(lanes)is list and len(lanes)==2 and all(type(v)is list and len(v)==8 and all(type(i)is int and i>=0 for i in v)for v in lanes)
  and len(set(lanes[0]+lanes[1]))==16,"Exactly two disjoint8CPU lanes; actual availability later checked by original qualified")
 need(type(capacity)is int and capacity>=33554432,"ROOT actual capacity for genuine32MiB admin only")
 a=dict(template)
 need(a["active"]is False and a["source_manifest_ref"]==MANIFEST and a["source_review_ref"]is None,"Exact inactive template")
 a.update(active=True,execute_registered_simulations_and_ordinary_closes=True,
  ROOT_exclusive_initial_admin_admission_until_coordinator_registered=True,source_review_ref=REVIEW,scope_id=scope,CPU_lanes=lanes,
  run_directory=BATCH+"/ROOT_actual_batch_"+scope,admission_directory=BATCH+"/ROOT_batch_admission_"+scope,
  created_at_utc=now.isoformat(),expires_at_utc=end.isoformat(),
  actual_initial_capacity=dict(root_authorizer="root_agent",actual_current_observation=True,source_only_estimate=False,
   observed_at_utc=observed.isoformat(),admissible_new_bytes=capacity))
 return a
def save(directory,name,value):
 data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode();need(len(data)<=65536,"Compact ROOT metadata")
 with(directory/name).open("xb")as f:need(f.write(data)==len(data),"Complete exclusive write");f.flush();os.fsync(f.fileno())
 return ref(directory/name)
def argv(flag,rr):
 return[str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",MODULE,flag,json.dumps(rr,separators=(",",":"))]
def build_once(parameters):
 template=static_packet();now=datetime.now(timezone.utc);a=authority_value(parameters,template,now)
 output=local(HERE+"/ROOT_once_"+a["scope_id"])
 need(not output.exists()and not local(a["run_directory"]).exists()and not local(a["admission_directory"]).exists(),"Fresh ROOT metadata/admission/run namespaces; never replay")
 output.mkdir()
 pr=save(output,"actual_parameters.json",parameters);ar=save(output,"batch_authority.json",a)
 return dict(status="ROOT_METADATA_ONLY_BUILT",authority_ref=ar,actual_parameters_ref=pr,admission_argv=argv("--admit-batch-ref",ar),
  cwd=str(ROOT),required_environment=dict(PYTHONPATH=".:src:scripts",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1"),
  actual_resource_probes=False,guard_imported=False,admission_started=False,
  next_requirement="ROOT exclusive monitored admission once; consume actualOS0 before using genuine returned input. Preserve unknowns without replay.")
def run_argv(input_ref):
 static_packet();inp=read(input_ref)
 need(inp["schema"]=="ROOT_TRAIN002_019_batch_admitted_input_v1"and inp["source_manifest_ref"]==MANIFEST,"Actual admitted input metadata")
 a=read(inp["authority_ref"])
 need(a["source_manifest_ref"]==MANIFEST and a["source_review_ref"]==REVIEW and a["active"]is True,"Same exact batch source/review")
 need(inp["storage"]["reservation"]["reserved_bytes"]==33554432 and inp["coordinator"]["path"]==a["run_directory"]+"/accounting_coordinator.lock",
  "Recorded32MiB batch and exact coordinator path; original run_batch performs live original journal/inode checks")
 return dict(status="ARGV_ONLY_NO_EXECUTION",run_argv=argv("--run-batch-ref",input_ref),cwd=str(ROOT),
  source_review_ref=REVIEW,required_recipe_ref=RECIPE,guard_imported=False,execution_started=False,
  requirement="ROOT already consumed genuine admissionOS0; dedicated monitored session, shared accounting helper, final external terminal still required.")
def main(args=None):
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
 g.add_argument("--build-once-parameters-json");g.add_argument("--run-argv-from-admitted-input-ref")
 a=p.parse_args(args)
 if not(a.build_once_parameters_json or a.run_argv_from_admitted_input_ref):
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",resource_probes=False,guard_imported=False,metadata_written=False)));return 0
 result=build_once(json.loads(a.build_once_parameters_json))if a.build_once_parameters_json else run_argv(json.loads(a.run_argv_from_admitted_input_ref))
 print(json.dumps(result,sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
