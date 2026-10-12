"""Draft canonical native GPU consumer. Root promotes exact bytes before admission."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,sys
from arggnn import expanded_native_argmix_v1 as adapter
from arggnn import prepare_expanded_native_argmix_v3 as prep
from arggnn import project_storage_terminal_retired_scopes_v1 as accounting
native=adapter.native;ROOT=accounting.ROOT
SOURCE="src/arggnn/expanded_native_argmix_gpu_fit_root_v1.py"
DRAFT="research/reviews/expanded_full_panel_v1_root/native_launch_helpers_v1/argmix_native_gpu_fit_worker_v1.py"
HELPER="research/reviews/expanded_full_panel_v1_root/native_launch_helpers_v1/argmix_native_gpu_fit_v1.py"
UTILITY="research/reviews/expanded_full_panel_v1_root/retire_inactive_scopes_before_fast_cache_v1.py"
GPU_WRAPPER="scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py"
SOURCE_PATHS=tuple(dict.fromkeys(prep.SOURCE_PATHS+(SOURCE,DRAFT,HELPER,UTILITY,
 "src/arggnn/resources.py","src/arggnn/native_argformer_exec.py","scripts/qualify_official_argformer_runtime.py",
 GPU_WRAPPER,"scripts/launch_reserved_gpu_full_storage_v1.py")))
SCHEMA="expanded_native_argmix_GPU_fit_root_v1"
ADAPTER_SHA="00c8e57423d38b484bda3cfeb126c188add63c383e399df874a1a2e55e4bca23"


def admit(input_ref,first):
 if Path(__file__).resolve()!=ROOT/SOURCE:raise ValueError("Root-promoted canonical consumer required")
 spec=accounting._bound(input_ref)
 if (spec.get("schema")!=SCHEMA or spec.get("status")!="active" or spec.get("root_authorizer")!="root_agent"
  or spec.get("fit_authorized") is not True or spec.get("GPU_authorized") is not True
  or spec.get("all8_VAL_inference_authorized") is not True or spec.get("outer_included") is not False):
  raise ValueError("Exact root native fit/eight VAL authority required")
 expiry=datetime.fromisoformat(spec["expires_at_utc"])
 if expiry.tzinfo is None or expiry<=datetime.now(timezone.utc):raise ValueError("Expired native fit authority")
 sources=spec["source_sha256"]
 if set(sources)!=set(SOURCE_PATHS):raise ValueError("Exact native GPU consumer/source closure required")
 for name,sha in sources.items():
  if accounting._ref(name)!={"path":name,"sha256":sha}:raise ValueError("Native fit source changed: "+name)
 if sources[adapter.SOURCE]!=ADAPTER_SHA:raise ValueError("Frozen scientific adapter required")
 refs=[spec[k] for k in ("config_ref","runtime_ref","prepared_ref","prep_completed_ref","prep_input_ref",
  "prep_root_wait_ref","build_ref","CPU_library_admission_ref","size_bounds_ref")]
 for ref in refs:accounting._bound(ref)
 cfg=adapter.load_config_v1(spec["config_ref"])
 runtime=accounting._bound(spec["runtime_ref"])
 if (runtime["device"]!="cuda" or runtime["native_prefix"]!=str(native.PREFIX.relative_to(ROOT))
  or runtime["source_ref"]!=accounting._ref(SOURCE) or runtime["job_id"]!=spec["job_id"]
  or os.environ.get("ARGGNN_JOB_NAME")!=spec["job_id"]):raise ValueError("Bound actual native GPU runtime required")
 needed=refs+[accounting._ref(name) for name in SOURCE_PATHS]
 if any(ref not in spec["storage"]["reservation"]["consumer_bindings"] for ref in needed):raise ValueError("Unreserved native binding")
 if first:accounting.validate_storage_v9(spec,dict(cfg,output_cap_bytes=spec["output_cap_bytes"]),spec["storage"],
  source_ref=accounting._ref(SOURCE),config_ref=spec["config_ref"],runtime_ref=spec["runtime_ref"])
 else:accounting.budget_check_v9(spec["storage"],boundary=True)
 guard=native.load_frozen(native.GUARD,native.GUARD_SHA,"_native_fit_lease")
 lease=guard.validate_lease();guard.apply_and_validate_title()
 if lease["physical_gpu"] not in (2,3) or lease["physical_gpu"]!=runtime["physical_GPU"]:raise ValueError("Actual GPU2/3 lease required")
 return spec,cfg,runtime,lease


def execute(input_ref,is_native):
 spec,cfg,runtime,lease=admit(input_ref,first=not is_native)
 guard=prep.make_guard_v1(spec,accounting);output=prep.path_v1(spec["output_directory"])
 short=prep.path_v1(spec["short_tmp_directory"])
 if not is_native:
  guard(1_000_000);output.mkdir();short.mkdir()
  for name in ("sandbox","sandbox/tmp","sandbox/cache","sandbox/config","sandbox/data","GPU_runtime"):
   (output/name).mkdir()
  env=prep.native_env_v1(output)
  env.update(CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
   PYTHONPATH=str(ROOT)+":"+str(ROOT/"scripts")+":"+str(ROOT/"src"))
  for key in ("TMPDIR","TMP","TEMP"):env[key]=str(short)
  if len(os.fsencode(short/"pymp-12345678/listener-12345678"))>107:raise ValueError("Contained IPC path exceeds Unix socket limit")
  os.set_inheritable(lease["lease_fd"],True)
  argv=[str(native.PREFIX/"bin/python"),"-B","-m","arggnn.expanded_native_argmix_gpu_fit_root_v1",
   "--native","--execute","--input",input_ref["path"],"--expected-input-sha256",input_ref["sha256"]]
  prep.save_v1(output/"native_exec.json",dict(input_ref=input_ref,command=argv,lease=lease,
   process_title="AI Auto Research Pipeline Test",native_prefix=str(native.PREFIX)),guard)
  os.execve(native.PREFIX/"bin/python",argv,env)
 if Path(sys.prefix).resolve()!=native.PREFIX:raise ValueError("Existing qualified native interpreter required")
 bounds=accounting._bound(spec["size_bounds_ref"])
 if bounds["output_cap_bytes"]!=spec["output_cap_bytes"]:raise ValueError("Root size bound/cap drift")
 modules=adapter.admit_native_v1(spec["build_ref"],spec["CPU_library_admission_ref"],cfg)
 torch,ancpath,joint,trainer,loader=modules
 # Metadata-only checkpoint size check. Stock fit reseeds50 after this CPU model.
 import yaml
 prepared=accounting._bound(spec["prepared_ref"]);frames=accounting._bound(prepared["frames"])["parents"]
 first=frames["ap_expanded_train_000"]
 backend=yaml.safe_load(native.checked(prepared["backend"]).read_text())
 backend["paths"]["references"]=str(native.checked(first["references"],observed=True))
 native.verify_backend(backend,cfg,references_record=first["references"])
 model=joint.build_joint_model_from_config(backend,device=torch.device("cpu"))
 state_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values());del model
 if 2*state_bytes+16_000_000>bounds["checkpoint_upper_bytes"]:
  raise ValueError("Native checkpoint exceeds declared model/config bound; revise admission, never truncate")
 guard(16_000_000)
 actual,anchor,gpu_modules=native.gpu_runtime(output/"GPU_runtime")
 if actual["lease"]!=lease or actual["actual_uuid"]!=runtime["canonical_uuid"]:raise ValueError("Actual native context/lease UUID drift")
 prep.save_v1(output/"GPU_runtime"/"qualified_actual_runtime.json",actual,guard)
 guard(bounds["fit_upper_bytes"])
 fit_ref=adapter.fit_v1(spec["prepared_ref"],output/"fit",cfg,guard=guard,modules=gpu_modules)
 fit_record=accounting._bound(fit_ref)
 if native.checked(fit_record["checkpoint"]).stat().st_size>bounds["checkpoint_upper_bytes"]:
  raise ValueError("Actual native checkpoint exceeds declared bound")
 guard(bounds["VAL_upper_bytes"]+bounds["workspace_upper_bytes"])
 seals=adapter.infer_VAL_v1(fit_ref,output/"VAL",cfg,guard=guard,modules=gpu_modules)
 if len(seals)!=8:raise ValueError("All eight original VAL seals required")
 guard(0)
 return prep.save_v1(output/"completed.json",dict(schema=SCHEMA+"_completed",status="completed_native_fit_all8_VAL",
  input_ref=input_ref,fit_ref=fit_ref,VAL_seals=seals,prepared_ref=spec["prepared_ref"],
  native_checkpoint_selection="unchanged smoothed masked unweighted VAL batch-mean loss",
  source_seed=50,TRAIN_parents=32,VAL_parents=8,outer_parents=0,original_paper_reproduction=False,
  truth_opened_for_external_score=False,completed_utc=prep.utc_v1()),guard)


def main():
 p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true");p.add_argument("--native",action="store_true")
 p.add_argument("--input");p.add_argument("--expected-input-sha256");a=p.parse_args()
 if not a.execute:print(json.dumps(dict(status="source_only_native_fit_consumer",canonical_source=SOURCE)));return
 if not a.input or not a.expected_input_sha256:raise ValueError("Pinned active root input required")
 execute(dict(path=a.input,sha256=a.expected_input_sha256),a.native)
if __name__=="__main__":main()
