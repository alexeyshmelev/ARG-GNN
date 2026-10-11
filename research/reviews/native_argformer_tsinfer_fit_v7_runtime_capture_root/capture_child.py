import hashlib,importlib.util,json,os,pathlib,sys
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"research/reviews/native_argformer_tsinfer_fit_v7_runtime_capture_root"
OUT=ROOT/"experiments/results/native_argformer_tsinfer_fit_v7_root"
reg=json.loads((BASE/"registration.json").read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for b in [reg["implementation"],reg["config"],reg["independent_source_review"]]:
 if sha(ROOT/b["path"])!=b["sha256"]:raise ValueError("Frozen capture source/review differs")
spec=importlib.util.spec_from_file_location("_root_native_v7_capture",ROOT/reg["implementation"]["path"])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
c=m.policy();m.verify_source(c)
runtime=json.loads((BASE/"capture_binding.json").read_text())
value=m.capture_runtime_inventory(c,runtime)
payload=json.dumps(value,indent=2).encode()+b"\n"
if len(payload)>30000000:raise ValueError("Bounded inventory payload exceeded")
destination=OUT/"environment_inventory.json"
with destination.open("xb") as stream:stream.write(payload)
print(json.dumps({"inventory":{"path":str(destination.relative_to(ROOT)),"sha256":sha(destination)},"module_count":len(value["scientific_modules"]),"distribution_count":len(value["distributions"]),"import_file_count":len(value["import_file_sha256"]),"directory_symlink_count":len(value["directory_symlinks"]),"absent_executable_symlink_count":len(value["absent_executable_symlinks"]),"actual_helper_versions":[x["version"] for x in value["additive_helpers"]],"model_instantiated":False,"CUDA_context_created":False}),flush=True)
