from pathlib import Path
import os,json,hashlib,datetime,sys,importlib,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');os.chdir(ROOT)
OUT=ROOT/'research/reviews/H060_isolated_CPU_ROOT_preparation_fast_observed_root01_oct17_v1'
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def put(p,j):
 with p.open('x') as f:json.dump(j,f,indent=2);f.write('\n')
try:
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sys.dont_write_bytecode and sys.flags.no_user_site
 ids=list(range(128,160));allowed=sorted(os.sched_getaffinity(0));assert set(ids)<=set(allowed)
 os.sched_setaffinity(0,ids)
 import torch,torch_geometric,numpy
 torch.set_num_threads(32)
 assert not torch.cuda.is_initialized()
 base=ROOT/'research/reviews/H060_fast_observed_full_training_source_20261017_v1'
 manifest=base/'source_manifest.v1.json';j=json.loads(manifest.read_text())
 for path,sha in j['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
 binding=json.loads((base/'scientific_binding.v1.json').read_text())
 module=importlib.import_module(binding['runner_module']);assert ref(Path(module.__file__))==binding['runner_source_ref']
 assert not torch.cuda.is_initialized()
 runtime=json.loads((base/'runtime.INACTIVE.template.v1.json').read_text());runtime.update(actual_runtime_qualified=True,cpu_ids=ids)
 put(OUT/'runtime.root_v1.json',runtime)
 record={'schema':'ROOT_H060_fast_observed_actual_metadata_runtime_qualification_v1','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_manifest_ref':ref(manifest),'runner_source_ref':binding['runner_source_ref'],'runtime_ref':ref(OUT/'runtime.root_v1.json'),'root_authorizer':'root_agent','status':'qualified','unresolved_items':[],'standalone_token_update_probe':False,'scientific_probe_updates':0,'full_training_fit_measured':False,'production_FP32_equivalence_measured':False,'package_versions':{'torch':torch.__version__,'torch_geometric':torch_geometric.__version__,'numpy':numpy.__version__},'actual_interpreter':str(Path(sys.executable).resolve()),'cpu_ids':ids,'CPU_id_observation':'Allowed affinity identities, not exclusive idle-core reservation','torch_threads':torch.get_num_threads(),'CUDA_initialized':torch.cuda.is_initialized(),'all_source_pins_verified':len(j['source_sha256']),'observed_inputs_truth_checkpoints_models_or_updates_loaded':False,'meminfo_subset':{line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ('MemTotal','MemAvailable')}}
 put(OUT/'runtime_qualification.root_v1.json',record);print(json.dumps(record),flush=True)
except BaseException:
 text=traceback.format_exc();put(OUT/'runtime_qualification_failure.root_v1.json',{'traceback':text,'no_admission_or_training':True});print(text,flush=True);raise SystemExit(1)
