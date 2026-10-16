"""Independent H058v2 routing delta; inert metadata imports and handwritten writer only."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast,contextlib,hashlib,importlib,io,json,os,stat,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');HERE='research/reviews/H058_full_parent_CPU_runtime_wrapper_source_20261017_v2';BASE='research/reviews/H058_full_parent_CPU_runtime_wrapper_source_20261017_v1';REVIEW='research/reviews/H058_full_parent_CPU_runtime_wrapper_independent_review_20261017_v2';PRIOR='research/reviews/H058_full_parent_CPU_runtime_wrapper_independent_review_20261017_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
def need(ok,msg='source metadata check'):
 if not ok:raise ValueError(msg)
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def main():
 need(sys.dont_write_bytecode and sys.flags.no_user_site)
 expected={'root_driver_v1.py':'7fd85ace69044d63d4e4d99529555808e5fb5d0be32939ba766d8709a46b594b','root_contract_v1.py':'4ce42a5725565e2d1130abcc1741be36546c8fdecda60c6267a3c4421eaa054a','worker_entry_v1.py':'563ad97f6fb85b956867a075d8878674ce40b9180b92e4d899059fb1f2c946ac','source_manifest.v1.json':'90bb3a2da95c56b1f7ce0346099a8aeebde20697a94f585db6a04b81d8c70da5','source_handoff.v1.json':'5f686605619982741b95b149aa1285a40a1615685cdf96f5213a195af8ae4c04','routing_checks.v2.json':'ac50d96e46b0c9361c8aeda32cdebda0a96ff497c0cad1830471c1434a25ee60','audit_artifact_manifest.v1.json':'6a1502042160428e661f829810ac8db3a021d5f16ed31438bc1a1072f9f74b91'}
 for name,sha in expected.items():need(ref(HERE+'/'+name)['sha256']==sha,'candidate seal')
 manifest=json.loads((ROOT/HERE/'source_manifest.v1.json').read_text())
 for path,sha in manifest['source_sha256'].items():need(ref(path)['sha256']==sha,'actual source pin '+path)
 for name in ('root_driver_v1.py','root_contract_v1.py','worker_entry_v1.py'):
  text=(ROOT/HERE/name).read_text();compile(text,HERE+'/'+name,'exec')
  normalized=text.replace(HERE,BASE).replace(HERE.replace('/','.'),BASE.replace('/','.'))
  if name!='root_contract_v1.py':
   normalized=normalized.replace('from '+BASE.replace('/','.')+' import root_contract_v1 as c','from research.reviews.American_TreeLoom_matched_CPU_execution_source_20261017_v5 import root_contract_v1 as c')
  need(normalized==(ROOT/BASE/name).read_text(),'only namespace + both ImportFrom repairs')
  if name!='root_contract_v1.py':
   imports=[n for n in ast.parse(text).body if isinstance(n,ast.ImportFrom)and any(a.name=='root_contract_v1'and a.asname=='c'for a in n.names)]
   need(len(imports)==1 and imports[0].module==HERE.replace('/','.'),'exact executable ImportFrom')
 for name in ('caps.inactive.v1.json','scientific_binding.v1.json','dependencies.v1.json'):
  need((ROOT/HERE/name).read_bytes()==(ROOT/BASE/name).read_bytes(),'unchanged '+name)
 prior=json.loads((ROOT/PRIOR/'review.v1.json').read_text());producer=json.loads((ROOT/PRIOR/'producer_review.v1.json').read_text())
 need(ref(PRIOR+'/review.v1.json')['sha256']=='15caa4d8d06f1ef4b1453299da1b545e10875e0f24d3eec3ff0dea32a9774f7e')
 need(ref(PRIOR+'/producer_review.v1.json')['sha256']=='7c4e72f4814ca66fee786e6cdd4f7fe82fa6fb9194d61befe8b05723c6143a06')
 need(prior['verdict']=='SOURCE_BLOCKED'and [r['id']for r in prior['blockers']]==['R1']and producer['verdict']=='SOURCE_PASS'and producer['blockers']==[])
 c=importlib.import_module(HERE.replace('/','.')+'.root_contract_v1')
 d,w=[importlib.import_module(HERE.replace('/','.')+'.'+n)for n in ('root_driver_v1','worker_entry_v1')]
 need(d.c is w.c is c and c.HERE==d.HERE==HERE,'actual correct imported contract identity')
 need(Path(c.__file__)==ROOT/HERE/'root_contract_v1.py'and Path(d.__file__)==ROOT/HERE/'root_driver_v1.py'and Path(w.__file__)==ROOT/HERE/'worker_entry_v1.py')
 need(c.SOURCE==HERE+'/root_contract_v1.py'and c.DRIVER==d.SOURCE==HERE+'/root_driver_v1.py'and c.ENTRY==HERE+'/worker_entry_v1.py')
 need(c.PHASE=='train'and c.PARENT=='H058_European32_full_TRAIN_CPU'and len(c.TRAIN)==32)
 need(c.CAPS['max_cached_parent_bytes']==32*2**30 and c.CAPS['maximum_torch_files']==12 and c.CAPS['checkpoint_serialized_bytes']==1572864 and c.CAPS['dataloader_workers']==2)
 for module in(d,w):
  out=io.StringIO()
  with contextlib.redirect_stdout(out):need(module.main([])==0)
  need(json.loads(out.getvalue())['status']=='SOURCE_ONLY_INACTIVE')
 c.sources();c.scientific_review(c.ref(PRIOR+'/producer_review.v1.json'))
 b=c.binding();need(b['runner_source_ref']==producer['producer_ref']and b['source_manifest_ref']==producer['source_manifest_ref']and b['source_handoff_ref']==producer['source_handoff_ref'])
 # Actual twelve-name bounded writer under actual H058 contract, no real Torch or filesystem.
 files={};fds={};opens=[]
 class File(io.BytesIO):
  def __init__(self,name):super().__init__();self.name=name
  def fileno(self):return 42
  def __exit__(self,*args):files[self.name]=self.getvalue();return super().__exit__(*args)
 class P:
  def __init__(self,name=''):self.name=name
  def __truediv__(self,name):return P(name)
  def lstat(self):return NS(st_mode=stat.S_IFREG,st_nlink=1,st_dev=1,st_ino=2)
  def relative_to(self,root):return Path('SYNTHETIC_H058_OUTPUT')/self.name
 def opened(p,*args):need(p.name not in files);fds[42]=p.name;opens.append(p.name);return 42
 fakeos=NS(O_WRONLY=os.O_WRONLY,O_CREAT=os.O_CREAT,O_EXCL=os.O_EXCL,O_NOFOLLOW=os.O_NOFOLLOW,statvfs=lambda p:NS(f_bavail=10**9,f_frsize=4096),open=opened,fdopen=lambda fd,mode:File(fds[fd]),fsync=lambda fd:None,fstat=lambda fd:NS(st_dev=1,st_ino=2))
 class Scalar:
  def __init__(self,v):self.v=v
  def item(self):return self.v
 class Array:
  def __init__(self,v):self.v=v
  def tolist(self):return self.v
 class Tensor:
  def __init__(self,v):self.v=v
  def detach(self):return self
  def cpu(self):return self
  def numpy(self):return Array(self.v)
 torch=NS(Tensor=Tensor,save=lambda v,buf:buf.write(b'SYNTHETIC_H058_STATE_NOT_MODEL_BYTES'));np=NS(ndarray=Array,generic=Scalar)
 writer=object.__new__(w.BoundedWriter);writer.fast=lambda *a:None;writer.s={'storage':{}};writer.g=NS(budget_check_v9=lambda *a,**kw:None);writer.torch=torch;writer.jsonable=lambda v:w.jsonable_runtime_v1(v,torch,np);writer.path=P();writer.used=8192;writer.json_used=0;writer.torch_files=0;writer.names=set()
 old_os=w.os;w.os=fakeos
 try:
  writer.save('ROOT_started.json',dict(updates_per_arm=960,total_updates=1920,CPU_threads=32,physical_gpu=None))
  writer.save('CPU_complete_phase_preparation_cache.json',{'nested':(Array([Scalar(1)]),Tensor([2.0]))})
  for arm in ('aligned','alignment_null'):
   for at in (5,10,15,20,25,30):writer.save('same_marker_%s.pass_%02d.pt'%(arm,at),{},kind='torch')
  writer.save('completed.json',dict(status='completed_fixed30pass_full_TRAIN',steps={'same_marker_aligned':960,'same_marker_alignment_null':960}))
  writer.save('ROOT_completed.json',dict(status=c.STATUSES[c.PHASE],parent_id=c.PARENT))
  need(writer.torch_files==12 and json.loads(files['CPU_complete_phase_preparation_cache.json'])=={'nested':[[1],[2.0]]})
  before=len(opens)
  try:writer.save('same_marker_aligned.pass_30.pt',{},kind='torch')
  except ValueError:pass
  else:raise AssertionError('exclusive product duplicate rejected')
  need(len(opens)==before)
  capped=w.CappedBuffer(c.CAPS['checkpoint_serialized_bytes']);capped.write(b'x'*c.CAPS['checkpoint_serialized_bytes'])
  try:capped.write(b'y')
  except ValueError:pass
  else:raise AssertionError('actual H058 state cap rejected')
 finally:w.os=old_os
 need(c._h._GUARD is None and c.SELECTOR_REF is None and not any(n in sys.modules for n in ('torch','numpy','torch_geometric','arggnn.census_tsinfer_fit_v1','scripts.project_budget_v9','arggnn.project_budget_v4')))
 return dict(schema='H058_CPU_v2_independent_routing_delta_check_v1',status='SOURCE_PASS_INACTIVE_IMPORTS_AND_HANDWRITTEN_METADATA_ONLY',source_map_pins_authenticated=len(manifest['source_sha256']),exact_only_namespace_and_two_ImportFrom_delta=True,both_executable_ImportFrom_exact_v2=True,actual_driver_worker_contract_same_identity=True,actual_HERE_SOURCE_DRIVER_ENTRY_and_inactive_default_routes_correct=True,actual_H058_phase_parent_32TRAIN_cache32GiB_workers2_checkpoint12x1_5MiB_caps=True,producer_binding_caps_dependencies_byte_identical=True,v1_rejection_and_producer_source_PASS_preserved=True,actual_scientific_review_accepts_existing_standalone_producer_review=True,twelve_actual_writer_name_paths_handwritten_serialization=True,native_nested_JSON_and_distinct_ROOT_completed_products=True,H058_serialization_cap_and_duplicate_fail_before_open=True,inmemory_products=len(files),actual_optimizer_updates=0,actual_source_metadata_entry_imports_only=True,real_Torch_Numpy_scientific_model_payload_guard_mutex_admission_resource_process_probe_jobs_install_Git=False,limits='Inactive stdlib metadata entry imports and handwritten I/O/serialization seams only. No numerical model work or real guards/lifecycle. Full workload, actual binary serialization/runtime/resource fit/current direct role union and successful future admission/wait/drain remain unproved.')
if __name__=='__main__':
 value=main();(ROOT/REVIEW/'review_checks.v2.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');print(json.dumps(value,sort_keys=True,indent=2))
