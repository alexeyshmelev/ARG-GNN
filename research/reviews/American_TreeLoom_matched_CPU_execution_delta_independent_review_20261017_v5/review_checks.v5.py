"""Independent minimal serializer/source delta check. Stdlib and handwritten type seams only."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast,contextlib,copy,hashlib,importlib,io,json,os,stat,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/American_TreeLoom_matched_CPU_execution_source_20261017_v5'
BASE='research/reviews/American_TreeLoom_matched_CPU_execution_source_20261017_v4'
REVIEW='research/reviews/American_TreeLoom_matched_CPU_execution_delta_independent_review_20261017_v5'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
def need(ok,msg='independent check'):
 if not ok:raise AssertionError(msg)
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def node(text,name):return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name)
def main():
 need(sys.dont_write_bytecode and sys.flags.no_user_site)
 expected={'source_manifest.v1.json':'c90c715f4c44affa9f8c7335a89028a3e1e1fd322ce9d9092fd62a0ecdb561dc','source_handoff.v1.json':'f5d45fe5bdcdab3bcfe526e6d5515900e3724a058c6c383a726051a795e361cf','root_driver_v1.py':'a321fbe59409a929eda10e22deb2d9e466a75da26edd5fe4ac8354b293cf876f','worker_entry_v1.py':'5657f26ea1250d237c7213b5f8ef944d7ca731ea2d8fb7526ed8c011bbb91385'}
 for name,sha in expected.items():need(ref(HERE+'/'+name)['sha256']==sha,'candidate seal')
 manifest=json.loads((ROOT/HERE/'source_manifest.v1.json').read_text())
 for path,sha in manifest['source_sha256'].items():
  need(Path(path).suffix in ('.py','.sh','.json','.md','.patch'),'source/metadata only')
  need(ref(path)['sha256']==sha,'pinned source '+path)
 normalized=[]
 for name in ('root_contract_v1.py','root_driver_v1.py'):
  need((ROOT/HERE/name).read_text().replace(HERE,BASE).replace(HERE.replace('/','.'),BASE.replace('/','.'))==(ROOT/BASE/name).read_text(),'namespace-only '+name)
  normalized.append(name)
 same=('caps.inactive.v1.json','scientific_binding.v1.json','scientific_config.INACTIVE.template.v1.json','dependencies.v1.json','ROOT_selected_operational_failstops.v1.json','training_context.INACTIVE.template.v1.json')
 for name in same:need((ROOT/HERE/name).read_bytes()==(ROOT/BASE/name).read_bytes(),'unchanged '+name)
 actual=(ROOT/HERE/'worker_entry_v1.py').read_text()
 reduced=actual.replace(HERE.replace('/','.'),BASE.replace('/','.'))
 start=reduced.index('def jsonable_runtime_v1(');end=reduced.index('class CappedBuffer(',start)
 reduced=reduced[:start]+reduced[end:]
 reduced=reduced.replace(' writer=BoundedWriter(s,q,g,fast,torch,lambda value:jsonable_runtime_v1(value,torch,np))',' from arggnn.census_tsinfer_fit_v1 import _jsonable_v1\n writer=BoundedWriter(s,q,g,fast,torch,_jsonable_v1)')
 need(reduced==(ROOT/BASE/'worker_entry_v1.py').read_text(),'only declared worker repair and namespace')
 # Compare conversion algebra with frozen legacy source, without importing that module.
 legacy_text=(ROOT/'src/arggnn/census_tsinfer_fit_v1.py').read_text()
 old=node(legacy_text,'_jsonable_v1');new=copy.deepcopy(node(actual,'jsonable_runtime_v1'))
 class OldForm(ast.NodeTransformer):
  def visit_Call(self,n):
   self.generic_visit(n)
   if isinstance(n.func,ast.Name)and n.func.id=='jsonable_runtime_v1':n.func.id='_jsonable_v1';n.args=n.args[:1]
   return n
 new=OldForm().visit(new);new.name='_jsonable_v1';new.args.args=new.args.args[:1];new.body=new.body[1:]
 need(ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False),'same recursive conversion algebra')
 d,w=[importlib.import_module(HERE.replace('/','.')+'.'+name)for name in ('root_driver_v1','worker_entry_v1')];c=d.c
 need(w.c is c and c.HERE==HERE and c._h._GUARD is None and c.SELECTOR_REF is None)
 need(len(c.sources()['source_sha256'])==len(manifest['source_sha256']))
 b=c.binding();c.scientific_review(b['scientific_review_ref'])
 cfg=c.read(c.ref(HERE+'/scientific_config.INACTIVE.template.v1.json'));cfg.update(c.read(c.ref(HERE+'/ROOT_selected_operational_failstops.v1.json'))['values']);cfg['execution_enabled']=True;c.validate_config(cfg)
 for module in (d,w):
  with contextlib.redirect_stdout(io.StringIO()):need(module.main([])==0)
 need('from arggnn.census_tsinfer_fit_v1 import _jsonable_v1'not in actual)
 import_at=actual.index(' import numpy as np,torch,torch_geometric')
 need(actual.index('os.read(gate_fd,1)')<actual.index('g.validate_storage_v9')<import_at<actual.index(' writer=BoundedWriter('),'runtime imports after owned/admitted checks and before closure')
 # Handwritten stand-ins exercise only type conversion and stdlib JSON.
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
 torch=NS(Tensor=Tensor);np=NS(ndarray=Array,generic=Scalar)
 metadata={7:(Tensor([[1.,2.],[3.,4.]]),Array([Scalar(5),Scalar(6.)])), 'native':{'active':True,'none':None,'path':'metadata.json'},'nested':[{'scalar':Scalar(8)}]}
 expected_native={'7':[[[1.,2.],[3.,4.]],[5,6.]],'native':{'active':True,'none':None,'path':'metadata.json'},'nested':[{'scalar':8}]}
 need(w.jsonable_runtime_v1(metadata,torch,np)==expected_native)
 legacy_ns={'torch':None,'np':None}
 exec(compile(ast.Module(body=[old],type_ignores=[]),'frozen_legacy_helper','exec'),legacy_ns)
 legacy_failure=False
 try:legacy_ns['_jsonable_v1']({'native':'started'})
 except AttributeError as exc:legacy_failure='Tensor'in str(exc)
 need(legacy_failure,'reproduce exact lazy-global failure on native metadata')
 legacy_ns.update(torch=torch,np=np);need(legacy_ns['_jsonable_v1'](metadata)==expected_native)
 # The unchanged actual BoundedWriter.save receives the repaired closure. All I/O is in memory.
 files={};fds={};opens=[]
 class File(io.BytesIO):
  def __init__(self,name):super().__init__();self.name=name
  def fileno(self):return 42
  def __exit__(self,*args):files[self.name]=self.getvalue();return super().__exit__(*args)
 class P:
  def __init__(self,name=''):self.name=name
  def __truediv__(self,name):return P(name)
  def lstat(self):return NS(st_mode=stat.S_IFREG,st_nlink=1,st_dev=1,st_ino=2)
  def relative_to(self,root):return Path('SYNTHETIC_METADATA')/self.name
 def opened(p,*args):need(p.name not in files);fds[42]=p.name;opens.append(p.name);return 42
 fakeos=NS(O_WRONLY=os.O_WRONLY,O_CREAT=os.O_CREAT,O_EXCL=os.O_EXCL,O_NOFOLLOW=os.O_NOFOLLOW,
  statvfs=lambda p:NS(f_bavail=10**9,f_frsize=4096),open=opened,fdopen=lambda fd,mode:File(fds[fd]),fsync=lambda fd:None,fstat=lambda fd:NS(st_dev=1,st_ino=2))
 writer=object.__new__(w.BoundedWriter);writer.fast=lambda *a:None;writer.s={'storage':{}};writer.g=NS(budget_check_v9=lambda *a,**kw:None)
 writer.torch=torch;writer.jsonable=lambda value:w.jsonable_runtime_v1(value,torch,np);writer.path=P();writer.used=8192;writer.json_used=0;writer.torch_files=0;writer.names=set()
 old_os=w.os;w.os=fakeos
 try:
  writer.save('started.json',dict(status='synthetic_started',input_ref={'path':'synthetic/input.json','sha256':'a'*64},additional_updates_per_arm=600,total_additional_updates=1200,device='cpu',physical_GPU_lease=None))
  writer.save('TRAIN_observation_barrier.json',metadata)
  writer.save('completed_training.json',dict(epochs=30,additional_updates_per_arm=600,new_saved_logical_optimizer_updates_total=1200,trained_parent_ids=list(c.TRAIN),TEST_deferred=True))
  need(json.loads(files['TRAIN_observation_barrier.json'])==expected_native)
  count=len(opens)
  for value in ({'x':object()},{'x':Scalar(float('nan'))}):
   try:writer.save('worker_failure.json',value)
   except (TypeError,ValueError):pass
   else:raise AssertionError('must reject unrepresentable JSON')
   need(len(opens)==count,'serialize before file open')
 finally:w.os=old_os
 # Authenticate actual old failure metadata, original reservation and previous source review.
 lineage=c.read(c.ref(HERE+'/failure_lineage.v5.json'))
 values={key:c.read(lineage[key])for key in ('actual_v4_execution_outer_ref','actual_v4_wait_ref','actual_v4_log_ref','v4_reservation_preserved_ref')}
 outer=values['actual_v4_execution_outer_ref'];wait=values['actual_v4_wait_ref'];storage=values['v4_reservation_preserved_ref']
 need(outer['actual_terminal_consumed']is True and outer['actual_exit_code']==1 and outer['actual_tool_session_id']==47644 and outer['new_optimizer_updates']==0)
 need(wait['returncode']==1 and wait['all_owned_processes_and_groups_drained']is True and wait['completed_ref']is None)
 need("NoneType"in values['actual_v4_log_ref']['tail']and 'torch.Tensor'in values['actual_v4_log_ref']['tail'])
 need(storage['reserved_new_bytes']==32*2**20 and len(storage['reservation']['consumer_bindings'])==61 and storage['expires_at_utc']=='2026-10-17T23:36:50.841990+00:00')
 prior=c.read(c.ref('research/reviews/American_TreeLoom_matched_CPU_execution_delta_independent_review_20261017_v4/review.v1.json'))
 need(prior['verdict']=='SOURCE_PASS'and prior['blockers']==[])
 template=c.read(c.ref(HERE+'/ROOT_review.INACTIVE.template.v1.json'))
 stale=template['source_ref']!=c.ref(c.DRIVER)
 need(stale and template['independent']is False and template['verdict']=='UNREVIEWED_INACTIVE')
 need(not any(n in sys.modules for n in ('torch','numpy','torch_geometric','arggnn.census_tsinfer_fit_v1','scripts.project_budget_v9','arggnn.project_budget_v4')))
 need(c._h._GUARD is None and c.SELECTOR_REF is None)
 return dict(schema='H060_v5_independent_serializer_delta_check_v1',status='PASS_SOURCE_AND_HANDWRITTEN_METADATA_ONLY',source_map_items_authenticated=len(manifest['source_sha256']),
  contract_driver_namespace_normalized_byte_identical=normalized,byte_identical_binding_caps_context_files=list(same),worker_delta_exactly_namespace_and_explicit_runtime_serializer=True,
  recursive_conversion_algebra_AST_equal_to_frozen_legacy_helper=True,legacy_lazy_global_None_Tensor_failure_reproduced=True,explicit_runtime_nested_tensor_array_scalar_tuple_dict_conversion_passed=True,
  actual_BoundedWriter_native_started_and_nested_and_completed_JSON_passed=True,actual_inmemory_JSON_products=len(files),actual_inmemory_JSON_bytes=writer.json_used,unsupported_and_nonfinite_values_rejected_before_open=True,
  old_v4_outer_47644_exit1_direct_worker1_drain_zero_updates_authenticated=True,old_reservation_charge_expiry_and_61_consumer_bindings_preserved=True,
  inactive_review_template_stale_driver_SHA_nonblocking=True,template_remedy='Use exact newly authored independent v5 review; do not activate inactive template with stale v4 driver SHA.',
  real_Torch_Numpy_models_payload_guards_mutex_admission_resource_process_probes_jobs_Git_install=False,actual_optimizer_updates=0,
  limits='Source and handwritten type/metadata seams only. Current guard role count/capacity, actual serialization on installed numerical runtime, full workload fit, successful preparation/wait/drain/outer remain future ROOT evidence. No broad qualification matrix.')
if __name__=='__main__':
 result=main();(ROOT/REVIEW/'review_checks.v5.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result,sort_keys=True,indent=2))
