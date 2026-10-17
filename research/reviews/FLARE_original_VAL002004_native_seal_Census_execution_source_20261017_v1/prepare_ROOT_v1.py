"""Inactive ROOT metadata-only preparation for one original native/seal parent and stage.
No guard, resource, reserve, admission or execution call. Import is inert.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,copy,hashlib,importlib,json,os,stat,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_original_VAL002004_native_seal_Census_execution_source_20261017_v1"
SOURCE=HERE+"/prepare_ROOT_v1.py"
_ALLOWED=set()
def require(ok,message):
 if not ok:raise ValueError(message)
def local(name):
 require(type(name)is str and not Path(name).is_absolute()and '..'not in Path(name).parts,'Contained relative ROOT path')
 p=ROOT/name;require(p.resolve()==p and ROOT in p.parents,'Canonical contained ROOT metadata/source path');return p
def bounded_bytes(record):
 require(type(record)is dict and set(record)=={'path','sha256'},'Exact bounded source/JSON ref')
 p=local(record['path']);before=p.lstat()
 require(p.suffix in('.json','.py','.md','.txt','.patch','.sh')and stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=2**20,'Only compact singly linked source or metadata; no scientific payload')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb')as f:opened=os.fstat(f.fileno());raw=f.read(2**20+1);after=os.fstat(f.fileno())
 named=p.lstat();identity=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 require(identity(before)==identity(opened)==identity(after)==identity(named)and len(raw)==before.st_size and hashlib.sha256(raw).hexdigest()==record['sha256'],'Exact stable bounded source/JSON bytes');return raw
def reference(name):
 p=local(name);r=dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest());bounded_bytes(r);return r
def read(record):
 require(Path(record['path']).suffix=='.json','Only bounded JSON metadata');return json.loads(bounded_bytes(record))
def put(path,value):
 require(path.parent in _ALLOWED,'Only one fresh authorized preparation directory may receive metadata')
 local(str(path.relative_to(ROOT)));data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();require(len(data)<=2**20,'Compact metadata only')
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 return reference(str(path.relative_to(ROOT)))
def metadata_core_v1(c,read_metadata):
 """Run only original source-pinned metadata predicates with a RAM-capable reader."""
 raw=c.raw(c.COMPLETION);tree=ast.parse(raw);p=ROOT/c.COMPLETION['path']
 core=types.ModuleType('_FLARE_original_metadata_only_core');scope=core.__dict__
 scope.update(ROOT=ROOT,HERE=p.parent,copy=copy,json=json,require=c.require,_tools=lambda:(read_metadata,lambda *a:(_ for _ in()).throw(ValueError('Metadata preparer cannot open payloads')),c.ref))
 for name in('OPTIONS','COMMIT','CLASSES'):
  matches=[n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets)]
  require(len(matches)==1,'Exact original metadata constant');scope[name]=ast.literal_eval(matches[0].value)
 scope['MISSING']=tuple('american_c3_val_%03d'%i for i in range(1,5))
 names=('_local','_local_ref','_sources','_context','_gzip_bound','_conversion_bound','_native_output_bound','inactive_stage_configs_v1')
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names];require(len(nodes)==len(names),'Only original context/config metadata functions')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p)+'#metadata-only','exec',dont_inherit=True,optimize=0),scope)
 return core

def construct_metadata_v1(instruction_ref,*,c,b,put_metadata,own_ref):
 v=c.validate_stage_instruction_v1(instruction_ref)
 require(v['preparation_source_ref']==own_ref,'Exact separately reviewed one-stage preparer source')
 parent=v['parent_id'];stage=v['stage'];prep=ROOT/v['preparation_directory']
 row=c.completion()._context(v['context_ref'],parent_id=parent)['targets'][parent]
 key='H041/FLARE-original-'+stage+'-'+Path(v['packet_directory']).name
 from .root_driver_v1 import phase_budget
 budget=phase_budget(stage)
 stamp=dict(schema='FLARE_original_parent_native_seal_source_registration_v1',registered_before_request=True,registered_utc=v['created_at_utc'],
  ROOT_metadata_preparation_instruction_ref=instruction_ref,source_manifest_ref=c.ref(c.MANIFEST),source_review_ref=v['source_review_ref'],
  parent_id=parent,stage=stage,phase_budget=budget,proposed_reservation_key=key)
 registration=dict(schema='FLARE_root_target_registration_v1',root_authorizer='root_agent',parent_id=parent,
  observation_ref=row['observation_ref'],focal_registry_ref=row['focal_registry_ref'],count_manifest_ref=row['count_manifest_ref'],
  scientific_role='registered_heldout_full30Mb_comparison',query_ancestry_truth_opened=False,score_authorized=False,
  hypothesis_id='H041',missing_original_cell=True,VAL000_rerun=False,one_stage_source_registration=stamp)
 registration_ref=put_metadata(prep/(parent+'.FLARE_registration.root_v1.json'),registration)
 runtime=dict(prefix=str(ROOT/'.runtime/env'),CPU_threads=4,cpu_ids=copy.deepcopy(v['cpu_ids']),CUDA_VISIBLE_DEVICES='',package_report_ref=c.BASE_RUNTIME)
 runtime_ref=put_metadata(prep/'runtime.root_active_v1.json',runtime)
 keys=('stage','parent_id','conversion_ref','native_run_ref','previous_stage_evidence_ref','context_ref','preparation_directory','packet_directory','output_directory','cpu_ids','created_at_utc','expires_at_utc','storage_policy_selector_ref','source_review_ref')
 q=dict(schema='FLARE_remaining_CPU_ROOT_request_v1',active=True,root_authorizer='root_agent',execution_authorized=True,
  phase=stage+'_'+parent,GPU_authorized=False,CUDA_authorized=False,physical_gpu=None,remaining_VAL_truth_authorized=False,TEST_authorized=False,
  automatic_retry_resume_or_fallback=False,VAL000_rerun=False,source_manifest_ref=c.ref(c.MANIFEST),
  registration_ref=registration_ref,runtime_ref=runtime_ref,phase_budget=budget,ROOT_metadata_preparation_instruction_ref=instruction_ref,
  historical_VAL001_failure_acceptance_ref=b['existing_VAL001_failure_acceptance_ref'],historical_failure_parent_id='american_c3_val_001',
  proposed_reservation_key=key,reuse_partial_outputs=False,efficiency_improvement_claimed=False,paper_runtime_result_claimed=False,
  **{k:copy.deepcopy(v[k])for k in keys})
 cfg=c.make_config(q);caps=c.stage_caps(stage,cfg);q['caps']=caps
 require(cfg['external_truth_ref']is None and cfg['max_sites_bound']==row['input_sites']and (stage=='seal'or cfg['native_options']==c.completion().OPTIONS)
  and caps['scientific_seconds']==budget['scientific_seconds'],'Exact original full support/native science, no truth')
 cfg_ref=put_metadata(prep/('expected_'+stage+'_config.review_only.root_v1.json'),cfg)
 directive=dict(q,schema='FLARE_remaining_CPU_ROOT_directive_v1',directive='Execute one exact missing original FLARE stage after genuine fresh admission0')
 q['root_directive_ref']=put_metadata(prep/'ROOT_directive.root_active_v1.json',directive)
 request_ref=put_metadata(prep/'ROOT_request.root_active_v1.json',q)
 return dict(status='PREPARED_METADATA_ONLY_NOT_ADMITTED',parent_id=parent,stage=stage,request_ref=request_ref,
  expected_config_review_only_ref=cfg_ref,context_ref=v['context_ref'],registration_ref=registration_ref,runtime_ref=runtime_ref,
  output_cap_bytes=caps['output_bytes'],whole_scope_reservation_bytes=caps['reservation_bytes'],CPU_ids=copy.deepcopy(v['cpu_ids']),
  preparation_source_ref=own_ref,admission_or_launch_performed=False,current_fit_or_role_union_proven=False,Census_fallback=False,no_truth_read=True)

def prepare_v1(instruction_ref):
 global _ALLOWED
 require(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0,'Exact ROOT fresh -B -s optimize0 interpreter')
 instruction=read(instruction_ref);own_ref=reference(SOURCE)
 sys.path[:0]=[str(ROOT),str(ROOT/'src')]
 d=importlib.import_module(HERE.replace('/','.')+'.root_driver_v1');c=d.contract();b=d.bindings();c.sources()
 original_read=c.read;overlay={}
 def metadata_read(r):return copy.deepcopy(overlay[(r['path'],r['sha256'])])if(r['path'],r['sha256'])in overlay else original_read(r)
 c.read=metadata_read;c.completion=lambda:metadata_core_v1(c,metadata_read)
 def ram_put(path,value):
  raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();r=dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest());overlay[(r['path'],r['sha256'])]=copy.deepcopy(value);return r
 # Every source/science/selector/clock/CPU/name join runs in RAM before first write.
 construct_metadata_v1(instruction_ref,c=c,b=b,put_metadata=ram_put,own_ref=own_ref)
 _ALLOWED={local(instruction['preparation_directory'])}
 require(all(not local(instruction[k]).exists()for k in('preparation_directory','packet_directory','output_directory')),
  'One preparation and two reserved namespaces fresh; no replay')
 overlay.clear()
 for prep in _ALLOWED:prep.mkdir(mode=0o700)
 result=construct_metadata_v1(instruction_ref,c=c,b=b,put_metadata=put,own_ref=own_ref)
 c.validate_request(result['request_ref'])
 result['metadata_preparation_receipt_ref']=put(local(instruction['preparation_directory'])/'prepared_metadata_only.root_v1.json',
  dict(result,admission_or_launch_performed=False))
 return result

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ROOT-instruction-ref');a=p.parse_args(argv)
 if not a.ROOT_instruction_ref:print(json.dumps(dict(status='INACTIVE_SOURCE_ONLY',preparation=False,admission=False,launch=False)));return 0
 result=prepare_v1(json.loads(a.ROOT_instruction_ref));print(json.dumps(result,sort_keys=True),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
