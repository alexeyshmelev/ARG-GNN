from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,sys,ast,copy,importlib.metadata,traceback
sys.excepthook=lambda t,v,b:traceback.print_exception(t,v,b,file=sys.stdout)
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
assert len(sys.argv)==2 and sys.argv[1] in ['21','22']
chrom=sys.argv[1]
B=ROOT/f'research/reviews/H056_chr{chrom}_bank_actual_root_20261016_v1'
assert not B.exists()
def ref(p):
 p=Path(p);return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def norm(r):return {k:r[k] for k in ['path','sha256']}
def save(n,v):
 p=B/n
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 return ref(p)
def unique(rs):
 d={}
 for r in rs:
  r=norm(r);assert Path(r['path']).suffix in ['.py','.sh','.json']
  if r['path'] in d:assert d[r['path']]==r
  d[r['path']]=r
 return [d[k] for k in sorted(d)]
D=ROOT/'research/reviews/H056_bank_application_runtime_source_20261016_v1/runtime_v2.py'
assert ref(D)['sha256']=='647e7ce205a19b7cf5847faf45b70e18825e6d28fdc7e00907157dd3a0b0436d'
tree=ast.parse(D.read_text())
fix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FIXED_REFS' for t in n.targets))
names={'need','ceil_mib','valid_counts','preparation_caps_v1'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==4
ns={'AUX':524288};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(D)+'::pure_caps','exec'),ns)
native_dir=ROOT/f'experiments/results/H056_chr{chrom}_H36_Relate_pre_size_CPU_v1'
native=json.loads((native_dir/'completed.json').read_text())
contract=json.loads((native_dir/'runtime/graph_contract.json').read_text())
wait_ref=ref(ROOT/f'research/reviews/H056_chr{chrom}_native_actual_root_20261016_v1/actual_native_tool_wait.root_sealed_v1.json')
wait=json.loads((ROOT/wait_ref['path']).read_text())
assert native['chromosome']==chrom and native['graph_only_ready'] and native['counts']==contract['counts']
assert wait['actual_tool_terminal_consumed'] and wait['actual_tool_exit_code']==0 and wait['actual_owned_group_absent']
assert wait['native_completion_ref']==ref(native_dir/'completed.json') and wait['native_root_input_ref']==native['root_input_ref']
caps=ns['preparation_caps_v1'](native['counts'])
templates=json.loads((ROOT/'research/reviews/H056_native_application_runtime_source_20261016_v1/binding_templates.inactive.v1.json').read_text())
Acq=ROOT/f'experiments/results/H056_chr{chrom}_H36_phased_acquisition_CPU_v1'
acq=json.loads((Acq/'completed.json').read_text())
roster=ref(ROOT/'research/reviews/H056_exact_prefix_asset_qualification_root_20261016_v1/planned_H36_roster.root_frozen.json')
obs={'original_VCF':norm(acq['artifacts'][f'H056.chr{chrom}.noAP.vcf.gz']),'bridge_haps':norm(acq['artifacts']['baseline.haps']),'bridge_sample':norm(acq['artifacts']['baseline.sample']),'bridge_ages':norm(acq['artifacts']['sample_ages.generations']),'frozen_sites':norm(acq['artifacts']['frozen_site_identities.json']),'roster':roster,'serialized_readback':contract['serialized_readback_ref'],'native_graph':native['native_graph_ref'],'native_mapping':native['native_mapping_ref'],'graph_contract':native['graph_contract_ref']}
science=copy.deepcopy(templates['reader_manifest_template']);science['observation_refs']=obs
for p,h in science['source_sha256'].items():assert ref(ROOT/p)['sha256']==h
B.mkdir()
sm=save('scientific_manifest.json',science)
cfg=copy.deepcopy(templates['reader_config_template'])
cfg.update(chromosome=chrom,expected_counts=native['counts'],genuine_regional_bp0=contract['coverage']['genuine_regional_bp0'],observation_refs=obs,source_manifest_ref=sm,status='root_admitted_observation_preparation',preparation_authorized=True,max_host_RSS_bytes=2000000000,wall_deadline_seconds=1800)
for k in ['input_read_cap_bytes','VCF_decompressed_cap_bytes','CPU_preparation_cap_bytes','CPU_tensor_cap_bytes','flag_cap_bytes','output_cap_bytes']:cfg[k]=caps[k]
cr=save('config.json',cfg)
now=datetime.now(timezone.utc);end=now+timedelta(seconds=1800)
out=f'experiments/results/H056_chr{chrom}_H36_original_GT_bank_CPU_v1';aux=out+'_aux';job=f'H056-bank-chr{chrom}-original-GT-v1'
assert not (ROOT/out).exists() and not (ROOT/aux).exists()
runtime={'prefix':str(ROOT/'.runtime/env'),'executable':str(Path(sys.executable).resolve()),'python':sys.version,'CUDA_VISIBLE_DEVICES':'','CPU_threads':4,'created_at_utc':now.isoformat(),'expires_at_utc':end.isoformat(),'packages':{k:importlib.metadata.version(k) for k in ['numpy','torch','torch_geometric','tskit']}}
rr=save('runtime.json',runtime)
original=json.loads((ROOT/f'research/reviews/H056_chr{chrom}_native_actual_root_20261016_v1/source_manifest.json').read_text())['original_bindings']
resources=unique([*original.values(),*fix.values()]);sources=dict(science['source_sha256'])
for r in [*resources,ref(D)]:
 if Path(r['path']).suffix=='.py':sources[r['path']]=r['sha256']
meta=unique([templates['protocol_ref'],cr,sm,rr,ref(native_dir/'completed.json'),wait_ref,roster,obs['serialized_readback'],obs['graph_contract'],obs['native_mapping']])
io={'schema':'H056_specific_CPU_IO_closure_v1','operation':'prepare','protocol_ref':templates['protocol_ref'],'scientific_manifest_ref':sm,'source_sha256':sources,'required_metadata_refs':meta,'original_bindings':original}
ir=save('io_source_manifest.json',io)
producer=ref(ROOT/'research/reviews/H056_H36_implementation_20261016_v1/reader_h36_v1.py')
bindings=unique([ref(D),producer,cr,sm,rr,ir,templates['protocol_ref'],*resources,*[dict(path=p,sha256=h) for p,h in sources.items()],*meta]);assert len(bindings)==59
for r in bindings:assert ref(ROOT/r['path'])==r
inp=copy.deepcopy(templates['preparation_root_input_template'])
inp.update(status='inactive_pending_actual_admission',operation='prepare',root_authorizer='root_agent',registered=True,execution_authorized=False,GPU_authorized=False,optimization_authorized=False,root_host_resource_enforcement_active=True,preparation_authorized=True,producer_ref=producer,config_ref=cr,runtime_ref=rr,storage_ref=None,storage_adapter_ref=fix['guard'],io_context_ref=ref(D),io_source_manifest_ref=ir,source_manifest_ref=sm,resource_binding_refs=resources,job_id=job,output_directory=out,auxiliary_directory=aux,expires_at_utc=end.isoformat(),native_completion_ref=ref(native_dir/'completed.json'),native_wait_ref=wait_ref,caps=caps)
input_ref=save('input.inactive.json',inp)
request={'key':f'H056/bank-chr{chrom}-original-GT-v1','output':str(ROOT/out),'auxiliary_scopes':[str(ROOT/aux)],'reserved_bytes':caps['output_cap_bytes'],'consumer_bindings':bindings}
req=save('reservation_request.root_v1.json',request)
result={'schema':'H056_bank_actual_binding_construction_v1','input_inactive_ref':input_ref,'actual_counts':native['counts'],'caps':caps,'direct_bindings':59,'resource_bindings':len(resources),'config_ref':cr,'runtime_ref':rr,'scientific_manifest_ref':sm,'io_manifest_ref':ir,'reservation_request_ref':req,'execution_authorized':False,'native_actual_outer_wait0_ref':wait_ref,'original_GT_only_features':True,'no_checkpoint_or_model_read':True}
save('binding_construction.actual_v1.json',result)
admit=(ROOT/'research/reviews/H056_chr20_bank_actual_root_20261016_v1/admit.root_v1.py').read_text()
admit=admit.replace('H056_chr20_bank_actual_root_20261016_v1',f'H056_chr{chrom}_bank_actual_root_20261016_v1')
assert admit.count('==14680064')==1
admit=admit.replace('==14680064','=='+str(caps['output_cap_bytes']))
with (B/'admit.root_v1.py').open('x') as f:f.write(admit)
print(json.dumps({'chromosome':chrom,'counts':native['counts'],'cap_bytes':caps['output_cap_bytes'],'direct_bindings':59,'input_inactive_ref':input_ref}),flush=True)
