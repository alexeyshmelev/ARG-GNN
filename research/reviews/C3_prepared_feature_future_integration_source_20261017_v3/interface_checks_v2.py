"""Exact source AST interface checks with RAM-only I/O and scientific stubs.
No operational imports, real guard calls, data arrays, models or GPU operations.
"""
from pathlib import Path as DiskPath
import ast,copy,hashlib,io,json,os,types
import numpy as np
R=DiskPath('/workspace-SR003.nfs2/ARG-GNN')
D=R/'research/reviews/C3_prepared_feature_future_integration_source_20261017_v3'
O=R/'research/reviews/C3_prepared_feature_future_integration_source_20261017_v1'
V5=R/'research/reviews/C3_remaining_cells_GPU0_source_20261017_v5'
positive=[];negative=[]
def require(ok,label):
    if not ok:raise ValueError(label)
def check(ok,label):
    if not ok:raise AssertionError(label)
    positive.append(label)
def rejected(fn,label,contains=None):
    try:fn()
    except (ValueError,KeyError,AssertionError,TypeError,FileExistsError)as error:
        if contains is not None:require(contains in str(error),'Expected exact rejection '+label)
        negative.append(label);return
    raise AssertionError('Accepted '+label)
def source_ref(path):
    p=DiskPath(path);p=p if p.is_absolute()else R/p
    return dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def node(path,name):
    text=path.read_text()
    return next(n for n in ast.walk(ast.parse(text))if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
def install(n,env):
    exec(compile(ast.Module(body=[copy.deepcopy(n)],type_ignores=[]),'<exact_RAM_'+n.name+'>','exec'),env)
pins=json.loads((D/'pins.v1.json').read_text())
review=json.loads((R/pins['feature_review']['path']).read_text())
component_manifest=json.loads((R/pins['feature_manifest']['path']).read_text())
readmap={json.dumps(pins['feature_review'],sort_keys=True):review,
         json.dumps(pins['feature_manifest'],sort_keys=True):component_manifest}
def metadata_read(ref):return copy.deepcopy(readmap[json.dumps(ref,sort_keys=True)])
env=dict(require=require,pins_v1=lambda:copy.deepcopy(pins),read=metadata_read)
install(node(D/'contract_v1.py','accepted_feature_component_values_v2'),env)
def join():return env['accepted_feature_component_values_v2']()
check(join()==(review,component_manifest),'Genuine immutable independent v3 candidate and manifest metadata join')
check(source_ref(pins['feature_component']['path'])==pins['feature_component']
      and source_ref(pins['feature_manifest']['path'])==pins['feature_manifest']
      and source_ref(pins['feature_review']['path'])==pins['feature_review'],
      'Exact component, manifest and independent review file hashes')
rk=json.dumps(pins['feature_review'],sort_keys=True)
mk=json.dumps(pins['feature_manifest'],sort_keys=True)
for field,value in [('schema','wrong_schema'),('independent',False),('verdict','PASS'),
                    ('blockers',['block']),('candidate_ref',dict(path='RAM/wrong',sha256='0'*64)),
                    ('candidate_manifest_ref',dict(path='RAM/wrong_manifest',sha256='0'*64))]:
    before=copy.deepcopy(readmap[rk]);readmap[rk][field]=value
    rejected(join,'Genuine review rejects '+field)
    readmap[rk]=before
for alias in ['source_ref','component_ref']:
    before=copy.deepcopy(readmap[rk]);readmap[rk].pop('candidate_ref');readmap[rk][alias]=pins['feature_component']
    rejected(join,'Legacy '+alias+' alias cannot rescue missing genuine candidate_ref')
    readmap[rk]=before
for field,value in [('schema','wrong_manifest_schema'),('component_ref',dict(path='RAM/wrong_component',sha256='0'*64))]:
    before=copy.deepcopy(readmap[mk]);readmap[mk][field]=value
    rejected(join,'Component manifest rejects '+field)
    readmap[mk]=before
validate=node(D/'contract_v1.py','validate_request_v1')
check(any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='accepted_feature_component_values_v2'for n in ast.walk(validate)),
      'Actual request validator consumes exact tested review helper')
# RAM filesystem and descriptor interface. The exact common JSON serializer runs.
fs={};opened=[];syncs=[];boundaries=[];fdnext=[100];descriptors={}
class MemoryStream(io.BytesIO):
    def __init__(self,key,initial=b''):
        super().__init__(initial);self.key=key
    def close(self):
        if not self.closed:fs[self.key]=self.getvalue()
        super().close()
    def fileno(self):return 999
class RamPath:
    def __init__(self,value):
        self.key=value.key if isinstance(value,RamPath)else str(value).strip('/')
    def __truediv__(self,other):return RamPath(self.key+'/'+str(other))
    @property
    def name(self):return self.key.rsplit('/',1)[-1]
    def __str__(self):return self.key
    def exists(self):return self.key in fs
    def stat(self):return types.SimpleNamespace(st_size=len(fs[self.key]))
    def relative_to(self,root):
        root=str(root).strip('/')
        require(not root or self.key.startswith(root+'/'),'RAM root containment')
        return RamPath(self.key[len(root)+1:]if root else self.key)
    def open(self,mode):
        require(mode=='xb','Only exact exclusive NPZ RAM mode')
        if self.exists():raise FileExistsError(self.key)
        fs[self.key]=b'';opened.append(dict(path=self.key,mode=mode))
        return MemoryStream(self.key)
class RamOS:
    O_WRONLY=os.O_WRONLY;O_CREAT=os.O_CREAT;O_EXCL=os.O_EXCL;O_NOFOLLOW=os.O_NOFOLLOW
    @staticmethod
    def open(path,flags,mode):
        key=str(path);require(flags==os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,'Exact original exclusive JSON flags')
        require(mode==0o600,'Exact original private JSON mode')
        if key in fs:raise FileExistsError(key)
        fs[key]=b'';fdnext[0]+=1;descriptors[fdnext[0]]=key
        opened.append(dict(path=key,flags=flags,mode=mode))
        return fdnext[0]
    @staticmethod
    def fdopen(fd,mode):
        require(mode=='wb','Exact JSON descriptor binary mode')
        return MemoryStream(descriptors.pop(fd))
    @staticmethod
    def fsync(fd):syncs.append(fd)
def ram_ref(path):
    key=str(path)
    return dict(path=key,sha256=hashlib.sha256(fs[key]).hexdigest())
common='research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/common_v1.py'
serializer_env=dict(require=require,Path=RamPath,json=json,os=RamOS,ROOT=RamPath(''),ref=ram_ref)
install(node(R/common,'save'),serializer_env)
exact_json_save=serializer_env['save']
fixture_input=dict(path='RAM/current_input.json',sha256='f'*64)
class C:
    MANIFEST=str((D/'source_manifest.v1.json').relative_to(R))
    require=staticmethod(require)
    local=staticmethod(RamPath)
    pins_v1=staticmethod(lambda:copy.deepcopy(pins))
    save=staticmethod(exact_json_save)
    current_amendment_ref_v1=staticmethod(lambda:copy.deepcopy(fixture_input))
    ref=staticmethod(source_ref)
    @staticmethod
    def active_amendment_v1():boundaries.append(('active',))
    @staticmethod
    def charged_census_v1(spec,*,required_write_bytes):boundaries.append(('charged',required_write_bytes))
class Budget:
    @staticmethod
    def budget_check_v9(storage,*,boundary,required_write_bytes):
        require(boundary is True,'Original boundary flag');boundaries.append(('budget',required_write_bytes))
g=Budget()
def make_writer(path,source):
    wenv=dict(c=C,Path=RamPath,json=json,stable_reference_v1=ram_ref)
    install(node(source,'GuardedWriter'),wenv)
    return wenv['GuardedWriter'](dict(output_directory=path),{},g,dict(RAM=True))
shared='src/arggnn/american_c3_gnn_shared30mb_isolated_v2.py'
predictor='src/arggnn/american_c3_gnn_predict_isolated_v2.py'
CLASSES=('EUR','EAS','AFR');VAL=tuple('american_c3_val_%03d'%i for i in range(5))
class DummyArchitecture:
    def __init__(self):
        self.class_order=CLASSES;self.hidden_dim=32;self.expected_haplotypes=1600
        self.donor_cohorts=2;self.donor_channels=6;self.site_chunk_size=256;self.checkpoint_sites=True
        self.site_mlp=[types.SimpleNamespace(in_features=15)]
        self.leaf_projection=[types.SimpleNamespace(in_features=37)]
        self.forward_gru=self.reverse_gru=types.SimpleNamespace(input_size=35)
        self.classifier=[types.SimpleNamespace(out_features=3)]
    def __call__(self,*args,**kwargs):raise AssertionError('No model execution allowed')
training=dict(trained_parent_ids=['RAM/train_%03d'%i for i in range(20)],train_sequence_lengths_bp=[2000000]*20,protocol_sha256='p'*64)
metadata=dict(epoch=30,global_step=600)
checkpoint_ref=dict(path='RAM/checkpoint',sha256='c'*64)
training_ref=dict(path='RAM/completion',sha256='d'*64)
ram_arrays={'probabilities':np.array([[0.2,0.3,0.5],[0.6,0.1,0.3]],dtype=np.float32),
            'query_index':np.array([0,1],dtype=np.int64)}
trace=[];drains=[]
def stream(*args,**kwargs):
    trace.append('fixture_stream')
    return {k:v.copy()for k,v in ram_arrays.items()},dict(training_metadata={'RAM_only':True},fixture_interface_only=True)
def eligible(cp,tr):
    require(cp==checkpoint_ref and tr==training_ref,'Exact RAM checkpoint fixtures')
    return copy.deepcopy(training),copy.deepcopy(metadata)
def shutdown(loader):drains.append(loader.dataset.direction);return dict(drained=True)
def noop(*args,**kwargs):return None
senv=dict(require=require,VAL=VAL,TEST=(),CLASSES=CLASSES,
          AmericanC3WholeTreeGNNV1=DummyArchitecture,DEVELOPMENT_BRANCH='RAM_disclosed_branch',LENGTH=30000000,
          eligible_checkpoint_v1=eligible,validate_shared_plan_v1=noop,validate_shared_metadata_v1=noop,
          require_baseline_v1=noop,validate_plan_v1=noop,streamed_focal_predictions=stream,
          registry=lambda ref:['RAM/focal0','RAM/focal1'],verify_prediction=noop,identity_sha=lambda records:'i'*64,
          shutdown_dataloader=shutdown)
install(node(R/predictor,'seal_prediction_v1'),senv)
install(node(R/shared,'seal_shared_val_prediction_v1'),senv)
parent=VAL[1];plan=dict(parent_id=parent,config={},focal_registry_ref=dict(path='RAM/focals',sha256='1'*64))
def loaders():
    return tuple(types.SimpleNamespace(_iterator=None,dataset=types.SimpleNamespace(plan_ref={'path':'RAM/plan','sha256':'2'*64},
          direction=d,plan=copy.deepcopy(plan)))for d in ['forward','reverse'])
def run(source,tag):
    writer=make_writer('RAM/'+tag,source);fwd,rev=loaders()
    result=senv['seal_shared_val_prediction_v1'](DummyArchitecture(),fwd,rev,parent_id=parent,
          checkpoint_ref=checkpoint_ref,training_completion_ref=training_ref,writer=writer,device='RAM',
          runtime_guard=lambda event:trace.append(event))
    return result,writer
result,writer=run(D/'worker_v1.py','v2_chain')
seal=json.loads(fs[result['path']])
npzpath=seal['prediction_ref']['path']
with np.load(io.BytesIO(fs[npzpath]),allow_pickle=False)as payload:
    check(set(payload.files)==set(ram_arrays)and all(np.array_equal(payload[k],v)for k,v in ram_arrays.items()),
          'Exact shared wrapper → predictor → actual GuardedWriter NPZ serializes tiny RAM fixture unchanged')
check(seal['fixture_interface_only']is True and seal['status']=='sealed_C3_original_heldout_predictions'
      and 'training_metadata'not in seal,'Exact predictor ordinary JSON status and tensor metadata removal preserved')
check(seal['training_2Mb_to_30Mb_length_transfer']is True and seal['original_full30Mb_matched_training_claimed']is False
      and seal['same_observation_and_native_focals']is True
      and seal['complete_original_5VAL_selection_required_before_TEST']is True,
      'Exact shared wrapper explicit JSON disclosures survive actual backend')
check(seal['actual_execution_source_manifest_ref']==source_ref(C.MANIFEST)
      and seal['actual_ROOT_phase_input_ref']==fixture_input
      and seal['CPU_preprocessing_component_ref']==pins['feature_component']
      and seal['bitwise_original_feature_equivalence_claimed']is False,
      'Actual GuardedWriter retains source, phase input and preprocessing disclosures')
check(drains==['forward','reverse'],'Exact predictor drains both RAM loaders on success')
check(len(syncs)==1 and len(opened)==2 and sum('flags'in row for row in opened)==1,
      'Actual original common JSON serializer runs exact exclusive no-follow mode, flush and fsync on RAM backend')
check(all(boundaries[i][0]in ('active','charged','budget')for i in range(len(boundaries)))
      and any(row[0]=='charged'and row[1]>8192 for row in boundaries),
      'Actual GuardedWriter executes inherited boundary callbacks using RAM-only guard boundaries')
check(seal['records']==10000 and len(ram_arrays['probabilities'])==2,
      'Two-row fixture is explicitly interface-only, never a scientific support validation')
# These negative controls must fail at the original explicit JSON dispatch after NPZ serialization.
for path,tag in [(O/'worker_v1.py','immutable_v1_chain'),(V5/'worker_v1.py','immutable_v5_chain')]:
    before=len(drains)
    rejected(lambda path=path,tag=tag:run(path,tag),tag+' rejects wrapper explicit JSON',contains='Only actual small JSON/NPZ writes')
    check(len(drains)==before+2,'Exact predictor drains both RAM loaders after '+tag+' interface rejection')
    check(len([p for p in fs if p.startswith('RAM/'+tag+'/')and p.endswith('.npz')])==1
          and not any(p.startswith('RAM/'+tag+'/')and p.endswith('.json')for p in fs),
          tag+' stores NPZ then rejects JSON without invented successful seal')
# Direct serializer controls preserve existing limits and exclusivity.
a=make_writer('RAM/direct_json',D/'worker_v1.py')
for kind,name in [(None,'default.json'),('json','explicit.json')]:
    rr=a.save(name,dict(visible='same payload'),kind=kind)
    check(json.loads(fs[rr['path']])['visible']=='same payload','Bounded JSON accepts '+repr(kind)+' via unchanged serializer')
rejected(lambda:a.save('explicit.json',{}), 'Existing scientific output remains exclusive')
rejected(lambda:a.save('../escape.json',{}),'Flat scientific names remain mandatory')
rejected(lambda:a.save('nan.json',{'bad':float('nan')},kind='json'),'Nonfinite JSON remains rejected')
rejected(lambda:a.save('too_big.json',{'x':'x'*(8*2**20)},kind='json'),'JSON8MiB bound remains enforced')
rejected(lambda:a.save('unknown.json',{},kind='yaml'),'Unknown serializer kind remains rejected')
rejected(lambda:a.save('objects.npz',{'x':np.array([object()],dtype=object)},kind='npz'),'NPZ object arrays remain rejected')
rejected(lambda:exact_json_save(RamPath('RAM/direct_json'),'serializer_large.json',{'x':'x'*(8*2**20)}),
         'Exact common serializer independently preserves8MiB bound')
rejected(lambda:exact_json_save(RamPath('RAM/direct_json'),'explicit.json',{}),
         'Exact common serializer independently preserves O_EXCL')
result=dict(schema='C3_prepared_integration_v2_exact_RAM_interface_checks_v1',
    status='AUTHOR_INTERFACE_PASS_PENDING_FRESH_INDEPENDENT_REVIEW',
    positive=len(positive),negative=len(negative),checks=positive,negative_cases=negative,
    exact_AST_sources=[source_ref(R/shared),source_ref(R/predictor),source_ref(R/common),
                       source_ref(D/'worker_v1.py'),source_ref(D/'contract_v1.py')],
    exact_original_JSON_serializer_executed=True,actual_GuardedWriter_executed=True,
    operational_module_imports=0,guard_imports=0,real_guard_calls=0,disk_scientific_outputs=0,
    original_payload_reads=0,truth_reads=0,model_calls=0,GPU_operations=0,Git_actions=0,
    tiny_RAM_NPZ_fixture_only=True,scientific_support_validation=False,
    unresolved_v5_success_dependency=True,whole_route_accepted=False)
(D/'interface_checks.result.v2.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],positive=len(positive),negative=len(negative))))
