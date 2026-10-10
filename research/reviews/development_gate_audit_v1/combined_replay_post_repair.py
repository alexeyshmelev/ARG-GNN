"""Independent combined-gate replay. Hand-built facts never admit an actual cohort.
Uses actual current producer proposal assemblers/contracts/pins, mocked only
positive cohort admission. No active executable config or final plan is emitted.
"""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,json,sys,os
import torch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/development_gate_audit_v1'
TEMP=ROOT/'.runtime/tmp/development_gate_audit_v1/combined_replay_post';TEMP.mkdir(parents=True,exist_ok=False)
from arggnn import development_selection_bindings as b,development_training_bindings as common
from arggnn import argmix_development_bindings as raw,canonical_development_bindings as canonical,allrecord_development_bindings as allrecord,priorart_development_bindings as prior
from arggnn import train_anchor_development as candidate
from arggnn.development_selection import tensor_fingerprint
from arggnn.development_execution import DETERMINISTIC_POLICY
torch.set_num_threads(1)
checks={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');return {'path':str(path),'sha256':sha(path)}
def record(name,value):
    checks[name]=value;(OUT/'combined_replay_post_repair.json').write_text(json.dumps({'actual_cohort_admitted':False,'executable_config_emitted':False,'positive_admission_mocked':True,'checks':checks},indent=2,sort_keys=True)+'\n')
    print(name+': '+json.dumps(value,sort_keys=True),flush=True)
def reject(call):
    try:call()
    except Exception as error:return {'rejected':True,'reason':str(error),'type':type(error).__name__}
    return {'rejected':False}
freeze=save(TEMP/'freeze.json',{'kind':'fresh_development_preparation_execution_freeze','confirmatory_access':False,'not_confirmatory_test_freeze':True,'audit_fixture_synthetic':True})
raw_catalog={}
raw_records=[]
for split,count in (('train',64),('val',16)):
    for i in range(count):
        row={'split':split,'replicate':i}
        for field,hash_field in (('metadata','metadata_sha256'),('tensor_file','tensor_sha256'),('prepared_tensor_file','prepared_tensor_sha256')):
            p=TEMP/'synthetic_raw_bytes'/split/str(i)/(field+'.synthetic')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text('Synthetic byte fixture: '+split+' '+str(i)+' '+field+'\n')
            raw_catalog[str(p)]=sha(p);row[field]=str(p);row[hash_field]=sha(p)
        raw_records.append(row)
source=save(TEMP/'source.json',{'protocol':{'splits':{'train':64,'val':16,'test':0},'master_seed':20261101,'inference':{'master_seed':20261102}},
    'replicates':raw_records})
oracle=save(TEMP/'oracle.json',{'audit_fixture_synthetic':True})
bprep=hashlib.sha256(b'synthetic boundary preparation').hexdigest();aprep=hashlib.sha256(b'synthetic age preparation').hexdigest()
boundary=save(TEMP/'boundary.json',{'status':'complete','preparation_sha256':bprep,'source':{'manifest_sha256':source['sha256']}})
age=save(TEMP/'age.json',{'status':'complete','preparation_sha256':aprep,'source_manifest_sha256':source['sha256'],'boundary_manifest_sha256':boundary['sha256'],'boundary_preparation_sha256':bprep})
facts={'oracle':oracle,'inferred':{**source,'protocol_path':source['path'],'protocol_sha256':source['sha256']},
    'boundary':{**boundary,'preparation_sha256':bprep},'age':{**age,'preparation_sha256':aprep}}
completion={'all_stages_complete':True,'train_parents':64,'validation_parents':16,'test_parents':0,'inferred_boundary_and_age_tiles':640,
    'training_or_confirmatory_selection_frozen':False,'development_execution_freeze_file_sha256':freeze['sha256'],
    'development_execution_freeze_path':freeze['path'],'stage_manifests':facts,'audit_fixture_synthetic':True}
cp=save(TEMP/'completion.json',completion)
artifact=TEMP/'source_artifact.txt';artifact.write_text('Synthetic observed-only bytes.\n')
lock=common.SourceSnapshot({str(artifact):sha(artifact),**raw_catalog})
class ObservedCase:
    def __init__(self,i):
        self.parent_id=source['sha256']+':val:'+str(i)
        self.positions=torch.tensor([.25,.75],dtype=torch.float64)
        self.query_node_ids=torch.arange(64,dtype=torch.long)
    @property
    def labels(self):raise AssertionError('Audit binder accessed labels')
    @property
    def baseline_probabilities(self):raise AssertionError('Audit binder accessed scores')
cases=[ObservedCase(i) for i in reversed(range(16))]
dataset=SimpleNamespace(cases=lambda split:cases if split=='val' else (_ for _ in ()).throw(AssertionError('Unexpected split')))
settings=common.make_configs(completion,cp['path'],cp['sha256'])
for module,key in ((raw,'argmix_full_tree'),(canonical,'canonical_gru')):
    plan=json.loads((ROOT/module.PLAN).read_bytes());module.check_plan(plan);plan_path=ROOT/module.PLAN
    args=(plan,str(plan_path),sha(plan_path),completion,cp['path'],cp['sha256'])
    settings[key]=module.bound_settings(*args,raw_catalog) if module is canonical else module.bound_settings(*args)
allplanpath=ROOT/'configs/allrecord_development_search_plan_v2.json';allplan=json.loads(allplanpath.read_bytes())
settings.update(allrecord.bound_configs(allplan,allplanpath,sha(allplanpath),completion,cp['path'],cp['sha256']))
priorplanpath=ROOT/prior.PLAN;priorplan=json.loads(priorplanpath.read_bytes())
settings.update(prior.bound_settings(priorplan,priorplanpath,sha(priorplanpath),completion,cp['path'],cp['sha256']))
refs=[]
for family in b.FAMILIES:
    ref=save(TEMP/'synthetic_input_configs'/(family+'.json'),settings[family]);refs.append({'family_id':family,**ref})
amendment=ROOT/b.SCOPE_AMENDMENT;amendsha=sha(amendment)
def run(refs_arg=None):
    return b.bind_development_selection(cp['path'],cp['sha256'],expected_scope_amendment_sha256=amendsha,family_configs=refs if refs_arg is None else refs_arg)
# The same handmade facts explicitly fail real source admission before any data load.
record('synthetic_completion_is_non_executable',reject(lambda:common.verify_completion(cp['path'],cp['sha256'],with_snapshot=True)))
assert checks['synthetic_completion_is_non_executable']['rejected']
# No real completion is queried. This CLI-style missing path is inside audit temp.
record('missing_completion_real_gate',reject(lambda:b.bind_development_selection(TEMP/'missing_completion.json','a'*64,
    expected_scope_amendment_sha256=amendsha,family_configs=refs)))
record('incomplete_completion_real_gate',reject(lambda:b.bind_development_selection(
    save(TEMP/'incomplete_completion.json',{**completion,'all_stages_complete':False})['path'],
    sha(TEMP/'incomplete_completion.json'),expected_scope_amendment_sha256=amendsha,family_configs=refs)))
def admitted(path,digest,**kwargs):
    assert str(path)==cp['path'] and digest==cp['sha256'] and kwargs=={'with_snapshot':True}
    return deepcopy(completion),dataset,{},lock
with patch.object(common,'verify_completion',admitted):
    result=run()
    record('ten_real_producer_contracts_under_synthetic_admission',{'validated':True,'output_path':result.path,
        'families':list(result.plan['families']),'required_baselines':result.plan['required_baseline_families'],
        'dependency_keys':len(result.plan['implementation_artifacts']),'resolved_artifacts':len({x['path'] for x in result.plan['implementation_artifacts'].values()}),
        'configuration_seed_cells':result.receipt['configuration_seed_cells'],'optimizer_budget':result.receipt['maximum_prescribed_optimizer_starts'],
        'policy_fields':result.receipt['family_deterministic_policy_fields'],'scores_inspected':result.receipt['scores_inspected'],
        'selection_performed':result.receipt['selection_performed'],'final_test_access':result.receipt['final_test_access'],
        'authorizes_confirmatory_generation':result.receipt['authorizes_confirmatory_generation'],'plan_sha256':result.sha256})
    assert result.path is None and result.receipt['selection_performed'] is False
    # Freeze/rebind only synthetic input JSON bytes. Actual sources/plans unchanged.
    for family,mutator,name in [
        ('candidate',lambda s:s.update(num_classes=2),'candidate_wrong_class_count'),
        ('candidate',lambda s:s.update(hidden_dim=16),'candidate_wrong_width'),
        ('gru',lambda s:s['model_specs'][s['models'][0]].update(temporal_depth=1),'gru_wrong_temporal_depth'),
        ('candidate',lambda s:s.update(boundary_data=str(ROOT/'data/final_test_hypothetical')),'candidate_wrong_heldout_cache_path')]:
        value=deepcopy(settings[family]);mutator(value);changed=save(TEMP/'mutated_input_configs'/(name+'.json'),value)
        mutatedrefs=[{'family_id':family,**changed} if r['family_id']==family else r for r in refs]
        outcome=reject(lambda:run(mutatedrefs))
        if family=='candidate' and name!='candidate_wrong_heldout_cache_path':producer=reject(lambda:candidate.checked_development_protocol(changed['path']))
        elif family=='gru':producer=reject(lambda:common.checked_protocol(changed['path'],changed['sha256']))
        else:producer=reject(lambda:common.check_completion_settings(value,completion))
        record(name,{'combined_gate':outcome,'producer_local_guard':producer,'mutated_config_sha256':changed['sha256'],'active_file_modified':False})
    assert all(checks[name]['combined_gate']['rejected'] for name in ('candidate_wrong_class_count','candidate_wrong_width','gru_wrong_temporal_depth','candidate_wrong_heldout_cache_path'))
    # Ambiguous/missing aliases and required family omission must remain rejected.
    for family in ('candidate','canonical_gru'):
        value=deepcopy(settings[family]);other='deterministic_policy' if b.POLICY_FIELDS[family]=='deterministic_execution_policy' else 'deterministic_execution_policy'
        value[other]=deepcopy(DETERMINISTIC_POLICY);changed=save(TEMP/'mutated_input_configs'/('ambiguous_'+family+'.json'),value)
        mutatedrefs=[{'family_id':family,**changed} if r['family_id']==family else r for r in refs]
        record('ambiguous_policy_'+family,reject(lambda:run(mutatedrefs)))
    record('missing_required_family',reject(lambda:run(refs[:-1])))
# In-memory input ref mutation during admission is not a file-content rewrite.
with patch.object(Path,'read_bytes',side_effect=AssertionError('Held-out read attempted')):
    record('lexical_guard_before_any_read',reject(lambda:b.bind_development_selection(
        ROOT/'data/confirmation/hypothetical.json','a'*64,expected_scope_amendment_sha256=amendsha,family_configs=refs)))

# Pure exact assembly admission must not open any raw synthetic byte artifact.
from arggnn import development_selection as selector
# The end-to-end binder reads the source lock only to seal explicit bytes;
# the validator itself only reads captured control/plan/manifest files.
capture=selector._Capture(ROOT)
saved_read=Path.read_bytes
def raw_guard(path):
    if str(path) in raw_catalog:raise AssertionError('Pure settings validator read raw fixture')
    return saved_read(path)
with patch.object(Path,'read_bytes',raw_guard):
    for family in b.FAMILIES:
        b.validate_producer_settings(family,settings[family],completion=completion,completion_path=cp['path'],
            completion_sha256=cp['sha256'],capture=capture,source_snapshot=lock)
record('pure_ten_producer_assembly_no_raw_reads',{'validated':10,'raw_artifacts':len(raw_catalog),'source_snapshot_artifacts':len(lock.hashes)})

# Owned partial emission is deliberately interrupted; no final plan survives.
original_open=Path.open
class BrokenWriter:
    def __init__(self,handle):self.handle=handle
    def __enter__(self):return self
    def __exit__(self,*args):return self.handle.__exit__(*args)
    def write(self,payload):
        self.handle.write(payload[:17]);raise OSError('Independent synthetic receipt write interruption')
def broken_open(path,*args,**kwargs):
    handle=original_open(path,*args,**kwargs)
    if path.name=='binding_receipt.json' and args and args[0]=='xb':return BrokenWriter(handle)
    return handle
output=TEMP/'synthetic_interrupted_output'
with patch.object(common,'verify_completion',admitted),patch.object(Path,'open',broken_open):
    outcome=reject(lambda:b.bind_development_selection(cp['path'],cp['sha256'],expected_scope_amendment_sha256=amendsha,
        family_configs=refs,output_directory=output))
assert outcome['rejected'] and not output.exists()
record('partial_emission_cleanup',{'rejection':outcome,'final_directory_exists':output.exists()})

assert not torch.cuda.is_initialized()
record('final_cuda_state',{'initialized':False,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')})
