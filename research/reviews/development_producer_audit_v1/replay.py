"""Independent CPU-only audit. All data/source facts below are synthetic."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import json, sys, os
import pytest
import torch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
REVIEW=ROOT/'research/reviews/development_producer_audit_v1'
TEMP=ROOT/'.runtime/tmp/development_producer_audit_v1/replay'
TEMP.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'tests'))
from arggnn import development_execution as execution
from arggnn import development_training_bindings as common
from arggnn import train_argmix_development as raw
from arggnn import train_allrecord_development as allrecord
from arggnn import train_canonical_development as canonical
from arggnn import train_priorart_development as prior
from arggnn import canonical_development_bindings as canonical_binding
from arggnn import priorart_development_bindings as prior_binding
from arggnn import allrecord_development_bindings as allrecord_binding
from arggnn import argmix_development_bindings as raw_binding
from arggnn import train_anchor_development as candidate
from arggnn import train_anchor_temporal_development as temporal
from test_argmix_development import tiny_parents, tiny_settings
from test_allrecord_development import one_parent_cases
from test_allrecord_boundary_temporal import model as allrecord_model
from test_development_training_bindings import linked_sources
from test_train_anchor_development import test_tiny_cpu_fit_stores_hash_bound_raw_logit_predictions
from test_canonical_development import tiny_parents as canonical_parents, tiny_settings as canonical_settings, tiny_binding
from test_priorart_development import tiny_parent as prior_parent, tiny_settings as prior_settings, tiny_configuration as prior_configuration

torch.set_num_threads(1)
results={'kind':'synthetic_CPU_producer_audit_replay','no_actual_cohort_or_gpu_access':True,'checks':{}}
def record(name,value):
    results['checks'][name]=value
    (REVIEW/'replay.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    print(name+': '+json.dumps(value,sort_keys=True),flush=True)
def folder(name):
    path=TEMP/name;path.mkdir(exist_ok=False);return path
def rejected(call):
    try:call()
    except Exception as error:return {'rejected':True,'type':type(error).__name__,'reason':str(error)}
    return {'rejected':False}
for name,checker in [('argmix_v2',raw_binding.check_plan),('allrecord_v2',allrecord_binding.validate_plan),('canonical_v1',canonical_binding.check_plan),('priorart_v1',prior_binding.check_plan)]:
    relative={'argmix_v2':raw_binding.PLAN,'allrecord_v2':'configs/allrecord_development_search_plan_v2.json','canonical_v1':canonical_binding.PLAN,'priorart_v1':prior_binding.PLAN}[name]
    plan=json.loads((ROOT/relative).read_bytes());checker(plan)
    record(name+'_published_plan',{'validated':True,'sha256':common.sha(ROOT/relative),'policy_field':'deterministic_policy' if 'deterministic_policy' in plan else 'deterministic_execution_policy'})
configuration={'id':'synthetic_policy_drift','hidden_dim':8,'dropout':0.,'learning_rate':.001}
execution.configure_deterministic_development();snap=raw_binding.IntegritySnapshot()
path=folder('raw_policy')/'selected.pt'
result,_=raw.train_configuration(configuration,{**tiny_settings(),'epochs':2},tiny_parents(),torch.device('cpu'),101,path,
    {'protocol_sha256':'a'*64,'implementation_sha256':snap.implementation_sha256},snap.check,
    lambda row:torch.use_deterministic_algorithms(False))
record('raw_runtime_policy_drift',{'fit_accepted':True,'deterministic_algorithms_at_return':torch.are_deterministic_algorithms_enabled(),
    'checkpoint_replay_nll_delta':result['checkpoint_replay_nll_delta'],'verifier_after_return':rejected(execution.deterministic_development_manifest),
    'source_checker':'real IntegritySnapshot.check; synthetic parents and two-epoch helper settings'})
execution.configure_deterministic_development();path=folder('allrecord_policy')/'selected.pt'
case,banks=one_parent_cases();lock_file=path.parent/'source.json';lock_file.write_text('{}')
lock=common.SourceSnapshot({str(lock_file):common.sha(lock_file)})
result,_=allrecord.train_configuration(allrecord_model('allrecord_gru'),{'model_learning_rates':{'synthetic':.001}},'synthetic',101,
    {'train':[case],'val':[case]},banks,path,{'protocol_sha256':'a'*64,'implementation_sha256':allrecord._IMPORTED_PINS},
    integrity_check=lambda **kw:lock.check(full=kw['full']),callback=lambda row:torch.use_deterministic_algorithms(False))
record('allrecord_runtime_policy_drift',{'fit_accepted':True,'epochs':len(result['history']),
    'deterministic_algorithms_at_return':torch.are_deterministic_algorithms_enabled(),'checkpoint_replay_nll_delta':result['checkpoint_replay_nll_delta'],
    'verifier_after_return':rejected(execution.deterministic_development_manifest),'source_checker':'real SourceSnapshot.check; synthetic case/bank'})
execution.configure_deterministic_development();path=folder('candidate_policy')
mp=pytest.MonkeyPatch();original=candidate.evaluate
calls=[]
def policy_drift_evaluate(*args,**kwargs):
    calls.append(len(calls));torch.use_deterministic_algorithms(False)
    return original(*args,**kwargs)
mp.setattr(candidate,'evaluate',policy_drift_evaluate)
try:test_tiny_cpu_fit_stores_hash_bound_raw_logit_predictions(path,mp,'none')
finally:mp.undo()
run=json.loads((path/'run/results.json').read_text());body=next(iter(run['models'].values()))
record('candidate_main_runtime_policy_drift',{'fit_accepted':True,'evaluation_calls':len(calls),
    'recorded_deterministic_algorithms':run['deterministic_execution_policy']['deterministic_algorithms'],
    'deterministic_algorithms_at_return':torch.are_deterministic_algorithms_enabled(),'checkpoint_replay_nll_delta':body['checkpoint_replay_nll_delta'],
    'receipt':str(path/'run/results.json'),'source_admission':'existing tiny synthetic test explicitly mocks production cohort admission'})
execution.configure_deterministic_development();path=folder('canonical_policy')/'selected.pt'
r=rejected(lambda:canonical.train_configuration(configuration,{**canonical_settings(),'epochs':2},canonical_parents(),torch.device('cpu'),101,path,
    {**tiny_binding(),'deterministic_policy':deepcopy(execution.DETERMINISTIC_POLICY)},callback=lambda row:torch.use_deterministic_algorithms(False)))
record('canonical_runtime_policy_drift',r)
execution.configure_deterministic_development();path=folder('priorart_policy')/'selected.pt';parent=prior_parent()
r=rejected(lambda:prior.train_configuration(prior_configuration('gst_ca_span'),prior_settings('gst_ca_span'),
    {'train':[parent],'val':[parent]},torch.device('cpu'),101,path,{},integrity_check=lambda **kw:None,
    callback=lambda row:torch.use_deterministic_algorithms(False)))
record('priorart_runtime_policy_drift',r)
execution.configure_deterministic_development();path=folder('settings_drift');mp=pytest.MonkeyPatch()
try:
    completion,cp,artifacts=linked_sources(path,mp)
    values=common.make_configs(completion,cp,common.sha(cp))
    for family in ('candidate','gru','transformer','reference_smoother'):
        settings=values[family];config=path/(family+'.json');config.write_text(json.dumps(settings));digest=common.sha(config)
        _,lock=common.capture_completion_sources(cp,common.sha(cp))
        common.check_live(settings,config,digest,source_snapshot=lock,full_sources=True)
        changed=deepcopy(settings)
        if family=='candidate':changed['hidden_dim']=16
        elif family=='reference_smoother':changed['model_specs'][changed['models'][0]]['bandwidth_bp']=1
        else:changed['model_specs'][changed['models'][0]]['hidden_dim']=16
        outcome=rejected(lambda:common.check_live(changed,config,digest,source_snapshot=lock,full_sources=True))
        record(family+'_common_live_accepts_mutated_settings',{'original_config_sha256':digest,'current_file_sha256':common.sha(config),
            'mutation_rejected':outcome['rejected'],'source_artifacts':len(lock.hashes),'registry_mocked':True})
finally:mp.undo()
path=folder('captured_control_drift')/'control.json';path.write_text('{"hidden_dim":32}')
captured=allrecord_binding.capture_control(path,common.sha(path));captured.values['hidden_dim']=16
record('captured_control_value_drift',{'full_check':rejected(lambda:captured.check(full=True)),
    'captured_value':captured.values,'original_bytes':captured.content.decode()})
for namespace in ('test','confirmation','confirmatory','final_test','final-test'):
    target=ROOT/'data'/namespace/'hypothetical_control.json'
    reads=[]
    def forbidden_read(self):
        reads.append(str(self));raise AssertionError('Read attempted by before-open spy')
    with patch.object(Path,'read_bytes',forbidden_read):
        outcome=rejected(lambda:temporal.captured_protocol(target,'a'*64))
    record('temporal_namespace_'+namespace,{'attempted_reads':reads,**outcome})
execution.configure_deterministic_development()
assert not torch.cuda.is_initialized()
record('final_cuda_state',{'initialized':torch.cuda.is_initialized(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')})
