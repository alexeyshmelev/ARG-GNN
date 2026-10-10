"""Retest the original synthetic policy/settings/namespace mutations on repaired guards."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import hashlib,json,sys,os
import pytest,torch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/development_producer_audit_v1'
TEMP=ROOT/'.runtime/tmp/development_producer_audit_v1/post_repair'
TEMP.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(ROOT/'tests'))
from arggnn import development_execution as ex,development_training_bindings as common,argmix_development_bindings as rb,allrecord_development_bindings as ab,canonical_development_bindings as cb,priorart_development_bindings as pb
from arggnn import train_argmix_development as raw,train_allrecord_development as ar,train_anchor_development as candidate,train_anchor_temporal_development as temporal
from test_argmix_development import tiny_parents,tiny_settings
from test_allrecord_development import one_parent_cases
from test_allrecord_boundary_temporal import model as allrecord_model
from test_development_training_bindings import linked_sources
from test_train_anchor_development import test_tiny_cpu_fit_stores_hash_bound_raw_logit_predictions
torch.set_num_threads(1)
checks={}
def record(name,value):
    checks[name]=value;(OUT/'replay_post_repair.json').write_text(json.dumps({'no_gpu_or_actual_cohort_access':True,'checks':checks},indent=2,sort_keys=True)+'\n')
    print(name+': '+json.dumps(value,sort_keys=True),flush=True)
def folder(name):
    p=TEMP/name;p.mkdir();return p
def reject(call):
    try:call()
    except Exception as error:
        if not isinstance(error,(RuntimeError,ValueError)):raise
        return {'rejected':True,'type':type(error).__name__,'reason':str(error)}
    raise AssertionError('Mutation unexpectedly accepted')
for name,checker,relative in [('argmix',rb.check_plan,rb.PLAN),('allrecord',ab.validate_plan,'configs/allrecord_development_search_plan_v2.json'),('canonical',cb.check_plan,cb.PLAN),('priorart',pb.check_plan,pb.PLAN)]:
    checker(json.loads((ROOT/relative).read_bytes()));record(name+'_published_plan',{'valid':True,'sha256':common.sha(ROOT/relative)})
ex.configure_deterministic_development();p=folder('raw_policy')/'selected.pt';snap=rb.IntegritySnapshot()
record('raw_same_callback_policy_drift',reject(lambda:raw.train_configuration({'id':'synthetic_policy_drift','hidden_dim':8,'dropout':0.,'learning_rate':.001},
    {**tiny_settings(),'epochs':2},tiny_parents(),torch.device('cpu'),101,p,{'protocol_sha256':'a'*64,'implementation_sha256':snap.implementation_sha256},
    snap.check,lambda row:torch.use_deterministic_algorithms(False))))
ex.configure_deterministic_development();p=folder('allrecord_policy')/'selected.pt';case,banks=one_parent_cases()
source=p.parent/'source.json';source.write_text('{}');lock=common.SourceSnapshot({str(source):common.sha(source)})
record('allrecord_same_callback_policy_drift',reject(lambda:ar.train_configuration(allrecord_model('allrecord_gru'),{'model_learning_rates':{'synthetic':.001}},'synthetic',101,
    {'train':[case],'val':[case]},banks,p,{'protocol_sha256':'a'*64,'implementation_sha256':ar._IMPORTED_PINS},
    integrity_check=lambda **kw:lock.check(full=kw['full']),callback=lambda row:torch.use_deterministic_algorithms(False))))
ex.configure_deterministic_development();p=folder('candidate_policy');mp=pytest.MonkeyPatch();original=candidate.evaluate
def drift_eval(*args,**kwargs):
    torch.use_deterministic_algorithms(False);return original(*args,**kwargs)
mp.setattr(candidate,'evaluate',drift_eval)
try:record('candidate_same_main_policy_drift',reject(lambda:test_tiny_cpu_fit_stores_hash_bound_raw_logit_predictions(p,mp,'none')))
finally:mp.undo()
failure=json.loads((p/'run/failure.json').read_bytes())
record('candidate_same_main_failure_receipt',{'receipt':failure,'results_json_exists':(p/'run/results.json').exists()})
assert not (p/'run/results.json').exists() and failure['test_access'] is False
ex.configure_deterministic_development();torch.use_deterministic_algorithms(False)
with patch.object(temporal,'check_live',side_effect=AssertionError('No source operation before effective-policy rejection')):
    record('temporal_active_policy_guard',reject(lambda:temporal.check_active_execution({},None,None)))
ex.configure_deterministic_development();p=folder('settings');mp=pytest.MonkeyPatch()
try:
    completion,cp,artifacts=linked_sources(p,mp);values=common.make_configs(completion,cp,common.sha(cp))
    for family in ('candidate','gru','transformer','reference_smoother'):
        settings=values[family];config=p/(family+'.json');config.write_text(json.dumps(settings));digest=common.sha(config)
        _,lock=common.capture_completion_sources(cp,common.sha(cp));common.check_live(settings,config,digest,source_snapshot=lock,full_sources=True)
        changed=deepcopy(settings)
        if family=='candidate':changed['hidden_dim']=16
        elif family=='reference_smoother':changed['model_specs'][changed['models'][0]]['bandwidth_bp']=1
        else:changed['model_specs'][changed['models'][0]]['hidden_dim']=16
        record(family+'_same_settings_mutation',reject(lambda:common.check_live(changed,config,digest,source_snapshot=lock,full_sources=True)))
        assert common.sha(config)==digest
finally:mp.undo()
ex.configure_deterministic_development();p=folder('raw_runtime')/'config.json';p.write_text('{"hidden_dim":32}')
snap=rb.IntegritySnapshot();settings=snap.control(p,common.sha(p));snap.bind_runtime_configuration(p,settings);snap.check(full_sources=True)
settings['hidden_dim']=16;record('raw_bound_runtime_settings_mutation',reject(lambda:snap.check(full_sources=True)))
settings['hidden_dim']=32;torch.use_deterministic_algorithms(False);record('raw_bound_runtime_policy_mutation',reject(lambda:snap.check(full_sources=True)))
# Exercise unchanged shared composition with real runtime registration.
for family,constructor,member in [('canonical',cb.CanonicalIntegritySnapshot,'shared'),('priorart',pb.PriorArtSnapshot,'source')]:
    ex.configure_deterministic_development();p=folder(family+'_runtime')/'config.json';p.write_text('{"hidden_dim":32}')
    snapshot=constructor();settings=snapshot.control(p,common.sha(p))
    getattr(snapshot,member).bind_runtime_configuration(p,settings);snapshot.check(full_sources=True)
    settings['hidden_dim']=16
    record(family+'_composed_runtime_settings_mutation',reject(lambda:snapshot.check(full_sources=True)))
    settings['hidden_dim']=32;torch.use_deterministic_algorithms(False)
    record(family+'_composed_runtime_policy_mutation',reject(lambda:snapshot.check(full_sources=True)))
ex.configure_deterministic_development();p=folder('captured_control')/'config.json';p.write_text('{"hidden_dim":32}')
captured=ab.capture_control(p,common.sha(p));captured.values['hidden_dim']=16
record('captured_control_same_value_mutation',reject(lambda:captured.check(full=True)))
for namespace in ('test','confirmation','confirmatory','final_test','final-test'):
    target=ROOT/'data'/namespace/'hypothetical_control.json';reads=[]
    def forbidden_read(self):
        reads.append(str(self));raise AssertionError('Read attempted by before-open spy')
    with patch.object(Path,'read_bytes',forbidden_read):outcome=reject(lambda:temporal.captured_protocol(target,'a'*64))
    assert not reads;record('temporal_same_namespace_'+namespace,{'attempted_reads':reads,**outcome})
ex.configure_deterministic_development();assert not torch.cuda.is_initialized()
record('final_cuda_state',{'initialized':False,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')})
