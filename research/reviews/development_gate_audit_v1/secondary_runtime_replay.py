"""Active secondary guard replay. Main/cohort/producer admission are explicit mocks.
Real matched controls, numerical verifier, SourceSnapshot and SecondaryAdmission.check
are used. No actual cohort is accessed and no executable config is emitted.
"""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,json,sys,importlib.util,os
import pytest,torch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/development_gate_audit_v1'
TEMP=ROOT/'.runtime/tmp/development_gate_audit_v1/secondary_runtime';TEMP.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(ROOT/'tests'))
spec=importlib.util.spec_from_file_location('audit_frozen_secondary_fixture',OUT/'secondary_frozen_snapshot/tests/test_secondary_development.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
from arggnn import secondary_control_bindings as b,train_secondary_development as runner,development_execution as ex
class Factory:
    def mktemp(self,name):
        p=TEMP/name;p.mkdir();return p
evidence=fixture.evidence.__wrapped__(Factory())
checks={}
def record(name,value):
    checks[name]=value;(OUT/'secondary_runtime_replay.json').write_text(json.dumps({'positive_cohort_and_producer_admission_mocked':True,'no_actual_cohort_access':True,'checks':checks},indent=2,sort_keys=True)+'\n')
    print(name+': '+json.dumps(value,sort_keys=True),flush=True)
def reject(call):
    try:call()
    except Exception as error:return {'rejected':True,'type':type(error).__name__,'reason':str(error)}
    return {'rejected':False}
torch.set_num_threads(1)
for stage,mutation in (('validation','policy'),('validation','settings'),('reload','policy'),('reload','settings')):
    mp=pytest.MonkeyPatch();key=stage+'_'+mutation
    try:
        study=fixture.admitted.__wrapped__(evidence,mp,SimpleNamespace(node=SimpleNamespace(name=key)))
        bound=fixture.bind(study);settings=bound.settings
        original_candidate=deepcopy(settings['selected_candidate']);original_baseline=deepcopy(settings['primary_strongest_baseline'])
        train=deepcopy(study['cases']['train']);validation=deepcopy(study['cases']['val'][:1])
        # Distinct validation class induces legitimate patience stopping; feature input is unchanged.
        validation[0].labels.fill_(3);validation[0].tiles[0].labels=validation[0].labels
        cases={'train':train,'val':validation}
        packs={id(tile):study['pack'] for group in cases.values() for case in group for tile in case.tiles}
        ex.configure_deterministic_development();runner.set_seed(101)
        name=settings['models'][0];model=runner.make_control(name,settings).double()
        checkpoint=study['directory']/'weights.pt';prediction=study['directory']/'prediction.pt'
        mutation_calls=[];validation_calls=[];saved_validate=runner.validate;saved_reload=model.load_state_dict
        def mutate():
            mutation_calls.append(None)
            if mutation=='policy':torch.use_deterministic_algorithms(False)
            else:settings['hidden_dim']=16
        def validate(*args,**kwargs):
            result=saved_validate(*args,**kwargs);validation_calls.append(None)
            if stage=='validation' and len(validation_calls)==1:mutate()
            return result
        def reload(*args,**kwargs):
            result=saved_reload(*args,**kwargs)
            if stage=='reload':mutate()
            return result
        mp.setattr(runner,'validate',validate);mp.setattr(model,'load_state_dict',reload)
        def live(full):
            ex.deterministic_development_manifest()
            return bound.check(full)
        outcome=reject(lambda:runner.fit_configuration(model,cases,packs,name=name,seed=101,
            settings=settings,config_sha=bound.sha256,checkpoint=checkpoint,prediction=prediction,live_check=live))
        assert outcome['rejected'] and len(mutation_calls)==1
        record(key,{'outcome':outcome,'mutations':len(mutation_calls),'actual_validation_calls':len(validation_calls),
            'checkpoint_exists_after_failed_fit':checkpoint.exists(),'prediction_exists_after_failed_fit':prediction.exists(),
            'success_returned':False,'full_production_epoch_cap_preserved':settings['epochs']==48,
            'primary_candidate_unchanged':settings['selected_candidate']==original_candidate,
            'primary_baseline_unchanged':settings['primary_strongest_baseline']==original_baseline})
    finally:mp.undo();ex.configure_deterministic_development()
# Complete/partial secondary synthesis with the documented template states.
# The three seed rows below are synthetic serialized clones, not three actual fits.
mp=pytest.MonkeyPatch()
try:
    study=fixture.admitted.__wrapped__(evidence,mp,SimpleNamespace(node=SimpleNamespace(name='synthesis')))
    # Existing fixture writes an executable-shaped synthetic input config only
    # after mocked admission; it cannot pass genuine cohort admission.
    runs=fixture.secondary_runs.__wrapped__(study)
    complete=fixture.synth(runs)['values'];partial=fixture.synth(runs,runs['run_refs'][:2])['values']
    record('public_two_lr_selector',{'complete_cells':len(complete['cells']),'complete_matrix_resolved':complete['matrix_resolved'],
        'partial_matrix_resolved':partial['matrix_resolved'],'partial_choices':[row['choice'] for row in partial['controls'].values()],
        'primary_selection_changed':complete['primary_selection_changed'],'final_test_access':complete['final_test_access'],
        'authorizes_confirmatory_generation':complete['authorizes_confirmatory_generation'],
        'evidence_limit':'Four actual two-epoch toy template fits; three-seed cells are serialized synthetic clones under mocked admission.'})
    assert complete['matrix_resolved'] and not partial['matrix_resolved'] and all(row['choice'] is None for row in partial['controls'].values())
finally:mp.undo();ex.configure_deterministic_development()
assert not torch.cuda.is_initialized()
record('final_cuda_state',{'initialized':False,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')})
