"""Focused handoff, resource and scientific-parity tests; all signals mocked."""
import copy
import json
import signal
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile
from research.joint_evaluation_20261029.queue import successor_v2_gpu as gpu
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue import successor_v2_support as support
from research.joint_evaluation_20261029.queue import successor_v2_handoff as handoff
from research.joint_evaluation_20261029.queue import successor_v2_scheduler as scheduler

HERE = Path(__file__).resolve().parent


def main():
    checks = []
    def check(name, value):
        assert value, name
        checks.append(name)
    def rejects(name, function):
        try:
            function()
        except (ValueError, RuntimeError, AssertionError):
            checks.append(name)
        else:
            raise AssertionError(name)
    config_ref = r.ref(HERE/'successor_v2_config.json')
    config = r.read(config_ref)
    plan = r.read(config['successor_template_ref'])
    source = r.read(config['effective_plan_ref'])
    state = json.loads(r.path(config['original_state_path']).read_text())
    original = {t['id']: t for t in source['tasks']}
    tasks = {t['id']: t for t in plan['tasks']}
    check('Exact252original-task namespace plusnewbarrier DAG validates', scheduler.validate(plan) is plan and len(tasks)==253)
    check('Exactly 36 disjoint sidecar aliases; no child duplicated', len(plan['alias_task_ids'])==36 and all(tasks[n]['kind']=='external' for n in plan['alias_task_ids']))
    changed = 0
    for name, task in tasks.items():
        if name in plan['added_task_ids']:
            check('New joint barrier is external and waits exact future receipt', task['kind']=='external' and task['source_task_id']=='joint_argmix.prediction_barrier' and task['expected_result_path']=='experiments/results/joint_argmix_20261029/prediction_barrier/barrier.json')
            continue
        if name in plan['alias_task_ids']:
            check('External alias keeps ID and original dependency edges '+name, task['dependencies']==original[name]['dependencies'])
            continue
        normalized = copy.deepcopy(task)
        if name in plan['operational_amendments']:
            changed += 1
            check('Arbitrary controller wall limit removed '+name, normalized['max_seconds']==support.MAX_SECONDS)
            normalized['max_seconds'] = original[name]['max_seconds']
            if 'argv' in normalized:
                args = normalized['argv']
                old_args = original[name]['argv']
                if '--timeout-seconds' in args:
                    index = args.index('--timeout-seconds')+1
                    check('CLI wall limit removed '+name, args[index]==str(support.MAX_SECONDS))
                    args[index] = old_args[index]
                if name.endswith('.native_relate'):
                    check('Only native CLI timeout adapter used '+name, args[3]=='research.joint_evaluation_20261029.queue.successor_native')
                    args[3] = old_args[3]
                    normalized['source_refs'] = normalized['source_refs'][:-2]
        if name=='paper_argmix_paired.prediction_barrier':
            check('Paired comparison explicitly uses new joint barrier', 'joint_argmix.prediction_barrier' in normalized['dependencies'] and normalized['argv'][normalized['argv'].index('--other-barrier')+1].endswith('/joint_argmix_20261029/prediction_barrier/barrier.json'))
            normalized['dependencies']=original[name]['dependencies']
            normalized['argv'][normalized['argv'].index('--other-barrier')+1]=original[name]['argv'][original[name]['argv'].index('--other-barrier')+1]
        check('Exact native science/caps/source and explicit comparison delta '+name, normalized==original[name])
    check('All pending nonalias work explicitly amended', changed==len(plan['operational_amendments']))
    check('Only authorized physical GPUs', plan['physical_gpus']==[3,4,5])
    check('Foreign RAM includes512GiB native+64GiB CNN+64GiB American+64GiB new ARGMix+64GiB European4', plan['foreign_ram_reserve_bytes']==768*2**30)
    check('Original scope cap retained', plan['scope_limits']==source['scope_limits'])
    fixture = {'tasks': {'a':dict(status='blocked',error='Output already exists, no overwrite'),
                         'b':dict(status='blocked',error='Required predecessor failed'),
                         'c':dict(status='failed',pid=99,error='Real historical failure'),
                         'd':dict(status='blocked',error='Required predecessor failed'),
                         'e':dict(status='blocked',error='Required predecessor failed')}}
    defs = {'a':dict(dependencies=[]),'b':dict(dependencies=['a']),'c':dict(dependencies=[]),
            'd':dict(dependencies=['a','c']),'e':dict(dependencies=[])}
    rows, prior = support.reset_rows(fixture, defs, ['a'])
    check('Only alias and its genuinely freed descendant reset', rows['a']['status']=='pending' and rows['b']['status']=='pending' and set(prior)=={'a','b'})
    check('Genuine failed row and failed-dependent row unchanged', rows['c']==fixture['tasks']['c'] and rows['d']==fixture['tasks']['d'])
    check('Unrelated blocked history unchanged', rows['e']==fixture['tasks']['e'])
    bad=copy.deepcopy(fixture);bad['tasks']['a']['pid']=123
    rejects('Launched attempt cannot be replaced', lambda: support.reset_rows(bad,defs,['a']))
    bad=copy.deepcopy(fixture);bad['tasks']['c']['status']='running'
    rejects('Running original work prevents seeding', lambda: support.reset_rows(bad,defs,['a']))
    good = copy.deepcopy(state)
    for row in good['tasks'].values():
        if row['status']=='running':
            row.update(status='succeeded',actual_wait=True,actual_returncode=0,owned_process_group_absent=True)
    for name in config['required_successful_boundary_tasks']:
        good['tasks'][name]=dict(status='succeeded',actual_wait=True,actual_returncode=0)
    good['tasks'][config['required_claim_block_task']]=dict(status='blocked',error='Output already exists, no overwrite')
    check('Genuine idle claim boundary recognized', handoff.boundary_reason(good,config,[]) is None)
    check('Live descendant blocks initial discovery', handoff.boundary_reason(good,config,[123])=='Original controller has live descendants')
    bad=copy.deepcopy(good);bad['tasks'][config['required_successful_boundary_tasks'][0]]['actual_wait']=False
    check('Fabricated completion cannot establish boundary', handoff.boundary_reason(bad,config,[]) is not None)
    ident=dict(pid=config['prior_controller']['pid'],start_ticks=config['prior_controller']['start_ticks'],state='S')
    with patch.object(handoff,'identity',return_value=ident), patch.object(handoff,'load_original',return_value=state), patch.object(handoff,'descendants',return_value=[1]), patch.object(handoff.processes,'_pidfd_open') as opening, patch.object(handoff.processes,'_pidfd_send') as sending:
        result=handoff.attempt_idle_stop(config)
        check('Busy original receives zero signals and no pidfd', not result['stopped'] and opening.call_count==sending.call_count==0)
    with ExitStack() as stack:
        stack.enter_context(patch.object(handoff,'identity',side_effect=[ident,ident,{**ident,'state':'T'}]))
        stack.enter_context(patch.object(handoff,'load_original',return_value=good))
        stack.enter_context(patch.object(handoff,'descendants',side_effect=[[],[123]]))
        stack.enter_context(patch.object(handoff.processes,'_pidfd_open',return_value=77))
        sending=stack.enter_context(patch.object(handoff.processes,'_pidfd_send'))
        stack.enter_context(patch.object(handoff.os,'close'))
        result=handoff.attempt_idle_stop(config)
        check('Launch race receives STOP then CONT only; never TERM', not result['stopped'] and [c.args[1] for c in sending.call_args_list]==[signal.SIGSTOP,signal.SIGCONT])
    with ExitStack() as stack:
        stack.enter_context(patch.object(handoff,'identity',side_effect=[ident,ident,{**ident,'state':'T'},None]))
        stack.enter_context(patch.object(handoff,'load_original',return_value=good))
        stack.enter_context(patch.object(handoff,'descendants',return_value=[]))
        stack.enter_context(patch.object(handoff.processes,'session_members',return_value=[]))
        stack.enter_context(patch.object(handoff.processes,'_pidfd_open',return_value=77))
        sending=stack.enter_context(patch.object(handoff.processes,'_pidfd_send'))
        stack.enter_context(patch.object(handoff.os,'close'))
        stack.enter_context(patch.object(handoff.r,'ref',return_value=dict(path='fixture',sha256='fixture')))
        result=handoff.attempt_idle_stop(config)
        check('Verified frozen idle controller receives exact pidfd STOP TERM CONT', result['stopped'] and [c.args for c in sending.call_args_list]==[(77,signal.SIGSTOP),(77,signal.SIGTERM),(77,signal.SIGCONT)])
        check('Controller exit observation is never a fabricated wait', result['old_controller_wait_consumed'] is False)
    real_alias = tasks['paper_american_argformer_american_test_003_prepare']
    real_result = support.external_result(real_alias)
    evidence = support.external_evidence(real_alias)
    check('Actual completed native sidecar result authenticates', real_result and r.read(evidence['external_actual_wait_ref'])['actual_wait'] is True)
    check('Alias evidence does not invent local process wait', 'actual_wait' not in evidence and 'pid' not in evidence)
    with patch.object(support,'validate_wait',side_effect=ValueError('Changed wait')):
        rejects('Changed sidecar wait prevents completion adoption', lambda: support.external_result(real_alias))
    with patch.object(support.r,'scope_bytes',return_value=0):
        scope_plan=dict(scope_limits=[dict(path='experiments/testscope',max_bytes=100)])
        candidate=dict(output_directory='experiments/testscope/a',max_output_bytes=101)
        check('Candidate growth cannot overcommit128GiB scope analog', not support.task_fits_scope(scope_plan,{},candidate,[]))
        candidate['max_output_bytes']=60
        active=[dict(output_directory='experiments/testscope/b',max_output_bytes=41)]
        check('Active plus candidate caps cannot overcommit scope', not support.task_fits_scope(scope_plan,{},candidate,active))
        candidate['max_output_bytes']=59
        check('Exact scope bound is admissible', support.task_fits_scope(scope_plan,{},candidate,active))
    accounting=support.commitment(dict(plan,handoff_config_ref=config_ref))
    check('Full ARGMix scope and American allocation precharged', accounting['items']['Entire original ARGMix scope cap']==128*2**30 and accounting['items']['American full additive allocation']==64*2**30)
    check('Old envelope is replaced with explicit owned scopes', not any('growth envelope' in name for name in accounting['items']))
    check('Admission uses exact1.2TB ceiling', accounting['admitted']==(accounting['total_bytes']<=1_200_000_000_000))
    check('Absent root-priority release keeps every GPU closed', gpu.allowed(dict(plan,handoff_config_ref=config_ref))==[])
    with patch.object(gpu,'root_release',return_value={'path':'test','sha256':'test'}):
        check('Unfinished native lanes restrict successor to GPU3', gpu.allowed(dict(plan,handoff_config_ref=config_ref))==[3])
    release=dict(schema='root_joint_models_gpu_release_v1',successor_v2_config_ref=config_ref,
                 completed_models={'american_joint':{'path':'fixture_a','sha256':'fixture'},'argmix_joint':{'path':'fixture_b','sha256':'fixture'},'european_four_joint':{'path':'fixture_c','sha256':'fixture'}})
    wait=dict(pid=999,actual_wait=True,actual_returncode=0,owned_process_group_absent=True)
    with patch.object(gpu.r,'path',return_value=Mock(is_file=lambda:True)), patch.object(gpu.r,'ref',return_value={'path':'fixture_release','sha256':'fixture'}), patch.object(gpu.r,'read',side_effect=[release,wait,wait,wait]), patch.object(gpu.processes,'session_members',return_value=[]):
        check('All three genuine root model chain waits permit release', gpu.root_release(dict(plan,handoff_config_ref=config_ref)) is not None)
    with patch.object(gpu.r,'path',return_value=Mock(is_file=lambda:True)), patch.object(gpu.r,'ref',return_value={'path':'fixture_release','sha256':'fixture'}), patch.object(gpu.r,'read',side_effect=[release,dict(wait,actual_wait=False)]):
        rejects('Missing root consumed wait cannot release GPU', lambda:gpu.root_release(dict(plan,handoff_config_ref=config_ref)))
    with tempfile.TemporaryDirectory(dir=HERE,prefix='.gpu_lock_fixture_') as directory:
        with patch.object(gpu.r,'path',return_value=Path(directory)/'lock'):
            first=gpu.acquire(3)
            try:
                check('GPU3 shared flock is exclusive across independent opens', first is not False and gpu.acquire(3) is False)
            finally:
                gpu.release(first)
            next_lease=gpu.acquire(3)
            check('GPU3 is available after owned lease closes', next_lease is not False)
            gpu.release(next_lease)
    check('GPU4and5 do not acquire GPU3 lock', gpu.acquire(4) is None and gpu.acquire(5) is None)
    with patch.object(support.r,'scope_bytes',return_value=256*2**20+1):
        check('Controller metadata overflow fails own admission bound', not support.metadata_within_bound(plan))
    with patch.object(support.r,'scope_bytes',return_value=256*2**20):
        check('Exact controller metadata bound admitted', support.metadata_within_bound(plan))
    spec=r.read(plan['supersession_ref'])
    replacement_rows,replaced=support.apply_supersession(state['tasks'],spec)
    check('Exactly11unlaunched legacy rows explicitlysuperseded', len(replaced)==11 and all(replacement_rows[n]['status']=='superseded' and replacement_rows[n]['scientifically_complete'] is False for n in replaced))
    check('Supersession preserves all unrelated and genuinefailed history', all(replacement_rows[n]==row for n,row in state['tasks'].items() if n not in replaced))
    bad=copy.deepcopy(state['tasks']);bad[next(iter(replaced))]['pid']=123
    rejects('Supersession refuses any launched legacyattempt',lambda:support.apply_supersession(bad,spec))
    check('Unavailable futurebarrier remains pending without inventedwait',support.external_result(tasks['joint_argmix.prediction_barrier']) is None)
    upgrade=support.commitment(dict(plan,handoff_config_ref=config_ref),argmix_override_bytes=25*2**30)
    check('Productionproposal charges full25GiB, not old8',upgrade['items']['New ARGMix full additive allocation']==25*2**30 and upgrade['total_bytes']-accounting['total_bytes']==17*2**30)
    admission=dict(schema='joint_argmix_cache_admission_v1',admitted=True,root_confirmed=True,
        prior_allocation_ref=config['argmix_joint_allocation_ref'],successor_config_ref=config_ref,
        cache_scope='experiments/cache/joint_argmix_20261029',results_scope='experiments/results/joint_argmix_20261029',
        cache_max_bytes=24*2**30,results_max_bytes=2**30,owned_RAM_max_bytes=64*2**30,
        combined_reserved_bytes=1_190_000_000_000,owner_ceiling_bytes=1_200_000_000_000,
        protocol_ref={'path':'fixture_protocol','sha256':'fixture'},actual_ledger_receipt_ref={'path':'fixture_ledger','sha256':'fixture'})
    ledger=dict(admitted=True,cache_scope=admission['cache_scope'],cache_max_bytes=admission['cache_max_bytes'],
        combined_reserved_bytes=admission['combined_reserved_bytes'],accounting=dict(total_bytes=admission['combined_reserved_bytes'],items={'New ARGMix full additive allocation':25*2**30}))
    with patch.object(support.r,'path',return_value=Mock(exists=lambda:True)),patch.object(support.r,'ref',return_value={'path':'fixture_admission','sha256':'fixture'}),patch.object(support.r,'read',side_effect=[admission,{},ledger]):
        check('Authorized exact production ledger charges25GiB',support.argmix_reserved_bytes(dict(plan,handoff_config_ref=config_ref),config,{'additional_total_bytes':8*2**30})==25*2**30)
    for change,label in [({'root_confirmed':False},'Unapproved production proposal cannot activate'),({'combined_reserved_bytes':1_200_000_000_001},'Production upgrade rejects one byte over ceiling'),({'successor_config_ref':{'path':'wrong','sha256':'wrong'}},'Production upgrade rejects another controller binding')]:
        with patch.object(support.r,'path',return_value=Mock(exists=lambda:True)),patch.object(support.r,'ref',return_value={'path':'fixture_admission','sha256':'fixture'}),patch.object(support.r,'read',return_value=dict(admission,**change)):
            rejects(label,lambda:support.argmix_reserved_bytes(dict(plan,handoff_config_ref=config_ref),config,{'additional_total_bytes':8*2**30}))
    bad_ledger=copy.deepcopy(ledger);bad_ledger['accounting']['items']['New ARGMix full additive allocation']=8*2**30
    with patch.object(support.r,'path',return_value=Mock(exists=lambda:True)),patch.object(support.r,'ref',return_value={'path':'fixture_admission','sha256':'fixture'}),patch.object(support.r,'read',side_effect=[admission,{},bad_ledger]):
        rejects('Production admission cannot retain obsolete8GiB ledger charge',lambda:support.argmix_reserved_bytes(dict(plan,handoff_config_ref=config_ref),config,{'additional_total_bytes':8*2**30}))
    future=tasks['joint_argmix.prediction_barrier'];future_wait=dict(task_id=future['source_task_id'],model_id=future['producer_model_id'],actual_wait=True,actual_returncode=0,owned_process_group_absent=True,pid=999,result_ref={'path':future['expected_result_path'],'sha256':'fixture'})
    actual_path=support.r.path
    def future_paths(value):
        return Mock(is_file=lambda:True) if value==future['future_wait_path'] else actual_path(value)
    from arggnn.benchmarking.paper_argmix import paper_argmix_score as native_score
    with patch.object(support.r,'path',side_effect=future_paths),patch.object(support.r,'ref',return_value={'path':'fixture_wait','sha256':'fixture'}),patch.object(support.r,'read',side_effect=[future_wait,{},{}]),patch.object(support.processes,'session_members',return_value=[]),patch.object(native_score,'verify_barrier') as verification:
        check('Future result requires native complete-barrier verification',support.future_result(future)==future_wait['result_ref'] and verification.call_count==1)
    for change,label in [({'actual_wait':False},'Future alias rejects unconsumed wait'),({'task_id':'wrong'},'Future alias rejects wrong producer task')]:
        with patch.object(support.r,'path',side_effect=future_paths),patch.object(support.r,'ref',return_value={'path':'fixture_wait','sha256':'fixture'}),patch.object(support.r,'read',return_value=dict(future_wait,**change)):
            rejects(label,lambda:support.future_result(future))
    handoff.check_config(config)
    check('Every reviewed source pin authentic', True)
    receipt=dict(schema='successor_focused_tests_v1',passed=True,check_count=len(checks),checks=checks,
                 config_ref=config_ref,template_ref=config['successor_template_ref'],current_accounting=accounting,
                 signals_sent=0,workers_launched=0,claims_created=0,live_state_mutations=0)
    r.save(HERE/'successor_v2_tests.json',receipt)
    print(json.dumps(dict(passed=True,check_count=len(checks),config_ref=config_ref,signals_sent=0,workers_launched=0)))


if __name__=='__main__':
    main()
