"""CPU metadata and mocked wiring only; no children,data generator or native tools."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
from . import contracts_v3 as c, guardian_v3 as guardian, worker_v3 as worker
from .retained_allowance_v1 import configure_retained_allowance, aggregate

BASE=Path(__file__).parent
RESULTS=[]


def require_rejection(label,callback):
    try:callback()
    except (ValueError,RuntimeError):RESULTS.append(label);return
    raise AssertionError('Expected rejection: '+label)


def row(pid,parent,start,session=None,rss=1):
    return dict(pid=pid,ppid=parent,start=start,session=pid if session is None else session,
        pgrp=pid,state='S',rss_bytes=rss)


def release_tests():
    manifest=dict(path='manifest',sha256='m');review=dict(path='review',sha256='r')
    value=dict(schema='European_short_VAL_ROOT_execution_release_v1',passed=True,execution_authorized=True,
        source_manifest_ref=manifest,independent_source_review_ref=review,plan_ref=c.plan.PLAN_REF,
        aggregate_directory=str(c.AGGREGATE.relative_to(c.ROOT)),host_cap_bytes=c.HOST_CAP,
        reservation_bytes=c.RESERVATION,operating_stop_bytes=c.OPERATING,max_seconds=c.MAX_SECONDS,
        CUDA_VISIBLE_DEVICES='',peer_directories=[],global_project_ceiling_bytes=1000000000000,
        ROOT_live_global_storage_and_peer_admission_required=True,model_fit_prediction_or_scoring_authorized=False,
        ROOT_status_max_age_seconds=c.ROOT_STATUS_MAX_AGE_SECONDS,
        growth_refresh_interval_seconds=c.GROWTH_INTERVAL_SECONDS,retained_attempt_ref=c.ref(c.RETAINED_REF_PATH))
    assert c.validate_release(value,manifest,review,[]) is value
    for key,replacement in [('passed',False),('execution_authorized',False),('source_manifest_ref',{}),
            ('plan_ref',{}),('host_cap_bytes',1),('reservation_bytes',1),('operating_stop_bytes',1),
            ('max_seconds',1),('CUDA_VISIBLE_DEVICES','0'),('model_fit_prediction_or_scoring_authorized',True),
            ('ROOT_status_max_age_seconds',1),('growth_refresh_interval_seconds',5),('retained_attempt_ref',{})]:
        bad=dict(value,**{key:replacement})
        require_rejection('release_reject_'+key,lambda:c.validate_release(bad,manifest,review,[]))
    bad=dict(value,peer_directories=[str(c.AGGREGATE.relative_to(c.ROOT))])
    require_rejection('own_aggregate_not_peer',lambda:c.validate_release(bad,manifest,review,bad['peer_directories']))
    bad=dict(value,peer_directories=[str(c.RETAINED_AGGREGATE.relative_to(c.ROOT))])
    require_rejection('retained_attempt_cannot_claim_second_peer_reservation',
        lambda:c.validate_release(bad,manifest,review,bad['peer_directories']))
    RESULTS.append('exact_ROOT_release_pass')


def family_tests():
    supervisor=1;leader=row(10,1,100);native=row(11,10,101);grandchild=row(12,11,102,session=11)
    known={10:leader}
    table={10:leader,11:native,12:grandchild,99:row(99,90,90)}
    assert set(guardian.capture_family(leader,known,supervisor,table))=={10,11,12}
    table={11:row(11,1,101),12:row(12,11,102,session=11),13:row(13,1,103),
           10:row(10,90,200),99:row(99,90,90)}
    assert set(guardian.capture_family(leader,known,supervisor,table))=={11,12,13}
    assert 10 not in guardian.capture_family(leader,known,supervisor,table)
    RESULTS.append('recursive_sessions_adopted_children_and_recycled_PID')
    class Child:
        pid=10
        def wait(self,timeout=None):return -15
        def poll(self):return -15
    observed=[];calls=[{10:leader},{10:leader},{},{ }]
    def capture(*args):return calls.pop(0) if calls else {}
    with patch.object(guardian,'capture_family',capture),patch.object(guardian,'reap_adopted',lambda *a:[]),\
         patch.object(guardian,'signal_owned',lambda family,signum:observed.append((set(family),signum))),\
         patch.object(guardian.time,'sleep',lambda delay:None):
        receipt=guardian.wait_owned(Child(),leader,{10:leader},terminate=True)
    assert receipt['actual_wait'] and receipt['actual_returncode']==-15 and receipt['owned_family_absent']
    assert observed==[({10},guardian.signal.SIGTERM)]
    RESULTS.append('mocked_actual_wait_and_exact_family_shutdown')


def runtime_gate_tests(fixture):
    work=fixture/'runtime';work.mkdir()
    pid=os.getpid();parent=os.getppid();own=row(pid,parent,12);supervisor=row(parent,1,11)
    table={pid:own,parent:supervisor};argv=['--metadata-fixture'];process_reads=[]
    class Clock:
        tick=100.
        def monotonic(self):return self.tick
        def time(self):return time.time()
    clock=Clock()
    def processes():process_reads.append(clock.tick);return copy.deepcopy(table)
    command=['.runtime/env/bin/python3.11','-B','-m',c.MODULE,*argv]
    now=int(time.time())
    request=dict(command=command,host_cap_bytes=c.HOST_CAP,output_cap_bytes=c.RESERVATION,
        operating_stop_bytes=c.OPERATING,max_seconds=c.MAX_SECONDS,CUDA_VISIBLE_DEVICES='',
        aggregate_directory=str(c.AGGREGATE.relative_to(c.ROOT)),supervisor_pid=parent,
        supervisor_start_ticks='11',created_unix=now,deadline_unix=now+c.MAX_SECONDS)
    launch=dict(actual_argv=command,pid=pid,process_start_ticks='12')
    for name,value in [('request.json',request),('launch.json',launch),('release.json',{})]:
        (work/name).write_text(json.dumps(value))
    status=work/'status.json';status.write_text(json.dumps(dict(status='running',observed_unix=time.time())))
    original_read=Path.read_bytes
    def read_bytes(path):
        if str(path)=='/proc/self/cmdline':return ('\0'.join(command)+'\0').encode()
        return original_read(path)
    with patch.object(worker,'AGGREGATE',work),patch.object(worker,'process_table',processes),\
         patch.object(worker,'time',clock),patch.object(Path,'read_bytes',read_bytes),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
        check,remaining=worker.execution_checker(c.ref(work/'request.json'),c.ref(work/'launch.json'),
            c.ref(work/'release.json'),request,launch,argv)
        assert 0<remaining<=c.MAX_SECONDS;check()
        RESULTS.append('actual_argv_identity_deadline_and_fresh_ROOT_gate')
        before=len(process_reads)
        for _ in range(100):check()
        assert len(process_reads)==before
        clock.tick+=1;check();assert len(process_reads)==before+1
        RESULTS.append('one_process_sample_per_second_despite100_callbacks')
        clock.tick+=1
        status.write_text(json.dumps(dict(status='running',observed_unix=time.time()-17)))
        check();assert 17<=check.diagnostics['last_status_age_seconds']<18
        RESULTS.append('measured17_second_synchronous_admission_delay_remains_healthy')
        clock.tick+=1
        status.write_text(json.dumps(dict(status='running',observed_unix=time.time()-121)))
        require_rejection('stale_ROOT_status_past120_seconds',check)
        prior_reads=len(process_reads);require_rejection('failed_status_sample_not_cached',check)
        assert len(process_reads)==prior_reads+1
        status.write_text(json.dumps(dict(status='running',observed_unix=time.time()+2)))
        require_rejection('future_ROOT_status',check)
        status.write_text(json.dumps(dict(status='running',observed_unix=time.time())))
        clock.tick+=1
        table[parent]['start']=13;require_rejection('recycled_guardian_generation',check);table[parent]['start']=11
        clock.tick+=1
        table[pid]['ppid']=999;require_rejection('orphan_worker_rejected',check);table[pid]['ppid']=parent
        clock.tick+=1
        table[999]=row(999,pid,13,rss=c.HOST_CAP)
        require_rejection('native_new_session_counts_in_total_RSS',check);del table[999]
        check()
        with patch.object(clock,'time',lambda:request['deadline_unix']):
            require_rejection('Unix_deadline_immediate_despite120_second_lease',check)
        saved=clock.tick;clock.tick=100.+c.MAX_SECONDS
        require_rejection('monotonic_deadline_immediate_despite120_second_lease',check);clock.tick=saved
        (work/'execution_hold').write_text('CPU fixture hold\n')
        require_rejection('immediate_ROOT_hold_even_with_fresh_cached_sample',check)


def stage_wiring_tests(fixture):
    work=fixture/'stages';work.mkdir();events=[];holder={}
    frozen=c.plan.validate_plan()
    class Owner:
        def __init__(self):self.execution_check=self.check
        def path(self,path):assert path.is_relative_to(work);return path
        def check(self):events.append('check')
        def __call__(self,amount):assert type(amount) is int and amount>=0
    owner=Owner();tools={'reviewed':'hash'};config={'native':'metadata-only'}
    def generate(recipe,parent,output,**kwargs):
        assert recipe==frozen['recipe'] and parent['seeds']==frozen['registration']['seeds']
        assert kwargs['guard'] is owner and kwargs['execution_check'] is owner.execution_check
        assert not list(output.iterdir());(output/'observed').mkdir();files={}
        for name in worker.native_relate_v3.OBSERVED_FILES:
            path=output/'observed'/name;path.write_text('Tiny metadata fixture\n')
            files['observed/'+name]=dict(sha256=c.ref(path)['sha256'],bytes=path.stat().st_size)
        events.append('generation');return {'generation':{'files':files}}
    def supervisor(shared,*,reviewed_native_tools):
        assert shared is owner and reviewed_native_tools is tools
        holder['native']=object();return holder['native']
    def infer(recipe,parent,observed,destination,**kwargs):
        assert kwargs['guard'] is owner and kwargs['execution_check'] is owner.execution_check
        assert kwargs['native_step'] is holder['native'] and kwargs['native_config'] is config
        assert kwargs['max_metadata_bytes']==256*2**20 and kwargs['write_headroom_bytes']==2**30
        assert set(kwargs['expected_input_refs'])==set(worker.native_relate_v3.OBSERVED_FILES)
        assert all('/observed/' in value['path'] for value in kwargs['expected_input_refs'].values())
        destination.mkdir();(destination/'observed_sample_manifest.json').write_text('[]\n')
        (destination/'tree.fixture').write_text('No tree data\n');(destination/'commands.json').write_text('[]\n')
        value=dict(observed_sample_manifest=c.ref(destination/'observed_sample_manifest.json'),
            inferred_tree=c.ref(destination/'tree.fixture'),commands=c.ref(destination/'commands.json'),
            post_five_iterations={'metadata_only':True})
        (destination/'inference.json').write_text(json.dumps(value));events.append('native');return value
    def sample(sampling_plan,parent_id,rows,metadata,*,observation_ref):
        assert sampling_plan==frozen['sampling_plan'] and parent_id==frozen['registration']['parent_id']
        assert rows==[] and metadata=={'metadata_only':True}
        assert observation_ref==c.ref(work/'native/inference.json')
        events.append('sampling')
        return dict(examples=10000,labels_present=False,query_truth_read=False,fixture=True)
    with patch.object(worker,'AGGREGATE',work),patch.object(worker.generation_v3,'generate_parent',generate),\
         patch.object(worker,'NativeSupervisor',supervisor),patch.object(worker.native_relate_v3,'infer_parent',infer),\
         patch.object(worker.sampling_v1,'build_validation_registry',sample):
        result=worker.run_stages(owner,frozen,config,tools)
    assert [item for item in events if item!='check']==['generation','native','sampling']
    assert result['sampled_registry_ref']==c.ref(work/'sampled_registry.json')
    assert result['independent_chromosome_replication'] is False and not result['model_fit_prediction_or_scoring_performed']
    assert not (work/'generation/truth').exists()
    RESULTS.append('same_owner_exact_frozen_stages_observed_whitelist_registry_seal')


def retained_allowance_tests(fixture):
    actual_ref,actual_charge=c.retained_attempt()
    assert actual_ref==c.ref(c.RETAINED_REF_PATH) and actual_charge==45056
    RESULTS.append('exact_failed_wait_family_absence_and45056_real_retained_bytes')
    with patch.object(guardian,'measure',lambda path:dict(charged_bytes=actual_charge if path==c.RETAINED_AGGREGATE else 1234)):
        combined=guardian.combined_storage(actual_charge)
        assert combined['combined_charged_bytes']==actual_charge+1234
    with patch.object(guardian,'measure',lambda path:dict(charged_bytes=actual_charge+1)):
        require_rejection('retained_attempt_growth_rejected',lambda:guardian.combined_storage(actual_charge))
    RESULTS.append('guardian_combines_both_attempt_charges_once')
    with patch.object(aggregate,'OPERATING',c.OPERATING),patch.object(aggregate,'RESERVATION',c.RESERVATION):
        limits=configure_retained_allowance(actual_charge)
        assert limits['attempt_operating_stop_bytes']+actual_charge==c.OPERATING
        assert limits['attempt_reservation_bytes']+actual_charge==c.RESERVATION
        assert 0<aggregate.OPERATING<aggregate.RESERVATION
        require_rejection('duplicate_retained_allowance_rejected',lambda:configure_retained_allowance(actual_charge))
        scope=fixture/'retained_owner';scope.mkdir()
        owner=aggregate.AggregateOwner(scope,execution_check=lambda:None,max_family_rss_bytes=c.HOST_CAP,max_seconds=60)
        try:
            assert type(owner) is worker.AggregateOwner
            snapshot=dict(owner._last,charged_bytes=aggregate.OPERATING-4096,supervisor_metadata_bytes=0)
            with patch.object(aggregate,'measure',lambda path:dict(snapshot)):
                require_rejection('prospective_write_respects_combined12GiB_limit',lambda:owner(4096))
            owner._last=dict(snapshot)
            require_rejection('stream_credit_respects_combined12GiB_limit',lambda:owner.reserve_stream_write(1))
            emergency=dict(snapshot,charged_bytes=aggregate.RESERVATION-1)
            with patch.object(aggregate,'measure',lambda path:dict(emergency)):
                require_rejection('emergency_write_respects_combined16GiB_limit',
                    lambda:owner.write_json(scope/'never_written.json',{},emergency=True))
            assert not (scope/'never_written.json').exists()
        finally:owner.close()
    require_rejection('nonpositive_remaining_operating_limit_rejected',lambda:configure_retained_allowance(c.OPERATING))
    RESULTS.append('original_exact_owner_type_and_positive_tightened_limits_preserved')


def main():
    fixture=BASE/'cpu_fixture_v3';fixture.mkdir()
    for name in ('contracts_v3.py','guardian_v3.py','worker_v3.py','retained_allowance_v1.py'):
        ast.parse((BASE/name).read_text(),filename=name)
    assert 'torch' not in sys.modules
    assert not c.AGGREGATE.exists(),'No live or prepared scientific aggregate was created'
    release_tests();family_tests();runtime_gate_tests(fixture);stage_wiring_tests(fixture);retained_allowance_tests(fixture)
    config,tools,review=c.native_bindings();assert len(tools)==7
    frozen=c.plan.validate_plan();assert frozen['registration']['initial_parent_count']==1
    assert not frozen['registration']['additional_parent_admitted']
    assert not c.AGGREGATE.exists()
    result=dict(schema='European_short_VAL_ROOT_runner_CPU_contract_tests_v3',passed=True,checks=RESULTS,
        tested_source_refs=[c.ref(BASE/name) for name in ('contracts_v3.py','guardian_v3.py','worker_v3.py','retained_allowance_v1.py','test_contracts_cpu_v3.py')],
        native_chain_review_ref=review,torch_imported=False,children_launched=False,native_tools_executed=False,
        generation_executed=False,model_or_scoring_executed=False,scientific_aggregate_created=False,
        fixtures='Tiny retained metadata files and mocked process tables only')
    output=BASE/'CONTRACTS_CPU_v3.json'
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(passed=True,checks=len(RESULTS),result_ref=c.ref(output)),sort_keys=True))

if __name__=='__main__':main()
