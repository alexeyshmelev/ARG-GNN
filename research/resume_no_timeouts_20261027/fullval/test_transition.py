"""Deterministic admission-transition tests; no workers or scientific work."""
import errno
import json
from pathlib import Path
from unittest.mock import patch
from . import guardian as g, contracts as c
from research.efficiency_20261027.root import run_guarded_profile_with_peer as shared

HERE=Path(__file__).resolve().parent

class Clock:
    def __init__(self,on_sleep=None):
        self.now=0.0
        self.sleeps=[]
        self.on_sleep=on_sleep
    def monotonic(self):return self.now
    def sleep(self,seconds):
        self.sleeps.append(seconds)
        self.now+=seconds
        if self.on_sleep:self.on_sleep()

def run():
    base=HERE/'transition_fixtures_v2'
    base.mkdir()
    checks=[]
    canonical=base/'state.json'
    plan=base/'plan.json'
    plan.write_text(json.dumps(dict(existing_workers=[],records_directory='unused')))
    state=dict(updated_unix=g.time.time(),project_reserved_bytes=123,
        plan_ref=dict(path=str(plan.relative_to(c.ROOT))),
        tasks={'ordinary_worker':dict(status='running',pid=2147483647,process_start_ticks='not-live')})
    canonical.write_text(json.dumps(state))
    def publish_completion():
        state['tasks']['ordinary_worker']['status']='succeeded'
        canonical.write_text(json.dumps(state))
    clock=Clock(publish_completion)
    with patch.object(g,'AGGREGATE',base),patch.object(shared,'STATE',canonical), \
         patch.object(shared,'identity',return_value=None), \
         patch.object(g,'current_memory',return_value=g.HOST_MARGIN) as memory, \
         patch.object(g.time,'monotonic',clock.monotonic),patch.object(g.time,'sleep',clock.sleep):
        result=g.campaign_growth_stable()
    assert result[0]['tasks']['ordinary_worker']['status']=='succeeded'
    assert result[1:3]==(0,[]) and result[3]['campaign_transition_retries']==1
    assert len(clock.sleeps)==1 and memory.call_count==4
    checks.append(dict(name='real_shared_helper_rereads_canonical_completion_after_exact_transition',sleeps=clock.sleeps))

    clock=Clock()
    with patch.object(g,'AGGREGATE',base),patch.object(g,'current_memory',return_value=g.HOST_MARGIN), \
         patch.object(g,'campaign_growth',side_effect=RuntimeError(g.CAMPAIGN_TRANSITION_ERROR)) as growth, \
         patch.object(g.time,'monotonic',clock.monotonic),patch.object(g.time,'sleep',clock.sleep):
        try:g.campaign_growth_stable()
        except ValueError as error:assert '60-second' in str(error)
        else:raise AssertionError('Unresolved campaign transition accepted')
    assert clock.now==60 and growth.call_count==60
    checks.append(dict(name='persistent_transition_fails_at_snapshot_freshness_bound',elapsed=clock.now,scans=growth.call_count))

    failures=[RuntimeError('Campaign snapshot is stale'),RuntimeError('Different failure'),
              OSError(errno.EIO,'true I/O failure'),FileNotFoundError(errno.ENOENT,'missing state')]
    class Derived(RuntimeError):pass
    failures.append(Derived(g.CAMPAIGN_TRANSITION_ERROR))
    for expected in failures:
        clock=Clock()
        with patch.object(g,'AGGREGATE',base),patch.object(g,'current_memory',return_value=g.HOST_MARGIN), \
             patch.object(g,'campaign_growth',side_effect=expected) as growth, \
             patch.object(g.time,'monotonic',clock.monotonic),patch.object(g.time,'sleep',clock.sleep):
            try:g.campaign_growth_stable()
            except BaseException as error:assert error is expected
            else:raise AssertionError('Non-transition error was swallowed')
        assert not clock.sleeps and growth.call_count==1
    checks.append(dict(name='genuine_IO_stale_state_other_runtime_and_subclass_failures_propagate_immediately',failure_cases=len(failures)))

    hold=base/'execution_hold'
    clock=Clock(lambda:hold.write_text('test hold\n'))
    with patch.object(g,'AGGREGATE',base),patch.object(g,'current_memory',return_value=g.HOST_MARGIN), \
         patch.object(g,'campaign_growth',side_effect=RuntimeError(g.CAMPAIGN_TRANSITION_ERROR)) as growth, \
         patch.object(g.time,'monotonic',clock.monotonic),patch.object(g.time,'sleep',clock.sleep):
        try:g.campaign_growth_stable()
        except ValueError as error:assert 'hold' in str(error)
        else:raise AssertionError('Hold ignored while retrying')
    assert growth.call_count==1
    hold.rename(base/'observed_execution_hold')
    checks.append(dict(name='execution_hold_interrupts_retry_before_another_campaign_read'))

    clock=Clock()
    with patch.object(g,'AGGREGATE',base),patch.object(g,'current_memory',side_effect=[g.HOST_MARGIN,g.HOST_MARGIN,g.HOST_MARGIN-1]), \
         patch.object(g,'campaign_growth',side_effect=RuntimeError(g.CAMPAIGN_TRANSITION_ERROR)) as growth, \
         patch.object(g.time,'monotonic',clock.monotonic),patch.object(g.time,'sleep',clock.sleep):
        try:g.campaign_growth_stable()
        except ValueError as error:assert 'Host memory pressure' in str(error)
        else:raise AssertionError('Host margin ignored while retrying')
    assert growth.call_count==1
    checks.append(dict(name='host_margin_interrupts_retry_before_another_campaign_read'))

    clock=Clock()
    def slow_success():
        clock.now=61
        return {},0,[]
    with patch.object(g,'AGGREGATE',base),patch.object(g,'current_memory',return_value=g.HOST_MARGIN), \
         patch.object(g,'campaign_growth',side_effect=slow_success), \
         patch.object(g.time,'monotonic',clock.monotonic),patch.object(g.time,'sleep',clock.sleep):
        try:g.campaign_growth_stable()
        except ValueError as error:assert 'snapshot freshness' in str(error)
        else:raise AssertionError('Slow stale successful scan accepted')
    assert g.ROOT_STATUS_MAX_AGE_SECONDS==120 and g.CAMPAIGN_TRANSITION_MAX_SECONDS==60
    checks.append(dict(name='slow_success_rejected_and_120_second_worker_status_freshness_unchanged'))
    return checks

def main():
    checks=run()
    path=HERE/'TRANSITION_TESTS.json'
    path.write_text(json.dumps(dict(passed=True,check_count=len(checks),checks=checks,
        original_shared_helper_unchanged=True,no_workload_launch=True,
        source_refs=[c.ref(HERE/name) for name in ('guardian.py','contracts.py','test_transition.py')]
            +[c.ref(Path(shared.__file__))]),indent=2)+'\n')
    print(json.dumps(dict(passed=True,check_count=len(checks),receipt=c.ref(path))))

if __name__=='__main__':main()
