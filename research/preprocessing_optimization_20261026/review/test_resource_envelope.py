"""Integer-only pool scheduling assessment; no banks, models or large allocations."""
import concurrent.futures
import json
from pathlib import Path
import random
import threading
from arggnn.benchmarking.paper_campaign import runtime as r
from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.pool import MemoryGovernor, PipelinedCompactParents

GIB = 2**30
BASE = 'research/preprocessing_optimization_20261026/review'
BOUND = 'research/reviews/workflow_progress_diagnosis_20261018/treeloom/storage_bound_v2.json'
PROGRESS = 'experiments/results/paper_american_20261024/treeloom_fit/native/progress.jsonl'
LIVE = 'experiments/results/paper_american_20261024/treeloom_fit/native/training_live.json'


def main():
    bound = json.loads(r.path(BOUND).read_text())
    rows = [json.loads(line) for line in r.path(PROGRESS).read_text().splitlines()]
    measured = {row['parent_id']: row['compact_charged_bytes'] for row in rows if row.get('event') == 'compact_parent_ready'}
    bounds = {row['parent_id']:row['complete_compact_cache_upper_bound_bytes']+160000 for row in bound['parents']}
    assert len(bounds) == len(measured) == 20 and set(bounds) == set(measured)
    assert all(measured[parent] <= bounds[parent] for parent in bounds)
    peak = json.loads(r.path(LIVE).read_text())['peak_RSS_bytes']
    checks, scenarios = [], []
    assert sum(bounds.values()) > 320*GIB and sum(bounds.values()) < 352*GIB
    checks.append('pinned all20 bound refuses320GiB and fits352GiB')
    residual = peak - sum(measured.values())
    # This is a calibrated scenario using retained evidence, not an analytical
    # proof that future allocator or concurrent preparation peaks cannot grow.
    for maximum in (576,592):
        pool = None
        def rss():
            return residual + (pool.charged if pool is not None else 0)
        def prepare(parent,resource_check):
            return parent, {'parent_id':parent}
        def pack(parent,metadata,**kwargs):
            return dict(parent=parent,charged_bytes=measured[parent],read_peak_upper_bound_bytes=128*GIB)
        def restore(record,**kwargs):
            return record['parent'], {'parent_id':record['parent']}
        pool=PipelinedCompactParents(parent_bounds=bounds,input_refs={parent:{} for parent in bounds},
            external_arrays={},max_cache_bytes=352*GIB,max_rss_bytes=maximum*GIB,
            phase_reserve_bytes=128*GIB,rss_bytes=rss,resource_check=lambda stage:None,
            event=lambda *args:None,prepare=prepare,pack=pack,restore=restore)
        visits=[]
        try:
            for epoch in range(1,31):
                order=sorted(bounds);random.Random(70002+epoch).shuffle(order)
                for parent in order:
                    actual,_=pool.load(parent)
                    assert actual==parent
                    with pool.training_phase():visits.append(parent)
            receipt=pool.receipt()
            assert len(visits)==600 and receipt['parents']==20
            assert receipt['preparations']=={parent:1 for parent in bounds}
            assert receipt['restorations']=={parent:30 for parent in bounds}
            assert receipt['compact_charged_bytes']==sum(measured.values())
            assert receipt['evictions']==receipt['cold_rebuilds']==receipt['disk_cache_bytes']==0
            scenarios.append(dict(maximum_GiB=maximum,calibrated_peak_GiB=peak/GIB,
                no_active_admission_margin_GiB=maximum-128-peak/GIB,
                retained_parents=20,loads=600,preparations=20,
                note='Calibration to existing warm peak; not a future memory guarantee'))
        finally:pool.close()
    checks.append('both envelopes retain20 parents with600 loads and one preparation each in measured-peak integer scenario')
    for maximum in (576,592):
        governor=MemoryGovernor(maximum_bytes=maximum*GIB,rss_bytes=lambda:352*GIB,
            resource_check=lambda stage:None,reserve_bytes=128*GIB)
        entered=threading.Event();attempted=threading.Event()
        def second():
            attempted.set()
            with governor.phase('second'):entered.set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
            with governor.phase('first'):
                future=worker.submit(second)
                assert attempted.wait(2)
                assert not entered.wait(.05)
            future.result(timeout=2)
            assert entered.is_set()
        governor.close()
    checks.append('overlapping128GiB phases wait and serialize when reservations do not fit')
    for maximum,current,should_pass in ((576,449,False),(592,449,True),(592,465,False)):
        governor=MemoryGovernor(maximum_bytes=maximum*GIB,rss_bytes=lambda:current*GIB,
            resource_check=lambda stage:None,reserve_bytes=128*GIB)
        passed=False
        try:
            with governor.phase('no_active_phase'):passed=True
        except ValueError:
            assert not should_pass
        finally:governor.close()
        assert passed==should_pass
    checks.append('576 fails at449GiB with no active phase;592 admits449GiB and fails465GiB')
    report=dict(status='pass',schema='read_only_tree_resource_envelope_assessment_v1',checks=checks,
        source_refs=[r.ref(BOUND),r.ref(PROGRESS),r.ref(LIVE),r.ref(__file__),
            r.ref('research/repairs/TreeLoom_lazy_compact_prefetch_20261024_v1/pool.py')],
        all20_pinned_upper_bound_bytes=sum(bounds.values()),all20_historical_charged_bytes=sum(measured.values()),
        historical_warm_peak_RSS_bytes=peak,cache_required_GiB=352,scenarios=scenarios,
        model_executed=False,large_arrays_allocated=False,scheduler_changed=False,
        future_peak_proven=False,
        conclusion='352GiB cache is required by unchanged pinned bounds.576GiB is close to its no-active admission threshold;592GiB gives16GiB more worker margin if host admission remains available.')
    reference=r.save(BASE+'/resource_envelope_assessment.json',report)
    print(json.dumps(dict(report_ref=reference,checks=checks,scenarios=scenarios),indent=2))


if __name__=='__main__':main()
