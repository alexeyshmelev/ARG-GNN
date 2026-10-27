"""Bounded final-monitor-race reproduction; no simulation or native operation."""
import hashlib
import json
from pathlib import Path
import threading
from unittest.mock import patch
from research.efficiency_20261027.execution.european_full_val_runner import aggregate_guard_v1 as guard

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/efficiency_20261027/execution'

def ref(path):return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    scope=OUT/'fullval_guard_close_v1_fixture';scope.mkdir()
    owner=guard.AggregateOwner(scope,execution_check=lambda:None,max_family_rss_bytes=2**30,max_seconds=10)
    entered=threading.Event();release=threading.Event();original_refresh=owner.refresh
    def delayed_failure():
        if threading.current_thread().name=='full-validation-storage':
            entered.set()
            assert release.wait(1.)
            raise guard.GuardFailure('Synthetic resource failure after final foreground check')
        return original_refresh()
    closed_without_error=False
    with patch.object(guard,'SCAN_INTERVAL',.01),patch.object(owner,'refresh',side_effect=delayed_failure):
        owner.start_monitor()
        assert entered.wait(1.)
        owner.check()
        timer=threading.Timer(.02,release.set);timer.start()
        try:
            owner.close()
            closed_without_error=True
        finally:
            release.set();timer.join(timeout=1.)
    assert closed_without_error and owner._error is not None and not owner._thread.is_alive()
    report=dict(schema='European_full_VAL_guard_close_review_v1',passed=False,
        source_manifest_ref=ref(OUT/'european_full_val_runner/source_pins_v1.json'),
        source_refs=[ref(Path(guard.__file__)),ref(OUT/'european_full_val_runner/worker_v1.py'),ref(Path(__file__))],
        deterministic_bounded_thread_reproduction=True,
        findings=[dict(priority='required_before_launch',
            detail='A monitor scan fails after the final foreground check while close waits for it. close returns successfully despite owner._error being set, so worker can exit0 and guardian can mark finished.',
            required_fix='After stopping and joining the monitor, propagate any stored monitor error before permitting successful close. Preserve lease cleanup using finally. Add a deterministic late-failure test and a clean-close test.')],
        monitor_failure=owner._error,close_returned_without_error=closed_without_error,
        scientific_payloads_opened=False,native_tool_executions=0,model_or_GPU_calls=0,
        existing_jobs_modified=False)
    destination=OUT/'FULL_VAL_V1_GUARD_CLOSE_REVIEW.json'
    with destination.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(review=ref(destination),passed=False)))

if __name__=='__main__':main()
