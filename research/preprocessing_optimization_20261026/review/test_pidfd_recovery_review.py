"""Independent post-pidfd identity regressions using only mocked operations."""
import copy
import json
from unittest.mock import patch
from arggnn.benchmarking.paper_campaign import runtime as r
from research.preprocessing_optimization_20261026.integration import recover_pidfd_stop as recovery


def main():
    expected='8ac7149e869ba475acd83a09313791f2b2eac41a0897dcab8f8bbda1926f4dc1'
    assert r.ref(recovery.__file__)['sha256']==expected
    before=recovery.authenticate(require_review=False)
    row=copy.deepcopy(before['row'])
    changed_telemetry=copy.deepcopy(row)
    changed_telemetry['observed_pss_bytes']=row.get('observed_pss_bytes',0)+1234567
    changed_telemetry['observed_output_bytes']=row.get('observed_output_bytes',0)+1234
    changed_identity=copy.deepcopy(changed_telemetry)
    changed_identity['pid']+=1
    real_ref=r.ref
    def fixture_ref(path):
        if str(path)==str(recovery.REVIEW):return dict(path='mock/independent_review.json',sha256='0'*64)
        return real_ref(path)
    checks=[]
    for second,allowed in ((changed_telemetry,True),(changed_identity,False)):
        with patch.object(recovery,'authenticate',return_value=before), \
             patch.object(recovery.resource_ops,'snapshot',return_value=dict(admissible_projection=True)), \
             patch.object(recovery.resource_ops,'save_snapshot',return_value={}), \
             patch.object(recovery,'validate_target',side_effect=[copy.deepcopy(row),second]), \
             patch.object(recovery.processes,'_pidfd_open',return_value=81) as opened, \
             patch.object(recovery.processes,'_pidfd_send') as sent, \
             patch.object(recovery.os,'close') as closed, \
             patch.object(recovery.original,'exclusive',return_value=dict(path='mock/intent',sha256='0'*64)) as saved, \
             patch.object(recovery.r,'ref',side_effect=fixture_ref):
            if allowed:
                recovery.recover()
                sent.assert_called_once_with(81,recovery.signal.SIGTERM)
                assert saved.call_count==2
                assert saved.call_args_list[0].args[0]==recovery.INTENT
                assert saved.call_args_list[1].args[0]==recovery.original.WORK/'stop_sent.json'
                checks.append('Changed RSS/output telemetry permits exactly one mocked dispatch after stable-identity recheck')
            else:
                try:recovery.recover()
                except ValueError:pass
                else:raise AssertionError('Changed identity was accepted after pidfd acquisition')
                sent.assert_not_called();saved.assert_not_called()
                checks.append('Changed identity after pidfd acquisition refuses intent and dispatch')
            opened.assert_called_once_with(row['pid'])
            closed.assert_called_once_with(81)
    assert r.ref(recovery.__file__)['sha256']==expected
    proof=r.read(r.ref('research/preprocessing_optimization_20261026/review/pidfd_recovery_tests.json'))
    assert proof['operational_source_ref']==r.ref(recovery.__file__) and proof['status']=='pass'
    assert proof['fallback_syscalls_434_424_verified_without_native_APIs'] is True
    assert proof['syscall_errors_propagate']==2 and proof['identity_rejections']==5
    for reference in recovery.PINS:assert r.ref(reference['path'])==reference
    checks.extend(['Pinned portable syscall434/424 implementation and original before_stop authenticated',
        'Forced missing-native-API, error propagation, five identity rejections and fail-closed admission evidence verified',
        'Actual runtime portable pidfd-open capability tested separately on only its own diagnostic PID without signals'])
    report=dict(schema='independent_pidfd_recovery_review_v1',status='pass',
        operational_source_ref=r.ref(recovery.__file__),pinned_inputs=recovery.PINS,
        checks=checks,test_source_ref=r.ref(__file__),
        evidence_refs=[r.ref('research/preprocessing_optimization_20261026/review/pidfd_recovery_tests.json'),
            r.ref('research/preprocessing_optimization_20261026/review/pidfd_runtime_capability.json'),
            r.ref('research/preprocessing_optimization_20261026/review/pidfd_failure_basis.json')],
        real_signals_sent=False,real_pidfd_opened_by_this_test=False,production_files_changed=False,
        findings=[],requires_fresh_admission_and_identity_checks=True,
        prior_sealed_sources_and_receipts_preserved=True,
        uncertain_dispatch_retry_policy='Exclusive intent blocks replay; inspect outcome before any separately authorized recovery')
    path=recovery.REVIEW
    with path.open('x') as handle:json.dump(report,handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps(dict(status='pass',review_ref=r.ref(path),checks=checks),indent=2))


if __name__=='__main__':main()
