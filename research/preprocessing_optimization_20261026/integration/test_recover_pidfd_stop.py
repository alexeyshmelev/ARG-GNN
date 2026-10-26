"""Bounded recovery tests with fake libc syscalls, never real signals or pidfds."""
from types import SimpleNamespace
from unittest.mock import patch
import copy
import errno
import json
import signal
from arggnn.benchmarking.paper_campaign import processes, runtime as r
from . import recover_pidfd_stop as recovery


def main():
    calls=[]
    def syscall(*args):
        calls.append(args)
        return 81 if args[0]==434 else 0
    with patch.object(processes,'os',SimpleNamespace(strerror=lambda code:'mock operating-system error')), patch.object(processes,'signal',SimpleNamespace()), \
         patch.object(processes.platform,'machine',return_value='x86_64'), \
         patch.object(processes.ctypes,'CDLL',return_value=SimpleNamespace(syscall=syscall)):
        fd=processes._pidfd_open(123456)
        processes._pidfd_send(fd,signal.SIGTERM)
    assert calls==[(434,123456,0),(424,81,signal.SIGTERM,0,0)]
    failed=0
    with patch.object(processes,'os',SimpleNamespace(strerror=lambda code:'mock operating-system error')), patch.object(processes,'signal',SimpleNamespace()), \
         patch.object(processes.platform,'machine',return_value='x86_64'), \
         patch.object(processes.ctypes,'CDLL',return_value=SimpleNamespace(syscall=lambda *args:-1)), \
         patch.object(processes.ctypes,'get_errno',return_value=errno.EPERM):
        for operation in [lambda:processes._pidfd_open(123456),lambda:processes._pidfd_send(81,signal.SIGTERM)]:
            try:operation()
            except OSError as error:
                assert error.errno==errno.EPERM
                failed+=1
            else:raise AssertionError('Syscall error ignored')
    before=r.read(recovery.PINS[2])
    task=copy.deepcopy(before['task'])
    state=dict(plan_ref=before['plan_ref'],plan_revision=before['plan_revision'])
    row=copy.deepcopy(before['row'])
    recovery.check_saved_identity(before,state,task,row)
    negatives=0
    for field,value in [('pid',row['pid']+1),('process_start_ticks','0'),('attempt',row['attempt']+1),
                        ('physical_gpu',0),('status','succeeded')]:
        wrong=copy.deepcopy(row)
        wrong[field]=value
        try:recovery.check_saved_identity(before,state,task,wrong)
        except ValueError:negatives+=1
        else:raise AssertionError(field)
    with patch.object(recovery,'authenticate',return_value=before), \
         patch.object(recovery.resource_ops,'snapshot',return_value={'admissible_projection':False}), \
         patch.object(recovery.resource_ops,'save_snapshot',return_value={}), \
         patch.object(recovery.processes,'_pidfd_open') as opened, \
         patch.object(recovery.processes,'_pidfd_send') as sent:
        try:recovery.recover()
        except ValueError:pass
        else:raise AssertionError('Refused RAM admission ignored')
        opened.assert_not_called()
        sent.assert_not_called()
    with patch.object(recovery,'authenticate',return_value=before), \
         patch.object(recovery.resource_ops,'snapshot',return_value={'admissible_projection':True}), \
         patch.object(recovery.resource_ops,'save_snapshot',return_value={}), \
         patch.object(recovery,'validate_target',side_effect=ValueError('Changed identity')), \
         patch.object(recovery.processes,'_pidfd_open') as opened, \
         patch.object(recovery.processes,'_pidfd_send') as sent:
        try:recovery.recover()
        except ValueError:pass
        else:raise AssertionError('Changed identity ignored')
        opened.assert_not_called()
        sent.assert_not_called()
    for ref in recovery.PINS:
        assert r.ref(ref['path'])==ref
    result=dict(status='pass',operational_source_ref=r.ref(recovery.__file__),test_source_ref=r.ref(__file__),
        pinned_inputs=recovery.PINS,fallback_syscalls_434_424_verified_without_native_APIs=True,
        syscall_errors_propagate=failed,identity_rejections=negatives,
        negative_admission_and_identity_perform_no_pidfd_or_signal=True,
        model_executed=False,real_pidfd_opened=False,real_signal_sent=False,production_files_changed=False)
    path=r.ROOT/'research/preprocessing_optimization_20261026/review/pidfd_recovery_tests.json'
    with path.open('x') as handle:
        json.dump(result,handle,indent=2)
        handle.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
