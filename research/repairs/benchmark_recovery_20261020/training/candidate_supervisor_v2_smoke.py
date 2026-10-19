"""One CPU-only invocation verifies real success and owned termination waits."""
from pathlib import Path
import os
import signal
import sys
from .candidate_common import ROOT, need, file_ref, write_json
from .candidate_supervisor_v2 import run_owned


def main():
    base = Path(__file__).parent
    child = base/'candidate_supervisor_smoke_child.py'
    command = [sys.executable,'-B',str(child),'--sleep-seconds']
    def pass_check(observed,elapsed):
        need(elapsed < 5., 'Synthetic success child wall cap')
    success = run_owned(command+['0.1'],base/'candidate_supervisor_v2_smoke.success.log',pass_check,.02)
    def controlled_stop(observed,elapsed):
        need(elapsed < .2, 'Synthetic controlled stop')
    stopped = run_owned(command+['10'],base/'candidate_supervisor_v2_smoke.stopped.log',controlled_stop,.02)
    need(success['actual_exit_code'] == 0 and success['resource_violation'] is None
         and success['owned_child_absent'] is True, 'Actual successful wait and absence required')
    need(stopped['actual_exit_code'] == -signal.SIGTERM and stopped['owned_child_absent'] is True
         and stopped['resource_violation'] == 'ValueError: Synthetic controlled stop',
         'Actual controlled owned termination wait and absence required')
    need('torch' not in sys.modules, 'CPU lifecycle smoke must not import Torch')
    receipt = dict(schema='treeloom_supervisor_v2_CPU_smoke_v1',status='passed',
        pidfd_open_available=hasattr(os,'pidfd_open'),
        pidfd_send_signal_available=hasattr(signal,'pidfd_send_signal'),
        success_child=success,controlled_stop_child=stopped,CUDA_imported=False,
        source_refs=[file_ref(base/name) for name in (
            'candidate_supervisor_v2.py','candidate_supervisor_v2_smoke.py','candidate_supervisor_smoke_child.py')])
    write_json(base/'candidate_supervisor_v2_smoke.result.json',receipt)
    print(receipt,flush=True)


if __name__ == '__main__':
    main()
