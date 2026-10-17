"""Inactive common worker entry; delegates only after exact admitted successor selection."""
import argparse,json
from . import root_driver_v1 as d
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--worker-ref");p.add_argument("--gate-fd",type=int);a=p.parse_args(argv)
 if not a.worker_ref:
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",models_imported=False,guard_imported=False)));return 0
 d.need(a.gate_fd is not None,"Original actual owned startup gate required")
 completed=d.worker(json.loads(a.worker_ref),a.gate_fd)
 print(json.dumps(dict(completed_ref=completed)),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
