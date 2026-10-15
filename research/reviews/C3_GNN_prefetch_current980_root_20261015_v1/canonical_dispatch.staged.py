"""Canonical project dispatch for the reviewed prefetch current980 C3 GNN source packet."""
from pathlib import Path
import sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
def main(argv=None):
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts"),
        str(ROOT/"research/reviews/C3_GNN_prefetch_current980_root_20261015_v1")]
    from worker_entry_v1 import main as execute
    return execute(argv)
if __name__=="__main__":raise SystemExit(main())
