"""Canonical unchanged-launcher entry for inactive full native MLM successor."""
from pathlib import Path
import sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_MLM_full_source_v3"
def main(argv=None):
    sys.path.insert(0,str(ROOT/REL))
    import supervisor_v1
    return supervisor_v1.main(argv)
if __name__=="__main__":raise SystemExit(main())
