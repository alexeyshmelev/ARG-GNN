"""Exact v6 execution-amendment entry. No accounting admission."""
from pathlib import Path
import sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_MLM_amendment_v7"
def main(argv=None):
    sys.path[:0]=[str(ROOT/REL),str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT)]
    import amendment_v7 as amendment
    amendment.check_lifetime()
    import supervisor_v1
    return supervisor_v1.main(argv)
if __name__=="__main__":raise SystemExit(main())
