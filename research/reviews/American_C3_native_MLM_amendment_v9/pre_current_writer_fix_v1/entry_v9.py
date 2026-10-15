"""Truthful v9 execution entry; original canonical v8 remains frozen."""
from pathlib import Path
import sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_native_MLM_amendment_v9"
def main(argv=None):
    sys.path[:0]=[str(ROOT/REL),str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT)]
    import amendment_v9 as amendment
    amendment.check_lifetime()
    import supervisor_v1
    return supervisor_v1.main(argv)
if __name__=="__main__":raise SystemExit(main())
