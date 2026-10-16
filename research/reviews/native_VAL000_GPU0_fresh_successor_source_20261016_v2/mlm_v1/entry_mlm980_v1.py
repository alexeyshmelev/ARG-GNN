"""Fresh VAL000 native MLM980 entry; no import-time authority/state action."""
from pathlib import Path
import sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
def main(argv=None):
    sys.path[:0]=[str(ROOT/'research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v2/mlm_v1'),
        str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),
        str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),
        str(ROOT/"research/reviews/American_C3_native_MLM_amendment_v12"),
        str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT)]
    import supervisor_v1
    return supervisor_v1.main(argv)
if __name__=="__main__":raise SystemExit(main())
