import json,re
from pathlib import Path
D=Path(__file__).resolve().parent

def test_all_initial_documented_permit_kinds_match_frozen_source():
    text=(D/'sealed_source/docs/anchor_compact_cuda_bootstrap_v2.md').read_text()
    kinds=re.findall(r'root_compact_cuda_bootstrap_v\d+_permit',text)
    assert kinds and set(kinds)=={'root_compact_cuda_bootstrap_v1_permit'}
