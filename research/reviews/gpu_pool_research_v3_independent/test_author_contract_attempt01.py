import ast,json
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
def test_inactive_contract_kind_matches_frozen_bootstrap():
    tree=ast.parse((ROOT/'src/arggnn/anchor_compact_cuda_bootstrap_v2.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='admit_precontext')
    kinds=[n.value for n in ast.walk(fn) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('root_compact_cuda_bootstrap_') and n.value.endswith('_permit')]
    assert kinds==['root_compact_cuda_bootstrap_v1_permit']
    contract=json.loads((Path(__file__).parent/'author_final_attempt01/minimal_actual_permit_contract.json').read_text())
    assert contract['required_root_permit_kind']==kinds[0]
