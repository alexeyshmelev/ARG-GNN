import ast,json
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
def test_inactive_contract_kind_matches_frozen_bootstrap():
    tree=ast.parse((ROOT/'src/arggnn/anchor_compact_cuda_bootstrap_v2.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='admit_precontext')
    kinds=[n.value for n in ast.walk(fn) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('root_compact_cuda_bootstrap_') and n.value.endswith('_permit')]
    assert kinds==['root_compact_cuda_bootstrap_v1_permit']
    contract=json.loads((Path(__file__).parent/'author_final_attempt02/minimal_actual_permit_contract.json').read_text())
    assert contract['required_root_permit_kind']==kinds[0]

def test_final_documentation_kind_matches_inherited_schema():
    import re
    text=(Path(__file__).parent/'sealed_source_final/docs/anchor_compact_cuda_bootstrap_v2.md').read_text()
    kinds=re.findall(r'root_compact_cuda_bootstrap_v\d+_permit',text)
    assert kinds and set(kinds)=={'root_compact_cuda_bootstrap_v1_permit'}

def test_fixed_contract_is_inactive_and_matches_all_core_pins():
    contract=json.loads((Path(__file__).parent/'author_final_attempt02/minimal_actual_permit_contract.json').read_text())
    assert contract['actual_execution_admitted'] is False
    assert contract['bootstrap_source']['sha256']=='d6f1504901b926d45a59664fcc9d211892eb087586aca1a53d34256e05db1fe5'
    assert contract['bootstrap_config']['sha256']=='bf94468e5b8d613e027c1d46fbb7991a672870d5a6f0159ec9511af1d8285dac'
    assert contract['pool_extension']['source']['sha256']=='10176c2b2580aa0e27b46aabe3ceadf88981d5e279408277d70e830675045504'
    assert contract['pool_extension']['config']['sha256']=='e67fce8bbd59042f3deba8bbf434c9d8759f1f9d26781df74eccf84b432e29c7'
