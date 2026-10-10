"""Independent narrow-source and handmade equivalence audit. Writes only own receipt."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
import arggnn.anchor_compact_actual_bank_v1 as current
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OWN=ROOT/'research/reviews/anchor_compact_actual_bank_v1_independent'
AUTHOR=ROOT/'research/reviews/anchor_compact_actual_bank_v1_author'
BEFORE=AUTHOR/'storage/revision_v2/before'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old_path=BEFORE/'src/arggnn/anchor_compact_actual_bank_v1.py'
assert sha(old_path)=='5830a3471ce3e7145db1731a71ca58b3fa13161f6855fe805fb2e29d9827c33f'
assert sha(Path(current.__file__))=='b3e88afc1d36a7b4b8f1d3eebef68a4c034a719b32b722751a19e040fbe33ec3'
assert sha(OWN/'review_receipt.json')=='38d63aa9bce0a73dc4a157ec396cebdb6f8e8dd640c5fa1dbc0d1a8b7937f6c2'
archive=json.loads((AUTHOR/'storage/revision_v2/before_archive_manifest.json').read_text())
assert all(sha(ROOT/x['archive_path'])==x['sha256'] for x in archive['records'].values())

old_tree=ast.parse(old_path.read_text())
new_tree=ast.parse(Path(current.__file__).read_text())
def find(tree,name):return next(n for n in tree.body if getattr(n,'name',None)==name)
def dump(tree):return ast.dump(tree,include_attributes=False)
unchanged=[]
changed=[]
for item in old_tree.body:
    if isinstance(item,(ast.FunctionDef,ast.ClassDef)):
        if dump(item)==dump(find(new_tree,item.name)):unchanged.append(item.name)
        else:changed.append(item.name)
assert set(changed)=={'qualification_models_v1','compare_model_step_v1','compare_action_VJPs_v1'},changed
# For direct action, replace checked local identity values with old attribute access,
# discard the prework guard and metadata-only receipt key, then compare full AST.
class StripIdentity(ast.NodeTransformer):
    def visit_Dict(self,n):
        n=self.generic_visit(n)
        pairs=[(k,v) for k,v in zip(n.keys,n.values) if not isinstance(k,ast.Constant) or k.value!='cell_identity']
        n.keys=[x[0] for x in pairs];n.values=[x[1] for x in pairs]
        return n
    def visit_Call(self,n):
        n=self.generic_visit(n)
        n.keywords=[kw for kw in n.keywords if kw.arg!='cell_identity']
        return n
class LocalToCheckedModel(StripIdentity):
    def visit_Name(self,n):
        if n.id in {'geometry','coupling'} and isinstance(n.ctx,ast.Load):
            return ast.Attribute(value=ast.Name(id='model',ctx=ast.Load()),attr=n.id,ctx=ast.Load())
        return n
new_action=copy.deepcopy(find(new_tree,'compare_action_VJPs_v1'))
assert isinstance(new_action.body[0],ast.Expr)
new_action.body=new_action.body[:1]+new_action.body[4:]
new_action=LocalToCheckedModel().visit(new_action)
assert dump(new_action)==dump(find(old_tree,'compare_action_VJPs_v1'))
new_step=copy.deepcopy(find(new_tree,'compare_model_step_v1'))
new_step.body=new_step.body[:1]+new_step.body[4:]
new_step=StripIdentity().visit(new_step)
assert dump(new_step)==dump(find(old_tree,'compare_model_step_v1'))
old_cfg=json.loads((BEFORE/'configs/anchor_compact_actual_bank_v1.json').read_text())
new_cfg=json.loads((ROOT/'configs/anchor_compact_actual_bank_v1.json').read_text())
assert old_cfg['numerical_cells']['tolerances']==new_cfg['numerical_cells']['tolerances']
for key in old_cfg['numerical_cells']:
    if key!='controls':assert old_cfg['numerical_cells'][key]==new_cfg['numerical_cells'][key]
for key,value in old_cfg['numerical_cells']['controls'].items():assert new_cfg['numerical_cells']['controls'][key]==value
assert old_cfg['profile']==new_cfg['profile']
assert old_cfg['gpu']==new_cfg['gpu'] and old_cfg['storage']==new_cfg['storage']
assert all(sha(ROOT/p)==expected for p,expected in new_cfg['source_sha256'].items())

old_metrics=json.loads((OWN/'independent_metrics.json').read_text())
new_metrics=json.loads((OWN/'independent_metrics_v2.json').read_text())
num_kinds={'direct_dense_oracle_small_full_model','candidate_sized_handmade'}
old_numeric=[x for x in old_metrics if x['kind'] in num_kinds]
new_numeric=[x for x in new_metrics if x['kind'] in num_kinds]
def remove_cell_identity(value):
    if isinstance(value,dict):return {k:remove_cell_identity(v) for k,v in value.items() if k!='cell_identity'}
    if isinstance(value,list):return [remove_cell_identity(x) for x in value]
    return value
assert old_numeric==remove_cell_identity(new_numeric)
assert len(old_numeric)==14
old_author=BEFORE/'research/reviews/anchor_compact_actual_bank_v1_author/handmade_parity_metrics.json'
assert old_author.read_bytes()==(AUTHOR/'handmade_parity_metrics.json').read_bytes()

# Raw bitwise action replay uses independent fixtures, not author fixtures.
name='arggnn._independent_legacy_actual_adapter_v2_audit'
spec=importlib.util.spec_from_file_location(name,old_path)
legacy=importlib.util.module_from_spec(spec);sys.modules[name]=legacy;spec.loader.exec_module(legacy)
spec=importlib.util.spec_from_file_location('independent_v2_handmade_fixture',OWN/'test_independent_v2.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
torch.set_num_threads(1)
def capture(module,model,bank,view):
    arrays={}
    action_native=module.anchor_kernel_action;action_compact=module.anchor_spectral_compact_action_v1
    def native(*a,**kw):
        result=action_native(*a,**kw);arrays['native']=result.detach().clone();return result
    def compact(*a,**kw):
        result=action_compact(*a,**kw);arrays['compact']=result.detach().clone();return result
    module.anchor_kernel_action=native;module.anchor_spectral_compact_action_v1=compact
    try:result=module.compare_action_VJPs_v1(model,bank,view)
    finally:module.anchor_kernel_action=action_native;module.anchor_spectral_compact_action_v1=action_compact
    return result,arrays
raw_cells=[]
for cell in new_numeric:
    dtype={'torch.float64':torch.float64,'torch.float32':torch.float32}[cell['dtype']]
    if cell['kind']=='candidate_sized_handmade':
        bank,_=fixture.handmade(184,96,47,32)
        native,_,bank,view=fixture.pair(bank,dtype,geometry=cell['geometry'],stress=cell['stress'],candidate=True)
    else:
        bank,_=fixture.handmade()
        native,_,bank,view=fixture.pair(bank,dtype,coupling=cell['coupling'],stress=True)
    before,before_raw=capture(legacy,native,bank,view)
    after,after_raw=capture(current,native,bank,view)
    assert before==remove_cell_identity(after)
    assert all(torch.equal(before_raw[k],after_raw[k]) for k in before_raw)
    raw_cells.append(dict(kind=cell['kind'],dtype=cell['dtype'],geometry=cell.get('geometry','age'),
        coupling=cell.get('coupling','full'),stress=cell.get('stress',True),
        native_and_compact_raw_action_bitwise_equal_before_after=True,action_numeric_receipts_exact=True))
assert not torch.cuda.is_initialized()
receipt=dict(kind='independent_compact_actual_narrow_v2_equivalence',archived_source_sha256=sha(old_path),
    revised_source_sha256=sha(Path(current.__file__)),initial_FAIL_sha256=sha(OWN/'review_receipt.json'),
    archived_records_all_exact=True,archived_records_count=len(archive['records']),
    unchanged_definition_AST=unchanged,changed_definition_names=changed,
    action_math_AST_equal_after_removing_prework_guard_and_receipt_metadata=True,
    full_step_math_AST_equal_after_removing_prework_guard_and_receipt_metadata=True,
    frozen_numerical_tolerances_and_existing_protocol_cells_unchanged=True,
    profile_gpu_storage_protocol_unchanged=True,all_current_protocol_source_hashes_match=True,
    independent_prior_14_numeric_cells_exact_receipts_after_discarding_new_identity=True,
    independent_raw_action_replays=raw_cells,independent_raw_replays_count=len(raw_cells),
    author_original_16_cell_metric_JSON_still_byte_identical=True,author_metric_JSON_sha256=sha(old_author),
    author_74_tests_are_acknowledged_not_independent_verdict=True,
    actual_arrays_opened=False,CUDA_context_initialized=False)
(OWN/'source_and_numerical_equivalence_v2.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['independent_raw_action_replays','unchanged_definition_AST']},indent=2))
