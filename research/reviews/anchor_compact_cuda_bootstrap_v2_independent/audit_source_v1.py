"""Independent frozen-byte and mechanical AST/config comparison; no scientific import."""
import ast,copy,hashlib,json
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BOOT=ROOT/'research/reviews/anchor_compact_cuda_bootstrap_v2_independent'
EXT=ROOT/'research/reviews/gpu_pool_research_v3_independent'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads((ROOT/path).read_text())
def dump(node):return ast.dump(node,include_attributes=False)
def tree(path):return ast.parse((ROOT/path).read_text())
bc1=read('configs/anchor_compact_cuda_bootstrap_v1.json');bc2=read('configs/anchor_compact_cuda_bootstrap_v2.json')
pc2=read('configs/gpu_pool_research_v2.json');pc3=read('configs/gpu_pool_research_v3.json')
version_map=[('anchor_compact_cuda_bootstrap_v2','anchor_compact_cuda_bootstrap_v1'),('anchor_compact_cuda_bootstrap_protocol_v2','anchor_compact_cuda_bootstrap_protocol_v1'),('gpu_pool_research_v3','gpu_pool_research_v2')]
hash_map=[(sha(ROOT/'configs/anchor_compact_cuda_bootstrap_v2.json'),sha(ROOT/'configs/anchor_compact_cuda_bootstrap_v1.json')),(sha(ROOT/'configs/gpu_pool_research_v3.json'),sha(ROOT/'configs/gpu_pool_research_v2.json')),(sha(ROOT/'src/arggnn/anchor_compact_cuda_bootstrap_v2.py'),sha(ROOT/'src/arggnn/anchor_compact_cuda_bootstrap_v1.py'))]
class Normalize(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            for new,old in version_map+hash_map:node.value=node.value.replace(new,old)
        return node
old=tree('src/arggnn/anchor_compact_cuda_bootstrap_v1.py');new=tree('src/arggnn/anchor_compact_cuda_bootstrap_v2.py')
normalized=Normalize().visit(copy.deepcopy(new))
old_functions={n.name:dump(n) for n in old.body if isinstance(n,ast.FunctionDef)}
new_functions={n.name:dump(n) for n in normalized.body if isinstance(n,ast.FunctionDef)}
assert set(old_functions)==set(new_functions)
changed=[n for n in old_functions if old_functions[n]!=new_functions[n]]
assert changed==['bound_bytes'],changed
new_constants={'DEFAULT_BOUND_FILE_LIMIT_BYTES','METADATA_MANIFEST_LIMIT_BYTES','METADATA_MANIFEST_SHA256'}
normalized.body=[n for n in normalized.body if not(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in new_constants for t in n.targets))]
old_bound=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='bound_bytes')
normalized.body=[copy.deepcopy(old_bound) if isinstance(n,ast.FunctionDef) and n.name=='bound_bytes' else n for n in normalized.body]
assert dump(normalized)==dump(old),'Unexpected non-bound helper/module change'
old_pool=tree('scripts/gpu_pool_research_v2.py');new_pool=Normalize().visit(tree('scripts/gpu_pool_research_v3.py'))
assert dump(old_pool)==dump(new_pool),'Nonidentity pool source change'
def norm_value(value):
    if isinstance(value,dict):return {norm_value(k):norm_value(v) for k,v in value.items()}
    if isinstance(value,list):return [norm_value(v) for v in value]
    if isinstance(value,str):
        for new,old in version_map+hash_map:value=value.replace(new,old)
    return value
bc2_old=norm_value(copy.deepcopy(bc2))
added=set(bc2_old)-set(bc1)
assert added=={'default_bound_file_limit_bytes','metadata_manifest_limit_bytes','metadata_manifest_sha256'},added
for k in added:bc2_old.pop(k)
assert bc2_old==bc1,'Unexpected bootstrap config drift'
assert norm_value(pc3)==pc2,'Unexpected pool config drift'
assert bc2['source_sha256']==bc1['source_sha256']
checks={p:{'expected':h,'actual':sha(ROOT/p)} for p,h in bc2['source_sha256'].items()}
assert all(v['expected']==v['actual'] for v in checks.values())
preserved=read('research/reviews/anchor_compact_cuda_bootstrap_v2_author/preserved_source_baseline.json')
old_checks={p:{'expected':h,'actual':sha(ROOT/p)} for p,h in preserved.items()}
assert all(v['expected']==v['actual'] for v in old_checks.values())
prior=read('research/reviews/anchor_compact_cuda_bootstrap_v1_independent/preserved_dependency_audit_final_v1.json')
assert all(preserved[p]==v['expected'] for p,v in prior['checks'].items())
assigned={p:sha(ROOT/p) for p in ['src/arggnn/anchor_compact_cuda_bootstrap_v2.py','configs/anchor_compact_cuda_bootstrap_v2.json','scripts/gpu_pool_research_v3.py','configs/gpu_pool_research_v3.json']}
metadata=read('configs/anchor_compact_actual_bank_v1.json')
expected_pins={metadata[k]['path']:metadata[k]['sha256'] for k in ['prepared_manifest','observed_manifest']}
assert bc2['metadata_manifest_sha256']==expected_pins
assert bc2['default_bound_file_limit_bytes']==1048576 and bc2['metadata_manifest_limit_bytes']==67108864
# Module-scope imports are all unchanged inert stdlib + source guard modules.
imports=[dump(n) for n in new.body if isinstance(n,(ast.Import,ast.ImportFrom))]
assert imports==[dump(n) for n in old.body if isinstance(n,(ast.Import,ast.ImportFrom))]
contracts={'assigned_sha256':assigned,'bootstrap_helper_count':len(old_functions),'unchanged_bootstrap_helpers':[n for n in old_functions if n!='bound_bytes'],'only_behavioral_helper_change':'bound_bytes','bootstrap_full_module_AST_equal_after_explicit_bound_change_and_identity_normalization':True,'extension_full_module_AST_equal_after_identity_normalization':True,'normalization_exact_strings':version_map+hash_map,'bootstrap_config_added_keys_only':sorted(added),'bootstrap_config_all_other_values_unchanged':True,'pool_config_all_other_values_unchanged':True,'unchanged_frozen_protocol_source_map_count':len(checks),'unchanged_frozen_protocol_source_map':checks,'old_baseline_count':len(old_checks),'all_old_baseline_artifacts_unchanged':True,'old_baseline_checks':old_checks,'all_32_prior_independent_frozen_pins_identical':True,'metadata_pins_equal_frozen_adapter':expected_pins,'numeric_model_math_tolerance_dependencies_unchanged':True,'actual_execution_admitted':False,'static_nvtop_hash_only_no_execution':True,'actual_metadata_or_arrays_read':False,'live_state_or_ledger_read':False}
for d in (BOOT,EXT):
    (d/'source_audit_v1.json').write_text(json.dumps(contracts,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:contracts[k] for k in ['bootstrap_helper_count','only_behavioral_helper_change','bootstrap_full_module_AST_equal_after_explicit_bound_change_and_identity_normalization','extension_full_module_AST_equal_after_identity_normalization','unchanged_frozen_protocol_source_map_count','old_baseline_count','all_32_prior_independent_frozen_pins_identical']},indent=2))
