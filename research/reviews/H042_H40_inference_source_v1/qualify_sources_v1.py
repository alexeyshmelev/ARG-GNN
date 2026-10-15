"""Standard-library source/AST checks; no Torch or model imports."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,json
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
REL='research/reviews/H042_H40_inference_source_v1'
HERE=ROOT/REL

def reference(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def cls(path,name):return next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
def method(node,name):return next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(node):return ast.dump(node,include_attributes=False)
def number(node):
    if isinstance(node,ast.Constant):return node.value
    if isinstance(node,ast.Name) and node.id=='hidden_dim':return 32
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return number(node.left)+number(node.right)
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult):return number(node.left)*number(node.right)
    raise AssertionError(ast.unparse(node))

def learned_shapes():
    out={}
    def layer(prefix,node):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute):return
        kind=node.func.attr;kw={x.arg:x.value for x in node.keywords}
        if kind=='Sequential':
            for i,item in enumerate(node.args):layer(prefix+'.'+str(i),item)
        elif kind=='Linear':
            inp,width=map(number,node.args[:2]);out[prefix+'.weight']=[width,inp]
            if 'bias' not in kw or number(kw['bias']):out[prefix+'.bias']=[width]
        elif kind=='LayerNorm':
            width=number(node.args[0]);out[prefix+'.weight']=[width];out[prefix+'.bias']=[width]
        elif kind=='GRU':
            inp,width=map(number,node.args[:2]);assert number(kw['num_layers'])==1 and number(kw['bidirectional']) is True
            for suffix in ('','_reverse'):
                out[prefix+'.weight_ih_l0'+suffix]=[3*width,inp];out[prefix+'.weight_hh_l0'+suffix]=[3*width,width]
                out[prefix+'.bias_ih_l0'+suffix]=[3*width];out[prefix+'.bias_hh_l0'+suffix]=[3*width]
    for path,name,prefix in [('src/arggnn/mutation_routed_whole_tree_gnn_v1.py','MutationRoutedWholeTreeGNNV1',''),('src/arggnn/anonymous_clade_flag_control_v1.py','RankEightCladeFlagAdapterV1','clade_coupling.')]:
        for n in method(cls(path,name),'__init__').body:
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Attribute) and isinstance(n.targets[0].value,ast.Name) and n.targets[0].value.id=='self':layer(prefix+n.targets[0].attr,n.value)
    return out

class RestoreOriginalCounts(ast.NodeTransformer):
    def visit_Attribute(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=='self':
            if node.attr=='expected_haplotypes':return ast.Constant(value=952)
            if node.attr=='expected_queries':return ast.Constant(value=686)
        return self.generic_visit(node)

def qualify_sources():
    manifest=json.loads((HERE/'source_manifest.json').read_text());checks={}
    for record in manifest['frozen_source_refs'].values():checks['frozen_pin:'+record['path']]=reference(record['path'])==record
    for path,sha in manifest['new_source_sha256'].items():checks['new_source_pin:'+path]=reference(path)['sha256']==sha
    for record in manifest['owned_metadata_refs'].values():checks['metadata_pin:'+record['path']]=reference(record['path'])==record
    source_path=REL+'/h042_panel_inference_v1.py';source=(ROOT/source_path).read_text();tree=ast.parse(source)
    for path in [source_path,REL+'/qualify_sources_v1.py']:compile((ROOT/path).read_text(),path,'exec');checks['compile_text_without_import:'+path]=True
    allowed={'pathlib','collections.abc','dataclasses','hashlib','json','math'};top=[]
    for n in tree.body:
        if isinstance(n,ast.Import):top.extend(a.name for a in n.names)
        if isinstance(n,ast.ImportFrom):top.append(n.module)
    checks['adapter_module_import_is_standard_library_only']=all(x in allowed for x in top)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    newclass=next(n for n in ast.walk(functions['_model_class_v1']) if isinstance(n,ast.ClassDef));original=cls('src/arggnn/anonymous_clade_flag_control_v1.py','AnonymousCladeFlagControlGNNV1')
    checks['all_forward_arithmetic_identical_after_two_count_guards_restored']=dump(RestoreOriginalCounts().visit(copy.deepcopy(method(newclass,'forward_holds'))))==dump(method(original,'forward_holds'))
    checks['only_constructor_and_forward_count_guard_overridden']={n.name for n in newclass.body if isinstance(n,ast.FunctionDef)}=={'__init__','forward_holds'}
    checks['inherits_frozen_H042_encode_sweeps_and_query_readouts']=len(newclass.bases)==1 and ast.unparse(newclass.bases[0])=='AnonymousCladeFlagControlGNNV1'
    constructor=ast.unparse(method(newclass,'__init__'))
    checks['same_original_backbone_and_adapter_initialization']='MutationRoutedFullPanelPyGGNNV1.__init__(self, hidden_dim=32, expected_haplotypes=expected_haplotypes, site_chunk_size=256, checkpoint_sites=True)' in constructor and 'self.clade_coupling = RankEightCladeFlagAdapterV1()' in constructor
    checks['no_new_learned_layers_parameters_or_buffers']=not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'Linear','GRU','LayerNorm','Parameter','register_buffer','register_parameter'} for n in ast.walk(tree))
    schema=json.loads((HERE/'state_schema.json').read_text());shapes=learned_shapes();count=0
    for shape in shapes.values():
        n=1
        for d in shape:n*=d
        count+=n
    checks['all42_state_shapes_derived_from_original_source']=schema['shapes']==shapes and len(shapes)==42
    checks['all30788_values_same_capacity_no_H_dimension']=count==schema['parameter_count']==30788 and schema['state_values_read'] is False
    checks['full_H042_strict_state_loading_no_warmstart']='model.load_state_dict(selected_model_state,strict=True)' in source and 'load_h038_flag_warmstart' not in source
    checks['CPU_FP32_exact_state_and_value_digests_enforced']="value.device.type=='cpu'" in source and 'value.dtype==torch.float32' in source and '_tensor_sha256(value)==digests[name]' in source and '.equal(value)' in source
    checks['model_frozen_eval_after_load']='model.eval();model.requires_grad_(False)' in source
    checks['original_flag_array_math_and_payload_reused']='arrays,meta=build_clade_flag_arrays_v1(parent,samples)' in source and 'AnonymousCladeFlagsV1.from_verified_arrays(arrays,meta)' in source
    checks['no_checkpoint_reader_training_or_fixture_call']=not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'load','backward','step','train','fit','calibrate'} for n in ast.walk(tree))
    checks['H40_explicit_R16_Q24_counts']=source.count('expected_haplotypes=40,expected_queries=24')>=3 and "proof['query_columns']==list(range(16,40))" in source
    checks['twenty_real_diploid_identity_and_age_pairs_required']="len({keys[i][0] for i in range(0,40,2)})==20" in source and 'ages[i]==ages[i+1]' in source
    checks['donor_age350_alpha0_5_log2000001_fixed']="proof.get('age_split_generations')==350." in source and "proof.get('donor_alpha')==.5" in source and "proof.get('learned_span_log_denominator')==2000001." in source and 'math.log(2000001.)' in source
    checks['H018_actual_two_donors_each_channel_proxy_purity_unverified']='groups==[2]*8' in source and "proxy_pure_ancestry_status='unverified'" in source
    checks['separate_loader_content_fingerprint_required']="bank_fingerprint_sha256_v1(bank)==proof.get('bank_fingerprint_sha256')" in source and 'validate_inputs_v1(bank,expected_haplotypes=expected_haplotypes)' in source
    checks['selected_best_identity_not_certified_here']=json.loads((HERE/'selected_state_identity.inactive.json').read_text())['qualified'] is False
    checks['real_observation_identity_not_certified_here']=json.loads((HERE/'H018_H40_observation_identity.inactive.json').read_text())['qualified'] is False
    if not all(checks.values()):raise AssertionError(json.dumps([k for k,v in checks.items() if not v]))
    return dict(schema='H042_H40_source_qualification_receipt_v1',created_utc=datetime.now(timezone.utc).isoformat(),status='passed_source_AST_and_shape_derivation_only',checks=checks,source_manifest_ref=reference(REL+'/source_manifest.json'),strict_state_factory_executed=False,original_panel_parity_executed=False,real_loader_or_domain_transfer_qualified=False,actual_checkpoint_VCF_tree_model_or_target_read=False,Torch_project_or_guard_imports=0,process_GPU_state_journal_inventory_admission_accounting_execution_Git_actions=0,scientific_claim='software shape generalization only; no real ancestry truth, accuracy, purity, calibration or novelty result')

if __name__=='__main__':print(json.dumps(qualify_sources(),indent=2,sort_keys=True))
