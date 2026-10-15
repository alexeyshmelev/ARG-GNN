"""Static source/count-metadata qualification only; no project module imports.

Reads fixed source and compact metadata pins, compiles text without execution,
and compares ASTs. No guard, process, payload, journal or scope inventory call.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,json,stat
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
REL='research/reviews/American_C3_native_paths_root_v3'
HERE=ROOT/REL
CONSUMER='src/arggnn/american_c3_native_paths_consumer_v3.py'
OLD_CONSUMER='src/arggnn/american_c3_native_paths_consumer_v2.py'
OLD_PATHS='src/arggnn/american_c3_paths_v1.py'
OLD_REL='research/reviews/American_C3_native_paths_root_v2'

def reference(path):
    raw=(ROOT/path).read_bytes();return dict(path=path,sha256=hashlib.sha256(raw).hexdigest())
def source(path):return (ROOT/path).read_text()
def functions(path):return {n.name:n for n in ast.parse(source(path)).body if isinstance(n,ast.FunctionDef)}
def dump(node):return ast.dump(node,include_attributes=False)
def literal_assignment(path,name):
    for n in ast.parse(source(path)).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='dict':
                return {k.arg:ast.literal_eval(k.value) for k in n.value.keywords}
            return ast.literal_eval(n.value)
    raise AssertionError(name)
def call_message(node):
    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
        values=node.value.args
        if values and isinstance(values[-1],ast.Constant) and isinstance(values[-1].value,str):return values[-1].value
    return ''

class NormalizeBudgetExecute(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        if node.module=='american_c3_common_v1':return None
        return node
    def visit_Assign(self,node):
        names={x.id for x in ast.walk(node.targets[0]) if isinstance(x,ast.Name)}
        if names & {'reference_paths','reference_tokens','max_raw'}:return None
        node=self.generic_visit(node)
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='planned':
            node.value=ast.parse('5*sum(map(len,selected))+32*len(selected)+64*v+pickle_margin',mode='eval').body
        return node
    def visit_If(self,node):
        if dump(node.test)==dump(ast.parse('key[0]<600',mode='eval').body):return None
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if call_message(node) in {
            'Exact original first600 reference / last1000 query role partition required; no fallback',
            'Exact full original tree dimensions for occurrence serialization required',
            'Exact full-count maximum29 required for serialization proof',
            'Exact complete corpus/reference counts required before durable products; no subset fallback',
            'Exact original complete vocabulary required'}:return None
        return self.generic_visit(node)
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='npz_save_v3':
            assert isinstance(node.args[0],ast.Name) and node.args[0].id=='w'
            node.func=ast.Attribute(value=ast.Name(id='w',ctx=ast.Load()),attr='npz',ctx=ast.Load());node.args=node.args[1:]
        if isinstance(node.func,ast.Attribute) and node.func.attr=='pickle' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='sequence_to_tree_indices.pkl':
            node.args[2]=ast.parse('5*tokens+64*len(seqs)+5*ts.num_samples*ts.num_trees+pickle_margin',mode='eval').body
        if isinstance(node.func,ast.Name) and node.func.id=='dict' and any(k.arg=='status' and isinstance(k.value,ast.Constant) and k.value.value=='completed_native_C3_path_preparation' for k in node.keywords):
            added={'reference_unique_paths','reference_path_tokens','max_raw_path_nodes','budget_successor_source_ref','scientific_preparation_original_source_ref','exact_per_sample_counts_ref','native_serialization_bound_bytes'}
            node.keywords=[k for k in node.keywords if k.arg not in added]
        return node

class NormalizeSupervisor(ast.NodeTransformer):
    def visit_Expr(self,node):
        if call_message(node)=='Actual complete reference/serialization evidence and original science required':return None
        node=self.generic_visit(node)
        if call_message(node)=='Actual complete original native corpus metadata':node.value.args[-1].value='Actual complete unchanged native corpus metadata'
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='dict' and any(k.arg=='actual_direct_worker_wait' for k in node.keywords):
            node.keywords=[k for k in node.keywords if k.arg!='budget_successor_source_ref']
            for k in node.keywords:
                if k.arg=='unmodified_native_scientific_source_ref':k.value=ast.parse('c.PATHS_REF',mode='eval').body
        return node

def qualify():
    manifest=json.loads((HERE/'source_manifest.json').read_text());checks={}
    for path,sha in manifest['source_sha256'].items():
        checks['source_pin:'+path]=reference(path)['sha256']==sha
    for name,record in manifest['metadata_refs'].items():checks['metadata_pin:'+name]=reference(record['path'])==record
    for path in [CONSUMER,REL+'/native_paths_contract_v3.py',REL+'/root_native_paths_driver.py',REL+'/qualify_sources_v3.py']:
        compile(source(path),path,'exec');checks['compile_without_execution:'+path]=True
    new=functions(CONSUMER);old=functions(OLD_PATHS)
    checks['entire_execute_scientific_AST_preserved_after_exact_budget_assertion_normalization']=dump(NormalizeBudgetExecute().visit(copy.deepcopy(new['execute'])))==dump(old['execute'])
    oldc=functions(OLD_REL+'/native_paths_contract_v2.py');newc=functions(REL+'/native_paths_contract_v3.py')
    for name in ['validate_counts_v1','original_guard_v1','validate_close_receipt_v1']:checks['original_function_unchanged:'+name]=dump(oldc[name])==dump(newc[name])
    oldd=functions(OLD_REL+'/root_native_paths_driver.py');newd=functions(REL+'/root_native_paths_driver.py')
    checks['owned_RSS_source_unchanged']=dump(oldd['owned_group_RSS_v1'])==dump(newd['owned_group_RSS_v1'])
    checks['entire_supervision_wait_guard_cleanup_preserved_after_terminal_metadata_normalization']=dump(NormalizeSupervisor().visit(copy.deepcopy(newd['launch'])))==dump(oldd['launch'])
    main_text=ast.get_source_segment(source(CONSUMER),new['main'])
    checks['canonical_common_CLI_uses_new_source_and_preserved_original_author_dependency']='common.cli(SOURCE,DEFAULTS,execute,' in main_text and 'ORIGINAL_PATHS_REF["path"]' in main_text
    checks['no_stale_paths_module_call_in_main']=not any(isinstance(n,ast.Name) and n.id=='paths' for n in ast.walk(new['main']))
    checks['consumer_original_memo_install_kept']='memo.install(v4,expected_source_path=ROOT/c.V4_REF["path"]' in main_text
    cfg=json.loads((HERE/'config.inactive.json').read_text());oldcfg=json.loads((ROOT/OLD_REL/'config.inactive.json').read_text())
    checks['only_budget_count_metadata_config_fields_added_or_changed']=all(cfg[k]==v for k,v in oldcfg.items() if k!='output_cap_bytes') and cfg['output_cap_bytes']==6000000000
    checks['inactive_preview_and_no_issued_storage']=cfg['execution_enabled'] is False and json.loads((HERE/'input.inactive.json').read_text())['storage'] is None
    count_ref=literal_assignment(REL+'/native_paths_contract_v3.py','EXACT_COUNT_REF');exact=json.loads((ROOT/count_ref['path']).read_text())
    per_ref=literal_assignment(REL+'/native_paths_contract_v3.py','PER_SAMPLE_REF');detail=json.loads((ROOT/per_ref['path']).read_text());rows=detail['per_sample_counts']
    checks['original_complete_count_bindings']=exact['per_sample_counts_ref']==per_ref and reference(per_ref['path'])==per_ref
    checks['count_metadata_all1600_ordered_samples_and_exact_reference_aggregates']=[r['sample_index'] for r in rows]==list(range(1600)) and [r['sample_node'] for r in rows]==list(range(1600)) and sum(r['unique_paths'] for r in rows[:600])==9583977 and sum(r['path_tokens'] for r in rows[:600])==116080787
    checks['role_partition_and_exact_count_stops_precede_first_durable_product']=source(CONSUMER).index('Exact original first600 reference / last1000 query role partition')<source(CONSUMER).index("focal_ref=npz_save_v3") and source(CONSUMER).index('Exact complete corpus/reference counts required before durable products')<source(CONSUMER).index("focal_ref=npz_save_v3")
    checks['unchanged_original_author_function_delegation']='return original.author_functions()' in source(CONSUMER)
    checks['exact_CPython_protocol5_required']='sys.version_info[:3]==(3,11,17)' in main_text and 'pickle.HIGHEST_PROTOCOL==5' in main_text
    proof=json.loads((HERE/'storage_forecast.json').read_text())
    checks['aggregate_bound_and_6GB_margin']=proof['total_proposed_bound_bytes']==5942029370 and sum(proof['terms_bytes'].values())==5942029370 and proof['output_and_auxiliary_scope_cap_bytes']==6000000000 and proof['remaining_cap_margin_bytes']==57970630
    checks['per_file_prospective_budget_formulas_present']='return 5*t+8*n+64*v+MARGIN' in source(CONSUMER) and 'return 5*tokens+8*paths+404779200+MARGIN' in source(CONSUMER) and 'planned=raw+(raw+99)//100+1048576' in source(CONSUMER)
    checks['NPZ_same_standard_format_prepare_finish']='np.savez_compressed(stream,**arrays)' in source(CONSUMER) and 'p=w.prepare(name,planned)' in source(CONSUMER) and 'return w.finish(p,planned)' in source(CONSUMER)
    checks['original64_binding_guard_and_root_accounting_mutation_path_kept']='len(consumers)<=64' in source(REL+'/root_native_paths_driver.py') and 'm.apply_reserve_v1(authority_ref,expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)' in source(REL+'/root_native_paths_driver.py')
    checks['root_guard_never_memoized']='m=c.original_guard_v1()' in source(REL+'/root_native_paths_driver.py') and 'memo.install' not in source(REL+'/root_native_paths_driver.py')
    direct=[*manifest['native_scientific_closure'],*manifest['integration_closure_refs'],*manifest['metadata_refs'].values()]
    checks['named_direct_source_metadata_refs_are_canonical_single_link_regular']=all((ROOT/r['path']).resolve()==ROOT/r['path'] and stat.S_ISREG((ROOT/r['path']).lstat().st_mode) and (ROOT/r['path']).lstat().st_nlink==1 and (ROOT/r['path']).lstat().st_size<=32*1024*1024 for r in direct)
    checks['hardlinked_runtime_stdlib_is_read_only_nested_provenance_not_direct_guard_binding']=all(r['path']!='.runtime/env/lib/python3.11/pickle.py' for r in direct)
    if not all(checks.values()):raise AssertionError(json.dumps([k for k,v in checks.items() if not v]))
    return dict(schema='American_C3_native_paths_v3_source_qualification_v1',status='passed_static_source_metadata_checks_only',created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,
        source_manifest_ref=reference(REL+'/source_manifest.json'),source_refs=[reference(p) for p in [CONSUMER,REL+'/native_paths_contract_v3.py',REL+'/root_native_paths_driver.py',REL+'/qualify_sources_v3.py']],
        compiled_modules_imported_or_executed=False,scientific_preparation='same original scientific control flow, author functions, corpus/views/split/focals/native objects; budget expressions/checks/metadata changed',
        actions=dict(scientific_or_guard_module_imports=0,payload_roster_tree_label_reads=0,process_GPU_state_journal_checks=0,scope_inventories=0,admissions=0,scientific_execution=0,files_deleted=0,Git_operations=0))

if __name__=='__main__':
    value=qualify();raw=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    with (HERE/'source_qualification_receipt.json').open('xb') as stream:stream.write(raw)
    print(json.dumps(dict(status=value['status'],checks=len(value['checks']),receipt_ref=reference(REL+'/source_qualification_receipt.json'))))
