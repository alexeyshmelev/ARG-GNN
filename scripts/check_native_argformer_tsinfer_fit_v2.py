"""Targeted v2 binding/bootstrap checks. Reuse unchanged v1 scientific evidence."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types
from unittest.mock import patch

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
REVIEW=ROOT/'research/reviews/native_argformer_tsinfer_fit_v2_author'
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

m=load('_nativefitv2_source_only',ROOT/'src/arggnn/native_argformer_tsinfer_fit_v2.py')
v1=load('_sealed_nativefitv1_source_only',ROOT/'src/arggnn/native_argformer_tsinfer_fit_v1.py')
c=m.policy();oldc=v1.policy()
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
def reject(name,call):
    try:call()
    except (ValueError,RuntimeError,KeyError,TypeError,FileNotFoundError):checks.append(name);return
    raise AssertionError('unexpected acceptance: '+name)
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    return m.reference(path)

receipt=v1.read_json(v1.reference(ROOT/'research/reviews/native_argformer_tsinfer_fit_v1_author/source_receipt.json'))
check('sealed_v1_receipt_identity',v1.sha(ROOT/'research/reviews/native_argformer_tsinfer_fit_v1_author/source_receipt.json')=='6379fad0a0cfd78c6abc3aed69753bcbff9c2b358a20db233ac4da5e23356042')
for ref in receipt['files']:
    check('sealed_v1_file_unchanged_'+ref['path'],m.sha(ROOT/ref['path'])==ref['sha256'])
source=m.SCRIPT.read_text()
normalized=source.replace('native_argformer_tsinfer_fit_v2','native_argformer_tsinfer_fit_v1').replace('_admitted_native_tsinfer_fit_v2','_admitted_native_tsinfer_fit_v1').replace('_admitted_native_tsinfer_retrieval_v2','_admitted_native_tsinfer_retrieval_v1').replace('project_budget_v4_root_reserve_admission','project_budget_v3_root_reserve_admission')
normalized=re.sub("CONFIG_SHA256 = '[0-9a-f]{64}'",f"CONFIG_SHA256 = '{v1.CONFIG_SHA256}'",normalized)
check('entire_executor_AST_unchanged_after_only_binding_literal_normalization',ast.dump(ast.parse(normalized),include_attributes=False)==ast.dump(ast.parse(v1.SCRIPT.read_text()),include_attributes=False))
normalized_config=copy.deepcopy(c)
normalized_config['kind']=oldc['kind'];normalized_config['result_root']=oldc['result_root'];normalized_config['short_tmp_root']=oldc['short_tmp_root']
normalized_config['storage']['reservation_key_prefix']=oldc['storage']['reservation_key_prefix']
for r in normalized_config['storage']['companion_contract'].values():normalized_config['frozen_files_sha256'].pop(r['path'])
normalized_config['storage']['companion_contract']=oldc['storage']['companion_contract']
for r in oldc['storage']['companion_contract'].values():normalized_config['frozen_files_sha256'][r['path']]=r['sha256']
check('entire_configuration_unchanged_except_version_scopes_and_exact_budget4_bindings',m.exact(normalized_config,oldc))
proof=m.ast_proof(c)
check('all19_native_fit_ASTs_identical_to_sealed_v1',proof[m.FIT]==receipt['native_ASTs'][m.FIT])
check('all11_native_retrieval_ASTs_identical_to_sealed_v1',proof[m.RETRIEVAL]==receipt['native_ASTs'][m.RETRIEVAL])
check('native_recipe_and_model_source_commit_unchanged',c['native_recipe']==oldc['native_recipe'] and c['source_commit']==oldc['source_commit'])
check('actual_prepared_dispatchers_unchanged_no_reextraction',c['prepared_dispatchers']==oldc['prepared_dispatchers'])
check('qualified_corpus_counts_receipts_unchanged',c['qualified_corpus_metadata']==oldc['qualified_corpus_metadata'])
check('source_RNG_optimizer_retrieval_and_walltime_contracts_unchanged',c['source_AST']==oldc['source_AST'] and c['retrieval_rule']==oldc['retrieval_rule'] and c['maximum_walltime_seconds']==oldc['maximum_walltime_seconds'])
check('v4_exact_source_pin',c['storage']['companion_contract']['implementation']['sha256']=='a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47')
check('v4_exact_config_pin',c['storage']['companion_contract']['config']['sha256']=='5788f87d11825cf4fbe94a25abe056132107b155848111354e70637334e91d63')
check('v1_scientific_fixture_pass_can_be_reused_with_exact_unchangedness',receipt['checks']['status']=='passed' and receipt['checks']['count']==116)
m.verify_source(c)
check('native_author_source_inventory_unchanged',True)
fit=m.compile_fit(c);retrieval=m.compile_retrieval(c)
check('native_capsule_constructs_without_model_execution',callable(fit['execute_native']) and callable(retrieval['seal_parent']))

with tempfile.TemporaryDirectory(prefix='binding_fixture_',dir=REVIEW) as temp:
    directory=Path(temp)
    parent=c['parent_ids'][0]
    inventory=write_json(directory/'runtime_inventory.json',{'handmade_fixture_only':True})
    a=m.admission_bindings(c,parent,'probe-memory','fixture')
    review={'kind':'native_argformer_tsinfer_fit_v2_independent_review','review_role':'independent_reviewer','verdict':'PASS','implementation':a['implementation'],'config':a['config']}
    runtime={'kind':'native_argformer_tsinfer_fit_v2_root_runtime','authorizer':'root_agent','active':True,'CPU_only':False,**c['runtime'],'python_executable_sha256':m.sha(ROOT/c['runtime']['interpreter']),'environment_inventory_receipt':inventory,'additive_helpers':[]}
    reg={'kind':'native_argformer_tsinfer_fit_v2_root_experiment_registration','authorizer':'root_agent','active':True,'condition':c['condition'],'scope_parents':[parent],'stage':'probe-memory','implementation':a['implementation'],'config':a['config'],'query_truth_authorized':False,'hypothesis_id':'H013','repeat_reason':'budget binding revision only'}
    check('new_v2_exact_root_gate_accepts_handmade_schema',m.validate_admission(c,parent,'probe-memory','fixture',a,review,runtime,reg)==a)
    for field,value in [('kind','native_argformer_tsinfer_fit_v1_root_admission'),('implementation',v1.reference(v1.SCRIPT)),('config',v1.reference(v1.CONFIG)),('short_tmp',str(v1.scopes(oldc,parent,'probe-memory','fixture')[2])),('output',str(v1.scopes(oldc,parent,'probe-memory','fixture')[1])),('active',False)]:
        bad=copy.deepcopy(a);bad[field]=value
        reject('reject_old_or_inactive_binding_'+field,lambda bad=bad:m.validate_admission(c,parent,'probe-memory','fixture',bad,review,runtime,reg))
    old_review=copy.deepcopy(review);old_review['kind']='native_argformer_tsinfer_fit_v1_independent_review'
    reject('v1_review_cannot_activate_v2',lambda:m.validate_admission(c,parent,'probe-memory','fixture',a,old_review,runtime,reg))
    old_runtime=copy.deepcopy(runtime);old_runtime['kind']='native_argformer_tsinfer_fit_v1_root_runtime'
    reject('v1_runtime_authority_cannot_activate_v2',lambda:m.validate_admission(c,parent,'probe-memory','fixture',a,review,old_runtime,reg))
    for p in c['parent_ids']:
        for stage in m.STAGES:
            t=m.templates(c,p,stage,'root_assign_fresh_run');scope,output,tmp,log=m.scopes(c,p,stage,'root_assign_fresh_run')
            check('v2_exact_stage_scope_'+p+'_'+stage,t['admission']['output']==str(output) and t['storage']['output']==str(scope) and t['storage']['write_scopes']==[str(scope),str(tmp)])
            check('v2_inactive_templates_'+p+'_'+stage,not t['admission']['active'] and not t['runtime']['active'] and not t['storage']['active'])
            check('v2_native_short_socket_limit_'+p+'_'+stage,len(os.fsencode(tmp/'pymp-12345678/listener-12345678'))<=107)
            check('new_namespace_disjoint_from_v1_'+p+'_'+stage,scope!=v1.scopes(oldc,p,stage,'root_assign_fresh_run')[0] and tmp!=v1.scopes(oldc,p,stage,'root_assign_fresh_run')[2])
    # Reuse unchanged complete engineering bootstrap fixture function. Only its
    # explicit fixture temporary-root literal follows the new version namespace.
    old_tests=ast.parse((ROOT/'scripts/check_native_argformer_tsinfer_fit_v1.py').read_text())
    node=next(n for n in old_tests.body if isinstance(n,ast.FunctionDef) and n.name=='verify_bootstrap_fixture')
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='.runtime/tmp/afts1':n.value='.runtime/tmp/afts2'
    module=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(module)
    ns={'m':m,'c':c,'check':check,'reject':reject,'write_json':write_json,'ROOT':ROOT,'copy':copy,'types':types,'patch':patch,'os':os,'Path':Path,'json':json}
    exec(compile(module,'_unchanged_complete_bootstrap_fixture_v2','exec'),ns)
    ns['verify_bootstrap_fixture'](directory)
    # Exercise the successor adapter API without reading either actual ledger.
    fc=copy.deepcopy(c);fc['result_root']=str(directory/'successor_scope');fc['short_tmp_root']=str(directory/'successor_tmp')
    scope,output,tmp,log=m.scopes(fc,parent,'probe-memory','fixture')
    migration={'kind':'project_budget_v4_root_reserve_admission','root_authorizer':'root_agent','active':True,
      'no_legacy_mutation_authorized':True,'scientific_execution_authorized':False,'global_rebase_authorized':False,**c['storage']['companion_contract']}
    migref=write_json(directory/'root_reserve_authority.json',migration)
    reservation={'key':c['storage']['reservation_key_prefix']+'/'+parent,'output':str(scope),'auxiliary_scopes':[str(tmp)],'reserved_bytes':20000000000,'consumer_bindings':[m.reference(m.SCRIPT),m.reference(m.CONFIG)],'root_admission':migref}
    storage_receipt={'kind':'native_argformer_tsinfer_fit_v2_root_storage','authorizer':'root_agent','active':True,'implementation':m.reference(m.SCRIPT),'config':m.reference(m.CONFIG),'scope_parents':[parent],
      'reservation_key':reservation['key'],'reserved_bytes':20000000000,'output':str(scope),'write_scopes':[str(scope),str(tmp)],'stage_boundary_cap_not_OS_quota':True,
      'mode':'separately_audited_successor','migration_authority':migref,'companion':c['storage']['companion_contract'],'reservation':reservation,
      'legacy_ledger_sha256_at_admission':'1'*64,'supplemental_ledger_sha256_at_admission':'2'*64}
    calls=[]
    def fake_account(root,**kwargs):
        calls.append(kwargs)
        return {'projected_bytes':400000000000,'limit_bytes':950000000000,'legacy_ledger_sha256':('1' if len(calls)==1 else '3')*64,'supplemental_ledger_sha256':'2'*64}
    fake=types.SimpleNamespace(check_reservation=fake_account)
    fake_spec=types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda module:None))
    with patch.object(m.importlib.util,'spec_from_file_location',return_value=fake_spec),patch.object(m.importlib.util,'module_from_spec',return_value=fake):
        st=m.Storage(fc,parent,'probe-memory','fixture',storage_receipt)
        st.check()
        check('budget4_adapter_receives_full_new_owned_reservation',calls[0]['reservation']==reservation)
        check('budget4_adapter_receives_exact_first_live_hash_pair',calls[0]['initial_ledger_pins']=={'legacy_ledger_sha256':'1'*64,'supplemental_ledger_sha256':'2'*64})
        check('budget4_adapter_allows_dynamic_unrelated_reservations_after_initial_bind',calls[1]['initial_ledger_pins'] is None)
        check('budget4_adapter_uses_real_boundary_measurement_API',all(x['boundary'] is True for x in calls))
        check('budget4_adapter_receives_exact_reviewed_source_and_config',calls[0]['expected_source_sha256']==c['storage']['companion_contract']['implementation']['sha256'] and calls[0]['expected_config_sha256']==c['storage']['companion_contract']['config']['sha256'])
        wrong=copy.deepcopy(migration);wrong['kind']='project_budget_v3_root_reserve_admission'
        wref=write_json(directory/'old_budget_reserve.json',wrong)
        bad=copy.deepcopy(storage_receipt);bad['migration_authority']=wref;bad['reservation']['root_admission']=wref
        reject('budget3_root_reserve_cannot_activate_budget4_adapter',lambda:m.Storage(fc,parent,'probe-memory','fixture',bad))
        old_storage=copy.deepcopy(storage_receipt);old_storage['kind']='native_argformer_tsinfer_fit_v1_root_storage'
        reject('old_v1_storage_receipt_cannot_borrow_new_scope',lambda:m.Storage(fc,parent,'probe-memory','fixture',old_storage))

# Sealed v1 artifacts are verified again after fixture construction.
for ref in receipt['files']:check('sealed_v1_preserved_after_targeted_checks_'+ref['path'],m.sha(ROOT/ref['path'])==ref['sha256'])
report={'kind':'native_argformer_tsinfer_fit_v2_targeted_author_checks','status':'passed','check_count':len(checks),'checks':checks,
 'implementation':m.reference(m.SCRIPT),'config':m.reference(m.CONFIG),'test_source':m.reference(__file__),
 'v1_source_receipt':v1.reference(ROOT/'research/reviews/native_argformer_tsinfer_fit_v1_author/source_receipt.json'),
 'reused_v1_scientific_fixture_report':receipt['checks']['report'],'reused_v1_check_count':116,
 'whole_source_AST_binding_normalization_equal':True,'whole_configuration_only_binding_diff':True,'native_scientific_ASTs_unchanged':True,
 'large_scientific_fixture_not_repeated':True,'actual_arrays_opened':False,'native_model_executed':False,'GPU':False,'actual_live_ledger_read':False,
 'successor_adapter_accounting_stubbed_only_for_contract_test':True,'complete_real_bootstrap_code_exercised_with_exec_stub':True}
write_json(REVIEW/'targeted_checks.json',report)
print(json.dumps({'status':'passed','check_count':len(checks),'implementation':report['implementation'],'config':report['config']},indent=2))
