"""Read-only author seal validation. Writes only its bounded local validation receipt."""
from __future__ import annotations
import ast,copy,hashlib,json,os
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/census_biological_fit_v2_author'
checks=[]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)
check('bytecode_disabled_and_CPU_visibility_empty',os.environ.get('PYTHONDONTWRITEBYTECODE')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='')
receipt_path=OUT/'author_receipt.json';receipt=json.loads(receipt_path.read_text())
check('author_only_pending_independent_review',receipt['schema']=='census_biological_fit_author_receipt_v2' and receipt['status']=='source_selection_checks_completed_pending_fresh_independent_review' and receipt['actual_admission'] is False and receipt['independent_review'] is False and receipt['grants_execution_authority'] is False)
for p,h in receipt['artifact_sha256'].items():check('sealed_author_artifact_'+p,sha(ROOT/p)==h)
for p,h in receipt['baseline_pins'].items():check('preserved_v1_baseline_'+p,sha(ROOT/p)==h)
config=json.loads((ROOT/'configs/census_biological_fit_v2.json').read_text())
for p,h in receipt['source_dependency_sha256'].items():check('unchanged_source_dependency_'+p,sha(ROOT/p)==h)
check('all35_source_dependency_bindings_exact',len(receipt['source_dependency_sha256'])==35 and config['source_sha256']==receipt['source_dependency_sha256'])
s1=(ROOT/'src/arggnn/census_biological_fit_v1.py').read_text();s2=(ROOT/'src/arggnn/census_biological_fit_v2.py').read_text()
delta=json.loads((OUT/'exact_delta_proof.json').read_text());expected=s1
for item in delta['source_token_replacements']:
    check('registered_replacement_count_'+item['old'],s1.count(item['old'])==item['count'])
    expected=expected.replace(item['old'],item['new'])
check('entire_source_exact_authorized_byte_delta',s2==expected)
c1=(ROOT/'configs/census_biological_fit_v1.json').read_text();c2=(ROOT/'configs/census_biological_fit_v2.json').read_text()
check('entire_config_exact_authorized_byte_delta',c2==c1.replace('census_biological_fit_v1','census_biological_fit_v2'))
check('successor_syntax_compiles_without_module_execution',compile(ast.parse(s2),'sealed_fit_v2','exec') is not None)
report=json.loads((OUT/'selection_checks.json').read_text());attempt=json.loads((OUT/'checks_attempt01.json').read_text())
check('319_source_selection_checks_passed_first_attempt',report['status']=='passed_bounded_exact_selection_path_and_complete_delta' and report['check_count']==319 and len(report['checks'])==319 and attempt['returncode']==0)
check('seven_exact_source_selection_fixtures',len(report['fixtures'])==7 and all(f['epochs']==30 and len(f['controls'])==5 and f['score_calls']==300 and f['actual_steps_performed']==0 for f in report['fixtures']))
byname={f['label']:f for f in report['fixtures']}
for label,want in [('v2_all_infinite30epochs',1),('v2_infinity_to_first_finite_then_ties',5),('v2_equal_finite30epoch_ties',1),('v2_later_infinite_never_overwrites_finite',2),('preserved_v1_all_infinite_counterexample',None)]:
    check('sealed_selection_outcome_'+label,set(byname[label]['selected_epoch_by_control'].values())=={want})
check('raw_infinity_preserved_no_actual_biological_occurrence_claim',receipt['actual_all_infinite_biological_occurrence_observed'] is False and receipt['raw_infinity_preserved'] is True and report['actual_targets_opened'] is False)
check('no_scientific_model_optimizer_fitting_GPU_or_live_resources',report['scientific_model_instantiated'] is False and report['optimizer_instantiated'] is False and report['optimizer_or_fitting_steps_performed']==0 and report['GPU_or_device_query_or_context'] is False and report['live_ledger_or_lock_access'] is False)
for label,r in report['current_poolv2_refs'].items():check('poolv2_source_config_still_unchanged_'+label,sha(ROOT/r['path'])==r['sha256'])
pool=json.loads((ROOT/report['current_poolv2_refs']['config']['path']).read_text())
check('poolv2_still_has_no_fit2_eligibility','arggnn.census_biological_fit_v2' not in pool['allowed_modules'] and 'arggnn.census_biological_fit_v2' not in pool['additional_worker_contracts'])
check('same_tsinfer_still_separate_unqualified',receipt['same_tsinfer_consumer']=='separate_unqualified_not_added' and report['same_tsinfer_consumer']=='separate_unqualified_not_added')
validation={'schema':'census_biological_fit_v2_author_validation_v1','status':'passed','check_count':len(checks),'checks':checks,'author_receipt':{'path':str(receipt_path.relative_to(ROOT)),'sha256':sha(receipt_path)},'actual_admission':False,'independent_review':False,'grants_execution_authority':False,'actual_targets_models_optimizers_fitting_GPU_live_resources_accessed':False}
with (OUT/'validator_receipt.json').open('x') as stream:json.dump(validation,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'status':'passed','check_count':len(checks),'author_receipt':validation['author_receipt']},indent=2))
