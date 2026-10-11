"""Read-only independent receipt/consumer validation. No native runtime capture."""
import datetime,hashlib,importlib.util,json,os,sys
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D=ROOT/'research/reviews/native_argformer_tsinfer_fit_v6_independent'
sys.dont_write_bytecode=True
checks=[]
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def check(n,v):
 if not v:raise AssertionError(n)
 checks.append(n)
check('CPU_visibility_and_disabled_bytecode',os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1')
r=json.loads((D/'review_receipt.json').read_text());s=json.loads((D/'independent_review_schema_template.json').read_text())
for k in ('kind','review_role','implementation','config'):check('consumer_schema_'+k,r[k]==s[k])
check('source_only_PASS_is_final_independent_review',r['verdict']=='PASS' and r['template_not_review'] is False and r['scientific_execution_authorized'] is False)
for k in ('actual_model_fit_executed','actual_scientific_arrays_opened','actual_GPU_inventory_context_or_live_ledger_read','actual_native_runtime_inventory_captured','actual_native_scientific_imports','authority_activation_performed','runtime_installation_or_prefix_mutation'):check('source_only_'+k,r[k] is False)
for entry in r['files']:check('sealed_evidence_'+entry['path'],ref(ROOT/entry['path'])==entry)
for key in ('implementation','config','author_source_receipt','preserved_v5_PASS','preserved_v5_validator','directory_metadata','absent_executable_metadata'):check('requested_pin_'+key,ref(ROOT/r[key]['path'])==r[key])
author=json.loads((ROOT/r['author_source_receipt']['path']).read_text())
for entry in author['files']:check('sealed_author_'+entry['path'],ref(ROOT/entry['path'])==entry)
h=json.loads((D/'handmade_checks.json').read_text());attempt=json.loads((D/'attempt01.json').read_text())
check('fresh270_checks_and107_rejection_cases_passed',h['status']=='passed' and h['check_count']==270 and len(h['checks'])==270 and len(h['details']['rejections'])==107 and not h['issues'] and attempt['returncode']==0)
for p,sha in h['pinned_inputs'].items():check('final_baseline_input_'+p,ref(ROOT/p)['sha256']==sha)
spec=importlib.util.spec_from_file_location('_v6_review_consumer_validator',ROOT/r['implementation']['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);c=m.policy()
check('all30_native_ASTs_bound',m.ast_proof(c)==h['details']['scientific_ASTs'] and sum(len(x) for x in h['details']['scientific_ASTs'].values())==30)
for p in c['parent_ids']:
 for stage in m.STAGES:
  a=m.admission_bindings(c,p,stage,'review_validator')
  try:m.validate_admission(c,p,stage,'review_validator',a,r,{}, {})
  except ValueError as error:check('source_PASS_stops_at_missing_runtime_'+p+'_'+stage,str(error)=='exact fresh isolated runtime authority required')
  else:raise AssertionError('source PASS must not supply runtime or actual stage authority')
  pending=dict(r,verdict='PENDING')
  try:m.validate_admission(c,p,stage,'review_validator',a,pending,{}, {})
  except ValueError as error:check('PENDING_verdict_denied_'+p+'_'+stage,str(error)=='fresh independent source/config PASS required')
  else:raise AssertionError('PENDING verdict must deny')
result={'kind':'native_argformer_tsinfer_fit_v6_independent_review_validator_receipt','status':'passed','check_count':len(checks),'checks':checks,'review_receipt':ref(D/'review_receipt.json'),'validator':ref(Path(__file__)),'executor_receipt_contract_validated':True,'source_PASS_does_not_supply_runtime_or_execution_authority':True,'actual_native_runtime_scientific_imports_models_GPU_live_ledgers_or_authority_activation':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (D/'validator_receipt.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'status':'passed','check_count':len(checks),'review_receipt':result['review_receipt']},indent=2))
