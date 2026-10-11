"""Validate exact independent v3 REJECT contract and all12 admission rejections."""
import datetime,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
REVIEW=ROOT/'research/reviews/native_argformer_tsinfer_fit_v3_independent'
sys.dont_write_bytecode=True
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return {'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
r=json.loads((REVIEW/'review_receipt.json').read_text())
schema=json.loads((REVIEW/'independent_review_schema_template.json').read_text())
checks=[]
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
for key in ('kind','review_role','implementation','config'):check('exact_schema_'+key,r[key]==schema[key])
check('final_independent_REJECT',r['verdict']=='REJECT' and r['template_not_review'] is False)
for key in ('actual_model_fit_executed','actual_scientific_arrays_opened','actual_GPU_inventory_context_or_live_ledger_read','authority_activation_performed','actual_native_runtime_inventory_captured'):
 check('source_only_'+key,r[key] is False)
for record in r['files']:check('exact_evidence_'+record['path'],sha(ROOT/record['path'])==record['sha256'])
for key in ('implementation','config','author_source_receipt','preserved_v2_REJECT'):
 record=r[key];check('exact_pin_'+key,sha(ROOT/record['path'])==record['sha256'])
h=json.loads((REVIEW/'handmade_checks.json').read_text())
check('fresh134_checks',h['check_count']==134)
check('one_reproduced_owned_policy_blocker',len(h['counterexamples'])==1 and h['counterexamples'][0]['id']=='OWNED_CHILD_ENTRY_POLICY_FAILURE_NO_TERMINAL')
small=h['details']['fresh_handmade_retrieval']
check('small_full_reference_native_kernel',small['full_paths']==280 and small['pure_reference_full_paths']==96 and small['query_full_paths']==120 and small['exhaustive_pairs']==11520 and small['native_fit_evidence'] is False)
spec=importlib.util.spec_from_file_location('_exact_v3_verdict_validator',ROOT/r['implementation']['path'])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);c=m.policy()
check('all30_ASTs_bound',m.ast_proof(c)==h['details']['scientific_ASTs'])
for parent in c['parent_ids']:
 for stage in m.STAGES:
  a=m.admission_bindings(c,parent,stage,'review_validator')
  try:m.validate_admission(c,parent,stage,'review_validator',a,r,{}, {})
  except ValueError as error:check('REJECT_blocks_'+parent+'_'+stage,str(error)=='fresh independent source/config PASS required')
  else:raise AssertionError('REJECT must never admit execution')
result={'kind':'native_argformer_tsinfer_fit_v3_independent_review_validator_receipt','status':'passed','checks':checks,'check_count':len(checks),'review_receipt':ref(REVIEW/'review_receipt.json'),'schema_template':ref(REVIEW/'independent_review_schema_template.json'),'validator':ref(Path(__file__)),'executor_receipt_contract_validated':True,'REJECT_blocks_all12_parent_stages':True,'actual_execution_or_authority_activation':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (REVIEW/'validator_receipt.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps({'status':'passed','check_count':len(checks),'review_receipt':result['review_receipt']},indent=2))
