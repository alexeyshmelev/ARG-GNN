"""AST, immutable metadata, source order and scientific preservation audit only."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OLD='research/reviews/native_VAL000_full_state_resume_source_20261016_v11'
NEW='research/reviews/native_VAL000_full_state_resume_source_20261016_v12'
REVIEW='research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v12_complete_integration'
checks=[]
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def good(name,condition):
    assert condition,name
    checks.append(dict(name=name,status='PASS_SOURCE_CHECK'))
def normalize(s):return s.replace(OLD,NEW).replace(OLD.replace('/','.'),NEW.replace('/','.')).replace('same_charge_full_state_resume_v11','same_charge_full_state_resume_v12').replace('prepare_continuation_root_v11.py','prepare_continuation_root_v12.py')
def body(path):return ast.parse((ROOT/path).read_text()).body
def dump(x):return ast.dump(x,include_attributes=False)
old=body(OLD+'/amendment_contract_v1.py');new=body(NEW+'/amendment_contract_v1.py')
changed={'guard_v1','charged_census_v1','fresh_guard_runtime_v11','validate_liveness_component_v11'}
added={'_consumer_guard_integrity_v12','_consumer_guards_v12'}
oldf={n.name:n for n in ast.parse(normalize((ROOT/OLD/'amendment_contract_v1.py').read_text())).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
newf={n.name:n for n in new if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
good('only_two_new_factory_lifecycle_definitions',set(newf)-set(oldf)==added)
unchanged=sorted(set(oldf)-changed)
good('every_other_contract_function_AST_unchanged_after_exact_namespace_substitution',all(dump(oldf[n])==dump(newf[n]) for n in unchanged))
for n in ['execution_only_v1.py','corpus_metadata_bridge_v1.py','resume_scientific_bridge_v5.py','alias_namespace_v7.py','fresh_entry_v9.py']:
    good(n+'_exact_namespace_only',normalize((ROOT/OLD/n).read_text())==(ROOT/NEW/n).read_text())
expected=normalize((ROOT/OLD/'prepare_continuation_root_v11.py').read_text()).replace('ROOT_amendment.INACTIVE.template.v11.json','ROOT_amendment.INACTIVE.template.v12.json').replace('amendment.root_active_v11.json','amendment.root_active_v12.json')
good('ROOT_helper_only_exact_namespace_and_receipt_filename_changes',expected==(ROOT/NEW/'prepare_continuation_root_v12.py').read_text())
a=json.loads((ROOT/OLD/'ROOT_amendment.INACTIVE.template.v11.json').read_text());b=json.loads((ROOT/NEW/'ROOT_amendment.INACTIVE.template.v12.json').read_text())
a['fresh_child_name']='same_charge_full_state_resume_v12';a['resume_policy_ref']['path']=NEW+'/resume_policy.INACTIVE.v5.json'
good('all92_amendment_template_fields_preserved_except_two_exact_namespace_paths',a==b and len(b)==92)
m=json.loads((ROOT/NEW/'source_manifest.v1.json').read_text());component=json.loads((ROOT/m['liveness_component_manifest_ref']['path']).read_text());component_review=json.loads((ROOT/m['liveness_component_independent_review_ref']['path']).read_text())
direct=ast.parse((ROOT/NEW/'direct_gpu0_v1.py').read_text());frozen=ast.parse((ROOT/component['direct_source_ref']['path']).read_text())
imports=[n for n in direct.body if isinstance(n,ast.ImportFrom) and n.module==NEW.replace('/','.')+'.amendment_contract_v1']
good('exactly_one_direct_contract_import',len(imports)==1)
imports[0].module=OLD.replace('/','.')+'.amendment_contract_v1'
good('liveness_direct_component_AST_exact_after_one_allowed_import_substitution',dump(direct)==dump(frozen))
authority=newf['terminal_snapshot_authority_v11']
good('terminal_authority_AST_exact',hashlib.sha256(dump(authority).encode()).hexdigest()==component['terminal_authority_AST_sha256']=='a481daf147c78599566d9ba7c0b75a5294bfbfc65b3b204c48be2947d9c681a0')
good('component_independent_review_is_exact_seal_and_PASS',component_review['verdict']=='PASS_SOURCE_ONLY' and component_review['candidate_manifest_ref']==m['liveness_component_manifest_ref'])
prep=(ROOT/NEW/'prepare_continuation_root_v12.py').read_text()
good('ROOT_mutation_factory_and_real_reserve_precede_consumer_guard',prep.index('guard=c.fresh_guard_runtime_v11(consumer=False)')<prep.index('rr=guard.apply_reserve_v1(')<prep.index('g=c.guard_v1(actual_q)'))
old_validate=next(n for n in body('research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/common_v1.py') if isinstance(n,ast.FunctionDef) and n.name=='validate_input_v1')
good('original_input_validation_does_not_install_guard',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='guard_v1' for n in ast.walk(old_validate)))
from research.reviews.native_VAL000_full_state_resume_source_20261016_v12 import amendment_contract_v1 as c
good('original7GiB_and_separate5GiB_only_CAPS_changes',c.CAPS==dict(c.ORIGINAL_CAPS,reservation_bytes=5*2**30,scientific_output_bytes=c.ORIGINAL_CAPS['scientific_output_bytes']-2*2**30) and c.ORIGINAL_CAPS['reservation_bytes']==7*2**30)
good('unchanged_GPU0_32GiB_C3_26GiB_4GiB_headroom_cleanup30_roster4096',c.GPU0_UUID=='GPU-ff06ca1f-403a-f2d0-b584-71e177452667' and c.CAPS['max_cuda_reserved_bytes']==32*2**30 and c.CAPS['C3_fixed_envelope_bytes']==26*2**30 and c.CAPS['headroom_bytes']==4*2**30 and c.CAPS['cleanup_seconds']==30 and c.OWNED_ROSTER_CAP==4096)
good('only_explicit5GiB_or17quarterGiB_accounting_choices',c.ACCOUNTING_CHOICES_V11==(5*2**30,17*2**28))
audit=json.loads((ROOT/c.V11_SAVED22000_AUDIT_REF['path']).read_text());counter=audit['actual_saved_optimizer_counters']
good('exact_saved22000_77AdamW_metadata_partial227990',audit['actual_saved_optimizer_updates']==22000 and audit['checkpoint_ref']==c.V11_RESUME_CHECKPOINT_REF and counter['parameter_state_count']==77 and counter['optimizer_names']==['DecoupledAdamW'] and counter['unique_step_values']==[22000] and audit['full_update_budget']==227990 and audit['full_budget_complete'] is False and audit['live_updates_after_saved_checkpoint_unknown'] is True)
good('policy_numeric_science_and_no_weight_only_restore',c.read(c.RESUME_POLICY_REF)['full_epochs']==5 and c.read(c.RESUME_POLICY_REF)['full_optimizer_updates']==227990 and c.read(c.RESUME_POLICY_REF)['batch']==512 and c.read(c.RESUME_POLICY_REF)['seed']==17 and c.read(c.RESUME_POLICY_REF)['load_weights_only'] is False)
# All repeated public checks run their policy/state/namespace/measurement functions;
# the memo affects only three exact v4 functions within one mutex transaction.
for path in (c.GUARD['path'],c.FRESH_GUARD_REF['path']):
    nodes={n.name:n for n in body(path) if isinstance(n,ast.FunctionDef)}
    check=(ROOT/path).read_text().splitlines()[nodes['check_storage_locked_v1'].lineno-1:nodes['check_storage_locked_v1'].end_lineno]
    text='\n'.join(check)
    good(path+'_policy_state_bindings_and_whole_scope_measurement_remain_in_every_check',all(s in text for s in ('_policy()','_state_locked(','for record in own[','check_prefix(','budget.v4.measure_scopes(')))
good('no_scientific_runtime_imported',not any(n in sys.modules for n in ('torch','numpy','composer','torch_geometric')))
value=dict(status='PASS_SOURCE_CHECK',checks=checks,check_count=len(checks),unchanged_contract_definitions=unchanged,only_changed_contract_definitions=sorted(changed),new_contract_definitions=sorted(added),candidate_manifest_ref=ref(NEW+'/source_manifest.v1.json'),candidate_ROOT_preparation_ref=ref(NEW+'/prepare_continuation_root_v12.py'),liveness_component_review_ref=m['liveness_component_independent_review_ref'],original_caps=c.ORIGINAL_CAPS,fresh_caps=c.CAPS,checkpoint_payload_opened=False,actual_guards_resources_processes_accounting_admission_activation_jobs_or_Git=False,finished_utc=datetime.now(timezone.utc).isoformat())
with (ROOT/REVIEW/'independent_static_invariants.v12.json').open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
