"""Original full-target metadata joins and ROOT batch identity; no operational call here."""
from pathlib import Path
from datetime import datetime,timezone
import copy

def install_v1(c,b,phase_budget):
 original_caps=c.stage_caps;original_validate=c.validate_request;original_roles=c.direct_roles;original_input=c.input_spec
 def stage_caps(stage,cfg):
  caps=original_caps(stage,cfg)
  rows=[row for row in b['original_targets'].values()if row['target_row_for_later_ROOT_binding']['observation_ref']==cfg.get('observation_ref')]
  c.require(stage=='convert'and len(rows)==1 and cfg['max_sites']==rows[0]['input_sites']
   and caps['output_bytes']==rows[0]['conversion_output_cap_bytes']and caps['reservation_bytes']==rows[0]['whole_scope_reservation_bytes']
   and caps['cpu_threads']==4 and caps['rss_bytes']==4*2**30,'Count-derived full original parent conversion caps, unchanged CPU/RSS')
  caps.update(scientific_seconds=10800,wall_seconds=11400);return caps
 c.stage_caps=stage_caps
 def historical_failure_acceptance_v1(record):
     v = c.read(record)
     old = b['old_failure']
     c.require(v.get('schema') == 'ROOT_FLARE_failed_unsealed_acceptance_v2' and v.get('root_authorizer') == 'root_agent' and (v.get('accepted') is True) and all((type(v.get(k)) is type(old[k]) and v.get(k) == old[k] for k in ('original_input_ref', 'original_request_ref', 'old_created_at_utc', 'old_expires_at_utc', 'old_reservation_key', 'actual_wait_ref', 'outer_wait_ref', 'outer_tool', 'outer_returncode', 'direct_worker_returncode', 'owned_drain', 'completed_ref', 'old_charge_bytes', 'recorded_operational_failure', 'actual_cancel_signal', 'exact_sites_written'))) and (v.get('old_launch_resume_write_authority_revoked') is True) and (v.get('old_files_scopes_key_charge_clocks_and_failure_unchanged') is True) and (v.get('original_conversion_unsealed') is True), 'Genuine exact failed12977 acceptance, no completion/cause/credit relabel')
     for key in ('original_input_ref', 'original_request_ref', 'actual_wait_ref', 'outer_wait_ref'):
         c.raw(old[key])
     latest = b['setupv3_failure']
     accepted = v.get('latest_setupv3_failure_acceptance')
     c.require(isinstance(accepted, dict) and accepted.get('schema') == 'ROOT_FLARE_setupv3_failed_unsealed_acceptance_v1' and (accepted.get('root_authorizer') == 'root_agent') and (accepted.get('accepted') is True) and all((type(accepted.get(k)) is type(value) and accepted.get(k) == value for (k, value) in latest.items())) and (accepted.get('old_launch_resume_write_authority_revoked') is True) and (accepted.get('old_files_scopes_key_charge_clocks_and_failure_unchanged') is True) and (accepted.get('original_conversion_unsealed') is True), 'Genuine exact97739outer1 worker1/drain acceptance; no old clock extension, reuse, credit or success')
     for key in ('original_input_ref', 'original_request_ref', 'actual_wait_ref', 'outer_wait_ref', 'failure_diagnosis_ref'):
         c.raw(latest[key])
 def batch_instruction(record):
  v=c.read(record)
  c.require(v.get('schema')=='ROOT_FLARE_VAL002004_conversion_metadata_preparation_v1'
   and v.get('root_authorizer')=='root_agent'and v.get('active')is True and v.get('metadata_preparation_only')is True
   and v.get('admission_authorized')is False and v.get('launch_authorized')is False
   and v.get('ROOT_reviewed_preparer_source')is True and v.get('conversion_request_metadata_authorized')is True
   and v.get('preparation_source_ref')==c.ref(c.HERE+'/prepare_ROOT_v1.py')
   and v.get('source_manifest_ref')==c.ref(c.MANIFEST)and set(v.get('targets',{}))==set(c.VAL),
   'Separate exact reviewed ROOT metadata instruction for three original parents')
  c.reviewed(v['source_review_ref']);c.selected(v['storage_policy_selector_ref'])
  created=datetime.fromisoformat(v['created_at_utc']);expires=datetime.fromisoformat(v['expires_at_utc'])
  c.require(created.tzinfo is not None and expires.tzinfo is not None and(expires-created).total_seconds()==8*3600,
   'Fresh finite8h metadata instruction retains full18600combined work and margin')
  identities=[];names=[]
  for parent in c.VAL:
   row=v['targets'][parent];ids=row['cpu_ids'];label='VAL'+parent[-3:]
   c.require(type(ids)is list and len(ids)==len(set(ids))==4 and all(type(i)is int and i>=0 for i in ids),'Explicit ROOT CPU4 group for every original parent')
   identities.extend(ids)
   prefixes=dict(preparation_directory='research/reviews/FLARE_remaining_ROOT_preparation_'+label+'_convert_',
    packet_directory='research/reviews/FLARE_remaining_ROOT_actual_'+label+'_convert_',output_directory='data/FLARE_original_remaining_'+label+'_convert_')
   for key,prefix in prefixes.items():
    c.require(type(row[key])is str and row[key].startswith(prefix),'Fresh parent-specific original conversion namespace')
    p=c.local(row[key]);c.require(all(not(p==q or p in q.parents or q in p.parents)for q in names),'All nine preparation/packet/output targets disjoint')
    names.append(p)
  c.require(len(set(identities))==12,'All three actual CPU4 groups must be disjoint; no CPU reuse')
  historical_failure_acceptance_v1(b['existing_VAL001_failure_acceptance_ref'])
  completed=c.read(b['existing_VAL001_completed_conversion_ref'])
  c.require(completed.get('status')=='completed_C3_FLARE_observation_conversion'and completed.get('parent_id')=='american_c3_val_001'
   and completed.get('input_sites')==227316 and completed.get('query_ancestry_truth_opened')is False,
   'Preserve genuine completed VAL001 history; no rerun or transplanted predecessor')
  return v
 c.validate_batch_instruction_v1=batch_instruction
 def validate_request(record):
  q=c.read(record);parent=q.get('parent_id')
  c.require(q.get('stage')=='convert'and parent in c.VAL,'Only fresh full original VAL002/003/004 conversions')
  authority=batch_instruction(q['ROOT_metadata_preparation_instruction_ref']);slot=authority['targets'][parent]
  c.require(all(q.get(k)==slot[k]for k in('preparation_directory','packet_directory','output_directory','cpu_ids'))
   and all(q.get(k)==authority[k]for k in('created_at_utc','expires_at_utc','storage_policy_selector_ref','source_review_ref')),
   'Exact ROOT-authorized parent scopes/CPU4/current Census source and clocks')
  key='H041/FLARE-original-convert-'+Path(q['packet_directory']).name
  c.require(q.get('proposed_reservation_key')==key and q.get('historical_VAL001_failure_acceptance_ref')==b['existing_VAL001_failure_acceptance_ref']
   and q.get('historical_failure_parent_id')=='american_c3_val_001','Exact new key; original failures never relabelled as new-parent failure')
  condition=b['fresh_condition'];budget=phase_budget('convert')
  c.require(q.get('phase_budget')==budget and all(type(q.get(k))is type(value)and q[k]==value for k,value in condition.items()),'Exact full3h scientific condition; no partial reuse or runtime-result claim')
  registration=c.read(q['registration_ref']);registered=registration.get('fresh_completion_condition_registration')
  c.require(type(registered)is dict and registered.get('schema')=='FLARE_original_parent_fresh3h_conversion_registration_v1'
   and registered.get('root_authorizer')=='root_agent'and registered.get('parent_id')==parent and registered.get('registered_before_request')is True
   and registered.get('source_manifest_ref')==c.ref(c.MANIFEST)and registered.get('source_review_ref')==q['source_review_ref']
   and registered.get('ROOT_metadata_preparation_instruction_ref')==q['ROOT_metadata_preparation_instruction_ref']
   and registered.get('proposed_reservation_key')==key and registered.get('phase_budget')==budget
   and registered.get('historical_VAL001_failure_acceptance_ref')==b['existing_VAL001_failure_acceptance_ref']
   and all(type(registered.get(k))is type(value)and registered[k]==value for k,value in condition.items()),'Genuine prospective original-target3h registration')
  c.require(datetime.fromisoformat(registered['registered_utc'])<=datetime.fromisoformat(q['created_at_utc']),'Target condition registered before request')
  directive=c.read(q['root_directive_ref'])
  c.require(all(directive.get(k)==q[k]for k in('phase_budget','ROOT_metadata_preparation_instruction_ref','historical_VAL001_failure_acceptance_ref','historical_failure_parent_id','proposed_reservation_key'))
   and all(type(directive.get(k))is type(value)and directive[k]==value for k,value in condition.items()),'Separate ROOT directive retains original condition and history')
  return original_validate(record)
 c.validate_request=validate_request
 def direct_roles(request_ref,q,cfgref,closure):
  roles=original_roles(request_ref,q,cfgref,closure)
  for r in(c.DEPS['SELECTOR'],c.GUARD_REF,q['historical_VAL001_failure_acceptance_ref'],q['ROOT_metadata_preparation_instruction_ref']):
   if r not in roles:roles.append(copy.deepcopy(r))
  c.require(c.GUARD_REF==c.selected(q['storage_policy_selector_ref'])['source_ref']and c.GUARD_REF in roles,'Exact Census child remains direct; original static selector preserved')
  return roles
 c.direct_roles=direct_roles
 def input_spec(record):
  result=original_input(record);s,q,cfg=result
  c.require(s['storage']['reservation']['key']==q['proposed_reservation_key']
   and s['job_id']==q['proposed_reservation_key'].split('/',1)[1],'Actual admitted input/key binds its exact new original parent')
  return result
 c.input_spec=input_spec
