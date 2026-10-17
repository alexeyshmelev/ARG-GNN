"""Inactive native/seal joins for original VAL002–004. No operational call on import."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy

def derive_conversion_evidence_node_v1(node):
 require=lambda ok,message:None if ok else (_ for _ in()).throw(ValueError(message))
 require(isinstance(node,ast.FunctionDef)and node.name=='stage_evidence'and not node.decorator_list,'Exact original predecessor evidence function')
 patterns=[(ast.parse(text,mode='eval').body,name)for text,name in(
  ('ref(DRIVER)','_CONVERSION_DRIVER_REF'),('ref(ENTRY)','_CONVERSION_ENTRY_REF'),("sources()['source_sha256']",'_CONVERSION_SOURCE_SHA256'))]
 counts=[0,0,0]
 class Exact(ast.NodeTransformer):
  def visit(self,n):
   for i,(pattern,name)in enumerate(patterns):
    if ast.dump(n,include_attributes=False)==ast.dump(pattern,include_attributes=False):
     counts[i]+=1;return ast.copy_location(ast.Name(id=name,ctx=ast.Load()),n)
   return super().visit(n)
 result=Exact().visit(copy.deepcopy(node));result.name='_conversion_stage_evidence_v1'
 require(counts==[1,1,1],'Exactly three predecessor-source comparison operands change')
 return ast.fix_missing_locations(result)

def install_v1(c,b,phase_budget):
 original_validate=c.validate_request;original_roles=c.direct_roles;original_input=c.input_spec
 original_evidence=c.stage_evidence;original_caps=c.stage_caps
 source=c.raw(b['original_contract_ref']);node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='stage_evidence')
 derived=derive_conversion_evidence_node_v1(node)
 # Globals stay the live contract dictionary: later RAM readers and exact selected source remain effective.
 c.__dict__.update(_CONVERSION_DRIVER_REF=copy.deepcopy(b['conversion_driver_ref']),_CONVERSION_ENTRY_REF=copy.deepcopy(b['conversion_worker_ref']))
 exec(compile(ast.Module(body=[derived],type_ignores=[]),b['operational_join_ref']['path']+'#original-three-source-operands','exec',dont_inherit=True,optimize=0),c.__dict__)
 def stage_evidence(record,*,stage,parent,completion_ref):
  c.require(parent in c.VAL and stage in('convert','native'),'Only original002–004 genuine predecessor stages')
  if stage=='convert':
   m=c.read(b['conversion_manifest_ref'])
   c.require(isinstance(m.get('source_sha256'),dict),'Exact passed conversion source map')
   review=c.read(b['conversion_source_review_ref'])
   c.require(review.get('independent')is True and review.get('verdict')=='SOURCE_PASS'and review.get('blockers')==[]
    and review.get('source_ref')==b['conversion_driver_ref']and review.get('source_manifest_ref')==b['conversion_manifest_ref']
    and review.get('source_handoff_ref')==b['conversion_handoff_ref'],'Exact whole conversion source review; no arbitrary previous-source substitution')
   c._CONVERSION_SOURCE_SHA256=m['source_sha256']
   return c._conversion_stage_evidence_v1(record,stage=stage,parent=parent,completion_ref=completion_ref)
  return original_evidence(record,stage=stage,parent=parent,completion_ref=completion_ref)
 c.stage_evidence=stage_evidence
 def stage_caps(stage,cfg):
  caps=original_caps(stage,cfg);budget=phase_budget(stage)
  c.require(stage in('native','seal')and caps['cpu_threads']==4 and caps['rss_bytes']==(24 if stage=='native'else 4)*2**30
   and caps['scientific_seconds']==budget['scientific_seconds']and caps['wall_seconds']==budget['wall_seconds']
   and caps['reservation_bytes']==caps['output_bytes']+8*2**20,'Exact original count-derived output+8MiB, scientific/wall/CPU/RSS caps')
  return caps
 c.stage_caps=stage_caps
 # The full historical predicate is reused byte-for-byte in AST from the passed conversion integration.
 history_tree=ast.parse(c.raw(b['conversion_operational_join_ref']))
 history_install=next(n for n in history_tree.body if isinstance(n,ast.FunctionDef)and n.name=='install_v1')
 history=next(n for n in history_install.body if isinstance(n,ast.FunctionDef)and n.name=='historical_failure_acceptance_v1')
 history=copy.deepcopy(history);history.name='_historical_VAL001_failure_acceptance_v1'
 scope=dict(c=c,b=b)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[history],type_ignores=[])),b['operational_join_ref']['path']+'#unchanged-historical-VAL001-only','exec',dont_inherit=True,optimize=0),scope)
 def instruction(record):
  v=c.read(record);stage=v.get('stage');parent=v.get('parent_id')
  c.require(v.get('schema')=='ROOT_FLARE_VAL002004_native_seal_metadata_preparation_v1'
   and v.get('root_authorizer')=='root_agent'and v.get('active')is True and v.get('metadata_preparation_only')is True
   and v.get('admission_authorized')is False and v.get('launch_authorized')is False
   and v.get('ROOT_reviewed_preparer_source')is True and v.get('one_stage_request_metadata_authorized')is True
   and v.get('preparation_source_ref')==c.ref(c.HERE+'/prepare_ROOT_v1.py')
   and v.get('source_manifest_ref')==c.ref(c.MANIFEST)and parent in c.VAL and stage in c.STAGES,
   'Explicit separate reviewed ROOT instruction for exactly one original parent and one native/seal stage')
  c.reviewed(v['source_review_ref']);c.selected(v['storage_policy_selector_ref'])
  created=datetime.fromisoformat(v['created_at_utc']);expires=datetime.fromisoformat(v['expires_at_utc'])
  c.require(created.tzinfo is not None and expires.tzinfo is not None and(expires-created).total_seconds()==8*3600,
   'Fresh finite8h authority; no predecessor clock reset')
  ids=v['cpu_ids'];c.require(type(ids)is list and len(ids)==len(set(ids))==4 and all(type(i)is int and i>=0 for i in ids),'Explicit ROOT CPU4')
  names=[];label='VAL'+parent[-3:]
  for key,prefix in dict(preparation_directory='research/reviews/FLARE_remaining_ROOT_preparation_'+label+'_'+stage+'_',
   packet_directory='research/reviews/FLARE_remaining_ROOT_actual_'+label+'_'+stage+'_',output_directory='data/FLARE_original_remaining_'+label+'_'+stage+'_').items():
   c.require(type(v[key])is str and v[key].startswith(prefix),'Fresh exact parent/stage-specific namespace')
   p=c.local(v[key]);c.require(all(p!=q and p not in q.parents and q not in p.parents for q in names),'Preparation/packet/output disjoint');names.append(p)
  prior_stage='convert'if stage=='native'else'native';previous=v['conversion_ref']if stage=='native'else v['native_run_ref']
  e=c.stage_evidence(v['previous_stage_evidence_ref'],stage=prior_stage,parent=parent,completion_ref=previous)
  inp=c.read(e['input_ref']);q=c.read(inp['root_stage_request_ref'])
  c.require(q.get('parent_id')==parent and q.get('stage')==prior_stage and q.get('context_ref')==v['context_ref']
   and q.get('source_manifest_ref')==(b['conversion_manifest_ref']if stage=='native'else c.ref(c.MANIFEST)),
   'Reuse exact actual predecessor context/count/plan metadata and same parent; no alternate support')
  if stage=='native':
   c.require(v.get('native_run_ref')is None,'Native cannot reuse an earlier native result')
  else:
   c.require(q.get('conversion_ref')==v['conversion_ref'],'Seal keeps exact native conversion lineage')
  row=c.completion()._context(v['context_ref'],parent_id=parent)['targets'][parent]
  expected=b['original_targets'][parent]['target_row_for_later_ROOT_binding']
  c.require(all(row.get(k)==value for k,value in expected.items()if k!='count_manifest_ref')and isinstance(row.get('count_manifest_ref'),dict),
   'Every original scientific identity/support ref unchanged; count comes from actual prior context')
  scope['_historical_VAL001_failure_acceptance_v1'](b['existing_VAL001_failure_acceptance_ref'])
  c.current(v,phase_budget(stage)['combined_seconds']+600)
  return v
 c.validate_stage_instruction_v1=instruction
 def validate_request(record):
  q=c.read(record);v=instruction(q['ROOT_metadata_preparation_instruction_ref']);stage=q.get('stage')
  keys=('stage','parent_id','conversion_ref','native_run_ref','previous_stage_evidence_ref','context_ref','preparation_directory','packet_directory','output_directory','cpu_ids','created_at_utc','expires_at_utc','storage_policy_selector_ref','source_review_ref')
  c.require(all(q.get(k)==v[k]for k in keys),'Exact one-stage ROOT instruction metadata/scopes/clocks/CPU/source joins')
  key='H041/FLARE-original-'+stage+'-'+Path(q['packet_directory']).name
  c.require(q.get('proposed_reservation_key')==key and q.get('phase_budget')==phase_budget(stage)
   and q.get('historical_VAL001_failure_acceptance_ref')==b['existing_VAL001_failure_acceptance_ref']
   and q.get('historical_failure_parent_id')=='american_c3_val_001'
   and q.get('reuse_partial_outputs')is False and q.get('efficiency_improvement_claimed')is False
   and q.get('paper_runtime_result_claimed')is False,'Exact new key/full stage; historical failures remain VAL001 only; no runtime/fit claim')
  reg=c.read(q['registration_ref']);stamp=reg.get('one_stage_source_registration')
  c.require(type(stamp)is dict and stamp.get('schema')=='FLARE_original_parent_native_seal_source_registration_v1'
   and stamp.get('registered_before_request')is True and stamp.get('registered_utc')==q['created_at_utc']
   and stamp.get('ROOT_metadata_preparation_instruction_ref')==q['ROOT_metadata_preparation_instruction_ref']
   and stamp.get('source_manifest_ref')==c.ref(c.MANIFEST)and stamp.get('source_review_ref')==q['source_review_ref']
   and stamp.get('parent_id')==q['parent_id']and stamp.get('stage')==stage and stamp.get('phase_budget')==q['phase_budget']
   and stamp.get('proposed_reservation_key')==key,'Prospective exact original parent/stage source registration')
  directive=c.read(q['root_directive_ref'])
  c.require(all(directive.get(k)==q[k]for k in('phase_budget','ROOT_metadata_preparation_instruction_ref','historical_VAL001_failure_acceptance_ref','historical_failure_parent_id','proposed_reservation_key','reuse_partial_outputs','efficiency_improvement_claimed','paper_runtime_result_claimed')),'Separate exact original stage ROOT directive')
  return original_validate(record)
 c.validate_request=validate_request
 def direct_roles(request_ref,q,cfgref,closure):
  roles=original_roles(request_ref,q,cfgref,closure)
  for r in(c.DEPS['SELECTOR'],c.GUARD_REF,q['historical_VAL001_failure_acceptance_ref']):
   if r not in roles:roles.append(copy.deepcopy(r))
  c.require(c.ref(c.DRIVER)==c.ref(c.ENTRY)and c.GUARD_REF==c.selected(q['storage_policy_selector_ref'])['source_ref']and c.GUARD_REF in roles,
   'Same authenticated driver/worker source is one distinct ref; every selected guard role remains direct')
  return roles
 c.direct_roles=direct_roles
 def input_spec(record):
  result=original_input(record);s,q,cfg=result
  c.require(s['storage']['reservation']['key']==q['proposed_reservation_key']and s['job_id']==q['proposed_reservation_key'].split('/',1)[1],
   'Exact actual admitted original stage/key')
  return result
 c.input_spec=input_spec
