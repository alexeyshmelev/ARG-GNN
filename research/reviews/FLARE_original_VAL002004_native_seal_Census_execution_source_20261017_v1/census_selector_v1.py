"""Explicit Census-only source join. Import inert; metadata selection configures no guard."""
import ast,copy

def install_v1(c,b):
 profile=b['Census_source_profile'];selected_ref=None;selected_value=None
 def selected(record):
  c.require(type(record)is dict and set(record)=={'path','sha256'},'ROOT must supply exact actual Census selector; no fallback')
  c.raw(profile['source_ref']);tree=ast.parse(c.raw(profile['source_ref']))
  wanted=('selector_draft_v1','_selected_record')
  nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in wanted]
  c.require(len(nodes)==len(wanted),'Exact pinned child metadata selector functions')
  names=('SOURCE','MANIFEST','MEASURE_SOURCE','MODE','JOURNAL','BASE_REF','BASE_POLICY_REF','BASE_SELECTOR_REF','BASE_MODE','CEILING_BYTES','HUMAN_LIMIT_BYTES','WARNING_BYTES')
  scope=dict(copy=copy,_json=c.read,_raw=c.raw,_ref=c.ref,require=c.require,_POLICY_REF=None,_REPAIR_AUTHORITY_REF=None)
  for name in names:
   matches=[n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets)]
   c.require(len(matches)==1,'Exact literal pinned child selector constants')
   scope[name]=ast.literal_eval(matches[0].value)
  exec(compile(ast.Module(body=nodes,type_ignores=[]),profile['source_ref']['path']+'#original-selector-metadata-only','exec',dont_inherit=True,optimize=0),scope)
  value=scope['_selected_record'](record)
  c.require(value['source_ref']==value['selector_source_ref']==profile['source_ref']
   and value['source_manifest_ref']==profile['source_manifest_ref']and value['measurement_source_ref']==profile['measurement_source_ref']
   and value['source_review_ref']==profile['source_review_ref']and value['accounting_mode']==profile['accounting_mode'],
   'Exact independently reviewed Census child source/manifest/measurement/review; old or mixed child rejected')
  review=c.read(profile['source_review_ref'])
  c.require(review.get('independent')is True and review.get('verdict')=='SOURCE_PASS'and review.get('blockers')==[]
   and review.get('source_ref')==profile['source_ref']and review.get('source_manifest_ref')==profile['source_manifest_ref']
   and review.get('measurement_source_ref')==profile['measurement_source_ref'],'Exact genuine independently passed child source')
  policy=c.read(value['policy_ref']);activation=c.read(policy['root_activation_ref'])
  c.require(policy.get('root_authorizer')=='root_agent'and policy.get('schema')=='Census_observation_exact_retained_allowance_policy_v1'
   and activation.get('active')is True and activation.get('root_authorizer')=='root_agent'
   and activation.get('source_ref')==profile['source_ref']and activation.get('source_manifest_ref')==profile['source_manifest_ref']
   and activation.get('ROOT_explicitly_releases_current_selector_for_this_child')is True,
   'Later genuine active ROOT Census policy/activation required, never planning authority')
  return value
 def bind_selector(record):
  nonlocal selected_ref,selected_value
  value=selected(record)
  c.require(selected_ref is None or(selected_ref==record and selected_value==value),'One immutable selected child/source/policy lifetime')
  selected_ref=copy.deepcopy(record);selected_value=copy.deepcopy(value)
  c.SELECTOR_REF=copy.deepcopy(record);c._CHOSEN=copy.deepcopy(value)
  c.SELECTOR=copy.deepcopy(value['selector_source_ref']);c.GUARD_REF=copy.deepcopy(value['source_ref']);c.POLICY_REF=copy.deepcopy(value['policy_ref'])
  return value
 def guard(consumer):
  c.require(type(consumer)is bool and selected_ref is not None and c.SELECTOR_REF==selected_ref and c._CHOSEN==selected_value
   and c.GUARD_REF==c.SELECTOR==profile['source_ref']and c.POLICY_REF==selected_value['policy_ref'],
   'Exact explicit lifetime Census binding before original guard/cache/memo path')
  # Existing singleton/cache-before-consumer-memo implementation and every budget check.
  c._h.selection=selected;c._h.SELECTOR=copy.deepcopy(c.SELECTOR)
  return c._h.guard(selected_ref,consumer)
 c.selected=selected;c.bind_selector=bind_selector;c.guard=guard
