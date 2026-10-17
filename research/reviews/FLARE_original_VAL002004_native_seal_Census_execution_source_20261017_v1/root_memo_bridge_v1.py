"""Narrow Census source bridge for accepted ROOT reserve memo v2; import inert.
Original memo transaction/cache/final checks are preserved. No public guard or operation here.
"""
from pathlib import Path
import ast,copy

def derive_memo_methods_v1(base,source,census_ref,uncached_paths,bridge_ref):
 tree=ast.parse(source);classes=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='RootMutationMemo']
 base.need(len(classes)==1,'Exact pinned reviewed memo class')
 methods={n.name:n for n in classes[0].body if isinstance(n,ast.FunctionDef)}
 check=copy.deepcopy(methods['_check']);run=copy.deepcopy(methods['_run'])
 patterns=[ast.parse("PINS['guard']",mode='eval').body,ast.parse("(() if authority_ref is None else (authority_ref['path'],))",mode='eval').body]
 counts=[0,0]
 class Exact(ast.NodeTransformer):
  def visit(self,node):
   for i,pattern in enumerate(patterns):
    if ast.dump(node,include_attributes=False)==ast.dump(pattern,include_attributes=False):
     counts[i]+=1
     value=ast.Name(id='_CENSUS_GUARD_REF',ctx=ast.Load())if i==0 else ast.parse("_CURRENT_UNCACHED_PATHS + (() if authority_ref is None else (authority_ref['path'],))",mode='eval').body
     return ast.copy_location(value,node)
   return super().visit(node)
 check=Exact().visit(check);run=Exact().visit(run)
 base.need(counts==[1,1],'Only exact child-source authentication and extra current-path exclusion operands change')
 def stable_source(record):
  value=base.source_bytes(record)
  base.source_bytes(bridge_ref) # Exact source bridge also reauthenticated before return/append.
  return value
 scope=dict(base.__dict__,_CENSUS_GUARD_REF=copy.deepcopy(census_ref),_CURRENT_UNCACHED_PATHS=tuple(uncached_paths),source_bytes=stable_source)
 code=ast.fix_missing_locations(ast.Module(body=[check,run],type_ignores=[]))
 exec(compile(code,bridge_ref['path']+'#exact-reviewed-memo-v2-Census-source-join','exec',dont_inherit=True,optimize=0),scope)
 return dict(_check=scope['_check'],_run=scope['_run']),counts

def current_uncached_paths_v1(c,q,request_ref,b):
 paths=set()
 def collect(value):
  if type(value)is dict:
   if set(value)=={'path','sha256'}and type(value['path'])is str and Path(value['path']).suffix=='.json':paths.add(value['path'])
   else:
    for child in value.values():collect(child)
  elif type(value)in(list,tuple):
   for child in value:collect(child)
 collect(request_ref);collect(q)
 manifest=c.read(q['source_manifest_ref']);collect(manifest)
 for path in manifest['source_sha256']:
  if Path(path).suffix=='.json':paths.add(path)
 for r in(q['ROOT_metadata_preparation_instruction_ref'],q['context_ref'],q['registration_ref'],q['root_directive_ref'],q['storage_policy_selector_ref']):collect(c.read(r))
 selector=c.read(q['storage_policy_selector_ref']);policy=c.read(selector['policy_ref']);collect(policy);collect(c.read(policy['root_activation_ref']))
 # Config/closure/authority are staged after constructor. Their exact future names stay uncached.
 prep=q['preparation_directory'];stem=Path(q['packet_directory']).name
 for name in(stem+'.config.json',stem+'.source_and_input_closure.json',stem+'.reserve_authority.json'):paths.add(prep+'/'+name)
 c.require(len(paths)<=2048,'Bounded explicit uncached metadata/control paths')
 return tuple(sorted(paths))

def make_memo_v1(g,q,request_ref,c,b,base):
 review=c.read(b['ROOT_memo_review_ref'])
 c.require(review.get('independent')is True and review.get('verdict')=='SOURCE_PASS'
  and review.get('source_ref')==b['ROOT_memo_source_ref']and review.get('blockers')==[],'Exact accepted memo v2 component source review before constructor')
 source=c.raw(b['ROOT_memo_source_ref']);base.source_bytes(b['ROOT_memo_source_ref']);base.source_bytes(b['ROOT_memo_bridge_ref'])
 uncached=current_uncached_paths_v1(c,q,request_ref,b)
 methods,counts=derive_memo_methods_v1(base,source,b['selected_child_source_ref'],uncached,b['ROOT_memo_bridge_ref'])
 Memo=type('ExactCensusRootMutationMemoV2',(base.RootMutationMemo,),methods)
 manifest=c.sources();cache=[dict(path=path,sha256=sha)for path,sha in manifest['source_sha256'].items()
  if Path(path).suffix in('.py','.sh')and Path(path).parts[0]in('src','scripts','research')]
 return Memo(g,integration_manifest_ref=q['source_manifest_ref'],adapter_ref=b['ROOT_memo_source_ref'],cache_refs=cache)

def derive_admission_v1(node):
 # Source authentication precedes this exact original source-shape assertion.
 result=copy.deepcopy(node)
 expected=("g = c.guard(False)","impl = c.ref(g.budget.SOURCE_REL)","launcher = c.ref(g.budget.LAUNCHER_REL)")
 matched=[ast.unparse(n)for n in node.body[10:13]]
 if matched!=list(expected):raise ValueError('Exact original guard construction positions')
 block=node.body[13]
 if not(isinstance(block,ast.With)and len(block.items)==1 and block.items[0].optional_vars is None
  and ast.unparse(block.items[0].context_expr)=='g.budget.v4.mutex(c.ROOT)'
  and len(block.body)==1 and ast.unparse(block.body[0])=="state = g._state_locked(impl['sha256'], launcher['sha256'], fresh=True)"):
  raise ValueError('Exact original first locked fresh state call')
 # Configure exact child and construct memo before first staging write; staging order unchanged.
 pre=copy.deepcopy(node.body[10:13])+ast.parse('_transaction_memo = _memo_for_guard(g,q,request_ref)').body
 state=ast.parse("state = _transaction_memo.read_state_v1(expected_source_sha256=impl['sha256'], expected_launcher_sha256=launcher['sha256'])").body
 result.body=copy.deepcopy(node.body[:7])+pre+copy.deepcopy(node.body[7:10])+state+copy.deepcopy(node.body[14:])
 pattern=ast.parse("g.apply_reserve_v1(ar, expected_source_sha256=impl['sha256'], expected_launcher_sha256=launcher['sha256'])",mode='eval').body
 hits=[0]
 class Reserve(ast.NodeTransformer):
  def visit_Call(self,n):
   if ast.dump(n,include_attributes=False)==ast.dump(pattern,include_attributes=False):
    hits[0]+=1;n=copy.deepcopy(n);n.func=ast.Attribute(value=ast.Name(id='_transaction_memo',ctx=ast.Load()),attr='reserve_v1',ctx=ast.Load());return n
   return self.generic_visit(n)
 result=Reserve().visit(result)
 if hits!=[1]:raise ValueError('Exactly one unchanged reserve call site')
 return ast.fix_missing_locations(result)

def install_admission_v1(driver,c,b):
 base=c.load(b['ROOT_memo_source_ref'],'_FLARE_exact_reviewed_ROOT_memo_v2')
 original,node=base.functions_from_source(driver,b['original_driver_ref'],('admit',))['admit']
 derived=derive_admission_v1(node)
 scope=dict(original.__globals__,_memo_for_guard=lambda g,q,r:make_memo_v1(g,q,r,c,b,base))
 exec(compile(ast.Module(body=[derived],type_ignores=[]),b['ROOT_memo_bridge_ref']['path']+'#exact-original-admission-two-transactions','exec',dont_inherit=True,optimize=0),scope)
 driver.admit=scope['admit']
