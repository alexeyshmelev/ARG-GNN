"""Private source-authenticated selector binding only; passed scientific bodies remain exact."""
import ast,copy,importlib.util,sys
from . import contract_v1 as c
_INTEGRATION=_ENTRY=None
_SELECTED=None
def integration_v1(selector_ref):
 global _INTEGRATION,_SELECTED
 c.configure_selector_v1(selector_ref)
 if _INTEGRATION is not None:
  c.require(_SELECTED==selector_ref,"Same scientific binding selector for lifetime")
  return _INTEGRATION
 c.raw(c.SAVED_SCIENCE_REVIEW_REF);review=c.read(c.SAVED_SCIENCE_REVIEW_REF)
 c.require(review.get("verdict")=="SOURCE_PASS"and review.get("blocking_findings",review.get("blockers",[]))==[],
  "Passed saved-MLM scientific integration review stays separate from operational review")
 module=c.load(c.SAVED_INTEGRATION_REF,"_Census64_original_saved_MLM_scientific_integration")
 source=c.raw(c.SAVED_INTEGRATION_REF).decode();tree=ast.parse(source)
 fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="bind_stage_v1"))
 target=ast.dump(ast.parse('PINS["current_selector"]',mode="eval").body,include_attributes=False);count=[0]
 class ActualSelector(ast.NodeTransformer):
  def visit_Subscript(self,node):
   if ast.dump(node,include_attributes=False)==target:
    count[0]+=1;return ast.copy_location(ast.Name(id="_actual_Census_selector_ref_v1",ctx=ast.Load()),node)
   return self.generic_visit(node)
 fn=ActualSelector().visit(fn);c.require(count[0]==2,"Only two exact original selector-comparison operands change")
 ns=module.__dict__;ns["_actual_Census_selector_ref_v1"]=copy.deepcopy(selector_ref);ast.fix_missing_locations(fn)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(c.local(c.SCIENCE_BINDING))+":actual_selector_operands","exec"),ns)
 _SELECTED=copy.deepcopy(selector_ref);_INTEGRATION=module
 return module
def entry_v1(selector_ref):
 global _ENTRY
 integration=integration_v1(selector_ref)
 if _ENTRY is None:
  source=c.raw(c.SAVED_SCIENTIFIC_ENTRY_REF).decode();tree=ast.parse(source)
  matches=[n for n in tree.body if isinstance(n,ast.ImportFrom)and any(a.asname=="integration"for a in n.names)]
  c.require(len(matches)==1,"Only original scientific entry integration import is operationally bound")
  tree.body=[ast.Assign(targets=[ast.Name(id="integration",ctx=ast.Store())],value=ast.Name(id="_actual_scientific_integration_v1",ctx=ast.Load()))if n is matches[0]else n for n in tree.body]
  ast.fix_missing_locations(tree);module=type(integration)("_Census64_original_scientific_entry")
  module.__dict__["_actual_scientific_integration_v1"]=integration
  exec(compile(tree,str(c.local(c.SAVED_SCIENTIFIC_ENTRY_REF["path"]))+":same_private_integration_import","exec"),module.__dict__)
  _ENTRY=module
 return _ENTRY
