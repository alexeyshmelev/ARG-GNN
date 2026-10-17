"""Tiny pure integer structural checks and AST equivalence; no scientific imports."""
from pathlib import Path
import ast,builtins,hashlib,json,symtable
HERE=Path("research/reviews/H060_isolated_hold_compatibility_source_20261017_v1")
OLD=Path("research/reviews/American_TreeLoom_warmstart_training_source_20261017_v2")
def need(ok,message):
 if not ok:raise AssertionError(message)
def parsed(path):return ast.parse(Path(path).read_text(),filename=str(path))
def functions(tree):return {x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
def dump(node):return ast.dump(node,include_attributes=False)
def suffix(function,first_name):
 for at,node in enumerate(function.body):
  if isinstance(node,ast.Assign)and any(isinstance(t,ast.Tuple)and any(isinstance(x,ast.Name)and x.id==first_name for x in t.elts)
   or isinstance(t,ast.Name)and t.id==first_name for t in node.targets):return function.body[at:]
 raise AssertionError(first_name)
def main():
 files=sorted(HERE.glob("*.py"))
 for p in files:
  s=p.read_text();compile(s,str(p),"exec");table=symtable.symtable(s,str(p),"exec")
  bound={x.get_name()for x in table.get_symbols()if x.is_assigned()or x.is_imported()or x.is_namespace()}
  def visit(t):
   for x in t.get_symbols():
    if x.is_referenced()and x.is_global():need(x.get_name()in bound or x.get_name()in vars(builtins)or x.get_name()=="__file__","Unresolved global "+x.get_name())
   for child in t.get_children():visit(child)
  visit(table)
 # Execute ONLY four pure Python bookkeeping helpers; no arrays/floats/Torch.
 helpers=parsed(HERE/"structural_index_v1.py")
 scope={};exec(compile(helpers,"pure_integer_structural_helpers","exec"),scope)
 classify=scope["classify_sample_roots"];partition=scope["validate_root_partition"];clades=scope["isolated_clades"]
 parent=[3,3,3,-1,-1,-1,-1,10,10,10,-1]
 head=[-1,-1,-1,2,-1,-1,-1,-1,-1,-1,9]
 root_of=[3,3,3,3,4,5,6,10,10,10,10]
 rows=[(0,1,2),(4,5,6),(7,8,9)]
 ordered=[classify(parent,head,root_of,row)for row in rows]
 need(ordered==[3,(4,5,6),10],"Native connected/isolated/connected roots, no virtual node")
 need(partition(ordered,(3,4,5,6,10),(1,))==(1,),"Every actual root retained exactly once")
 need(clades(ordered[1],rows[1],parent,head)==({},3,0),"No proper clade; every isolated native sample visited")
 checks=["connected_isolated_connected_native_roots","all_real_roots_exactly_once","isolated_empty_clades_all_nodes_visited"]
 cases=[
  ("mixed_informative_forest",lambda:classify([3,3,-1,-1],[-1,-1,-1,1],[3,3,2,3],(0,1,2))),
  ("duplicate_sample",lambda:classify(parent,head,root_of,(4,4,6))),
  ("missing_isolated_declaration",lambda:partition(ordered,(3,4,5,6,10),())),
  ("hidden_native_component",lambda:partition(ordered,(3,4,5,6,10,11),(1,))),
  ("reused_component_across_holds",lambda:partition([3,3],(3,),())),
  ("isolated_sample_with_child",lambda:clades((4,5,6),(4,5,6),parent,head[:4]+[0]+head[5:])),
 ]
 for label,call in cases:
  try:call()
  except ValueError:checks.append(label+"_rejected")
  else:raise AssertionError(label)
 original=functions(parsed(OLD/"warmstart_train_v2.py"));own=functions(parsed(HERE/"warmstart_train_v2.py"))
 need(set(original)==set(own),"No added/replaced training algorithm")
 for name in original:
  if name not in("_global_bundle_v1","_signature_v1"):need(dump(original[name])==dump(own[name]),"Unchanged runner algorithm "+name)
 # Entire model/loss/optimizer implementations remain direct original imports.
 imports=[x for x in parsed(HERE/"warmstart_train_v2.py").body if isinstance(x,ast.ImportFrom)]
 need(any(x.module=="research.reviews.American_TreeLoom_warmstart_training_source_20261017_v2.warmstart_models_v1"for x in imports),
  "Original model/loss/optimizer identities retained")
 # The original saved600 Dataset and provider constructor method bodies are exact.
 own_classes={x.name:x for x in parsed(HERE/"observation_provider_v1.py").body if isinstance(x,ast.ClassDef)}
 for path,name,method in[
  ("src/arggnn/american_c3_gnn_dataset_isolated_v2.py","AmericanC3WholeTreeDatasetV1","__getitem__"),
  ("src/arggnn/american_c3_gnn_stream_train_isolated_v2.py","SealedObservationProviderV1","__init__")]:
  old_class=next(x for x in parsed(path).body if isinstance(x,ast.ClassDef)and x.name==name)
  old_method=next(x for x in old_class.body if isinstance(x,ast.FunctionDef)and x.name==method)
  new_method=next(x for x in own_classes[name].body if isinstance(x,ast.FunctionDef)and x.name==method)
  need(dump(old_method)==dump(new_method),"Exact effective saved600 method "+name+"."+method)
 old_obs=functions(parsed("src/arggnn/american_c3_gnn_observations_isolated_v2.py"))
 own_obs=functions(parsed(HERE/"observations_v1.py"))
 for name in("donor_frequencies","native_tree","_layout_with_isolated_samples"):need(dump(old_obs[name])==dump(own_obs[name]),"Original isolated helper "+name)
 # Original structural recurrence/group/threshold algebra stays byte-equivalent AST.
 old_episode=functions(parsed("src/arggnn/ordered_clade_episode_gnn_v1.py"))["build_clade_episode_arrays_v1"]
 own_episode=functions(parsed(HERE/"episode_preparation_v1.py"))["build_clade_episode_arrays_v1"]
 need([dump(x)for x in suffix(old_episode,"flags")]==[dump(x)for x in suffix(own_episode,"flags")],
  "Exact original proper-clade episode flags/arrays/algebra")
 old_cut=functions(parsed("research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py"))["prepare_cut_plan_v1"]
 own_cut=functions(parsed(HERE/"cut_preparation_v1.py"))["prepare_cut_plan_v1"]
 need([dump(x)for x in suffix(old_cut,"representatives")]==[dump(x)for x in suffix(own_cut,"representatives")],
  "Exact original350cut/group/episode/plan algebra")
 call_names={ast.unparse(x.func)for p in files for x in ast.walk(parsed(p))if isinstance(x,ast.Call)}
 need(not any(x.endswith((".simplify",".add_row",".delete_sites",".keep_intervals"))for x in call_names),
  "No pruning/virtual-root/fake-edge operation")
 need("_connected_encode_block"in call_names and "_episodes.build_clade_episode_arrays_v1"in call_names
  and "_original.prepare_cut_plan_v1"in call_names,"Connected recipes delegate original implementations")
 return dict(schema="H060_isolated_hold_compatibility_source_checks_v1",status="SOURCE_CHECKS_PASS_NOT_INDEPENDENT_REVIEW",
  pure_integer_structural_checks=checks,compiled_sources=[str(p)for p in files],runner_algorithm_AST_unchanged=True,
  original_model_loss_optimizer_identities=True,effective_saved600_provider_methods_AST_unchanged=True,
  original_episode_and_cut_numeric_suffix_AST_unchanged=True,connected_delegation=True,
  no_Torch_or_numpy_imports=True,no_payloads_opened=True,no_FP64_or_neural_or_update_tests=True,
  actual_guards_resources_jobs_or_Git=False,execution_authorized=False)
if __name__=="__main__":print(json.dumps(main(),sort_keys=True,indent=2))
