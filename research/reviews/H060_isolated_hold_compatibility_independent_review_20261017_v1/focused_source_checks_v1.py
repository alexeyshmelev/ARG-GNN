from pathlib import Path
import ast,hashlib,json,runpy,sys,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");BASE=ROOT/"research/reviews/H060_isolated_hold_compatibility_source_20261017_v1"
assert hashlib.sha256((BASE/"source_manifest.v1.json").read_bytes()).hexdigest()=="36d8cb6c8dccdf79faa99ce176df0c32e93650f4e01b661e33552128fd71c431"
for source_name,source_sha in json.loads((BASE/"source_manifest.v1.json").read_text())["source_sha256"].items():
 source_path=ROOT/source_name
 assert source_path.suffix in(".py",".json",".md",".patch",".sh") and source_path.resolve()==source_path
 assert hashlib.sha256(source_path.read_bytes()).hexdigest()==source_sha
def parsed(p):return ast.parse(p.read_text(),filename=str(p))
def function(p,name):return next(x for x in parsed(p).body if isinstance(x,ast.FunctionDef)and x.name==name)
def load_function(node,scope):
 exec(compile(ast.Module(body=[node],type_ignores=[]),"<source-only structural function>","exec"),scope)
 return scope[node.name]
def need(ok,msg):
 if not ok:raise AssertionError(msg)
try:
 author=runpy.run_path(str(BASE/"source_check_v1.py"))["main"]()
 need("torch"not in sys.modules and"numpy"not in sys.modules,"No scientific import")
 scope={};exec(compile(parsed(BASE/"structural_index_v1.py"),"<actual pure structural helpers>","exec"),scope)
 classify=scope["classify_sample_roots"];partition=scope["validate_root_partition"];isolated=scope["isolated_clades"]
 cases=[]
 def rejection(name,fn):
  try:fn()
  except ValueError:cases.append(dict(name=name,result="PASS"));return
  raise AssertionError(name)
 need(classify([-1,-1,-1],[-1,-1,-1],[0,1,2],[2,0,1])==(2,0,1),"Original sample order kept")
 cases.append(dict(name="permuted_isolated_sample_order_preserved",result="PASS"))
 rejection("negative_component_root_rejected",lambda:classify([-1]*3,[-1]*3,[-1,1,2],[0,1,2]))
 rejection("wrong_isolated_index_rejected",lambda:partition([3,(4,5,6),10],[3,4,5,6,10],[0]))
 rejection("extra_isolated_declaration_rejected",lambda:partition([3],[3],[0]))
 rejection("isolated_tuple_row_order_mismatch_rejected",lambda:isolated((1,0,2),[0,1,2],[-1]*3,[-1]*3))
 # Execute the actual original pure clade traversal and actual successor wrapper,
 # plus ONLY the actual original-equivalent episode loop on thirteen toy nodes.
 connected_scope={}
 connected=load_function(function(ROOT/"src/arggnn/anonymous_clade_flag_control_v1.py","_tree_clades"),connected_scope)
 wrapper_scope=dict(isolated_clades=isolated,_connected_tree_clades=connected)
 wrapper=load_function(function(BASE/"episode_preparation_v1.py","_tree_clades"),wrapper_scope)
 parent=[3,3,4,4,-1,-1,-1,-1,11,11,12,12,-1]
 head=[-1,-1,-1,1,3,-1,-1,-1,-1,-1,-1,9,11]
 sibling=[-1,0,-1,2,-1,-1,-1,-1,-1,8,-1,10,-1]
 roots=[4,(5,6,7),12];samples=[(0,1,2),(5,6,7),(8,9,10)]
 class Flags:
  def __init__(self,n):self.rows=[[0,0]for _ in range(n)]
  def __setitem__(self,key,value):self.rows[key[0]][key[1]]=value
 flags=Flags(13)
 env=dict(roots=roots,samples=samples,parent=parent,head=head,sibling=sibling,_tree_clades=wrapper,_require=need,
  visited=0,clade_counts=[],node_counts=[],unary_counts=[],previous=None,previous_episode={},
  members=0,episode_count=0,n=13,nodes=[None]*13,holds=[None]*13,episodes=[None]*13,sizes=[0]*13,flags=flags)
 builder=function(BASE/"episode_preparation_v1.py","build_clade_episode_arrays_v1")
 loop=next(x for x in builder.body if isinstance(x,ast.For)and ast.unparse(x.target)=="(hold, (root, row))")
 exec(compile(ast.Module(body=[loop],type_ignores=[]),"<actual structural episode loop>","exec"),env)
 need(env["visited"]==13 and env["node_counts"]==[5,3,5] and env["clade_counts"]==[1,0,1],"Every native vertex preserved")
 need(env["members"]==2 and env["episode_count"]==2 and env["nodes"][:2]==[3,11]and env["holds"][:2]==[0,2]and env["episodes"][:2]==[0,1]and env["sizes"][:2]==[1,1],"No bridging empty hold")
 need(flags.rows[3]==[0,1]and flags.rows[11]==[1,0]and all(flags.rows[u]==[0,0]for u in(5,6,7)),"Actual lost/gained boundaries")
 cases.append(dict(name="actual_episode_loop_breaks_clade_across_isolated_hold_and_visits_all13nodes",result="PASS"))
 # Execute only original-equivalent topology cut traversal at integer time seams.
 cutfn=function(BASE/"cut_preparation_v1.py","prepare_cut_plan_v1")
 cutloop=next(x for x in cutfn.body if isinstance(x,ast.For)and ast.unparse(x.target)=="(tree_id, root)")
 class RootList(list):
  def tolist(self):return list(self)
 cutenv=dict(roots=RootList([0,1,2]),owner=[-1]*3,cut=[-1]*3,p=[-1]*3,times=[0,350,351],HORIZON=350,head=[-1]*3,sibling=[-1]*3,_require=need)
 exec(compile(ast.Module(body=[cutloop],type_ignores=[]),"<actual pure cut topology traversal>","exec"),cutenv)
 representative=next(n.value for n in ast.walk(cutfn)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="representative"for t in n.targets))
 expression=compile(ast.Expression(representative),"<actual representative expression>","eval")
 reps=[eval(expression,dict(node=i,i=i,above=[False,False,True],cut=cutenv["cut"]))for i in range(3)]
 need(cutenv["cut"]==[0,1,-1]and reps==[0,1,2],"350 seam keeps real sample representatives")
 cases.append(dict(name="actual_cut_topology_and_representative_expression_preserve0_350_351_sample_rules",result="PASS"))
 need("torch"not in sys.modules and"numpy"not in sys.modules,"No scientific import after checks")
 print(json.dumps(dict(status="INDEPENDENT_STRUCTURAL_SOURCE_CHECKS_PASS",author_structural_checks=len(author["pure_integer_structural_checks"]),independent_checks=cases,scientific_imports=False,neural_or_FP64_or_optimizer_updates=False,source_check_receipt_matches=(author==json.loads((BASE/"source_check.result.v1.json").read_text()))),sort_keys=True))
except Exception:print(traceback.format_exc())
