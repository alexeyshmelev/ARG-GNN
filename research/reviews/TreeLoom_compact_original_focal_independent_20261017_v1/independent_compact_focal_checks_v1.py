"""Independent compact focal equivalence on a distinct tiny RAM genealogy."""
from pathlib import Path
import ast,copy,hashlib,io,json
import numpy as np
import tskit
B=Path("research/reviews/TreeLoom_compact_original_focal_source_20261017_v1")
checks=[]
def check(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def need(ok,label):
 if not ok:raise ValueError(label)
def reject(fn,label):
 try:fn()
 except(ValueError,KeyError,TypeError):checks.append(label);return
 raise AssertionError("Bad control/fixture accepted "+label)
def ref(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
manifest=json.loads((B/"source_manifest.v1.json").read_text())
check(ref(B/"source_manifest.v1.json")["sha256"]=="0d81e5144865f8db202901182c69dfcde235305b15eaab8049a95fbc3cad540e","Exact compact manifest")
check(ref(B/"compact_focals_v1.py")["sha256"]=="74ec68ceb13264b01bc432f74df375a8b229a81afd0b101b656a2eb98049d2f9","Exact compact core")
for r in manifest["dependency_refs"]+manifest["own_refs"]:
 check(ref(r["path"])==r,"Exact source/metadata pin "+r["path"])
original=Path("src/arggnn/american_c3_paths_v1.py")
author=Path(".runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py")
native_ast=ast.parse(original.read_text());author_ast=ast.parse(author.read_text());candidate_ast=ast.parse((B/"compact_focals_v1.py").read_text())
author_ns=dict(tskit=tskit,random=__import__("random"),RANDOM_SEED=42,SPECIAL_TOKENS={"fixture":0},SPECIAL_OFFSET=4)
for name in["get_path_key","build_vocab_from_sequences","split_unique_sequences"]:
 n=next(n for n in author_ast.body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[n],type_ignores=[]),"original-author#RAM","exec"),author_ns)
native_ns=dict(require=need,CLASSES=("EUR","EAS","AFR"),author_functions=lambda:{k:author_ns[k]for k in["get_path_key","build_vocab_from_sequences","split_unique_sequences"]})
for name in["storage_bound","execute"]:
 n=next(n for n in native_ast.body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[n],type_ignores=[]),"original-native#RAM","exec"),native_ns)
candidate=dict(require=need,CLASSES=("EUR","EAS","AFR"),ast=ast,hashlib=hashlib,json=json,Path=Path)
for n in candidate_ast.body:
 if isinstance(n,ast.Assign)and all(isinstance(v,ast.Name)for v in n.targets):
  exec(compile(ast.Module(body=[n],type_ignores=[]),"compact-constants#RAM","exec"),candidate)
candidate["original_native_v1"]=lambda:__import__("types").SimpleNamespace(author_functions=lambda:{"get_path_key":author_ns["get_path_key"]})
candidate["read"]=lambda r:json.loads(Path(r["path"]).read_text())
candidate["bound"]=lambda r:Path(r["path"])
for name in["focal_arrays_v1","make_config_v1","identity_sha_v1"]:
 n=next(n for n in candidate_ast.body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[n],type_ignores=[]),"compact-exact#RAM","exec"),candidate)
# Distinct6-sample6-tree fixture. Repeat A after B, B after C/D.
t=tskit.TableCollection(sequence_length=137.)
for _ in range(6):t.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0.)
for tm in[1.,1.,1.,3.,4.,1.,1.,1.,5.]:t.nodes.add_row(time=tm)
patterns={
"A":[(6,0),(6,2),(7,1),(7,4),(8,3),(8,5),(9,6),(9,7),(9,8)],
"B":[(6,0),(6,2),(7,1),(7,4),(8,3),(8,5),(10,6),(10,7),(10,8)],
"C":[(11,0),(11,5),(12,1),(12,2),(13,3),(13,4),(14,11),(14,12),(14,13)],
"D":[(11,0),(11,5),(12,1),(12,2),(13,3),(13,4),(10,11),(10,12),(10,13)]}
for left,right,pattern in[(0.,7.,"A"),(7.,19.,"B"),(19.,44.,"A"),(44.,68.,"C"),(68.,106.,"D"),(106.,137.,"B")]:
 for parent,child in patterns[pattern]:t.edges.add_row(left,right,parent,child)
t.sort();ts=t.tree_sequence()
rows=[dict(sample_index=i,observed_node_id=i,role="pure_source_reference"if i<2 else"query")for i in range(6)]
check(ts.num_samples==6 and ts.num_trees==6 and ts.num_sites==0,"Own6sample6tree RAM fixture")
class Capture(Exception):pass
class Writer:
 def __init__(self):self.arrays=None;self.names=[]
 def npz(self,name,**arrays):
  self.names.append(name);self.arrays={k:v.copy()for k,v in arrays.items()};raise Capture()
def original_arrays(cfg):
 observation=dict(schema="american_c3_observed_v1",parent_id=cfg["parent_id"],class_order=["EUR","EAS","AFR"],genealogy_kind=candidate["GENEALOGY"],query_ancestry_truth_in_inputs=False,roster_ref="ram/roster",tree_ref="ram/tree")
 roster=dict(parent_id=cfg["parent_id"],class_order=["EUR","EAS","AFR"],samples=rows)
 native_ns["read"]=lambda r:observation if r=="ram/observation"else roster
 native_ns["bound"]=lambda r:Path("ram/tree")
 oldload=tskit.load;tskit.load=lambda p:ts if p=="ram/tree"else None
 w=Writer()
 try:
  try:native_ns["execute"](dict(cfg,observation_ref="ram/observation",genealogy_kind=candidate["GENEALOGY"],output_cap_bytes=2**30),w)
  except Capture:pass
 finally:tskit.load=oldload
 check(w.names==["focals.npz"],"Original exact execute stopped at first focal NPZ")
 return w.arrays
def packed(arrays):
 f=io.BytesIO();np.savez_compressed(f,**arrays);return f.getvalue()
base=dict(parent_id="independent_RAM6",max_unique_paths=100,max_path_tokens=1000,max_path_occurrences=36,max_vocabulary_nodes=100,max_raw_path_nodes=10,query_focal_count=7,path_sampling_seed=92007)
cases=[]
for seed,count in[(92003,7),(92011,12),(92019,16)]:
 cfg=dict(base,path_sampling_seed=seed,query_focal_count=count)
 expected=original_arrays(cfg);actual,stats=candidate["focal_arrays_v1"](ts,rows,cfg)
 check(list(expected)==list(actual),"Original7field insertion order "+str(seed))
 for k in expected:
  check(expected[k].dtype==actual[k].dtype and expected[k].shape==actual[k].shape and expected[k].tobytes()==actual[k].tobytes(),"Exact array "+k+" seed"+str(seed))
 check(packed(expected)==packed(actual),"Exact same-serializer NPZ bytes "+str(seed))
 check(candidate["identity_sha_v1"](expected)==candidate["identity_sha_v1"](actual),"Original six-field identity "+str(seed))
 check((actual["record_ids"]>=8).all(),"Reference paths counted globally "+str(seed))
 check(not np.isin(actual["positions_bp0"],[31.5,121.5]).any(),"Nonadjacent repeated A/B paths retain first midpoint "+str(seed))
 check(stats==dict(unique_paths=24,path_tokens=72,path_occurrences=36,observed_vocabulary_node_count=15,unique_query_paths=16),"Full traversal stats "+str(seed))
 cases.append(dict(seed=seed,count=count,original_arrays_equal=True,serialized_npz_equal=True,identity=candidate["identity_sha_v1"](actual)))
for key,value in[("max_unique_paths",23),("max_path_tokens",71),("max_path_occurrences",35),("max_vocabulary_nodes",14),("max_raw_path_nodes",2),("query_focal_count",17)]:
 reject(lambda key=key,value=value:candidate["focal_arrays_v1"](ts,rows,dict(base,**{key:value})),"Whole-parent bound rejects "+key)
bad=copy.deepcopy(rows);bad[1]["sample_index"]=5
reject(lambda:candidate["focal_arrays_v1"](ts,bad,base),"Reject changed sample order")
fake=dict(path="RAM_UNOPENED",sha256="0"*64)
for index in[0,1,19]:
 cfg=candidate["make_config_v1"]("american_c3_train_%03d"%index,fake)
 check(cfg["path_sampling_seed"]==92001+index and cfg["query_focal_count"]==10000 and cfg["execution_enabled"]is False,"Original production config inactive seed "+str(index))
 check({k:cfg[k]for k in candidate["ORIGINAL_LIMITS"]}==candidate["ORIGINAL_LIMITS"],"Original production traversal caps "+str(index))
for p in["american_c3_val_000","american_c3_test_000","american_c3_train_020"]:
 reject(lambda p=p:candidate["make_config_v1"](p,fake),"Reject nonTRAIN config "+p)
# Production execute uses the exact checked config and admits only sanitized full inputs.
prod=next(n for n in candidate_ast.body if isinstance(n,ast.FunctionDef)and n.name=="execute")
prodtext=ast.unparse(prod)
check("make_config_v1"in prodtext and "execution_enabled=True"in prodtext.replace(" ",""),"Production exact enabled config check")
check("30000000"in prodtext and "1600"in prodtext and "[None] * 1000"in prodtext,"Production full30Mb1600sample and masked query checks")
check("native_corpus_materialized=False"in prodtext.replace(" ","")and "native_train_val_split_executed=False"in prodtext.replace(" ",""),"Honest compact completion rejects native corpus/split claims")
check(not any(isinstance(n,(ast.Import,ast.ImportFrom))and any(x in ast.unparse(n)for x in["torch","subprocess","resource","socket","fcntl"])for n in ast.walk(candidate_ast)),"Core contains no resource/launcher/GPU imports")
check(manifest["status"]=="SOURCE_ONLY_NO_OPERATIONAL_CONTROLLER"and manifest["whole_scope_reservation_not_derived"]is True,"No operational controller or whole-scope reservation claimed")
result=dict(schema="TreeLoom_compact_focals_independent_RAM_checks_v1",passed=True,check_count=len(checks),checks=checks,cases=cases,author_fixture_reused=False,actual_guard_imported=False,actual_operations=False,real_scientific_payload_reads=False,truth_payload_reads=False,model_execution=False,full_parent_equivalence_claimed=False)
print(json.dumps(result,sort_keys=True,indent=2))
