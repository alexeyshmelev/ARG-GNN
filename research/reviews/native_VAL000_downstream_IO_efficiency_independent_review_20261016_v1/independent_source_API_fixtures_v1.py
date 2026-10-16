"""Independent source/API fixtures; stdlib only, no scientific imports or payloads."""
import ast, copy, hashlib, json, math, sys, types, __future__
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OWN=ROOT/"research/reviews/native_VAL000_downstream_IO_efficiency_independent_review_20261016_v1"
CAND=ROOT/"research/reviews/native_VAL000_downstream_IO_efficiency_source_20261016_v1"
ORIG=ROOT/"research/reviews/native_VAL000_downstream_scientific_stages_source_20261016_v1"
EXPECTED={
 str(CAND.relative_to(ROOT)/"source_handoff.v1.json"):"a10ce4add8fc5420ffa1a9d216ddfab168605cae41fb419b2e2416169ed16976",
 str(CAND.relative_to(ROOT)/"source_manifest.v1.json"):"ff4630314ddc36d3e87b8b8c1927931cf05f4c08d91cf043d7632a9dd3267aa3",
 str(ORIG.relative_to(ROOT)/"source_handoff.v1.json"):"e4056a2c500c4bfb0817252f89b95e7e9a941496d20c4d39bd7c9c323b7891fe"}
reads={}
def rb(path):
    path=Path(path); assert path.resolve()==path and path.is_relative_to(ROOT)
    assert path.suffix in {".py",".json",".md",".yaml"} and path.stat().st_size<=32*2**20
    b=path.read_bytes(); reads[str(path.relative_to(ROOT))]={"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
    return b
def verify(r):
    b=rb(ROOT/r["path"]); assert hashlib.sha256(b).hexdigest()==r["sha256"],r["path"]
for p,h in EXPECTED.items(): verify({"path":p,"sha256":h})
manifest=json.loads(rb(CAND/"source_manifest.v1.json"))
def walk(o):
    if isinstance(o,dict):
        if "path" in o and "sha256" in o: verify(o)
        for v in o.values(): walk(v)
    elif isinstance(o,list):
        for v in o: walk(v)
walk(manifest);walk(json.loads(rb(CAND/"source_handoff.v1.json")))
walk(json.loads(rb(ORIG/"source_handoff.v1.json")))
src=rb(CAND/"efficient_downstream_v1.py").decode()
tree=ast.parse(src); original=ast.parse(rb(ORIG/"scientific_stages_v1.py"))
def fn(t,name): return next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def run(nodes,ns):
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),"<independent-source-API-fixture>","exec",flags=__future__.annotations.compiler_flag),ns)
def dump(n):return ast.dump(n,include_attributes=False)
results=[]
def passed(name,**detail): results.append(dict(test=name,status="PASS",**detail))
# Exact train and final checkpoint calls remain identical to original consumer.
for target in ("module.train","helper._final_checkpoint"):
    def call(t,n):
        return next(x for x in ast.walk(fn(t,n)) if isinstance(x,ast.Call) and ast.unparse(x.func)==target)
    assert dump(call(tree,"contrastive_keep1_v1"))==dump(call(original,"contrastive_v1"))
passed("unchanged_author_train_and_final_checkpoint_calls")
# Candidate import-free definitions and helpers are compiled, never imported.
ns={"Path":Path,"ast":ast,"copy":copy,"types":types,"ROOT":ROOT,"SOURCE":str((CAND/"efficient_downstream_v1.py").relative_to(ROOT))}
ns["IO_POLICY_REF"]=ast.literal_eval(next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="IO_POLICY_REF").value)
run([fn(tree,"require"),fn(tree,"RAMCombinationWriter")],ns)
class Writer:
    path=OWN
    def __init__(self):self.calls=[]
    def save(self,*args,**kwargs):self.calls.append((args,kwargs));return {"fixture_receipt":len(self.calls)}
w=Writer(); outputs={v+"/"+s:{"embedding_ref":{"fixture":v+"/"+s}} for v in ("query_paths","contrastive") for s in ("train","val")}
prep={"full_path_id_order_refs":{k:{"fixture_order":k} for k in outputs}}
ram=ns["RAMCombinationWriter"](w,{"outputs":outputs,"checkpoint_ref":{"fixture":"final2135"}},prep)
for v in ("query_paths","contrastive"):
    value={"embeddings":object()};order={"record_ids":object()}
    desc=ram.save(v+"_train_then_val_embeddings.pt",value,kind="torch")
    assert "path" not in desc and "sha256" not in desc and desc["actual_combined_payload_file_exists"] is False
    assert ram.values[v] is value
    ram.save(v+"_train_then_val_full_path_order.npz",order,kind="npz")
ram.save("completed.json",{"status":"completed_C3_native_CPU_row_preserving_concat"})
assert len(w.calls)==1 and w.calls[0][0][0]=="RAM_combination.json"
try:ram.save("query_paths_train_then_val_embeddings.pt",{},kind="torch")
except ValueError:pass
else:raise AssertionError("duplicate RAM write accepted")
passed("RAM_writer_preserves_identity_and_explicit_virtual_descriptors",persistent_calls=1)
# Execute actual candidate clone constructor with a source-extracted original function.
concat_tree=ast.parse(rb(ROOT/"src/arggnn/american_c3_native_concat_v1.py"))
concat_ns={};run([fn(concat_tree,"concatenate_embeddings_v1")],concat_ns)
orig_concat=concat_ns["concatenate_embeddings_v1"]
refs_node=next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="REFS")
refs=ast.literal_eval(refs_node.value)
def source_bytes(r):verify(r);return rb(ROOT/r["path"])
concat_mod=types.SimpleNamespace(__file__=str(ROOT/refs["concat"]["path"]),concatenate_embeddings_v1=orig_concat)
ns.update(REFS=refs,source_bytes=source_bytes,importlib=types.SimpleNamespace(import_module=lambda name:concat_mod))
run([fn(tree,"_original_concat_function_v1")],ns)
callbacks={k:(lambda *a,**kw:None) for k in ("preparation_v1","counts_v1","fit_completion_v1","metadata")}
cloned=ns["_original_concat_function_v1"]({"validated_VAL000_"+k:v for k,v in callbacks.items()})
assert cloned.__code__ is orig_concat.__code__
assert all(cloned.__globals__[k] is v for k,v in callbacks.items())
passed("candidate_concat_clone_keeps_exact_code_object_and_four_metadata_callbacks")
# Candidate seal constructor changes only the existing embedding-load AST.
eval_tree=ast.parse(rb(ROOT/refs["evaluator"]["path"]))
eval_ns={};run([fn(eval_tree,"native_top1_seal")],eval_ns)
eval_mod=types.SimpleNamespace(__file__=str(ROOT/refs["evaluator"]["path"]),native_top1_seal=eval_ns["native_top1_seal"])
ns["importlib"]=types.SimpleNamespace(import_module=lambda name:eval_mod)
run([fn(tree,"_original_RAM_seal_v1")],ns)
adapted=ns["_original_RAM_seal_v1"]({},ram)
resolver=adapted.__globals__["_load_RAM_embedding_v1"]
for v in ("query_paths","contrastive"):assert resolver(ram.descriptor(v,"RAM_tensors")) is ram.values[v]
try:resolver({"fixture":"alien"})
except ValueError:pass
else:raise AssertionError("alien RAM input accepted")
expected=fn(eval_tree,"native_top1_seal")
capture={}
def capture_compile(t,*a,**kw):
    capture["tree"]=copy.deepcopy(t);return compile(t,*a,**kw)
ns["compile"]=capture_compile
adapted=ns["_original_RAM_seal_v1"]({},ram)
changed=copy.deepcopy(capture["tree"].body[0])
nested=next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name=="embedding")
orig_nested=next(n for n in expected.body if isinstance(n,ast.FunctionDef) and n.name=="embedding")
assert ast.unparse(nested.body[0].value)=="_load_RAM_embedding_v1(record)"
nested.body[0].value=copy.deepcopy(orig_nested.body[0].value)
assert dump(changed)==dump(expected)
passed("candidate_seal_exact_one_input_load_substitution_reverses_to_original_AST")
# Extract the candidate's exact RAM main-clone statements and exercise original main with API fakes.
retrieve_tree=ast.parse(rb(ROOT/refs["author_retrieval"]["path"]))
class Vec:
    def __init__(self,name,shape):self.name=name;self.shape=shape
class Torch:
    cuda=types.SimpleNamespace(is_available=lambda:True)
    saved=None
    @staticmethod
    def device(name):assert name=="cuda";return name
    @staticmethod
    def is_tensor(value):return False
    @staticmethod
    def save(value,path):Torch.saved=(value,path)
E=Vec("E",(3,768));Q=Vec("Q",(2,768));B=Vec("scores",(2,1));I=Vec("indices",(2,1));normal_calls=[];retrieval_calls=[]
def normalize(x,p,dim):
    assert p==2 and dim==1;normal_calls.append(x);return x
def topk(Q,E,**kw):retrieval_calls.append((Q,E,kw));return B,I
main_ns={"torch":Torch,"F":types.SimpleNamespace(normalize=normalize),"argparse":__import__("argparse"),"topk_indices_streaming":topk}
run([fn(retrieve_tree,"ensure_l2_normalized"),fn(retrieve_tree,"labels_from_metadata"),fn(retrieve_tree,"main")],main_ns)
mod=types.SimpleNamespace(main=main_ns["main"])
ram.values={"contrastive":{"embeddings":E,"labels":[2,0,1],"metadata":{}},"query_paths":{"embeddings":Q,"metadata":{}}}
bridge_fn=fn(tree,"RAM_retrieval_and_seal_v1")
start=next(i for i,n in enumerate(bridge_fn.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="tokens")
end=next(i for i,n in enumerate(bridge_fn.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="main")
bridge_ns={"ram":ram,"module":mod,"types":types,"require":ns["require"]}
run(bridge_fn.body[start:end+1],bridge_ns)
assert bridge_ns["main"].__code__ is mod.main.__code__
argv=sys.argv;sys.argv=["fixture","--corpus","ARGGNN_RAM_reference_train_then_val","--queries","ARGGNN_RAM_query_train_then_val","--out","VIRTUAL.pt","--topk","1","--q_batch","1024","--e_chunk","100000"]
try:bridge_ns["main"]()
finally:sys.argv=argv
assert normal_calls==[E,Q] and retrieval_calls==[(Q,E,{"device":"cuda","k":1,"q_batch":1024,"e_chunk":100000})]
saved,path=Torch.saved
assert path=="VIRTUAL.pt" and saved["top1_indices"] is I and saved["top1_scores"] is B and saved["corpus_labels"]==["2","0","1"] and "query_labels" not in saved
try:bridge_ns["load_RAM"]("other.pt")
except ValueError:pass
else:raise AssertionError("alien retrieval load accepted")
passed("actual_candidate_RAM_bridge_runs_original_main_with_identity_and_API_fakes",numerical_test=False,unchanged_topk_source_sha256=refs["author_retrieval"]["sha256"])
# Execute installed scheduler and rotate code using in-memory API fakes.
misc=ast.parse(rb(ROOT/refs["interval_scheduler"]["path"]));scheduler=copy.deepcopy(fn(misc,"create_interval_scheduler"))
scheduler.body=[n for n in scheduler.body if not isinstance(n,ast.ImportFrom)]
units=types.SimpleNamespace(**{k:k for k in ("EPOCH","ITERATION","BATCH","TOKEN","SAMPLE","DURATION","SECOND")})
events=types.SimpleNamespace(**{k:k for k in ("EPOCH_CHECKPOINT","ITERATION_CHECKPOINT","BATCH_CHECKPOINT","EPOCH_END","ITERATION_END","BATCH_END")})
class Interval:
    unit="BATCH";value=20
    def __int__(self):return 20
class Time:
    @staticmethod
    def from_input(interval,unit):assert interval=="20ba";return Interval()
scns={"math":math,"Event":events,"State":object,"Time":Time,"TimeUnit":units};run([scheduler],scns)
scheduled=scns["create_interval_scheduler"]("20ba")
class Stamp:
    def __init__(self,b):self.batch=b
    def get(self,unit):return self.batch
    def state_dict(self):return {"batch":self.batch}
class State:
    def __init__(self,b):self.timestamp=Stamp(b);self.previous_timestamp=Stamp(b-1)
    def get_elapsed_duration(self):return self.timestamp.batch/2135
event_batches=[];last=-1
for b in range(1,2136):
    for ev in ("BATCH_CHECKPOINT","EPOCH_CHECKPOINT"):
        if scheduled(State(b),ev) and last!=b:event_batches.append(b);last=b
assert event_batches==list(range(20,2135,20))+[2135]
saver=ast.parse(rb(ROOT/refs["checkpoint_saver"]["path"]))
klass=fn(saver,"CheckpointSaver")
rot=fn(klass,"_rotate_checkpoints");state_dict=fn(klass,"state_dict")
virtual=set();removed=[]
def remove(p):assert p in virtual;virtual.remove(p);removed.append(p)
rns={"Path":Path,"os":types.SimpleNamespace(remove=remove)}
run([rot,state_dict],rns)
rows=[]
for keep in (-1,1):
    virtual.clear();removed.clear()
    obj=types.SimpleNamespace(saved_checkpoints=[],num_checkpoints_to_keep=keep,all_saved_checkpoints_to_timestamp={})
    peak=0
    for b in event_batches:
        p="/VIRTUAL/new/ep1-ba%d-rank0.pt"%b;virtual.add(p);peak=max(peak,len(virtual))
        obj.all_saved_checkpoints_to_timestamp[p]=Stamp(b)
        snapshot=rns["state_dict"](obj)
        assert len(snapshot["all_saved_checkpoints_to_timestamp"])==event_batches.index(b)+1
        obj.saved_checkpoints.append(p)
        if keep>=0:rns["_rotate_checkpoints"](obj,False)
    assert len(virtual)==(107 if keep==-1 else 1)
    assert "/VIRTUAL/new/ep1-ba2135-rank0.pt" in virtual
    rows.append({"keep":keep,"save_events":len(event_batches),"peak_regular":peak,"retained_regular":len(virtual),"removed":len(removed),"serialized_timestamp_history":len(snapshot["all_saved_checkpoints_to_timestamp"])})
assert rows[1]["peak_regular"]==2 and rows[1]["removed"]==106
passed("installed_scheduler_rotation_and_timestamp_state_fixture",rows=rows,actual_files_deleted=0)
# Source-derived payload and cap arithmetic.
rows={"contrastive/train":8748479,"contrastive/val":972757,"query_paths/train":9003,"query_paths/val":997}
payload={k:n*(768*4+128*8+128*8+8+(8 if k.startswith("contrastive/") else 0)) for k,n in rows.items()}
total=sum(payload.values());sidecars=9721236*4*8+10000*3*8
assert total==49979548096 and sidecars==311319552
proposal=json.loads(rb(CAND/"storage_proposal.v1.json"))
assert proposal["exact_original_four_embedding_tensor_payload_bytes"]==payload
assert total+sidecars==proposal["duplicate_disk_payload_bytes_avoided"]==50290867648
assert sum(proposal["proposed_separate_future_scope_budgets"][k] for k in ("RAM_retrieval_seal_bytes","contrastive_bytes","original_four_embedding_outputs_bytes"))==65*2**30
assert (108-2)*3*2**30==proposal["checkpoint_comparison"]["comparison_reduction_of_file_bound_bytes"]==341449900032
passed("independent_payload_and_hypothetical_cap_arithmetic",embedding_tensor_payload_bytes=total,sidecar_payload_bytes=sidecars,total_proposed_bytes=65*2**30)
report={"schema":"native_VAL000_downstream_IO_independent_source_API_fixture_results_v1","status":"PASS","tests":results,
"scientific_imports":False,"payload_or_truth_or_weights_read":False,"actual_runtime_or_numerical_equivalence_test":False,
"jobs_probes_GPU_packages_accounting_Git":False,"real_checkpoint_deletions":0,"input_file_count":len(reads)}
(OWN/"fixture_results.v1.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
(OWN/"source_hashes.v1.json").write_text(json.dumps({"schema":"independent_review_source_read_hash_inventory_v1","files":reads},sort_keys=True,indent=2)+"\n")
print(json.dumps({"status":"PASS","test_count":len(results),"input_file_count":len(reads),"result":str((OWN/"fixture_results.v1.json").relative_to(ROOT))}))
