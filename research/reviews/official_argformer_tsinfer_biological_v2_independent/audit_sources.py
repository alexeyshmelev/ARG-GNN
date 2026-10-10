import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OWN=ROOT/"research/reviews/official_argformer_tsinfer_biological_v2_independent"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def new_json(name,value):
    with (OWN/name).open("x") as f:json.dump(value,f,sort_keys=True,indent=2);f.write(chr(10))
expected={
"src/arggnn/official_argformer_tsinfer_biological_v2.py":"24fbf9c4f06e532fc87fe9ee570ca875e0cad56ff5c719cfd64230930eac760e",
"configs/official_argformer_tsinfer_biological_v2.json":"502d1de0bc548b936753e84a9e1e1807474768d528bdf68fa90cbb8c921ba0a8",
"research/reviews/official_argformer_tsinfer_biological_v2_author/review_receipt.json":"eaf9a5e0853a20b88198f0362dafdbb438f0645ece8cb62b0dcd4a7b7f27d6a4"}
actual={p:sha(ROOT/p) for p in expected};assert actual==expected
config=json.loads((ROOT/"configs/official_argformer_tsinfer_biological_v2.json").read_text())
receipt=json.loads((ROOT/"research/reviews/official_argformer_tsinfer_biological_v2_author/review_receipt.json").read_text())
source_pins={p:sha(ROOT/p) for p in config["frozen_files_sha256"]}
assert source_pins==config["frozen_files_sha256"]==receipt["all_original_files_pinned_and_unchanged"]
author=ROOT/config["author_source"]["root"]
records=[]
for p in sorted(author.rglob("*")):
    assert not p.is_symlink()
    if p.is_file() and p.name!="ARG_GNN_SOURCE_RECEIPT.json":
        records.append({"path":p.relative_to(author).as_posix(),"sha256":sha(p)})
tree_sha=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
assert len(records)==60 and tree_sha==config["author_source"]["tree_sha256"]
function_names={
"src/arggnn/train_official_argformer_biological.py":("sha","canonical","json_new","observed_path","load_frozen","prepared_inputs","cache_loss","cache_adapter","object_tensor_manifest","native_runtime","source_model","assert_backbone","audit_loader","audit_callback","train_phase","memory_or_parity","readout","verify_encoder_initializer","execute_native","bootstrap"),
"scripts/score_official_argformer_retrieval_biological_v1.py":("corpus_roles","validate_support","lexicographic_topk","exhaustive_neighbors","hard_votes","project_grid")}
proof={}
for p,names in function_names.items():
    nodes={n.name:n for n in ast.parse((ROOT/p).read_text()).body if isinstance(n,ast.FunctionDef)}
    proof[p]={name:hashlib.sha256(ast.dump(nodes[name],include_attributes=False).encode()).hexdigest() for name in names}
assert proof==receipt["source_only_scientific_invariance"]["native_function_ASTs"]
fit=ast.parse((ROOT/"src/arggnn/train_official_argformer_biological.py").read_text())
prepare=next(n for n in fit.body if isinstance(n,ast.FunctionDef) and n.name=="prepare_paths")
start=next(i for i,n in enumerate(prepare.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="tree_path" for t in n.targets))
tail=hashlib.sha256(ast.dump(ast.Module(body=prepare.body[start:],type_ignores=[]),include_attributes=False).encode()).hexdigest()
assert tail==receipt["source_only_scientific_invariance"]["preparation_tail_AST_sha256"]
owned_sources={p:sha(ROOT/p) for p in receipt["owned_source_sha256"]}
assert owned_sources==receipt["owned_source_sha256"]
artifact_pins={p:sha(ROOT/p) for p in receipt["artifact_sha256"]}
assert artifact_pins==receipt["artifact_sha256"]
spec=importlib.util.spec_from_file_location("independent_source_audit",ROOT/"src/arggnn/official_argformer_tsinfer_biological_v2.py")
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
report=bridge.source_report(bridge.policy())
assert report["native_function_ASTs"]==proof and report["preparation_tail_AST_sha256"]==tail
metadata=bridge.metadata_report(bridge.policy())
assert len(metadata["parents"])==4 and not metadata["arrays_opened"]
new_json("source_pins.json",{"input_handoff_pins":actual,"owned_source_pins":owned_sources,"preserved_original_pins":source_pins,"native_function_ASTs":proof,"preparation_tail_AST_sha256":tail,"author60_files":records,"author60_tree_sha256":tree_sha,"author_artifact_pins":artifact_pins})
new_json("source_report.json",report)
new_json("metadata_report.json",metadata)
print(json.dumps({"input_handoff_pins_exact":True,"original_frozen_files":len(source_pins),"native_ASTs":len(proof["src/arggnn/train_official_argformer_biological.py"]),"retrieval_ASTs":len(proof["scripts/score_official_argformer_retrieval_biological_v1.py"]),"preparation_tail_exact":True,"author_files":len(records),"four_parent_JSON_metadata":True,"array_tree_model_payloads_opened":False},sort_keys=True))
