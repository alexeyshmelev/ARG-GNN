"""Independent source/compact-count audit. No project/science/guard imports or execution."""
from pathlib import Path
import ast, copy, hashlib, json, stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/American_C3_native_paths_v3_independent_review_20261015_v1"
PKG="research/reviews/American_C3_native_paths_root_v3"
OLD="research/reviews/American_C3_native_paths_root_v2"
CONSUMER="src/arggnn/american_c3_native_paths_consumer_v3.py"
ORIGINAL="src/arggnn/american_c3_paths_v1.py"

def raw(p):return (ROOT/p).read_bytes()
def text(p):return raw(p).decode()
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def ref(p):return dict(path=p,sha256=sha(p))
def parse(p):return ast.parse(text(p),p)
def funcs(p):return {x.name:x for x in parse(p).body if isinstance(x,ast.FunctionDef)}
def dump(x):return ast.dump(x,include_attributes=False)
def message(x):
    if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and x.value.args:
        end=x.value.args[-1]
        if isinstance(end,ast.Constant) and isinstance(end.value,str):return end.value
    return None

# Explicitly remove the additions that were inspected against the actual source,
# and replace budget-only call sites. No executable producer definitions are loaded.
class ScienceComparison(ast.NodeTransformer):
    def visit_ImportFrom(self,n):
        if n.module=="american_c3_common_v1":return None
        return n
    def visit_Assign(self,n):
        targets={x.id for t in n.targets for x in ast.walk(t) if isinstance(x,ast.Name)}
        if targets & {"reference_paths","reference_tokens","max_raw"}:return None
        n=self.generic_visit(n)
        if targets=={"planned"}:
            n.value=ast.parse("5*sum(map(len,selected))+32*len(selected)+64*v+pickle_margin",mode="eval").body
        return n
    def visit_If(self,n):
        if dump(n.test)==dump(ast.parse("key[0]<600",mode="eval").body):return None
        return self.generic_visit(n)
    def visit_Expr(self,n):
        if message(n) in {
            "Exact original first600 reference / last1000 query role partition required; no fallback",
            "Exact full original tree dimensions for occurrence serialization required",
            "Exact full-count maximum29 required for serialization proof",
            "Exact complete corpus/reference counts required before durable products; no subset fallback",
            "Exact original complete vocabulary required"}:return None
        return self.generic_visit(n)
    def visit_Call(self,n):
        n=self.generic_visit(n)
        if isinstance(n.func,ast.Name) and n.func.id=="npz_save_v3":
            assert dump(n.args[0])==dump(ast.Name(id="w",ctx=ast.Load()))
            n.func=ast.Attribute(value=n.args[0],attr="npz",ctx=ast.Load());n.args=n.args[1:]
        if isinstance(n.func,ast.Attribute) and n.func.attr=="pickle" and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=="sequence_to_tree_indices.pkl":
            n.args[2]=ast.parse("5*tokens+64*len(seqs)+5*ts.num_samples*ts.num_trees+pickle_margin",mode="eval").body
        if isinstance(n.func,ast.Name) and n.func.id=="dict" and any(k.arg=="status" and isinstance(k.value,ast.Constant) and k.value.value=="completed_native_C3_path_preparation" for k in n.keywords):
            added={"reference_unique_paths","reference_path_tokens","max_raw_path_nodes","budget_successor_source_ref","scientific_preparation_original_source_ref","exact_per_sample_counts_ref","native_serialization_bound_bytes"}
            n.keywords=[k for k in n.keywords if k.arg not in added]
        return n

manifest=json.loads(text(PKG+"/source_manifest.json"))
source_hashes={p:sha(p) for p in manifest["source_sha256"]}
metadata_hashes={k:ref(v["path"]) for k,v in manifest["metadata_refs"].items()}
checks={"all_manifest_source_pins_match":source_hashes==manifest["source_sha256"],
        "all_manifest_metadata_pins_match":metadata_hashes==manifest["metadata_refs"],
        "reviewed_consumer_hash":sha(CONSUMER)=="3957212a54e852575173fcdbc12d4e4fabf6bb08e44f2fe4008634d6263fb4e6"}
new=funcs(CONSUMER);old=funcs(ORIGINAL)
checks["whole_science_AST_equal_after_inspected_additions_and_budget_calls_only"]=dump(ScienceComparison().visit(copy.deepcopy(new["execute"])))==dump(old["execute"])
newc=funcs(PKG+"/native_paths_contract_v3.py");oldc=funcs(OLD+"/native_paths_contract_v2.py")
for name in ("validate_counts_v1","original_guard_v1","validate_close_receipt_v1"):
    checks["frozen_contract_function_unchanged:"+name]=dump(newc[name])==dump(oldc[name])
checks["owned_RSS_function_unchanged"]=dump(funcs(PKG+"/root_native_paths_driver.py")["owned_group_RSS_v1"])==dump(funcs(OLD+"/root_native_paths_driver.py")["owned_group_RSS_v1"])
exact_path="research/reviews/American_C3_native_paths_capacity_source_v2/actual_counts_v1/completed.json"
per_path="research/reviews/American_C3_native_paths_capacity_source_v2/actual_counts_v1/per_sample_counts.json"
assert len(raw(per_path))<=1048576
exact=json.loads(text(exact_path));detail=json.loads(text(per_path));rows=detail["per_sample_counts"]
checks["count_receipt_per_sample_binding"]=exact["per_sample_counts_ref"]==ref(per_path)
checks["all_1600_count_rows_ordered_and_builtin_integral"]=len(rows)==1600 and all(type(r) is dict and r["sample_index"]==r["sample_node"]==i and all(type(x) is int for x in r.values()) for i,r in enumerate(rows))
p,t,o=(sum(r[k] for r in rows) for k in ("unique_paths","path_tokens","path_occurrences"))
pr,tr=(sum(r[k] for r in rows[:600]) for k in ("unique_paths","path_tokens"))
v=exact["vocabulary_nodes"];l=max(r["max_raw_path_nodes"] for r in rows);f=10000;n=p+pr+f
checks["independent_count_sums"]=(p,t,o,v,l,pr,tr)==(25574587,316672003,122980800,106769,29,9583977,116080787)
checks["each_count_row_visits_all_trees"]=all(r["path_occurrences"]==76863 and 0<r["unique_paths"]<=76863 and r["unique_paths"]<=r["path_tokens"]<=29*r["unique_paths"] and 0<r["max_raw_path_nodes"]<=29 for r in rows)
k=1600*(256*2+(65536-256)*3+(76863-65536)*5)
margin=16*16777216
terms=dict(raw_path_integer_opcodes=5*(2*t+tr+l*f),containers=8*n+8*p,tree_index_integer_opcodes=k,
    row_sidecar_int64_payloads=24*n+8*pr,reference_labels=5*pr,vocabulary=7*64*v,fixed_margin=margin,
    root_log=16777216,root_metadata=1048576)
aggregate=sum(terms.values());native=aggregate-terms["root_log"]-terms["root_metadata"]
checks["independent_aggregate_arithmetic"]=(native,aggregate,6000000000-aggregate)==(5924203578,5942029370,57970630)
map_container_unframed=7*p+2*(o//1000)+2*((p+999)//1000)+1024
checks["mapping_container_allowance_includes_empty_final_set_batches"]=map_container_unframed<8*p
# Source-fixed sidecar scalar fields and focal arrays, with sum-of-ceil allowances.
sidecar_scalars=6*(4*len("american_c3_train_000")+3*4*3)
focal_raw=6*8*f+4*len("american_c3_train_000")+3*4*3
npz_raw=terms["row_sidecar_int64_payloads"]+sidecar_scalars+focal_raw
npz_extra_over_row_term=sidecar_scalars+focal_raw+(npz_raw+99)//100+6+7*1048576
checks["NPZ_expansion_focal_metadata_fit_six_remaining_margins"]=npz_extra_over_row_term<6*16777216
serializer_path=".runtime/env/lib/python3.11/pickle.py"
serializer_actual=ref(serializer_path)
checks["actual_pickle_py_source_hash"]=serializer_actual["sha256"]=="227b9550b1491f5ed3bf7b444a97f91113d47a3bb0e82fb49da24cbc255d407f"
direct=[*manifest["native_scientific_closure"],*manifest["integration_closure_refs"],*manifest["metadata_refs"].values()]
checks["serializer_source_not_a_direct_guard_binding"]=all(r["path"]!=serializer_path for r in direct)
checks["named_direct_bindings_remain_single_link_regular_canonical"]=all((ROOT/r["path"]).resolve()==ROOT/r["path"] and stat.S_ISREG((ROOT/r["path"]).lstat().st_mode) and (ROOT/r["path"]).lstat().st_nlink==1 and (ROOT/r["path"]).stat().st_size<=32*1024*1024 for r in direct)
for pth in (CONSUMER,PKG+"/native_paths_contract_v3.py",PKG+"/root_native_paths_driver.py"):
    compile(text(pth),pth,"exec")
checks["compile_text_only_no_project_execution"]=True
assert all(checks.values()),checks
result=dict(schema="American_C3_native_paths_v3_independent_source_audit_v1",checks=checks,
    source_sha256=source_hashes,metadata_refs=metadata_hashes,exact_counts=dict(unique_paths=p,path_tokens=t,path_occurrences=o,vocabulary_nodes=v,max_raw_path_nodes=l,first600_reference_unique_paths=pr,first600_reference_path_tokens=tr,remaining1000_unique_paths=p-pr,remaining1000_path_tokens=t-tr),
    independent_bound_terms_bytes=terms,native_bound_bytes=native,aggregate_bound_bytes=aggregate,cap_bytes=6000000000,cap_margin_bytes=6000000000-aggregate,
    mapping_container_bound_before_frames_and_fixed_headers=map_container_unframed,mapping_8P_allowance=8*p,
    NPZ_raw_bytes_including_focals_and_scalar_fields=npz_raw,NPZ_extra_over_sidecar_row_term_bytes=npz_extra_over_row_term,
    residual_six_margin_bytes=6*16777216-npz_extra_over_row_term,
    serializer_actual_source_ref=serializer_actual,serializer_source_link_count=(ROOT/serializer_path).stat().st_nlink,
    executed_project_or_guard_modules=False,scientific_data_labels_roster_tree_checkpoint_reads=False,accounting_admission_process_GPU_or_Git_actions=False,
    limits="Source/compact existing metadata evidence only; no empirical serialization, focal generation, native file comparison, live runtime verification or capacity assessment. Public CPython3.11.17 _pickle.c was separately fetched as source, not imported/executed, and matched b8f6875c572368b62fcd2edeb7c094a77acd2aa58943d96e26b29485800d958f.")
with (OUT/"source_audit_receipt.json").open("x") as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write("\n")
print(json.dumps(dict(checks=len(checks),all_passed=all(checks.values()),aggregate_bound_bytes=aggregate,receipt_ref=ref((OUT/"source_audit_receipt.json").relative_to(ROOT).as_posix()))))
