from pathlib import Path
import ast,copy,hashlib,importlib,io,json,math,sys,types
from contextlib import redirect_stdout
import numpy as np
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/"research/reviews/European_outer8_preparation_independent_review_20261016_v3"
CAND=ROOT/"research/reviews/European_outer8_preparation_source_20261016_v3"
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
V=importlib.import_module("research.reviews.European_outer8_preparation_source_20261016_v3.outer_prediction_validator_v2")
A=importlib.import_module("research.reviews.European_outer8_preparation_source_20261016_v3.outer_adapters_v2")
E=importlib.import_module("research.reviews.European_outer8_preparation_source_20261016_v3.check_prediction_barrier_v2")
F=importlib.import_module("research.reviews.European_outer8_preparation_source_20261016_v3.software_contract_fixtures_v3")
checks=[]
def ok(name):checks.append(dict(name=name,status="PASS"))
def reject(name,call):
    try:call()
    except (ValueError,KeyError,TypeError,AssertionError):ok(name);return
    raise AssertionError(name)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((CAND/"source_manifest.v1.json").read_text())
verified=[];deferred=[]
for r in manifest["refs"]:
    # No data-path files, real arrays, trees, model payloads, or checkpoints are opened.
    if r["path"].startswith("data/") or Path(r["path"]).suffix.lower() in (".npz",".bin",".pt",".keras",".pkl",".trees"):
        deferred.append(r);continue
    assert sha(ROOT/r["path"])==r["sha256"],r["path"]
    verified.append(r)
assert len(verified)+len(deferred)==226
ok("manifest_nonpayload_source_metadata_pins")
handoff=json.loads((CAND/"source_handoff.v3.json").read_text())
assert sha(CAND/"source_handoff.v3.json")=="49ab4effe1f65d4116bf80080eb0c99a3f9124408e45309a1dd5cce5c7564c05"
assert sha(CAND/"source_manifest.v1.json")=="2bf82a2cffd3b0100eb4a569bb2b8f605ee113e8491c42be953f76d8b8d4f7fb"
assert sha(ROOT/"research/reviews/European_outer8_preparation_independent_review_20261016_v2/review.v2.json")=="e84877c82e311eb1b46b711183fe435eaabc96872c74bc7f73fc40eabb338e97"
for r in handoff["authored_files"]:
    if Path(r["path"]).suffix not in (".bin",".npz"):assert sha(ROOT/r["path"])==r["sha256"]
ok("candidate_handoff_all_authored_nonfixture_files_exact")
prior=[]
for x in handoff["source_seals_preserved"]:
    h=json.loads((ROOT/x["handoff_ref"]["path"]).read_text());assert sha(ROOT/x["handoff_ref"]["path"])==x["handoff_ref"]["sha256"]
    count=0
    for r in h["authored_files"]:
        if Path(r["path"]).suffix in (".npz",".bin"):continue
        assert sha(ROOT/r["path"])==r["sha256"],r["path"];count+=1
    prior.append(dict(version=x["version"],verified_nonpayload_files=count))
ok("prior_v1_v2_seals_preserved")
for p in CAND.glob("*.py"):compile(p.read_text(),str(p),"exec")
plan=json.loads((CAND/"source_clone_plan.v2.json").read_text())
for clone in plan["clones"]:
    path=ROOT/clone["original_source_ref"]["path"];text=path.read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==clone["function"])
    body=ast.get_source_segment(text,node)
    for before,after in clone["exact_literal_replacements"]:
        assert body.count(before)==1,(clone["function"],before);body=body.replace(before,after)
    compile("from __future__ import annotations\n"+body,str(path)+":review_only","exec")
ok("all12_native_exact_literal_clones_compile_without_execution")
old=ROOT/"research/reviews/European_outer8_preparation_source_20261016_v2/prepare_outer000_v1.py"
assert old.read_text().replace("European_outer8_preparation_source_20261016_v2","European_outer8_preparation_source_20261016_v3")== (CAND/"prepare_outer000_v1.py").read_text()
ok("first_parent_wrapper_only_HERE_changed")
config=json.loads((ROOT/"configs/expanded_full_panel_v1.json").read_text());registry=json.loads((CAND/"outer8_registry.v1.json").read_text())
rows=config["parent_registry"]
assert registry["outer_rows"]==[r for r in rows if r["split"]=="outer"]
assert [sum(r["split"]==s for r in rows) for s in ("train","val","outer")]==[32,8,8]
seed_values=[]
for row in rows:
    for value in row["seeds"].values():
        seed_values.extend(value if isinstance(value,list) else [value])
    seed_values.extend(row["relate_size_iteration_seeds"])
    assert row["relate_final_reestimate_seed"]==row["seeds"]["relate_size_base"]+5
assert len(seed_values)==432 and len(set(seed_values))==432
ok("exact48_registry_32TRAIN8VAL8outer_and432_distinct_seeds")
# Candidate fixture constructors create only small in-memory synthetic contracts.
for m in V.METHODS:
    p,r,cell,identity=F.native_case(m)
    V.native_lineage(cell,p,identity,r)
    ok("native_lineage_interface_"+m)
    payload=r.decode_json(cell["actual_wait_ref"]);payload["actual_OS_wait_status"]=1
    bad=r.put(cell["actual_wait_ref"]["path"],payload);broken=dict(cell,actual_wait_ref=bad)
    reject("reject_nonzero_OS_wait_"+m,lambda:V.native_lineage(broken,p,identity,r))
p=F.parent();reader,roster=F.support_reader(p)
for m in V.METHODS:
    z=F.products(p,m);ref=reader.npz("review_synthetic/"+m+".npz",z)
    V.validate_product(ref,p,m,reader,argformer_roster_ref=roster);ok("actual_archive_interface_"+m)
    key="C4_probabilities" if m.startswith("H050") else "source_probabilities" if m=="AncestralPaths_native_Dense5" else "probabilities"
    bad=copy.deepcopy(z);bad[key][0,0,0]=np.nan
    reject("reject_nonfinite_archive_"+m,lambda:V.validate_product(reader.npz("review_synthetic/bad.npz",bad),p,m,reader,argformer_roster_ref=roster))
metrics=dict(calibration_bins=15,boundary_tolerance_bp=1000.,short_tract_cutoff_bp=10000.)
view=V.metric_parent_view_v3(p);V.preflight_metric_contract_v3(view,metrics);ok("complete_metric_parent_pretruth_preflight")
bad=copy.deepcopy(view);bad["samples"][1]["haplotype"]=0
reject("reject_bad_diploid_pair_pretruth",lambda:V.preflight_metric_contract_v3(bad,metrics))
bad=copy.deepcopy(view);bad["cohorts"][0]="Bronze_Age"
reject("reject_mismatched_metric_cohort_pretruth",lambda:V.preflight_metric_contract_v3(bad,metrics))
helper=ast.parse((ROOT/"scripts/train_ancestralpaths_biological_v1.py").read_text())
names={"intersections","boundary_match","tracts","length_summary","full_span_score","json_value"}
env=dict(np=np,math=math)
exec(compile(ast.Module(body=[n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),"original_metrics:synthetic_only","exec"),env)
truth=types.SimpleNamespace(intervals=np.array([[0.,2000000.]]),c4=np.zeros((1,686),dtype=np.int64))
for cols in (np.arange(182),np.arange(686),np.arange(182,686)):
    out=env["full_span_score"](F.products(p,V.METHODS[0])["C4_probabilities"],view,truth,cols,metrics)
    assert out["query_count"]==len(cols) and out["accuracy"]==1 and math.isfinite(out["nll"])
    json.dumps(env["json_value"](out),allow_nan=False)
    ok("inherited_metric_and_serializer_full_stratum_"+str(len(cols)))
query=p["query_slots"];intervals=np.array([[0.,100.],[100.,200.]])
V.validate_h050_bank_geometry_v3(query,np.array([0,1]),np.array([100.,100.],np.float32),(2,952),intervals,p);ok("H050_actual_query_hold_span_readout_contract")
reject("H050_reject_wrong_query_order",lambda:V.validate_h050_bank_geometry_v3(query[::-1],np.array([0,1]),np.array([100.,100.],np.float32),(2,952),intervals,p))
reject("H050_reject_wrong_readout",lambda:V.validate_h050_bank_geometry_v3(query,np.array([1,0]),np.array([100.,100.],np.float32),(2,952),intervals,p))
reject("H050_reject_wrong_native_interval",lambda:V.validate_h050_bank_geometry_v3(query,np.array([0,1]),np.array([100.,100.],np.float32),(2,952),intervals+1,p))
node=next(n for n in ast.parse((ROOT/"research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/producer_v1.py").read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="actual_ref")
actual_env=dict(ROOT=BASE,hashlib=hashlib);exec(compile(ast.Module(body=[node],type_ignores=[]),"original_actual_ref:synthetic_only","exec"),actual_env)
(BASE/"checkpoint_identity_fixture.bin").write_bytes(b"synthetic identity only")
fixed=actual_env["actual_ref"]("checkpoint_identity_fixture.bin")
assert A.exact_checkpoint_identity_v3(types.SimpleNamespace(actual_ref=actual_env["actual_ref"]),fixed)==V.ROOT/fixed["path"]
assert actual_env["actual_ref"](BASE/"checkpoint_identity_fixture.bin")!=fixed
ok("H050_relative_string_authentication_vs_original_absolute_Path_defect")
writer_node=next(n for n in ast.parse((CAND/"h050_observed_bridge_v3.py").read_text()).body if isinstance(n,ast.ClassDef) and n.name=="_CutWriterV3")
writer_env=dict(ROOT=BASE,Path=Path,need=V.need,record=V.record)
exec(compile(ast.Module(body=[writer_node],type_ignores=[]),"cut_writer:synthetic_only","exec"),writer_env)
class Writer:
    def save(self,name,value,kind=None):
        path=BASE/"cell/cuts"/name;path.parent.mkdir(parents=True,exist_ok=True)
        data=b"synthetic cut bytes";path.write_bytes(data)
        return dict(path=str(path.relative_to(BASE)),sha256=hashlib.sha256(data).hexdigest())
bounds=types.SimpleNamespace(extra_arrays={});w=writer_env["_CutWriterV3"](Writer(),bounds,"cell",100,lambda *a:None)
r=w.save("tiny.npz",{},"npz");assert bounds.extra_arrays[r["path"],r["sha256"]]==19;ok("H050_canonical_cut_writer_actual_compressed_readback_bound")
reject("H050_cut_writer_rejects_traversal",lambda:w.save("../bad.npz",{},"npz"))
w=writer_env["_CutWriterV3"](Writer(),bounds,"cell",1,lambda *a:None)
reject("H050_cut_writer_rejects_overcap_after_preserving_failure_bytes",lambda:w.save("overcap.npz",{},"npz"))
stream=io.StringIO()
with redirect_stdout(stream):assert E.main([])==2
assert json.loads(stream.getvalue())["status"]=="INACTIVE_SOURCE_ONLY";ok("CLI_default_inactive_exit2_no_guard_or_payload")
called=[];dispatch=types.FunctionType(E.main.__code__,dict(E.main.__globals__,run_v3=lambda r,s:(called.append((r,s)) or {})))
for flag,stage in (("--validate-ref","validate"),("--score-ref","score")):
    with redirect_stdout(io.StringIO()):assert dispatch([flag,"{}"])==0
    assert called[-1]==({},stage);ok("CLI_actual_mode_dispatch_"+stage)
assert "torch" not in sys.modules;ok("synthetic_review_imported_no_Torch")
result=dict(schema="European_outer8_independent_v3_verification",source_only=True,checks=checks,cases=len(checks),source_metadata_refs_verified=verified,
    data_path_or_payload_refs_deferred_unopened=deferred,original_assets_deferred_unopened=manifest["expected_original_metadata_and_assets"],
    prior_seals=prior,all12_clone_bodies_compiled_without_execution=True,no_actual_data_truth_tree_checkpoint_reads=True,
    no_runtime_resource_process_GPU_probe_or_accounting_invocation=True,no_jobs_installs_or_Git=True,
    fixture_only_native_lineage_records_are_synthetic_not_actual_wait_evidence=True)
(BASE/"verification.v3.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status="PASS",cases=len(checks),manifest_refs_verified=len(verified),deferred=len(deferred))))
