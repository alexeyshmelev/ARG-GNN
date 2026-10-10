"""Independent adversarial CPU/metadata fixtures; no executable permit or real arrays."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import test_author_replay as author

h=author.h
ACTUAL_ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
PARENT="bio_paths_pilot_train_000"

@pytest.mark.parametrize("field,value",[("active",1),("CPU_only",1),("threads",True),("preparation_only",1),("native_fit_or_retrieval_authorized",True),("scientific_fallback",True),("paper_reproduction_claimed",True),("scope_parents",["bio_paths_pilot_train_001"]),("condition","AmericanAdmixture"),("output","data/official_argformer_tsinfer_parent_v1/bio_paths_pilot_train_001")])
def test_typed_admission_rejects_bool_number_substitution_and_method_scope(field,value):
    c=h.policy();a,r,rt=author.fake_admission(c,PARENT)
    a[field]=value
    with pytest.raises(ValueError):h.validate_admission(c,PARENT,a,r,rt)

@pytest.mark.parametrize("change",["TRAIN_truth","role","leaf_id","source_id","mapping_reorder","extra_truth_key"])
def test_exact_original_roster_join_rejects_changed_lineage(change):
    c=h.policy();bridge,bc=h.load_bridge(c)
    source=[];rows=[];mapping=[]
    for i in range(280):
        role="pure_source_reference" if i<96 else ("mixed_reference_only" if i<160 else "query")
        src={"sample_index":i,"sample_id":f"handmade_{i//2}","haplotype":i%2,"observed_cohort":"handmade","age_generations":10.,"role":role,"observed_reference_class":i//24 if i<96 else None,"observed_node_id":i}
        source.append(src);row=copy.deepcopy(src);row["source_observed_node_id"]=row.pop("observed_node_id");row["observed_node_id"]=row["inferred_tree_node_id"]=1000+i;rows.append(row);mapping.append({"sample_index":i,"inferred_tree_node_id":1000+i})
    bridge.validate_roster(bc,rows,source,mapping)
    if change=="TRAIN_truth":rows[160]["observed_reference_class"]=2
    elif change=="role":rows[160]["role"]="pure_source_reference"
    elif change=="leaf_id":rows[160]["inferred_tree_node_id"]=1000
    elif change=="source_id":rows[160]["source_observed_node_id"]=159
    elif change=="mapping_reorder":mapping[0],mapping[1]=mapping[1],mapping[0]
    else:rows[160]["query_ancestry"]=[0,0,1,0]
    with pytest.raises(ValueError):bridge.validate_roster(bc,rows,source,mapping)

def test_cpu_hidden_devices_gate_before_any_import(monkeypatch):
    c=h.policy();monkeypatch.setenv("CUDA_VISIBLE_DEVICES","2")
    rt={"python_executable_sha256":"0"*64}
    seen=[];original=builtins.__import__
    def imports(name,*args,**kwargs):
        seen.append(name);return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",imports)
    with pytest.raises(ValueError,match="hide every"):h.runtime_preflight(c,rt)
    assert not seen

def test_no_runtime_authority_stops_cpu_library_import(monkeypatch):
    seen=[];original=builtins.__import__
    def imports(name,*args,**kwargs):
        if name.split(".")[0] in {"torch","tskit","numpy"}:seen.append(name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",imports)
    class Denied:
        def require(self):raise RuntimeError("handmade denied authority")
    with pytest.raises(RuntimeError):h.CPU_libraries(Denied())
    assert not seen

def test_scheduler_mutex_is_true_RDWR_existing_readonly_bytes(tmp_path,monkeypatch):
    c,parent,r,module,path,state,key=author.storage_fixture(tmp_path,monkeypatch)
    lock=tmp_path/c["storage"]["scheduler_lock"];lock.write_text("handmade mutex marker")
    before=lock.read_bytes();ledger_before=path.read_bytes();seen=[];original=h.os.open
    def opener(p,flags,*args,**kwargs):
        if Path(p)==lock:seen.append(flags)
        return original(p,flags,*args,**kwargs)
    monkeypatch.setattr(h.os,"open",opener)
    storage=h.Storage(c,parent,r,module);storage.check()
    assert len(seen)==2
    assert all(flags & os.O_ACCMODE == os.O_RDWR and flags & os.O_NOFOLLOW for flags in seen)
    assert all(not flags & (os.O_CREAT|os.O_TRUNC|os.O_APPEND|os.O_EXCL) for flags in seen)
    assert lock.read_bytes()==before and path.read_bytes()==ledger_before

@pytest.mark.parametrize("change",["lock_symlink","ledger_symlink","warning_boundary","disk_headroom"])
def test_storage_boundary_and_containment_fail_closed(tmp_path,monkeypatch,change):
    c,parent,r,module,path,state,key=author.storage_fixture(tmp_path,monkeypatch)
    if change.endswith("symlink"):
        target=tmp_path/"handmade_metadata";target.write_text("{}");victim=tmp_path/c["storage"]["scheduler_lock"] if change=="lock_symlink" else path;victim.unlink();victim.symlink_to(target)
    elif change=="warning_boundary":
        base=167336640512+130000000000+30000000000+67108864
        state["jobs"]["other"]={"reserved_bytes":450000000000-base,"output":str(tmp_path/"other")};path.write_text(json.dumps(state));r["ledger_sha256_at_admission"]=h.sha(path)
    else:monkeypatch.setattr(h.os,"statvfs",lambda p:SimpleNamespace(f_bavail=1,f_frsize=1))
    with pytest.raises((ValueError,RuntimeError)):h.Storage(c,parent,r,module)

def test_local_cap_includes_outer_log_not_only_output(tmp_path,monkeypatch):
    c,parent,r,module,path,state,key=author.storage_fixture(tmp_path,monkeypatch)
    output,log=h.selected_scopes(c,parent)
    monkeypatch.setattr(h,"local_bytes",lambda p:1200000000 if p==output else 800000001)
    with pytest.raises(ValueError,match="combined"):h.Storage(c,parent,r,module)

def test_output_symlink_and_write_once_block_source_overwrite(tmp_path,monkeypatch):
    c=copy.deepcopy(h.FIXED_CONFIG);c["output_root"]="owned";c["launch_log_root"]="logs";monkeypatch.setattr(h,"ROOT",tmp_path)
    a=h.Authority(c,PARENT,{"CPU_only":True,"executable_sha256":"0"*64},SimpleNamespace(check=lambda:None));a.output.mkdir(parents=True);a.owns_output=True
    p=a.output/"first.json";a.json_new(p,{"one":1});before=p.read_bytes()
    with pytest.raises(FileExistsError):a.json_new(p,{"overwrite":True})
    protected=tmp_path/"protected.py";protected.write_text("preserved")
    link=a.output/"alias.json";link.symlink_to(protected)
    with pytest.raises(ValueError,match="canonical"):a.json_new(link,{})
    assert p.read_bytes()==before and protected.read_text()=="preserved"

def test_diagnostic_failure_receipt_small_write_once_and_owned(tmp_path,monkeypatch):
    c=copy.deepcopy(h.FIXED_CONFIG);c["output_root"]="owned";c["launch_log_root"]="logs";monkeypatch.setattr(h,"ROOT",tmp_path)
    a=h.Authority(c,PARENT,{"CPU_only":True,"executable_sha256":"0"*64},SimpleNamespace(check=lambda:(_ for _ in ()).throw(ValueError("over cap"))))
    with pytest.raises(RuntimeError):a.failure_evidence({})
    a.output.mkdir(parents=True);a.owns_output=True
    with pytest.raises(ValueError,match="allowance"):a.failure_evidence({"error":"x"*65536})
    a.failure_evidence({"status":"failed_partial_preserved"})
    assert (a.output/"dispatcher_receipt.json").stat().st_size<65536
    with pytest.raises(FileExistsError):a.failure_evidence({})

@pytest.mark.parametrize("change",["mixed_support_gap","overlapping_occurrence","half_open_boundary","missing_reference_VAL","wrong_row_order","cyclic_path","ancestorless_path"])
def test_all_roles_full_occurrences_and_full_path_membership_reject_corruption(tmp_path,monkeypatch,change):
    c,bridge,bc,a,p,np,directory,corpus,vocab,joins=author.corpus_fixture(tmp_path,monkeypatch)
    if change in {"mixed_support_gap","overlapping_occurrence","half_open_boundary"}:
        occ=list(corpus["occurrences"])
        if change=="mixed_support_gap":occ[96]=(96,corpus["paths"][96][0],1.,400.)
        elif change=="overlapping_occurrence":occ.append(occ[0])
        else:occ[160]=(160,corpus["paths"][160][0],0.,340.)
        corpus["occurrences"]=tuple(occ);author.write_primitive(directory/"corpus.pkl",corpus)
    elif change=="missing_reference_VAL":
        folder=directory/"contrastive/val";ids=json.loads((folder/"path_ids.json").read_text());(folder/"path_ids.json").write_text(json.dumps(ids[:-1]))
    elif change=="wrong_row_order":
        path=directory/"mlm/train/raw_train_sequences_and_vocab.pkl";payload=h.primitive_load(path);payload["raw_sequences"][0],payload["raw_sequences"][1]=payload["raw_sequences"][1],payload["raw_sequences"][0];author.write_primitive(path,payload)
    else:
        paths=list(corpus["paths"]);paths[0]=(paths[0][0],777,777) if change=="cyclic_path" else (paths[0][0],);corpus["paths"]=tuple(paths);author.write_primitive(directory/"corpus.pkl",corpus)
    author.reseal(directory,p)
    with pytest.raises(ValueError):h.qualify_corpus(c,bridge,bc,a,p,np)

def test_genuine_tokenizer_truncation_preserves_full_ids_and_leaf_metadata():
    methods,hashes=author.fixture_methods()
    ancestors=tuple(range(10000,10128))
    paths=((1,*ancestors,20000),(2,*ancestors,20001))
    vocab={"special":{"[PAD]":0,"[MASK]":1,"[CLS]":2,"[SEP]":3},"node_id_tokens":{f"node_{n}":i+4 for i,n in enumerate(sorted({n for p in paths for n in p}))}}
    rows=h.token_rows(paths,vocab,methods);first=rows(0);second=rows(1)
    assert first["input_ids"]==second["input_ids"] and first["extant_node_id_raw"]==1 and second["extant_node_id_raw"]==2
    q=h.validate_tokens(paths,vocab,methods)
    assert q["tokenized_rows"]==2 and q["duplicate_tokenized_inputs_retained"]==1 and q["full_paths_truncated_by_author128_policy"]==2
    assert first["input_ids"][0]==2 and first["input_ids"][-1]==3
    assert hashes["_tokenize_raw_sequence_from_vocab"]=="0423ff1c6ed07fc3011901266ab74db9f7b94c626a2827ff5cc0897a3db11c86"

@pytest.mark.parametrize("pure_train",[4096,4095])
def test_pair_cost_guard_required_even_if_source_budget_infeasible(pure_train):
    c=h.policy();bridge,bc=h.load_bridge(c)
    pure_val=13000;queries=2400000;total=pure_train+pure_val+queries+64;train_total=int(.9*total)
    query_train=train_total-pure_train-57
    roles={"train":{"pure_source_reference":pure_train,"mixed_reference_only":57,"query":query_train},"val":{"pure_source_reference":pure_val,"mixed_reference_only":7,"query":queries-query_train}}
    assert sum(roles["train"].values())==train_total and sum(roles["val"].values())==total-train_total
    assert queries<=20000*120 # maximum distinct query IDs on the full scored centers
    assert (pure_train+pure_val)*queries>bc["retrieval"]["max_exhaustive_pairs_per_parent"]
    with pytest.raises(ValueError,match="cost"):h.budget(bridge,bc,roles,queries)


def mock_execute_fixture(tmp_path,monkeypatch,format_error=False,feasible=True):
    c=copy.deepcopy(h.FIXED_CONFIG);c["output_root"]="owned";c["launch_log_root"]="logs"
    monkeypatch.setattr(h,"ROOT",tmp_path)
    script=tmp_path/"handmade_dispatcher_marker.py";script.write_text("handmade source marker")
    config=tmp_path/"handmade_policy_marker.json";config.write_text("{}")
    (tmp_path/"admission.json").write_text("{}") # inert file marker, active dictionary is memory only
    monkeypatch.setattr(h,"SCRIPT",script);monkeypatch.setattr(h,"CONFIG",config)
    a,review,runtime=author.fake_admission(c,PARENT)
    refs={name:{"path":name+".json","sha256":"0"*64} for name in ["independent_review","runtime","storage","experiment_registration"]}
    a.update(refs)
    registration={"kind":"official_argformer_tsinfer_parent_v1_root_experiment_registration","authorizer":"root_agent","status":"registered_CPU_preparation_and_feasibility_only","condition":c["condition"],"scope_parents":[PARENT],"implementation":h.ref(script),"config":h.ref(config),"combined_observed_manifest":c["combined_observed_manifest"],"query_truth_authorized":False,"hypothesis_id":"handmade_only_never_registered"}
    native={"inferred_tree":{"path":"handmade_observed.trees","sha256":"0"*64},"observed_sample_manifest":{"path":"handmade_roster.json","sha256":"0"*64},"original_observed_sample_manifest":{"path":"handmade_source_roster.json","sha256":"0"*64},"completion_provenance":{"path":"handmade_inference.json","sha256":"0"*64},"coordinate_contract":{"fixed_grid":{"path":"handmade_grid.npz","sha256":"0"*64}},"sites_in_scored_support":700}
    row={"parent_id":PARENT,"frozen_helper_result":native};rows=[{"inferred_tree_node_id":i} for i in range(280)]
    metadata={"parents":{PARENT:{"namespace":"handmade:namespace"}}}
    lookup={"admission.json":a,"independent_review.json":review,"runtime.json":runtime,"storage.json":{},"experiment_registration.json":registration,c["combined_observed_manifest"]["path"]:{"replicates":[row]},"handmade_roster.json":rows,"handmade_source_roster.json":rows}
    monkeypatch.setattr(h,"read_json",lambda reference,expected=None:copy.deepcopy(lookup[reference["path"]]))
    events=[]
    monkeypatch.setattr(h,"runtime_preflight",lambda *x:(events.append("runtime") or {"CPU_only":True,"executable_sha256":"0"*64}))
    class FakeStorage:
        def __init__(self,*x):events.append("storage")
        def check(self):events.append("storage_check");return {"handmade":True}
    monkeypatch.setattr(h,"Storage",FakeStorage);monkeypatch.setattr(h,"load_pool",lambda *x:None)
    def contain(authority):
        events.append("contain");authority.require();authority.output.mkdir(parents=True);authority.owns_output=True
    monkeypatch.setattr(h,"contain",contain)
    class FileFormatTooNewError(ValueError):pass
    def load(path):
        events.append("tskit.load:"+str(path))
        if format_error:raise FileFormatTooNewError("handmade unsupported writer1.0.3 reader0.6.4")
        return SimpleNamespace(num_samples=280,num_sites=700,num_nodes=400,samples=lambda:SimpleNamespace(tolist=lambda:list(range(280))))
    tskit=SimpleNamespace(load=load,__version__="0.6.4")
    monkeypatch.setattr(h,"CPU_libraries",lambda authority:(events.append("CPU_libraries") or (None,tskit)))
    monkeypatch.setattr(h,"bound",lambda reference,expected=None,json_only=False:(events.append("bound:"+reference["path"]) or tmp_path/reference["path"]))
    def prepare(c,bridge,bc,authority):
        events.append("compile_core")
        def core(*x):
            events.append("core");assert bc["execution_enabled"] is False
            (authority.output/"prepared").mkdir();authority.json_new(authority.output/"prepared/prepared_manifest.json",{"handmade":True});return {"handmade":True}
        return core
    monkeypatch.setattr(h,"preparation_callable",prepare)
    monkeypatch.setattr(h,"qualify_corpus",lambda *x:{"status":"qualified_full_native_preparation" if feasible else "prepared_native_source_budget_infeasible","budget":{"native_MLM512_contrastive4096_feasible":feasible}})
    monkeypatch.setattr(h,"verify_source",lambda *x:None)
    bridge=SimpleNamespace(metadata_report=lambda *x:metadata,native_metadata=lambda *x:{"namespace":"handmade:namespace"},preparation_entry=lambda record:{"handmade":True},phase_for=lambda *x:"canary",verify_source=lambda *x:None)
    bc={"execution_enabled":False,"manifest_contract":{"path":"handmade/combined.json"}}
    args=SimpleNamespace(parent=PARENT,expected_source_sha256=h.sha(script),expected_config_sha256=h.sha(config),admission="admission.json",expected_admission_sha256="0"*64)
    return c,bridge,bc,args,a,events

@pytest.mark.parametrize("feasible",[True,False])
def test_active_execute_reachable_via_handmade_stages_and_preserves_negative(tmp_path,monkeypatch,feasible):
    c,bridge,bc,args,a,events=mock_execute_fixture(tmp_path,monkeypatch,feasible=feasible)
    result=h.execute(c,bridge,bc,args)
    assert result["exit_code"]==(0 if feasible else 2)
    assert events.index("runtime")<events.index("contain")<events.index("CPU_libraries")
    load_at=next(i for i,v in enumerate(events) if v.startswith("tskit.load:"))
    assert load_at<events.index("compile_core")<events.index("core") and bc["execution_enabled"] is False
    assert not result["query_truth_used"] and not result["model_fit_or_retrieval_executed"]
    assert json.loads((tmp_path/c["output_root"]/PARENT/"dispatcher_receipt.json").read_text())["exit_code"]==result["exit_code"]

def test_original_format_failure_stops_before_core_and_preserves_exact_error(tmp_path,monkeypatch):
    c,bridge,bc,args,a,events=mock_execute_fixture(tmp_path,monkeypatch,format_error=True)
    result=h.execute(c,bridge,bc,args)
    assert result["exit_code"]==1 and result["status"]=="failed_partial_preserved" and result["error_type"]=="FileFormatTooNewError"
    assert result["original_tree_format_compatibility_attempted"] and not result["actual_native_preparation_completed"]
    assert "core" not in events and "compile_core" not in events
    assert not (tmp_path/c["output_root"]/PARENT/"input_tree_compatibility.json").exists()
    assert (tmp_path/c["output_root"]/PARENT/"dispatcher_receipt.json").is_file()

@pytest.mark.parametrize("change",["inactive","wrong_source","bad_review","wrong_parent"])
def test_execute_bad_authority_stops_before_runtime_storage_and_containment(tmp_path,monkeypatch,change):
    c,bridge,bc,args,a,events=mock_execute_fixture(tmp_path,monkeypatch)
    if change=="inactive":a["active"]=False
    elif change=="wrong_source":args.expected_source_sha256="1"*64
    elif change=="bad_review":a["independent_review"]={"path":"storage.json","sha256":"0"*64}
    else:args.parent=c["parent_ids"][1]
    with pytest.raises(ValueError):h.execute(c,bridge,bc,args)
    assert not events and not (tmp_path/c["output_root"]).exists()

@pytest.mark.parametrize("pure_train,queries,over_limit",[(4095,2000000,False),(4095,2000001,True),(4096,2000000,False),(4096,2000001,True)])
def test_cost_threshold_equality_and_one_over_are_independent_of_feasibility(pure_train,queries,over_limit):
    c=h.policy();bridge,bc=h.load_bridge(c)
    pure_val=20000-pure_train
    total=20000+queries+64;train_total=int(.9*total)
    query_train=train_total-pure_train-57
    roles={"train":{"pure_source_reference":pure_train,"mixed_reference_only":57,"query":query_train},"val":{"pure_source_reference":pure_val,"mixed_reference_only":7,"query":queries-query_train}}
    assert sum(roles["train"].values())==train_total and queries<=20000*120
    if over_limit:
        with pytest.raises(ValueError,match="full-corpus cost"):h.budget(bridge,bc,roles,queries)
    else:
        got=h.budget(bridge,bc,roles,queries)
        assert got["exhaustive_query_corpus_pairs"]==40000000000
        assert got["native_MLM512_contrastive4096_feasible"]==(pure_train==4096)
        assert got["contrastive_optimizer_updates"]==(1 if pure_train==4096 else 0)

def test_over_cost_execute_preserves_completed_partial_preparation_and_cost_failure(tmp_path,monkeypatch):
    actual_c=h.policy();actual_bridge,actual_bc=h.load_bridge(actual_c)
    c,bridge,bc,args,a,events=mock_execute_fixture(tmp_path,monkeypatch,feasible=False)
    roles={"train":{"pure_source_reference":4095,"mixed_reference_only":57,"query":2171291},"val":{"pure_source_reference":13000,"mixed_reference_only":7,"query":228709}}
    def qualify(*args):
        return h.budget(actual_bridge,actual_bc,roles,2400000)
    monkeypatch.setattr(h,"qualify_corpus",qualify)
    result=h.execute(c,bridge,bc,args)
    assert result["exit_code"]==1 and result["status"]=="failed_partial_preserved"
    assert result["actual_native_preparation_completed"] and result["error_type"]=="ValueError"
    assert "full-corpus cost" in result["error"] and "core" in events
    output=tmp_path/c["output_root"]/PARENT
    assert (output/"prepared/prepared_manifest.json").is_file()
    assert not (output/"corpus_qualification.json").exists()
    receipt=json.loads((output/"dispatcher_receipt.json").read_text())
    assert receipt["exit_code"]==1 and "full-corpus cost" in receipt["error"]
