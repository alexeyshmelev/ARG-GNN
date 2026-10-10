"""Source and handmade CPU fixture checks; no live permits, ledger or real data."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
import pytest
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
spec=importlib.util.spec_from_file_location("parent_cpu_dispatcher",ROOT/"scripts/prepare_official_argformer_tsinfer_parent_v1.py")
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

def fixture_methods():
    p=ROOT/".runtime/author_sources/argformer_f7545be7bca0/src/arg_data.py"
    cls=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name=="ARGDataset")
    nodes=[copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {"_tokenize_raw_sequence_from_vocab","__getitem__"}]
    mod=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),*nodes],type_ignores=[]);ast.fix_missing_locations(mod);ns={}
    exec(compile(mod,str(p),"exec"),ns)
    return ns,{n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in nodes}

def fake_admission(c,parent):
    a=h.admission_bindings(c,parent)
    a["actual_input_tree_roster_grid_validation_authorized"]=True
    review={"kind":"official_argformer_tsinfer_parent_v1_independent_source_review","review_role":"independent_reviewer","verdict":"PASS","implementation":a["implementation"],"config":a["config"],"bridge":c["bridge"]}
    runtime={"kind":"official_argformer_tsinfer_parent_v1_root_CPU_runtime","authorizer":"root_agent",**copy.deepcopy(c["runtime"]),"python_executable_sha256":"0"*64}
    return a,review,runtime

def test_default_source_mode_cannot_import_native_or_read_actual_arrays(monkeypatch):
    original_import=builtins.__import__;original_open=Path.open
    def imports(name,*args,**kwargs):
        if name.split(".")[0] in {"numpy","torch","tskit","tsinfer","tsdate","composer","transformers","omegaconf"}:
            raise AssertionError("native import during source-only")
        return original_import(name,*args,**kwargs)
    def opens(p,*args,**kwargs):
        if p.suffix in {".trees",".npz",".vcf",".pkl",".pt",".npy"}:
            raise AssertionError("actual array read during source-only")
        return original_open(p,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",imports);monkeypatch.setattr(Path,"open",opens)
    c=h.policy();h.verify_source(c);bridge,bc=h.load_bridge(c)
    report=h.source_report(c,bridge,bc)
    assert not report["CPU_runtime_inspected"] and not report["live_ledger_read"] and not report["actual_arrays_opened"]
    assert len(report["native20_function_ASTs"])==20
    assert report["native_preparation_tail_AST_sha256"]==json.loads((ROOT/"research/reviews/official_argformer_tsinfer_biological_v2_author/source_only_final.stdout.json").read_text())["preparation_tail_AST_sha256"]

def test_execute_without_optin_stops_before_runtime_ledger_or_arrays(monkeypatch):
    c=h.policy();bridge,bc=h.load_bridge(c)
    monkeypatch.setattr(h,"runtime_preflight",lambda *a:(_ for _ in ()).throw(AssertionError("runtime inspected")))
    monkeypatch.setattr(h,"load_pool",lambda *a:(_ for _ in ()).throw(AssertionError("ledger source entered")))
    args=SimpleNamespace(parent=c["default_parent"],expected_source_sha256=None,expected_config_sha256=None,admission=None,expected_admission_sha256=None)
    with pytest.raises(ValueError,match="opt-in"):h.execute(c,bridge,bc,args)

def test_positive_one_parent_admission_in_memory_fixture_only():
    c=h.policy();a,r,rt=fake_admission(c,c["default_parent"])
    got=h.validate_admission(c,c["default_parent"],a,r,rt)
    assert got["scope_parents"]==["bio_paths_pilot_train_000"] and got["preparation_only"]

@pytest.mark.parametrize("change",["two_parents","query_truth","threads","source","review_role","review_verdict","runtime","tree_authority","inactive"])
def test_admission_schema_blocks_scope_runtime_or_review_drift(change):
    c=h.policy();a,r,rt=fake_admission(c,c["default_parent"])
    if change=="two_parents":a["scope_parents"]=c["parent_ids"][:2]
    elif change=="query_truth":a["query_truth_authorized"]=True
    elif change=="threads":a["threads"]=2
    elif change=="source":a["implementation"]["sha256"]="1"*64
    elif change=="review_role":r["review_role"]="author"
    elif change=="review_verdict":r["verdict"]="BLOCKED"
    elif change=="runtime":rt["distributions"]["tskit"]="1.0.3"
    elif change=="tree_authority":a["actual_input_tree_roster_grid_validation_authorized"]=False
    else:a["active"]=False
    with pytest.raises(ValueError):h.validate_admission(c,c["default_parent"],a,r,rt)

def test_private_preparation_namespace_has_no_unwrapped_write_or_import_route():
    tree=ast.parse(Path(h.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="preparation_callable")
    assert "load_frozen" not in ast.get_source_segment(Path(h.__file__).read_text(),fn)
    assignments=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="ns" for t in n.targets)]
    values={k.value:ast.unparse(v) for k,v in zip(assignments[0].value.keys,assignments[0].value.values)}
    assert values["json_new"]=="authority.json_new"
    class Denied:
        def require(self):raise RuntimeError("runtime authority missing")
    with pytest.raises(RuntimeError,match="runtime authority"):h.preparation_callable({},None,None,Denied())
    with pytest.raises(RuntimeError,match="runtime authority"):h.author_tokenizer({},Denied())

def test_unready_write_cannot_touch_existing_source_or_import(tmp_path,monkeypatch):
    c=copy.deepcopy(h.FIXED_CONFIG);c["output_root"]="owned";c["launch_log_root"]="logs"
    monkeypatch.setattr(h,"ROOT",tmp_path)
    authority=h.Authority(c,c["default_parent"],{"CPU_only":False},SimpleNamespace(check=lambda:None))
    protected=tmp_path/"protected.py";protected.write_text("original source")
    with pytest.raises(RuntimeError,match="runtime"):authority.json_new(protected,{"bad":1})
    assert protected.read_text()=="original source"
    authority.ready=True
    with pytest.raises(RuntimeError,match="own output"):authority.json_new(authority.output/"x.json",{})
    authority.output.mkdir(parents=True);authority.owns_output=True
    with pytest.raises(ValueError,match="one admitted output"):authority.json_new(protected,{})
    assert protected.read_text()=="original source"

def role_counts():
    return {"train":{"pure_source_reference":5000,"mixed_reference_only":1500,"query":3500},
            "val":{"pure_source_reference":500,"mixed_reference_only":150,"query":350}}

def test_budget_crosschecks_source_counts_and_reports_exact_full4096_updates():
    c=h.policy();bridge,bc=h.load_bridge(c);roles=role_counts()
    result=h.budget(bridge,bc,roles,3000,{"mlm":{"train":10000,"val":1000},"contrastive":{"train":5000,"val":500}},11000)
    assert result["MLM_optimizer_updates"]==95 and result["contrastive_optimizer_updates"]==1
    assert result["pure_reference_corpus_paths_all_train_plus_val"]==5500
    assert result["native_MLM512_contrastive4096_feasible"] and not result["query_truth_trains_any_model"]

@pytest.mark.parametrize("change",["negative","bool","query_exceeds","reference_record","MLM_record","unique_record"])
def test_budget_rejects_inconsistent_real_count_claims(change):
    c=h.policy();bridge,bc=h.load_bridge(c);roles=role_counts();q=3000;decl={"mlm":{"train":10000,"val":1000},"contrastive":{"train":5000,"val":500}};total=11000
    if change=="negative":roles["train"]["query"]=-1
    elif change=="bool":roles["train"]["query"]=True
    elif change=="query_exceeds":q=4000
    elif change=="reference_record":decl["contrastive"]["train"]=12000
    elif change=="MLM_record":decl["mlm"]["val"]=2
    else:total=1
    with pytest.raises(ValueError):h.budget(bridge,bc,roles,q,decl,total)

def test_infeasible_source4096_is_preserved_as_negative_no_fallback():
    c=h.policy();bridge,bc=h.load_bridge(c);roles=role_counts();roles["train"]["pure_source_reference"]=4095
    r=h.budget(bridge,bc,roles,3000)
    assert not r["native_MLM512_contrastive4096_feasible"]
    assert r["contrastive_optimizer_updates"]==0 and r["contrastive_full_logical_and_microbatch"]==4096

def write_primitive(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as stream:pickle.dump(value,stream,protocol=4)

def corpus_fixture(tmp_path,monkeypatch):
    import numpy as np
    c=h.policy();bridge,bc=h.load_bridge(c);methods,method_hashes=fixture_methods()
    monkeypatch.setattr(h,"ROOT",tmp_path)
    c=copy.deepcopy(c);c["output_root"]="owned";c["launch_log_root"]="logs"
    authority=h.Authority(c,c["default_parent"],{"CPU_only":True,"executable_sha256":"0"*64},SimpleNamespace(check=lambda:None))
    authority.output.mkdir(parents=True);authority.owns_output=True
    directory=authority.output/"prepared";directory.mkdir()
    rows=[];paths=[]
    for i in range(280):
        role="pure_source_reference" if i<96 else ("mixed_reference_only" if i<160 else "query")
        node=1000+2*i;paths.append((node,777,999))
        rows.append({"inferred_tree_node_id":node,"role":role,"observed_reference_class":i//24 if i<96 else None})
    paths=tuple(paths);namespace="handmade_only:no_real_data"
    corpus={"paths":paths,"occurrences":tuple((i,p[0],0.,400.) for i,p in enumerate(paths)),"namespace":namespace}
    write_primitive(directory/"corpus.pkl",corpus)
    vocab={"special":{"[PAD]":0,"[MASK]":1,"[CLS]":2,"[SEP]":3},"node_id_tokens":{f"node_{n}":i+4 for i,n in enumerate(sorted({n for p in paths for n in p}))}}
    write_primitive(directory/"embedding/all/raw_all_sequences_and_vocab.pkl",{"raw_sequences":[list(p) for p in paths],"vocab":vocab})
    ids=list(range(280));random.Random(42).shuffle(ids);cut=int(.9*len(ids));splits={"train":ids[:cut],"val":ids[cut:]}
    counts={"mlm":{},"contrastive":{}}
    labels={r["inferred_tree_node_id"]:r["observed_reference_class"] for r in rows if r["role"]=="pure_source_reference"}
    for split,all_ids in splits.items():
        for task in ["mlm","contrastive"]:
            selected=all_ids if task=="mlm" else [i for i in all_ids if paths[i][0] in labels]
            folder=directory/task/split;folder.mkdir(parents=True)
            (folder/"path_ids.json").write_text(json.dumps(selected))
            write_primitive(folder/("raw_"+split+"_sequences_and_vocab.pkl"),{"raw_sequences":[list(paths[i]) for i in selected],"vocab":vocab})
            if task=="contrastive":write_primitive(folder/"labels.pkl",[labels[paths[i][0]] for i in selected])
            counts[task][split]=len(selected)
    (directory/"observed_roster.json").write_text(json.dumps(rows))
    q=np.array([r["inferred_tree_node_id"] for r in rows if r["role"]=="query"])
    refs=np.array(list(labels));grid=np.array([0,1,2,3])
    joins={"grid_indexes":grid,"original_positions_bp":50.+100*grid,"local_positions_bp":40.+100*grid,
       "query_nodes":q,"reference_nodes":refs,"reference_classes":np.array(list(labels.values())),
       "query_path_ids":np.tile(np.arange(160,280),(4,1)),"reference_path_ids":np.tile(np.arange(96),(4,1))}
    np.savez_compressed(directory/"coordinate_path_joins.npz",**joins)
    prepared={"policy_sha256":h.sha(bridge.CONFIG),"true_query_ancestry_read":False,"unique_full_paths":280,"path_counts":counts,"namespace":namespace,
      "coordinate_contract":{"coverage_original_bp0":[10,410],"local_to_original_offset_bp":10,"retained_fixed100bp_indexes":[0,1,2,3]},
      "files_sha256":{str(p.relative_to(directory)):h.sha(p) for p in directory.rglob("*") if p.is_file()}}
    monkeypatch.setattr(h,"author_tokenizer",lambda *args:(methods,method_hashes))
    return c,bridge,bc,authority,prepared,np,directory,corpus,vocab,joins

def reseal(directory,prepared):
    prepared["files_sha256"]={str(p.relative_to(directory)):h.sha(p) for p in directory.rglob("*") if p.is_file()}

def test_280_leaf_handmade_corpus_retains_duplicate_tokenizations(tmp_path,monkeypatch):
    c,bridge,bc,a,p,np,directory,*_=corpus_fixture(tmp_path,monkeypatch)
    q=h.qualify_corpus(c,bridge,bc,a,p,np)
    assert q["all_unique_paths_before_truncate"]==280 and q["full_pure_reference_corpus_paths"]==96
    assert q["tokenization"]["tokenized_rows"]==280 and q["tokenization"]["distinct_tokenized_inputs"]==1
    assert q["tokenization"]["duplicate_tokenized_inputs_retained"]==279
    assert q["full_readout_occurrence_partitions_validated"] and q["native_original_node_IDs_used"]
    assert q["status"]=="prepared_native_source_budget_infeasible"
    assert not q["query_truth_used"] and not q["model_fit_or_retrieval_executed"]

@pytest.mark.parametrize("change",["full_path_duplicate","vocab","post_truncate_dedup","query_into_contrastive","contrastive_label","support_gap","wrong_readout","namespace"])
def test_handmade_corpus_corruption_cannot_shrink_or_relabel_baseline(tmp_path,monkeypatch,change):
    c,bridge,bc,a,p,np,directory,corpus,vocab,joins=corpus_fixture(tmp_path,monkeypatch)
    if change=="full_path_duplicate":
        corpus["paths"]=(corpus["paths"][0],corpus["paths"][0],*corpus["paths"][2:]);write_primitive(directory/"corpus.pkl",corpus)
    elif change=="vocab":
        vocab["node_id_tokens"].pop("node_777");payload=h.primitive_load(directory/"embedding/all/raw_all_sequences_and_vocab.pkl");payload["vocab"]=vocab;write_primitive(directory/"embedding/all/raw_all_sequences_and_vocab.pkl",payload)
    elif change=="post_truncate_dedup":
        payload=h.primitive_load(directory/"embedding/all/raw_all_sequences_and_vocab.pkl");payload["raw_sequences"]=payload["raw_sequences"][:1];write_primitive(directory/"embedding/all/raw_all_sequences_and_vocab.pkl",payload)
    elif change=="query_into_contrastive":
        path=directory/"contrastive/train/path_ids.json";ids=json.loads(path.read_text());ids[0]=160;path.write_text(json.dumps(ids))
    elif change=="contrastive_label":
        path=directory/"contrastive/train/labels.pkl";labels=h.primitive_load(path);labels[0]=(labels[0]+1)%4;write_primitive(path,labels)
    elif change=="support_gap":
        occurrences=list(corpus["occurrences"]);pid,leaf,left,right=occurrences[0];occurrences[0]=(pid,leaf,1.,right);corpus["occurrences"]=tuple(occurrences);write_primitive(directory/"corpus.pkl",corpus)
    elif change=="wrong_readout":
        joins["query_path_ids"][0,0]=0;np.savez_compressed(directory/"coordinate_path_joins.npz",**joins)
    else:
        corpus["namespace"]="other_parent";write_primitive(directory/"corpus.pkl",corpus)
    reseal(directory,p)
    with pytest.raises(ValueError):h.qualify_corpus(c,bridge,bc,a,p,np)

def test_primitive_corpus_loader_rejects_global_code(tmp_path):
    class Global:
        def __reduce__(self):return (print,("must never execute",))
    path=tmp_path/"fake.pkl";write_primitive(path,Global())
    with pytest.raises(ValueError,match="primitive"):h.primitive_load(path)

def test_first_real_load_is_original_tree_format_canary_and_failure_preserved():
    source=Path(h.__file__).read_text();tree=ast.parse(source)
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="execute")
    calls=[(n.lineno,ast.unparse(n.func)) for n in ast.walk(execute) if isinstance(n,ast.Call)]
    load_line=min(line for line,name in calls if name=="tskit.load")
    core_line=min(line for line,name in calls if name=="core")
    assert load_line<core_line
    assert "original_tree_format_compatibility_attempted" in source
    assert "failed_partial_preserved" in source and "conversion_upgrade_fallback" in source

def storage_fixture(tmp_path,monkeypatch):
    c=copy.deepcopy(h.FIXED_CONFIG)
    module=h.load_pool(c)
    monkeypatch.setattr(h,"ROOT",tmp_path)
    script=tmp_path/"script.py";script.write_text("handmade source marker")
    config=tmp_path/"config.json";config.write_text("{}")
    monkeypatch.setattr(h,"SCRIPT",script);monkeypatch.setattr(h,"CONFIG",config)
    c["output_root"]="owned";c["launch_log_root"]="logs"
    snapshot=tmp_path/"snapshot.json"
    snapshot.write_text(json.dumps({"schema":"biological_candidate_storage_snapshot_v1","status":"complete","returncode":0,
      "project_root":str(tmp_path),"command":["du","-s","-B1",str(tmp_path)],"used_bytes":167336640512,
      "completed_utc":"2026-10-10T17:31:06.897981+00:00"}))
    c["storage"]["snapshot"]={"path":"snapshot.json","sha256":h.sha(snapshot)}
    parent=c["default_parent"];out,log=h.selected_scopes(c,parent)
    p=c["storage"];key=p["reservation_key_prefix"]+"/"+parent
    ledger=tmp_path/p["ledger_path"];ledger.parent.mkdir(parents=True)
    lock=tmp_path/p["scheduler_lock"];lock.touch()
    state={"schema":"gpu_pool_storage_ledger_v1","root":str(tmp_path),"snapshot_sha256":p["snapshot"]["sha256"],
      "baseline_bytes":167336640512,"external_reserve_bytes":130000000000,"limit_bytes":450000000000,
      "controllers":{"other":{"reserved_bytes":67108864}},
      "jobs":{key:{"reserved_bytes":2000000000,"output":str(out)},
              "existing_observed":{"reserved_bytes":20000000000,"output":str(tmp_path/"observed")},
              "retained_runtime":{"reserved_bytes":8000000000,"output":str(tmp_path/"runtime")}}}
    ledger.write_text(json.dumps(state))
    receipt={"kind":"official_argformer_tsinfer_parent_v1_root_storage","authorizer":"root_agent","implementation":h.ref(script),
       "config":h.ref(config),"scope_parents":[parent],"reservation_key":key,"reserved_bytes":p["reserved_bytes"],
       "local_write_cap_bytes":p["local_write_cap_bytes"],"output":str(out),"write_scopes":[str(out),str(log)],
       "snapshot":p["snapshot"],"external_reserve_bytes":p["external_reserve_bytes"],"warning_bytes":p["warning_bytes"],
       "hard_limit_bytes":p["hard_limit_bytes"],"stage_boundary_cap_not_OS_quota":True,"ledger_sha256_at_admission":h.sha(ledger)}
    monkeypatch.setattr(h.os,"statvfs",lambda path:SimpleNamespace(f_bavail=10**15,f_frsize=1))
    return c,parent,receipt,module,ledger,state,key

def test_readonly_storage_counts_every_slot_controller_and_external_reserve(tmp_path,monkeypatch):
    c,parent,r,module,path,state,key=storage_fixture(tmp_path,monkeypatch)
    before=path.read_bytes();context=h.Storage(c,parent,r,module)
    expected=167336640512+130000000000+30000000000+67108864
    assert context.last["projected_bytes"]==expected
    assert path.read_bytes()==before
    state["jobs"]["new_other"]={"reserved_bytes":1234,"output":str(tmp_path/"extra")}
    path.write_text(json.dumps(state));before=path.read_bytes()
    assert context.check()["projected_bytes"]==expected+1234
    assert path.read_bytes()==before

@pytest.mark.parametrize("change",["missing_slot","borrowed_output","stale","other_overcommit","local_cap"])
def test_fake_storage_blocks_borrow_stale_overcommit_and_shared_scope_cap(tmp_path,monkeypatch,change):
    c,parent,r,module,path,state,key=storage_fixture(tmp_path,monkeypatch)
    if change=="missing_slot":state["jobs"].pop(key)
    elif change=="borrowed_output":state["jobs"][key]["output"]=str(tmp_path/"observed")
    elif change=="stale":r["ledger_sha256_at_admission"]="0"*64
    elif change=="other_overcommit":state["jobs"]["large_other"]={"reserved_bytes":200000000000,"output":str(tmp_path/"extra")}
    else:monkeypatch.setattr(h,"local_bytes",lambda p:2000000001 if p==h.selected_scopes(c,parent)[0] else 0)
    if change not in {"stale","local_cap"}:
        path.write_text(json.dumps(state));r["ledger_sha256_at_admission"]=h.sha(path)
    before=path.read_bytes()
    with pytest.raises((ValueError,RuntimeError)):h.Storage(c,parent,r,module)
    assert path.read_bytes()==before

def test_admitted_private_stage_reaches_core_without_changing_inactive_bridge(tmp_path,monkeypatch):
    c=h.policy();bridge,bc=h.load_bridge(c);before=copy.deepcopy(bc)
    monkeypatch.setattr(h,"ROOT",tmp_path)
    c=copy.deepcopy(c);c["output_root"]="owned";c["launch_log_root"]="logs"
    checks=[];a=h.Authority(c,c["default_parent"],{"CPU_only":True,"executable_sha256":"0"*64},SimpleNamespace(check=lambda:checks.append("storage")))
    a.output.mkdir(parents=True);a.owns_output=True
    # Handmade core stand-in checks dispatcher reachability; separate source proof
    # checks the actual original preparation tail without executing real arrays.
    body=ast.parse("return {'parent':parent_id,'output':str(output),'inactive_bridge':config['execution_enabled']}").body
    monkeypatch.setattr(h,"preparation_definition",lambda *args:body)
    core=h.preparation_callable(c,bridge,bc,a)
    result=core({},c["default_parent"],a.output/"prepared",bc["manifest_contract"]["sha256"],bc)
    assert result["parent"]==c["default_parent"] and result["inactive_bridge"] is False
    assert checks==["storage"] and bc==before
    with pytest.raises(ValueError,match="one admitted parent"):core({},c["parent_ids"][1],a.output/"prepared","0"*64,bc)

@pytest.mark.parametrize("query_needed,over_limit",[(10000001,True),(9765625,False)])
def test_full_corpus_pair_guard_applies_even_when_source4096_infeasible(query_needed,over_limit):
    c=h.policy();bridge,bc=h.load_bridge(c)
    roles={"train":{"pure_source_reference":4095,"mixed_reference_only":0,"query":query_needed},
           "val":{"pure_source_reference":1,"mixed_reference_only":0,"query":0}}
    if over_limit:
        assert 4096*query_needed==40960004096
        with pytest.raises(ValueError,match="full-corpus cost"):
            h.budget(bridge,bc,roles,query_needed)
    else:
        result=h.budget(bridge,bc,roles,query_needed)
        assert result["exhaustive_query_corpus_pairs"]==40000000000
        assert not result["native_MLM512_contrastive4096_feasible"]
        assert result["contrastive_optimizer_updates"]==0
        assert result["contrastive_full_logical_and_microbatch"]==4096
