"""Source and handmade ARGformer bridge checks. No real arrays/native model."""
import ast
import builtins
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
spec=importlib.util.spec_from_file_location("tsinfer_argformer_v2",ROOT/"src/arggnn/official_argformer_tsinfer_biological_v2.py")
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def fake_reference(path,h="0"*64):
    return {"path":str(path),"sha256":h}

def fake_rosters():
    source=[];derived=[];mapping=[]
    for i in range(280):
        role="pure_source_reference" if i<96 else ("mixed_reference_only" if i<160 else "query")
        label=i//24 if i<96 else None
        original={"sample_index":i,"sample_id":"subject_"+str(i//2),"haplotype":i%2,
                  "observed_cohort":"fake_observed","age_generations":0 if i<224 else 200,
                  "role":role,"observed_reference_class":label,"observed_node_id":3000+i}
        source.append(original)
        node=4000+i*2
        row={k:v for k,v in original.items() if k!="observed_node_id"}
        row.update(source_observed_node_id=original["observed_node_id"],observed_node_id=node,inferred_tree_node_id=node)
        derived.append(row);mapping.append({"sample_index":i,"inferred_tree_node_id":node})
    return source,derived,mapping

def fake_metadata():
    c=copy.deepcopy(b.FIXED_CONFIG);rows=[];rs={};ss={}
    for p in c["parent_ids"]:
        phase=b.phase_for(c,p);work=Path(c["manifest_contract"]["path"]).parent/phase/p
        source,roster,mapping=fake_rosters();rs[p]=roster;ss[p]=source
        native={"parent_id":p,"condition":c["native_helper_condition"],"status":"qualified_observed_cpu_inference",
          "inferred_tree":{**fake_reference(work/"observed.trees"),"condition":c["native_helper_condition"]},
          "observed_sample_manifest":{**fake_reference(work/"observed_sample_manifest.json"),"rows":280},
          "completion_provenance":fake_reference(work/"inference.json"),
          "original_observed_sample_manifest":fake_reference(Path("data/biological_paths_execution_pilot_v1")/p/"observed/sample_manifest.json"),
          "input_observed_sha256":{"sample_manifest.json":"0"*64},
          "sample_index_to_inferred_tree_node_id":mapping,
          "truth_or_oracle_read":False,"no_input_variant_filtering":True,
          "native_node_ids_and_times_preserved_during_sanitation":True,
          "coordinate_contract":{"original_parent_interval_bp0":[0,2000000],"normalization_length_bp":2000000,
             "coverage_original_bp0":[10,361],"local_to_original_offset_bp":10,"retained_fixed100bp_indexes":[0,1,2,3],
             "fixed_grid":{**fake_reference(Path("observed_grid.npz")),"total":20000,"center_formula_bp0":"50 + 100*grid_index"}},
          "connected_identity":{"internal_reuse_occurrences":2,"uncovered_interval_inside_support_admitted":False}}
        adequacy={"expected_parent":p,"phase":phase,"native_parent_directory":str(work),
          "native_verification":fake_reference(work/"native_verification.json"),"stage_journal":fake_reference(work/"stages.json")}
        for key in ["source_input_hashes_exact","original_genotypes_and_sample_ages_exact","strict_native_times_and_mutation_branches","entire_common_support_connected","native_ids_preserved_by_sanitation"]:
            adequacy[key]=True
        rows.append({"parent_id":p,"condition":c["observation_condition"],"source_helper_condition":c["native_helper_condition"],
          "rescaling_intervals":1000,"native_paper_reproduction_claimed":False,"frozen_helper_result":native,
          "adequacy":adequacy,"task_transfer_provenance":fake_reference(work/"task_transfer_v2.json")})
    manifest={"kind":c["manifest_contract"]["kind"],"status":c["manifest_contract"]["status"],
      "condition":c["observation_condition"],"rescaling_intervals":1000,"truth_or_oracle_read":False,
      "native_paper_reproduction_claimed":False,"scope_parents":c["parent_ids"],"replicates":rows,
      **copy.deepcopy(c["observation_provenance"]),
      "phase_manifests":{phase:fake_reference(Path(c["manifest_contract"]["path"]).parent/relative) for phase,relative in c["manifest_contract"]["phase_paths"].items()}}
    for field,path in [("implementation","scripts/infer_biological_paths_tsinfer_observed_v2.py"),("config","configs/biological_paths_tsinfer_observed_v2.json")]:
        manifest[field]=fake_reference(path,c["frozen_files_sha256"][path])
    return c,manifest,rs,ss

def test_source_only_excludes_native_imports_and_array_reads(monkeypatch):
    original_import=builtins.__import__;original_open=Path.open
    def imports(name,*args,**kwargs):
        if name.split(".")[0] in {"torch","numpy","tskit","tsinfer","tsdate","composer","transformers","omegaconf"}:
            raise AssertionError("native dependency imported")
        return original_import(name,*args,**kwargs)
    def opens(path,*args,**kwargs):
        if path.suffix in {".trees",".vcf",".npz",".pt",".pkl",".npy"}:
            raise AssertionError("array payload read")
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",imports);monkeypatch.setattr(Path,"open",opens)
    report=b.source_report(b.policy())
    assert report["status"]=="inactive_source_only_no_native_or_array_execution"
    assert not report["native_imported"] and not report["actual_arrays_opened"] and not report["GPU"]
    assert len(report["native_function_ASTs"][b.FIT])==20
    assert len(report["native_function_ASTs"][b.RETRIEVAL])==6

@pytest.mark.parametrize("name",["source_model","execute_native","bootstrap","native_runtime","train_phase","readout"])
def test_native_capsule_and_CLI_are_inactive_before_side_effects(name):
    c=b.policy();ns=b.native_capsule(c)
    with pytest.raises(RuntimeError,match="inactive"):
        ns[name]()
    with pytest.raises(RuntimeError,match="inactive"):
        b.main(["--execute"])

def test_preparation_tail_and_runtime_reuse_are_exact_and_inactive():
    c=b.policy();core,h=b.preparation_core(c)
    n=b.function_nodes(b.FIT,c,["prepare_paths"])["prepare_paths"]
    start=next(i for i,x in enumerate(n.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="tree_path" for t in x.targets))
    import hashlib
    expected=hashlib.sha256(ast.dump(ast.Module(body=n.body[start:],type_ignores=[]),include_attributes=False).encode()).hexdigest()
    assert h==expected
    with pytest.raises(RuntimeError,match="inactive"):core()
    changed=copy.deepcopy(c);changed["execution_enabled"]=True
    with pytest.raises(RuntimeError,match="inactive"):b.execution_policy(changed,b.digest(b.CONFIG))

def test_good_all_parent_metadata_retains_gapped_native_node_IDs():
    c,m,r,s=fake_metadata();records=b.validate_combined(c,m,r,s)
    assert list(records)==c["parent_ids"]
    for p,record in records.items():
        assert record["namespace"]==p+":"+"0"*64
        assert record["native"]["sample_index_to_inferred_tree_node_id"][1]["inferred_tree_node_id"]==4002
        assert record["metadata_only_actual_tree_and_corpus_unqualified"]

@pytest.mark.parametrize("change",["inner_parent","outer_condition","helper_condition","wrong_phase","foreign_tree","source_sample","source_haplotype","query_label","missing_support","runtime_pin","truth_field","bool_index","bool_mapping"])
def test_corrupt_metadata_fails_before_any_tree_read(monkeypatch,change):
    c,m,r,s=fake_metadata();row=m["replicates"][0];native=row["frozen_helper_result"];p=row["parent_id"]
    if change=="inner_parent":native["parent_id"]=c["parent_ids"][1]
    elif change=="outer_condition":row["condition"]="Relate"
    elif change=="helper_condition":native["condition"]="other"
    elif change=="wrong_phase":row["adequacy"]["phase"]="remaining"
    elif change=="foreign_tree":native["inferred_tree"]["path"]="foreign/observed.trees"
    elif change=="source_sample":r[p][0]["sample_id"]="other_subject"
    elif change=="source_haplotype":r[p][0]["haplotype"]=1
    elif change=="query_label":r[p][160]["observed_reference_class"]=0;s[p][160]["observed_reference_class"]=0
    elif change=="missing_support":native["coordinate_contract"]["retained_fixed100bp_indexes"]=[0,2,3]
    elif change=="runtime_pin":m["runtime"]["sha256"]="1"*64
    elif change=="truth_field":s[p][160]["query_truth_label"]=0;r[p][160]["query_truth_label"]=0
    elif change=="bool_index":r[p][0]["sample_index"]=False;s[p][0]["sample_index"]=False;native["sample_index_to_inferred_tree_node_id"][0]["sample_index"]=False
    elif change=="bool_mapping":native["sample_index_to_inferred_tree_node_id"][0]["inferred_tree_node_id"]=True
    monkeypatch.setattr(b,"digest",lambda *a:(_ for _ in ()).throw(AssertionError("real bytes read")))
    with pytest.raises(ValueError):b.validate_combined(c,m,r,s)

def test_source_roster_join_is_explicit_copy_without_relabelling_condition():
    c,m,r,s=fake_metadata();records=b.validate_combined(c,m,r,s);native=m["replicates"][0]["frozen_helper_result"];before=copy.deepcopy(native)
    entry=b.preparation_entry(records[c["parent_ids"][0]])
    assert native==before and "source_manifest" not in native["observed_sample_manifest"]
    assert entry["observed_sample_manifest"]["source_manifest"]==native["original_observed_sample_manifest"]
    assert entry["condition"]==c["native_helper_condition"] and entry["inferred_tree"]==native["inferred_tree"]

def good_counts():
    return {"unique_full_paths":11000,"MLM_train":10000,"MLM_val":1000,"pure_reference_train":5000,"pure_reference_val":500,"query_full_paths_scored":4000}

def test_exact_budget_reports_all_paths_and_native_drop_last_updates():
    result=b.corpus_budget(b.policy(),good_counts())
    assert result["MLM_optimizer_updates"]==95 and result["contrastive_optimizer_updates"]==1
    assert result["pure_reference_corpus_paths_all_train_plus_val"]==5500
    assert result["exhaustive_query_corpus_pairs"]==22000000
    assert not result["query_truth_trains_any_model"]

@pytest.mark.parametrize("kind",["below4096","pair_cost","bool_count"])
def test_corpus_budget_never_silently_shrinks_or_accumulates_contrastive_batch(kind):
    c=b.policy();counts=good_counts()
    if kind=="below4096":counts["pure_reference_train"]=4095
    elif kind=="pair_cost":counts.update(unique_full_paths=1000000,MLM_train=900000,MLM_val=100000,pure_reference_train=500000,pure_reference_val=100000,query_full_paths_scored=100000)
    else:counts["MLM_val"]=True
    with pytest.raises(ValueError):b.corpus_budget(c,counts)

def test_full_reference_corpus_keeps_all_train_val_rows_and_duplicate_embeddings():
    import numpy as np
    ns=b.retrieval_core(b.policy())
    paths=((10,777),(11,777),(12,888),(13,888),(20,777),(21,999),(30,777))
    ids,labels=ns["corpus_roles"](paths,np.array([20,21]),np.array([10,11,12,13]),np.array([0,1,2,3]))
    assert ids.tolist()==[0,1,2,3] and labels.tolist()==[0,1,2,3]
    corpus=np.array([[1,0],[1,0],[0,1],[0,-1]],np.float32)
    query=np.array([[1,0]],np.float32)
    values,neighbors,ties=ns["exhaustive_neighbors"](query,corpus,ids,labels,1,2,save_k=4)
    assert neighbors.tolist()==[[0,1,2,3]] and values.tolist()==[[1,1,0,0]]
    vote=ns["hard_votes"](neighbors,ids,labels,1)
    assert vote.tolist()==[[1,0,0,0]]
    assert ties.tolist()==[[1,1,0,0]]
    projected=ns["project_grid"](np.array([4]),vote,np.array([[4,4],[4,4]]))
    assert projected.shape==(2,2,4) and np.all(projected[:,:,0]==1)
