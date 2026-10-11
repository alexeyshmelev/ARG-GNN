#!/usr/bin/env python3
"""Run unchanged compiled preparation on handmade data only; no live admission."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
AUDIT=ROOT/"research/reviews/official_argformer_tsinfer_parent_v3_author"
ALLOWED=AUDIT/"tmp"

def load_dispatcher(version=3):
    path=ROOT/("scripts/prepare_official_argformer_tsinfer_parent_v%d.py"%version)
    spec=importlib.util.spec_from_file_location("_handmade_parent_v%d"%version,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class HandmadeStorage:
    def __init__(self): self.check_calls=0
    def check(self):
        self.check_calls+=1
        return {"handmade_only":True,"live_ledger_inspected":False}

def run_handmade_case(case_dir,dispatcher=None):
    case=Path(case_dir)
    if case.is_symlink() or case.resolve()!=case or ALLOWED not in case.parents or case.exists():
        raise ValueError("Fresh owned audit temp case required")
    case.mkdir(parents=True)
    h=dispatcher or load_dispatcher()
    c=h.policy();h.verify_source(c);bridge,bc=h.load_bridge(c)
    import numpy as np
    import tskit
    table=tskit.TableCollection(sequence_length=400)
    rows=[]
    for sample in range(280):
        age=float(sample%7)
        node=table.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=age)
        role="pure_source_reference" if sample<96 else ("mixed_reference_only" if sample<160 else "query")
        rows.append(dict(sample_index=sample,inferred_tree_node_id=node,observed_node_id=node,
                         role=role,observed_reference_class=sample//24 if sample<96 else None,age_generations=age))
    roots=[table.nodes.add_row(time=20),table.nodes.add_row(time=30)]
    for sample in range(280):
        table.edges.add_row(0,200,roots[0],sample)
        table.edges.add_row(200,400,roots[1],sample)
    table.sort()
    tree=table.tree_sequence()
    assert tree.num_samples==280 and tree.num_trees==2 and tree.num_populations==0
    inputs=case/"handmade_inputs";inputs.mkdir()
    tree_path=inputs/"handmade_observed.trees";tree.dump(tree_path)
    def metadata_file(name,value):
        path=inputs/name
        with path.open("x") as stream: stream.write(json.dumps(value,sort_keys=True)+"\n")
        return path
    roster=metadata_file("inferred_roster.json",rows)
    original=metadata_file("source_roster.json",{"handmade_only":True})
    provenance=metadata_file("completion.json",{"handmade_only":True,"condition":"observed_only"})
    grid=inputs/"fixed_grid.npz"
    np.savez_compressed(grid,positions_bp=np.arange(50,2000000,100,dtype=np.float64))
    def ref(path):return dict(path=str(path),sha256=h.sha(path))
    coordinate=dict(original_parent_interval_bp0=[0,2000000],normalization_length_bp=2000000,
                    fixed_grid={**ref(grid),"total":20000,"center_formula_bp0":"50 + 100*grid_index"},
                    retained_fixed100bp_indexes=[10,11,12,13],local_to_original_offset_bp=1000,
                    coverage_original_bp0=[1000,1400])
    parent=c["default_parent"]
    entry=dict(parent_id=parent,split="train",status="inferred_both_conditions",
               inferred_tree={**ref(tree_path),"condition":c["observation_condition"]},
               observed_sample_manifest={**ref(roster),"rows":280,"source_manifest":ref(original)},
               completion_provenance=ref(provenance),coordinate_contract=coordinate,
               sample_index_to_inferred_tree_node_id=[dict(sample_index=i,inferred_tree_node_id=i) for i in range(280)])
    input_pins={str(path):h.sha(path) for path in inputs.iterdir() if path.is_file()}
    fixture_config=copy.deepcopy(c)
    fixture_config["output_root"]=str(case/"owned")
    fixture_config["launch_log_root"]=str(case/"logs")
    storage=HandmadeStorage()
    authority=h.Authority(fixture_config,parent,{"CPU_only":True,"executable_sha256":"0"*64},storage)
    authority.output.mkdir(parents=True)
    authority.owns_output=True
    core=h.preparation_callable(fixture_config,bridge,bc,authority)
    body=h.preparation_definition(fixture_config,bridge,bc)
    body_sha=hashlib.sha256(ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False).encode()).hexdigest()
    manifest_sha=hashlib.sha256(b"handmade_only_manifest_no_real_observations").hexdigest()
    prepared=core(entry,parent,authority.output/"prepared",manifest_sha,bc)
    directory=authority.output/"prepared"
    assert prepared["schema"]=="official_argformer_prepared_observed_arg_v1"
    assert prepared["status"]=="prepared_observed_paths_only" and prepared["true_query_ancestry_read"] is False
    assert prepared["unique_full_paths"]==560 and prepared["path_counts"]["mlm"]=={"train":504,"val":56}
    assert sum(prepared["path_counts"]["contrastive"].values())==192
    assert prepared["inferred_tree_sha256"]==h.sha(tree_path)
    assert prepared["namespace"]==parent+":"+h.sha(tree_path)
    assert prepared["coordinate_contract"]==coordinate
    assert json.loads((directory/"prepared_manifest.json").read_text())==prepared
    for relative,digest in prepared["files_sha256"].items():
        assert h.sha(directory/relative)==digest
    qualification=h.qualify_corpus(fixture_config,bridge,bc,authority,prepared,np)
    assert qualification["status"]=="prepared_native_source_budget_infeasible"
    assert qualification["budget"]["native_MLM512_contrastive4096_feasible"] is False
    assert qualification["budget"]["contrastive_optimizer_updates"]==0
    assert qualification["tokenization"]["tokenized_rows"]==560
    assert qualification["full_pure_reference_corpus_paths"]==192
    assert qualification["query_truth_used"] is False
    assert qualification["model_fit_or_retrieval_executed"] is False
    assert qualification["full_readout_occurrence_partitions_validated"] is True
    assert input_pins=={str(path):h.sha(path) for path in inputs.iterdir() if path.is_file()}
    report=dict(kind="official_argformer_v3_handmade_full_preparation_smoke",status="PASS",
                source_unchanged_preparation_body_executed=True,native_preparation_tail_AST_sha256=body_sha,
                dispatcher_source_sha256=h.sha(h.SCRIPT),dispatcher_config_sha256=h.sha(h.CONFIG),
                runtime={"numpy":np.__version__,"tskit":tskit.__version__,"python":list(sys.version_info[:3])},
                handmade_samples=280,handmade_trees=2,handmade_role_counts={"pure":96,"mixed":64,"query":120},
                unique_full_paths=560,path_counts=prepared["path_counts"],prepared_files_bound=len(prepared["files_sha256"]),
                qualification_status=qualification["status"],tokenization=qualification["tokenization"],
                storage_check_calls=storage.check_calls,authority_json_new_used=True,live_admission_or_ledger_read=False,
                actual_project_arrays_read=False,query_ancestry_read=False,model_fit_or_retrieval_executed=False,
                input_files_unchanged=True,prepared_manifest_path=str(directory/"prepared_manifest.json"),
                prepared_manifest_sha256=h.sha(directory/"prepared_manifest.json"))
    with (case/"smoke_receipt.json").open("x") as stream:
        stream.write(json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir",required=True)
    args=parser.parse_args()
    print(json.dumps(run_handmade_case(args.case_dir),sort_keys=True))
