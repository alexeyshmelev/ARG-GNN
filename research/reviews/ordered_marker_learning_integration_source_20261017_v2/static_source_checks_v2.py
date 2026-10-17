"""Standard-library source inspection only; never import candidate/model code."""
from __future__ import annotations
import ast,hashlib,json
from pathlib import Path

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/ordered_marker_learning_integration_source_20261017_v2"


def main():
    checks={};locators={};sources={}
    for path in sorted(HERE.glob("*.py")):
        text=path.read_text();tree=ast.parse(text,filename=str(path))
        compile(text,str(path),"exec",dont_inherit=True)
        sources[path.name]=text
        locators[path.name]={n.name:dict(line=n.lineno,end=n.end_lineno)
            for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    checks["all_new_python_AST_and_compile_without_execution"]=True
    body=sources["ordered_learning_gnn_v2.py"]
    checks["forward_and_reverse_block_consumers_checkpointed"]=(
        "checkpoint(forward_block,*args" in body and "checkpoint(reverse_block,*args" in body
        and body.count("kernel.marker_block_v2(")==2)
    checks["no_full_hold_feature_sequence_or_detached_recurrence"]=(
        "marker_sequence(" not in body and "torch.cat(features" not in body
        and ".detach(" not in body and "calls[first:last]" in body)
    checks["mandatory_added_and_backbone_checkpoint_flags"]=(
        "checkpoint_backbone is True and checkpoint_added is True" in body)
    checks["all_parameter_trainability_and_FP32_failstop"]=(
        "all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters())" in body
        and "validate_trainable_fp32_v2(model)" in body)
    producer=sources["producer_v2.py"]
    checks["real_original_objective_and_full960_schedule"]=(
        "row=h45.train_one(" in producer and "for at in range(1,31):" in producer
        and "order=list(TRAIN);random.Random(101+at).shuffle(order)" in producer
        and "steps=={a:960 for a in arms}" in producer and "if at in PASSES:" in producer)
    checks["step_trainability_failstop_and_finite_all_parameter_first_gradients"]=(
        "model.validate_trainable_fp32_v2(value)" in producer and "set(checks)==set(named)" in producer)
    evaluate=producer[producer.index("def run_evaluate_v2("):]
    checks["complete_RAM_matrix_verified_before_either_VAL_truth"]=(
        evaluate.index("ram.verify()")<evaluate.index("reader.truth_gate=True")
        and "load_placement_checkpoints_v2" in evaluate)
    checks["no_execution_CLI_or_new_process_controller"]=(
        "__main__" not in producer and "subprocess" not in producer
        and "argparse" not in producer)
    checks["immutable_pretruth_original_observations_positive_bridge"]=(
        "episodes.freeze_v1(banks,keys,ids)" in sources["observations_v2.py"]
        and "prepare_positive_parent_v2(" in sources["observations_v2.py"]
        and "ORIGINAL_H058_ARMS[0]" in sources["observations_v2.py"])
    snapshot=json.loads((HERE/"SOURCE_REGISTRATION_SNAPSHOT.v2.json").read_text())
    protocol=snapshot["scientific_protocol"]
    checks["exact_full_workload_metadata"]=(
        len(protocol["TRAIN_parent_ids"])==32 and len(protocol["VAL_parent_ids"])==8
        and protocol["passes"]==30 and protocol["updates_per_arm"]==960
        and protocol["updates_all_new_arms"]==1920
        and protocol["VAL_prediction_products_required_before_truth"]==96
        and protocol["observed_haplotypes"]==952 and protocol["query_haplotypes"]==686
        and protocol["observed_references"]==266 and protocol["region_length_bp"]==2000000
        and protocol["checkpoint_passes"]==[5,10,15,20,25,30])
    checks["immutable_source_protocol_hash_authenticates"]=(
        hashlib.sha256((ROOT/snapshot["source_protocol_ref"]["path"]).read_bytes()).hexdigest()
        ==snapshot["source_protocol_ref"]["sha256"])
    contract=json.loads((HERE/"SOURCE_CONTRACT.v2.json").read_text())
    refs=contract["source_refs"]
    checks["mutable_ledger_not_active_dependency"]=(
        not any(x["path"]=="research/experiment_ledger.md" for x in refs)
        and snapshot["historical_drift_evidence"]["active_dependency"] is False)
    checks["all_declared_direct_source_refs_match"]=all(
        hashlib.sha256((ROOT/x["path"]).read_bytes()).hexdigest()==x["sha256"] for x in refs)
    # The accepted source algebra methods must remain identical to inactive v1.
    old=ast.parse((ROOT/"research/reviews/ordered_marker_learning_integration_source_20261017_v1/positive_marker_backend_v1.py").read_text())
    new=ast.parse(sources["positive_marker_backend_v2.py"])
    def methods(tree):
        return {n.name:ast.dump(n,include_attributes=False)
            for cls in tree.body if isinstance(cls,ast.ClassDef)
            for n in cls.body if isinstance(n,ast.FunctionDef)}
    a,b=methods(old),methods(new)
    checks["five_accepted_positive_feature_algebra_methods_unchanged"]=all(
        a[k]==b[k] for k in ("_features","_positive_log_fields","_cut","_path","_group_fields"))
    if not all(checks.values()):raise ValueError("Source checks failed: "+str([k for k,v in checks.items() if not v]))
    report=dict(schema="ordered_marker_static_source_checks_v2",status="SOURCE_CHECKS_PASS_ONLY_NOT_NUMERICAL_OR_FIT_PASS",
        checks=checks,function_locators=locators,declared_direct_refs_checked=len(refs),
        source_files=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(HERE.glob("*.py"))],
        candidate_modules_imported=False,model_instantiation_or_numerical_execution=False,
        data_payload_truth_guard_state_resource_process_job_or_Git_operations=False,
        execution_authority=False)
    path=HERE/"STATIC_CHECKS.v2.json"
    with path.open("x") as f:json.dump(report,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps(dict(status=report["status"],checks=len(checks),Python_files=len(sources),
        direct_refs=len(refs),receipt=str(path.relative_to(ROOT))),indent=2))


if __name__=="__main__":
    main()
