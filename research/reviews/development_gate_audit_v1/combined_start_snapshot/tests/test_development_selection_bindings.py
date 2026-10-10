"""Small synthetic files with explicit cohort/review-contract mocks, no fitting."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from arggnn.resources import ROOT
from arggnn import development_selection_bindings as binding
from arggnn import development_selection as selection
from arggnn.development_execution import DETERMINISTIC_POLICY


def save(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    content=(json.dumps(value,sort_keys=True,indent=2)+"\n").encode()
    path.write_bytes(content)
    return {"path":str(path.relative_to(root)),"sha256":hashlib.sha256(content).hexdigest()}


class ObservedOnlyCase:
    def __init__(self,parent_id,index):
        self.parent_id=parent_id
        self.positions=torch.tensor([index+.25,index+.75],dtype=torch.float64 if index%2 else torch.float32)
        self.query_node_ids=torch.arange(10,74,dtype=torch.long)
    @property
    def labels(self):raise AssertionError("Binder inspected targets")
    @property
    def baseline_probabilities(self):raise AssertionError("Binder inspected baseline scores")


@pytest.fixture
def study(tmp_path,monkeypatch):
    root=tmp_path
    producer=root/"src/arggnn/producer.py";producer.parent.mkdir(parents=True)
    producer.write_text("# Synthetic reviewed bytes, never executed.\n")
    code_sha=hashlib.sha256(producer.read_bytes()).hexdigest()
    script=root/"scripts/bind_development_selection.py";script.parent.mkdir()
    script.write_text("# Synthetic script bytes, never executed.\n")
    monkeypatch.setattr(binding,"_IMPORTED_SOURCES",{str(producer):code_sha})
    freeze=save(root,"freeze.json",{"kind":"fresh_development_preparation_execution_freeze",
        "confirmatory_access":False,"not_confirmatory_test_freeze":True})
    source=save(root,"source.json",{"protocol":{"splits":{"train":64,"val":16,"test":0},
        "master_seed":20261101,"inference":{"master_seed":20261102}},
        "replicates":[{"split":s,"replicate":i} for s,n in (("train",64),("val",16)) for i in range(n)]})
    oracle=save(root,"oracle.json",{"synthetic_fixture":True})
    bprep=hashlib.sha256(b"boundary preparation").hexdigest();aprep=hashlib.sha256(b"age preparation").hexdigest()
    boundary=save(root,"boundary.json",{"status":"complete","preparation_sha256":bprep,
        "source":{"manifest_sha256":source["sha256"]}})
    age=save(root,"age.json",{"status":"complete","preparation_sha256":aprep,
        "source_manifest_sha256":source["sha256"],"boundary_manifest_sha256":boundary["sha256"],
        "boundary_preparation_sha256":bprep})
    completion_value={"all_stages_complete":True,"train_parents":64,"validation_parents":16,"test_parents":0,
        "inferred_boundary_and_age_tiles":640,"training_or_confirmatory_selection_frozen":False,
        "development_execution_freeze_file_sha256":freeze["sha256"],"development_execution_freeze_path":str(root/freeze["path"]),
        "stage_manifests":{"oracle":oracle,"inferred":source,
            "boundary":{**boundary,"preparation_sha256":bprep},"age":{**age,"preparation_sha256":aprep}}}
    # Actual completion facts use absolute paths, as production receipts may.
    for ref in completion_value["stage_manifests"].values():ref["path"]=str(root/ref["path"])
    completion=save(root,"completion.json",completion_value)
    common={"completion_receipt_sha256":completion["sha256"],"source_dataset_manifest_sha256":source["sha256"],
        "boundary_dataset_manifest_sha256":boundary["sha256"],"boundary_preparation_sha256":bprep,
        "age_dataset_manifest_sha256":age["sha256"],"age_preparation_sha256":aprep,
        "preparation_execution_freeze_sha256":freeze["sha256"]}
    cases=[ObservedOnlyCase(source["sha256"]+":val:"+str(i),i) for i in reversed(range(16))]
    class Dataset:
        def cases(self,split):
            assert split=="val"
            return cases
    tracked=root/"source_artifact.bin";tracked.write_bytes(b"observed-only fixture bytes")
    lock=binding.preparation.SourceSnapshot({str(tracked):hashlib.sha256(tracked.read_bytes()).hexdigest()})
    calls=[]
    def admit(path,digest,*,with_snapshot):
        assert Path(path)==root/"completion.json" and digest==completion["sha256"] and with_snapshot is True
        calls.append("genuine_preparation_admission_mock")
        return deepcopy(completion_value),Dataset(),{},lock
    monkeypatch.setattr(binding.preparation,"verify_completion",admit)
    contracts={};configs={};refs=[]
    for family in binding.FAMILIES:
        pins={"producer.py" if family in binding.FAMILIES[:4] else "src/arggnn/producer.py":code_sha}
        contract={"input_scope":"Synthetic observed-only "+family,"cross_tree_aggregation":"Synthetic "+family+" aggregation","pins":pins}
        contracts[family]=contract
        config={"family":family,"development_namespace":"development-v1","training_seeds":[101,202,303],
            "splits":{"train":64,"val":16,"test":0},"nll_policy":selection.RAW_NLL,**common,
            "epochs":48,"patience":8,"weight_decay":.0001,"selection":binding.preparation.SELECTION,
            "input_scope":contract["input_scope"],"cross_tree_aggregation":contract["cross_tree_aggregation"],
            binding.POLICY_FIELDS[family]:deepcopy(DETERMINISTIC_POLICY),"bound_implementation_sha256":pins,
            "models":[],"model_specs":{},"model_learning_rates":{}}
        if family=="candidate":
            for i,(geometry,depth,rate) in enumerate((g,d,r) for g in ("plain","age") for d in (1,2) for r in (.0003,.001)):
                name=family+"_"+str(i);config["models"].append(name)
                config["model_specs"][name]={"geometry":geometry,"kernel_depth":depth,"coupling":"full","rate_parameterization":"log"}
                config["model_learning_rates"][name]=rate
        elif family=="reference_smoother":
            for i,(bw,pc) in enumerate((b,p) for b in (10000,30000,100000,300000) for p in (.001,.01)):
                name=family+"_"+str(i);config["models"].append(name)
                config["model_specs"][name]={"bandwidth_bp":bw,"pseudocount":pc}
        else:
            for i,(width,dropout,rate) in enumerate((w,d,r) for w in (32,64) for d in (0.,.1) for r in (.0003,.001)):
                name=family+"_"+str(i);config["models"].append(name)
                config["model_specs"][name]={"hidden_dim":width,"dropout":dropout}
                config["model_learning_rates"][name]=rate
            if family in ("canonical_gru","argmix_full_tree","gst_ca_span","graphssm_zoh"):
                config["configurations"]=[{"id":name,**config["model_specs"][name],"learning_rate":config["model_learning_rates"][name]} for name in config["models"]]
                for key in ("models","model_specs","model_learning_rates"):del config[key]
        ref=save(root,"configs/"+family+".json",config)
        configs[family]=config;refs.append({"family_id":family,**ref})
    monkeypatch.setattr(binding,"_family_contract",lambda family:deepcopy(contracts[family]))
    amendment=json.loads((ROOT/binding.SCOPE_AMENDMENT).read_text())
    amendment_ref=save(root,binding.SCOPE_AMENDMENT,amendment)
    return {"root":root,"completion":completion,"completion_value":completion_value,"refs":refs,
        "configs":configs,"contracts":contracts,"amendment":amendment,"amendment_ref":amendment_ref,
        "calls":calls,"cases":cases,"tracked":tracked,"producer":producer,"lock":lock}


def run(study,*,refs=None,output=None,completion_sha=None,amendment_sha=None):
    return binding.bind_development_selection(study["completion"]["path"],completion_sha or study["completion"]["sha256"],
        expected_scope_amendment_sha256=amendment_sha or study["amendment_ref"]["sha256"],
        family_configs=study["refs"] if refs is None else refs,output_directory=output,project_root=study["root"])


def rewrite_config(study,family,change):
    value=deepcopy(study["configs"][family]);change(value)
    ref=save(study["root"],"configs/"+family+".json",value)
    study["configs"][family]=value
    for current in study["refs"]:
        if current["family_id"]==family:current.update(ref)


def test_full_validate_only_and_observed_ordered_contracts(study):
    result=run(study)
    assert result.path is None and result.receipt_path is None
    assert list(result.plan["families"])==list(binding.FAMILIES)
    assert result.plan["required_baseline_families"]==list(binding.FAMILIES[1:])
    assert set(result.plan["source_binding"])==set(selection.BINDINGS)
    assert len(result.plan["source_artifacts"])==5
    assert set(result.plan["implementation_artifacts"])=={"producer.py","src/arggnn/producer.py"}
    assert len({ref["path"] for ref in result.plan["implementation_artifacts"].values()})==1
    assert result.sha256==hashlib.sha256(result.content).hexdigest()
    assert result.receipt["maximum_prescribed_optimizer_starts"]==binding.BUDGET
    assert result.receipt["configuration_seed_cells"]==240 and result.receipt["prescribed_main_optimizer_starts"]==216
    assert result.receipt["scores_inspected"] is False and result.receipt["selection_performed"] is False
    assert result.receipt["authorizes_confirmatory_generation"] is False
    for i,parent in enumerate(result.plan["validation_parents"]):
        case=next(case for case in study["cases"] if case.parent_id.endswith(":val:"+str(i)))
        assert parent=={"parent_id":case.parent_id,"positions_sha256":selection.tensor_fingerprint(case.positions),
            "query_ids_sha256":selection.tensor_fingerprint(case.query_node_ids),"logit_shape":[2,64,4]}
    assert study["calls"]==["genuine_preparation_admission_mock"]
    assert not (study["root"]/"configs/development_selection_bound_v2").exists()
    assert not list((study["root"]/".runtime/tmp").iterdir())


def test_explicit_exclusive_write_is_a_plan_binding_not_selection(study):
    result=run(study,output="configs/bound")
    assert Path(result.path).read_bytes()==result.content
    assert Path(result.receipt_path).read_bytes()==result.receipt_content
    assert selection.load_development_selection_plan(result.path,result.sha256,project_root=study["root"]).values==result.plan
    assert result.receipt["plans_written"] is True and result.receipt["selection_performed"] is False
    with pytest.raises(ValueError,match="fresh output"):run(study,output="configs/bound")
    assert Path(result.path).read_bytes()==result.content


@pytest.mark.parametrize("fault",["missing","wrong_completion_sha","incomplete"])
def test_no_final_output_for_missing_or_incomplete_completion(study,monkeypatch,fault):
    if fault=="missing":(study["root"]/study["completion"]["path"]).unlink()
    elif fault=="incomplete":
        def reject(*args,**kwargs):raise RuntimeError("Successful64/16/TEST0 preparation completion required")
        monkeypatch.setattr(binding.preparation,"verify_completion",reject)
    with pytest.raises((ValueError,RuntimeError)):
        run(study,output="configs/bound",completion_sha="f"*64 if fault=="wrong_completion_sha" else None)
    assert not (study["root"]/"configs/bound").exists()
    assert not (study["root"]/".runtime/tmp").exists()


@pytest.mark.parametrize("fault",["missing_family","duplicate","unknown","missing_config","wrong_config_sha"])
def test_ten_factual_family_references_are_mandatory(study,fault):
    refs=deepcopy(study["refs"])
    if fault=="missing_family":refs.pop()
    elif fault=="duplicate":refs[-1]=deepcopy(refs[0])
    elif fault=="unknown":refs[-1]["family_id"]="unreviewed"
    elif fault=="missing_config":(study["root"]/refs[-1]["path"]).unlink()
    else:refs[-1]["sha256"]="f"*64
    with pytest.raises((ValueError,RuntimeError)):run(study,refs=refs,output="configs/bound")
    assert not (study["root"]/"configs/bound").exists()


@pytest.mark.parametrize("fault",["source","scope","aggregation","dependency","missing_policy","non_strict_policy",
    "boolean_policy_as_integer","missing_selection","epochs","patience","decay","seeds","nll","grid","family"])
def test_wrong_config_scope_source_grid_budget_or_policy_refused(study,fault):
    def change(value):
        if fault=="source":value.pop("age_preparation_sha256")
        elif fault=="scope":value["input_scope"]="unsupported access"
        elif fault=="aggregation":value["cross_tree_aggregation"]="none"
        elif fault=="dependency":value["bound_implementation_sha256"]={}
        elif fault=="missing_policy":value.pop("deterministic_execution_policy")
        elif fault=="non_strict_policy":value["deterministic_execution_policy"]["warn_only"]=True
        elif fault=="boolean_policy_as_integer":value["deterministic_execution_policy"]["deterministic_algorithms"]=1
        elif fault=="missing_selection":value.pop("selection")
        elif fault=="epochs":value["epochs"]=49
        elif fault=="patience":value["patience"]=9
        elif fault=="decay":value["weight_decay"]=.001
        elif fault=="seeds":value["training_seeds"]=[101,202]
        elif fault=="nll":value["nll_policy"]="probability floor"
        elif fault=="grid":value["model_specs"][value["models"][0]]["hidden_dim"]=16
        else:value["family"]="gru_alias"
    rewrite_config(study,"gru",change)
    with pytest.raises((ValueError,RuntimeError)):run(study,output="configs/bound")
    assert not (study["root"]/"configs/bound").exists()


@pytest.mark.parametrize("fault",["required_baseline","allrecord","budget","reserve","checkpoint_rule","family_rule",
    "control_policy","failure_policy","final_access","scores","typed_grid","empty_scope"])
def test_amendment_required_families_and_288_start_policy_guards(study,fault):
    value=deepcopy(study["amendment"])
    if fault=="required_baseline":value["required_baseline_families"].pop()
    elif fault=="allrecord":value["families"][6]["id"]="substitute"
    elif fault=="budget":value["maximum_prescribed_optimizer_starts"]["total"]=289
    elif fault=="reserve":value["maximum_prescribed_optimizer_starts"]["secondary_control_reserve"]=0
    elif fault in ("checkpoint_rule","family_rule","failure_policy"):value.pop(fault)
    elif fault=="control_policy":value.pop("secondary_control_policy")
    elif fault=="final_access":value["authorizes_confirmatory_generation"]=True
    elif fault=="scores":value["no_fresh_development_scores_inspected"]=False
    elif fault=="typed_grid":value["neural_grid"]["gradient_clip_norm"]=True
    else:value["families"][0]["scope"]=""
    ref=save(study["root"],binding.SCOPE_AMENDMENT,value)
    with pytest.raises(ValueError):run(study,output="configs/bound",amendment_sha=ref["sha256"])
    assert not (study["root"]/"configs/bound").exists()


def test_proposal_aliases_normalize_transparently_without_config_relabeling(study):
    refs=deepcopy(study["refs"])
    for ref in refs:
        if ref["family_id"].startswith("allrecord_"):
            ref["family_id"]=ref["family_id"].replace("allrecord_","allrecord_boundary_")
    result=run(study,refs=refs)
    assert result.receipt["proposal_to_actual_family_aliases"]==binding.ALIASES
    assert all(family==study["configs"][family]["family"] for family in result.plan["families"])


@pytest.mark.parametrize("artifact",["source_artifact","config","implementation"])
def test_source_drift_during_strict_temporary_validation_prevents_emit(study,monkeypatch,artifact):
    original=selection.load_development_selection_plan
    def mutate(*args,**kwargs):
        result=original(*args,**kwargs)
        path=study["tracked"] if artifact=="source_artifact" else study["producer"] if artifact=="implementation" else study["root"] / study["refs"][0]["path"]
        before=path.stat();content=path.read_bytes()
        path.write_bytes(content[:-1]+(b"x" if content[-1:]!=b"x" else b"y"))
        os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        return result
    monkeypatch.setattr(selection,"load_development_selection_plan",mutate)
    with pytest.raises((ValueError,RuntimeError),match="changed"):run(study,output="configs/bound")
    assert not (study["root"]/"configs/bound").exists()
    assert not list((study["root"]/".runtime/tmp").iterdir())


def test_hash_conflicts_across_dependency_path_aliases_refused(study):
    family="gst_ca_span"
    study["contracts"][family]["pins"]["src/arggnn/producer.py"]="f"*64
    rewrite_config(study,family,lambda value:value.update(bound_implementation_sha256=deepcopy(study["contracts"][family]["pins"])))
    with pytest.raises(ValueError,match="Conflicting|conflict"):run(study,output="configs/bound")
    assert not (study["root"]/"configs/bound").exists()


@pytest.mark.parametrize("field",["completion","amendment","config","output"])
def test_heldout_paths_rejected_before_any_byte_read(study,monkeypatch,field):
    refs=deepcopy(study["refs"])
    kwargs={"completion_path":study["completion"]["path"],"expected_completion_sha256":study["completion"]["sha256"],
        "expected_scope_amendment_sha256":study["amendment_ref"]["sha256"],"family_configs":refs,
        "output_directory":"configs/bound","project_root":study["root"]}
    if field=="completion":kwargs["completion_path"]="data/final_test_probe/completion.json"
    elif field=="amendment":kwargs["scope_amendment_path"]="research/confirmation/amendment.json"
    elif field=="config":refs[0]["path"]="configs/test/protocol.json"
    else:kwargs["output_directory"]="configs/final-test-plan"
    monkeypatch.setattr(Path,"read_bytes",lambda *_:pytest.fail("Held-out input read preceded guard"))
    with pytest.raises(ValueError,match="Held-out"):binding.bind_development_selection(**kwargs)


def test_heldout_lexical_symlink_alias_is_refused(study):
    target=study["root"]/"final_test_alias.json"
    target.symlink_to(study["root"]/study["refs"][0]["path"])
    refs=deepcopy(study["refs"]);refs[0]["path"]="final_test_alias.json"
    with pytest.raises(ValueError,match="Held-out"):run(study,refs=refs)


@pytest.mark.parametrize("field",["completion","amendment","config"])
def test_placeholder_hashes_refused(study,field):
    refs=deepcopy(study["refs"])
    if field=="config":refs[0]["sha256"]="0"*64
    with pytest.raises(ValueError,match="Real SHA256"):
        run(study,refs=refs,completion_sha="0"*64 if field=="completion" else None,
            amendment_sha="0"*64 if field=="amendment" else None)


@pytest.mark.parametrize("fault",["missing_parent","duplicate_parent","unordered_positions","duplicate_queries"])
def test_observed_parent_contract_refuses_incomplete_or_invalid_coordinates(study,fault):
    if fault=="missing_parent":study["cases"].pop()
    elif fault=="duplicate_parent":study["cases"][-1]=study["cases"][0]
    elif fault=="unordered_positions":study["cases"][0].positions=study["cases"][0].positions.flip(0)
    else:study["cases"][0].query_node_ids[1]=study["cases"][0].query_node_ids[0]
    with pytest.raises(ValueError):run(study,output="configs/bound")
    assert not (study["root"]/"configs/bound").exists()


def cli(study,monkeypatch):
    spec=importlib.util.spec_from_file_location("selection_binder_cli",ROOT/"scripts/bind_development_selection.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,"guard_development_path",lambda path:binding.guard_development_path(path,project_root=study["root"]))
    original=binding.bind_development_selection
    monkeypatch.setattr(module,"bind_development_selection",lambda *args,**kwargs:original(*args,**kwargs,project_root=study["root"]))
    return module


def test_cli_default_missing_real_completion_creates_no_plan(study,monkeypatch):
    module=cli(study,monkeypatch)
    with pytest.raises(RuntimeError,match="No successful"):module.main([])
    assert study["calls"]==[]
    assert not (study["root"]/"configs/development_selection_bound_v2").exists()


def test_cli_existing_completion_requires_both_explicit_hashes(study,monkeypatch):
    module=cli(study,monkeypatch)
    with pytest.raises(RuntimeError,match="Explicit reviewed"):
        module.main(["--completion",study["completion"]["path"],"--write-plan"])
    assert study["calls"]==[]


def test_cli_validate_only_has_no_final_output(study,monkeypatch,capsys):
    module=cli(study,monkeypatch)
    args=["--completion",study["completion"]["path"],"--expected-completion-sha256",study["completion"]["sha256"],
        "--expected-scope-amendment-sha256",study["amendment_ref"]["sha256"]]
    for ref in study["refs"]:args.extend(["--family-config",ref["family_id"],ref["path"],ref["sha256"]])
    module.main(args)
    event=json.loads(capsys.readouterr().out)
    assert event["plans_written"] is False and event["selection_performed"] is False
    assert event["families"]==10 and event["required_baselines"]==9
    assert not (study["root"]/"configs/development_selection_bound_v2").exists()


@pytest.mark.parametrize("field",["stage","freeze"])
def test_declared_completed_heldout_alias_refused_before_admission(study,field):
    value=deepcopy(study["completion_value"])
    if field=="stage":value["stage_manifests"]["inferred"]["path"]="data/final_test_alias/manifest.json"
    else:value["development_execution_freeze_path"]="configs/confirmation/freeze.json"
    study["completion"].update(save(study["root"],study["completion"]["path"],value))
    with pytest.raises(ValueError,match="Held-out"):run(study,output="configs/bound")
    assert study["calls"]==[]
    assert not (study["root"]/"configs/bound").exists()


def test_failed_output_write_rolls_back_owned_partial_bytes(study,monkeypatch):
    original=Path.open
    class BrokenWriter:
        def __init__(self,handle):self.handle=handle
        def __enter__(self):return self
        def __exit__(self,*args):return self.handle.__exit__(*args)
        def write(self,payload):
            self.handle.write(payload[:10]);raise OSError("Synthetic failed disk write")
    def broken(path,*args,**kwargs):
        handle=original(path,*args,**kwargs)
        if path.name=="binding_receipt.json" and args and args[0]=="xb":return BrokenWriter(handle)
        return handle
    monkeypatch.setattr(Path,"open",broken)
    with pytest.raises(OSError,match="Synthetic failed disk write"):run(study,output="configs/bound")
    assert not (study["root"]/"configs/bound").exists()


@pytest.mark.parametrize("family",["canonical_gru","gst_ca_span","graphssm_zoh"])
def test_exact_frozen_producer_policy_field_required(study,family):
    def change(value):value["deterministic_execution_policy"]=value.pop("deterministic_policy")
    rewrite_config(study,family,change)
    with pytest.raises(ValueError,match="Ambiguous producer"):run(study,output="configs/bound")
    assert not (study["root"]/"configs/bound").exists()


def test_receipt_records_strict_producer_policy_field_mapping(study):
    result=run(study)
    assert result.receipt["family_deterministic_policy_fields"]==binding.POLICY_FIELDS
    assert result.receipt["family_deterministic_policy_fields"]["canonical_gru"]=="deterministic_policy"
    assert result.receipt["family_deterministic_policy_fields"]["candidate"]=="deterministic_execution_policy"
