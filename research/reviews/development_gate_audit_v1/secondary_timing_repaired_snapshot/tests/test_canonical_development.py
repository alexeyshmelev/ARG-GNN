"""Synthetic-only completion/source/byte/fit/selection tests for canonical development."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pytest
import torch

from arggnn import argmix_development_bindings as shared
import arggnn.canonical_development_bindings as bindings
import arggnn.train_canonical_development as runner
from arggnn.canonical_development_bindings import (CanonicalIntegritySnapshot,bind_completed_canonical_development,
    capture_training_snapshot,check_plan,grid,implementation_hashes)
from arggnn.train_canonical_development import (evaluate,factory,join_external_targets,prepare_observed_parent,
    prepare_verified_parents,select_family,train_configuration)
from test_argmix_development import completed_files,save
from test_canonical_gru import genuine_sequence


@pytest.fixture(autouse=True)
def one_cpu_thread():torch.set_num_threads(1)


def current_plan():
    root=Path(__file__).resolve().parents[1]
    return json.loads((root/bindings.PLAN).read_bytes())


@pytest.fixture
def canonical_files(completed_files):
    f=completed_files
    f.canonical_plan_path=f.root/bindings.PLAN
    f.canonical_plan_sha=save(f.canonical_plan_path,current_plan())
    return f


def prepared(sequence=None):
    return prepare_observed_parent(sequence or genuine_sequence(dtype=torch.float32),parent_id="synthetic_parent",
        split="train",source_tensor_sha256="a"*64,source_prepared_sha256="b"*64)


def tiny_settings():
    return {"num_layers":2,"temporal_layers":2,"epochs":5,"patience":8,"weight_decay":.0001,"gradient_clip_norm":1.}


def tiny_parents():
    seq=genuine_sequence(dtype=torch.float32);order=torch.tensor([2,0,1])
    seq=replace(seq,positions=seq.positions[order],target_graph=seq.target_graph[order])
    value=prepared(seq)
    return {split:[join_external_targets(value,torch.zeros(3,2,dtype=torch.long),split+"_parent",{},2)]
        for split in ("train","val")}


def tiny_binding():return {"protocol_sha256":"a"*64,"implementation_sha256":{"code":"b"*64}}


def test_nonexecuting_reviewed_plan_extends_shared_catalog_and_frozen_capacity():
    plan=current_plan();check_plan(plan)
    assert len(plan["configurations"])*len(plan["training_seeds"])==24
    pins=implementation_hashes()
    assert set(pins)==set(shared.implementation_hashes())|set(bindings.extra_implementation_paths())
    assert all(pins[name]==value for name,value in shared.implementation_hashes().items())
    for width,capacity in ((32,37988),(64,145604)):
        net=factory({"id":"capacity","hidden_dim":width,"dropout":0.,"learning_rate":.001},tiny_settings())
        assert sum(p.numel() for p in net.parameters())==capacity
        manifest=net.configuration_manifest()
        assert manifest["GNN_depth"]==manifest["GRU_depth"]==2 and manifest["bidirectional"]
    assert "every retained canonical record" in plan["input_scope"]


@pytest.mark.parametrize("fault",["test","pilot","seed","epochs","source_depth","temporal_depth","dropout",
    "width","lr","id","grid","boolean","scope","canonicalization","nll","pins"])
def test_fixed_plan_scope_budget_and_configuration_drift_are_rejected(fault):
    plan=current_plan()
    if fault=="test":plan["splits"]["test"]=1
    elif fault=="pilot":plan["splits"]["train"]=16
    elif fault=="seed":plan["training_seeds"].append(404)
    elif fault=="epochs":plan["epochs"]=49
    elif fault=="source_depth":plan["num_layers"]=1
    elif fault=="temporal_depth":plan["temporal_layers"]=1
    elif fault=="dropout":plan["configurations"][0]["dropout"]=.2
    elif fault=="width":plan["configurations"][0]["hidden_dim"]=16
    elif fault=="lr":plan["configurations"][0]["learning_rate"]=.003
    elif fault=="id":plan["configurations"][0]["id"]="../escape"
    elif fault=="grid":plan["configurations"]=plan["configurations"][:-1]
    elif fault=="boolean":plan["configurations"][0]["dropout"]=False
    elif fault=="scope":plan["input_scope"]="anchor subset"
    elif fault=="canonicalization":plan["canonicalization"]["merge_identical_adjacent"]=False
    elif fault=="nll":plan["selection"]="floored probabilities"
    else:plan["reviewed_implementation_sha256"]={}
    with pytest.raises((RuntimeError,ValueError)):check_plan(plan)


def test_actual_receipt_bound_config_and_all_parent_preprocessing(canonical_files):
    f=canonical_files;path=f.root/"configs/canonical_bound.json"
    settings,content,digest=bind_completed_canonical_development(f.completion_path,f.completion_sha,
        f.canonical_plan_path,f.canonical_plan_sha,path)
    assert path.read_bytes()==content and shared.sha(path)==digest
    assert settings["splits"]==shared.SPLITS and settings["input_scope"]==bindings.INPUT_SCOPE
    assert settings["cross_tree_aggregation"]==bindings.AGGREGATION
    for key in bindings.SOURCE_KEYS:assert shared.valid_sha(settings[key])
    captured,manifest,snapshot=capture_training_snapshot(path,digest)
    assert captured==settings and snapshot.bytes_for(path)==content
    parents=prepare_verified_parents(manifest,snapshot)
    assert len(parents["train"])==64 and len(parents["val"])==16
    assert [p.metadata["replicate"] for p in parents["val"]]==list(range(16))
    for split in ("train","val"):
        for parent in parents[split]:
            assert parent.parent_id==settings["dataset_manifest_sha256"]+":"+split+":"+str(parent.metadata["replicate"])
            assert parent.prepared.sequence.x.shape[1]==48 and parent.prepared.sequence.num_queries==64
            assert len(parent.tiles)==8
            torch.testing.assert_close(parent.prepared.sequence.positions,f.sequence.positions,rtol=0,atol=0)
            assert parent.metadata["adapter"]["truth_annotations_used"]==[]
    with pytest.raises(RuntimeError,match="Never overwrite"):
        bind_completed_canonical_development(f.completion_path,f.completion_sha,f.canonical_plan_path,f.canonical_plan_sha,path)
    with pytest.raises(RuntimeError,match="Control SHA256 mismatch"):capture_training_snapshot(path,"e"*64)
    changed=deepcopy(settings);changed["configurations"][0]["hidden_dim"]=16
    changed_sha=save(path,changed)
    with pytest.raises(RuntimeError,match="differs from captured plan"):
        capture_training_snapshot(path,changed_sha)


@pytest.mark.parametrize("fault",["incomplete","test","freeze","source"])
def test_bad_completion_blocks_inputs_models_cuda_and_config_output(canonical_files,monkeypatch,fault):
    f=canonical_files;completion=deepcopy(f.completion)
    if fault=="incomplete":completion["all_stages_complete"]=False
    elif fault=="test":completion["test_parents"]=1
    elif fault=="freeze":completion["development_execution_freeze_file_sha256"]="e"*64
    else:Path(f.records[0]["tensor_file"]).write_bytes(b"bad source")
    completion_sha=save(f.completion_path,completion)
    def forbidden(*args,**kwargs):raise AssertionError("Input/model/CUDA admitted before verified completion")
    monkeypatch.setattr(runner,"prepare_verified_parents",forbidden)
    monkeypatch.setattr(runner,"factory",forbidden);monkeypatch.setattr(runner,"require_device",forbidden)
    output=f.root/"configs/must_not_exist.json"
    with pytest.raises(RuntimeError):
        bind_completed_canonical_development(f.completion_path,completion_sha,f.canonical_plan_path,f.canonical_plan_sha,output)
    assert not output.exists()


def test_main_bad_receipt_cannot_admit_input_model_or_cuda(canonical_files,monkeypatch):
    f=canonical_files;snapshot=CanonicalIntegritySnapshot()
    _,_,snapshot=shared.verify_completed_development(f.completion_path,f.completion_sha,snapshot)
    bad={**f.completion,"all_stages_complete":False};bad_sha=save(f.completion_path,bad)
    settings=bindings.bound_settings(current_plan(),f.canonical_plan_path,f.canonical_plan_sha,bad,f.completion_path,bad_sha,snapshot.sources)
    config=f.root/"configs/bad_completion_config.json";config_sha=save(config,settings)
    def forbidden(*args,**kwargs):raise AssertionError("Input/model/CUDA admitted before verified receipt")
    for name in ("prepare_verified_parents","factory","require_device","configure_deterministic_development"):monkeypatch.setattr(runner,name,forbidden)
    monkeypatch.setattr(torch.cuda,"device_count",forbidden)
    with pytest.raises(RuntimeError,match="completed preparation"):
        runner.main(["--config",str(config),"--expected-config-sha256",config_sha,"--device","cuda","--validate-config"])


@pytest.mark.parametrize("which",["control","source","canonical_code"])
def test_composed_snapshot_rejects_control_source_or_additional_code_drift(canonical_files,monkeypatch,which):
    f=canonical_files;snapshot=CanonicalIntegritySnapshot()
    _,_,snapshot=shared.verify_completed_development(f.completion_path,f.completion_sha,snapshot)
    if which=="control":f.completion_path.write_bytes(f.completion_path.read_bytes()+b" ")
    elif which=="source":Path(f.records[0]["prepared_tensor_file"]).write_bytes(b"changed prepared")
    else:monkeypatch.setattr(bindings,"implementation_hashes",lambda:{"changed":"e"*64})
    with pytest.raises(RuntimeError,match="drift"):snapshot.check()


def test_observed_preparation_poison_label_counterfactual_and_explicit_unsorted_scatter():
    parents=tiny_parents();parent=parents["val"][0];seq=genuine_sequence(dtype=torch.float32)
    class ObservedOnly:
        def __getattr__(self,name):
            if name in {"labels","metadata","truth","tree_sequence","populations"}:raise AssertionError("Target/truth read")
            return getattr(seq,name)
    observed=prepare_observed_parent(ObservedOnly(),parent_id="poison",split="train",source_tensor_sha256="a"*64,source_prepared_sha256="b"*64)
    assert observed.sequence.x.shape[1]==48
    net=factory({"id":"tiny","hidden_dim":8,"dropout":0.,"learning_rate":.02},tiny_settings()).eval()
    records,logits=evaluate(net,[parent],torch.device("cpu"))
    counterfactual=replace(parent,labels=torch.full_like(parent.labels,3))
    torch.testing.assert_close(logits[0],evaluate(net,[counterfactual],torch.device("cpu"))[1][0],rtol=0,atol=0)
    expected=torch.empty_like(logits[0])
    for tile in parent.tiles:expected[tile.parent_position_indices]=net(tile.model_input).double()
    torch.testing.assert_close(logits[0],expected,rtol=0,atol=0)
    assert records[0]["weighting"]=="uniform fixed coordinates and query haplotypes within one parent"
    broken=replace(parent,tiles=[replace(parent.tiles[0],parent_position_indices=parent.tiles[0].parent_position_indices.flip(0)),*parent.tiles[1:]])
    with pytest.raises(RuntimeError,match="scatter correspondence"):evaluate(net,[broken],torch.device("cpu"))


def test_tiny_fitting_gradients_selected_checkpoint_and_equal_parent_weighting(tmp_path):
    parents=tiny_parents();configuration={"id":"tiny_fit","hidden_dim":8,"dropout":.1,"learning_rate":.02}
    net=factory(configuration,tiny_settings());net(parents["train"][0].tiles[0].model_input).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    result,logits=train_configuration(configuration,tiny_settings(),parents,torch.device("cpu"),101,tmp_path/"tiny.pt",tiny_binding())
    assert result["history"][-1]["train_nll"]<result["history"][0]["train_nll"]
    assert logits[0].dtype==torch.float64 and result["best_epoch"]>=0
    saved=torch.load(tmp_path/"tiny.pt",map_location="cpu",weights_only=True)
    assert saved["protocol_sha256"]=="a"*64 and saved["seed"]==101
    assert result["ending_config_sha256"]=="a"*64
    original=parents["val"][0]
    raw=genuine_sequence(dtype=torch.float32);raw=replace(raw,positions=raw.positions[:1],target_graph=raw.target_graph[:1])
    short=join_external_targets(prepared(raw),torch.full((1,2),3,dtype=torch.long),"short",{},2)
    parents["val"].append(short);settings=tiny_settings();settings["epochs"]=2
    unequal,_=train_configuration({**configuration,"dropout":0.},settings,parents,torch.device("cpu"),101,tmp_path/"equal.pt",tiny_binding())
    a,b=unequal["per_parent"]
    assert unequal["validation_nll"]==pytest.approx((a["nll"]+b["nll"])/2,abs=1e-14)
    assert abs(unequal["validation_nll"]-(3*a["nll"]+b["nll"])/4)>1e-4


def test_epoch_input_control_drift_is_detected_before_checkpoint(tmp_path):
    snapshot=CanonicalIntegritySnapshot();control=tmp_path/"control.json";digest=save(control,{"immutable":True})
    snapshot.control(control,digest)
    def mutate(item):control.write_bytes(control.read_bytes()+b" ")
    path=tmp_path/"must_not_save.pt"
    with pytest.raises(RuntimeError,match="drift"):
        train_configuration({"id":"mutation","hidden_dim":8,"dropout":0.,"learning_rate":.02},tiny_settings(),
            tiny_parents(),torch.device("cpu"),101,path,tiny_binding(),snapshot.check,mutate)
    assert not path.exists()


def test_family_selection_requires_original_bytes_spec_capacity_bindings_and_all_seeds():
    settings={**tiny_settings(),"configurations":grid(),"family_tie_tolerance_nats":.0001,
        "dataset_manifest_sha256":"b"*64,"dataset_protocol_sha256":"c"*64,
        "bound_implementation_sha256":{"code":"d"*64},"source_artifact_sha256":{"data/synthetic_input":"e"*64},
        **{key:"f"*64 for key in bindings.SOURCE_KEYS}}
    content=(json.dumps(settings,indent=2)+"\n").encode();digest=hashlib.sha256(content).hexdigest()
    records=[{"configuration_id":c["id"],"configuration":c,"training_seed":seed,"protocol_sha256":digest,
        "parameters":sum(p.numel() for p in factory(c,settings).parameters()),
        "validation_nll":1. if c["hidden_dim"]==64 else 1.00005,
        "implementation_sha256":settings["bound_implementation_sha256"],"source_artifact_sha256":settings["source_artifact_sha256"],
        **{key:settings[key] for key in (*bindings.SOURCE_KEYS,"dataset_manifest_sha256","dataset_protocol_sha256")}}
        for c in grid() for seed in shared.SEEDS]
    assert select_family(records,content,digest)["selected"]["configuration_id"]==grid()[0]["id"]
    with pytest.raises(ValueError,match="Exactly24"):select_family(records[:-1],content,digest)
    altered=deepcopy(settings);altered["configurations"][0]["hidden_dim"]=16
    wrong=deepcopy(records);wrong[0]["configuration"]["hidden_dim"]=16
    with pytest.raises(ValueError,match="Original canonical config bytes"):
        select_family(wrong,json.dumps(altered).encode(),digest)
    with pytest.raises(ValueError,match="original config/spec/capacity"):
        select_family(wrong,content,digest)
    wrong=deepcopy(records);wrong[0]["parameters"]=1
    with pytest.raises(ValueError,match="original config/spec/capacity"):select_family(wrong,content,digest)
    with pytest.raises(ValueError,match="Original canonical config bytes"):select_family(records,settings,digest)


@pytest.mark.parametrize("namespace",["confirmatory_v1","final_test","final-test","confirmation","test"])
def test_heldout_namespace_and_alias_are_refused_before_any_byte_read(tmp_path,monkeypatch,namespace):
    target=tmp_path/"allowed_control.json";save(target,{"allowed":True})
    folder=tmp_path/namespace;folder.mkdir()
    alias=folder/"aliased_control.json";alias.symlink_to(target)
    reads=[]
    def forbidden_read(path):reads.append(path);raise AssertionError("Held-out bytes opened")
    monkeypatch.setattr(Path,"read_bytes",forbidden_read)
    with pytest.raises(ValueError,match="Held-out canonical development"):
        CanonicalIntegritySnapshot().control(alias,"a"*64)
    with pytest.raises(ValueError,match="Held-out canonical development"):
        bindings.admit_consumed_control_paths({"replicates":[{"tensor_file":str(alias)}]})
    assert reads==[]


def test_generic_synthesis_accepts_actual_canonical_cell_and_keeps_missing_starts_ineligible(tmp_path):
    """One real tiny fit under synthetic source facts, never a development result."""
    from test_development_selection import study,write_json,write_tensor
    from arggnn.development_selection import synthesize_development_selection
    # This existing fixture hand-builds a synthetic evidence plan. It does not
    # simulate chromosomes, run inference, or touch actual cohort/test labels.
    evidence=study.__wrapped__(tmp_path)
    plan=evidence["plan"];old=evidence["configs"]["canonical"]
    configuration=grid()[0]
    settings={**{key:value for key,value in old.items() if key not in ("models","model_specs","model_learning_rates")},
        "family":bindings.FAMILY,"configurations":grid(),"num_layers":2,"temporal_layers":2,"gradient_clip_norm":1.,
        "input_scope":bindings.INPUT_SCOPE,"cross_tree_aggregation":bindings.AGGREGATION,
        "selection":shared.SELECTION,"bound_implementation_sha256":implementation_hashes(),
        "deterministic_policy":runner.DETERMINISTIC_POLICY}
    paths=shared.implementation_paths()|bindings.extra_implementation_paths()
    for name,digest in settings["bound_implementation_sha256"].items():
        target=tmp_path/"implementation"/name;target.parent.mkdir(parents=True,exist_ok=True)
        content=paths[name].read_bytes();target.write_bytes(content)
        assert hashlib.sha256(content).hexdigest()==digest
        plan["implementation_artifacts"][name]={"path":str(target.relative_to(tmp_path)),"sha256":digest}
    config_ref=write_json(tmp_path,"configs/canonical_tree_gru.json",settings)
    plan["families"].pop("canonical")
    plan["families"][bindings.FAMILY]={"role":"canonical_tree_gru","grid_kind":"width_dropout_lr",
        "protocol":config_ref,"expected_implementation_sha256":settings["bound_implementation_sha256"],
        "input_scope":bindings.INPUT_SCOPE,"cross_tree_aggregation":bindings.AGGREGATION}
    plan["required_baseline_families"]=[bindings.FAMILY if name=="canonical" else name for name in plan["required_baseline_families"]]
    plan_ref=write_json(tmp_path,"canonical_selection_plan.json",plan)
    raw=genuine_sequence(dtype=torch.float32)
    positions=torch.tensor([.25,.75],dtype=torch.float64)
    containing=((positions[:,None]>=raw.intervals[:,0])&(positions[:,None]<raw.intervals[:,1])).long().argmax(1)
    raw=replace(raw,positions=positions,target_graph=containing,query_node_ids=torch.tensor([10,20]))
    canonical=prepared(raw)
    assert canonical.sequence.query_node_ids.tolist()==[10,20]
    training=join_external_targets(canonical,torch.zeros(2,2,dtype=torch.long),"synthetic_train",{},2)
    validation=[join_external_targets(canonical,torch.full((2,2),3,dtype=torch.long),p["parent_id"],{},2)
        for p in plan["validation_parents"]]
    policy=runner.configure_deterministic_development()
    binding={"protocol_sha256":config_ref["sha256"],"implementation_sha256":settings["bound_implementation_sha256"],
        "deterministic_policy":policy,
             **plan["source_binding"]}
    checkpoint=tmp_path/"artifacts/canonical_selected_checkpoint.pt";checkpoint.parent.mkdir(exist_ok=True)
    result,logits=train_configuration(configuration,settings,{"train":[training],"val":validation},torch.device("cpu"),101,checkpoint,binding)
    prediction=write_tensor(tmp_path,"artifacts/canonical_selected_logits.pt",{
        "logits":logits,"parent_ids":[p.parent_id for p in validation],
        "positions":[p.prepared.sequence.positions for p in validation],
        "query_node_ids":[p.prepared.sequence.query_node_ids for p in validation],
        "seed":101,"nll_policy":bindings.NLL_POLICY,**binding})
    result.update(validation_logits=prediction["path"],validation_logits_sha256=prediction["sha256"])
    run={"development_namespace":"development-v1","evaluation_split":"fresh_development_validation_only",
        "training_seeds":shared.SEEDS,"training_seed":101,"family":bindings.FAMILY,"nll_policy":bindings.NLL_POLICY,
        **binding,"initial_implementation_sha256":settings["bound_implementation_sha256"],
        "models":{configuration["id"]:result}}
    run_ref={"family_id":bindings.FAMILY,**write_json(tmp_path,"runs/canonical_actual_cell.json",run)}
    receipt=synthesize_development_selection(plan_ref["path"],plan_ref["sha256"],run_receipts=[run_ref],project_root=tmp_path)
    assert receipt.payload["matrix_counts"]["success"]==1
    assert receipt.payload["families"][bindings.FAMILY]["family_selection_eligible"] is False
    assert receipt.payload["families"][bindings.FAMILY]["winner"] is None
    assert receipt.payload["baseline_selection_eligible"] is False
    assert result["parameters"]==37988 and logits[0].dtype==torch.float64
    assert result["checkpoint_replay_nll_delta"]==0.
    assert result["original_selected_validation_nll"]==result["reloaded_validation_nll"]
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert saved["deterministic_policy"]==runner.DETERMINISTIC_POLICY


def test_declared_policy_drift_and_missing_cuda_policy_fail_before_model(tmp_path,monkeypatch):
    policy=runner.configure_deterministic_development()
    def forbidden(*args,**kwargs):raise AssertionError("Model admitted under invalid execution policy")
    monkeypatch.setattr(runner,"factory",forbidden)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":16:8")
    with pytest.raises(RuntimeError,match="execution policy drift"):
        train_configuration({"id":"policy_guard","hidden_dim":8,"dropout":0.,"learning_rate":.02},tiny_settings(),
            tiny_parents(),torch.device("cpu"),101,tmp_path/"must_not_save.pt",{**tiny_binding(),"deterministic_policy":policy})
    with pytest.raises(RuntimeError,match="Declared strict deterministic"):
        train_configuration({"id":"policy_guard","hidden_dim":8,"dropout":0.,"learning_rate":.02},tiny_settings(),
            tiny_parents(),torch.device("cuda"),101,tmp_path/"must_not_save_cuda.pt",tiny_binding())
    assert not (tmp_path/"must_not_save.pt").exists() and not (tmp_path/"must_not_save_cuda.pt").exists()


def test_replay_failure_preserves_original_and_reloaded_NLL_evidence(tmp_path,monkeypatch):
    calls=0
    def controlled_evaluation(model,parents,device):
        nonlocal calls
        calls+=1
        return [{"nll":1. if calls==1 else 1.001}],[]
    monkeypatch.setattr(runner,"evaluate",controlled_evaluation)
    settings=tiny_settings();settings["epochs"]=1
    with pytest.raises(RuntimeError,match="strict deterministic execution") as result:
        train_configuration({"id":"replay_guard","hidden_dim":8,"dropout":0.,"learning_rate":.02},settings,
            tiny_parents(),torch.device("cpu"),101,tmp_path/"selected.pt",tiny_binding())
    evidence=result.value.replay_evidence
    assert evidence["original_selected_validation_nll"]==1.
    assert evidence["reloaded_validation_nll"]==1.001
    assert evidence["checkpoint_replay_nll_delta"]==pytest.approx(.001)
