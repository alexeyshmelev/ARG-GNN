"""Synthetic CPU source/binder/training contracts. No simulations or native jobs."""
from copy import deepcopy
from dataclasses import replace
import gzip
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

import arggnn.argmix_development_bindings as bindings
from arggnn.argmix_development_bindings import (IntegritySnapshot,bind_completed_development,
    capture_training_snapshot,check_plan,development_path,grid,safe_id,sha,verify_completed_development)
from arggnn.train_argmix_development import (FEATURE_NAMES,TreeTrainingParent,evaluate,factory,
    load_verified_parents,observed_fixed_tiles,select_family,tensor_sha,train_configuration,verify_parent_contract)
from arggnn.data import GraphSequence
from arggnn.reference_encoder import ReferenceFeatureAdapter
from test_argformer_control import example_sequence


@pytest.fixture(autouse=True)
def one_cpu_thread():torch.set_num_threads(1)


def save(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(payload,indent=2)+"\n").encode())
    return sha(path)


def synthetic_observed_parent():
    times=torch.tensor([0.]*64+[260.]*6+[300.]*6+[250.]*12+[1000.])
    x=torch.zeros(89,8)
    x[:64,4:6]=1
    x[64:88,5]=1
    for c in range(4):x[64+c*6:70+c*6,c]=1
    x[:,6]=times.log1p()/math.log1p(10000.)
    x[:88,7]=(times[-1]-times[:88]).log1p()/math.log1p(10000.)
    a=torch.arange(88);root=torch.full((88,),88)
    edges=torch.cat((torch.stack((root,a)),torch.stack((a,root))),dim=1)
    clades=x[:,:4].clone();clades[-1]=1
    positions=(torch.arange(128,dtype=torch.float64)+.5)/128
    seq=GraphSequence(x,edges,torch.zeros(89,dtype=torch.long),torch.arange(89).masked_fill(torch.arange(89)==88,-1),
        clades,torch.tensor([[0.,1.]],dtype=torch.float64),torch.arange(64),positions,
        torch.zeros(128,dtype=torch.long),torch.tensor([0.,1.],dtype=torch.float64))
    seq.validate()
    return seq


@pytest.fixture
def completed_files(tmp_path,monkeypatch):
    import arggnn.resources as resources
    real_root=Path(__file__).resolve().parents[1]
    plan=json.loads((real_root/bindings.PLAN).read_bytes())
    monkeypatch.setattr(bindings,"ROOT",tmp_path)
    monkeypatch.setattr(resources,"ROOT",tmp_path)
    plan_path=tmp_path/bindings.PLAN;plan_sha=save(plan_path,plan)
    registry=[{"split":split,"replicate":i,"seeds":{"ancestry":100000+(0 if split=="train" else 64)+i,
        "mutations":200000+(0 if split=="train" else 64)+i},
        "inference_seed":300000+(0 if split=="train" else 64)+i}
        for split,count in (("train",64),("val",16)) for i in range(count)]
    prep_code={"synthetic_preparation.py":"f"*64}
    oracle_protocol=json.loads((real_root/"configs/ancient_development64_v1.json").read_bytes())
    oracle_config_sha=save(tmp_path/"configs/ancient_development64_v1.json",oracle_protocol)
    prep_plan=json.loads((real_root/"configs/fresh_development_preparation_v1.json").read_bytes())
    prep_plan_sha=save(tmp_path/"configs/fresh_development_preparation_v1.json",prep_plan)
    freeze={"plan":prep_plan,"simulation_protocol":oracle_protocol,
        "config_files":{"configs/ancient_development64_v1.json":oracle_config_sha,
            "configs/fresh_development_preparation_v1.json":prep_plan_sha},"kind":"fresh_development_preparation_execution_freeze","confirmatory_access":False,
        "not_confirmatory_test_freeze":True,"parent_registry":registry,"source_code_sha256":prep_code}
    freeze_path=tmp_path/bindings.PREPARATION_FREEZE;freeze_sha=save(freeze_path,freeze)
    oracle_protocol=json.loads((real_root/"configs/ancient_development64_v1.json").read_bytes())
    oracle={"protocol":oracle_protocol,"protocol_sha256":oracle_config_sha,"replicates":deepcopy(registry),
        "development_freeze_file_sha256":freeze_sha,"generation_code_sha256":prep_code}
    oracle_path=tmp_path/bindings.PATHS["oracle"];oracle_sha=save(oracle_path,oracle)
    source_parent={"manifest_sha256":oracle_sha,"manifest":str(oracle_path),
        "config":"configs/ancient_development64_v1.json","config_sha256":oracle_config_sha}
    protocol={**deepcopy(oracle_protocol),"benchmark":"ancient-europe-stdpopsim-0.3.0-relate-inferred",
        "inference":{"master_seed":20261102,"method":"Relate","version":"1.2.4"},"source_parent":source_parent}
    protocol_path=(tmp_path/bindings.PATHS["inferred"]).parent/"protocol.json"
    protocol_sha=save(protocol_path,protocol)
    seq=synthetic_observed_parent();labels=torch.zeros(128,64,dtype=torch.long)
    raw=io.BytesIO();torch.save({"sequence":seq.as_dict(),"labels":labels,
        "baseline_probabilities":torch.full((128,64,4),.25)},raw)
    raw_bytes=gzip.compress(raw.getvalue(),mtime=0)
    import hashlib
    raw_sha=hashlib.sha256(raw_bytes).hexdigest()
    adapter=ReferenceFeatureAdapter(8,**bindings.ADAPTER)
    prepared=adapter.precompute(seq)
    provenance=adapter.manifest_provenance(FEATURE_NAMES,raw_sha)
    cache=io.BytesIO();torch.save({"sequence":prepared.as_dict(),"adapter":provenance},cache)
    cache_bytes=cache.getvalue();cache_sha=hashlib.sha256(cache_bytes).hexdigest()
    samples=[{"observed_node_id":i,"sample_id":f"synthetic_q{i//2}","haplotype":i%2,
        "cohort_role":"query","reference_class":None} for i in range(64)]
    samples.extend({"observed_node_id":i,"sample_id":f"synthetic_r{i//2}","haplotype":i%2,
        "cohort_role":"reference","reference_class":(i-64)//6} for i in range(64,88))
    alignment={"targets_equal":True,"positions_equal":True,"query_order_equal":True,
        "retained_target_sha256":tensor_sha(labels),"retained_positions_sha256":tensor_sha(seq.positions),
        "retained_parent_grid_indexes":list(range(128)),
        "query_haplotype_keys":[[row["sample_id"],row["haplotype"]] for row in samples[:64]]}
    records=[]
    for entry in registry:
        directory=(tmp_path/bindings.PATHS["inferred"]).parent/entry["split"]/f"rep{entry['replicate']:03d}"/"inference/dataset"
        directory.mkdir(parents=True)
        tensor=directory/"tensors.pt.gz";tensor.write_bytes(raw_bytes)
        prepared_path=directory/"prepared.pt";prepared_path.write_bytes(cache_bytes)
        metadata={"split":entry["split"],"replicate":entry["replicate"],"seeds":entry["seeds"],
            "master_seed":20261101,"num_classes":4,"feature_names":FEATURE_NAMES,"feature_time_scale":10000,
            "internal_population_annotations_present":False,"paired_alignment":alignment,"samples":samples}
        metadata_path=directory/"metadata.json";metadata_sha=save(metadata_path,metadata)
        records.append({**entry,"directory":str(directory),"metadata":str(metadata_path),"metadata_sha256":metadata_sha,
            "tensor_file":str(tensor),"tensor_sha256":raw_sha,"prepared_tensor_file":str(prepared_path),
            "prepared_tensor_sha256":cache_sha,"adapter_configuration_sha256":provenance["configuration_sha256"],
            "paired_alignment":alignment})
    inferred={"protocol":protocol,"protocol_sha256":protocol_sha,"replicates":records,"source_parent":source_parent,
        "development_freeze_file_sha256":freeze_sha,"generation_code_sha256":prep_code}
    inferred_path=tmp_path/bindings.PATHS["inferred"];inferred_sha=save(inferred_path,inferred)
    boundary={"status":"complete","preparation_sha256":"b"*64,"source":{"manifest_sha256":inferred_sha}}
    boundary_path=tmp_path/bindings.PATHS["boundary"];boundary_sha=save(boundary_path,boundary)
    age={"status":"complete","preparation_sha256":"c"*64,"source_manifest_sha256":inferred_sha,
        "boundary_manifest_sha256":boundary_sha,"boundary_preparation_sha256":"b"*64}
    age_path=tmp_path/bindings.PATHS["age"];age_sha=save(age_path,age)
    facts={stage:{"path":str(tmp_path/relative),"sha256":digest} for (stage,relative),digest in
        zip(bindings.PATHS.items(),(oracle_sha,inferred_sha,boundary_sha,age_sha))}
    facts["inferred"].update(protocol_sha256=protocol_sha,protocol_path=str(protocol_path))
    facts["boundary"]["preparation_sha256"]="b"*64;facts["age"]["preparation_sha256"]="c"*64
    completion={"all_stages_complete":True,"train_parents":64,"validation_parents":16,"test_parents":0,
        "inferred_boundary_and_age_tiles":640,"training_or_confirmatory_selection_frozen":False,
        "stage_manifests":facts,"development_execution_freeze_file_sha256":freeze_sha,
        "parent_registry_sha256":bindings.digest(registry),"preparation_source_code_sha256":prep_code}
    completion_path=tmp_path/bindings.COMPLETION;completion_sha=save(completion_path,completion)
    return SimpleNamespace(root=tmp_path,plan=plan,plan_path=plan_path,plan_sha=plan_sha,
        completion=completion,completion_path=completion_path,completion_sha=completion_sha,
        manifest=inferred,manifest_path=inferred_path,records=records,sequence=prepared,labels=labels)


def test_reviewed_nonexecuting_plan_grid_counts_and_factory_capacity():
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/bindings.PLAN).read_bytes());check_plan(plan)
    assert len(plan["configurations"])*len(plan["training_seeds"])==24
    settings={"num_layers":2,"reference_adapter":bindings.ADAPTER}
    for width,count in ((32,123268),(64,295684)):
        config={"id":"capacity","hidden_dim":width,"dropout":0.,"learning_rate":.001}
        model=factory(config,settings)
        assert sum(p.numel() for p in model.parameters())==count
        assert model.configuration()["token_scope"]=="full_tree"


@pytest.mark.parametrize("fault",["test","pilot","seed","epochs","depth","width","dropout","bool","lr","grid","id","nll","hash"])
def test_plan_boundary_and_budget_drift_fails(fault):
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/bindings.PLAN).read_bytes())
    if fault=="test":plan["splits"]["test"]=1
    elif fault=="pilot":plan["splits"]["train"]=16
    elif fault=="seed":plan["training_seeds"].append(404)
    elif fault=="epochs":plan["epochs"]=49
    elif fault=="depth":plan["num_layers"]=3
    elif fault=="width":plan["configurations"][0]["hidden_dim"]=16
    elif fault=="dropout":plan["configurations"][0]["dropout"]=.2
    elif fault=="bool":plan["configurations"][0]["dropout"]=False
    elif fault=="lr":plan["configurations"][0]["learning_rate"]=.003
    elif fault=="grid":plan["configurations"]=plan["configurations"][:-1]
    elif fault=="id":plan["configurations"][0]["id"]="../escape"
    elif fault=="nll":plan["selection"]="clipped probabilities"
    else:plan["reviewed_implementation_sha256"]={}
    with pytest.raises((RuntimeError,ValueError)):check_plan(plan)


@pytest.mark.parametrize("value",["../escape","/tmp/escape","a/b","a\\b",".","..","", "a"*129,"а"])
def test_safe_ids_reject_path_components(value):
    with pytest.raises(ValueError,match="Unsafe ID"):safe_id(value)


@pytest.mark.parametrize("value",["/tmp/argmix_external", "../argmix_escape", "data/final_test/secret",
    "data/confirmatory_v1/secret", "data/x/test/secret"])
def test_development_path_rejects_external_traversal_and_heldout_before_open(value):
    with pytest.raises(ValueError):development_path(value)


def test_completion_binding_writes_only_factual_exclusive_config_and_exact_snapshots(completed_files):
    f=completed_files;output=f.root/"configs/argmix_bound.json"
    settings,content,digest=bind_completed_development(f.completion_path,f.completion_sha,f.plan_path,f.plan_sha,output)
    assert output.read_bytes()==content and sha(output)==digest
    assert settings["dataset_manifest_sha256"]==sha(f.manifest_path)
    assert settings["splits"]==bindings.SPLITS and settings["confirmatory_access"] is False
    settings2,manifest,snapshot=capture_training_snapshot(output,digest)
    assert settings2==settings and snapshot.bytes_for(output)==content and manifest==f.manifest
    snapshot.check(full_sources=True)
    with pytest.raises(RuntimeError,match="Never overwrite"):
        bind_completed_development(f.completion_path,f.completion_sha,f.plan_path,f.plan_sha,output)
    with pytest.raises(RuntimeError,match="Control SHA256 mismatch"):
        capture_training_snapshot(output,"e"*64)


@pytest.mark.parametrize("fault",["incomplete","test","missingstage","stagepath","factualhash","registry","sourcechain"])
def test_completion_adversarial_facts_are_rejected_without_config_output(completed_files,fault):
    f=completed_files;completion=deepcopy(f.completion)
    if fault=="incomplete":completion["all_stages_complete"]=False
    elif fault=="test":completion["test_parents"]=1
    elif fault=="missingstage":completion["stage_manifests"].pop("age")
    elif fault=="stagepath":completion["stage_manifests"]["inferred"]["path"]="data/final_test/manifest.json"
    elif fault=="factualhash":completion["stage_manifests"]["inferred"]["protocol_sha256"]="e"*64
    elif fault=="registry":completion["parent_registry_sha256"]="e"*64
    else:
        manifest=deepcopy(f.manifest);manifest["source_parent"]["manifest_sha256"]="e"*64
        completion["stage_manifests"]["inferred"]["sha256"]=save(f.manifest_path,manifest)
    new_sha=save(f.completion_path,completion);output=f.root/"configs/must_not_write.json"
    with pytest.raises((ValueError,RuntimeError)):
        bind_completed_development(f.completion_path,new_sha,f.plan_path,f.plan_sha,output)
    assert not output.exists()


@pytest.mark.parametrize("which",["control","source","code"])
def test_run_integrity_rejects_midrun_control_source_and_code_mutations(completed_files,monkeypatch,which):
    f=completed_files;_,_,snapshot=verify_completed_development(f.completion_path,f.completion_sha)
    if which=="control":f.completion_path.write_bytes(f.completion_path.read_bytes()+b" ")
    elif which=="source":Path(f.records[0]["tensor_file"]).write_bytes(b"altered")
    else:monkeypatch.setattr(bindings,"implementation_hashes",lambda:{"altered":"e"*64})
    with pytest.raises(RuntimeError,match="drift"):snapshot.check()


def test_frozen_loader_checks_raw_prefix_and_pair_target_query_schema(completed_files):
    f=completed_files;_,manifest,snapshot=verify_completed_development(f.completion_path,f.completion_sha)
    parents=load_verified_parents(manifest,snapshot)
    assert len(parents["train"])==64 and len(parents["val"])==16
    parent=parents["train"][0]
    assert len(parent.tiles)==8 and parent.sequence.x.shape[1]==48
    torch.testing.assert_close(parent.sequence.x[:,:8],synthetic_observed_parent().x,rtol=0,atol=0)
    metadata=json.loads(Path(f.records[0]["metadata"]).read_bytes())
    case=SimpleNamespace(sequence=parent.sequence,labels=parent.labels,metadata=metadata)
    changed=SimpleNamespace(sequence=case.sequence,labels=1-case.labels,metadata=case.metadata)
    with pytest.raises(RuntimeError,match="Paired external target"):
        verify_parent_contract(changed,f.records[0])
    leaked=replace(parent.sequence,x=parent.sequence.x.clone());leaked.x[0,0]=1
    with pytest.raises(RuntimeError,match="Query ancestry label"):
        verify_parent_contract(SimpleNamespace(sequence=leaked,labels=case.labels,metadata=metadata),f.records[0])


def test_existing_loader_rejects_prefix_tampering_despite_rebound_prepared_hash(completed_files):
    f=completed_files;record=f.records[0];path=Path(record["prepared_tensor_file"])
    bundle=torch.load(path,map_location="cpu",weights_only=True);bundle["sequence"]["x"][0,6]+=.01
    torch.save(bundle,path)
    record["prepared_tensor_sha256"]=sha(path)
    with pytest.raises(RuntimeError,match="original feature prefix"):
        from arggnn.train_study import load_cases
        load_cases(f.manifest,"train")


def tiny_parents():
    seq=example_sequence().to("cpu");seq.x=seq.x.float()
    order=torch.tensor([3,0,2,1]);seq=replace(seq,positions=seq.positions[order],target_graph=seq.target_graph[order])
    tiles=observed_fixed_tiles(seq,2)
    return {split:[TreeTrainingParent(seq,tiles,torch.zeros(4,2,dtype=torch.long),split+"_rep000",{})]
            for split in ("train","val")}


def tiny_settings():
    return {"num_layers":2,"reference_adapter":bindings.ADAPTER,"epochs":5,"patience":8,
            "weight_decay":.0001,"gradient_clip_norm":1.}


def test_observed_only_tiling_label_independence_and_unsorted_prediction_scatter():
    parents=tiny_parents();parent=parents["val"][0]
    class ObservedOnly:
        def __getattr__(self,name):
            if name in {"labels","metadata","truth","tree_sequence","populations"}:
                raise AssertionError("Observed preprocessing accessed target/truth")
            return getattr(parent.sequence,name)
    tiles=observed_fixed_tiles(ObservedOnly(),2)
    assert [t.parent_position_indices.tolist() for t in tiles]==[[1,2,3],[0]]
    torch.manual_seed(31)
    model=factory({"id":"tiny","hidden_dim":8,"dropout":0.,"learning_rate":.02},tiny_settings()).eval()
    records,logits=evaluate(model,[parent],torch.device("cpu"))
    other=replace(parent,labels=torch.full_like(parent.labels,3))
    torch.testing.assert_close(evaluate(model,[other],torch.device("cpu"))[1][0],logits[0],rtol=0,atol=0)
    torch.testing.assert_close(logits[0],model(parent.sequence).double(),rtol=0,atol=0)
    assert records[0]["observations"]==8 and records[0]["nll"]>0
    bad=replace(parent,tiles=[replace(parent.tiles[0],parent_position_indices=torch.tensor([0,2])),parent.tiles[1]])
    with pytest.raises(RuntimeError,match="correspondence"):evaluate(model,[bad],torch.device("cpu"))


def test_factory_synthetic_finite_allfeature_parameter_gradients_and_CPU_fitting(tmp_path):
    parents=tiny_parents();settings=tiny_settings()
    configuration={"id":"synthetic","hidden_dim":8,"dropout":.1,"learning_rate":.02}
    model=factory(configuration,settings);seq=parents["train"][0].sequence
    seq.x.requires_grad_(True);model(seq).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert seq.x.grad is not None and bool((seq.x.grad.abs().sum(0)>0).all())
    seq.x=seq.x.detach()
    binding={"protocol_sha256":"a"*64,"dataset_manifest_sha256":"b"*64,"implementation_sha256":{"code":"c"*64}}
    result,logits=train_configuration(configuration,settings,parents,torch.device("cpu"),101,
        tmp_path/"tiny.pt",binding)
    assert result["history"][-1]["train_nll"]<result["history"][0]["train_nll"]
    assert result["best_epoch"]>=0 and logits[0].dtype==torch.float64
    assert result["model_configuration"]["scientific_status"]==bindings.STATUS


def test_epoch_mutation_fails_before_checkpoint(tmp_path):
    snapshot=IntegritySnapshot();control=tmp_path/"control.json"
    expected=save(control,{"frozen":True});snapshot.control(control,expected)
    def mutation(item):control.write_bytes(control.read_bytes()+b" ")
    configuration={"id":"synthetic","hidden_dim":8,"dropout":0.,"learning_rate":.02}
    checkpoint=tmp_path/"must_not_save.pt"
    with pytest.raises(RuntimeError,match="drift"):
        train_configuration(configuration,tiny_settings(),tiny_parents(),torch.device("cpu"),101,
            checkpoint,{"config":"a"*64},snapshot.check,mutation)
    assert not checkpoint.exists()


def test_raw_float64_equal_parent_NLL_and_complete_seed_tie_selection():
    from arggnn.confirmatory_metrics import logit_metrics
    small=logit_metrics(torch.tensor([[[0.,-1000.,-2000.,-3000.]]]).numpy(),torch.tensor([[3]]).numpy())
    assert small["nll"]==3000. and small["historical_floor_1e12_nll"]<28.
    settings={"num_layers":2,"reference_adapter":bindings.ADAPTER,"configurations":grid(),"dataset_manifest_sha256":"b"*64,"dataset_protocol_sha256":"c"*64,
        "completion_receipt_sha256":"d"*64,"bound_implementation_sha256":{"code":"e"*64},"family_tie_tolerance_nats":.0001}
    records=[{"configuration_id":c["id"],"configuration":c,"training_seed":seed,"validation_nll":1. if c["hidden_dim"]==64 else 1.00005,
        "parameters":295684 if c["hidden_dim"]==64 else 123268,"protocol_sha256":"a"*64,
        **{key:settings[key] for key in ("dataset_manifest_sha256","dataset_protocol_sha256","completion_receipt_sha256")},
        "implementation_sha256":settings["bound_implementation_sha256"]} for c in settings["configurations"] for seed in bindings.SEEDS]
    assert select_family(records,settings,"a"*64)["selected"]["configuration_id"]==grid()[0]["id"]
    with pytest.raises(ValueError,match="All24"):select_family(records[:-1],settings,"a"*64)
    altered=deepcopy(records);altered[0]["protocol_sha256"]="f"*64
    with pytest.raises(ValueError,match="exact config/source/code"):select_family(altered,settings,"a"*64)



def test_unequal_parent_sizes_use_equal_parent_selected_checkpoint_NLL(tmp_path):
    parents=tiny_parents();base=parents["val"][0]
    sequence=replace(base.sequence,positions=base.sequence.positions[:1],target_graph=base.sequence.target_graph[:1])
    short=TreeTrainingParent(sequence,observed_fixed_tiles(sequence,2),torch.full((1,2),3,dtype=torch.long),"short_parent",{})
    parents["val"].append(short)
    settings=tiny_settings();settings["epochs"]=2
    result,_=train_configuration({"id":"equal_parent","hidden_dim":8,"dropout":0.,"learning_rate":.02},
        settings,parents,torch.device("cpu"),101,tmp_path/"equal_parent.pt",
        {"protocol_sha256":"a"*64,"implementation_sha256":{"code":"b"*64}})
    first,second=result["per_parent"]
    assert first["observations"]==8 and second["observations"]==2
    assert result["validation_nll"]==pytest.approx((first["nll"]+second["nll"])/2,abs=1e-14)
    assert abs(result["validation_nll"]-(4*first["nll"]+second["nll"])/5)>1e-4
    saved=torch.load(tmp_path/"equal_parent.pt",map_location="cpu",weights_only=True)
    assert saved["seed"]==101 and saved["protocol_sha256"]=="a"*64
    assert result["ending_config_sha256"]=="a"*64
    assert all(row["val_nll"]==row["validation_nll"] for row in result["history"])


def test_full_source_SHA_guard_catches_changed_bytes_with_stale_stat(completed_files,monkeypatch):
    f=completed_files;_,_,snapshot=verify_completed_development(f.completion_path,f.completion_sha)
    source=Path(f.records[0]["tensor_file"]);source.write_bytes(b"changed input bytes")
    monkeypatch.setattr(bindings,"signature",lambda path:snapshot.source_signatures[str(path)])
    snapshot.check()
    with pytest.raises(RuntimeError,match="source SHA256 drift"):snapshot.check(full_sources=True)
