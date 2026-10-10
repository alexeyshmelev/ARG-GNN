"""Canonical comparator/runner contracts, synthetic training and observed input QA."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from arggnn.canonical_gru import (CanonicalTreeGNNGRU,canonical_fixed_tiles,
    prepare_canonical_gnn_input,prepare_canonical_parent)
from arggnn.train_canonical_gru import (bind_training_parent,capture_run_snapshot,evaluate,logits_metrics,
    pilot_path,safe_component,select_family_configuration,source_artifacts,train_configuration,verify_protocol)
from test_tree_canonicalization import example,duplicate_intervals,permute_nodes,assert_same_sequence


ADAPTER={"reference_columns":[0,1,2,3],"time_column":6,"branch_column":7,
         "time_scale":10000,"kernel_scales":[.05,.1,.2]}


@pytest.fixture(autouse=True)
def one_cpu_thread(): torch.set_num_threads(1)


def prepare(seq=None):
    return prepare_canonical_parent(seq or example(),ADAPTER)


def network(dropout=0.,width=8):
    torch.manual_seed(304)
    return CanonicalTreeGNNGRU(48,4,hidden_dim=width,num_layers=2,temporal_layers=2,dropout=dropout).double()


def genuine_sequence(dtype=torch.float64):
    seq=duplicate_intervals(example(dtype=dtype))
    n=len(example().x)
    seq.x[n+10,6]+=.02
    seq.x[2*n+10,6]+=.04
    return seq


def tiny_settings():
    return {"reference_adapter":ADAPTER,"base_feature_count":8,"input_dim":48,"num_classes":4,
        "num_queries":2,"windows":2,"num_layers":2,"temporal_layers":2,"epochs":5,
        "patience":8,"weight_decay":.0001,"gradient_clip_norm":1.}


@pytest.mark.parametrize("dropout",[0.,.1])
def test_own_GNN_GRU_predictions_match_after_unary_subdivision_storage_canonicalization(dropout):
    first=prepare(example())
    expanded=permute_nodes(duplicate_intervals(example(True,True)))
    second=prepare(expanded)
    assert_same_sequence(first.sequence,second.sequence)
    assert second.correspondence.source_graph_to_canonical.tolist()==[0,0,0]
    assert second.sequence.target_graph.tolist()==[0,0,0]
    net=network(dropout).eval()
    before=net(prepare_canonical_gnn_input(first.sequence))
    after=net(prepare_canonical_gnn_input(second.sequence))
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    assert before.shape==(3,2,4) and torch.isfinite(before).all()
    # The identical canonical input also consumes identical stochastic masks
    # when training state is deliberately held fixed.
    net.train()
    torch.manual_seed(305); before=net(prepare_canonical_gnn_input(first.sequence))
    torch.manual_seed(305); after=net(prepare_canonical_gnn_input(second.sequence))
    torch.testing.assert_close(before,after,rtol=0,atol=0)


def test_prefix_summary_refresh_and_external_target_graph_mapping():
    seq=duplicate_intervals(example(True,True))
    seq.x=torch.cat((seq.x,torch.randn(len(seq.x),40,dtype=seq.x.dtype)),dim=-1)
    actual=prepare(seq)
    assert actual.sequence.x.shape[1]==48
    assert actual.correspondence.sequence.x.shape[1]==8
    assert actual.correspondence.provenance["discarded_appended_feature_count"]==40
    torch.testing.assert_close(actual.sequence.x[:,:8],actual.correspondence.sequence.x,rtol=0,atol=0)
    changed=replace(seq,x=torch.cat((seq.x[:,:8],torch.full_like(seq.x[:,8:],1e6)),dim=-1))
    assert_same_sequence(actual.sequence,prepare(changed).sequence)
    torch.testing.assert_close(actual.sequence.positions,seq.positions,rtol=0,atol=0)
    torch.testing.assert_close(actual.sequence.query_node_ids,seq.query_node_ids,rtol=0,atol=0)
    torch.testing.assert_close(actual.sequence.target_graph,actual.correspondence.source_graph_to_canonical[seq.target_graph],rtol=0,atol=0)
    assert actual.adapter_provenance["output_dim"]==48
    assert actual.adapter_provenance["truth_annotations_used"]==[]


def test_observed_only_poison_and_input_label_counterfactual():
    seq=example()
    class ObservedOnly:
        def __getattr__(self,name):
            if name in {"labels","populations","truth","tree_sequence","metadata"}:
                raise AssertionError("Preprocessing cannot access target/truth fields")
            return getattr(seq,name)
    prepared=prepare_canonical_parent(ObservedOnly(),ADAPTER)
    settings=tiny_settings()
    metadata={"replicate":0,"split":"train","num_classes":4}
    first=bind_training_parent(prepared,torch.zeros(3,2,dtype=torch.long),metadata,settings)
    second=bind_training_parent(prepared,torch.full((3,2),3,dtype=torch.long),metadata,settings)
    assert first.prepared is second.prepared
    for a,b in zip(first.tiles,second.tiles):
        assert_same_sequence(a.model_input.sequence,b.model_input.sequence)
    net=network().eval()
    torch.testing.assert_close(evaluate(net,[first],torch.device("cpu"))[1][0],
        evaluate(net,[second],torch.device("cpu"))[1][0],rtol=0,atol=0)


def test_observed_reference_input_counterfactual_reaches_model():
    seq=example()
    changed=replace(seq,x=seq.x.clone())
    changed.x[2,:4],changed.x[4,:4]=seq.x[4,:4].clone(),seq.x[2,:4].clone()
    changed.clades=seq.clades[:,torch.tensor([2,1,0,3])].clone()
    first,second=prepare(seq),prepare(changed)
    assert not torch.equal(first.sequence.x,second.sequence.x)
    net=network().eval()
    assert (net(prepare_canonical_gnn_input(first.sequence))-net(prepare_canonical_gnn_input(second.sequence))).abs().max()>1e-5


def test_query_correspondence_rename_permutation_and_readout_set_independence():
    seq=genuine_sequence()
    original=prepare(seq)
    net=network().eval()
    expected=net(prepare_canonical_gnn_input(original.sequence))
    renamed=replace(seq,node_ids=seq.node_ids.clone())
    mapping={10:90,20:80,30:70,40:60,50:50,60:40}
    for old,new in mapping.items(): renamed.node_ids[seq.node_ids==old]=new
    renamed.query_node_ids=torch.tensor([mapping[int(q)] for q in seq.query_node_ids])
    actual=net(prepare_canonical_gnn_input(prepare(renamed).sequence))
    torch.testing.assert_close(actual,expected,rtol=3e-12,atol=3e-12)
    qorder=torch.tensor([1,0])
    reordered=replace(seq,query_node_ids=seq.query_node_ids[qorder])
    torch.testing.assert_close(net(prepare_canonical_gnn_input(prepare(reordered).sequence)),expected[:,qorder],rtol=3e-12,atol=3e-12)
    order=torch.tensor([2,0,1])
    reordered=replace(seq,positions=seq.positions[order],target_graph=seq.target_graph[order])
    torch.testing.assert_close(net(prepare_canonical_gnn_input(prepare(reordered).sequence)),expected[order],rtol=0,atol=0)
    added=replace(seq,positions=torch.cat((seq.positions,torch.tensor([.33],dtype=seq.positions.dtype))),
        target_graph=torch.cat((seq.target_graph,torch.tensor([1]))))
    torch.testing.assert_close(net(prepare_canonical_gnn_input(prepare(added).sequence))[:3],expected,rtol=0,atol=0)


def test_unsorted_parent_coordinates_scatter_by_explicit_indices():
    seq=genuine_sequence()
    order=torch.tensor([2,0,1])
    seq=replace(seq,positions=seq.positions[order],target_graph=seq.target_graph[order])
    prepared=prepare(seq)
    parent=bind_training_parent(prepared,torch.zeros(3,2,dtype=torch.long),
        {"replicate":0,"split":"val","num_classes":4},tiny_settings())
    assert [t.parent_position_indices.tolist() for t in parent.tiles]==[[1,2],[0]]
    net=network().eval()
    records,predictions=evaluate(net,[parent],torch.device("cpu"))
    expected=torch.empty_like(predictions[0])
    for tile in parent.tiles: expected[tile.parent_position_indices]=net(tile.model_input)
    torch.testing.assert_close(predictions[0],expected,rtol=0,atol=0)
    assert records[0]["observations"]==6


@pytest.mark.parametrize("width,dropout",[(8,0.),(12,.1)])
def test_all_graph_GRU_head_parameters_have_finite_gradients(width,dropout):
    prepared=prepare(genuine_sequence())
    net=network(dropout,width)
    output=net(prepare_canonical_gnn_input(prepared.sequence))
    loss=torch.nn.functional.cross_entropy(output.reshape(-1,4),torch.tensor([0,1,2,3,1,0]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert not any(isinstance(m,torch.nn.Embedding) for m in net.modules())


def test_small_CPU_training_checkpoint_loss_and_external_validation(tmp_path):
    settings=tiny_settings()
    prepared=prepare(genuine_sequence(dtype=torch.float32))
    labels=torch.zeros(3,2,dtype=torch.long)
    parents={split:[bind_training_parent(prepared,labels,{"replicate":0,"split":split,"num_classes":4},settings)] for split in ("train","val")}
    checkpoint=tmp_path/"synthetic_canonical.pt"
    configuration={"id":"synthetic_smoke","hidden_dim":8,"dropout":0.,"learning_rate":.02}
    result,predictions=train_configuration(configuration,settings,parents,torch.device("cpu"),101,checkpoint,"a"*64,"b"*64)
    assert checkpoint.exists() and len(predictions)==1 and predictions[0].shape==(3,2,4)
    assert result["history"][-1]["train_nll"]<result["history"][0]["train_nll"]
    assert result["best_epoch"]>=0 and torch.isfinite(predictions[0]).all()
    assert result["protocol_sha256"]=="a"*64 and result["dataset_manifest_sha256"]=="b"*64


def test_primary_raw_logit_NLL_has_no_historical_probability_floor():
    logits=torch.tensor([[[0.,-1000.,-2000.,-3000.]]],dtype=torch.float64)
    result=logits_metrics(logits,torch.tensor([[3]]))
    assert result["nll_stable_log_softmax"]==3000.
    assert result["nll_uniform_grid"]<28.
    assert len(result["raw_logit_metrics"]["calibration"])==15
    assert result["raw_logit_metrics"]["per_class"][1]["recall"] is None


def test_frozen_config_source_binding_and_equal_grid_budget():
    root=Path(__file__).resolve().parents[1]
    settings=json.loads((root/"configs/canonical_inferred_cohort64_pilot_v1.json").read_text())
    manifest_path=root/"data/ancient_inferred_cohort64_pilot_v1/manifest.json"
    if not manifest_path.exists(): pytest.skip("Frozen training/validation source unavailable")
    manifest=json.loads(manifest_path.read_text())
    verify_protocol(settings,manifest,manifest_path)
    changed={**settings,"windows":4}
    with pytest.raises(RuntimeError,match="dataset field"):
        verify_protocol(changed,manifest,manifest_path)
    assert len(settings["configurations"])*len(settings["training_seeds"])==24


def test_family_selection_requires_all_seeds_and_uses_capacity_lexicographic_tie():
    configurations=[{"id":name,"hidden_dim":12 if name=="large" else 8,"dropout":0.,"learning_rate":.001}
        for name in ("large","small_a","small_b")]
    settings={"configurations":configurations,"input_dim":48,"num_classes":4,"num_layers":2,"temporal_layers":2,
        "training_seeds":[101,202,303],"family_tie_tolerance_nats":.0001,
        "dataset":{"manifest_sha256":"b"*64,"source_artifact_sha256":{"data/synthetic_canonical_source":"d"*64}},
        "implementation_sha256":{"code":"c"*64}}
    import hashlib
    config_bytes=json.dumps(settings).encode()
    expected=hashlib.sha256(config_bytes).hexdigest()
    records=[{"configuration_id":c["id"],"configuration":c,"training_seed":seed,
        "validation_nll":1. if c["id"]=="large" else 1.00005,
        "parameters":sum(p.numel() for p in CanonicalTreeGNNGRU(48,4,c["hidden_dim"],2,2,c["dropout"]).parameters()),
        "protocol_sha256":expected,"dataset_manifest_sha256":"b"*64,
        "implementation_sha256":settings["implementation_sha256"],"source_artifact_sha256":settings["dataset"]["source_artifact_sha256"]}
        for c in configurations for seed in settings["training_seeds"]]
    assert select_family_configuration(records,settings,expected,config_bytes=config_bytes)["selected"]["configuration_id"]=="small_a"
    with pytest.raises(ValueError,match="every frozen"):
        select_family_configuration(records[:-1],settings,expected,config_bytes=config_bytes)
    for field,value,pattern in (("implementation_sha256",{"code":"e"*64},"implementation/dependency"),
        ("protocol_sha256","f"*64,"exact expected original config"),
        ("configuration",{**configurations[0],"hidden_dim":999},"configuration/capacity"),
        ("parameters",1,"configuration/capacity"),
        ("source_artifact_sha256",{"data/synthetic_canonical_source":"e"*64},"raw/prepared/metadata")):
        mismatch=[dict(row) for row in records];mismatch[0][field]=value
        with pytest.raises(ValueError,match=pattern):select_family_configuration(mismatch,settings,expected,config_bytes=config_bytes)


def test_actual_pinned_train_input_preprocessing_and_untrained_forward_only():
    root=Path(__file__).resolve().parents[1]
    manifest_path=root/"data/ancient_inferred_cohort64_pilot_v1/manifest.json"
    if not manifest_path.exists(): pytest.skip("Pinned inferred train source unavailable")
    import hashlib
    manifest=json.loads(manifest_path.read_text())
    record=next(r for r in manifest["replicates"] if r["split"]=="train" and r["replicate"]==0)
    path=Path(record["prepared_tensor_file"])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record["prepared_tensor_sha256"]
    bundle=torch.load(path,map_location="cpu",weights_only=True)
    from arggnn.data import GraphSequence
    seq=GraphSequence.from_dict(bundle["sequence"])
    prepared=prepare_canonical_parent(seq,manifest["protocol"]["reference_adapter"])
    assert prepared.sequence.x.shape[-1]==48 and prepared.sequence.num_queries==64
    torch.testing.assert_close(prepared.sequence.positions,seq.positions,rtol=0,atol=0)
    tile=canonical_fixed_tiles(prepared,8)[0]
    net=CanonicalTreeGNNGRU(hidden_dim=8).eval()
    output=net(tile.model_input)
    assert output.shape==(16,64,4) and torch.isfinite(output).all()
    # No raw target bundle, real labels, fitted optimizer or metric is used.


@pytest.mark.parametrize("value",["", ".", "..", "../escape", "/tmp/escape", "a/b", "a\\b",
    "a.b", "a b", "a\x00b", "а", "-leading", "a"*129,123,None])
def test_safe_filename_components_reject_adversarial_ids(value):
    with pytest.raises(ValueError,match="Unsafe"):
        safe_component(value,"configuration ID")


def test_unsafe_ids_fail_before_path_or_model_input_access(tmp_path):
    assert safe_component("canonical_gru_h32_drop00_lr02")=="canonical_gru_h32_drop00_lr02"
    assert safe_component("0")=="0" and safe_component("a"*128)=="a"*128
    with pytest.raises(ValueError,match="Unsafe configuration ID"):
        verify_protocol({"configurations":[{"id":"../escape"}]},{"replicates":[]},tmp_path/"missing.json")
    with pytest.raises(ValueError,match="Unsafe parent ID"):
        verify_protocol({"configurations":[]},{"replicates":[{"split":"train","replicate":"/tmp/escape"}]},tmp_path/"missing.json")
    with pytest.raises(ValueError,match="Unsafe parent ID"):
        bind_training_parent(None,None,{"replicate":"../escape"},{})
    with pytest.raises(ValueError,match="Unsafe configuration ID"):
        train_configuration({"id":"/tmp/escape"},None,None,None,None,tmp_path/"escaped.pt",None,None)
    assert not (tmp_path/"escaped.pt").exists()


@pytest.mark.parametrize("value",["/tmp/canonical_external_artifact","../../canonical_external_artifact"])
def test_external_absolute_and_traversal_sources_are_rejected(value):
    manifest={"replicates":[{"split":"train","replicate":0,"metadata":value,"metadata_sha256":"a"*64}]}
    with pytest.raises(ValueError,match="outside project"):
        source_artifacts(manifest)


@pytest.fixture
def frozen_snapshot_files(tmp_path,monkeypatch):
    import hashlib
    import arggnn.train_canonical_gru as runner
    root=Path(__file__).resolve().parents[1]
    settings=json.loads((root/"configs/canonical_inferred_cohort64_pilot_v1.json").read_bytes())
    dataset=tmp_path/"dataset"
    dataset.mkdir()
    code=tmp_path/"tiny_dependency.py"
    code.write_bytes(b"reviewed source\n")
    monkeypatch.setattr(runner,"implementation_paths",lambda:{"tiny_dependency.py":code})
    def confined_path(value):
        path=Path(value)
        path=(path if path.is_absolute() else tmp_path/path).resolve()
        if path!=tmp_path and tmp_path not in path.parents:
            raise ValueError("Output path is outside project")
        return path
    monkeypatch.setattr(runner,"project_path",confined_path)
    records=[]
    for split,count in (("train",16),("val",4)):
        for index in range(count):
            record={"split":split,"replicate":index}
            for field,hash_field in (("metadata","metadata_sha256"),("tensor_file","tensor_sha256"),
                ("prepared_tensor_file","prepared_tensor_sha256")):
                path=dataset/f"{split}_{index}_{field}"
                payload=f"{split} {index} {field}\n".encode()
                path.write_bytes(payload)
                record[field]=str(path)
                record[hash_field]=hashlib.sha256(payload).hexdigest()
            records.append(record)
    manifest={"protocol_sha256":settings["dataset"]["protocol_sha256"],"replicates":records,
        "protocol":{"benchmark":settings["benchmark"],"simulation":{"query_diploids":32},
            "reference_adapter":settings["reference_adapter"],"windows":8,"splits":settings["splits"]}}
    manifest_path=dataset/"manifest.json"
    manifest_bytes=(json.dumps(manifest,indent=2)+"\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    settings["dataset"]["directory"]=str(dataset)
    settings["dataset"]["manifest_sha256"]=hashlib.sha256(manifest_bytes).hexdigest()
    settings["implementation_sha256"]=runner.implementation_hashes()
    settings["dataset"]["source_artifact_sha256"]=runner.source_artifacts(manifest)
    config_path=tmp_path/"config.json"
    config_path.write_bytes((json.dumps(settings,indent=2)+"\n").encode())
    return config_path,manifest_path,code,Path(records[0]["tensor_file"])


def test_snapshot_copies_original_bytes_and_enforces_supplied_config_and_reviewed_code(frozen_snapshot_files):
    config_path,manifest_path,code,_=frozen_snapshot_files
    import hashlib
    config_bytes=config_path.read_bytes()
    manifest_bytes=manifest_path.read_bytes()
    digest=hashlib.sha256(config_bytes).hexdigest()
    snapshot=capture_run_snapshot(config_path,manifest_path,digest)
    assert snapshot.config_bytes==config_bytes and snapshot.manifest_bytes==manifest_bytes
    assert snapshot.config_sha256==digest
    snapshot.check(full_sources=True)
    with pytest.raises(RuntimeError,match="Supplied expected config SHA256"):
        capture_run_snapshot(config_path,manifest_path,"0"*64)
    with pytest.raises(ValueError,match="Invalid expected config"):
        capture_run_snapshot(config_path,manifest_path,"../escape")
    code.write_bytes(b"unreviewed source\n")
    with pytest.raises(RuntimeError,match="Reviewed implementation/dependency"):
        capture_run_snapshot(config_path,manifest_path)
    assert snapshot.config_bytes==config_bytes


@pytest.mark.parametrize("which,pattern",[(0,"config changed"),(1,"manifest changed"),
    (2,"implementation/dependency source changed"),(3,"source artifact changed")])
def test_run_snapshot_rejects_midrun_config_manifest_code_and_source_mutations(frozen_snapshot_files,which,pattern):
    snapshot=capture_run_snapshot(*frozen_snapshot_files[:2])
    path=frozen_snapshot_files[which]
    path.write_bytes(path.read_bytes()+b"mutation\n")
    with pytest.raises(RuntimeError,match=pattern): snapshot.check()
    with pytest.raises(RuntimeError,match=pattern): snapshot.check(full_sources=True)


def test_epoch_integrity_guard_detects_mutation_before_checkpoint(tmp_path,frozen_snapshot_files):
    settings=tiny_settings()
    prepared=prepare(genuine_sequence(dtype=torch.float32))
    labels=torch.zeros(3,2,dtype=torch.long)
    parents={split:[bind_training_parent(prepared,labels,{"replicate":0,"split":split,"num_classes":4},settings)] for split in ("train","val")}
    snapshot=capture_run_snapshot(*frozen_snapshot_files[:2])
    checkpoint=tmp_path/"must_not_save.pt"
    def mutate_after_epoch(item):
        path=frozen_snapshot_files[0]
        path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(RuntimeError,match="config changed"):
        train_configuration({"id":"synthetic_smoke","hidden_dim":8,"dropout":0.,"learning_rate":.02},
            settings,parents,torch.device("cpu"),101,checkpoint,snapshot.config_sha256,
            snapshot.manifest_sha256,mutate_after_epoch,snapshot.check)
    assert not checkpoint.exists()



def test_duplicate_parent_ids_are_rejected_before_input_cache_paths(tmp_path):
    with pytest.raises(ValueError,match="Duplicate split/parent IDs"):
        verify_protocol({"configurations":[]},{"replicates":[{"split":"train","replicate":0},
            {"split":"train","replicate":0}]},tmp_path/"missing.json")


def test_initial_and_boundary_source_hash_guards_reject_changed_bytes(frozen_snapshot_files,monkeypatch):
    import arggnn.train_canonical_gru as runner
    snapshot=capture_run_snapshot(*frozen_snapshot_files[:2])
    source=frozen_snapshot_files[3]
    source.write_bytes(b"changed source\n")
    with pytest.raises(RuntimeError,match="source artifact hash changed"):
        capture_run_snapshot(*frozen_snapshot_files[:2])
    # Full boundary hashing verifies bytes even if a filesystem cache reports
    # the old stat signature. Ordinary epoch checks use the actual signatures.
    monkeypatch.setattr(runner,"source_signature",lambda path:snapshot.source_signatures[str(path)])
    snapshot.check()
    with pytest.raises(RuntimeError,match="source artifact hash changed"):
        snapshot.check(full_sources=True)



@pytest.mark.parametrize("namespace",["confirmatory_v1","final_test","final-test","confirmation","test"])
def test_heldout_paths_are_rejected_before_any_control_or_source_file_read(tmp_path,monkeypatch,namespace):
    import arggnn.train_canonical_gru as runner
    forbidden=tmp_path/namespace/"manifest.json"
    opened=[]
    original=Path.read_bytes
    def spy(path):
        opened.append(path)
        if namespace in path.parts:raise AssertionError("Held-out file was opened")
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",spy)
    with pytest.raises(ValueError,match="Held-out paths"):
        capture_run_snapshot(tmp_path/"safe_config.json",forbidden)
    with pytest.raises(ValueError,match="Held-out paths"):
        capture_run_snapshot(forbidden,tmp_path/"safe_manifest.json")
    with pytest.raises(ValueError,match="Held-out paths"):
        source_artifacts({"replicates":[{"split":"train","replicate":0,"metadata":str(forbidden),"metadata_sha256":"a"*64}]})
    assert opened==[]


def test_audit_family_wrong_common_protocol_and_record_specs_are_rejected():
    root=Path(__file__).resolve().parents[1]
    config=root/"configs/canonical_inferred_cohort64_pilot_v1.json"
    import hashlib
    expected=hashlib.sha256(config.read_bytes()).hexdigest()
    settings=json.loads(config.read_bytes())
    rows=[{"configuration_id":c["id"],"training_seed":seed,"validation_nll":1.,"parameters":1,
        "protocol_sha256":"f"*64,"dataset_manifest_sha256":settings["dataset"]["manifest_sha256"],
        "implementation_sha256":settings["implementation_sha256"],"source_artifact_sha256":settings["dataset"]["source_artifact_sha256"],
        "configuration":{"id":c["id"],"hidden_dim":999,"dropout":.5,"learning_rate":1.}}
        for c in settings["configurations"] for seed in settings["training_seeds"]]
    with pytest.raises(ValueError,match="exact expected original config"):
        select_family_configuration(rows,settings,expected,config_bytes=config.read_bytes())
    for row in rows:row["protocol_sha256"]=expected
    with pytest.raises(ValueError,match="configuration/capacity"):
        select_family_configuration(rows,settings,expected,config_bytes=config.read_bytes())
    with pytest.raises(TypeError):select_family_configuration(rows,settings)


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_main_rejects_unadmitted_inputs_before_cuda_queries(tmp_path,monkeypatch,device):
    import sys
    import arggnn.train_canonical_gru as runner
    touched=[]
    def forbidden(*args,**kwargs):
        touched.append("cuda")
        raise AssertionError("CUDA queried before input admission")
    monkeypatch.setattr(runner.torch.cuda,"is_available",forbidden)
    monkeypatch.setattr(runner.torch.cuda,"device_count",forbidden)
    monkeypatch.setattr(runner,"set_research_title",lambda:None)
    monkeypatch.setattr(runner,"require_storage",lambda **kwargs:None)
    monkeypatch.setattr(sys,"argv",["train_canonical_gru",
        "--config",str(tmp_path/"confirmatory"/"config.json"),
        "--data",str(tmp_path/"data"),"--output",str(tmp_path/"output"),
        "--device",device])
    with pytest.raises(ValueError,match="Held-out paths"):runner.main()
    assert touched==[]


def test_family_selection_rejects_coherent_settings_drift_under_unchanged_original_sha():
    import copy,hashlib
    config=Path(__file__).resolve().parents[1]/"configs/canonical_inferred_cohort64_pilot_v1.json"
    original=config.read_bytes()
    expected=hashlib.sha256(original).hexdigest()
    settings=json.loads(original)
    changed=copy.deepcopy(settings)
    changed["configurations"][0]["hidden_dim"]=16
    rows=[{"configuration_id":c["id"],"configuration":c,"training_seed":seed,
        "validation_nll":1.,
        "parameters":sum(p.numel() for p in CanonicalTreeGNNGRU(48,4,c["hidden_dim"],2,2,c["dropout"]).parameters()),
        "protocol_sha256":expected,"dataset_manifest_sha256":changed["dataset"]["manifest_sha256"],
        "implementation_sha256":changed["implementation_sha256"],"source_artifact_sha256":changed["dataset"]["source_artifact_sha256"]}
        for c in changed["configurations"] for seed in changed["training_seeds"]]
    with pytest.raises(ValueError,match="settings differ from captured original"):
        select_family_configuration(rows,changed,expected,config_bytes=original)
    with pytest.raises(ValueError,match="original config bytes SHA256"):
        select_family_configuration(rows,changed,expected,config_bytes=json.dumps(changed).encode())
    with pytest.raises(TypeError):select_family_configuration(rows,changed,expected)
