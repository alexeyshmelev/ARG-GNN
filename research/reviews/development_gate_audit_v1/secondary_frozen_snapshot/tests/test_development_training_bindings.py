"""CPU synthetic protocol/binding/readout tests. No simulation or native jobs."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import pytest
import torch
from arggnn.resources import ROOT
from arggnn.development_training_bindings import (
    make_configs,checked_protocol,verify_completion,sha,check_live,no_test_path)
from arggnn.train_anchor_temporal_development import factory,captured_protocol
from arggnn.train_anchor_temporal import evaluate_temporal
from test_anchor_temporal_baselines import prepared,readouts


def completed_facts(tmp_path):
    ids=[{"split":s,"replicate":i} for s,n in [("train",64),("val",16)] for i in range(n)]
    protocol={"splits":{"train":64,"val":16,"test":0},"master_seed":20261101}
    manifests={
        "oracle":{"protocol":protocol,"replicates":ids},
        "inferred":{"protocol":{**protocol,"inference":{"master_seed":20261102}},"replicates":ids},
        "boundary":{"status":"complete","preparation_sha256":"b"*64},
        "age":{"status":"complete","preparation_sha256":"c"*64}}
    facts={}
    def save(stage):
        path=tmp_path/(stage+"_manifest.json")
        path.write_text(json.dumps(manifests[stage]))
        facts[stage]={"path":str(path),"sha256":sha(path)}
        if "preparation_sha256" in manifests[stage]:
            facts[stage]["preparation_sha256"]=manifests[stage]["preparation_sha256"]
    save("oracle")
    manifests["inferred"]["source_parent"]={"manifest_sha256":facts["oracle"]["sha256"]}
    save("inferred")
    manifests["boundary"]["source"]={"manifest_sha256":facts["inferred"]["sha256"]}
    save("boundary")
    manifests["age"].update(boundary_manifest_sha256=facts["boundary"]["sha256"],
        source_manifest_sha256=facts["inferred"]["sha256"],boundary_preparation_sha256="b"*64)
    save("age")
    completion={"all_stages_complete":True,"train_parents":64,"validation_parents":16,
        "test_parents":0,"inferred_boundary_and_age_tiles":640,
        "training_or_confirmatory_selection_frozen":False,
        "development_execution_freeze_file_sha256":"d"*64,"stage_manifests":facts}
    path=tmp_path/"completion.json"
    path.write_text(json.dumps(completion))
    return completion,path


def configs(tmp_path):
    completion,path=completed_facts(tmp_path)
    return make_configs(completion,path,sha(path))


def write_config(tmp_path,settings):
    path=tmp_path/"config.json"
    path.write_text(json.dumps(settings))
    return path


def test_exact_all_baseline_grids_and_root_candidate_contract(tmp_path):
    from arggnn.train_anchor_development import checked_development_protocol
    values=configs(tmp_path)
    assert set(values)=={"candidate","gru","transformer","reference_smoother"}
    for family,settings in values.items():
        path=write_config(tmp_path,settings)
        if family=="candidate":assert checked_development_protocol(path)==settings
        else:assert checked_protocol(path,sha(path))==settings


@pytest.mark.parametrize("fault",["test","pilot","seed","namespace","grid","width","dropout","lr","budget","hash","id","nll"])
def test_baseline_grid_drift_rejected(tmp_path,fault):
    p=configs(tmp_path)["gru"];name=p["models"][0]
    if fault=="test":p["splits"]["test"]=1
    elif fault=="pilot":p["splits"]["train"]=16
    elif fault=="seed":p["training_seeds"].append(404)
    elif fault=="namespace":p["development_namespace"]="pilot"
    elif fault=="grid":
        p["model_specs"][p["models"][1]]=deepcopy(p["model_specs"][name])
        p["model_learning_rates"][p["models"][1]]=p["model_learning_rates"][name]
    elif fault=="width":p["model_specs"][name]["hidden_dim"]=16
    elif fault=="dropout":p["model_specs"][name]["dropout"]=.2
    elif fault=="lr":p["model_learning_rates"][name]=.03
    elif fault=="budget":p["epochs"]=49
    elif fault=="hash":p["completion_receipt_sha256"]="0"*64
    elif fault=="id":
        p["models"][0]="../unsafe"
        p["model_specs"]["../unsafe"]=p["model_specs"].pop(name)
        p["model_learning_rates"]["../unsafe"]=p["model_learning_rates"].pop(name)
    else:p["nll_policy"]="clipped probabilities"
    path=write_config(tmp_path,p)
    with pytest.raises(RuntimeError):checked_protocol(path,sha(path))


def test_exact_config_sha_and_mutation_after_capture_are_rejected(tmp_path,monkeypatch):
    import arggnn.train_anchor_temporal_development as runner
    settings=configs(tmp_path)["gru"];path=write_config(tmp_path,settings);h=sha(path)
    content,value,loaded=captured_protocol(path,h)
    assert content==path.read_bytes() and value==h and loaded==settings
    with pytest.raises(RuntimeError,match="exact supplied"):captured_protocol(path,"f"*64)
    original=runner.checked_protocol
    def mutate(p,s):
        result=original(p,s);p.write_text("{}")
        return result
    monkeypatch.setattr(runner,"checked_protocol",mutate)
    with pytest.raises(RuntimeError,match="changed during capture"):captured_protocol(path,h)


def test_live_config_edit_rejected_before_cache_access(tmp_path):
    settings=configs(tmp_path)["gru"];path=write_config(tmp_path,settings);h=sha(path)
    path.write_text("{}")
    with pytest.raises(RuntimeError,match="SHA256 mismatch"):check_live(settings,path,h)


def test_completion_must_be_successful_before_cache_loading(tmp_path,monkeypatch):
    import arggnn.boundary_dataset as boundary
    completion,path=completed_facts(tmp_path)
    completion["all_stages_complete"]=False;path.write_text(json.dumps(completion))
    monkeypatch.setattr(boundary,"load_boundary_dataset",lambda *_a,**_k:pytest.fail("Cache read before valid completion"))
    with pytest.raises(RuntimeError,match="completion required"):verify_completion(path,sha(path))


def test_completion_facts_and_full_loader_bindings(tmp_path,monkeypatch):
    import arggnn.boundary_dataset as boundary
    import arggnn.age_dataset as age
    completion,path=completed_facts(tmp_path)
    calls=[]
    class Dataset:
        records=[None]*640
        def cases(self,split):
            return [SimpleNamespace(metadata={"split":split,"master_seed":20261101,"num_classes":4},
                query_node_ids=torch.arange(64)) for _ in range(64 if split=="train" else 16)]
    def loaded(*args,**kw):
        calls.append(("boundary",kw));return Dataset()
    def ages(*args,**kw):
        calls.append(("age",kw));return {i:None for i in range(640)}
    monkeypatch.setattr(boundary,"load_boundary_dataset",loaded)
    monkeypatch.setattr(age,"load_age_caches",ages)
    # This isolated API-call test mocks source admission; adversarial tests below
    # exercise the real source catalog and production freeze/registry checks.
    import arggnn.development_training_bindings as bindings
    monkeypatch.setattr(bindings,"capture_completion_sources",
        lambda *_a,**_k:(json.loads(path.read_text()),SimpleNamespace(check=lambda **_:None)))
    verify_completion(path,sha(path))
    assert [c[0] for c in calls]==["boundary","age"]
    assert calls[0][1]["expected_source_manifest_sha256"]==completion["stage_manifests"]["inferred"]["sha256"]
    completion["stage_manifests"]["age"]["preparation_sha256"]="e"*64
    path.write_text(json.dumps(completion))
    with pytest.raises(RuntimeError,match="facts differ"):verify_completion(path,sha(path))


@pytest.mark.parametrize("family",["gru","transformer","reference_smoother"])
def test_new_factory_independent_readouts_and_label_counterfactual(tmp_path,family):
    torch.set_num_threads(1)
    pack=prepared(anchors=32)
    settings=configs(tmp_path)[family];name=settings["models"][0]
    model=factory(name,settings).double().eval()
    output=model(pack)
    assert output.shape==(7,2,4)
    extra=torch.tensor([.12],dtype=torch.float64)
    larger=readouts(pack,torch.cat((pack.bank.positions,extra)))
    with torch.no_grad():after=model(larger)[:7]
    torch.testing.assert_close(after,output,rtol=1e-10,atol=1e-10)
    tile=SimpleNamespace(bank=pack.bank,parent_position_indices=torch.arange(7))
    case=SimpleNamespace(tiles=[tile],positions=pack.bank.positions,query_node_ids=pack.bank.query_node_ids,
        labels=torch.zeros(7,2,dtype=torch.long),metadata={"replicate":0},parent_id="synthetic-parent")
    first,z=evaluate_temporal(model,[case],{id(tile):pack})
    case.labels=torch.ones_like(case.labels)
    second,z2=evaluate_temporal(model,[case],{id(tile):pack})
    torch.testing.assert_close(z[0],z2[0],rtol=0,atol=0)
    assert first[0]["nll"]!=second[0]["nll"]
    if family=="reference_smoother":assert sum(p.numel() for p in model.parameters())==0
    else:
        output.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_final_path_refused_before_file_read():
    with pytest.raises(RuntimeError,match="Final held-out"):no_test_path(ROOT/"data/ancestry_confirmatory_v1/manifest.json")


def test_binder_default_is_validation_only(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("binder",ROOT/"scripts/bind_development_training.py")
    binder=importlib.util.module_from_spec(spec);spec.loader.exec_module(binder)
    completion,path=completed_facts(tmp_path)
    monkeypatch.setattr(binder,"verify_completion",lambda *_:(completion,None,None))
    target=tmp_path/"do_not_write"
    binder.main(["--completion",str(path),"--output-directory",str(target)])
    assert not target.exists()
    with pytest.raises(RuntimeError,match="explicit reviewed"):
        binder.main(["--completion",str(path),"--write-configs","--output-directory",str(target)])

def test_family_templates_do_not_share_mutable_identity_constants(tmp_path):
    from arggnn.development_training_bindings import SPLITS,SEEDS
    values=configs(tmp_path)
    values["gru"]["splits"]["test"]=1
    values["gru"]["training_seeds"].append(404)
    assert SPLITS=={"train":64,"val":16,"test":0} and SEEDS==[101,202,303]
    assert all(v["splits"]==SPLITS and v["training_seeds"]==SEEDS
               for k,v in values.items() if k!="gru")

def test_config_cache_facts_must_match_its_completion_receipt(tmp_path):
    from arggnn.development_training_bindings import check_completion_settings
    completion,path=completed_facts(tmp_path)
    settings=make_configs(completion,path,sha(path))["gru"]
    check_completion_settings(settings,completion)
    settings["source_dataset_manifest_sha256"]="f"*64
    with pytest.raises(RuntimeError,match="completed preparation receipt"):
        check_completion_settings(settings,completion)


def linked_sources(tmp_path,monkeypatch):
    """Self-consistent byte artifacts with mocked seed admission, no simulation."""
    import arggnn.development_training_bindings as b
    dirs={s:tmp_path/s for s in ("oracle","inferred","boundary","age")}
    for p in dirs.values():p.mkdir()
    rows=[{"split":s,"replicate":i,"seeds":{"ancestry":10000+3*j,"mutations":10001+3*j},
           "inference_seed":10002+3*j}
        for j,(s,i) in enumerate(( (s,i) for s,n in (("train",64),("val",16)) for i in range(n)))]
    monkeypatch.setattr(b,"expected_parent_registry",lambda:deepcopy(rows))
    def file(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value,(dict,list)) else value)
        return str(path),sha(path)
    op={"splits":{"train":64,"val":16,"test":0},"master_seed":20261101}
    ip={**op,"inference":{"master_seed":20261102}}
    oc,och=file(dirs["oracle"]/"protocol.json",op)
    ic,ich=file(dirs["inferred"]/"protocol.json",ip)
    oracle=[];inferred=[];one={}
    for row in rows:
        ident=(row["split"],row["replicate"])
        od=dirs["oracle"]/row["split"]/f'rep{row["replicate"]:03d}'
        raw=dirs["inferred"]/row["split"]/f'rep{row["replicate"]:03d}'/"inference/dataset"
        meta={**row,"master_seed":20261101}
        originals={}
        for directory,record in ((od,originals),(raw,{})):
            m,mh=file(directory/"metadata.json",meta)
            t,th=file(directory/"tensors.pt.gz","raw")
            p,ph=file(directory/"prepared.pt","prepared")
            rec={"directory":str(directory),"split":row["split"],"replicate":row["replicate"],
                "seeds":row["seeds"],"metadata":m,"metadata_sha256":mh,
                "tensor_file":t,"tensor_sha256":th,"prepared_tensor_file":p,"prepared_tensor_sha256":ph}
            if directory==od:
                originals.update(rec);oracle.append(rec)
                _,truth=file(od/"truth.trees.gz","truth")
                _,observed=file(od/"observed.trees.gz","observed")
            else:
                rec["source_parent"]={**originals,"truth_archive_sha256":truth,
                    "clean_observed_archive_sha256":observed}
                rec["inference_seed"]=row["inference_seed"]
                native=raw.parent
                _,rec["native_inferred_tree_sha256"]=file(native/"inferred.trees","native")
                _,rec["execution_sha256"]=file(native/"execution.json",{})
                obs,oh=file(raw/"observed.trees.gz","inferred observed")
                rec.update(observed_tree_file=obs,observed_tree_sha256=oh)
                genotype,gh=file(native.parent/"inputs/input.haps","genotype")
                _,rec["preparation_sha256"]=file(native.parent/"inputs/preparation.json",
                    {"paths":{"haps":genotype},"files_sha256":{"haps":gh}})
                inferred.append(rec)
        if not one:one={"oracle_raw":originals["tensor_file"],"inferred_raw":rec["tensor_file"],
                       "genotype":genotype,"oracle_metadata":originals["metadata"]}
    om={"protocol":op,"protocol_sha256":och,"replicates":oracle}
    ompath,omh=file(dirs["oracle"]/"manifest.json",om)
    im={"protocol":ip,"protocol_sha256":ich,"replicates":inferred,
        "source_parent":{"manifest_sha256":omh}}
    impath,imh=file(dirs["inferred"]/"manifest.json",im)
    supervision,superh=file(dirs["boundary"]/"train/rep000/supervision.pt","labels")
    bank,bankh=file(dirs["boundary"]/"train/rep000/window000/bank.pt","bank")
    bm={"status":"complete","preparation_sha256":"b"*64,
        "source":{"manifest_sha256":imh},
        "parents":[{"supervision_file":"train/rep000/supervision.pt","supervision_sha256":superh}],
        "records":[{"bank_file":"train/rep000/window000/bank.pt","bank_sha256":bankh}]}
    bmpath,bmh=file(dirs["boundary"]/"manifest.json",bm)
    age,ageh=file(dirs["age"]/"train/rep000/window000/age_operators.pt","age")
    am={"status":"complete","preparation_sha256":"c"*64,"boundary_manifest_sha256":bmh,
        "boundary_preparation_sha256":"b"*64,"source_manifest_sha256":imh,
        "records":[{"operator_file":"train/rep000/window000/age_operators.pt","operator_sha256":ageh}]}
    ampath,amh=file(dirs["age"]/"manifest.json",am)
    plan,planh=file(tmp_path/"plan.json",{})
    recipe,recipeh=file(tmp_path/"recipe.py","recipe")
    frozen={"kind":"fresh_development_preparation_execution_freeze","confirmatory_access":False,
        "not_confirmatory_test_freeze":True,"parent_registry":rows,
        "plan":{"expected_splits":op["splits"],"expected_simulation_master_seed":20261101,
                "relate_master_seed":20261102,"oracle_config":oc,"paths":{s:str(p) for s,p in dirs.items()}},
        "config_files":{oc:och,plan:planh},"source_code_sha256":{recipe:recipeh}}
    fp,fh=file(tmp_path/"freeze.json",frozen)
    facts={s:{"path":path,"sha256":h} for s,path,h in
           (("oracle",ompath,omh),("inferred",impath,imh),("boundary",bmpath,bmh),("age",ampath,amh))}
    facts["inferred"]["protocol_path"]=ic
    facts["boundary"]["preparation_sha256"]="b"*64
    facts["age"]["preparation_sha256"]="c"*64
    completion={"all_stages_complete":True,"train_parents":64,"validation_parents":16,
        "test_parents":0,"inferred_boundary_and_age_tiles":640,"training_or_confirmatory_selection_frozen":False,
        "development_execution_freeze_file_sha256":fh,"development_execution_freeze_path":fp,
        "stage_manifests":facts}
    cp,ch=file(tmp_path/"completion.json",completion)
    one.update(freeze=fp,oracle_manifest=ompath,inferred_manifest=impath,boundary_bank=bank,
               age_cache=age,supervision=supervision,completion=cp,recipe=recipe)
    return completion,Path(cp),one


@pytest.mark.parametrize("artifact",["oracle_manifest","inferred_manifest","oracle_raw","inferred_raw",
    "oracle_metadata","genotype","boundary_bank","age_cache","supervision","freeze","recipe"])
def test_source_snapshot_rejects_every_bound_artifact_change(tmp_path,monkeypatch,artifact):
    from arggnn.development_training_bindings import capture_completion_sources
    completion,path,artifacts=linked_sources(tmp_path,monkeypatch)
    _,lock=capture_completion_sources(path,sha(path))
    target=Path(artifacts[artifact])
    target.write_bytes(target.read_bytes()+b" ")
    with pytest.raises(RuntimeError,match="artifact changed"):lock.check(full=False)
    # Full seals independently verify expected bytes even if a stat cache is refreshed.
    from arggnn.development_training_bindings import artifact_signature
    lock.signatures[str(target)]=artifact_signature(target)
    with pytest.raises(RuntimeError,match="SHA256 changed"):lock.check(full=True)


def test_actual_freeze_registry_rejected_even_when_rehashed(tmp_path,monkeypatch):
    from arggnn.development_training_bindings import capture_completion_sources
    completion,path,artifacts=linked_sources(tmp_path,monkeypatch)
    fp=Path(artifacts["freeze"]);f=json.loads(fp.read_text())
    f["parent_registry"][0]["inference_seed"]+=1
    fp.write_text(json.dumps(f))
    completion["development_execution_freeze_file_sha256"]=sha(fp)
    path.write_text(json.dumps(completion))
    with pytest.raises(RuntimeError,match="freeze/parent registry"):
        capture_completion_sources(path,sha(path))


def test_check_live_original_source_mutation_now_rejected(tmp_path,monkeypatch):
    completion,path,artifacts=linked_sources(tmp_path,monkeypatch)
    settings=make_configs(completion,path,sha(path))["gru"]
    config=write_config(tmp_path,settings)
    check_live(settings,config,sha(config))
    Path(artifacts["inferred_manifest"]).write_text('{"mutated_source":true}')
    with pytest.raises(RuntimeError,match="SHA256 mismatch"):check_live(settings,config,sha(config))


def test_heldout_guard_precedes_config_receipt_and_source_reads(tmp_path,monkeypatch):
    from arggnn.development_training_bindings import read_bound,capture_completion_sources
    from arggnn.train_anchor_temporal_development import captured_protocol
    target=ROOT/"data/final_test_probe/manifest.json"
    def forbidden(_):pytest.fail("Held-out bytes opened")
    monkeypatch.setattr(Path,"read_bytes",forbidden)
    for call in (lambda:read_bound(target,"a"*64),
                 lambda:capture_completion_sources(target,"a"*64),
                 lambda:captured_protocol(target,"a"*64)):
        with pytest.raises(RuntimeError,match="Final held-out"):call()


def test_epoch_stat_checks_do_not_hash_or_reload_tensors(tmp_path,monkeypatch):
    import arggnn.development_training_bindings as b
    _,path,_=linked_sources(tmp_path,monkeypatch)
    _,lock=b.capture_completion_sources(path,sha(path))
    monkeypatch.setattr(b,"sha",lambda *_:pytest.fail("Epoch stat check hashed source bytes"))
    lock.check(full=False)


@pytest.mark.parametrize("kind",["missing","wrong_sha","incomplete"])
def test_startup_completion_absence_or_mismatch_refused(tmp_path,monkeypatch,kind):
    from arggnn.development_training_bindings import admit_run,snapshot
    completion,path=completed_facts(tmp_path)
    settings=make_configs(completion,path,sha(path))["gru"]
    config=write_config(tmp_path,settings);config_sha=sha(config)
    if kind=="missing":path.unlink()
    elif kind=="wrong_sha":path.write_text("{}")
    else:
        completion["all_stages_complete"]=False
        path.write_text(json.dumps(completion))
        settings["completion_receipt_sha256"]=sha(path)
        config=write_config(tmp_path,settings);config_sha=sha(config)
    with pytest.raises(RuntimeError):
        admit_run(settings,config,config_sha,imported_pins=snapshot())


def test_wrong_frozen_stage_path_refused_before_manifest_open(tmp_path,monkeypatch):
    from arggnn.development_training_bindings import capture_completion_sources
    completion,path,artifacts=linked_sources(tmp_path,monkeypatch)
    # Non-token path also must match the actual freeze before it can be read.
    target=tmp_path/"other_cohort"/"manifest.json"
    completion["stage_manifests"]["inferred"]["path"]=str(target)
    path.write_text(json.dumps(completion))
    original=Path.read_bytes
    def guarded(self):
        if self==target:pytest.fail("Unexpected source manifest was opened")
        return original(self)
    monkeypatch.setattr(Path,"read_bytes",guarded)
    with pytest.raises(RuntimeError,match="Stage directory"):
        capture_completion_sources(path,sha(path))


def test_stat_epoch_guard_catches_same_size_content_and_restored_mtime(tmp_path,monkeypatch):
    import os
    from arggnn.development_training_bindings import capture_completion_sources
    _,path,artifacts=linked_sources(tmp_path,monkeypatch)
    _,lock=capture_completion_sources(path,sha(path))
    target=Path(artifacts["boundary_bank"]);before=target.stat()
    target.write_bytes(b"xxxx")  # Same byte count as bank.
    os.utime(target,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(RuntimeError,match="artifact changed"):lock.check(full=False)
