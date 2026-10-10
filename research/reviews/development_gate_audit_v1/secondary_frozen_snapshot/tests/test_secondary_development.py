"""Conditional secondary CPU fixtures. Cohort and reviewed-code contracts are explicit mocks."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from arggnn import secondary_control_bindings as secondary
from arggnn import train_secondary_development as runner
from arggnn import development_selection as main
from arggnn import development_selection_bindings as combined
from arggnn.development_execution import configure_deterministic_development,DETERMINISTIC_POLICY
from test_development_selection import study as original_study,write_json,write_tensor
from test_anchor_temporal_baselines import prepared,readouts


def proposal_ref(root):return write_json(root,secondary.PROPOSAL,secondary.proposal_values())


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    # Expand the existing handmade six-family artifact fixture into ten actual IDs.
    root=tmp_path_factory.mktemp("secondary_evidence")
    base=original_study.__wrapped__(root)
    completion=json.loads((root/"completion.json").read_text())
    completion["development_execution_freeze_path"]=str(root/"freeze.json")
    for fact in completion["stage_manifests"].values():fact["path"]=str(root/fact["path"])
    for stage,key in (("boundary","boundary_preparation_sha256"),("age","age_preparation_sha256")):
        completion["stage_manifests"][stage]["preparation_sha256"]=base["plan"]["source_binding"][key]
    completed=write_json(root,"completion.json",completion)
    base["plan"]["source_artifacts"]["completion"]=completed
    base["plan"]["source_binding"]["completion_receipt_sha256"]=completed["sha256"]
    old_configs=deepcopy(base["configs"])
    for value in old_configs.values():value["completion_receipt_sha256"]=completed["sha256"]
    templates={(ref["family_id"],json.loads((root/ref["path"]).read_text())["training_seed"]):json.loads((root/ref["path"]).read_text()) for ref in base["refs"]}
    source={"candidate":"candidate","gru":"gru","transformer":"transformer","reference_smoother":"smoother",
        "canonical_gru":"canonical","argmix_full_tree":"argmix","allrecord_gru":"gru","allrecord_transformer":"transformer",
        "gst_ca_span":"gru","graphssm_zoh":"gru"}
    configs={};families={};contracts={};refs=[]
    q=torch.tensor([20,10]);x=torch.tensor([.25,.75],dtype=torch.float64)
    plan=deepcopy(base["plan"])
    for parent in plan["validation_parents"]:parent["query_ids_sha256"]=main.tensor_fingerprint(q)
    for family in combined.FAMILIES:
        old=source[family];config=deepcopy(old_configs[old]);original_names=config["models"]
        names={name:family+"_"+str(i) for i,name in enumerate(original_names)}
        config.update(family=family,input_scope="Synthetic observed "+family,cross_tree_aggregation="Synthetic "+family,
            **secondary.FIXED)
        config[combined.POLICY_FIELDS[family]]=deepcopy(DETERMINISTIC_POLICY)
        config["models"]=[names[name] for name in original_names]
        config["model_specs"]={names[name]:value for name,value in config["model_specs"].items()}
        config["model_learning_rates"]={names[name]:value for name,value in config["model_learning_rates"].items()}
        ref=write_json(root,"main_configs/"+family+".json",config);configs[family]=config
        role=combined.ROLES[family]
        families[family]={"role":role,"grid_kind":"candidate" if family=="candidate" else "bandwidth_pseudocount" if family=="reference_smoother" else "width_dropout_lr",
            "protocol":ref,"expected_implementation_sha256":config["bound_implementation_sha256"],
            "input_scope":config["input_scope"],"cross_tree_aggregation":config["cross_tree_aggregation"]}
        contracts[family]={"input_scope":config["input_scope"],"cross_tree_aggregation":config["cross_tree_aggregation"],"pins":config["bound_implementation_sha256"]}
        for seed in (101,202,303):
            run=deepcopy(templates[(old,seed)]);run["family"]=family;run["protocol_sha256"]=ref["sha256"];run["completion_receipt_sha256"]=completed["sha256"]
            models={}
            for old_name,body in run["models"].items():
                body=deepcopy(body);name=names[old_name]
                prediction=torch.load(root/body["validation_logits"],weights_only=True)
                prediction.update(protocol_sha256=ref["sha256"],completion_receipt_sha256=completed["sha256"],query_node_ids=[q.clone() for _ in range(16)])
                pred=write_tensor(root,f"main_evidence/{family}/{name}_s{seed}_logits.pt",prediction)
                if body["checkpoint"] is not None:
                    checkpoint=torch.load(root/body["checkpoint"],weights_only=True);checkpoint["protocol_sha256"]=ref["sha256"];checkpoint["completion_receipt_sha256"]=completed["sha256"]
                    check=write_tensor(root,f"main_evidence/{family}/{name}_s{seed}_weights.pt",checkpoint)
                    body.update(checkpoint=check["path"],checkpoint_sha256=check["sha256"])
                body.update(validation_logits=pred["path"],validation_logits_sha256=pred["sha256"],ending_config_sha256=ref["sha256"])
                models[name]=body
            run["models"]=models
            refs.append({"family_id":family,**write_json(root,f"main_runs/{family}_s{seed}.json",run)})
    plan["families"]=families;plan["required_baseline_families"]=list(combined.FAMILIES[1:])
    plan_ref=write_json(root,"main_plan.json",plan)
    receipt=main.synthesize_development_selection(plan_ref["path"],plan_ref["sha256"],run_receipts=refs,project_root=root)
    receipt_ref=write_json(root,"main_selection.json",receipt.payload)
    producer=root/"src/arggnn/producer.py";producer.parent.mkdir(parents=True)
    producer.write_bytes((root/"producer.py").read_bytes())
    script=root/"scripts/bind_secondary_development.py";script.parent.mkdir()
    script.write_bytes((secondary.ROOT/"scripts/bind_secondary_development.py").read_bytes())
    raw=root/"observed_source.bin";raw.write_bytes(b"Explicit synthetic observed-source artifact")
    pack=readouts(prepared(anchors=32),x)
    assert torch.equal(pack.bank.query_node_ids,q)
    def case(parent_id,index):
        labels=torch.zeros(2,2,dtype=torch.long)
        tile=SimpleNamespace(bank=pack.bank,labels=labels,parent_position_indices=torch.arange(2),provenance={"tile_id":parent_id+":window0"})
        return SimpleNamespace(parent_id=parent_id,labels=labels,positions=x,query_node_ids=q,tiles=[tile],metadata={"replicate":index})
    val=[case(parent["parent_id"],i) for i,parent in enumerate(plan["validation_parents"])]
    train=[case(plan["source_binding"]["source_dataset_manifest_sha256"]+":train:0",0)]
    dataset=SimpleNamespace(cases=lambda split:val if split=="val" else train)
    return {"root":root,"plan":plan,"plan_ref":plan_ref,"receipt":json.loads((root/receipt_ref["path"]).read_text()),"receipt_ref":receipt_ref,
        "contracts":contracts,"producer":producer,"raw":raw,"dataset":dataset,"cases":{"train":train,"val":val},"pack":pack}


@pytest.fixture
def admitted(evidence,monkeypatch,request):
    root=evidence["root"]
    pins={"producer.py":secondary.common.sha(evidence["producer"])}
    monkeypatch.setattr(secondary,"implementation_hashes",lambda:dict(pins))
    monkeypatch.setattr(combined,"_family_contract",lambda family:deepcopy(evidence["contracts"][family]))
    # Complete producer admission is explicitly mocked for handmade main files.
    def producer_admission_mock(family,settings,**context):
        assert context["source_snapshot"].hashes
        assert context["completion_sha256"]==evidence["plan"]["source_binding"]["completion_receipt_sha256"]
        assert settings["family"]==family
    monkeypatch.setattr(combined,"validate_producer_settings",producer_admission_mock,raising=False)
    snapshot=secondary.common.SourceSnapshot({str(evidence["raw"]):secondary.common.sha(evidence["raw"])})
    completion=json.loads((root/evidence["plan"]["source_artifacts"]["completion"]["path"]).read_text())
    def genuine_admission_mock(*args,**kwargs):
        assert kwargs=={"with_snapshot":True}
        return deepcopy(completion),evidence["dataset"],{},snapshot
    monkeypatch.setattr(secondary.common,"verify_completion",genuine_admission_mock)
    directory=root/"case_runs"/request.node.name
    directory.mkdir(parents=True)
    proposal=proposal_ref(root)
    return {**evidence,"proposal_ref":proposal,"directory":directory,"pins":pins,"snapshot":snapshot}


def bind(study,*,receipt_ref=None,output=None):
    return secondary.bind_secondary_development(study["proposal_ref"],study["plan_ref"],receipt_ref or study["receipt_ref"],output_path=output,project_root=study["root"])


@pytest.mark.parametrize("geometry,depth,third",[("plain",1,None),("plain",2,"noncontextual"),("age",1,"matched_scalar_age_full"),("age",2,"matched_scalar_age_full")])
def test_selected_geometry_depth_conditional_controls(geometry,depth,third):
    controls=secondary.controls_for({"geometry":geometry,"kernel_depth":depth,"coupling":"full","rate_parameterization":"log"})
    assert list(controls)[:2]==["gain_matched_diagonal","static_local"]
    assert list(controls)[2:]==([] if third is None else [third])
    assert all(spec["kernel_depth"]==depth for spec in controls.values())
    assert len(controls)*2*3<=18<=72
    if third=="matched_scalar_age_full":assert controls[third]["geometry"]=="matched_scalar_age" and controls[third]["coupling"]=="full"


def test_complete_main_evidence_replay_and_conditional_binding(admitted):
    result=bind(admitted)
    assert result.settings["controls"]==["gain_matched_diagonal","static_local"]
    assert result.settings["optimizer_starts"]==12
    assert result.settings["selected_candidate_spec"]["geometry"]=="plain"
    assert result.settings["primary_strongest_baseline"]==admitted["receipt"]["strongest_required_baseline"]
    assert result.settings["selected_candidate_checkpoints"][0]["checkpoint_sha256"]
    assert result.sha256==hashlib.sha256(result.content).hexdigest()
    assert result.settings["final_test_access"] is False


@pytest.mark.parametrize("field",["matrix_resolved","baseline_selection_eligible","candidate_choice","strongest_required_baseline","plan_sha256","cells"])
def test_incomplete_or_forged_main_receipt_refused(admitted,field):
    value=deepcopy(admitted["receipt"])
    value[field]=False if field in ("matrix_resolved","baseline_selection_eligible") else None if field.endswith("choice") or field=="strongest_required_baseline" else "f"*64 if field=="plan_sha256" else value[field][:-1]
    ref=write_json(admitted["root"],str((admitted["directory"]/"invalid_main.json").relative_to(admitted["root"])),value)
    with pytest.raises(ValueError):bind(admitted,receipt_ref=ref,output=admitted["directory"]/"config.json")
    assert not (admitted["directory"]/"config.json").exists()


def test_forged_selected_score_cannot_replay(admitted):
    value=deepcopy(admitted["receipt"]);value["candidate_choice"]["mean_three_seed_nll"]-=.01
    ref=write_json(admitted["root"],str((admitted["directory"]/"invalid_main.json").relative_to(admitted["root"])),value)
    with pytest.raises(ValueError,match="replay"):bind(admitted,receipt_ref=ref)


def test_settings_and_config_mutation_refused(admitted):
    path=admitted["directory"]/"config.json";result=bind(admitted,output=path)
    checked=secondary.admit_secondary(path,result.sha256,project_root=admitted["root"])
    checked.settings["epochs"]=1
    with pytest.raises(ValueError,match="In-memory"):checked.check(False)
    value=json.loads(path.read_text());value["selected_candidate_spec"]["kernel_depth"]=2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):secondary.admit_secondary(path,secondary.common.sha(path),project_root=admitted["root"])


def test_absent_completion_no_config(admitted,monkeypatch):
    def absent(*args,**kwargs):raise RuntimeError("No genuine complete preparation")
    monkeypatch.setattr(secondary.common,"verify_completion",absent)
    with pytest.raises(RuntimeError):bind(admitted,output=admitted["directory"]/"config.json")
    assert not (admitted["directory"]/"config.json").exists()


def test_code_drift_and_fresh_output_guard(admitted,monkeypatch):
    result=bind(admitted,output=admitted["directory"]/"config.json")
    with pytest.raises(ValueError,match="Fresh"):bind(admitted,output=admitted["directory"]/"config.json")
    monkeypatch.setattr(secondary,"implementation_hashes",lambda:{"producer.py":"f"*64})
    with pytest.raises(ValueError,match="dependency"):result.check(False)


def test_tiny_real_matched_control_cpu_fit_and_label_counterfactual(admitted):
    torch.set_num_threads(1);configure_deterministic_development()
    result=bind(admitted);settings=deepcopy(result.settings);settings["epochs"]=2
    cases={"train":admitted["cases"]["train"],"val":admitted["cases"]["val"][:1]}
    packs={id(tile):admitted["pack"] for parents in cases.values() for case in parents for tile in case.tiles}
    name=settings["models"][0];model=runner.make_control(name,settings).double()
    body=runner.fit_configuration(model,cases,packs,name=name,seed=101,settings=settings,config_sha=result.sha256,
        checkpoint=admitted["directory"]/"weights.pt",prediction=admitted["directory"]/"prediction.pt",live_check=lambda full:admitted["pins"])
    assert len(body["history"])==2 and body["checkpoint_replay_nll_delta"]==0.
    assert body["checkpoint_replay_max_abs_logit_delta"]==0.
    original=runner.validate(model,cases["val"],packs)
    case=cases["val"][0];before=case.labels.clone()
    try:
        case.labels.fill_(1);case.tiles[0].labels=case.labels
        changed=runner.validate(model,cases["val"],packs)
        assert changed[0]!=original[0]
        torch.testing.assert_close(changed[2][0],original[2][0],rtol=0,atol=0)
    finally:case.labels.copy_(before);case.tiles[0].labels=case.labels


def test_device_query_never_precedes_main_source_admission(tmp_path,monkeypatch):
    def reject(*args,**kwargs):raise ValueError("Incomplete main study")
    def forbidden(*args,**kwargs):pytest.fail("Device query before mandatory admission")
    monkeypatch.setattr(secondary,"admit_secondary",reject);monkeypatch.setattr(runner,"require_device",forbidden)
    monkeypatch.setattr(torch.cuda,"device_count",forbidden);monkeypatch.setattr(torch.cuda,"is_available",forbidden)
    with pytest.raises(ValueError,match="Incomplete"):
        runner.main(["--config",str(tmp_path/"absent.json"),"--expected-config-sha256","f"*64,"--output",str(tmp_path/"run"),"--device","cuda"])


def test_heldout_ref_precedes_byte_read(admitted,monkeypatch):
    monkeypatch.setattr(Path,"read_bytes",lambda *_:pytest.fail("Held-out bytes read"))
    with pytest.raises(ValueError,match="Held-out"):
        secondary.bind_secondary_development(admitted["proposal_ref"],admitted["plan_ref"],{"path":"data/final_test/selection.json","sha256":"f"*64},project_root=admitted["root"])


@pytest.fixture
def secondary_runs(admitted):
    configure_deterministic_development();torch.set_num_threads(1)
    config_path=admitted["directory"]/"config.json";bound=bind(admitted,output=config_path)
    settings=bound.settings
    template_cases={"train":admitted["cases"]["train"],"val":admitted["cases"]["val"][:1]}
    packs={id(tile):admitted["pack"].to(torch.device("cpu"),torch.float32) for parents in template_cases.values() for case in parents for tile in case.tiles}
    binding={key:settings[key] for key in main.BINDINGS};runs={seed:{} for seed in (101,202,303)}
    for name in settings["models"]:
        toy=deepcopy(settings);toy["epochs"]=2
        body=runner.fit_configuration(runner.make_control(name,toy),template_cases,packs,name=name,seed=101,settings=toy,
            config_sha=bound.sha256,checkpoint=admitted["directory"]/(name+"_template.pt"),
            prediction=admitted["directory"]/(name+"_template_prediction.pt"),live_check=lambda full:admitted["pins"])
        weights=torch.load(body["checkpoint"],weights_only=True)
        prediction=torch.load(body["validation_predictions"],weights_only=True)
        z=prediction["logits"][0]
        rows=[{**body["per_parent"][0],"parent_id":case.parent_id} for case in admitted["cases"]["val"]]
        weights["original_selected_logit_sha256"]=[main.tensor_fingerprint(z)]*16
        prediction.update(logits=[z.clone() for _ in range(16)],positions=[case.positions for case in admitted["cases"]["val"]],
            query_node_ids=[case.query_node_ids for case in admitted["cases"]["val"]],parent_ids=[case.parent_id for case in admitted["cases"]["val"]])
        for seed in (101,202,303):
            value=deepcopy(body);cp=deepcopy(weights);pred=deepcopy(prediction);cp["seed"]=seed;pred["seed"]=seed
            checkpoint=write_tensor(admitted["root"],str((admitted["directory"]/(name+f"_s{seed}.pt")).relative_to(admitted["root"])),cp)
            logits=write_tensor(admitted["root"],str((admitted["directory"]/(name+f"_s{seed}_logits.pt")).relative_to(admitted["root"])),pred)
            value.update(checkpoint=checkpoint["path"],checkpoint_sha256=checkpoint["sha256"],validation_predictions=logits["path"],
                validation_predictions_sha256=logits["sha256"],per_parent=rows,ending_input_artifact_sha256=dict(admitted["snapshot"].hashes))
            runs[seed][name]=value
    refs=[]
    for seed,models in runs.items():
        value={"kind":"secondary_development_run","protocol_sha256":bound.sha256,"training_seed":seed,"models":models,
            "main_selection_receipt_sha256":settings["main_selection_receipt"]["sha256"],"source_binding":binding,
            "initial_implementation_sha256":admitted["pins"],"deterministic_execution_policy":deepcopy(DETERMINISTIC_POLICY)}
        refs.append(write_json(admitted["root"],str((admitted["directory"]/f"run_s{seed}.json").relative_to(admitted["root"])),value))
    return {**admitted,"bound":bound,"config_path":config_path,"runs":runs,"run_refs":refs}


def synth(study,refs=None,ledgers=()):
    return secondary.synthesize_secondary_selection(study["config_path"],study["bound"].sha256,
        run_receipts=study["run_refs"] if refs is None else refs,failure_ledgers=ledgers,project_root=study["root"])


def test_complete_two_lr_three_seed_secondary_synthesis(secondary_runs):
    result=synth(secondary_runs)["values"]
    assert result["matrix_resolved"] is True and result["optimizer_starts_recorded"]==12
    assert all(control["choice"] is not None for control in result["controls"].values())
    assert result["selected_candidate"]==secondary_runs["bound"].settings["selected_candidate"]
    assert result["primary_strongest_baseline"]==secondary_runs["bound"].settings["primary_strongest_baseline"]
    assert result["primary_selection_changed"] is False and result["final_test_access"] is False
    partial=synth(secondary_runs,secondary_runs["run_refs"][:2])["values"]
    assert partial["matrix_resolved"] is False and all(control["choice"] is None for control in partial["controls"].values())
    with pytest.raises(ValueError,match="duplicated"):synth(secondary_runs,[*secondary_runs["run_refs"],secondary_runs["run_refs"][0]])


@pytest.mark.parametrize("field",["control_spec","replay","source","policy","integer_policy","nll"])
def test_secondary_wrong_record_cannot_select(secondary_runs,field):
    refs=deepcopy(secondary_runs["run_refs"])
    value=json.loads((secondary_runs["root"]/refs[0]["path"]).read_text());body=next(iter(value["models"].values()))
    if field=="control_spec":body["control_spec"]["kernel_depth"]=2
    elif field=="replay":body["checkpoint_replay_nll_delta"]=.01
    elif field=="source":body["ending_input_artifact_sha256"]={}
    elif field=="policy":value["deterministic_execution_policy"]["warn_only"]=True
    elif field=="integer_policy":value["deterministic_execution_policy"]["deterministic_algorithms"]=1
    else:
        body["per_parent"][0]["nll"]+=.01
        body["validation_nll"]=sum(row["nll"] for row in body["per_parent"])/16
    ref=write_json(secondary_runs["root"],str((secondary_runs["directory"]/"invalid_run.json").relative_to(secondary_runs["root"])),value)
    refs[0]=ref
    with pytest.raises((ValueError,RuntimeError)):synth(secondary_runs,refs)


def test_terminal_failures_preserved_without_primary_reselection(admitted):
    path=admitted["directory"]/"config.json";bound=bind(admitted,output=path);settings=bound.settings
    name=settings["models"][0]
    entry={"configuration_id":name,"seed":101,"attempt_id":"synthetic_1","terminal":True,"reason":"synthetic numerical failure","test_access":False}
    ledger=write_json(admitted["root"],str((admitted["directory"]/"failures.json").relative_to(admitted["root"])),{
        "kind":"secondary_development_failure_ledger","config_sha256":bound.sha256,
        "main_selection_receipt_sha256":settings["main_selection_receipt"]["sha256"],"entries":[entry]})
    result=secondary.synthesize_secondary_selection(path,bound.sha256,failure_ledgers=[ledger],project_root=admitted["root"])["values"]
    assert len(result["failures"])==1 and result["optimizer_starts_recorded"]==1 and result["matrix_resolved"] is False
    assert result["selected_candidate"]==settings["selected_candidate"]
    with pytest.raises(ValueError,match="Duplicate"):
        secondary.synthesize_secondary_selection(path,bound.sha256,failure_ledgers=[ledger,ledger],project_root=admitted["root"])


def test_actual_observed_query_contract_drift_refused(admitted,monkeypatch):
    original=secondary.common.verify_completion
    def altered(*args,**kwargs):
        completion,dataset,ages,lock=original(*args,**kwargs)
        values=[SimpleNamespace(**vars(case)) for case in dataset.cases("val")]
        values[0].query_node_ids=values[0].query_node_ids.flip(0)
        return completion,SimpleNamespace(cases=lambda split:values if split=="val" else dataset.cases(split)),ages,lock
    monkeypatch.setattr(secondary.common,"verify_completion",altered)
    with pytest.raises(ValueError,match="observed coordinate/query"):bind(admitted)


def test_effective_numerical_policy_drift_blocks_tiny_fit(admitted):
    configure_deterministic_development();bound=bind(admitted);settings=deepcopy(bound.settings);settings["epochs"]=2
    cases={"train":admitted["cases"]["train"],"val":admitted["cases"]["val"][:1]}
    packs={id(tile):admitted["pack"] for values in cases.values() for case in values for tile in case.tiles}
    def drift(full):torch.use_deterministic_algorithms(False);return admitted["pins"]
    try:
        with pytest.raises(RuntimeError,match="deterministic"):
            runner.fit_configuration(runner.make_control(settings["models"][0],settings).double(),cases,packs,
                name=settings["models"][0],seed=101,settings=settings,config_sha=bound.sha256,
                checkpoint=admitted["directory"]/"no_weights.pt",prediction=admitted["directory"]/"no_predictions.pt",live_check=drift)
        assert not (admitted["directory"]/"no_weights.pt").exists()
    finally:configure_deterministic_development()



def test_non_executable_conditional_proposal_cannot_run(admitted):
    with pytest.raises(ValueError,match="genuine completion/selection-bound"):
        secondary.admit_secondary(admitted["proposal_ref"]["path"],admitted["proposal_ref"]["sha256"],project_root=admitted["root"])



def test_every_main_family_consumes_complete_producer_guard(admitted,monkeypatch):
    visited=[]
    def guard(family,settings,**context):
        assert context["source_snapshot"] is admitted["snapshot"]
        assert context["completion_path"]==admitted["root"]/admitted["plan"]["source_artifacts"]["completion"]["path"]
        visited.append(family)
    monkeypatch.setattr(combined,"validate_producer_settings",guard,raising=False)
    bind(admitted)
    assert set(visited)==set(combined.FAMILIES) and len(visited)==10
    def reject(family,settings,**context):
        if family=="gru":raise ValueError("Producer temporal-depth guard")
    monkeypatch.setattr(combined,"validate_producer_settings",reject,raising=False)
    with pytest.raises(ValueError,match="Producer temporal-depth"):
        bind(admitted,output=admitted["directory"]/"no_config.json")
    assert not (admitted["directory"]/"no_config.json").exists()
