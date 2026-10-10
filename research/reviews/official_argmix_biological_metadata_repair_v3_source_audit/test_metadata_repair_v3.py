"""Handmade metadata/source repair checks. Never open native arrays/archives."""
from copy import deepcopy
import ast
import json
from pathlib import Path
from typing import Any, Dict
import types

import pytest
import yaml

from arggnn import official_argmix_biological as bridge

AUDIT = bridge.ROOT / "research/reviews/official_argmix_biological_metadata_repair_v3_source_audit"


def real_cfg():
    return json.loads(bridge.CONFIG.read_text())


def record(path, root):
    return {"path":str(path.relative_to(root)), "sha256":bridge.sha(path)}


@pytest.fixture
def handmade(tmp_path, monkeypatch):
    cfg=real_cfg()
    frames={"kind":"handmade_metadata", "config_sha256":bridge.METADATA_REPAIR_V3["prior_config_sha256"],
        "owned_code_sha256":bridge.METADATA_REPAIR_V3["prior_owned_code_sha256"], "parents":{}}
    prepared={"frames":{"path":"frames.json","sha256":"f"*64},
        "config_sha256":frames["config_sha256"],"owned_code_sha256":frames["owned_code_sha256"],
        "truth_provenance":{}}
    rows=[]
    tsv=[]
    for node in range(280):
        if node<120:role,klass,pop="query",None,"QUERY"
        elif node<184:role,klass,pop="mixed_reference_only",None,"UNLABELED_REF"
        else:
            klass=(node-184)//24
            role,pop="pure_source_reference",cfg["class_order"][klass]
        rows.append({"inferred_tree_node_id":node,"role":role,
            "observed_reference_class":klass,"age_generations":float(node%10)})
        tsv.append(f"{node} {pop} {float(node%10)}")
    for parent in bridge.PARENT_SPLITS:
        folder=tmp_path/parent
        folder.mkdir()
        references=folder/"references.tsv"
        references.write_text("\n".join(tsv)+"\n")
        roster=folder/"observed_roster.json"
        roster.write_text(json.dumps(rows))
        frame={"references":record(references,tmp_path),"roster":record(roster,tmp_path)}
        for key in ["tree","native_sample"]:
            p=folder/(key+".metadata-placeholder.txt")
            p.write_text("handmade hash placeholder, no array/tree/archive")
            frame[key]=record(p,tmp_path)
        truth=folder/"native-label-pointer.metadata-placeholder.txt"
        truth.write_text("handmade pointer placeholder, no label array")
        prepared["truth_provenance"][parent]={"native_truth":record(truth,tmp_path)}
        frames["parents"][parent]=frame
    monkeypatch.setattr(bridge,"ROOT",tmp_path)
    monkeypatch.setattr(bridge,"verify_frames",lambda *args:frames)
    for split in ["train","val"]:
        pairs=[]
        for parent, declared in bridge.PARENT_SPLITS.items():
            if split!=declared:continue
            frame=frames["parents"][parent]
            pairs.append({key:str(tmp_path/frame[field]["path"]) for key,field in
                [("rel","tree"),("references","references"),("relate_sample","native_sample")]})
            pairs[-1]["truth_npz"]=str(tmp_path/prepared["truth_provenance"][parent]["native_truth"]["path"])
        path=tmp_path/(split+"_pairs.yaml")
        path.write_text(yaml.safe_dump(pairs,sort_keys=False))
        prepared[split+"_pairs"]=record(path,tmp_path)
    return cfg,frames,prepared,tmp_path


def test_metadata_roles_all_four_and_first_train_pointer(handmade):
    cfg,frames,prepared,root=handmade
    result=bridge.reference_metadata_amendment(prepared,cfg)
    assert result["stock_available_predict_haplotypes"]==120
    assert result["selected_parent"]=="bio_paths_pilot_train_000"
    assert result["selected_reference"]==frames["parents"]["bio_paths_pilot_train_000"]["references"]
    assert len(result["parents"])==4
    for row in result["parents"].values():
        assert row["counts"]=={"QUERY":120,"UNLABELED_REF":64,"ANA":24,"CHG":24,"WHG":24,"EHG":24}
        assert row["exact_observed_roster_join"] is True
    assert len({row["eligibility_sha256"] for row in result["parents"].values()})==1
    assert result["inherited_products"]["frames"]["original_config_sha256"]==bridge.METADATA_REPAIR_V3["prior_config_sha256"]


@pytest.mark.parametrize("mutation",["reference","tree","sample","labels","extra","missing"])
def test_paired_pointers_remain_exact_before_metadata_addition(handmade,mutation):
    cfg,frames,prepared,root=handmade
    path=root/prepared["train_pairs"]["path"]
    pairs=yaml.safe_load(path.read_text())
    if mutation=="missing":pairs.pop()
    elif mutation=="extra":pairs[0]["extra"]=str(root/"irrelevant.txt")
    else:
        key={"reference":"references","tree":"rel","sample":"relate_sample","labels":"truth_npz"}[mutation]
        pairs[0][key]=str(root/"wrong-pointer.txt")
    path.write_text(yaml.safe_dump(pairs))
    prepared["train_pairs"]["sha256"]=bridge.sha(path)
    with pytest.raises(ValueError,match="pointers changed"):
        bridge.reference_metadata_amendment(prepared,cfg)


@pytest.mark.parametrize("mutation",["count","class","duplicate","age","roster","extra_field"])
def test_metadata_eligibility_and_roster_rejects_without_targets(handmade,mutation):
    cfg,frames,prepared,root=handmade
    frame=frames["parents"]["bio_paths_pilot_val_001"]
    path=root/frame["references"]["path"]
    lines=path.read_text().splitlines()
    if mutation=="count":lines[0]=lines[0].replace("QUERY","UNLABELED_REF")
    elif mutation=="class":lines[-1]=lines[-1].replace("EHG","OTHER")
    elif mutation=="duplicate":lines[1]=lines[0]
    elif mutation=="age":lines[0]="0 QUERY nan"
    elif mutation=="extra_field":lines[0]+=" extra"
    else:
        roster=root/frame["roster"]["path"]
        rows=json.loads(roster.read_text())
        rows[0]["observed_reference_class"]=0
        roster.write_text(json.dumps(rows))
        frame["roster"]["sha256"]=bridge.sha(roster)
    path.write_text("\n".join(lines)+"\n")
    frame["references"]["sha256"]=bridge.sha(path)
    with pytest.raises(ValueError):
        bridge.reference_metadata_amendment(prepared,cfg)


def test_backend_optional_pointer_is_observed_bound_and_no_other_schema_additions(handmade):
    cfg,frames,prepared,root=handmade
    backend=bridge.backend_config(cfg,root)
    bridge.verify_backend(backend,cfg)
    selected=frames["parents"]["bio_paths_pilot_train_000"]["references"]
    backend["paths"]["references"]=str(root/selected["path"])
    with pytest.raises(ValueError,match="exact observed binding"):
        bridge.verify_backend(backend,cfg)
    bridge.verify_backend(backend,cfg,references_record=selected)
    old=deepcopy(backend)
    old["paths"].pop("references")
    with pytest.raises(ValueError,match="requires stock metadata"):
        bridge.verify_backend(old,cfg,references_record=selected)
    wrong=deepcopy(backend)
    wrong["paths"]["references"]=str(root/frames["parents"]["bio_paths_pilot_val_000"]["references"]["path"])
    with pytest.raises(ValueError,match="exact observed binding"):
        bridge.verify_backend(wrong,cfg,references_record=selected)
    bad=deepcopy(backend)
    bad["paths"]["reference_override"]="unused"
    with pytest.raises(ValueError,match="schema"):
        bridge.verify_backend(bad,cfg,references_record=selected)
    infer=deepcopy(backend)
    infer["processing"]["exclude_edge_trees"]=False
    bridge.verify_backend(infer,cfg,inference=True,references_record=selected)
    assert backend["model"]==cfg["native_model"]
    assert backend["joint_model"]==cfg["joint_model"]


def test_stage_compatibility_requires_one_exact_record_kind_prior_config_and_owned_hashes(tmp_path,monkeypatch):
    cfg=real_cfg()
    old=cfg["compatibility_amendment"]
    kind="official_argmix_observed_frames_v1"
    payload={"kind":kind,"config_sha256":old["prior_config_sha256"],"owned_code_sha256":old["prior_owned_code_sha256"]}
    path=tmp_path/"admitted-product.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(bridge,"ROOT",tmp_path)
    binding=record(path,tmp_path)
    cfg["compatibility_amendment"]["inherited_products"][kind]=binding
    assert bridge.stage(binding,kind,cfg)==payload
    assert payload["config_sha256"]==old["prior_config_sha256"]
    for key,new in [("config_sha256","0"*64),("owned_code_sha256",{}),("kind","wrong_kind")]:
        bad=dict(payload)
        bad[key]=new
        path.write_text(json.dumps(bad))
        new_binding=record(path,tmp_path)
        cfg["compatibility_amendment"]["inherited_products"][kind]=new_binding
        with pytest.raises(ValueError):
            bridge.stage(new_binding,kind,cfg)
    path.write_text(json.dumps(payload))
    cfg["compatibility_amendment"]["inherited_products"][kind]=binding
    clone=tmp_path/"unregistered-copy.json"
    clone.write_bytes(path.read_bytes())
    with pytest.raises(ValueError,match="exact admitted"):
        bridge.stage(record(clone,tmp_path),kind,cfg)
    with pytest.raises(ValueError,match="Bound artifact changed"):
        bridge.stage(dict(binding,sha256="f"*64),kind,cfg)
    with pytest.raises(ValueError,match="exact admitted"):
        bridge.stage(dict(binding,unrequested_extension=True),kind,cfg)


def test_old_gpu_receipt_is_not_covered_by_compatibility(tmp_path,monkeypatch):
    cfg=real_cfg()
    payload={"status":"passed","result":{"phase":"qualify-gpu",
        "actual_source_joint_GPU_forward_backward":True,"actual_stock_Rust_TorchScript_GPU_fixture":True,
        "actual_cuda":True,"config_sha256":bridge.METADATA_REPAIR_V3["prior_config_sha256"],
        "owned_code_sha256":bridge.METADATA_REPAIR_V3["prior_owned_code_sha256"]}}
    path=tmp_path/"old_gpu.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(bridge,"ROOT",tmp_path)
    with pytest.raises(RuntimeError,match="Same-policy/code"):
        bridge.gpu_receipt(record(path,tmp_path),cfg)


def test_exact_unchanged_stock_count_function_only_reads_query_eligibility(handmade):
    cfg,frames,prepared,root=handmade
    source=bridge.SOURCE/"src"
    loader_text=(source/"utils/tsv_io.py").read_text()
    loader_node=next(n for n in ast.parse(loader_text).body if isinstance(n,ast.FunctionDef) and n.name=="load_node_metadata")
    joint_text=(source/"joint_interval_model.py").read_text()
    counter_node=next(n for n in ast.parse(joint_text).body if isinstance(n,ast.FunctionDef) and n.name=="count_joint_predict_haplotypes")
    scope={"Path":Path,"get_paths_config":lambda c:c["paths"],
        "get_predict_pops":lambda c:c["populations"]["predict"]}
    tree=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),loader_node,counter_node],type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree,"handmade_stock_metadata_definition_only","exec"),scope)
    backend=bridge.backend_config(cfg,root)
    with pytest.raises(ValueError,match="paths.references is required"):
        scope["count_joint_predict_haplotypes"](backend)
    for frame in frames["parents"].values():
        backend["paths"]["references"]=str(root/frame["references"]["path"])
        assert scope["count_joint_predict_haplotypes"](backend)==120


def test_scientific_bodies_author_inventory_and_original_products_unchanged():
    old=ast.parse((AUDIT/"prior_owned_source.py").read_text())
    new=ast.parse(bridge.MODULE.read_text())
    defs=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    old_functions,new_functions=defs(old),defs(new)
    changed={name for name in old_functions if old_functions[name]!=new_functions[name]}
    assert changed=={"policy","stage","verify_backend","stock_train","infer_and_seal"}
    assert set(new_functions)-set(old_functions)=={"reference_metadata_amendment"}
    assignments=lambda tree:{target.id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) for target in n.targets if isinstance(target,ast.Name) and target.id in {"NATIVE_MODEL","JOINT_MODEL"}}
    assert assignments(old)==assignments(new)
    assert bridge.sha(AUDIT/"prior_owned_source.py")==bridge.METADATA_REPAIR_V3["prior_owned_code_sha256"]["src/arggnn/official_argmix_biological.py"]
    assert bridge.sha(AUDIT/"prior_config.json")==bridge.METADATA_REPAIR_V3["prior_config_sha256"]
    # Source inventory reads only immutable original author source, no archive/runtime execution.
    assert len(bridge.source_inventory(require_copy=False))==86
    for binding in bridge.METADATA_REPAIR_V3["inherited_products"].values():
        assert bridge.sha(bridge.ROOT/binding["path"])==binding["sha256"]


def test_stock_count_value_flows_only_to_logs_and_metadata_summary():
    text=(bridge.SOURCE/"src/train_joint_model.py").read_text()
    tree=ast.parse(text)
    uses=sorted(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=="available_predict" and isinstance(n.ctx,ast.Load))
    assert uses==[347,599]
    assert "logger.info(" in "\n".join(text.splitlines()[331:351])
    assert "write_joint_training_data_log(" in "\n".join(text.splitlines()[593:606])
    counter=next(n for n in ast.parse((bridge.SOURCE/"src/joint_interval_model.py").read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="count_joint_predict_haplotypes")
    assert counter.lineno==719
