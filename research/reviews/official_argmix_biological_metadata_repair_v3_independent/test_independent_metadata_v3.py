"""Independent source and handmade ARGMix v3 audit, without scientific arrays."""
from copy import deepcopy
from pathlib import Path
import ast
import hashlib
import json

import pytest
import yaml
from arggnn import official_argmix_biological as bridge

AUDIT = bridge.ROOT / "research/reviews/official_argmix_biological_metadata_repair_v3_source_audit"
CONFIG_SHA = "25829cdb811da918db4cb7fad5ee607a44610bd1f04bc40dbe2f5564a6d58eb2"
MODULE_SHA = "4a6c44dfd81ba3f75e74c5287a4b55776bb9fe1612732c457748643cd75d7b1f"


def cfg():
    return bridge.policy(CONFIG_SHA)


def binding(p):
    return {"path": str(p.relative_to(bridge.ROOT)), "sha256": bridge.sha(p)}


def test_exact_scientific_conservation_and_author_copy_inventory():
    current = bridge.MODULE.read_text()
    prior = (AUDIT / "prior_owned_source.py").read_text()
    defs = lambda text: {n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    old, new = defs(prior), defs(current)
    assert set(new) - set(old) == {"reference_metadata_amendment"}
    assert set(old) - set(new) == set()
    assert {k for k in old if old[k] != new[k]} == {
        "policy", "stage", "verify_backend", "stock_train", "infer_and_seal"}
    original = json.loads((AUDIT / "prior_config.json").read_text())
    repaired = cfg()
    assert set(repaired) - set(original) == {"compatibility_amendment"}
    for key in original:
        if key != "owned_code_sha256":
            assert original[key] == repaired[key], key
    assert bridge.sha(bridge.MODULE) == MODULE_SHA
    assert bridge.sha(bridge.SCRIPT) == "4261fb1195d16f73370b5fa0144ba0f59afe6a67726af6567634b0bd282e223d"
    files = bridge.source_inventory(require_copy=True)
    assert len(files) == 86
    assert hashlib.sha256(bridge.canonical(files) + b"\n").hexdigest() == bridge.SOURCE_TREE_SHA


def test_exact_inherited_product_json_and_paired_observed_metadata():
    policy = cfg()
    amendment = policy["compatibility_amendment"]
    f_record = amendment["inherited_products"]["official_argmix_observed_frames_v1"]
    n_record = amendment["inherited_products"]["official_argmix_native_biological_archives_v1"]
    frames = bridge.stage(f_record, "official_argmix_observed_frames_v1", policy, observed=True)
    native = bridge.stage(n_record, "official_argmix_native_biological_archives_v1", policy)
    assert native["frames"] == f_record
    for product in (frames, native):
        assert product["config_sha256"] == amendment["prior_config_sha256"]
        assert product["owned_code_sha256"] == amendment["prior_owned_code_sha256"]
    reference_hashes = []
    signatures = []
    counts = {pop: 24 for pop in policy["class_order"]}
    counts.update(QUERY=120, UNLABELED_REF=64)
    for parent in bridge.PARENT_SPLITS:
        frame = frames["parents"][parent]
        assert frame["split"] == bridge.PARENT_SPLITS[parent]
        refs = bridge.checked(frame["references"], observed=True)
        roster_path = bridge.checked(frame["roster"], observed=True)
        roster = json.loads(roster_path.read_text())
        original_roster = json.loads(bridge.checked(
            frame["source_observed_entry"]["observed_sample_manifest"], observed=True).read_text())
        assert roster == original_roster
        rows = [(int(node), pop, float(age)) for node, pop, age
                in [line.split() for line in refs.read_text().splitlines()]]
        expected = []
        for row in roster:
            role, klass = row["role"], row["observed_reference_class"]
            if role == "pure_source_reference":
                assert type(klass) is int and 0 <= klass < 4
                pop = policy["class_order"][klass]
            else:
                assert klass is None and role in {"query", "mixed_reference_only"}
                pop = "QUERY" if role == "query" else "UNLABELED_REF"
            expected.append((row["inferred_tree_node_id"], pop, float(row["age_generations"])))
        assert rows == expected and len(rows) == 280
        assert {row[0] for row in rows} == set(range(280))
        assert {pop: sum(row[1] == pop for row in rows) for pop in set(row[1] for row in rows)} == counts
        signatures.append([(node, pop) for node, pop, age in rows])
        reference_hashes.append(frame["references"]["sha256"])
    assert len(set(reference_hashes)) == 1
    assert all(signature == signatures[0] for signature in signatures)
    assert amendment["metadata_reference_parent"] == "bio_paths_pilot_train_000"
    assert bridge.PARENT_SPLITS[amendment["metadata_reference_parent"]] == "train"
    for split in ("train", "val"):
        actual = yaml.safe_load(bridge.checked(native[split + "_pairs"]).read_text())
        expected = []
        for parent, declared in bridge.PARENT_SPLITS.items():
            if declared == split:
                frame = frames["parents"][parent]
                # Compare binary/truth pointers only, never open or hash their contents.
                expected.append(dict(rel=str(bridge.local(frame["tree"]["path"], observed=True)),
                    references=str(bridge.local(frame["references"]["path"], observed=True)),
                    relate_sample=str(bridge.local(frame["native_sample"]["path"], observed=True)),
                    truth_npz=str(bridge.local(native["truth_provenance"][parent]["native_truth"]["path"]))))
        assert actual == expected
    backend = yaml.safe_load(bridge.checked(native["backend"]).read_text())
    bridge.verify_backend(backend, policy)
    assert backend["paths"]["train_pairs"] == str(bridge.local(native["train_pairs"]["path"]))
    assert backend["paths"]["val_pairs"] == str(bridge.local(native["val_pairs"]["path"]))


def test_stock_counter_on_handmade_metadata_and_log_only_flow(tmp_path):
    base = bridge.SOURCE / "src"
    sources = {"load_node_metadata": base / "utils/tsv_io.py",
        "count_joint_predict_haplotypes": base / "joint_interval_model.py",
        "get_paths_config": base / "config_loader.py",
        "get_predict_pops": base / "config_loader.py"}
    nodes = []
    for name, file in sources.items():
        nodes.append(next(n for n in ast.parse(file.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name))
    tree = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace = {"Path": Path}
    exec(compile(tree, "independent_handmade_counter_only", "exec"), namespace)
    backend = bridge.backend_config(cfg(), tmp_path)
    with pytest.raises(ValueError, match="paths.references is required"):
        namespace["count_joint_predict_haplotypes"](backend)
    refs = tmp_path / "observed_metadata.tsv"
    refs.write_text("".join(f"{i} {'QUERY' if i < 120 else 'UNLABELED_REF' if i < 184 else ['ANA','CHG','WHG','EHG'][(i-184)//24]} {i / 2}\n" for i in range(280)))
    backend["paths"]["references"] = str(refs)
    assert namespace["count_joint_predict_haplotypes"](backend) == 120
    trainer = ast.parse((base / "train_joint_model.py").read_text())
    loads = sorted(n.lineno for n in ast.walk(trainer)
        if isinstance(n, ast.Name) and n.id == "available_predict" and isinstance(n.ctx, ast.Load))
    assert loads == [347, 599]
    assert nodes[1].lineno == 719


def test_old_gpu_rejected_and_current_handmade_receipt_requires_both_qualifications(tmp_path):
    policy = cfg()
    payload = {"status": "passed", "result": dict(phase="qualify-gpu", actual_cuda=True,
        actual_source_joint_GPU_forward_backward=True, actual_stock_Rust_TorchScript_GPU_fixture=True,
        config_sha256=policy["compatibility_amendment"]["prior_config_sha256"],
        owned_code_sha256=policy["compatibility_amendment"]["prior_owned_code_sha256"])}
    file = tmp_path / "handmade_gpu_admission.json"
    file.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="Same-policy/code"):
        bridge.gpu_receipt(binding(file), policy)
    payload["result"].update(config_sha256=CONFIG_SHA, owned_code_sha256=policy["owned_code_sha256"])
    file.write_text(json.dumps(payload))
    assert bridge.gpu_receipt(binding(file), policy) == binding(file)
    for key in ("actual_source_joint_GPU_forward_backward", "actual_stock_Rust_TorchScript_GPU_fixture"):
        altered = deepcopy(payload)
        altered["result"][key] = False
        file.write_text(json.dumps(altered))
        with pytest.raises(RuntimeError):
            bridge.gpu_receipt(binding(file), policy)


def test_fresh_fit_required_and_parent_specific_inference_reference_preserved():
    source = bridge.MODULE.read_text()
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    infer = ast.get_source_segment(source, functions["infer_and_seal"])
    assert 'stage(fit_record, "official_argmix_biological_stock_fit_v1", cfg)' in infer
    assert 'fit.get("metadata_reference_counting") != metadata_info' in infer
    assert '"--references", checked(frame["references"], observed=True)' in infer
    assert 'references_record=metadata_info["selected_reference"]' in infer
    script = bridge.SCRIPT.read_text()
    assert 'if args.phase != "qualify-gpu":' in script
    assert 'bridge.gpu_receipt(record(args.gpu_admission' in script
