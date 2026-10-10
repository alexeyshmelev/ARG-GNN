"""Handmade CPU tests only. No project arrays or CUDA contexts."""
from dataclasses import replace
import copy
import hashlib
import json
import os

import pytest
import torch

from arggnn.anchor_kernel import AnchorKernelBank, AnchorKernelModel, AnchorOperators, _layout
from arggnn.kron_interface import LeafInterfaceBank, QueryInterfaceCache
from arggnn.anchor_compact_actual_bank_v1 import (
    AnchorCompactActualBankModelV1, admit_actual_qualification_v1, checked_file,
    compare_action_VJPs_v1, compare_model_step_v1, comparison_v1,
    cuda_profile_pair_v1, cuda_isolated_memory_v1, derive_compact_actual_bank_v1,
    qualification_objective_v1, qualification_models_v1, qualification_direct_control_models_v2, select_train_tiles_v1)
import arggnn.anchor_compact_actual_bank_v1 as module


@pytest.fixture(autouse=True)
def cpu_only():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)


def fixture(dtype=torch.float64, coupling="full", stress=False):
    intervals = torch.tensor([[0., .22], [.22, .5], [.5, 1.]], dtype=torch.float64)
    positions = torch.tensor([.5, 0., .12, .999, .22, .5, 1/6, .82], dtype=torch.float64)
    base = torch.tensor([[[2., -1., 0.], [-1., 3., -1.], [0., -1., 2.]],
        [[3., 0., -1.], [0., 2., -.5], [-1., -.5, 2.5]],
        [[2.5, -.4, -.3], [-.4, 1.8, -.2], [-.3, -.2, 2.2]]], dtype=torch.float64)
    cqq = torch.stack((base, 1.3*base+.2*torch.eye(3, dtype=torch.float64)), dim=1)
    fractions = torch.tensor([[.6, .4], [.3, .7], [.8, .2]], dtype=torch.float64)
    cqr = -cqq.sum(-1)[..., None]*fractions
    eig, vec = torch.linalg.eigh(cqq)
    full = torch.zeros(3, 5, 5, dtype=torch.float64)
    full[:, :3, :3] = base
    full[:, :3, 3:] = cqr[:, 0]
    full[:, 3:, :3] = cqr[:, 0].transpose(-1, -2)
    full[:, 3:, 3:] = torch.diag_embed(-cqr[:, 0].sum(-2))
    q, r = torch.arange(3), torch.arange(3, 5)
    ids = torch.arange(5)+100
    cache = QueryInterfaceCache(full, torch.ones(3), base, cqr[:, 0],
        eig[:, 0], vec[:, 0], q, r, ids[q], ids[r], torch.tensor([.22, .28, .5]),
        base.mean(0), eig[0, 0], vec[0, 0])
    leaf = torch.randn(3, 5, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(771))
    bank = LeafInterfaceBank(full, ids, torch.arange(15).reshape(3, 5), leaf,
        tuple(range(4)), intervals, ids[q], positions,
        torch.searchsorted(intervals[:, 0].contiguous(), positions, right=True)-1,
        torch.tensor([0., 1.], dtype=torch.float64), (), .0001, 10000., query_cache=cache)
    ops = AnchorOperators(cqq, cqr, eig, vec)
    prepared = AnchorKernelBank(bank, _layout(bank, 3), {"age": ops, "plain": ops}, 2)
    args = dict(input_dim=4, num_classes=4, hidden_dim=6, num_layers=2, latent_dim=2,
                kernel_depth=2, num_scales=2, geometry="age", coupling=coupling,
                num_anchors=3, initial_coupling=(5., 20.), initial_damping=1.)
    torch.manual_seed(101)
    native = AnchorKernelModel(**args).to(dtype=dtype)
    compact = AnchorCompactActualBankModelV1(**args, readout_chunk_size=3).to(dtype=dtype)
    if stress:
        with torch.no_grad():
            for layer in native.layers:
                layer.raw_coupling.copy_(torch.linspace(-.7, .6, 4).reshape(2, 2))
                layer.raw_damping.copy_(torch.linspace(.4, -.3, 4).reshape(2, 2))
                layer.mixture_logits.copy_(torch.tensor([[.8, -.6], [-.4, .5]]))
    compact.load_state_dict(native.state_dict(), strict=True)
    view = derive_compact_actual_bank_v1(prepared, "age", coupling).to("cpu", dtype)
    return native, compact, prepared.to("cpu", dtype), view


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("coupling", ["full", "scalar", "diagonal", "gain_matched_diagonal"])
@pytest.mark.parametrize("stress", [False, True])
def test_all_Q_mass_VJP_and_private_clipped_AdamW(dtype, coupling, stress):
    native, compact, bank, view = fixture(dtype, coupling, stress)
    original = {k: v.clone() for k, v in native.state_dict().items()}
    result = compare_model_step_v1(native, compact, bank, view)
    assert result["status"] == "PASS", result
    assert result["preclip_native_norm"] > 1
    assert result["preclip_compact_norm"] > 1
    assert result["all_Q_checked"] and not result["labels_opened"]
    assert all(x["passed"] for x in result["checks"].values())
    assert native.state_dict().keys() == compact.state_dict().keys()
    for name, before in original.items():
        torch.testing.assert_close(native.state_dict()[name], before, rtol=0, atol=0)
    action = compare_action_VJPs_v1(native, bank, view)
    assert action["status"] == "PASS", action
    assert result["cell_identity"]["coupling"] == action["cell_identity"]["coupling"] == coupling
    assert result["cell_identity"]["geometry"] == action["cell_identity"]["geometry"] == "age"
    assert action["cell_identity"]["probe"] == "independent_signal_action_VJPs"
    assert result["cell_identity"]["probe"] == "full_model_VJPs_and_clipped_AdamW"
    assert not any(p.grad is not None for p in native.parameters())


def test_minimal_alias_view_and_coordinate_precision():
    native, compact, bank, view = fixture()
    raw = derive_compact_actual_bank_v1(bank, "age", "full")
    assert raw.leaf_x.data_ptr() == bank.bank.leaf_x.data_ptr()
    assert raw.geometry.eigenvectors.data_ptr() == bank.operators["age"].eigenvectors.data_ptr()
    assert raw.geometry.diagonal is None
    assert not hasattr(raw, "cqq") and not hasattr(raw, "query_cache")
    low = raw.to("cpu", torch.float32)
    assert low.leaf_x.dtype == torch.float32
    for name in ("anchors", "coordinates", "weights", "evaluation_left", "evaluation_right"):
        assert getattr(low.layout, name).dtype == torch.float64


@pytest.mark.parametrize("coupling", ["diagonal", "gain_matched_diagonal"])
def test_diagonal_view_has_original_diagonal(coupling):
    _, _, bank, _ = fixture()
    view = derive_compact_actual_bank_v1(bank, "age", coupling)
    assert view.geometry.diagonal.data_ptr() == bank.operators["age"].cqq.diagonal(
        dim1=-2, dim2=-1).data_ptr()


@pytest.mark.parametrize("change", ["geometry", "coupling", "features", "scales", "nan"])
def test_model_contract_rejected(change):
    _, compact, _, view = fixture()
    if change == "geometry":
        view = replace(view, geometry_name="plain")
    elif change == "coupling":
        view = replace(view, coupling="scalar")
    elif change == "features":
        view = replace(view, leaf_x=view.leaf_x[:, :, :3])
    elif change == "scales":
        view = replace(view, num_scales=3)
    else:
        leaf = view.leaf_x.clone()
        leaf[0, 0, 0] = float("nan")
        view = replace(view, leaf_x=leaf)
    with pytest.raises(ValueError):
        compact(view)


def test_no_optimizer_step_on_parity_failure(monkeypatch):
    native, compact, bank, view = fixture()
    original = module.comparison_v1
    called = []
    def failed(*args, **kwargs):
        result = original(*args, **kwargs)
        return dict(result, passed=False)
    monkeypatch.setattr(module, "comparison_v1", failed)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args, **kwargs: called.append(True))
    result = compare_model_step_v1(native, compact, bank, view)
    assert result["status"] == "parity_failed_optimizer_not_run"
    assert not called


def tile(g, p, number):
    return dict(tile_id=number, original_tree_ids=list(range(g)), position_indexes=list(range(p)),
        query_dynamics=184, pure_reference_sources=96, supervised_outputs=120,
        mixed_latent_outputs=64, mixed_query_flag=0, all_tree_nodes_retained=True,
        all280_leaves_retained=True, exact_serialization_roundtrip_checked=True,
        all_original_age_cache_fields_retained=True,
        bank={"path":f"handmade_train_tile_{number}", "sha256":"a"*64},
        full_bank_field_fingerprint_sha256="b"*64)


def manifest():
    return dict(status="complete_observed_only_Q184_R96_actualQ120", truth_opened=False,
        parents=[dict(parent_id="bio_paths_pilot_train_000", split="train",
                      tiles=[tile(2, 30, 0), tile(10, 20, 1)]),
                 dict(parent_id="bio_paths_pilot_train_001", split="train",
                      tiles=[tile(2, 4, 0), tile(7, 25, 1)]),
                 dict(parent_id="bio_paths_pilot_val_000", split="val",
                      tiles=[tile(1, 1, 0)])])


def test_selection_train_only_extrema_and_ties():
    chosen = select_train_tiles_v1(manifest())
    assert [(x["parent_id"], x["tile_id"]) for x in chosen] == [
        ("bio_paths_pilot_train_000", 0), ("bio_paths_pilot_train_000", 1),
        ("bio_paths_pilot_train_001", 0)]
    assert [x["selection_reasons"] for x in chosen] == [
        ["minimum_tree_count"], ["maximum_tree_count"], ["minimum_readout_count"]]


def test_selection_is_metadata_only_and_deduplicates():
    data = manifest()
    data["parents"][1]["tiles"] = [tile(2, 40, 0)]
    chosen = select_train_tiles_v1(data)
    assert len(chosen) == 2
    assert chosen[1]["selection_reasons"] == ["maximum_tree_count", "minimum_readout_count"]
    assert all(not x["parent_id"].startswith("bio_paths_pilot_val") for x in chosen)


@pytest.mark.parametrize("change", ["truth", "split", "role", "duplicate", "empty"])
def test_selection_bad_provenance_or_roles_fails(change):
    data = manifest()
    if change == "truth":
        data["truth_opened"] = True
    elif change == "split":
        data["parents"][0]["split"] = "val"
    elif change == "role":
        data["parents"][0]["tiles"][0]["mixed_query_flag"] = 1
    elif change == "duplicate":
        data["parents"][0]["tiles"].append(data["parents"][0]["tiles"][0])
    else:
        data["parents"][0]["tiles"][0]["original_tree_ids"] = []
    with pytest.raises(ValueError):
        select_train_tiles_v1(data)


def record(path):
    return {"path":str(path), "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def test_hash_paths_reject_drift_symlink_and_escape(tmp_path):
    file = tmp_path / "evidence.json"
    file.write_text("{}")
    item = record(file)
    assert checked_file(item, tmp_path) == file
    file.write_text('{"changed":true}')
    with pytest.raises(ValueError):
        checked_file(item, tmp_path)
    linked = tmp_path / "link"
    linked.symlink_to(file)
    with pytest.raises(ValueError):
        checked_file(record(linked), tmp_path)
    with pytest.raises(ValueError):
        checked_file({"path": "../outside", "sha256":"0"*64}, tmp_path)


def test_source_only_template_cannot_admit_actual_data(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(dict(kind="anchor_compact_actual_bank_protocol_v1",
        status="source_only_actual_execution_not_admitted", fit_allowed=False, truth_allowed=False)))
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps(dict(kind="root_compact_actual_bank_qualification_admission_v1",
        status="inactive_template_root_must_issue_fresh_receipt")))
    opened = []
    monkeypatch.setattr(torch, "load", lambda *a, **kw: opened.append(a))
    with pytest.raises(ValueError, match="root receipt"):
        admit_actual_qualification_v1(record(cfg), record(admission), root=tmp_path)
    assert not opened


def test_profile_blocked_before_any_cuda_call(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "recheck_admission_v1", lambda a: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: calls.append(True))
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="profiling"):
        cuda_profile_pair_v1({}, SimpleNamespace(binding={}), {}, {}, {}, stage="full_step")
    assert not calls


def test_comparison_records_failure_and_rejects_nonfinite():
    expected = torch.tensor([0., 1.], dtype=torch.float64)
    result = comparison_v1(expected+1e-6, expected)
    assert not result["passed"] and result["max_acceptance_units"] > 1
    with pytest.raises(ValueError):
        comparison_v1(torch.tensor([float("nan")]), torch.tensor([1.]))


@pytest.mark.parametrize("state", ["initial", "rate_mixture_stress"])
def test_candidate_sized_models_exact_parameter_identity_and_rng_restore(state):
    before = torch.random.get_rng_state().clone()
    native, compact = qualification_models_v1("age", state)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert native.state_dict().keys() == compact.state_dict().keys()
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value, compact.state_dict()[name], rtol=0, atol=0)
    assert native.input_dim == 47 and native.num_scales == 3
    assert native.hidden_dim == 32 and native.latent_dim == 8
    assert not next(native.parameters()).is_cuda


def test_isolated_memory_blocked_before_cuda(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(module, "recheck_admission_v1", lambda a: None)
    called = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: called.append(True))
    with pytest.raises(ValueError, match="profiling"):
        cuda_isolated_memory_v1({}, SimpleNamespace(binding={}), {}, {}, {})
    assert not called


def active_handmade_admission(tmp_path):
    def write(name, value):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return record(p)
    key = "src/arggnn/anchor_compact_actual_bank_v1.py"
    adapter = tmp_path / key
    adapter.parent.mkdir(parents=True)
    adapter.write_text("handmade synthetic source binding only")
    original = write("original.json", {"source_sha256":{}})
    observed = write("observed.json", {"truth_or_oracle_read":False})
    data = manifest()
    data.update(config_sha256=original["sha256"], observed_manifest_sha256=observed["sha256"],
                source_sha256={})
    prepared = write("prepared.json", data)
    cfg = write("config.json", dict(kind="anchor_compact_actual_bank_protocol_v1",
        status="source_only_actual_execution_not_admitted", fit_allowed=False, truth_allowed=False,
        source_sha256={key:record(adapter)["sha256"]}, original_preparation_source_sha256={},
        original_candidate_config=original, observed_manifest=observed, prepared_manifest=prepared))
    review = write("review.json", dict(verdict="PASS", adapter_sha256=record(adapter)["sha256"],
                                      protocol_sha256=cfg["sha256"], actual_execution_admitted=False))
    evidence = write("root_other_evidence.json", {"handmade_dummy":True})
    receipt = write("root.json", dict(kind="root_compact_actual_bank_qualification_admission_v1",
        status="active_root_admission", authorizer="root_agent", config=cfg,
        project_root=str(tmp_path.resolve()), truth_allowed=False, fit_allowed=False,
        devices=["cpu_float64","cuda_float32"], stages=["parity","profile_after_parity_pass"],
        expected_process_title="AI Auto Research Pipeline Test", independent_review=review,
        runtime_admission=evidence, storage_admission=evidence, launcher_admission=evidence,
        selected_train_tiles=select_train_tiles_v1(data)))
    return cfg, receipt


def test_active_root_binding_passes_only_exact_synthetic_metadata(tmp_path):
    cfg, receipt = active_handmade_admission(tmp_path)
    admission = admit_actual_qualification_v1(cfg, receipt, root=tmp_path)
    assert len(admission.selected) == 3
    assert not admission.config["truth_allowed"]
    module.recheck_admission_v1(admission)


@pytest.mark.parametrize("changed", ["source", "review", "selection", "runtime"])
def test_active_root_binding_detects_synthetic_provenance_change(tmp_path, changed):
    cfg, receipt = active_handmade_admission(tmp_path)
    admission = admit_actual_qualification_v1(cfg, receipt, root=tmp_path)
    if changed == "source":
        (tmp_path / "src/arggnn/anchor_compact_actual_bank_v1.py").write_text("changed")
    elif changed == "review":
        (tmp_path / "review.json").write_text('{"verdict":"BLOCKED"}')
    elif changed == "runtime":
        (tmp_path / "root_other_evidence.json").write_text("changed runtime")
    else:
        p = tmp_path / "root.json"
        data = json.loads(p.read_text())
        data["selected_train_tiles"][0]["parent_id"] = "bio_paths_pilot_val_000"
        p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.recheck_admission_v1(admission)


@pytest.mark.parametrize("coupling", ["scalar", "diagonal", "gain_matched_diagonal"])
def test_revision_v2_nonfull_view_full_model_rejects_before_encoder_or_action(coupling, monkeypatch):
    native, _, bank, _ = fixture()
    view = derive_compact_actual_bank_v1(bank, "age", coupling)
    calls = []
    def blocked(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Mismatch reached encoder or action")
    monkeypatch.setattr(native.encoder, "forward", blocked)
    monkeypatch.setattr(module, "anchor_kernel_action", blocked)
    monkeypatch.setattr(module, "anchor_spectral_compact_action_v1", blocked)
    with pytest.raises(ValueError, match="geometry or coupling contract"):
        compare_action_VJPs_v1(native, bank, view)
    assert not calls


@pytest.mark.parametrize("change", ["view_label", "model_label", "missing_native_geometry",
                                   "unknown_matching_coupling", "scale", "anchors", "features"])
def test_revision_v2_direct_identity_rejects_before_meaningful_work(change, monkeypatch):
    native, _, bank, view = fixture()
    if change == "view_label":
        view = replace(view, geometry_name="plain")
    elif change == "model_label":
        native.geometry = "plain"
    elif change == "missing_native_geometry":
        bank = replace(bank, operators={"plain": bank.operators["plain"]})
    elif change == "unknown_matching_coupling":
        native.coupling = "invented"
        view = replace(view, coupling="invented")
    elif change == "scale":
        view = replace(view, num_scales=7)
    elif change == "anchors":
        native.num_anchors = 8
    else:
        view = replace(view, leaf_x=view.leaf_x[:, :, :3])
    calls = []
    def blocked(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Mismatch reached meaningful work")
    monkeypatch.setattr(native.encoder, "forward", blocked)
    monkeypatch.setattr(module, "anchor_kernel_action", blocked)
    monkeypatch.setattr(module, "anchor_spectral_compact_action_v1", blocked)
    with pytest.raises(ValueError, match="contract"):
        compare_action_VJPs_v1(native, bank, view)
    assert not calls


@pytest.mark.parametrize("change", ["native_view_coupling", "compact_model_coupling",
                                   "native_view_geometry", "compact_model_geometry"])
def test_revision_v2_full_step_identity_rejects_before_copy_or_encoder(change, monkeypatch):
    native, compact, bank, view = fixture()
    if change == "native_view_coupling":
        view = replace(view, coupling="scalar")
    elif change == "compact_model_coupling":
        compact.coupling = "scalar"
    elif change == "native_view_geometry":
        view = replace(view, geometry_name="plain")
    else:
        compact.geometry = "plain"
    calls = []
    def blocked(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Mismatch reached copy or encoder")
    monkeypatch.setattr(native.encoder, "forward", blocked)
    monkeypatch.setattr(compact.encoder, "forward", blocked)
    monkeypatch.setattr(module.copy, "deepcopy", blocked)
    with pytest.raises(ValueError, match="contract"):
        compare_model_step_v1(native, compact, bank, view)
    assert not calls


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("coupling", ["full", "scalar", "diagonal", "gain_matched_diagonal"])
def test_revision_v2_positive_actual_dispatch_matches_receipt(dtype, coupling, monkeypatch):
    native, _, bank, view = fixture(dtype, coupling)
    seen = []
    original_native = module.anchor_kernel_action
    original_compact = module.anchor_spectral_compact_action_v1
    def trace_native(*args, **kwargs):
        seen.append(("native", kwargs["coupling"]))
        return original_native(*args, **kwargs)
    def trace_compact(*args, **kwargs):
        seen.append(("compact", kwargs["coupling"]))
        return original_compact(*args, **kwargs)
    monkeypatch.setattr(module, "anchor_kernel_action", trace_native)
    monkeypatch.setattr(module, "anchor_spectral_compact_action_v1", trace_compact)
    result = compare_action_VJPs_v1(native, bank, view)
    assert result["status"] == "PASS", result
    assert seen == [("native", coupling), ("compact", coupling)]
    identity = result["cell_identity"]
    assert identity["coupling"] == identity["native_coupling"] == identity["compact_coupling"] == coupling
    assert identity["geometry"] == identity["native_geometry"] == identity["compact_geometry"] == "age"
    assert identity["dtype"] == str(dtype) and identity["device"] == "cpu"
    assert (identity["Q"], identity["R"], identity["anchors"], identity["scales"]) == (3, 2, 3, 2)


def control_builder_bank(dtype):
    _, _, prepared, _ = fixture()
    bank = prepared.bank
    leaf = torch.sin(torch.arange(3*5*47, dtype=torch.float64).reshape(3, 5, 47)/17)
    bank = replace(bank, leaf_x=leaf, feature_columns=tuple(c for c in range(48) if c != 7))
    op = prepared.operators["age"]
    arrays = [torch.cat((getattr(op, name), getattr(op, name)[:, :1]), dim=1)
              for name in ("cqq", "cqr", "eigenvalues", "eigenvectors")]
    operators = AnchorOperators(*arrays)
    return AnchorKernelBank(bank, _layout(bank, 32), {"plain": operators, "age": operators}, 3).to("cpu", dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("coupling", ["scalar", "diagonal", "gain_matched_diagonal"])
def test_revision_v2_explicit_constructor_controls_match_seeded_weights_and_pass(dtype, coupling):
    before_rng = torch.random.get_rng_state().clone()
    native, compact = qualification_direct_control_models_v2("age", coupling)
    full, _ = qualification_models_v1("age", "initial")
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert native.coupling == compact.coupling == coupling
    assert native.geometry == compact.geometry == "age"
    for name, value in full.state_dict().items():
        torch.testing.assert_close(native.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(compact.state_dict()[name], value, rtol=0, atol=0)
    bank = control_builder_bank(dtype)
    view = derive_compact_actual_bank_v1(bank, "age", coupling).to("cpu", dtype)
    result = compare_action_VJPs_v1(native.to(dtype=dtype), bank, view)
    assert result["status"] == "PASS", result
    assert result["cell_identity"]["coupling"] == coupling
    assert result["cell_identity"]["anchors"] == 32 and result["cell_identity"]["scales"] == 3


@pytest.mark.parametrize("coupling", ["full", "invented"])
def test_revision_v2_nonfull_constructor_rejects_implicit_or_invalid_control(coupling):
    with pytest.raises(ValueError, match="explicit"):
        qualification_direct_control_models_v2("age", coupling)
