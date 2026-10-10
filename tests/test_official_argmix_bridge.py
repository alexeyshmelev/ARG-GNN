"""Actual unchanged author component smoke; no native extraction or real data."""
from pathlib import Path
import shutil
import sys

import pytest
import torch

from arggnn import official_argmix_bridge as bridge


@pytest.fixture
def copied_source(tmp_path):
    destination = tmp_path / "unchanged_author_copy"
    shutil.copytree(bridge.DEFAULT_AUTHOR_ROOT, destination)
    return destination


def test_complete_author_inventory_and_metadata_preflight():
    source = bridge.verify_author_source()
    report = bridge.inspect_official_source()
    assert len(source.file_hashes) == 86 and report["author_tree_sha256"] == bridge.AUTHOR_TREE_SHA256
    assert report["upstream_code_execution"] is False
    assert report["paper_v1_reproduction_established"] is False
    assert report["scientific_readiness"] is False


@pytest.mark.parametrize("mutation", ["bytes", "missing", "extra", "symlink"])
def test_complete_source_drift_rejected_before_native_import(copied_source, monkeypatch, mutation):
    source = copied_source / "src/tmrca_attn.py"
    if mutation == "bytes":
        source.write_text(source.read_text() + "\n# Explicit source-drift fixture\n")
    elif mutation == "missing":
        source.unlink()
    elif mutation == "extra":
        (copied_source / "unexpected.py").write_text("raise AssertionError('Must never execute')")
    else:
        source.unlink()
        source.symlink_to(bridge.DEFAULT_AUTHOR_ROOT / "src/tmrca_attn.py")
    def forbidden(*args, **kwargs):
        raise AssertionError("Native import before source admission")
    monkeypatch.setattr(bridge.importlib.util, "spec_from_file_location", forbidden)
    with pytest.raises(ValueError):
        bridge.native_model_smoke(copied_source)


def test_external_and_traversal_author_paths_reject():
    for path in [Path("/tmp/unadmitted_author"), ".runtime/author_sources/../argmix"]:
        with pytest.raises(ValueError):
            bridge.verify_author_source(path)


def test_immutable_source_snapshot_detects_later_copy_drift(copied_source):
    source = bridge.verify_author_source(copied_source)
    (copied_source / "LICENSE").write_text("Explicit later source drift")
    with pytest.raises(ValueError):
        source.check()


def test_native_smoke_requires_explicit_hidden_cuda(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="explicitly hidden CUDA"):
        bridge.native_model_smoke()


def test_actual_stock_native_model_and_torchscript_cpu_preserve_source_and_rng(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    def forbidden(*args, **kwargs):
        raise AssertionError("No CUDA API may be used for native CPU smoke")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    before = bridge.verify_author_source()
    rng = torch.get_rng_state().clone()
    bytecode = sys.dont_write_bytecode
    result = bridge.native_model_smoke()
    assert result["native_class"] == "TmrcaAttn"
    assert result["native_constructor"]["d_model"] == 128
    assert result["native_constructor"]["num_layer"] == 4
    assert result["input_ids_shape"] == [2, 16]
    assert result["raw_logits_shape"] == [2, 3]
    assert result["raw_logits_dtype"] == "torch.float32"
    assert result["finite"] and result["torchscript_exact_match"]
    assert result["untrained_random_weights"] and not result["optimization_performed"]
    assert not result["cuda_execution_performed"] and not result["full_native_pipeline_executed"]
    assert not result["real_data_or_labels_accessed"] and not result["scientific_readiness"]
    assert bridge.verify_author_source().file_hashes == before.file_hashes
    assert torch.equal(torch.get_rng_state(), rng)
    assert sys.dont_write_bytecode == bytecode
    assert not (bridge.DEFAULT_AUTHOR_ROOT / "src/__pycache__").exists()
