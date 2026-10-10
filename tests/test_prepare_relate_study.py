"""Frozen source and independent inference identities for the paired panel."""
from copy import deepcopy
import json

import pytest

from arggnn.prepare_relate_study import inference_seed, validate_protocol
from arggnn.resources import ROOT


def protocol():
    path = ROOT / "configs" / "inferred_common_input_pilot_v3.json"
    if not path.exists():
        pytest.skip("versioned inferred pilot protocol unavailable")
    return json.loads(path.read_text())


def test_inference_seeds_are_independent_deterministic_signed_integers():
    settings = protocol()
    seeds = [inference_seed(settings["inference"]["master_seed"], split, replicate)
             for split in ("train", "val") for replicate in range(settings["splits"][split])]
    assert len(set(seeds)) == 20
    assert all(1 <= seed <= 2**31-1 for seed in seeds)
    assert inference_seed(20261013, "train", 0) == 683243869
    assert inference_seed(20261013, "train", 0) != inference_seed(20261013, "val", 0)
    assert inference_seed(20261013, "train", 0) != inference_seed(20261014, "train", 0)
    with pytest.raises(ValueError, match="train/validation"):
        inference_seed(20261013, "test", 0)


def test_paired_protocol_preserves_twenty_train_validation_source_identities():
    _, records = validate_protocol(protocol())
    assert [(r["split"], r["replicate"]) for r in records] == [
        *[("train", i) for i in range(16)], *[("val", i) for i in range(4)]]


def test_source_hash_changes_and_common_input_changes_are_rejected():
    settings = deepcopy(protocol())
    settings["source_parent"]["manifest_sha256"] = "0"*64
    with pytest.raises(RuntimeError, match="integrity"):
        validate_protocol(settings)
    settings = deepcopy(protocol())
    settings["reference_adapter"]["kernel_scales"] = [.1]
    with pytest.raises(ValueError, match="reference_adapter"):
        validate_protocol(settings)
    settings = deepcopy(protocol())
    settings["splits"]["test"] = 1
    with pytest.raises(ValueError, match="splits"):
        validate_protocol(settings)
