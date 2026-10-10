"""Data reuse must bind the source version without changing data definitions."""
from copy import deepcopy
import pytest
from arggnn.train_study import verify_reused_dataset_protocol


def test_model_study_can_reuse_exact_frozen_parent_protocol():
    source={"benchmark":"inferred","master_seed":1,"simulation":{"length":100},
            "splits":{"train":16,"val":4,"test":0},"windows":8,"models":["independent"],
            "hidden_dim":32,"learning_rate":.001,"reference_adapter":{"time_column":6}}
    manifest={"protocol":source,"protocol_sha256":"a"*64}
    study=deepcopy(source)
    study.update(dataset_source_protocol_sha256="a"*64,models=["gst_ca_span"],learning_rate=.002)
    verify_reused_dataset_protocol(study,manifest)
    for field,change in [("master_seed",2),("windows",4),("simulation",{"length":200}),
                         ("benchmark","oracle"),("reference_adapter",{"time_column":5})]:
        wrong=deepcopy(study)
        wrong[field]=change
        with pytest.raises(RuntimeError,match="Reused dataset field changed"):
            verify_reused_dataset_protocol(wrong,manifest)
    study["dataset_source_protocol_sha256"]="b"*64
    with pytest.raises(RuntimeError,match="source protocol hash"):
        verify_reused_dataset_protocol(study,manifest)
