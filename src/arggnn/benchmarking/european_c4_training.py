"""Separate current four-head TreeLoom fit on the shared AP200Mb parents.

Reuse the reviewed30-pass loop and original32325-parameter four-class model.
The separately frozen AP-schema registry supplies this C4 support's exact
sampled coordinates/sample/record IDs; only its class strings use a declared C4
view. Its separately prepared C4TRAIN labels are checked against original
source-class truth by their producer. This does not assert shared sampled draws
with the unchanged native-six support.
No native6 head weights, probabilities or historical checkpoints are converted.
"""
from __future__ import annotations
import copy
import uuid
import zipfile
from . import contracts as c
from . import treeloom_fresh as fresh
from . import european_training as original
from . import european_treeloom as observed

PROTOCOL = "ancestralpaths_paper200mb_c4"
CLASSES = ("Ana","CHG","WHG","EHG")
_BOUND = None


def _need(ok,message):
    if not ok:
        raise c.ContractError(message)


def requirements():
    result = original.requirements()
    return dict(result,source_refs=[*result["source_refs"],c.file_ref(__file__)],
        protocol_id=PROTOCOL,output_classes=list(CLASSES),model_parameters=32325,
        coordinate_registry_view="provided_AP_schema_sampled_record_sample_coordinate_IDs_preserved_declared_C4_endpoint",
        target_schema="european_release_C4_sampled_targets_v1",
        separate_four_head_fit=True,native_six_head_or_probability_conversion=False)


def _parent_plan(ref,protocol_id):
    _need(protocol_id == PROTOCOL,"Only separately trained current AP200Mb C4 condition required")
    # Sampling quotas and independent parent identities are shared with AP6.
    # They do not dictate this model's separately declared four-class endpoint.
    return original._parent_plan(ref,"ancestralpaths_paper200mb_native6")


def _sampled_records(ref,**kwargs):
    import numpy as np
    records = observed._sampled_records(ref,**kwargs)
    _need(records["class_order"].tolist() == list(observed.NATIVE_SIX_CLASSES),
          "Actual AP-schema sampled registry with canonical native-six ID declaration required")
    return dict(records,class_order=np.asarray(CLASSES))


def _targets(ref,parent,bundle,model,*,max_metadata_bytes):
    import numpy as np
    from .targets import SampledFocalTargets
    metadata = observed._json_ref(ref,max_metadata_bytes)
    _need(metadata.get("schema") == "european_release_C4_sampled_targets_v1"
          and metadata.get("parent_id") == parent and metadata.get("split") == "train"
          and metadata.get("registry_ref") == bundle.sampled_registry_ref
          and metadata.get("source_class_order") == list(CLASSES)
          and tuple(model.output_classes) == CLASSES,
          "Same-parent source-verified canonical C4TRAIN labels required after observed barrier")
    path = observed._bound(metadata["arrays_ref"])
    with zipfile.ZipFile(path) as archive:
        _need(archive.namelist() == ["source_labels_0based.npy"]
              and sum(item.file_size for item in archive.infolist()) <= 2*2**20,
              "Only bounded canonical C4TRAIN label vector required")
    with np.load(path,allow_pickle=False) as archive:
        labels = archive["source_labels_0based"].copy()
    n = bundle.focal_hold.numel()
    _need(labels.dtype == np.int64 and labels.shape == (n,) and metadata.get("rows") == n
          and ((labels >= 0) & (labels < 4)).all(),
          "Every actual sampled C4TRAIN example must keep its canonical label")
    query_lookup = {int(value):i for i,value in enumerate(bundle.bank.query_column)}
    return SampledFocalTargets(readout_rows=tuple(map(int,bundle.focal_hold.tolist())),
        query_columns=tuple(query_lookup[int(value)] for value in bundle.focal_sample_column),
        labels=tuple(map(int,labels)),class_order=CLASSES,parent_id=parent,split="train",
        sampled_registry_ref=bundle.sampled_registry_ref)


def _bound():
    global _BOUND
    if _BOUND is None:
        tag = "current_C4_" + uuid.uuid4().hex
        # Reexecute verified maintained source in isolated namespaces. Only
        # observed registry class-view and target/parent-plan I/O bindings differ.
        view = fresh._private_module(observed,tag+"_observed",
                                     {"_sampled_records":_sampled_records})
        _BOUND = fresh._private_module(original,tag+"_training",{
            "european":view,"_parent_plan":_parent_plan,"_training_targets":_targets,
            "requirements":requirements})
    return _BOUND


def build_fit_spec(**kwargs):
    kwargs = dict(kwargs)
    kwargs.setdefault("protocol_id",PROTOCOL)
    return _bound().build_fit_spec(**kwargs)


def build_prediction_spec(**kwargs):
    kwargs = dict(kwargs)
    kwargs.setdefault("protocol_id",PROTOCOL)
    return _bound().build_prediction_spec(**kwargs)


def run_fit(spec,*,resource_check):
    _need(spec["protocol_id"] == PROTOCOL,"Genuine separately trained four-head condition required")
    return _bound().run_fit(spec,resource_check=resource_check)


def run_predictions(spec,*,resource_check):
    _need(spec["protocol_id"] == PROTOCOL,"Current four-head checkpoint only")
    return _bound().run_predictions(spec,resource_check=resource_check)


def fit_factory(**kwargs):
    spec = build_fit_spec(**kwargs)
    return dict(module="arggnn.benchmarking.european_c4_training",function="run_fit",
        spec=spec,python=".runtime/env/bin/python",source_refs=spec["requirements"]["source_refs"])


def prediction_factory(**kwargs):
    spec = build_prediction_spec(**kwargs)
    return dict(module="arggnn.benchmarking.european_c4_training",function="run_predictions",
        spec=spec,python=".runtime/env/bin/python",source_refs=spec["requirements"]["source_refs"])
