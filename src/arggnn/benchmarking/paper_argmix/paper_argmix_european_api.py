"""Plaintext European prediction API for AP952 and ARGMix3222.

Root owns saved-checkpoint authentication, actual GPU lease, observations,
output seals and real outer waits. Only physical GPU3,4,5 are campaign eligible.
"""
from __future__ import annotations

import torch
from arggnn.benchmarking import treeloom
from arggnn.benchmarking import european_treeloom as european
from .paper_argmix_treeloom_executor import PaperEuropeanBoundedExecutionV1
from .paper_argmix_gpu import validate_gpu_binding


def model_binding(model,bundle,metadata,resource_check,*,gpu_uuid=None,gpu_uuid_map=None):
    binding=validate_gpu_binding(gpu_uuid=gpu_uuid,gpu_uuid_map=gpu_uuid_map)
    implementation=treeloom._european_module()
    parameters=list(model.parameters())
    if not (type(model) is implementation.FrozenEuropeanTreeLoomV1 and
            type(bundle) is european.EuropeanWholeParent and model.condition==bundle.original.condition and
            callable(resource_check) and torch.cuda.device_count()==1 and
            all(p.dtype==torch.float32 and p.device.type=='cuda' and p.device.index in (None,0) for p in parameters) and
            metadata.get('parent_id')==bundle.parent_id and metadata.get('sampled_registry_ref')==bundle.sampled_registry_ref and
            metadata.get('focal_identity_sha256')==bundle.focal_identity_sha256 and metadata.get('private_truth_opened') is False):
        raise ValueError('Exact complete FP32 European model/bundle on campaign logical cuda0 required')
    bundle.validate();resource_check('European_complete_parent_campaign_GPU')
    return binding


def predict_logits(model,bundle,metadata,*,resource_check,gpu_uuid=None,gpu_uuid_map=None):
    """Return[N registered rows, native classes] in unchanged original order."""
    model_binding(model,bundle,metadata,resource_check,gpu_uuid=gpu_uuid,gpu_uuid_map=gpu_uuid_map)
    if bundle.split not in ('val_prediction','evaluation_prediction','iid_prediction'):
        raise ValueError('Held-out observations only for European prediction')
    model.eval()
    with torch.no_grad():
        return PaperEuropeanBoundedExecutionV1(model,bundle,metadata,
            resource_check=resource_check,holds_per_block=16,haplotypes_per_gru=8,
            history_holds_per_window=1024).forward()
