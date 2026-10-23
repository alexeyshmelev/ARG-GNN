"""Explicit in-memory adaptations of unchanged author code; no source rewriting."""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
import types

def inverse_frequency(labels):
    counts = Counter(int(x) for x in labels)
    if set(counts) != {0, 1, 2} or min(counts.values()) <= 0:
        raise ValueError("All three known source-reference classes must occur in TRAIN")
    total = sum(counts.values())
    return {c: total / (len(counts) * n) for c, n in sorted(counts.items())}

def crop_batch(batch):
    """Discard trailing padding only. Every real token and selected loss target stays."""
    width = int(batch["attention_mask"].sum(dim=1).max())
    for key in ("input_ids", "attention_mask", "token_type_ids", "labels"):
        value = batch.get(key)
        if value is not None and value.ndim == 2:
            batch[key] = value[:, :width].contiguous()
    return batch

class LiteralMaskCollator:
    def __init__(self, base, mask_id):
        self.base, self.mask_id = base, int(mask_id)
    def __call__(self, rows):
        batch = self.base(rows)
        selected = batch["labels"] != -100
        batch["input_ids"][selected] = self.mask_id
        return crop_batch(batch)

def extant_head_forward(self, hidden_states, pool=True):
    output = hidden_states[:, 1] if pool else hidden_states
    return self.drop(self.norm(self.act(self.dense(output))))

def install_extant_head(backbone):
    """Keep native projection, activation, norm and dropout and their state keys."""
    backbone.head.forward = types.MethodType(extant_head_forward, backbone.head)

def weighted_supcon(z, labels, temperature, class_weights, similarity="dot", normalize=True):
    """Author supervised loss with vectorized inverse-frequency anchor weighting."""
    import torch
    from torch.nn import functional as F
    if temperature <= 0 or similarity not in ("dot", "cosine"):
        raise ValueError("Positive temperature and native similarity required")
    if normalize or similarity == "cosine":
        z = F.normalize(z, p=2, dim=-1)
    sim = (z @ z.T) / temperature
    if len(z) <= 1:
        return sim.sum() * 0, sim.detach()
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    counts = positive.sum(1)
    log_prob = sim - torch.logsumexp(sim.masked_fill(eye, -torch.inf), dim=1, keepdim=True)
    per_anchor = -(log_prob * positive).sum(1) / counts.clamp_min(1)
    valid = counts > 0
    if not bool(valid.any()):
        return sim.sum() * 0, sim.detach()
    table = torch.tensor([class_weights[c] for c in range(3)], device=z.device, dtype=z.dtype)
    weights = table[labels[valid]]
    return (per_anchor[valid] * weights).sum() / weights.sum(), sim.detach()

def paper_forward(self, batch):
    states = self.backbone.bert(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"],
                               token_type_ids=batch.get("token_type_ids"))
    pooled = self.backbone.head(states)
    loss, sim = weighted_supcon(pooled, batch["labels"], self.temperature,
                               self.class_weights, self.similarity, self.normalize)
    return {"loss": loss, "sim": sim}

@contextmanager
def adaptations(module, cfg, *, phase, class_weights=None):
    """Restore imported globals after this one training invocation."""
    changed = []
    def replace(obj, name, value):
        changed.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)
    try:
        if phase == "mlm":
            owner = module.arg_data_module
            original = owner.build_arg_dataloader
            def loader(config=None, tokenizer=None, device_batch_size=None,
                       device_microbatch_size=None, **kwargs):
                config = config if config is not None else kwargs.pop("cfg")
                result = original(config, tokenizer, device_batch_size, device_microbatch_size)
                result.collate_fn = LiteralMaskCollator(result.collate_fn, tokenizer.mask_token_id)
                return result
            replace(owner, "build_arg_dataloader", loader)
        else:
            if class_weights is None:
                raise ValueError("Actual TRAIN class weights required; no uniform fallback")
            original = module.build_contrastive_model
            def builder(model_cfg):
                from omegaconf import OmegaConf
                local = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
                local.contrastive.use_class_weights = False
                model = original(local)
                model.base.class_weights = class_weights
                model.base.temperature = 0.05
                install_extant_head(model.base.backbone)
                model.base.forward = types.MethodType(paper_forward, model.base)
                return model
            replace(module, "build_contrastive_model", builder)
            original_collate = module.supervised_contrastive_collate_fn
            def collate(rows):
                return crop_batch(original_collate(rows))
            replace(module, "supervised_contrastive_collate_fn", collate)
        yield
    finally:
        for obj, name, previous in reversed(changed):
            setattr(obj, name, previous)
