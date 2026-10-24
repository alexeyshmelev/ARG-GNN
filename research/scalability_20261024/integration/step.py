"""Isolated selected full-parent update; frozen optimizer/history loop preserved."""
import torch
from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import (
    decode_parent_backward, geometry_from_spans, need)
from research.repairs.benchmark_recovery_20261020.training.progress import StageProgress
from .executors import build_executor


def train_step(model,bundle,targets,optimizer,metadata,*,resource_check,
               frozen_module,status_path,holds_per_block=128,
               haplotypes_per_gru=64,history_holds_per_window=1024,profile=None,
               executor_kind="integrated",layout_cache=None):
    need(type(model) is frozen_module.AmericanTreeLoomC3FullV1
         and type(bundle) is frozen_module.AmericanTreeLoomWholeParentV1
         and bundle.split=="train" and bundle.parent_id in frozen_module.TRAIN,
         "Authenticated complete TRAIN parent and unchanged model required")
    need(isinstance(targets,frozen_module.TrainFocalTargetVectorV1)
         and targets.parent_id==bundle.parent_id
         and targets.focal_registry_ref==bundle.focal_registry_ref
         and targets.focal_identity_sha256==bundle.focal_identity_sha256,
         "Exactly the sealed native focal targets required")
    need(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups)==1,
         "Unchanged AdamW required")
    parameters = list(model.parameters())
    group = optimizer.param_groups[0]
    need(group["lr"]==.001 and group["weight_decay"]==.0001
         and group["betas"]==(.9,.999) and group["eps"]==1e-8
         and group["maximize"] is False and group["amsgrad"] is False
         and len(parameters)==len(group["params"])==49
         and {id(p) for p in parameters}=={id(p) for p in group["params"]}
         and all(p.requires_grad and p.dtype==torch.float32 and p.device.type=="cuda"
                 for p in parameters),
         "All original49FP32 tensors and fixed optimizer settings")
    need(targets.labels_cpu.shape==(10000,) and targets.labels_cpu.dtype==torch.long
         and targets.labels_cpu.device.type=="cpu"
         and bool(((targets.labels_cpu>=0)&(targets.labels_cpu<3)).all()),
         "All10000 original TRAIN labels required")
    own_profile = profile is None
    profile = profile or StageProgress(status_path,parent_id=bundle.parent_id)
    def tick(label):
        resource_check(label)
        profile.mark(label)
    executor = None
    try:
        profile.mark("parent_begin",native_holds=bundle.native_holds,
            holds_per_block=holds_per_block,haplotypes_per_gru=haplotypes_per_gru,
            executor_kind=executor_kind)
        profile.flush()
        with profile.stage("bundle_validate"):
            bundle.validate()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with profile.stage("layout_constructor"):
            executor = build_executor(executor_kind,model,bundle,metadata,
                execution="staged",layout_cache=layout_cache,
                holds_per_block=holds_per_block,haplotypes_per_gru=haplotypes_per_gru,
                resource_check=tick,history_holds_per_window=history_holds_per_window,
                profile=profile,frozen_module=frozen_module)
        with torch.autocast(device_type="cuda",enabled=False):
            with profile.stage("complete_parent_context_forward"):
                context = executor.context_forward()
            with profile.stage("complete_parent_decoder"):
                loss,context_gradient,decoder_receipt = decode_parent_backward(
                    model,context,geometry_from_spans(bundle.bank.spans),
                    bundle.focal_hold,bundle.focal_sample_column,targets.labels_cpu,
                    tile_haplotypes=haplotypes_per_gru,profile=profile,resource_check=tick)
        tick("bounded_complete_parent_backward")
        with profile.stage("complete_parent_trunk_backward"):
            context.backward(context_gradient)
        del context,context_gradient
        with profile.stage("gradient_validation_and_clip"):
            gradients = [p.grad for p in parameters]
            need(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
                 and any(bool((g!=0).any()) for g in gradients),
                 "All49finite connected gradients and a nonzero global gradient required")
            norm = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            need(bool(torch.isfinite(norm)),"Finite original clip norm required")
        tick("bounded_complete_parent_optimizer")
        with profile.stage("complete_parent_optimizer"):
            optimizer.step()
        need(all(bool(torch.isfinite(p).all()) for p in parameters),
             "Finite updated original49tensors required")
        tick("bounded_complete_parent_end")
        profile.mark("parent_complete",loss=float(loss),gradient_norm=float(norm),
                     optimizer_updates_in_call=1)
        result = dict(parent_id=bundle.parent_id,native_holds=bundle.native_holds,
            original_sites=bundle.original_sites,native_focals=10000,haplotypes=1600,
            sequence_length_bp=30000000,loss=float(loss),gradient_norm=float(norm),
            one_complete_parent_optimizer_update=True,detached_history=False,
            artificial_block_boundaries=False,ancestry_accuracy=None,
            execution="staged_full_parent_backward_v1",
            executor_kind=executor_kind,decoder=decoder_receipt,
            holds_per_block=holds_per_block,stage_seconds={name:row["total_seconds"]
                for name,row in profile.snapshot()["stage_timings"].items()
                if name in ("layout_constructor","complete_parent_context_forward",
                            "complete_parent_decoder","complete_parent_trunk_backward")})
        if layout_cache is not None:
            result["layout_cache"] = layout_cache.snapshot()
        return result
    finally:
        if executor is not None and hasattr(executor,"release_layout"):
            executor.release_layout()
        if own_profile:
            profile.close()
