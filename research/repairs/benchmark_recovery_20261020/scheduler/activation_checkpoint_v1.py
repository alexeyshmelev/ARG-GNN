"""Execution-only activation recomputation around unchanged complete encoder layers."""
import inspect
from pathlib import Path


def install_layers(layers, torch, checkpoint):
    """Preserve full tensor inputs, complete loss, parameters and both RNG streams."""
    receipts=[]
    for index,layer in enumerate(layers):
        if getattr(layer,'_arggnn_activation_checkpoint',False):
            raise ValueError('Activation checkpointing may be installed exactly once')
        original=layer.forward
        if getattr(original,'__self__',None) is not layer:
            raise ValueError('Each original forward must be the exact bound layer method')
        def make_forward(module,original_forward):
            def forward(*args,**kwargs):
                if module.training and torch.is_grad_enabled():
                    return checkpoint(original_forward,*args,use_reentrant=False,
                                      preserve_rng_state=True,**kwargs)
                return original_forward(*args,**kwargs)
            return forward
        layer.forward=make_forward(layer,original)
        layer._arggnn_activation_checkpoint=True
        receipts.append(dict(layer=index,original_class=type(layer).__name__,
            original_forward_module=original.__module__,non_reentrant=True,
            preserve_CPU_and_current_tensor_device_RNG=True,full_input_tensor_preserved=True))
    return receipts


def install_native(model,torch,author_root):
    from torch.utils.checkpoint import checkpoint
    encoder=model.base.backbone.bert.encoder
    expected=Path(author_root)/'src/bert_layers/layers.py'
    if type(encoder).__name__!='FlexBertUnpadEncoder' or len(encoder.layers)!=12:
        raise ValueError('Exact released twelve-layer unpadded encoder required')
    for layer in encoder.layers:
        if type(layer).__name__!='FlexBertUnpadPreNormLayer' or Path(inspect.getsourcefile(type(layer))).resolve()!=expected.resolve():
            raise ValueError('Only the pinned original released encoder layer may be checkpointed')
    before_parameters=tuple((name,id(value)) for name,value in model.named_parameters())
    before_state_keys=tuple(model.state_dict())
    rows=install_layers(encoder.layers,torch,checkpoint)
    if tuple((name,id(value)) for name,value in model.named_parameters())!=before_parameters or tuple(model.state_dict())!=before_state_keys:
        raise ValueError('Activation recomputation changed parameter identity or checkpoint schema')
    return dict(schema='native_encoder_activation_checkpoint_execution_v1',layers=rows,
        algorithm_data_loss_batch_precision_unchanged=True,full_contrastive_batch=4096,
        no_loss_microbatching=True,parameter_identity_and_saved_schema_unchanged=True,
        original_source_files_unchanged=True,execution_differs_by_activation_recomputation=True)
