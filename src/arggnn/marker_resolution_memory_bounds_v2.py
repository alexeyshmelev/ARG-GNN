"""H044 resource-only erratum for frozen v1 96-channel H042 exported state.

The original GRU is bidirectional:32 context+2*32 history=96 channels.
Forward science, checkpoint/layout, decoder parameters and guards are unchanged.
"""
from .marker_resolution_readout_v1 import logical_bounds_v1

H042_CONTEXT_CHANNELS=32
H042_HISTORY_CHANNELS=64
H042_EXPORTED_STATE_CHANNELS=96

def logical_bounds_v2(s,t,p,*,h=952,q=686,genotype_bytes=1,query_block=64):
    value=logical_bounds_v1(s,t,p,h=h,q=q,genotype_bytes=genotype_bytes,query_block=query_block)
    value.update(exported_H042_state_bytes=4*t*h*H042_EXPORTED_STATE_CHANNELS,
        head_checkpoint_selected_state_and_feature_bytes=4*p*q*(H042_EXPORTED_STATE_CHANNELS+4*6),
        H042_context_channels=32,H042_bidirectional_history_channels=64,H042_exported_state_channels=96,
        resource_erratum="Frozen v1 counted64; actual unchanged forward state is96.",
        frozen_v1_forward_science_unchanged=True)
    return value

def constructor_interface_v2(model):
    """Bounded CPU constructor interface check; no observations or forward pass."""
    from torch import nn
    model=getattr(model,"backbone",model)
    gru=model.genomic_gru
    first=next(layer for layer in model.classifier.modules() if isinstance(layer,nn.Linear))
    context=32;history=gru.hidden_size*(2 if gru.bidirectional else 1)
    if not (gru.input_size==35 and gru.hidden_size==32 and gru.bidirectional
            and context+history==first.in_features==96):
        raise ValueError("Unchanged H042 bidirectional GRU/classifier96 interface required")
    return dict(context_channels=context,GRU_hidden_size=gru.hidden_size,
        GRU_bidirectional=gru.bidirectional,history_channels=history,
        exported_state_channels=context+history,classifier_in_features=first.in_features,
        observation_or_forward_execution=False)
