"""H050 reuse of reviewed H049 preparation authentication; no guard/runtime calls.

The preparation was registered under H049. H050's experimental preregistration
is separate. The explicit preparation view retains its genuine H049 identity.
"""
from research.reviews.H049_order2_cut_memory_gpu_exec_source_20261015_v2 import context_factory_v2 as framework


def authenticate_actual_cut_preparation_v1(reader,spec,cfg):
    require=framework.require
    require(cfg.get("hypothesis_id")=="H050"
        and cfg.get("cut_preparation_registration_ref")
        and cfg["cut_preparation_registration_ref"] != cfg["registration_ref"],
        "Separate H049 preparation and H050 experiment registration refs")
    preparation_view=dict(cfg,registration_ref=cfg["cut_preparation_registration_ref"])
    # All other exact refs, original40 observations, core/codec/counts and actual
    # admission/input/launch/config/runtime/completion/wait are unchanged.
    return framework.completed_cut_preparation_v1(reader,spec,preparation_view)


def required_context_fields_v1():
    return ("root_authorizer","admitted","hypothesis_id","arm","phase","physical_gpu",
        "device","cfg","reader","output","config_ref","registry","entry_refs",
        "observation_cache_ref","observation_source_refs","bindings","guard",
        "before_episode_build","before_cut_decode","cut_preparation",
        "cut_preparation_completion_ref","cut_preparation_actual_wait_ref")
