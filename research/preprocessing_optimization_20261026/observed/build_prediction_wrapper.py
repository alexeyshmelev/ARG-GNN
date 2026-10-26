"""Stage a readable copied prediction route with explicit preprocessing changes."""
from pathlib import Path
root=Path('/workspace-SR003.nfs2/ARG-GNN')
source=(root/'research/repairs/benchmark_recovery_20261025/european/ap_prediction_chunked_gru.py').read_text()
start=source.index('def run_predictions(')
end=source.index('\n\ndef prediction_factory(',start)
run=source[start:end]
run=run.replace('    torch,check,writer,binding = baseline.setup(spec,resource_check)',
'''    _validate_execution_options(spec)
    serialized = SerializedCheck(resource_check)
    torch,check,writer,binding = baseline.setup(spec,serialized)
    check = SerializedCheck(check, serialized.lock)''')
old='''    cache,predictions = OrderedDict(),{}
    load = original._observed_loader(prediction,cache,check,fit=False)
    for parent in plan["test_parent_ids"]:
        if parent not in prediction["heldout_inputs"]:
            continue
        bundle,metadata = load(parent)
        memory._admit_parent_headroom(cache,memory_envelope=prediction["memory_envelope"],
            stage="prediction",rss_bytes=memory._current_rss_bytes,
            resource_check=check,event=lambda *args:None,keep_parent=parent)
'''
new='''    cache,predictions = OrderedDict(),{}
    parents = tuple(parent for parent in plan["test_parent_ids"]
                    if parent in prediction["heldout_inputs"])
    def load(parent, callback):
        return _load_observed_parent(prediction, parent, callback, torch)
    def admit_overlap():
        if not spec["prefetch_next_parent"]:
            return False
        check("TreeLoom_prefetch_combined_RSS_admission")
        envelope = prediction["memory_envelope"]
        current = memory._current_rss_bytes()
        g.need(type(current) is int and current >= 0, "Actual current RSS required")
        return combined_headroom_available(current, envelope)
    with OneParentLookahead(parents, load, resource_check=check,
                            admit_overlap=admit_overlap) as lookahead:
        for parent in parents:
            bundle,metadata = lookahead.take(parent)
            memory._admit_parent_headroom(cache,memory_envelope=prediction["memory_envelope"],
                stage="prediction",rss_bytes=memory._current_rss_bytes,
                resource_check=check,event=lambda *args:None,keep_parent=parent)
            lookahead.start_next()
'''
assert old in run
run=run.replace(old,new)
body_start=run.index('        logits = predict_logits(')
body_end=run.index('    torch.cuda.synchronize(0)',body_start)
body=run[body_start:body_end]
run=run[:body_start]+''.join('    '+line if line.strip() else line for line in body.splitlines(keepends=True))+run[body_end:]
preamble='''"""Prediction-only execution adapter with exact parsing and bounded lookahead.

The trained model, recurrent execution, sampled targets, logits, softmax and
product construction remain the original qualified prediction implementation.
Only our observed preprocessing and its scheduling change. No author module or
currently running worker is patched.
"""
from collections import OrderedDict
from arggnn.benchmarking.paper_argmix import paper_argmix_ap_prediction as baseline
from arggnn.benchmarking.paper_argmix.paper_argmix_ap_prediction import (
    g,c,original,european,memory,predict_logits)
from research.repairs.benchmark_recovery_20261025.european import ap_prediction_chunked_gru as previous
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_gru
from . import adapter, parser, preparation, prefetch
from .prefetch import OneParentLookahead, SerializedCheck
from ..tree import observations, episodes, cut, kernels

MODULE = "research.preprocessing_optimization_20261026.observed.prediction"
SCHEMA = "AP_native6_exact_preprocessing_lookahead_prediction_v1"
PROTOCOL = previous.PROTOCOL
GRU_CHUNK_HOLDS = previous.GRU_CHUNK_HOLDS
install_model_execution = previous.install_model_execution


def requirements():
    prior = previous.requirements()
    sources = [*prior["source_refs"], *[c.file_ref(module.__file__) for module in
        (adapter,parser,preparation,prefetch,observations,episodes,cut,kernels)],
        c.file_ref("research/preprocessing_optimization_20261026/tree/kernels.cpp"),
        c.file_ref("research/preprocessing_optimization_20261026/tree/libstructural.so"),
        c.file_ref(__file__)]
    return dict(previous=prior,source_refs=list({r["path"]:r for r in sources}.values()),
        observed_parser="exact_vectorized_with_original_scalar_fallback",
        tree_preparation="qualified_serial_compiled_structural_kernels",
        next_parent_prefetch_limit=1,completed_parent_cache_bytes=0,
        scientific_model_and_prediction_algebra_unchanged=True,
        native_competitor_source_modified=False,training_repeated=False)


def _validate_execution_options(spec):
    envelope = spec["prediction_spec"]["memory_envelope"]
    g.need(type(spec["prefetch_next_parent"]) is bool,
           "Explicit next-parent prefetch setting required")
    g.need(spec["prediction_spec"]["observation_cache_bytes"] == 0,
           "This one-pass prediction route retains no completed-parent cache")
    g.need(envelope["max_rss_bytes"] <= 512*2**30
           and envelope["preparation_headroom_bytes"] >= 128*2**30
           and envelope["prediction_headroom_bytes"] >= 64*2**30,
           "Qualified complete-parent preparation/prediction reserves within512GiB required")


def combined_headroom_available(current, envelope):
    return current + envelope["preparation_headroom_bytes"] + envelope["prediction_headroom_bytes"] <= envelope["max_rss_bytes"]


def build_spec(*,prefetch_next_parent=True,**kwargs):
    spec = previous.build_spec(**kwargs)
    spec.update(schema=SCHEMA,requirements=requirements(),
                prefetch_next_parent=prefetch_next_parent)
    _validate_execution_options(spec)
    return spec


def _load_observed_parent(spec,parent,check,torch):
    item = spec["heldout_inputs"][parent]
    def prepare():
        state = adapter.load_relate_observation(
            item["inference_ref"],item["observed_vcf_ref"],protocol_id=spec["protocol_id"],
            max_sites=spec["max_sites"],max_metadata_bytes=spec["max_metadata_bytes"],resource_check=check)
        return preparation.prepare_observation(
            state,protocol_id=spec["protocol_id"],parent_id=parent,split="test",
            source_refs=state["source_refs"],sampled_registry_ref=item["sampled_registry_ref"],
            resource_check=check,max_sampled_metadata_bytes=spec["max_metadata_bytes"],
            **spec["preparation_limits"])
    # A separate empty cache avoids shared mutable cache state between the
    # prediction and its single background producer. No previous bank is kept.
    return memory._load_cached_parent(parent,OrderedDict(),prepare,
        observation_cache_bytes=0,memory_envelope=spec["memory_envelope"],
        rss_bytes=memory._current_rss_bytes,tensor_bytes=lambda bundle:memory._cpu_bytes(bundle,torch),
        resource_check=check,event=lambda *args:None)


'''
factory='''

def prediction_factory(**kwargs):
    spec = build_spec(**kwargs)
    return dict(module=MODULE,function="run_predictions",spec=spec,
        python=".runtime/env/bin/python",source_refs=spec["requirements"]["source_refs"])
'''
(root/'research/preprocessing_optimization_20261026/observed/prediction.py').write_text(preamble+run+factory)
